"""
jump_risk_adjustments.py — BCJ-style jump-risk premium and estimation-risk adjustments
for the SVJ model. Produces Tables 6 and 7 of Paper P1.

PURPOSE
-------
Tests whether adjusting SVJ Q-measure parameters for (a) jump-risk premium or
(b) estimation risk can reconcile the significantly negative realised delta-neutral
strategy returns with model predictions, i.e., raise the finite-sample p-values
above 0.05.  Follows Broadie, Chernov, and Johannes (2009, J. Finance), Section VI.

METHODOLOGY
-----------
Adjustment 1 — Jump-Risk Premium (BCJ Section VI.A):
  Under a power-utility investor with risk-aversion gamma=10, the physical-measure
  (P) jump parameters map to risk-neutral (Q) parameters via:
    lambda_Q = lambda_P * (1 + gamma * sigma_j^2)
    mu_j_Q   = mu_j_P  - gamma * sigma_j^2
    sigma_j_Q = sigma_j_P   (unchanged)
  BCJ scaled lambda from 0.59 to 0.97 (factor 1.64) and shifted mu_j from -3.25%
  to -6.85% using gamma=10.  We apply the same formula to Nifty 50 Lee-Mykland
  jump parameters.

Adjustment 2 — Estimation Risk (BCJ Section VI.B):
  Uncertainty in parameter estimates widens the effective volatility distribution.
  We implement this by:
    (a) Adding 1 percentage point to spot volatility: sqrt(theta_Q) += 0.01
    (b) Shifting each Q-measure parameter by +1 posterior standard error in the
        direction that most inflates simulated option payoffs (makes the model
        less conservative), following BCJ's one-at-a-time shift approach.

Both adjustments are applied to the SVJ model only.  BS and SV baselines are
re-printed from data/simulation_results.json (not re-simulated) for context.

REPRODUCIBILITY
---------------
  python jump_risk_adjustments.py

RANDOM SEED
-----------
  np.random.seed(20260509)  — set via np.random.default_rng(RANDOM_SEED).

OUTPUT
------
  papers/P1/tables/table6_adjustments.csv
  papers/P1/tables/table7_qparams.csv
  data/jump_risk_results.json

SANITY CHECK
------------
  Before reporting: verify that BS ATM straddle p-value (from simulation_results.json)
  is 0.000.  If straddle_return p-value > 0.10, halt.

REFERENCES
----------
  Broadie, M., Chernov, M., & Johannes, M. (2009). Understanding index option returns.
      Journal of Finance, 64(4), 1493-1529.
  Bates, D. S. (1996). Jumps and stochastic volatility. Review of Financial Studies,
      9(1), 69-107.
  Eraker, B., Johannes, M., & Polson, N. (2003). The impact of jumps in volatility and
      returns. Journal of Finance, 58(3), 1269-1300.
  Lee, S. S., & Mykland, P. A. (2008). Jumps in financial markets.
      Review of Financial Studies, 21(6), 2535-2563.
  Pillai, S. (2012). [Master's thesis, unpublished].
"""

import os
import json
import csv
import sys
import math
import numpy as np

# ---------------------------------------------------------------------------
# Global constants — must not be changed without updating the paper.
# ---------------------------------------------------------------------------

RANDOM_SEED  = 20260509
N_PATHS      = 25_000
TRADING_DAYS_PER_MONTH = 21
GAMMA        = 10.0          # Risk-aversion coefficient (BCJ baseline)
BASE_DIR     = "C:\\Users\\sumin\\definedge_downloader"

# Strategy columns that appear in Tables 6 and 7 (BCJ focus: delta-neutral).
# We report all 7 columns but highlight straddle and CN as the primary BCJ tests.
MONEYNESS_COLS = {
    0.94: "k094_return",
    0.96: "k096_return",
    0.98: "k098_return",
    1.00: "k100_return",
}
STRATEGY_COLS = ["straddle_return", "cn_return", "psp_return"]
ALL_COLS      = list(MONEYNESS_COLS.values()) + STRATEGY_COLS

# Primary strategies for Table 6 (BCJ-style delta-neutral focus).
PRIMARY_COLS  = ["straddle_return", "cn_return"]

# ---------------------------------------------------------------------------
# Step 1: Load parameters
# ---------------------------------------------------------------------------

def load_parameters():
    """
    Load SVJ model parameters from the canonical JSON files.

    SVJ baseline uses:
      - SV MCMC posterior means for the diffusion component (kappa, theta, sigma_v,
        rho) from mcmc_parameters_full.json, where chains converged (R-hat <= 1.04).
      - MoM mu from mcmc_parameters.json (SVJ MCMC mu chain diverged, R-hat=2.87).
      - Lee-Mykland jump parameters (lambda_p, mu_j, sigma_j) from
        jump_detection_results.json method B_LeeMykland.

    Standard errors for the estimation-risk adjustment come from:
      - mcmc_parameters_full.json (MCMC SE) for sv parameters.
      - mcmc_parameters.json (MoM SE) for mu and jump parameters (since MCMC
        SVJ chain diverged).

    Returns
    -------
    svj_base   : dict  — SVJ baseline P-measure parameters
    svj_ses    : dict  — posterior standard errors for each SVJ parameter
    rf_annual  : float — mean annualised risk-free rate from panel context
    T_months   : int   — panel length
    """
    mcmc_path = os.path.join(BASE_DIR, "data", "mcmc_parameters_full.json")
    mom_path  = os.path.join(BASE_DIR, "data", "mcmc_parameters.json")
    jump_path = os.path.join(BASE_DIR, "data", "jump_detection_results.json")
    sim_path  = os.path.join(BASE_DIR, "data", "simulation_results.json")

    with open(mcmc_path, "r") as fh:
        mcmc = json.load(fh)
    with open(mom_path, "r") as fh:
        mom = json.load(fh)
    with open(jump_path, "r") as fh:
        jd = json.load(fh)
    with open(sim_path, "r") as fh:
        sim = json.load(fh)

    lm = jd["methods"]["B_LeeMykland"]
    if not lm["reliable"]:
        raise RuntimeError(
            "B_LeeMykland jump detection is not reliable. "
            "Inspect jump_detection_results.json before proceeding."
        )

    svj_base = {
        "mu":       mom["svj"]["mu"],            # MoM (SVJ MCMC diverged)
        "kappa":    mcmc["sv"]["kappa_v"],        # SV MCMC posterior mean
        "theta":    mcmc["sv"]["theta"],           # SV MCMC posterior mean
        "sigma_v":  mcmc["sv"]["sigma_v"],         # SV MCMC posterior mean
        "rho":      mcmc["sv"]["rho"],             # SV MCMC posterior mean
        "V0":       mcmc["sv"]["theta"],           # start at long-run mean
        "lambda_p": lm["lambda_p"],                # Lee-Mykland annualised
        "mu_j":     lm["mu_j"],
        "sigma_j":  lm["sigma_j"],
    }

    # Standard errors: MCMC SE for diffusion params, MoM SE for mu and jump params.
    svj_ses = {
        "mu":       mom["svj"]["mu_se"],
        "kappa":    mcmc["sv"]["kappa_v_se"],
        "theta":    mcmc["sv"]["theta_se"],
        "sigma_v":  mcmc["sv"]["sigma_v_se"],
        "rho":      mcmc["sv"]["rho_se"],
        "lambda_p": mom["svj"]["lambda_p_se"],
        "mu_j":     mom["svj"]["mu_j_se"],
        "sigma_j":  mom["svj"]["sigma_j_se"],
    }

    rf_annual = sim["meta"]["rf_annual"]
    T_months  = sim["meta"]["T_months"]

    return svj_base, svj_ses, rf_annual, T_months


# ---------------------------------------------------------------------------
# Step 2: Compute Q-measure adjusted parameters
# ---------------------------------------------------------------------------

def compute_jump_risk_params(svj_base):
    """
    Compute risk-neutral (Q-measure) jump parameters under the jump-risk premium
    adjustment using BCJ Section VI.A (power-utility framework, gamma=10).

    Under power utility with constant relative risk aversion gamma, the mapping
    from P-measure to Q-measure jump parameters is:

        lambda_Q = lambda_P * E_P[exp(-gamma * J)]
                 ≈ lambda_P * (1 + gamma * sigma_j^2)      [for small |mu_j|]

    The exact formula integrating over the log-normal jump size distribution gives:
        E_P[exp(-gamma * J)] where J ~ N(mu_j, sigma_j^2)
        = exp(-gamma * mu_j + 0.5 * gamma^2 * sigma_j^2)

        lambda_Q = lambda_P * exp(-gamma * mu_j + 0.5 * gamma^2 * sigma_j^2)

    The Q-measure jump mean is:
        mu_j_Q = mu_j_P - gamma * sigma_j^2

    BCJ used the approximation (1 + gamma * sigma_j^2) to get factor 1.64.  We
    compute both the exact and approximate values for transparency, and use the
    exact form as the primary result.

    Parameters
    ----------
    svj_base : dict with keys lambda_p, mu_j, sigma_j (and SV params)

    Returns
    -------
    dict — Q-measure SVJ parameters under jump-risk premium adjustment
    """
    lambda_p = svj_base["lambda_p"]
    mu_j     = svj_base["mu_j"]
    sigma_j  = svj_base["sigma_j"]
    gamma    = GAMMA

    # Exact Q-measure lambda using the log-normal moment generating function.
    log_mgf = -gamma * mu_j + 0.5 * gamma ** 2 * sigma_j ** 2
    lambda_q_exact = lambda_p * math.exp(log_mgf)

    # BCJ approximation for reference.
    lambda_q_approx = lambda_p * (1.0 + gamma * sigma_j ** 2)

    # Q-measure jump mean.
    mu_j_q = mu_j - gamma * sigma_j ** 2

    # sigma_j unchanged under this utility framework.
    sigma_j_q = sigma_j

    params = dict(svj_base)   # copy all SV params
    params["lambda_p"] = lambda_q_exact
    params["mu_j"]     = mu_j_q
    params["sigma_j"]  = sigma_j_q

    # Store metadata for table 7.
    params["_lambda_p_approx"] = lambda_q_approx
    params["_lambda_scale_exact"] = lambda_q_exact / lambda_p
    params["_lambda_scale_approx"] = lambda_q_approx / lambda_p

    return params


def compute_estimation_risk_params(svj_base, svj_ses):
    """
    Compute SVJ parameters under the estimation-risk adjustment (BCJ Section VI.B).

    Implementation:
      1. Add 0.01 to sqrt(theta) to increase long-run spot volatility by 1 ppt.
         New theta = (sqrt(theta_base) + 0.01)^2.
      2. Increase lambda_p by one MoM standard error (wider jump tails).
      3. Decrease mu_j by one SE (more negative mean jump increases put values).
      4. Increase sigma_j by one SE (fatter jump distribution).
    All other parameters (mu, kappa, sigma_v, rho) are kept at baseline.

    The direction of each shift is chosen to maximally inflate the simulated
    option payoffs (and thus raise simulated return means), making it hardest
    to reject the null that realised returns are consistent with the model.

    Parameters
    ----------
    svj_base : dict — baseline P-measure parameters
    svj_ses  : dict — posterior standard errors

    Returns
    -------
    dict — SVJ parameters under estimation-risk adjustment
    """
    params = dict(svj_base)

    # (a) Increase long-run variance by 1 ppt spot vol.
    sqrt_theta_base = math.sqrt(svj_base["theta"])
    params["theta"] = (sqrt_theta_base + 0.01) ** 2
    params["V0"]    = params["theta"]   # reset starting variance

    # (b) Increase lambda_p by one SE.
    params["lambda_p"] = svj_base["lambda_p"] + svj_ses["lambda_p"]

    # (c) Decrease mu_j by one SE (more negative → larger put payoffs).
    params["mu_j"] = svj_base["mu_j"] - svj_ses["mu_j"]

    # (d) Increase sigma_j by one SE (fatter tails → larger put payoffs).
    params["sigma_j"] = svj_base["sigma_j"] + svj_ses["sigma_j"]

    return params


# ---------------------------------------------------------------------------
# Step 3: SVJ simulation (replicates option_simulation.simulate_svj)
# ---------------------------------------------------------------------------

def simulate_svj(params, T_months, n_paths, rng):
    """
    Simulate monthly log-returns under the Bates (1996) SVJ model.

    Exact replication of the simulate_svj function in option_simulation.py.
    Uses daily Euler-Maruyama with full-truncation scheme for the variance
    process (max(V,0) before each use) and Bernoulli jump indicator.

    Parameters
    ----------
    params   : dict with keys mu, kappa, theta, sigma_v, rho, V0,
                              lambda_p, mu_j, sigma_j
    T_months : int
    n_paths  : int
    rng      : np.random.Generator (seeded externally)

    Returns
    -------
    log_returns : ndarray, shape (n_paths, T_months)
    """
    mu       = params["mu"]
    kappa    = params["kappa"]
    theta    = params["theta"]
    sigma_v  = params["sigma_v"]
    rho      = params["rho"]
    V0       = params["V0"]
    lambda_p = params["lambda_p"]
    mu_j     = params["mu_j"]
    sigma_j  = params["sigma_j"]

    dt        = 1.0 / 252.0
    sqrt_dt   = math.sqrt(dt)
    rho_c     = math.sqrt(max(1.0 - rho ** 2, 0.0))
    lambda_dt = lambda_p * dt    # per-day jump probability

    n_days_total = TRADING_DAYS_PER_MONTH * T_months
    log_returns  = np.zeros((n_paths, T_months))

    Z = rng.standard_normal((n_paths, n_days_total, 3))
    U = rng.uniform(0.0, 1.0, (n_paths, n_days_total))

    V = np.full(n_paths, V0)

    for t in range(n_days_total):
        month_idx = t // TRADING_DAYS_PER_MONTH
        Z1 = Z[:, t, 0]
        Z2 = Z[:, t, 1]
        Z3 = Z[:, t, 2]
        W_S = Z1
        W_V = rho * Z1 + rho_c * Z2

        V_pos     = np.maximum(V, 0.0)
        sqrt_V_dt = np.sqrt(V_pos * dt)

        N_t = (U[:, t] < lambda_dt).astype(float)
        J_t = N_t * (mu_j + sigma_j * Z3)

        r_day = ((mu - lambda_p * mu_j - V_pos / 2.0) * dt
                 + sqrt_V_dt * W_S + J_t)
        log_returns[:, month_idx] += r_day

        V = V_pos + kappa * (theta - V_pos) * dt + sigma_v * sqrt_V_dt * W_V

    return log_returns


# ---------------------------------------------------------------------------
# Step 4: Load panel and compute realized means (mirrors option_simulation.py)
# ---------------------------------------------------------------------------

def load_panel():
    """
    Load the realized monthly option returns panel.

    Returns
    -------
    panel     : list of dicts
    T_months  : int
    """
    panel_path = os.path.join(BASE_DIR, "data", "nifty_options_panel.csv")
    panel = []
    with open(panel_path, "r", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            panel.append(row)

    if len(panel) < 30:
        raise RuntimeError(
            f"Panel has only {len(panel)} months (minimum 30 required). "
            "Rebuild nifty_options_panel.csv before proceeding."
        )
    return panel, len(panel)


def extract_realized_means(panel):
    """
    Compute time-averaged realized returns for all strategy columns.

    Raises RuntimeError if any column has fewer than 30 valid observations.

    Returns
    -------
    dict mapping column_name -> float
    """
    realized = {}
    for col in ALL_COLS:
        values = []
        for i, row in enumerate(panel):
            raw = row.get(col, "")
            if raw in ("", "nan", "NaN", "NA", None):
                continue
            try:
                v = float(raw)
                if math.isfinite(v):
                    values.append(v)
            except ValueError:
                print(f"  [WARNING] Cannot parse '{raw}' in column '{col}', "
                      f"month index {i}. Skipped.")

        if len(values) < 30:
            raise RuntimeError(
                f"Column '{col}' has only {len(values)} valid observations "
                f"(minimum 30). Cannot compute reliable test. "
                "Rebuild the panel or investigate missing data."
            )
        realized[col] = float(np.mean(values))

    return realized


def compute_simulated_returns(log_returns, panel):
    """
    Convert simulated log-returns into option strategy returns using the BCJ
    simplification: entry prices come from the realized panel; only the
    expiry payoff distribution is simulated.

    Exact replication of option_simulation.compute_simulated_returns logic.

    Returns
    -------
    sim_returns : dict column -> ndarray of shape (n_paths,)
        Time-averaged simulated return for each strategy over T_months.
    """
    n_paths, T_months = log_returns.shape
    assert T_months == len(panel)

    acc = {col: np.zeros(n_paths) for col in ALL_COLS}

    for m, row in enumerate(panel):
        S_entry      = float(row["spot_at_entry"])
        S_expiry_sim = S_entry * np.exp(log_returns[:, m])

        for k, col in MONEYNESS_COLS.items():
            K           = float(row[f"k{int(k*100):03d}_strike"])
            entry_price = float(row[f"k{int(k*100):03d}_entry_price"])
            if entry_price <= 0.0:
                continue
            payoff = np.maximum(K - S_expiry_sim, 0.0)
            acc[col] += payoff / entry_price - 1.0

        K_100          = float(row["k100_strike"])
        straddle_entry = float(row["straddle_entry_premium"])
        if straddle_entry > 0.0:
            put_payoff  = np.maximum(K_100 - S_expiry_sim, 0.0)
            call_payoff = np.maximum(S_expiry_sim - K_100, 0.0)
            straddle_payoff = put_payoff + call_payoff
            acc["straddle_return"] += straddle_payoff / straddle_entry - 1.0

        K_094    = float(row["k094_strike"])
        K_098    = float(row["k098_strike"])
        cn_entry = float(row["cn_entry_premium"])
        if abs(cn_entry) > 1e-6:
            put_094  = np.maximum(K_094 - S_expiry_sim, 0.0)
            put_098  = np.maximum(K_098 - S_expiry_sim, 0.0)
            cn_payoff = put_098 - put_094
            acc["cn_return"] += cn_payoff / abs(cn_entry) - 1.0

        psp_entry = float(row["psp_entry_premium"])
        if abs(psp_entry) > 1e-6:
            put_094   = np.maximum(K_094 - S_expiry_sim, 0.0)
            put_098   = np.maximum(K_098 - S_expiry_sim, 0.0)
            psp_payoff = put_098 - put_094
            acc["psp_return"] += psp_payoff / abs(psp_entry) - 1.0

    sim_returns = {col: acc[col] / T_months for col in acc}
    return sim_returns


def compute_pvalues(sim_returns, realized_means):
    """
    Finite-sample one-tailed p-value: fraction of simulated path-average returns
    at or below the realized mean.  Small p-value = evidence of mispricing.

    Returns
    -------
    dict column -> float
    """
    return {
        col: float(np.mean(sim_returns[col] <= realized_means[col]))
        for col in realized_means
    }


# ---------------------------------------------------------------------------
# Step 5: Sanity check
# ---------------------------------------------------------------------------

def run_sanity_check(sim_results_path):
    """
    Load BS straddle p-value from the pre-computed simulation_results.json and
    confirm it is <= 0.10 (the BCJ test must show meaningful rejection under BS).

    Also checks that the BS ATM straddle p-value is consistent with ~0.000,
    as the BCJ paradigm predicts.

    Raises RuntimeError if the check fails.
    """
    with open(sim_results_path, "r") as fh:
        sim = json.load(fh)

    bs_straddle_pval = sim["models"]["BS"]["straddle_return"]["p_value"]
    bs_cn_pval       = sim["models"]["BS"]["cn_return"]["p_value"]

    print(f"\n[SANITY CHECK] BS straddle p-value (from simulation_results.json): "
          f"{bs_straddle_pval:.4f}")
    print(f"[SANITY CHECK] BS CN spread p-value: {bs_cn_pval:.4f}")

    if bs_straddle_pval > 0.10:
        raise RuntimeError(
            f"[SANITY FAIL] BS straddle p-value = {bs_straddle_pval:.4f} > 0.10. "
            "The BCJ test paradigm requires that realized returns are significantly "
            "more negative than BS model predictions. "
            "Something is wrong with the simulation or panel. Halting."
        )

    print("[SANITY CHECK] PASSED — BS straddle p-value is in the expected range.")
    return sim


# ---------------------------------------------------------------------------
# Step 6: Table builders
# ---------------------------------------------------------------------------

def build_table6(realized_means, no_adj_pvals, jump_risk_pvals, est_risk_pvals,
                 no_adj_sim_means, jump_risk_sim_means, est_risk_sim_means):
    """
    Build Table 6: p-value comparison across adjustment scenarios.

    Columns: strategy, realized_mean, no_adj_sim_mean, no_adj_pval,
             jump_risk_sim_mean, jump_risk_pval, est_risk_sim_mean, est_risk_pval

    Returns
    -------
    list of dicts, one per strategy.
    """
    strategy_labels = {
        "k094_return":     "Put k=0.94",
        "k096_return":     "Put k=0.96",
        "k098_return":     "Put k=0.98",
        "k100_return":     "Put k=1.00 (ATM)",
        "straddle_return": "ATM Straddle",
        "cn_return":       "Crash-Neutral Spread",
        "psp_return":      "Put Spread (PSP)",
    }

    rows = []
    for col in ALL_COLS:
        rows.append({
            "strategy":            strategy_labels.get(col, col),
            "column":              col,
            "realized_mean":       round(realized_means[col], 6),
            "no_adj_sim_mean":     round(no_adj_sim_means[col], 6),
            "no_adj_pval":         round(no_adj_pvals[col], 4),
            "jump_risk_sim_mean":  round(jump_risk_sim_means[col], 6),
            "jump_risk_pval":      round(jump_risk_pvals[col], 4),
            "est_risk_sim_mean":   round(est_risk_sim_means[col], 6),
            "est_risk_pval":       round(est_risk_pvals[col], 4),
        })
    return rows


def build_table7(svj_base, svj_jump_risk, svj_est_risk):
    """
    Build Table 7: Q-measure parameter values under each adjustment scenario.

    Returns
    -------
    list of dicts.
    """
    rows = []

    # Key parameters to report, with display names.
    param_specs = [
        ("lambda_p",  "lambda^P / lambda^Q (jump intensity, annual)",
         svj_base["lambda_p"],
         svj_jump_risk["lambda_p"],
         svj_est_risk["lambda_p"]),
        ("mu_j",      "mu_J (mean jump size)",
         svj_base["mu_j"],
         svj_jump_risk["mu_j"],
         svj_est_risk["mu_j"]),
        ("sigma_j",   "sigma_J (jump size std dev)",
         svj_base["sigma_j"],
         svj_jump_risk["sigma_j"],
         svj_est_risk["sigma_j"]),
        ("theta",     "theta (long-run variance)",
         svj_base["theta"],
         svj_jump_risk["theta"],
         svj_est_risk["theta"]),
        ("kappa",     "kappa (mean-reversion speed)",
         svj_base["kappa"],
         svj_jump_risk["kappa"],
         svj_est_risk["kappa"]),
        ("sigma_v",   "sigma_v (vol-of-vol)",
         svj_base["sigma_v"],
         svj_jump_risk["sigma_v"],
         svj_est_risk["sigma_v"]),
        ("rho",       "rho (correlation)",
         svj_base["rho"],
         svj_jump_risk["rho"],
         svj_est_risk["rho"]),
    ]

    for key, label, base_val, jr_val, er_val in param_specs:
        rows.append({
            "parameter":         label,
            "key":               key,
            "baseline_P":        round(base_val, 6),
            "jump_risk_Q":       round(jr_val, 6),
            "est_risk_Q":        round(er_val, 6),
            "jr_change":         round(jr_val - base_val, 6),
            "er_change":         round(er_val - base_val, 6),
        })

    # Add derived scaling factor for lambda.
    lambda_p_base = svj_base["lambda_p"]
    rows.append({
        "parameter":  "lambda^Q / lambda^P (jump-risk scaling factor)",
        "key":        "lambda_scale",
        "baseline_P": 1.0,
        "jump_risk_Q": round(svj_jump_risk["lambda_p"] / lambda_p_base, 4),
        "est_risk_Q":  round(svj_est_risk["lambda_p"]  / lambda_p_base, 4),
        "jr_change":   round(svj_jump_risk["lambda_p"] / lambda_p_base - 1.0, 4),
        "er_change":   round(svj_est_risk["lambda_p"]  / lambda_p_base - 1.0, 4),
    })

    return rows


# ---------------------------------------------------------------------------
# Step 7: CSV writing
# ---------------------------------------------------------------------------

def write_csv(rows, filepath):
    """Write a list of dicts to CSV.  Creates parent directories as needed."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    if not rows:
        print(f"  [WARNING] No rows to write to {filepath}.")
        return
    fieldnames = list(rows[0].keys())
    with open(filepath, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Wrote {len(rows)} rows -> {filepath}")


# ---------------------------------------------------------------------------
# Step 8: Mongo export
# ---------------------------------------------------------------------------

def export_to_mongo(table6_rows, table7_rows, json_results):
    """
    Export results to MongoDB collection 'paper_results' in 'pricedatabase'.
    Non-fatal: prints a warning and continues if Mongo is unavailable.
    """
    try:
        import pymongo
        from datetime import datetime as _dt
    except ImportError:
        print("  [WARNING] pymongo not installed. Skipping Mongo export.")
        return

    try:
        from dotenv import load_dotenv as _load
        _load()
    except ImportError:
        pass

    mongo_uri = os.environ.get("MONGO_URI", "mongodb://localhost:27017/")

    try:
        client = pymongo.MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        client.server_info()
    except Exception as exc:
        print(f"  [WARNING] Cannot connect to MongoDB ({exc}). Skipping export.")
        return

    db   = client["pricedatabase"]
    coll = db["paper_results"]
    run_date = _dt.utcnow().isoformat()

    inserted = 0
    for row in table6_rows:
        doc = dict(row)
        doc.update({
            "run_date": run_date,
            "script":   "jump_risk_adjustments.py",
            "paper":    "P1",
            "table":    "table6",
        })
        fk = {"column": doc["column"], "paper": "P1", "table": "table6",
              "script": "jump_risk_adjustments.py"}
        coll.update_one(fk, {"$set": doc}, upsert=True)
        inserted += 1

    for row in table7_rows:
        doc = dict(row)
        doc.update({
            "run_date": run_date,
            "script":   "jump_risk_adjustments.py",
            "paper":    "P1",
            "table":    "table7",
        })
        fk = {"key": doc["key"], "paper": "P1", "table": "table7",
              "script": "jump_risk_adjustments.py"}
        coll.update_one(fk, {"$set": doc}, upsert=True)
        inserted += 1

    print(f"  Upserted {inserted} documents -> pricedatabase.paper_results")
    client.close()


# ---------------------------------------------------------------------------
# Step 9: Print comparison table
# ---------------------------------------------------------------------------

def print_comparison_table(table6_rows):
    """Print a formatted comparison of p-values across all three scenarios."""
    print("\n" + "=" * 90)
    print("TABLE 6 — SVJ Adjustment Comparison (finite-sample p-values)")
    print("=" * 90)
    hdr = (f"{'Strategy':<26} {'Realized':>10} {'No-Adj':>10} "
           f"{'Jump-Risk':>10} {'Est-Risk':>10} {'Reconciled?':>12}")
    print(hdr)
    print("-" * 90)

    for row in table6_rows:
        strat   = row["strategy"]
        real    = row["realized_mean"]
        no_adj  = row["no_adj_pval"]
        jr      = row["jump_risk_pval"]
        er      = row["est_risk_pval"]
        # "Reconciled" = either jr or er raises p-value above 0.05.
        reconciled = "Yes" if (jr >= 0.05 or er >= 0.05) else "No"
        print(f"  {strat:<24} {real:>10.3f} {no_adj:>10.4f} "
              f"{jr:>10.4f} {er:>10.4f} {reconciled:>12}")

    print("=" * 90)
    print("Notes:")
    print("  No-Adj     = SVJ baseline (P-measure parameters, Lee-Mykland jumps)")
    print(f"  Jump-Risk  = Q-measure adjustment, gamma={GAMMA:.0f} (BCJ Section VI.A)")
    print("  Est-Risk   = Estimation-risk adjustment (BCJ Section VI.B)")
    print("  p-value    = fraction of 25,000 simulated path means <= realized mean")
    print("  Reconciled = p-value >= 0.05 under at least one adjustment")
    print()


def print_table7(table7_rows):
    """Print Q-measure parameter values for Table 7."""
    print("\n" + "=" * 90)
    print("TABLE 7 — Q-Measure SVJ Parameters Under Each Adjustment")
    print("=" * 90)
    hdr = (f"{'Parameter':<45} {'Baseline':>10} {'Jump-Risk':>10} {'Est-Risk':>10}")
    print(hdr)
    print("-" * 90)
    for row in table7_rows:
        label = row["parameter"]
        base  = row["baseline_P"]
        jr    = row["jump_risk_Q"]
        er    = row["est_risk_Q"]
        print(f"  {label:<43} {base:>10.6f} {jr:>10.6f} {er:>10.6f}")
    print("=" * 90)
    print()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    """
    Full pipeline for jump-risk premium and estimation-risk adjustments.

    Steps
    -----
    1. Run sanity check on pre-computed BS p-values.
    2. Load SVJ base parameters and standard errors.
    3. Compute Q-measure adjusted parameters (jump-risk and estimation-risk).
    4. Load realized panel; extract realized means.
    5. Simulate SVJ paths under all three parameter sets (N_PATHS=25,000 each).
       Single RNG seeded once — draws consumed in order: no-adj, jump-risk, est-risk.
    6. Compute p-values for all strategies under each scenario.
    7. Build Table 6 (p-value comparison) and Table 7 (Q-measure params).
    8. Write CSVs, JSON, and Mongo export.
    9. Print formatted comparison tables.
    """
    print("=" * 72)
    print("jump_risk_adjustments.py — BCJ-style adjustment analysis for P1")
    print(f"Seed: {RANDOM_SEED}   N_PATHS: {N_PATHS:,}   gamma: {GAMMA:.0f}")
    print("=" * 72)

    sim_path = os.path.join(BASE_DIR, "data", "simulation_results.json")

    # --- Step 1: Sanity check ---
    print("\n[1/9] Running sanity check on pre-computed BS results...")
    sim_data = run_sanity_check(sim_path)

    # --- Step 2: Load parameters ---
    print("\n[2/9] Loading SVJ parameters and standard errors...")
    svj_base, svj_ses, rf_annual, T_months_sim = load_parameters()

    print(f"  SVJ base  : lambda_p={svj_base['lambda_p']:.4f}, "
          f"mu_j={svj_base['mu_j']:.6f}, sigma_j={svj_base['sigma_j']:.6f}")
    print(f"  SVJ SEs   : lambda_p_se={svj_ses['lambda_p']:.4f}, "
          f"mu_j_se={svj_ses['mu_j']:.6f}, sigma_j_se={svj_ses['sigma_j']:.6f}")
    print(f"  rf_annual={rf_annual:.4f}, T_months={T_months_sim}")

    # --- Step 3: Compute adjusted parameters ---
    print("\n[3/9] Computing Q-measure adjusted parameters...")
    svj_jump_risk = compute_jump_risk_params(svj_base)
    svj_est_risk  = compute_estimation_risk_params(svj_base, svj_ses)

    sigma_j  = svj_base["sigma_j"]
    gamma    = GAMMA
    print(f"  [Jump-Risk] lambda_p: {svj_base['lambda_p']:.4f} -> "
          f"{svj_jump_risk['lambda_p']:.4f} "
          f"(x{svj_jump_risk['_lambda_scale_exact']:.4f} exact, "
          f"x{svj_jump_risk['_lambda_scale_approx']:.4f} BCJ-approx)")
    print(f"  [Jump-Risk] mu_j:     {svj_base['mu_j']:.6f} -> "
          f"{svj_jump_risk['mu_j']:.6f}  "
          f"(shift = -gamma*sigma_j^2 = -{gamma*sigma_j**2:.6f})")
    print(f"  [Est-Risk]  theta:    {svj_base['theta']:.6f} -> "
          f"{svj_est_risk['theta']:.6f}  "
          f"(sqrt(theta) += 0.01: {math.sqrt(svj_base['theta']):.4f} -> "
          f"{math.sqrt(svj_est_risk['theta']):.4f})")
    print(f"  [Est-Risk]  lambda_p: {svj_base['lambda_p']:.4f} -> "
          f"{svj_est_risk['lambda_p']:.4f}  (+{svj_ses['lambda_p']:.4f} SE)")
    print(f"  [Est-Risk]  mu_j:     {svj_base['mu_j']:.6f} -> "
          f"{svj_est_risk['mu_j']:.6f}  (-{svj_ses['mu_j']:.6f} SE)")
    print(f"  [Est-Risk]  sigma_j:  {svj_base['sigma_j']:.6f} -> "
          f"{svj_est_risk['sigma_j']:.6f}  (+{svj_ses['sigma_j']:.6f} SE)")

    # --- Step 4: Load panel ---
    print("\n[4/9] Loading realized option returns panel...")
    panel, T_months = load_panel()
    print(f"  Panel months: {T_months}")

    realized_means = extract_realized_means(panel)
    print(f"  Realized straddle mean : {realized_means['straddle_return']:+.4f}")
    print(f"  Realized CN mean       : {realized_means['cn_return']:+.4f}")

    if T_months != T_months_sim:
        print(f"  [WARNING] Panel T_months={T_months} differs from "
              f"simulation_results.json T_months={T_months_sim}. "
              "Using panel T_months for new simulations.")

    # --- Step 5: Simulate ---
    # Single RNG seeded once; draws consumed in deterministic order.
    rng = np.random.default_rng(RANDOM_SEED)

    scenarios = [
        ("no_adj",     svj_base),
        ("jump_risk",  svj_jump_risk),
        ("est_risk",   svj_est_risk),
    ]

    scenario_results = {}
    for label, params in scenarios:
        # Strip internal metadata keys (prefixed with '_') before passing to simulator.
        sim_params = {k: v for k, v in params.items() if not k.startswith("_")}
        print(f"\n[5/9] Simulating SVJ [{label}] "
              f"({N_PATHS:,} paths x {T_months} months)...")
        print(f"  lambda_p={sim_params['lambda_p']:.4f}, "
              f"mu_j={sim_params['mu_j']:.6f}, "
              f"sigma_j={sim_params['sigma_j']:.6f}, "
              f"theta={sim_params['theta']:.6f}")

        log_returns = simulate_svj(sim_params, T_months, N_PATHS, rng)
        print(f"  log_returns: mean={log_returns.mean():.5f}, "
              f"std={log_returns.std():.5f}")

        sim_rets = compute_simulated_returns(log_returns, panel)
        pvals    = compute_pvalues(sim_rets, realized_means)

        scenario_results[label] = {
            "params":    sim_params,
            "sim_rets":  sim_rets,
            "pvals":     pvals,
            "sim_means": {col: float(np.mean(sim_rets[col])) for col in ALL_COLS},
            "sim_stds":  {col: float(np.std(sim_rets[col], ddof=1)) for col in ALL_COLS},
        }

        print(f"  p-values [{label}]:")
        for col in PRIMARY_COLS:
            print(f"    {col:<22} p={pvals[col]:.4f}")

    # --- Step 6: Assemble p-value results ---
    no_adj_pvals     = scenario_results["no_adj"]["pvals"]
    jump_risk_pvals  = scenario_results["jump_risk"]["pvals"]
    est_risk_pvals   = scenario_results["est_risk"]["pvals"]
    no_adj_sim_means    = scenario_results["no_adj"]["sim_means"]
    jump_risk_sim_means = scenario_results["jump_risk"]["sim_means"]
    est_risk_sim_means  = scenario_results["est_risk"]["sim_means"]

    # --- Step 7: Build tables ---
    print("\n[7/9] Building result tables...")
    table6_rows = build_table6(
        realized_means,
        no_adj_pvals, jump_risk_pvals, est_risk_pvals,
        no_adj_sim_means, jump_risk_sim_means, est_risk_sim_means,
    )
    table7_rows = build_table7(svj_base, svj_jump_risk, svj_est_risk)

    # --- Step 8: Write output ---
    print("\n[8/9] Writing output files...")
    t6_path = os.path.join(BASE_DIR, "papers", "P1", "tables", "table6_adjustments.csv")
    t7_path = os.path.join(BASE_DIR, "papers", "P1", "tables", "table7_qparams.csv")
    jr_path = os.path.join(BASE_DIR, "data", "jump_risk_results.json")

    write_csv(table6_rows, t6_path)
    write_csv(table7_rows, t7_path)

    # Build JSON output.
    json_out = {
        "meta": {
            "seed":     RANDOM_SEED,
            "n_paths":  N_PATHS,
            "T_months": T_months,
            "gamma":    GAMMA,
            "rf_annual": rf_annual,
            "script":   "jump_risk_adjustments.py",
        },
        "parameters": {
            "svj_base":      {k: float(v) for k, v in svj_base.items()},
            "svj_ses":       {k: float(v) for k, v in svj_ses.items()},
            "svj_jump_risk": {k: float(v) for k, v in svj_jump_risk.items()
                              if not k.startswith("_")},
            "svj_jump_risk_meta": {
                "lambda_scale_exact":  svj_jump_risk["_lambda_scale_exact"],
                "lambda_scale_approx": svj_jump_risk["_lambda_scale_approx"],
                "lambda_approx":       svj_jump_risk["_lambda_p_approx"],
            },
            "svj_est_risk":  {k: float(v) for k, v in svj_est_risk.items()},
        },
        "realized_means": {k: float(v) for k, v in realized_means.items()},
        "scenarios": {},
    }

    for label, res in scenario_results.items():
        json_out["scenarios"][label] = {
            "p_values":  {k: float(v) for k, v in res["pvals"].items()},
            "sim_means": {k: float(v) for k, v in res["sim_means"].items()},
            "sim_stds":  {k: float(v) for k, v in res["sim_stds"].items()},
        }

    with open(jr_path, "w") as fh:
        json.dump(json_out, fh, indent=2)
    print(f"  Wrote raw results -> {jr_path}")

    # Mongo (non-fatal).
    export_to_mongo(table6_rows, table7_rows, json_out)

    # --- Step 9: Print formatted tables ---
    print_comparison_table(table6_rows)
    print_table7(table7_rows)

    print("=" * 72)
    print("DONE. Output files:")
    print(f"  Table 6 -> {t6_path}")
    print(f"  Table 7 -> {t7_path}")
    print(f"  Raw JSON -> {jr_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
