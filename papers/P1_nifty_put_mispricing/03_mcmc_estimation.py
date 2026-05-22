"""
mcmc_estimation.py
==================
Estimates P-measure parameters for three option pricing models from Nifty 50
daily log-returns: Black-Scholes (BS), Heston Stochastic Volatility (SV), and
Bates SVJ (SV + compound Poisson jumps).

Methodology follows Broadie, Chernov & Johannes (2009) [BCJ] and
Eraker, Johannes & Polson (2003) [EJP], adapted to the Indian Nifty 50 market.
SV parameters use method-of-moments on monthly realised variance (Kim, Shephard
& Chib 1998 style). Jump days are identified via Barndorff-Nielsen & Shephard
(2006) bipower-variation thresholding.

Reproducibility
---------------
    python mcmc_estimation.py

Random seed: np.random.seed(20260509)  [project start date]

Inputs
------
    data/nifty_spot.csv   -- columns: date, open, high, low, close, adj_close,
                             volume, log_return
    data/tbill_91d.csv    -- columns: date, yield_pct

Outputs
-------
    data/mcmc_parameters.json
    papers/P1/tables/table2_parameters.csv
    MongoDB collection paper_results  (database optionsdatabase)

References
----------
    Broadie M, Chernov M, Johannes M (2009). Understanding Index Option Returns.
        Review of Financial Studies, 22(11), 4493-4529.
    Eraker B, Johannes M, Polson N (2003). The Impact of Jumps in Volatility and
        Returns. Journal of Finance, 58(3), 1269-1300.
    Barndorff-Nielsen OE, Shephard N (2006). Econometrics of Testing for Jumps
        in Financial Economics Using Bipower Variation. Journal of Financial
        Econometrics, 4(1), 1-30.
    Heston S (1993). A Closed-Form Solution for Options with Stochastic Volatility.
        Review of Financial Studies, 6(2), 327-343.
    Bates D (1996). Jumps and Stochastic Volatility: Exchange Rate Processes
        Implicit in Deutsche Mark Options. Review of Financial Studies, 9(1), 69-107.
    Pillai S (2012). Nifty 50 Index Put Option Mispricing (Master's thesis,
        unpublished).
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize_scalar

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = "C:\\Users\\sumin\\definedge_downloader"
DATA_DIR = os.path.join(BASE_DIR, "data")
PAPERS_DIR = os.path.join(BASE_DIR, "papers")
P1_TABLES_DIR = os.path.join(PAPERS_DIR, "P1", "tables")

SPOT_CSV = os.path.join(DATA_DIR, "nifty_spot.csv")
TBILL_CSV = os.path.join(DATA_DIR, "tbill_91d.csv")
PARAMS_JSON = os.path.join(DATA_DIR, "mcmc_parameters.json")
TABLE2_CSV = os.path.join(P1_TABLES_DIR, "table2_parameters.csv")

RANDOM_SEED = 20260509
TRADING_DAYS_PER_YEAR = 252
MONTHS_PER_YEAR = 12

# BCJ (2009) S&P 500 benchmark values — Table 1 in BCJ, 1987-2005 sample.
BCJ_BENCHMARK = {
    "bs":  {"mu": 0.0541, "sigma": 0.150},
    "sv":  {"kappa_v": 5.33, "theta_sqrt": 0.150, "sigma_v": 0.60, "rho": -0.58},
    "svj": {"lambda_p": 0.63, "mu_j": -0.0325, "sigma_j": 0.0434},
}

# Jump detection threshold multiplier (BNS 2006 recommend 3-sigma)
JUMP_THRESHOLD_SIGMA = 3.0

# Bootstrap configuration for standard errors
BOOTSTRAP_N_REPS = 2000
BOOTSTRAP_BLOCK_SIZE = 22   # roughly one trading month


# ---------------------------------------------------------------------------
# Step 1 — Data loading and validation
# ---------------------------------------------------------------------------

def load_spot_data(path: str) -> pd.DataFrame:
    """Load Nifty 50 spot data; validate required columns; parse dates."""
    required = {"date", "log_return"}
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower()
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"nifty_spot.csv is missing required columns: {missing}\n"
            f"Found columns: {list(df.columns)}"
        )
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    n_na = df["log_return"].isna().sum()
    if n_na > 0:
        print(f"  [WARN] {n_na} NaN log_return rows dropped.")
        df = df.dropna(subset=["log_return"]).reset_index(drop=True)
    if len(df) < 252:
        raise ValueError(
            f"Spot data has only {len(df)} rows after cleaning — need at least 252 "
            "(one full year) for parameter estimation. Halting."
        )
    print(f"  Spot data loaded: {len(df)} rows, {df['date'].min().date()} to "
          f"{df['date'].max().date()}")
    return df


def load_tbill_data(path: str) -> pd.DataFrame:
    """Load 91-day T-bill yield data; return as annual decimal fraction."""
    required = {"date", "yield_pct"}
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower()
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"tbill_91d.csv is missing required columns: {missing}\n"
            f"Found columns: {list(df.columns)}"
        )
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df["yield_decimal"] = df["yield_pct"] / 100.0
    print(f"  T-bill data loaded: {len(df)} rows. "
          f"Mean yield = {df['yield_decimal'].mean():.4f}")
    return df


def compute_mean_rfr(spot_df: pd.DataFrame, tbill_df: pd.DataFrame) -> float:
    """
    Compute the mean annualised risk-free rate aligned to the spot date range.
    T-bill yields are already annualised percentages in the data.
    """
    start = spot_df["date"].min()
    end = spot_df["date"].max()
    mask = (tbill_df["date"] >= start) & (tbill_df["date"] <= end)
    subset = tbill_df.loc[mask, "yield_decimal"]
    if len(subset) == 0:
        # Fall back to full sample mean if date ranges do not overlap
        rfr = tbill_df["yield_decimal"].mean()
        print("  [WARN] No T-bill observations inside spot date range. "
              f"Using full-sample mean rfr = {rfr:.4f}")
    else:
        rfr = subset.mean()
    print(f"  Mean annualised risk-free rate (rfr) = {rfr:.4f} ({rfr*100:.2f}%)")
    return float(rfr)


# ---------------------------------------------------------------------------
# Step 2 — Black-Scholes estimation (direct MLE / analytic)
# ---------------------------------------------------------------------------

def estimate_bs(returns: np.ndarray, rfr: float) -> dict:
    """
    Black-Scholes parameters from daily log-returns.

    mu    = annualised excess return (equity risk premium):
            E[log S_{t+1}/S_t] * 252 - rfr
            Under BS, the log-return under P has drift (mu + rfr - 0.5*sigma^2).
            We recover mu = annualised_mean - rfr + 0.5*sigma^2  (Ito correction).
    sigma = annualised volatility: std(daily log-returns) * sqrt(252)

    Standard errors are asymptotic MLE SEs:
        SE(mu_daily)    = sigma_daily / sqrt(T)
        SE(sigma_daily) = sigma_daily / sqrt(2*(T-1))
    """
    T = len(returns)
    mean_daily = float(np.mean(returns))
    std_daily = float(np.std(returns, ddof=1))

    sigma = std_daily * np.sqrt(TRADING_DAYS_PER_YEAR)
    # Ito-corrected annualised mean under P:
    mu_raw = mean_daily * TRADING_DAYS_PER_YEAR
    mu = mu_raw - rfr + 0.5 * sigma**2  # equity risk premium

    # Asymptotic SEs
    se_mean_daily = std_daily / np.sqrt(T)
    se_std_daily = std_daily / np.sqrt(2 * (T - 1))
    mu_se = se_mean_daily * TRADING_DAYS_PER_YEAR
    sigma_se = se_std_daily * np.sqrt(TRADING_DAYS_PER_YEAR)

    result = {
        "mu": round(mu, 6),
        "sigma": round(sigma, 6),
        "mu_se": round(mu_se, 6),
        "sigma_se": round(sigma_se, 6),
    }
    print(f"\n  [BS] mu={mu*100:.2f}%  sigma={sigma*100:.2f}%  "
          f"(SE mu={mu_se*100:.2f}%  SE sigma={sigma_se*100:.4f}%)")
    return result


# ---------------------------------------------------------------------------
# Step 3 — Monthly realised variance
# ---------------------------------------------------------------------------

def build_monthly_rv(spot_df: pd.DataFrame) -> pd.Series:
    """
    Compute monthly realised variance (RV) from daily squared log-returns.
    RV_m = sum_{t in month m} r_t^2  (annualised by multiplying by 12).

    Returns a pandas Series indexed by (year, month) period.
    """
    df = spot_df.copy()
    df["ym"] = df["date"].dt.to_period("M")
    rv = (
        df.groupby("ym")["log_return"]
        .apply(lambda x: float(np.sum(np.array(x) ** 2)) * 12)
    )
    # Drop months with fewer than 10 trading days (partial / holiday-heavy months)
    counts = df.groupby("ym")["log_return"].count()
    rv = rv[counts >= 10]
    print(f"  Monthly RV computed: {len(rv)} months, "
          f"mean annualised var = {rv.mean():.6f} "
          f"(sqrt = {np.sqrt(rv.mean())*100:.2f}%)")
    return rv


# ---------------------------------------------------------------------------
# Step 4 — Heston SV: method-of-moments
# ---------------------------------------------------------------------------

def estimate_sv_mom(returns: np.ndarray, rv_series: pd.Series,
                    rfr: float) -> dict:
    """
    Heston SV parameters via method-of-moments (Kim-Shephard-Chib style).

    Process (under P, annualised):
        dS/S  = (r + mu) dt + sqrt(V) dW_1
        dV    = kappa_v * (theta - V) dt + sigma_v * sqrt(V) dW_2
        corr(dW_1, dW_2) = rho

    Identification:
        theta   = E[RV_m]                      (unconditional variance)
        kappa_v via AR(1) on monthly RV:
                RV_m = a + b * RV_{m-1} + e_m
                b = exp(-kappa_v / 12)  =>  kappa_v = -12 * ln(b)
        sigma_v from the innovation volatility of the AR(1):
                Var(e_m) ~ (sigma_v^2 * theta) / (2 * kappa_v)  [exact OU formula]
                => sigma_v = sqrt(2 * kappa_v * Var(e) / theta)
        rho     = corr(r_t, r_{t+1}^2)   [leverage proxy; see Aït-Sahalia 1996]
        mu      = annualised mean - rfr + 0.5*theta (risk-premium, Ito-corrected)
    """
    rv = rv_series.values

    # --- theta ---
    theta = float(np.mean(rv))

    # --- AR(1) on monthly RV ---
    rv_lag = rv[:-1]
    rv_cur = rv[1:]
    slope, intercept, r_val, p_val, se_slope = stats.linregress(rv_lag, rv_cur)

    # Guard: slope must be in (0, 1) for a stationary process
    if not (0.0 < slope < 1.0):
        print(f"  [WARN] AR(1) slope = {slope:.4f} is outside (0,1). "
              "Clamping to (0.01, 0.99) for stability.")
        slope = float(np.clip(slope, 0.01, 0.99))

    kappa_v = float(-MONTHS_PER_YEAR * np.log(slope))

    # --- sigma_v from OU residual variance ---
    rv_hat = intercept + slope * rv_lag
    residuals = rv_cur - rv_hat
    var_resid = float(np.var(residuals, ddof=2))
    # Exact OU: Var(e) = sigma_v^2 * theta / (2 * kappa_v / 12)
    # => sigma_v^2 = Var(e) * 2 * (kappa_v/12) / theta
    kappa_m = kappa_v / MONTHS_PER_YEAR   # monthly mean-reversion speed
    sigma_v_sq = var_resid * 2.0 * kappa_m / max(theta, 1e-12)
    sigma_v = float(np.sqrt(max(sigma_v_sq, 1e-10)))

    # --- rho: leverage via correlation of r_t and r_{t+1}^2 ---
    r_arr = returns[:-1]
    r_sq_next = returns[1:] ** 2
    rho, _ = stats.pearsonr(r_arr, r_sq_next)
    rho = float(np.clip(rho, -0.999, 0.999))

    # --- mu ---
    mean_ann = float(np.mean(returns)) * TRADING_DAYS_PER_YEAR
    mu = mean_ann - rfr + 0.5 * theta

    result = {
        "mu": round(mu, 6),
        "kappa_v": round(kappa_v, 6),
        "theta": round(theta, 6),
        "sigma_v": round(sigma_v, 6),
        "rho": round(rho, 6),
    }
    print(f"\n  [SV] mu={mu*100:.2f}%  kappa_v={kappa_v:.4f}  "
          f"sqrt(theta)={np.sqrt(theta)*100:.2f}%  "
          f"sigma_v={sigma_v:.4f}  rho={rho:.4f}")
    return result


# ---------------------------------------------------------------------------
# Step 5 — BNS bipower-variation jump detection
# ---------------------------------------------------------------------------

def bipower_variation(returns: np.ndarray) -> np.ndarray:
    """
    Barndorff-Nielsen & Shephard (2006) bipower variation (daily, rolling
    21-day window for local scaling).

    BV_t = (pi/2) * |r_{t-1}| * |r_t|  — pair-wise sum over window.
    Returns array of per-day expected absolute-jump-free variation.
    """
    T = len(returns)
    abs_r = np.abs(returns)
    bv = np.full(T, np.nan)
    half_win = 10   # 21-day window centred on t
    mu1 = np.sqrt(2.0 / np.pi)   # E[|Z|] for Z ~ N(0,1)
    for t in range(T):
        lo = max(0, t - half_win)
        hi = min(T - 1, t + half_win)
        window = abs_r[lo: hi + 1]
        if len(window) < 3:
            bv[t] = np.mean(abs_r) ** 2 * (np.pi / 2.0)
        else:
            bv[t] = (np.pi / 2.0) * np.sum(window[:-1] * window[1:]) / (len(window) - 1)
    # Divide by mu1^2 to recover annualised-scale daily variance proxy
    bv = bv / (mu1 ** 2)
    return bv


def detect_jumps(returns: np.ndarray,
                 threshold_sigma: float = JUMP_THRESHOLD_SIGMA) -> np.ndarray:
    """
    Classify each day as a jump (True) or not, using BNS bipower variation.
    Jump criterion: |r_t| > threshold_sigma * sqrt(BV_t)
    where BV_t is the daily local volatility from bipower variation.
    Returns a boolean array of shape (T,).
    """
    bv = bipower_variation(returns)
    bv_daily_vol = np.sqrt(np.clip(bv, 1e-16, None))
    is_jump = np.abs(returns) > threshold_sigma * bv_daily_vol
    return is_jump


# ---------------------------------------------------------------------------
# Step 6 — Bates SVJ: SV + jump estimation
# ---------------------------------------------------------------------------

def estimate_svj(returns: np.ndarray, sv_params: dict, rfr: float) -> dict:
    """
    Bates SVJ parameters via BNS jump detection layered on top of SV estimates.

    Jump process (under P):
        N_t ~ Poisson(lambda_p)   [annual intensity]
        J   ~ N(mu_j, sigma_j^2) [log-price jump size]

    Identification:
        is_jump    = BNS threshold classifier
        lambda_p   = n_jumps / T_days * 252
        mu_j       = mean(r_t | is_jump)
        sigma_j    = std(r_t | is_jump)    [with ddof=1]
        mu_svj     = mu_sv + lambda_p * mu_j  (drift compensation under P)

    BCJ reference: Table 2, SVJ model, 1987-2005 S&P 500 sample.
    """
    is_jump = detect_jumps(returns)
    n_jumps = int(is_jump.sum())
    T = len(returns)

    if n_jumps < 5:
        print(f"  [WARN] Only {n_jumps} jump days detected. "
              "SVJ jump-size estimates will be unreliable (N < 5).")

    lambda_p = float(n_jumps / T * TRADING_DAYS_PER_YEAR)
    jump_returns = returns[is_jump]
    mu_j = float(np.mean(jump_returns)) if n_jumps > 0 else 0.0
    sigma_j = float(np.std(jump_returns, ddof=1)) if n_jumps > 1 else 0.0

    # Annualise jump sizes (they are daily; scale to annual equivalent magnitude)
    # Note: per BCJ, mu_j and sigma_j are reported as fraction of index level
    # (i.e. stay in log-return units, not annualised — just raw jump-day returns)

    # Adjust mu for jump compensation (Ito-correction for compound Poisson)
    # Under P: E[d ln S] = (mu_sv + rfr - 0.5*V + lambda*(exp(mu_j+0.5*sigma_j^2)-1)) dt
    # Equity risk premium mu_svj strips out the jump component:
    mu_svj = sv_params["mu"] - lambda_p * mu_j  # remove jump drift from SV mu

    result = {
        "mu": round(mu_svj, 6),
        "kappa_v": sv_params["kappa_v"],
        "theta": sv_params["theta"],
        "sigma_v": sv_params["sigma_v"],
        "rho": sv_params["rho"],
        "lambda_p": round(lambda_p, 6),
        "mu_j": round(mu_j, 6),
        "sigma_j": round(sigma_j, 6),
    }
    print(f"\n  [SVJ] lambda_p={lambda_p:.4f}  mu_j={mu_j*100:.2f}%  "
          f"sigma_j={sigma_j*100:.2f}%  n_jump_days={n_jumps}/{T}")
    return result


# ---------------------------------------------------------------------------
# Step 7 — Block bootstrap for standard errors
# ---------------------------------------------------------------------------

def block_bootstrap_bs(returns: np.ndarray, rfr: float,
                        n_reps: int = BOOTSTRAP_N_REPS,
                        block_size: int = BOOTSTRAP_BLOCK_SIZE) -> dict:
    """
    Block bootstrap (moving blocks) SEs for BS parameters.
    Returns dict of {param: se}.
    """
    np.random.seed(RANDOM_SEED)
    T = len(returns)
    n_blocks = int(np.ceil(T / block_size))

    mu_boot = []
    sigma_boot = []

    for _ in range(n_reps):
        starts = np.random.randint(0, T - block_size + 1, size=n_blocks)
        sample = np.concatenate([returns[s: s + block_size] for s in starts])[:T]
        mean_d = float(np.mean(sample))
        std_d = float(np.std(sample, ddof=1))
        sigma_b = std_d * np.sqrt(TRADING_DAYS_PER_YEAR)
        mu_b = mean_d * TRADING_DAYS_PER_YEAR - rfr + 0.5 * sigma_b ** 2
        mu_boot.append(mu_b)
        sigma_boot.append(sigma_b)

    return {
        "mu_se": round(float(np.std(mu_boot, ddof=1)), 6),
        "sigma_se": round(float(np.std(sigma_boot, ddof=1)), 6),
    }


def block_bootstrap_sv(returns: np.ndarray, spot_df: pd.DataFrame, rfr: float,
                        n_reps: int = BOOTSTRAP_N_REPS,
                        block_size: int = BOOTSTRAP_BLOCK_SIZE) -> dict:
    """
    Block bootstrap SEs for SV method-of-moments parameters.
    Bootstraps at the daily level then rebuilds monthly RV.
    Returns dict of {param: se}.
    """
    np.random.seed(RANDOM_SEED + 1)
    T = len(returns)
    n_blocks = int(np.ceil(T / block_size))
    spot_dates = spot_df["date"].values

    mu_b_list = []
    kappa_b_list = []
    theta_b_list = []
    sigma_v_b_list = []
    rho_b_list = []

    for _ in range(n_reps):
        starts = np.random.randint(0, T - block_size + 1, size=n_blocks)
        idx = np.concatenate([np.arange(s, s + block_size) for s in starts])[:T]
        r_boot = returns[idx]
        dates_boot = pd.to_datetime(spot_dates[idx])
        df_boot = pd.DataFrame({"date": dates_boot, "log_return": r_boot})
        try:
            rv_b = build_monthly_rv(df_boot)
            if len(rv_b) < 6:
                continue
            sv_b = estimate_sv_mom(r_boot, rv_b, rfr)
            mu_b_list.append(sv_b["mu"])
            kappa_b_list.append(sv_b["kappa_v"])
            theta_b_list.append(sv_b["theta"])
            sigma_v_b_list.append(sv_b["sigma_v"])
            rho_b_list.append(sv_b["rho"])
        except Exception:
            continue

    def _se(lst):
        return round(float(np.std(lst, ddof=1)), 6) if len(lst) > 1 else float("nan")

    return {
        "mu_se": _se(mu_b_list),
        "kappa_v_se": _se(kappa_b_list),
        "theta_se": _se(theta_b_list),
        "sigma_v_se": _se(sigma_v_b_list),
        "rho_se": _se(rho_b_list),
    }


def block_bootstrap_svj(returns: np.ndarray, sv_params: dict, rfr: float,
                         n_reps: int = BOOTSTRAP_N_REPS,
                         block_size: int = BOOTSTRAP_BLOCK_SIZE) -> dict:
    """
    Block bootstrap SEs for SVJ jump parameters (lambda_p, mu_j, sigma_j).
    Volatility parameters inherit SV SEs.
    """
    np.random.seed(RANDOM_SEED + 2)
    T = len(returns)
    n_blocks = int(np.ceil(T / block_size))

    lam_list = []
    mu_j_list = []
    sig_j_list = []

    for _ in range(n_reps):
        starts = np.random.randint(0, T - block_size + 1, size=n_blocks)
        idx = np.concatenate([np.arange(s, s + block_size) for s in starts])[:T]
        r_boot = returns[idx]
        try:
            svj_b = estimate_svj(r_boot, sv_params, rfr)
            lam_list.append(svj_b["lambda_p"])
            mu_j_list.append(svj_b["mu_j"])
            sig_j_list.append(svj_b["sigma_j"])
        except Exception:
            continue

    def _se(lst):
        return round(float(np.std(lst, ddof=1)), 6) if len(lst) > 1 else float("nan")

    return {
        "lambda_p_se": _se(lam_list),
        "mu_j_se": _se(mu_j_list),
        "sigma_j_se": _se(sig_j_list),
    }


# ---------------------------------------------------------------------------
# Step 8 — Sanity checks
# ---------------------------------------------------------------------------

def run_sanity_checks(bs: dict, sv: dict, svj: dict) -> None:
    """
    Halt with a clear error message if known-bad conditions are detected.
    Follows the BCJ (2009) Table 1 as a cross-reference for order-of-magnitude
    plausibility, adjusted for Indian equity premiums (typically higher than US).
    """
    errors = []
    warnings_list = []

    # BS: sigma must be between 5% and 80% for Nifty (plausible range)
    if not (0.05 <= bs["sigma"] <= 0.80):
        errors.append(f"BS sigma = {bs['sigma']:.4f} is outside [0.05, 0.80]. "
                      "Check data units — log_return should be in decimal, not %.")

    # BS: |mu| < 1.0 (100% equity risk premium would be absurd)
    if abs(bs["mu"]) > 1.0:
        errors.append(f"BS mu = {bs['mu']:.4f} has |mu| > 1.0. Likely a data-unit error.")

    # SV: kappa_v > 0 (mean-reverting volatility)
    if sv["kappa_v"] <= 0:
        errors.append(f"SV kappa_v = {sv['kappa_v']:.4f} <= 0. Volatility is not "
                      "mean-reverting. Check AR(1) on monthly RV.")

    # SV: sigma_v > 0
    if sv["sigma_v"] <= 0:
        errors.append(f"SV sigma_v = {sv['sigma_v']:.6f} <= 0. "
                      "OU residual variance was negative or zero.")

    # SV: Feller condition — 2*kappa*theta > sigma_v^2
    feller = 2.0 * sv["kappa_v"] * sv["theta"] - sv["sigma_v"] ** 2
    if feller < 0:
        warnings_list.append(
            f"SV Feller condition violated: 2*kappa*theta - sigma_v^2 = {feller:.6f} < 0. "
            "Variance process can reach zero. Results are still reported; "
            "adjust priors if running full MCMC."
        )

    # SV: rho should be negative for equity (leverage effect)
    if sv["rho"] > 0:
        warnings_list.append(
            f"SV rho = {sv['rho']:.4f} > 0. Expected negative leverage effect. "
            "Check correlation calculation."
        )

    # SVJ: lambda_p in reasonable range (0 to 10 jumps/year)
    if not (0.0 <= svj["lambda_p"] <= 10.0):
        errors.append(f"SVJ lambda_p = {svj['lambda_p']:.4f} is outside [0, 10]. "
                      "Jump detection threshold may need adjustment.")

    # SVJ: sigma_j > 0 if jumps detected
    if svj["lambda_p"] > 0 and svj["sigma_j"] <= 0:
        errors.append("SVJ sigma_j <= 0 despite nonzero lambda_p. "
                      "Fewer than 2 jump days detected.")

    for w in warnings_list:
        print(f"\n  [SANITY WARN] {w}")

    if errors:
        error_msg = "\n".join(f"  ERROR: {e}" for e in errors)
        raise RuntimeError(
            f"\n\nSanity check failed — halting before writing output.\n{error_msg}\n\n"
            "Please investigate the above issues before proceeding."
        )

    print("\n  [SANITY OK] All critical sanity checks passed.")


# ---------------------------------------------------------------------------
# Step 9 — BCJ comparison printout
# ---------------------------------------------------------------------------

def print_bcj_comparison(bs: dict, sv: dict, svj: dict) -> None:
    """Print a side-by-side comparison with BCJ (2009) S&P 500 values."""
    sep = "-" * 72
    print(f"\n{sep}")
    print("  BCJ (2009) S&P 500 vs. This Study — Nifty 50 Parameter Comparison")
    print(sep)
    print(f"  {'Parameter':<22} {'BCJ S&P500':>14} {'Nifty50':>14}")
    print(sep)

    b = BCJ_BENCHMARK
    rows = [
        ("BS: mu",         f"{b['bs']['mu']*100:.2f}%",        f"{bs['mu']*100:.2f}%"),
        ("BS: sigma",      f"{b['bs']['sigma']*100:.2f}%",     f"{bs['sigma']*100:.2f}%"),
        ("SV: kappa_v",    f"{b['sv']['kappa_v']:.4f}",        f"{sv['kappa_v']:.4f}"),
        ("SV: sqrt(theta)",f"{b['sv']['theta_sqrt']*100:.2f}%",
                           f"{np.sqrt(sv['theta'])*100:.2f}%"),
        ("SV: sigma_v",    f"{b['sv']['sigma_v']:.4f}",        f"{sv['sigma_v']:.4f}"),
        ("SV: rho",        f"{b['sv']['rho']:.4f}",            f"{sv['rho']:.4f}"),
        ("SVJ: lambda_p",  f"{b['svj']['lambda_p']:.4f}",      f"{svj['lambda_p']:.4f}"),
        ("SVJ: mu_j",      f"{b['svj']['mu_j']*100:.2f}%",     f"{svj['mu_j']*100:.2f}%"),
        ("SVJ: sigma_j",   f"{b['svj']['sigma_j']*100:.2f}%",  f"{svj['sigma_j']*100:.2f}%"),
    ]
    for label, bcj_val, nifty_val in rows:
        print(f"  {label:<22} {bcj_val:>14} {nifty_val:>14}")
    print(sep)


# ---------------------------------------------------------------------------
# Step 10 — Output: JSON and CSV
# ---------------------------------------------------------------------------

def assemble_output(bs: dict, bs_se: dict,
                    sv: dict, sv_se: dict,
                    svj: dict, svj_se: dict) -> dict:
    """Merge point estimates and SEs into the canonical output dict."""
    output = {
        "bs": {
            "mu": bs["mu"],
            "sigma": bs["sigma"],
            "mu_se": bs_se["mu_se"],
            "sigma_se": bs_se["sigma_se"],
        },
        "sv": {
            "mu": sv["mu"],
            "kappa_v": sv["kappa_v"],
            "theta": sv["theta"],
            "sigma_v": sv["sigma_v"],
            "rho": sv["rho"],
            "mu_se": sv_se["mu_se"],
            "kappa_v_se": sv_se["kappa_v_se"],
            "theta_se": sv_se["theta_se"],
            "sigma_v_se": sv_se["sigma_v_se"],
            "rho_se": sv_se["rho_se"],
        },
        "svj": {
            "mu": svj["mu"],
            "kappa_v": svj["kappa_v"],
            "theta": svj["theta"],
            "sigma_v": svj["sigma_v"],
            "rho": svj["rho"],
            "lambda_p": svj["lambda_p"],
            "mu_j": svj["mu_j"],
            "sigma_j": svj["sigma_j"],
            # SVJ volatility SEs inherited from SV
            "mu_se": sv_se["mu_se"],
            "kappa_v_se": sv_se["kappa_v_se"],
            "theta_se": sv_se["theta_se"],
            "sigma_v_se": sv_se["sigma_v_se"],
            "rho_se": sv_se["rho_se"],
            "lambda_p_se": svj_se["lambda_p_se"],
            "mu_j_se": svj_se["mu_j_se"],
            "sigma_j_se": svj_se["sigma_j_se"],
        },
    }
    return output


def write_json(output: dict, path: str) -> None:
    """Write parameter dict to JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  [OUT] Parameters written to: {path}")


def write_csv(output: dict, path: str) -> None:
    """
    Write Table 2 CSV with columns: model, parameter, estimate, std_error.
    Format follows Pillai (2012) Table 1 for continuity.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    rows = []
    # BS
    rows.append({"model": "BS", "parameter": "mu",
                 "estimate": output["bs"]["mu"],
                 "std_error": output["bs"]["mu_se"]})
    rows.append({"model": "BS", "parameter": "sigma",
                 "estimate": output["bs"]["sigma"],
                 "std_error": output["bs"]["sigma_se"]})
    # SV
    for param in ["mu", "kappa_v", "theta", "sigma_v", "rho"]:
        se_key = f"{param}_se"
        rows.append({"model": "SV", "parameter": param,
                     "estimate": output["sv"][param],
                     "std_error": output["sv"].get(se_key, float("nan"))})
    # SVJ
    for param in ["mu", "kappa_v", "theta", "sigma_v", "rho",
                  "lambda_p", "mu_j", "sigma_j"]:
        se_key = f"{param}_se"
        rows.append({"model": "SVJ", "parameter": param,
                     "estimate": output["svj"][param],
                     "std_error": output["svj"].get(se_key, float("nan"))})

    df_out = pd.DataFrame(rows, columns=["model", "parameter", "estimate", "std_error"])
    df_out.to_csv(path, index=False, float_format="%.6f")
    print(f"  [OUT] Table 2 written to: {path}")


def write_mongo(output: dict) -> None:
    """
    Attempt to write results to MongoDB collection paper_results in optionsdatabase.
    Gracefully skips if pymongo is not available or MongoDB is unreachable.
    """
    try:
        import pymongo  # noqa: PLC0415
        from dotenv import load_dotenv  # noqa: PLC0415
        load_dotenv(os.path.join(BASE_DIR, ".env"))
        mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
        client = pymongo.MongoClient(mongo_uri, serverSelectionTimeoutMS=3000)
        # Probe connection
        client.server_info()
        db = client["optionsdatabase"]
        col = db["paper_results"]
        doc = {
            "paper": "P1",
            "table": "table2_parameters",
            "script": "mcmc_estimation.py",
            "seed": RANDOM_SEED,
            "parameters": output,
        }
        col.replace_one(
            {"paper": "P1", "table": "table2_parameters"},
            doc,
            upsert=True,
        )
        print("  [OUT] Results upserted to MongoDB optionsdatabase.paper_results")
    except ImportError:
        print("  [SKIP] pymongo not installed — skipping MongoDB write.")
    except Exception as e:
        print(f"  [SKIP] MongoDB write skipped: {e}")


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def main():
    """
    Run full parameter estimation pipeline for BS, SV, and SVJ models.

    Pipeline:
        1. Load and validate inputs.
        2. Estimate BS parameters (direct analytic).
        3. Build monthly realised variance series.
        4. Estimate SV parameters (method-of-moments on monthly RV).
        5. Detect jumps via BNS bipower variation.
        6. Estimate SVJ parameters (jump-augmented SV).
        7. Block bootstrap standard errors for all three models.
        8. Sanity checks — halt on critical failures.
        9. Print BCJ comparison table.
       10. Write JSON, CSV, MongoDB.
    """
    np.random.seed(RANDOM_SEED)
    print("=" * 72)
    print("  mcmc_estimation.py — Nifty 50 Model Parameter Estimation")
    print(f"  Random seed: {RANDOM_SEED}")
    print("=" * 72)

    # 1. Load data
    print("\n[STEP 1] Loading data...")
    spot_df = load_spot_data(SPOT_CSV)
    tbill_df = load_tbill_data(TBILL_CSV)
    rfr = compute_mean_rfr(spot_df, tbill_df)
    returns = spot_df["log_return"].values.astype(float)

    # 2. Black-Scholes
    print("\n[STEP 2] Estimating BS parameters (analytic)...")
    bs_params = estimate_bs(returns, rfr)

    # 3. Monthly RV
    print("\n[STEP 3] Building monthly realised variance...")
    rv_series = build_monthly_rv(spot_df)

    # 4. Heston SV
    print("\n[STEP 4] Estimating SV parameters (method-of-moments)...")
    sv_params = estimate_sv_mom(returns, rv_series, rfr)

    # 5-6. SVJ
    print("\n[STEP 5-6] Detecting jumps and estimating SVJ parameters...")
    svj_params = estimate_svj(returns, sv_params, rfr)

    # 7. Bootstrap SEs
    print(f"\n[STEP 7] Block bootstrap SEs ({BOOTSTRAP_N_REPS} reps, "
          f"block={BOOTSTRAP_BLOCK_SIZE} days)...")
    print("  BS bootstrap...")
    bs_se = block_bootstrap_bs(returns, rfr)
    print(f"    mu_se={bs_se['mu_se']:.4f}  sigma_se={bs_se['sigma_se']:.4f}")
    print("  SV bootstrap (this may take 1-2 minutes)...")
    sv_se = block_bootstrap_sv(returns, spot_df, rfr)
    print(f"    kappa_v_se={sv_se['kappa_v_se']:.4f}  theta_se={sv_se['theta_se']:.6f}  "
          f"sigma_v_se={sv_se['sigma_v_se']:.4f}  rho_se={sv_se['rho_se']:.4f}")
    print("  SVJ bootstrap...")
    svj_se = block_bootstrap_svj(returns, sv_params, rfr)
    print(f"    lambda_p_se={svj_se['lambda_p_se']:.4f}  "
          f"mu_j_se={svj_se['mu_j_se']:.4f}  "
          f"sigma_j_se={svj_se['sigma_j_se']:.4f}")

    # 8. Sanity checks
    print("\n[STEP 8] Running sanity checks...")
    run_sanity_checks(bs_params, sv_params, svj_params)

    # 9. BCJ comparison
    print_bcj_comparison(bs_params, sv_params, svj_params)

    # 10. Output
    print("\n[STEP 10] Writing output files...")
    output = assemble_output(bs_params, bs_se, sv_params, sv_se, svj_params, svj_se)
    write_json(output, PARAMS_JSON)
    write_csv(output, TABLE2_CSV)
    write_mongo(output)

    print("\n" + "=" * 72)
    print("  DONE. Parameter estimation complete.")
    print(f"  JSON : {PARAMS_JSON}")
    print(f"  CSV  : {TABLE2_CSV}")
    print("=" * 72)


if __name__ == "__main__":
    main()
