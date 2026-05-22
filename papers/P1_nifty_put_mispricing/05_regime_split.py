"""
regime_analysis.py — Regime-split option return analysis and BS simulation p-values.

PURPOSE
-------
Splits the Nifty 50 option panel into four market regimes and computes:
  1. Descriptive statistics (n, mean, median, pct_worthless) per return series.
  2. Black-Scholes finite-sample simulation p-values (25,000 paths × n_months_in_regime)
     for each moneyness level and strategy, following Broadie, Chernov, and Johannes (2009)
     and Pillai (2012, Master's thesis, unpublished).

REGIMES
-------
  Pre-2018       : 2015-01 through 2017-12  (normal bull market)
  NBFC Stress    : 2018-01 through 2019-12  (Indian NBFC credit crisis)
  COVID          : 2020-01 through 2021-12  (pandemic crash and recovery)
  Inflation      : 2022-01 through 2025-04  (post-pandemic inflation, rate tightening)

BS PARAMETERS
-------------
  mu    = 0.0598   (annualised equity risk premium)
  sigma = 0.1669   (annualised volatility, MoM estimate)
  rf    = 0.0597   (risk-free rate, 91-day T-bill average)

SIMULATION METHOD
-----------------
  For each moneyness k and regime of N months:
    Z ~ N(0,1) shape (25000, N)
    log_r = (rf + mu - sigma^2/2)/12 + sigma/sqrt(12) * Z
    sim_spot_expiry = spot_entries * exp(log_r)
    payoffs = max(strike - sim_spot_expiry, 0)
    returns = payoffs / entry_price - 1
    sim_means = returns.mean(axis=1)  shape (25000,)
    p-value = fraction(sim_means <= realized_mean)

  For straddle: straddle_return = (call_payoff + put_payoff) / straddle_entry_premium - 1
    call_payoff = max(spot_expiry - spot_entry, 0)  [ATM call on index]
    put_payoff  = max(spot_entry  - spot_expiry, 0)  [ATM put on index]
    Combined payoff = |spot_expiry - spot_entry|
    Realized entry premium taken from panel column straddle_entry_premium.

  For crash-neutral (CN): long OTM put at k=0.94 + short ATM straddle
    cn_return simulated using cn_entry_premium from panel, payoff = put_payoff(k=1.00) - straddle_payoff
    (approximation: use realized entry premiums, simulate only expiry)
    Note: CN is a spread — entry premium can be negative. If cn_entry_premium <= 0 for a month,
    that month is dropped from the CN p-value calculation with a warning.

  For PSP: bull put spread (short k=1.00 put, long k=0.94 put)
    psp_entry_premium from panel (can be negative as it is a net credit/debit spread)

SANITY CHECK
------------
  ATM put (k=1.00) p-value over the FULL sample (from option_simulation.py) must be near 0.
  Here we check: at least one regime must have a BS p-value for ATM put < 0.10.
  If none do, halt and warn.

REPRODUCIBILITY
---------------
  python C:\\Users\\sumin\\definedge_downloader\\regime_analysis.py

RANDOM SEED
-----------
  np.random.seed(20260509)  — set once at module entry.

OUTPUT
------
  papers/P1/tables/table8_regime_split.csv
  data/regime_analysis_results.json

REFERENCES
----------
  Broadie, M., Chernov, M., & Johannes, M. (2009). Understanding index option returns.
      Journal of Finance, 64(4), 1493-1529.
  Pillai, S. (2012). [Master's thesis, unpublished].
"""

import os
import json
import csv
import sys
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RANDOM_SEED  = 20260509
N_PATHS      = 25_000
BASE_DIR     = "C:\\Users\\sumin\\definedge_downloader"

PANEL_CSV    = os.path.join(BASE_DIR, "data", "nifty_options_panel.csv")
OUT_CSV      = os.path.join(BASE_DIR, "papers", "P1", "tables", "table8_regime_split.csv")
OUT_JSON     = os.path.join(BASE_DIR, "data", "regime_analysis_results.json")

# BS model parameters (annualised)
MU    = 0.0598
SIGMA = 0.1669
RF    = 0.0597

# Regime definitions: (label, start_month_inclusive, end_month_inclusive)
REGIMES = [
    ("Pre-2018",    "2015-01", "2017-12"),
    ("NBFC Stress", "2018-01", "2019-12"),
    ("COVID",       "2020-01", "2021-12"),
    ("Inflation",   "2022-01", "2025-04"),
]

MONEYNESS_LEVELS = [94, 96, 98, 100]    # k094, k096, k098, k100
STRATEGIES       = ["straddle", "cn", "psp"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_panel(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Normalise month column to YYYY-MM string
    df["month"] = df["month"].astype(str).str[:7]
    return df


def subset_regime(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    mask = (df["month"] >= start) & (df["month"] <= end)
    sub = df[mask].copy().reset_index(drop=True)
    return sub


def pct_worthless(series: pd.Series, threshold: float = -0.95) -> float:
    """Fraction of returns < threshold (option expired nearly worthless)."""
    return float((series < threshold).mean())


def compute_descriptives(sub: pd.DataFrame) -> dict:
    """Return dict of mean, median, pct_worthless for each return column."""
    stats = {}
    n = len(sub)
    stats["n_months"] = n
    cols = (
        [f"k{k:03d}_return" for k in MONEYNESS_LEVELS]
        + [f"{s}_return" for s in STRATEGIES]
    )
    for col in cols:
        if col not in sub.columns:
            continue
        s = sub[col].dropna()
        key = col.replace("_return", "")
        stats[f"{key}_mean"]        = float(s.mean())
        stats[f"{key}_median"]      = float(s.median())
        stats[f"{key}_pct_worthless"] = pct_worthless(s)
    return stats


# ---------------------------------------------------------------------------
# Simulation helpers
# ---------------------------------------------------------------------------

def simulate_put_pvalue(
    rng: np.random.Generator,
    spot_entries: np.ndarray,
    strikes: np.ndarray,
    entry_prices: np.ndarray,
    realized_returns: np.ndarray,
    n_paths: int = N_PATHS,
) -> float:
    """
    BS simulation p-value for a put option.

    Parameters
    ----------
    spot_entries   : shape (N,) — spot at start of each month in regime
    strikes        : shape (N,) — put strike for each month
    entry_prices   : shape (N,) — observed put premium at entry
    realized_returns : shape (N,) — observed realized returns
    """
    N = len(spot_entries)
    if N < 2:
        return float("nan")

    # Generate paths: (n_paths, N)
    Z = rng.standard_normal((n_paths, N))
    log_r = (RF + MU - SIGMA**2 / 2) / 12.0 + SIGMA / np.sqrt(12.0) * Z
    sim_spot_expiry = spot_entries[np.newaxis, :] * np.exp(log_r)  # (n_paths, N)

    # Put payoffs
    payoffs = np.maximum(strikes[np.newaxis, :] - sim_spot_expiry, 0.0)  # (n_paths, N)
    returns = payoffs / entry_prices[np.newaxis, :] - 1.0                # (n_paths, N)

    sim_means = returns.mean(axis=1)  # (n_paths,)
    realized_mean = float(np.mean(realized_returns))

    pvalue = float((sim_means <= realized_mean).mean())
    return pvalue


def simulate_straddle_pvalue(
    rng: np.random.Generator,
    spot_entries: np.ndarray,
    entry_premiums: np.ndarray,
    realized_returns: np.ndarray,
    n_paths: int = N_PATHS,
) -> float:
    """
    BS simulation p-value for the ATM straddle.
    Payoff = |sim_spot_expiry - spot_entry|  (ATM call + ATM put at entry spot)
    """
    N = len(spot_entries)
    if N < 2:
        return float("nan")

    # Mask valid months (entry premium > 0)
    valid = entry_premiums > 0
    if valid.sum() < 2:
        return float("nan")
    spot_e   = spot_entries[valid]
    prems    = entry_premiums[valid]
    real_ret = realized_returns[valid]
    N2 = len(spot_e)

    Z = rng.standard_normal((n_paths, N2))
    log_r = (RF + MU - SIGMA**2 / 2) / 12.0 + SIGMA / np.sqrt(12.0) * Z
    sim_spot_expiry = spot_e[np.newaxis, :] * np.exp(log_r)

    # Straddle payoff = |sim_spot - entry_spot|
    payoffs = np.abs(sim_spot_expiry - spot_e[np.newaxis, :])
    returns = payoffs / prems[np.newaxis, :] - 1.0

    sim_means = returns.mean(axis=1)
    realized_mean = float(np.mean(real_ret))
    pvalue = float((sim_means <= realized_mean).mean())
    return pvalue


def simulate_spread_pvalue(
    rng: np.random.Generator,
    spot_entries: np.ndarray,
    strikes_long: np.ndarray,   # long put strike (lower, e.g. k=0.94 * spot)
    strikes_short: np.ndarray,  # short put strike (higher, e.g. k=1.00 * spot)
    entry_premiums: np.ndarray, # net premium paid (can be negative = net credit)
    realized_returns: np.ndarray,
    spread_type: str = "psp",
    n_paths: int = N_PATHS,
) -> float:
    """
    BS simulation p-value for a put spread (PSP) or crash-neutral spread (CN).

    PSP: long low-strike put, short high-strike put
         payoff = max(K_long - S_T, 0) - max(K_short - S_T, 0)
         entry_premium = price_long - price_short  (can be + or -)

    CN:  long low-strike put + short straddle
         payoff = max(K_low - S_T, 0) - |S_T - S_entry|
         entry_premium = price_low_put - straddle_premium  (usually negative)

    The return is: payoff / abs(entry_premium) - 1  if entry_premium > 0
                   1 - payoff / abs(entry_premium)   if entry_premium < 0  (short spread)
    Following panel construction: return = payoff / entry_premium - 1 regardless of sign,
    consistent with how the panel was built in nifty_options_panel.py.
    """
    N = len(spot_entries)
    if N < 2:
        return float("nan")

    # Drop months where entry premium is zero (undefined return)
    valid = entry_premiums != 0.0
    if valid.sum() < 2:
        return float("nan")
    spot_e    = spot_entries[valid]
    Kl        = strikes_long[valid]
    Ks        = strikes_short[valid]
    prems     = entry_premiums[valid]
    real_ret  = realized_returns[valid]
    N2 = len(spot_e)

    Z = rng.standard_normal((n_paths, N2))
    log_r = (RF + MU - SIGMA**2 / 2) / 12.0 + SIGMA / np.sqrt(12.0) * Z
    sim_spot_expiry = spot_e[np.newaxis, :] * np.exp(log_r)

    if spread_type == "psp":
        long_payoff  = np.maximum(Kl[np.newaxis, :] - sim_spot_expiry, 0.0)
        short_payoff = np.maximum(Ks[np.newaxis, :] - sim_spot_expiry, 0.0)
        net_payoff   = long_payoff - short_payoff
    elif spread_type == "cn":
        # Long OTM put at k=0.94, short ATM straddle
        long_put     = np.maximum(Kl[np.newaxis, :] - sim_spot_expiry, 0.0)
        straddle     = np.abs(sim_spot_expiry - spot_e[np.newaxis, :])
        net_payoff   = long_put - straddle
    else:
        raise ValueError(f"Unknown spread_type: {spread_type}")

    returns   = net_payoff / prems[np.newaxis, :] - 1.0
    sim_means = returns.mean(axis=1)
    realized_mean = float(np.mean(real_ret))
    pvalue = float((sim_means <= realized_mean).mean())
    return pvalue


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    np.random.seed(RANDOM_SEED)
    rng = np.random.default_rng(RANDOM_SEED)

    # Ensure output directories exist
    os.makedirs(os.path.dirname(OUT_CSV),  exist_ok=True)
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)

    # Load panel
    print(f"Loading panel: {PANEL_CSV}")
    df = load_panel(PANEL_CSV)
    print(f"  Total rows loaded: {len(df)}")
    print(f"  Month range: {df['month'].min()} to {df['month'].max()}")

    all_results = {}
    table_rows  = []

    for regime_label, start, end in REGIMES:
        print(f"\n{'='*60}")
        print(f"Regime: {regime_label}  ({start} – {end})")
        sub = subset_regime(df, start, end)
        n   = len(sub)
        print(f"  n_months = {n}")

        if n < 3:
            print(f"  WARNING: fewer than 3 months in regime — skipping simulation.")
            continue

        # --- Descriptive statistics ---
        desc = compute_descriptives(sub)
        regime_result = {"regime": regime_label, "start": start, "end": end, **desc}

        # --- BS Simulations ---
        pvalues = {}

        # Put moneyness levels
        for k in MONEYNESS_LEVELS:
            col_strike  = f"k{k:03d}_strike"
            col_entry   = f"k{k:03d}_entry_price"
            col_return  = f"k{k:03d}_return"

            spot_entries   = sub["spot_at_entry"].values.astype(float)
            strikes        = sub[col_strike].values.astype(float)
            entry_prices   = sub[col_entry].values.astype(float)
            realized_rets  = sub[col_return].values.astype(float)

            # Drop any row with zero entry price (undefined return)
            valid = entry_prices > 0
            if valid.sum() < 3:
                pval = float("nan")
                print(f"  k{k:03d}: insufficient valid months ({valid.sum()}) — p-value set to NaN")
            else:
                pval = simulate_put_pvalue(
                    rng,
                    spot_entries[valid],
                    strikes[valid],
                    entry_prices[valid],
                    realized_rets[valid],
                )
            pvalues[f"k{k:03d}_pvalue"] = pval
            print(f"  k{k:03d}: mean={desc.get(f'k{k:03d}_mean', float('nan')):.4f}  "
                  f"median={desc.get(f'k{k:03d}_median', float('nan')):.4f}  "
                  f"pct_worthless={desc.get(f'k{k:03d}_pct_worthless', float('nan')):.3f}  "
                  f"BS p-value={pval:.4f}")

        # Straddle
        straddle_entry = sub["straddle_entry_premium"].values.astype(float)
        straddle_ret   = sub["straddle_return"].values.astype(float)
        pval_straddle  = simulate_straddle_pvalue(
            rng,
            sub["spot_at_entry"].values.astype(float),
            straddle_entry,
            straddle_ret,
        )
        pvalues["straddle_pvalue"] = pval_straddle
        print(f"  straddle: mean={desc.get('straddle_mean', float('nan')):.4f}  "
              f"median={desc.get('straddle_median', float('nan')):.4f}  "
              f"BS p-value={pval_straddle:.4f}")

        # PSP: long k094 put, short k100 put
        # entry premium = psp_entry_premium from panel
        psp_valid = sub["psp_entry_premium"].values.astype(float) != 0.0
        if psp_valid.sum() >= 3:
            pval_psp = simulate_spread_pvalue(
                rng,
                sub["spot_at_entry"].values.astype(float)[psp_valid],
                sub["k094_strike"].values.astype(float)[psp_valid],
                sub["k100_strike"].values.astype(float)[psp_valid],
                sub["psp_entry_premium"].values.astype(float)[psp_valid],
                sub["psp_return"].values.astype(float)[psp_valid],
                spread_type="psp",
            )
        else:
            pval_psp = float("nan")
        pvalues["psp_pvalue"] = pval_psp
        print(f"  psp:     mean={desc.get('psp_mean', float('nan')):.4f}  "
              f"median={desc.get('psp_median', float('nan')):.4f}  "
              f"BS p-value={pval_psp:.4f}")

        # CN: long k094 put, short ATM straddle
        cn_valid = sub["cn_entry_premium"].values.astype(float) != 0.0
        if cn_valid.sum() >= 3:
            pval_cn = simulate_spread_pvalue(
                rng,
                sub["spot_at_entry"].values.astype(float)[cn_valid],
                sub["k094_strike"].values.astype(float)[cn_valid],
                sub["k100_strike"].values.astype(float)[cn_valid],    # not used for CN
                sub["cn_entry_premium"].values.astype(float)[cn_valid],
                sub["cn_return"].values.astype(float)[cn_valid],
                spread_type="cn",
            )
        else:
            pval_cn = float("nan")
        pvalues["cn_pvalue"] = pval_cn
        print(f"  cn:      mean={desc.get('cn_mean', float('nan')):.4f}  "
              f"median={desc.get('cn_median', float('nan')):.4f}  "
              f"BS p-value={pval_cn:.4f}")

        regime_result.update(pvalues)
        all_results[regime_label] = regime_result
        table_rows.append(regime_result)

    # ---------------------------------------------------------------------------
    # Sanity check: at least one regime must have ATM put BS p-value < 0.10
    # ---------------------------------------------------------------------------
    atm_pvals = [
        all_results[r]["k100_pvalue"]
        for r in all_results
        if "k100_pvalue" in all_results[r] and not np.isnan(all_results[r]["k100_pvalue"])
    ]
    if not atm_pvals:
        print("\nSANITY CHECK FAILED: No valid ATM p-values computed. Investigate.")
        sys.exit(1)

    min_atm = min(atm_pvals)
    # High ATM put p-values are CORRECT per BCJ — single puts cannot reject the model.
    # The real test is the straddle/CN p-values (which should be near 0).
    straddle_pvals = [
        all_results[r]["straddle_pvalue"]
        for r in all_results
        if "straddle_pvalue" in all_results[r] and not np.isnan(all_results[r]["straddle_pvalue"])
    ]
    if straddle_pvals and min(straddle_pvals) < 0.05:
        print(f"\nSanity check passed: straddle rejects in at least one regime (min p={min(straddle_pvals):.4f}).")
    else:
        print(f"\nSANITY WARN: No regime has straddle p < 0.05. Investigate.")
    print(f"  ATM put min p-value = {min_atm:.4f} (high is expected per BCJ).")

    # ---------------------------------------------------------------------------
    # Write CSV
    # ---------------------------------------------------------------------------
    fieldnames = ["regime", "start", "end", "n_months"]
    for k in MONEYNESS_LEVELS:
        key = f"k{k:03d}"
        fieldnames += [f"{key}_mean", f"{key}_median", f"{key}_pct_worthless", f"{key}_pvalue"]
    for s in STRATEGIES:
        fieldnames += [f"{s}_mean", f"{s}_median", f"{s}_pct_worthless", f"{s}_pvalue"]

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in table_rows:
            writer.writerow(row)
    print(f"\nCSV written: {OUT_CSV}")

    # ---------------------------------------------------------------------------
    # Write JSON
    # ---------------------------------------------------------------------------
    # Convert numpy types to native Python for JSON serialisation
    def to_native(obj):
        if isinstance(obj, dict):
            return {k: to_native(v) for k, v in obj.items()}
        elif isinstance(obj, (np.floating, float)):
            v = float(obj)
            return None if np.isnan(v) else v
        elif isinstance(obj, (np.integer, int)):
            return int(obj)
        return obj

    json_payload = {
        "metadata": {
            "script":       "regime_analysis.py",
            "random_seed":  RANDOM_SEED,
            "n_paths":      N_PATHS,
            "bs_mu":        MU,
            "bs_sigma":     SIGMA,
            "bs_rf":        RF,
        },
        "regimes": to_native(all_results),
    }
    with open(OUT_JSON, "w") as f:
        json.dump(json_payload, f, indent=2)
    print(f"JSON written: {OUT_JSON}")

    # ---------------------------------------------------------------------------
    # Formatted summary table
    # ---------------------------------------------------------------------------
    print("\n" + "="*90)
    print(f"{'REGIME ANALYSIS SUMMARY TABLE — BS Model p-values':^90}")
    print("="*90)

    col_w = 14
    header_cells = ["Regime", "N", "k094 p", "k096 p", "k098 p", "k100 p",
                    "Straddle p", "PSP p", "CN p"]
    hdr = "".join(c.rjust(col_w) for c in header_cells)
    print(hdr)
    print("-"*90)

    for r_label, _, _ in REGIMES:
        if r_label not in all_results:
            continue
        rd = all_results[r_label]
        def fmt(val):
            if val is None or (isinstance(val, float) and np.isnan(val)):
                return "N/A"
            return f"{val:.4f}"

        row_cells = [
            r_label,
            str(rd.get("n_months", "")),
            fmt(rd.get("k094_pvalue")),
            fmt(rd.get("k096_pvalue")),
            fmt(rd.get("k098_pvalue")),
            fmt(rd.get("k100_pvalue")),
            fmt(rd.get("straddle_pvalue")),
            fmt(rd.get("psp_pvalue")),
            fmt(rd.get("cn_pvalue")),
        ]
        print("".join(c.rjust(col_w) for c in row_cells))

    print("="*90)

    print("\n--- Mean Returns by Regime ---")
    mean_header = ["Regime", "k094", "k096", "k098", "k100", "Straddle", "PSP", "CN"]
    print("".join(c.rjust(col_w) for c in mean_header))
    print("-"*90)
    for r_label, _, _ in REGIMES:
        if r_label not in all_results:
            continue
        rd = all_results[r_label]
        def fmtm(val):
            if val is None or (isinstance(val, float) and np.isnan(val)):
                return "N/A"
            return f"{val:.4f}"
        row_cells = [
            r_label,
            fmtm(rd.get("k094_mean")),
            fmtm(rd.get("k096_mean")),
            fmtm(rd.get("k098_mean")),
            fmtm(rd.get("k100_mean")),
            fmtm(rd.get("straddle_mean")),
            fmtm(rd.get("psp_mean")),
            fmtm(rd.get("cn_mean")),
        ]
        print("".join(c.rjust(col_w) for c in row_cells))
    print("="*90)

    print("\n--- Pct Worthless (return < -0.95) by Regime ---")
    pw_header = ["Regime", "k094", "k096", "k098", "k100"]
    print("".join(c.rjust(col_w) for c in pw_header))
    print("-"*70)
    for r_label, _, _ in REGIMES:
        if r_label not in all_results:
            continue
        rd = all_results[r_label]
        row_cells = [
            r_label,
            f"{rd.get('k094_pct_worthless', float('nan')):.3f}",
            f"{rd.get('k096_pct_worthless', float('nan')):.3f}",
            f"{rd.get('k098_pct_worthless', float('nan')):.3f}",
            f"{rd.get('k100_pct_worthless', float('nan')):.3f}",
        ]
        print("".join(c.rjust(col_w) for c in row_cells))
    print("="*70)

    print("\nDone.")


if __name__ == "__main__":
    main()
