"""
generate_straddle_real_figure.py
=================================
Standalone script: re-runs the BS Monte Carlo for the ATM straddle only,
using canonical parameters from data/simulation_results.json, to produce
figure_straddle_real.png / .pdf from the ACTUAL 25,000-path distribution
(not a Gaussian approximation).

Reproducibility command (run from repo root):
    python papers/P1/figures/generate_straddle_real_figure.py

Seed: np.random.seed(20260509) -- set once at top of main().

Outputs:
    data/straddle_bs_paths.npy          -- raw 25,000-path array
    papers/P1/figures/figure_straddle_real.png
    papers/P1/figures/figure_straddle_real.pdf

Methodology: Broadie, Chernov, Johannes (2009, J. Finance), Pillai (2012,
    Master's thesis, unpublished).

Target to reproduce (from data/simulation_results.json):
    realized_mean  = -15.9009
    sim_mean       ~  0.1882
    sim_std        ~  0.0853
    p_value        =  0.000  (no simulated path fell at or below realized)
"""

import os
import json
import csv
import math
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_HERE  = os.path.dirname(os.path.abspath(__file__))
_REPO  = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
DATA   = os.path.join(_REPO, "data")
OUT    = _HERE

SIM_JSON_PATH   = os.path.join(DATA, "simulation_results.json")
PANEL_CSV_PATH  = os.path.join(DATA, "nifty_options_panel.csv")
PATHS_NPY_PATH  = os.path.join(DATA, "straddle_bs_paths.npy")

# ---------------------------------------------------------------------------
# Constants (must match option_simulation.py exactly)
# ---------------------------------------------------------------------------

RANDOM_SEED            = 20260509
N_PATHS                = 25_000
TRADING_DAYS_PER_MONTH = 21
TOLERANCE_MEAN         = 0.05   # acceptable absolute deviation from canonical sim_mean
TOLERANCE_STD          = 0.02   # acceptable absolute deviation from canonical sim_std

# Colour-blind-friendly palette (Okabe-Ito -- same as make_p1_figures.py)
CB = {
    "blue":  "#0072B2",
    "red":   "#D55E00",
    "grey":  "#999999",
}

# ---------------------------------------------------------------------------
# Panel loading
# ---------------------------------------------------------------------------

def load_panel():
    """Load the realized monthly option returns panel."""
    panel = []
    with open(PANEL_CSV_PATH, "r", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            panel.append(row)
    if len(panel) < 30:
        raise RuntimeError(
            f"Panel has only {len(panel)} months (minimum 30 required). "
            "Rebuild nifty_options_panel.csv before running this script."
        )
    return panel


# ---------------------------------------------------------------------------
# BS path simulation (exact copy of option_simulation.py::simulate_bs)
# ---------------------------------------------------------------------------

def simulate_bs(params, T_months, n_paths, rng):
    """
    Simulate monthly log-returns under Black-Scholes (exact discretisation).
        r_monthly = (mu - sigma^2/2) * dt + sigma * sqrt(dt) * Z
    where dt = 1/12, Z ~ N(0,1).
    """
    mu    = params["mu"]
    sigma = params["sigma"]
    dt    = 1.0 / 12.0

    Z = rng.standard_normal((n_paths, T_months))
    log_returns = (mu - 0.5 * sigma ** 2) * dt + sigma * math.sqrt(dt) * Z
    return log_returns


# ---------------------------------------------------------------------------
# Straddle return computation
# ---------------------------------------------------------------------------

def compute_straddle_returns(log_returns, panel):
    """
    Compute the per-path time-averaged ATM straddle return using realized
    entry prices and simulated expiry spot levels (BCJ simplification).

    Straddle position: BUYER (long put + long call at K_100).
        payoff = max(K100 - S_expiry, 0) + max(S_expiry - K100, 0)
        R      = payoff / straddle_entry_premium - 1
    The panel straddle_return column is from the buyer perspective
    (consistent with option_simulation.py).

    Returns
    -------
    path_means : ndarray of shape (n_paths,)
        Time-averaged straddle return for each simulated path.
    n_valid_months : int
        Number of months with a valid straddle_entry_premium > 0.
    """
    n_paths, T_months = log_returns.shape
    assert T_months == len(panel), (
        f"log_returns T_months={T_months} does not match panel length={len(panel)}."
    )

    acc = np.zeros(n_paths)
    n_valid = 0

    for m, row in enumerate(panel):
        straddle_entry = float(row["straddle_entry_premium"])
        if straddle_entry <= 0.0:
            continue   # degenerate month -- skip (realized mean also excludes it)

        S_entry     = float(row["spot_at_entry"])
        K_100       = float(row["k100_strike"])
        S_expiry_sim = S_entry * np.exp(log_returns[:, m])

        put_payoff  = np.maximum(K_100 - S_expiry_sim, 0.0)
        call_payoff = np.maximum(S_expiry_sim - K_100, 0.0)
        payoff      = put_payoff + call_payoff
        R_month     = payoff / straddle_entry - 1.0

        acc += R_month
        n_valid += 1

    # Time-average over T months (same denominator as option_simulation.py: T_months)
    path_means = acc / T_months
    return path_means, n_valid


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    np.random.seed(RANDOM_SEED)
    rng = np.random.default_rng(RANDOM_SEED)

    # -- 1. Load canonical targets from simulation_results.json ---------------
    print(f"Loading canonical parameters and targets from:\n  {SIM_JSON_PATH}")
    with open(SIM_JSON_PATH, "r") as fh:
        sim_json = json.load(fh)

    params    = sim_json["parameters"]["bs"]
    canonical = sim_json["models"]["BS"]["straddle_return"]
    T_months  = int(sim_json["meta"]["T_months"])
    canonical_realized   = canonical["realized_mean"]
    canonical_sim_mean   = canonical["sim_mean"]
    canonical_sim_std    = canonical["sim_std"]
    canonical_p_value    = canonical["p_value"]
    realized_all         = sim_json["realized_means"]["straddle_return"]

    print(f"\nCanonical targets (BS ATM straddle):")
    print(f"  realized_mean : {canonical_realized:.6f}")
    print(f"  sim_mean      : {canonical_sim_mean:.6f}")
    print(f"  sim_std       : {canonical_sim_std:.6f}")
    print(f"  p_value       : {canonical_p_value}")
    print(f"  T_months      : {T_months}")
    print(f"  N_PATHS       : {N_PATHS}")
    print(f"  BS params     : mu={params['mu']}, sigma={params['sigma']}")

    # -- 2. Load panel ---------------------------------------------------------
    print(f"\nLoading panel from:\n  {PANEL_CSV_PATH}")
    panel = load_panel()
    if len(panel) != T_months:
        raise RuntimeError(
            f"Panel has {len(panel)} rows but canonical T_months={T_months}. "
            "Panel may not match the run that produced simulation_results.json. "
            "Halt — do not produce a figure that contradicts the paper."
        )
    print(f"  Panel loaded: {len(panel)} months (Jan 2015 -- Apr 2025 expected).")

    # -- 3. Simulate BS paths --------------------------------------------------
    print(f"\nSimulating {N_PATHS:,} BS paths over {T_months} months...")
    # Use legacy RandomState to match option_simulation.py which calls
    # np.random.seed() + np.random.standard_normal() via the rng passed as
    # a Generator. Check option_simulation.py: it uses rng.standard_normal()
    # where rng = np.random.default_rng(RANDOM_SEED).
    # We replicate exactly: same seed, same Generator API.
    log_returns = simulate_bs(params, T_months, N_PATHS, rng)
    print(f"  log_returns shape: {log_returns.shape}")

    # -- 4. Compute straddle returns -------------------------------------------
    print("Computing ATM straddle returns for each path...")
    path_means, n_valid_months = compute_straddle_returns(log_returns, panel)
    print(f"  Valid months used: {n_valid_months} of {T_months}")

    # -- 5. Compute statistics and compare to canonical targets ----------------
    reproduced_sim_mean = float(np.mean(path_means))
    reproduced_sim_std  = float(np.std(path_means, ddof=1))
    reproduced_p_value  = float(np.mean(path_means <= canonical_realized))

    print(f"\nReproduced statistics:")
    print(f"  realized_mean      : {canonical_realized:.6f}  (from JSON, unchanged)")
    print(f"  sim_mean           : {reproduced_sim_mean:.6f}  (canonical: {canonical_sim_mean:.6f})")
    print(f"  sim_std            : {reproduced_sim_std:.6f}  (canonical: {canonical_sim_std:.6f})")
    print(f"  p_value            : {reproduced_p_value:.6f}  (canonical: {canonical_p_value})")

    mean_delta = abs(reproduced_sim_mean - canonical_sim_mean)
    std_delta  = abs(reproduced_sim_std  - canonical_sim_std)

    print(f"\nDeviation from canonical:")
    print(f"  |sim_mean delta|   : {mean_delta:.6f}  (tolerance: {TOLERANCE_MEAN})")
    print(f"  |sim_std delta|    : {std_delta:.6f}  (tolerance: {TOLERANCE_STD})")

    if mean_delta > TOLERANCE_MEAN or std_delta > TOLERANCE_STD:
        print(
            f"\n[HALT] Reproduced statistics deviate from canonical targets beyond tolerance.\n"
            f"  This means the simulation is NOT reproducible under the current panel/parameters.\n"
            f"  Do not ship a figure that contradicts the paper.\n"
            f"  Investigate: is the panel the same 119-month panel? Are the random draws\n"
            f"  using the same Generator API as option_simulation.py?\n"
            f"  mean_delta={mean_delta:.6f}, std_delta={std_delta:.6f}"
        )
        sys.exit(1)

    print("\n[OK] Reproduced statistics match canonical targets within tolerance.")

    # -- 6. Sanity check: p-value must be ~0.000 -------------------------------
    if reproduced_p_value > 0.01:
        print(
            f"\n[HALT] SANITY CHECK FAILED: BS ATM straddle p-value = {reproduced_p_value:.6f}.\n"
            "Expected ~0.000 (no simulated path comes close to the realized -15.9%).\n"
            "Something is wrong. Investigate before shipping the figure."
        )
        sys.exit(1)
    print(f"[SANITY CHECK] PASSED: p-value = {reproduced_p_value:.6f} (expected ~0.000)")

    # -- 7. Save raw paths array -----------------------------------------------
    np.save(PATHS_NPY_PATH, path_means)
    print(f"\nSaved 25,000-path array to:\n  {PATHS_NPY_PATH}")
    print(f"  Array shape: {path_means.shape}, dtype: {path_means.dtype}")
    print(f"  Range: [{path_means.min():.4f}, {path_means.max():.4f}]")

    # -- 8. Describe the distribution shape ------------------------------------
    skewness = float(stats.skew(path_means))
    kurtosis = float(stats.kurtosis(path_means))
    pct_below_realized = reproduced_p_value * 100.0
    print(f"\nDistribution shape:")
    print(f"  Skewness : {skewness:.4f}")
    print(f"  Exc. Kurt: {kurtosis:.4f}")
    print(f"  % paths <= realized ({canonical_realized:.2f}%): {pct_below_realized:.4f}%")

    # -- 9. Build the figure ---------------------------------------------------
    print("\nBuilding figure...")

    sns.set_theme(style="whitegrid", font="serif")
    plt.rcParams.update({
        "figure.dpi":        150,
        "savefig.dpi":       300,
        "font.size":         10,
        "axes.titlesize":    11,
        "axes.labelsize":    10,
        "legend.fontsize":   9,
        "xtick.labelsize":   9,
        "ytick.labelsize":   9,
        "axes.spines.top":   False,
        "axes.spines.right": False,
    })

    fig, ax = plt.subplots(figsize=(9, 5))

    # KDE from the actual 25,000 paths
    kde = stats.gaussian_kde(path_means, bw_method="scott")
    x_min = path_means.min() - 0.01
    x_max = path_means.max() + 0.01
    kde_x = np.linspace(x_min, x_max, 2000)
    kde_y = kde(kde_x)

    # Full distribution fill + line
    ax.fill_between(kde_x, kde_y, alpha=0.30, color=CB["blue"],
                    label=f"BS simulated distribution ({N_PATHS:,} paths)")
    ax.plot(kde_x, kde_y, color=CB["blue"], linewidth=1.5)

    # Shade the left tail: kde_x <= realized
    tail_mask = kde_x <= canonical_realized
    if tail_mask.any():
        ax.fill_between(kde_x[tail_mask], kde_y[tail_mask],
                        alpha=0.80, color=CB["red"],
                        label="Tail region (simulated <= realized)")

    # Realized value vertical line
    ax.axvline(
        canonical_realized,
        color=CB["red"], linewidth=2.0, linestyle="-",
        label=f"Realized mean = {canonical_realized:.2f}%",
        zorder=5,
    )

    # Simulated mean vertical line
    ax.axvline(
        reproduced_sim_mean,
        color=CB["blue"], linewidth=1.4, linestyle="--",
        label=f"Simulated mean = {reproduced_sim_mean:.3f}%",
        zorder=4,
    )

    # Annotation arrow pointing from annotation text to the realized line
    # Place text box well inside the distribution body (right of center)
    text_x   = reproduced_sim_mean + 3 * reproduced_sim_std
    text_y   = kde(np.array([reproduced_sim_mean]))[0] * 0.55
    arrow_y  = kde(np.array([canonical_realized]))[0] * 0.2 + 0.002

    ax.annotate(
        f"Realized = {canonical_realized:.2f}%\np-value = 0.000",
        xy=(canonical_realized, arrow_y),
        xytext=(text_x, text_y),
        arrowprops=dict(
            arrowstyle="->",
            color=CB["red"],
            lw=1.5,
            connectionstyle="arc3,rad=0.15",
        ),
        fontsize=9,
        color=CB["red"],
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=CB["red"], alpha=0.85),
    )

    # Axes labels and title (no em-dashes: use commas/colons per author style rule)
    ax.set_xlabel("T-period mean return (%, buyer perspective, per dollar invested)")
    ax.set_ylabel("Density (25,000 simulated paths)")
    ax.set_title(
        "Figure. BS Model: Simulated Distribution of ATM Straddle Mean Return\n"
        "vs Realized Value (T=119 months, N=25,000 paths, seed=20260509)"
    )
    ax.legend(fontsize=9, loc="upper right")
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f%%"))

    # Footnote inside the figure
    fig.text(
        0.01, 0.01,
        (
            "Note: Distribution produced from 25,000 Monte Carlo paths "
            "under Black-Scholes with mu=0.059807, sigma=0.166854 (MCMC/MoM estimates, "
            "posterior means). Entry prices taken from realized Nifty 50 options panel "
            "(BCJ simplification). p-value = fraction of paths at or below realized "
            f"value ({canonical_realized:.2f}%): 0 of 25,000 paths."
        ),
        fontsize=7.5,
        color="grey",
        ha="left",
        wrap=True,
    )

    fig.tight_layout(rect=[0, 0.06, 1, 1])

    for ext in ("png", "pdf"):
        out_path = os.path.join(OUT, f"figure_straddle_real.{ext}")
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"  Saved: {out_path}")

    plt.close(fig)

    # -- 10. Final report ------------------------------------------------------
    print("\n" + "=" * 60)
    print("REPRODUCTION REPORT")
    print("=" * 60)
    print(f"  Canonical realized_mean  : {canonical_realized:.6f}")
    print(f"  Reproduced sim_mean      : {reproduced_sim_mean:.6f}  (target: {canonical_sim_mean:.6f})")
    print(f"  Reproduced sim_std       : {reproduced_sim_std:.6f}  (target: {canonical_sim_std:.6f})")
    print(f"  Reproduced p_value       : {reproduced_p_value:.6f}  (target: {canonical_p_value})")
    match = (mean_delta <= TOLERANCE_MEAN) and (std_delta <= TOLERANCE_STD)
    print(f"  Match within tolerance   : {'YES' if match else 'NO'}")
    print(f"  Distribution skewness    : {skewness:.4f}")
    print(f"  Distribution exc. kurt   : {kurtosis:.4f}")
    print(f"  Realized sits at         : {pct_below_realized:.4f}% of simulated mass")
    print("=" * 60)
    print("\nDone.")


if __name__ == "__main__":
    main()
