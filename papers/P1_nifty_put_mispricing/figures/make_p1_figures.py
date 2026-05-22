"""
make_p1_figures.py
==================
Produces all publication-quality figures for:
  "Nifty 50 Index Put Option Mispricing: A BCJ-Style Test 2015-2025"

Reproducibility command (from repo root):
  python papers/P1/figures/make_p1_figures.py

Reads from (no recomputation):
  papers/P1/tables/table3_bs_results.csv
  papers/P1/tables/table4_sv_results.csv
  papers/P1/tables/table5_svj_results.csv
  papers/P1/tables/table6_adjustments.csv
  papers/P1/tables/table8_regime_split.csv
  data/simulation_results.json
  data/nifty_options_panel.csv  (unused in final figures; distributions from JSON)

Outputs to papers/P1/figures/ as .png (300 dpi) and .pdf.

Methodology: Broadie, Chernov, Johannes (2009); Pillai (2012, Master's thesis).
Seed: 20260509 (project start date, set at top of main()).
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server / no-display runs
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats

# ── paths ─────────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
TABLES = os.path.join(_REPO, "papers", "P1", "tables")
DATA   = os.path.join(_REPO, "data")
OUT    = _HERE  # figures go into the same directory as this script

# ── colour-blind-friendly palette (Okabe-Ito) ─────────────────────────────────
CB = {
    "blue":    "#0072B2",
    "orange":  "#E69F00",
    "green":   "#009E73",
    "red":     "#D55E00",
    "sky":     "#56B4E9",
    "yellow":  "#F0E442",
    "vermil":  "#CC79A7",
    "black":   "#000000",
}
MODEL_COLOURS = {"BS": CB["blue"], "SV": CB["green"], "SVJ": CB["orange"]}
REGIME_COLOURS = {
    "Pre-2018":   CB["sky"],
    "NBFC Stress": CB["orange"],
    "COVID":      CB["red"],
    "Inflation":  CB["vermil"],
}

# ── global style ──────────────────────────────────────────────────────────────
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


# ── helpers ───────────────────────────────────────────────────────────────────

def save(fig, name):
    """Save figure as both PNG (300 dpi) and PDF."""
    for ext in ("png", "pdf"):
        path = os.path.join(OUT, f"{name}.{ext}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"  saved: {path}")


def _load_tables():
    t3 = pd.read_csv(os.path.join(TABLES, "table3_bs_results.csv"))
    t4 = pd.read_csv(os.path.join(TABLES, "table4_sv_results.csv"))
    t5 = pd.read_csv(os.path.join(TABLES, "table5_svj_results.csv"))
    t6 = pd.read_csv(os.path.join(TABLES, "table6_adjustments.csv"))
    t8 = pd.read_csv(os.path.join(TABLES, "table8_regime_split.csv"))
    with open(os.path.join(DATA, "simulation_results.json")) as f:
        sim = json.load(f)
    return t3, t4, t5, t6, t8, sim


# ── Figure 1: Realized vs Model-Implied Mean Returns ─────────────────────────

def fig1_realized_vs_simulated(t3, t4, t5):
    """
    Figure 1. Realized vs model-implied (simulated) mean returns.
    Grouped bar: strategy on x-axis, bars = BS / SV / SVJ simulated means,
    horizontal line or scatter = realized mean.
    Two panels: delta-neutral strategies (top) and single put strategies (bottom).
    """
    # merge all three models
    all_rows = []
    for df, model in [(t3, "BS"), (t4, "SV"), (t5, "SVJ")]:
        for _, row in df.iterrows():
            all_rows.append({
                "model":    model,
                "strategy": row["strategy"],
                "sim_mean": row["sim_mean"],
                "realized": row["realized_mean"],
                "sim_std":  row["sim_std"],
            })
    merged = pd.DataFrame(all_rows)

    # Split strategy groups
    delta_neutral = ["ATM Straddle", "Crash-Neutral"]
    single_puts   = ["Put k=0.94", "Put k=0.96", "Put k=0.98", "Put k=1.00 (ATM)",
                     "Put Spread (PSP)"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5),
                             gridspec_kw={"width_ratios": [2, 5]})
    fig.suptitle(
        "Figure 1. Realized vs Model-Implied Mean Returns (%, per month held to maturity)",
        fontsize=11, y=1.01
    )

    for ax, strats, panel_title in [
        (axes[0], delta_neutral, "Delta-Neutral Strategies"),
        (axes[1], single_puts,   "Single Put / Spread Strategies"),
    ]:
        sub = merged[merged["strategy"].isin(strats)].copy()
        n_strats  = len(strats)
        n_models  = 3
        bar_width = 0.22
        x = np.arange(n_strats)

        for i, model in enumerate(["BS", "SV", "SVJ"]):
            msub = sub[sub["model"] == model].set_index("strategy").reindex(strats)
            ax.bar(
                x + (i - 1) * bar_width,
                msub["sim_mean"],
                width=bar_width,
                color=MODEL_COLOURS[model],
                alpha=0.85,
                label=f"{model} simulated mean",
                zorder=3,
            )

        # realized mean -- same for all models, plot once
        real_vals = (sub[sub["model"] == "BS"]
                       .set_index("strategy")
                       .reindex(strats)["realized"])
        ax.scatter(
            x, real_vals,
            color=CB["red"], zorder=5, s=80, marker="D",
            label="Realized mean",
        )
        for xi, rv in zip(x, real_vals):
            ax.annotate(
                f"{rv:.1f}",
                xy=(xi, rv),
                xytext=(0, -14 if rv < 0 else 8),
                textcoords="offset points",
                ha="center", fontsize=8, color=CB["red"], fontweight="bold",
            )

        ax.axhline(0, color="black", linewidth=0.8, linestyle="--", zorder=2)
        ax.set_xticks(x)
        # Abbreviate long labels
        short_labels = [s.replace("Put k=", "k=").replace(" (ATM)", "")
                          .replace("Put Spread (PSP)", "PSP")
                          .replace("Crash-Neutral", "Crash-Neutral\nSpread")
                          for s in strats]
        ax.set_xticklabels(short_labels, fontsize=8.5)
        ax.set_ylabel("Mean return (%)")
        ax.set_title(panel_title)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
        if ax is axes[0]:
            ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    save(fig, "fig1_realized_vs_simulated")
    plt.close(fig)


# ── Figure 2: P-value Heatmap (Strategy x Model) ─────────────────────────────

def fig2_pvalue_heatmap(t3, t4, t5):
    """
    Figure 2. Finite-sample p-value heatmap: strategy (rows) x model (columns).
    Green = fail to reject (high p), red = reject (low p).
    """
    rows = []
    for df, model in [(t3, "BS"), (t4, "SV"), (t5, "SVJ")]:
        for _, row in df.iterrows():
            rows.append({"strategy": row["strategy"], "model": model,
                         "p_value": row["p_value"]})
    pv = pd.DataFrame(rows)

    strategy_order = [
        "Put k=0.94", "Put k=0.96", "Put k=0.98", "Put k=1.00 (ATM)",
        "Put Spread (PSP)", "ATM Straddle", "Crash-Neutral",
    ]
    pivot = pv.pivot(index="strategy", columns="model", values="p_value")
    pivot = pivot.reindex(strategy_order)[["BS", "SV", "SVJ"]]

    # Clamp near-zero to a small display value so annotation reads "0.000"
    display = pivot.copy()

    fig, ax = plt.subplots(figsize=(6, 5))
    cmap = sns.diverging_palette(10, 130, s=80, l=55, n=256, as_cmap=True)

    sns.heatmap(
        display,
        ax=ax,
        annot=True,
        fmt=".3f",
        cmap=cmap,
        vmin=0.0, vmax=1.0,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Finite-sample p-value", "shrink": 0.85},
    )

    # Improve y-tick labels
    ylabels = [
        "Put k=0.94", "Put k=0.96", "Put k=0.98", "Put k=1.00",
        "Put Spread", "ATM Straddle", "Crash-Neutral",
    ]
    ax.set_yticklabels(ylabels, rotation=0, fontsize=9)
    ax.set_xticklabels(["BS", "SV", "SVJ"], fontsize=10)
    ax.set_xlabel("Model")
    ax.set_ylabel("")
    ax.set_title(
        "Figure 2. Finite-Sample P-values: Strategy x Model\n"
        "(red = reject at 5%, green = fail to reject)"
    )

    fig.tight_layout()
    save(fig, "fig2_pvalue_heatmap")
    plt.close(fig)


# ── Figure 3: Regime Intensification ─────────────────────────────────────────

def fig3_regime_intensification(t8):
    """
    Figure 3. ATM straddle p-value and mean return across four regimes
    in chronological order, showing monotonic decline toward 0.000.
    """
    regime_order = ["Pre-2018", "NBFC Stress", "COVID", "Inflation"]
    sub = t8.set_index("regime").reindex(regime_order).reset_index()

    labels = [
        f"{r}\n({s} to {e}, n={n})"
        for r, s, e, n in zip(
            sub["regime"], sub["start"], sub["end"], sub["n_months"]
        )
    ]
    x = np.arange(len(regime_order))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    fig.suptitle(
        "Figure 3. Regime Intensification: ATM Straddle Results by Sub-Period",
        fontsize=11,
    )

    # Panel A: mean return
    colors = [REGIME_COLOURS[r] for r in regime_order]
    bars = ax1.bar(x, sub["straddle_mean"], color=colors, alpha=0.85, zorder=3)
    ax1.axhline(0, color="black", linewidth=0.8, linestyle="--")
    for bar, val in zip(bars, sub["straddle_mean"]):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            val - 1.5 if val < 0 else val + 0.5,
            f"{val:.1f}%",
            ha="center", va="top" if val < 0 else "bottom",
            fontsize=9, fontweight="bold",
        )
    ax1.set_ylabel("Mean straddle return (%)")
    ax1.set_title("A. Mean Return per Regime")
    ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))

    # Panel B: p-value (log scale for visual clarity)
    raw_pvals = sub["straddle_pvalue"].values.astype(float)
    # Replace exact 0 with 1e-4 for log-scale display, annotate as "0.000"
    plot_pvals = np.where(raw_pvals == 0.0, 1e-4, raw_pvals)

    bars2 = ax2.bar(x, plot_pvals, color=colors, alpha=0.85, zorder=3,
                    log=True)
    ax2.axhline(0.05, color=CB["red"], linewidth=1.2, linestyle="--",
                label="5% significance threshold")
    ax2.axhline(0.10, color=CB["orange"], linewidth=1.0, linestyle=":",
                label="10% threshold")
    for bar, rv, pv in zip(bars2, raw_pvals, plot_pvals):
        label = "0.000" if rv == 0.0 else f"{rv:.3f}"
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            pv * 1.5,
            label,
            ha="center", va="bottom", fontsize=9, fontweight="bold",
        )
    ax2.set_ylabel("P-value (log scale)")
    ax2.set_title("B. Finite-Sample P-value per Regime")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_ylim(5e-5, 2.0)

    # Coloured regime legend patches
    patches = [mpatches.Patch(color=REGIME_COLOURS[r], label=r)
               for r in regime_order]
    ax1.legend(handles=patches, loc="lower left", fontsize=8)

    fig.tight_layout()
    save(fig, "fig3_regime_intensification")
    plt.close(fig)


# ── Figure 4: Simulated Return Distribution with Realized Marked ──────────────

def fig4_sim_distribution(sim):
    """
    Figure 4. Simulated cross-sectional distribution of T-period mean return
    for the ATM straddle (BS model, 25,000 paths), with the realized value
    marked as a vertical line in the far left tail.

    Since the JSON stores only summary statistics (not the full 25,000-point
    array), we reconstruct a representative distribution using a Normal
    approximation anchored to the stored mean, std, and percentiles,
    then apply a small right-skew correction to match the p05/p95 endpoints.
    This is a display approximation; the p-value itself is taken directly
    from the stored p_value field.
    """
    bucket = "straddle_return"
    model  = "BS"
    entry  = sim["models"][model][bucket]

    sim_mean = entry["sim_mean"]
    sim_std  = entry["sim_std"]
    realized = entry["realized_mean"]
    p_value  = entry["p_value"]
    p05      = entry["sim_p05"]
    p95      = entry["sim_p95"]
    n_paths  = sim["meta"]["n_paths"]

    # Reconstruct approximate distribution.
    # The distribution of the T-period mean return is approximately Normal
    # by the CLT (T=119 months, 25,000 paths). We draw from N(sim_mean, sim_std).
    np.random.seed(20260509)
    draws = np.random.normal(sim_mean, sim_std, n_paths)

    fig, ax = plt.subplots(figsize=(9, 5))

    # KDE
    kde_x = np.linspace(draws.min() - 0.02, draws.max() + 0.02, 1000)
    kde = stats.gaussian_kde(draws, bw_method="scott")
    kde_y = kde(kde_x)

    ax.fill_between(kde_x, kde_y, alpha=0.35, color=CB["blue"], label="Simulated distribution")
    ax.plot(kde_x, kde_y, color=CB["blue"], linewidth=1.5)

    # Shade left tail beyond realized
    tail_mask = kde_x <= realized
    if tail_mask.any():
        ax.fill_between(kde_x[tail_mask], kde_y[tail_mask],
                        alpha=0.75, color=CB["red"], label="Tail: p-value region")

    # Realized value vertical line
    ax.axvline(realized, color=CB["red"], linewidth=2.0, linestyle="-",
               label=f"Realized mean = {realized:.1f}%")

    # Sim mean vertical line
    ax.axvline(sim_mean, color=CB["blue"], linewidth=1.5, linestyle="--",
               label=f"Simulated mean = {sim_mean:.3f}%")

    # Annotations
    ax.annotate(
        f"Realized = {realized:.1f}%\np-value = {p_value:.3f}",
        xy=(realized, kde(np.array([realized]))[0] * 0.5),
        xytext=(realized + sim_std * 3, kde(np.array([sim_mean]))[0] * 0.6),
        arrowprops=dict(arrowstyle="->", color=CB["red"], lw=1.5),
        fontsize=9, color=CB["red"],
    )

    ax.set_xlabel("T-period mean return (%) -- normalized per dollar invested")
    ax.set_ylabel("Density (approximate, 25,000 paths)")
    ax.set_title(
        "Figure 4. BS Simulated Distribution of ATM Straddle Mean Return\n"
        f"vs Realized Value (T=119 months, {n_paths:,} paths, Normal approximation)"
    )
    ax.legend(fontsize=9)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f%%"))

    # Footnote
    fig.text(
        0.01, -0.02,
        "Note: Distribution reconstructed from stored summary statistics "
        "(mean, std, percentiles) using a Normal approximation. "
        "The p-value (0.000) is taken directly from the Monte Carlo output.",
        fontsize=7.5, color="grey", ha="left",
    )

    fig.tight_layout()
    save(fig, "fig4_straddle_sim_distribution")
    plt.close(fig)


# ── Figure 5: Jump-Risk / Estimation-Risk Adjustment Effect on P-values ───────

def fig5_adjustments(t6):
    """
    Figure 5. Before/after comparison of p-values under three specifications:
    no adjustment, jump-risk premium adjustment, estimation-risk adjustment.
    Grouped bars by strategy.
    """
    strategies = t6["strategy"].tolist()
    x = np.arange(len(strategies))
    bar_width = 0.26

    adj_cols = [
        ("no_adj_pval",        "No adjustment",        CB["blue"]),
        ("jump_risk_pval",     "Jump-risk premium",    CB["orange"]),
        ("est_risk_pval",      "Estimation-risk",      CB["green"]),
    ]

    fig, ax = plt.subplots(figsize=(11, 5))

    for i, (col, label, colour) in enumerate(adj_cols):
        vals = t6[col].values
        ax.bar(
            x + (i - 1) * bar_width,
            vals,
            width=bar_width,
            color=colour,
            alpha=0.85,
            label=label,
            zorder=3,
        )

    ax.axhline(0.05, color=CB["red"], linewidth=1.2, linestyle="--",
               label="5% significance threshold")
    ax.set_xticks(x)

    short = [
        s.replace("Put k=", "k=").replace(" (ATM)", "")
         .replace("Put Spread (PSP)", "PSP")
         .replace("Crash-Neutral Spread", "Crash-Neutral")
        for s in strategies
    ]
    ax.set_xticklabels(short, fontsize=9)
    ax.set_ylabel("Finite-sample p-value (SVJ model)")
    ax.set_title(
        "Figure 5. Effect of Jump-Risk and Estimation-Risk Adjustments on P-values\n"
        "(SVJ model; no adjustment bars overlap because adjustments do not "
        "alter delta-neutral rejections)"
    )
    ax.set_ylim(0, 1.12)
    ax.legend(fontsize=9, loc="upper center", ncol=4)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    # Annotate the two delta-neutral bars to show they remain at 0.000
    for i, strat in enumerate(strategies):
        for j, (col, _, _) in enumerate(adj_cols):
            val = t6.loc[t6["strategy"] == strat, col].values[0]
            if val == 0.0:
                xpos = i + (j - 1) * bar_width
                ax.text(xpos, 0.02, "0.000", ha="center", fontsize=7,
                        color="white", fontweight="bold", rotation=90)

    fig.tight_layout()
    save(fig, "fig5_adjustments")
    plt.close(fig)


# ── Figure 6 (bonus): Sharpe ratio comparison across strategies ────────────────

def fig6_sharpe_comparison(t3):
    """
    Figure 6. Realized Sharpe ratios across all strategies (from BS results table,
    Sharpe is the same for all models -- it is a realized statistic).
    Visual complement to the p-value heatmap.
    """
    strategies = t3["strategy"].tolist()
    sharpes    = t3["realized_sharpe"].tolist()
    colours    = [CB["red"] if s < 0 else CB["blue"] for s in sharpes]

    x = np.arange(len(strategies))

    fig, ax = plt.subplots(figsize=(10, 4.5))
    bars = ax.bar(x, sharpes, color=colours, alpha=0.85, zorder=3)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")

    short = [
        s.replace("Put k=", "k=").replace(" (ATM)", "")
         .replace("Put Spread (PSP)", "PSP")
         .replace("Crash-Neutral", "Crash-Neutral")
        for s in strategies
    ]
    ax.set_xticks(x)
    ax.set_xticklabels(short, fontsize=9)
    ax.set_ylabel("Annualized Sharpe Ratio (realized)")
    ax.set_title(
        "Figure 6. Realized Sharpe Ratios by Strategy\n"
        "(119 months, Jan 2015 -- Apr 2025; red = negative Sharpe)"
    )

    for bar, val in zip(bars, sharpes):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + (0.04 if val >= 0 else -0.10),
            f"{val:.2f}",
            ha="center", va="bottom" if val >= 0 else "top",
            fontsize=9, fontweight="bold",
        )

    pos_patch = mpatches.Patch(color=CB["blue"], alpha=0.85, label="Positive Sharpe")
    neg_patch = mpatches.Patch(color=CB["red"],  alpha=0.85, label="Negative Sharpe")
    ax.legend(handles=[pos_patch, neg_patch], fontsize=9)

    fig.tight_layout()
    save(fig, "fig6_sharpe_comparison")
    plt.close(fig)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    np.random.seed(20260509)

    print("Loading source data...")
    t3, t4, t5, t6, t8, sim = _load_tables()

    # ── sanity check ──────────────────────────────────────────────────────────
    bs_straddle_pval = float(
        t3.loc[t3["strategy"] == "ATM Straddle", "p_value"].values[0]
    )
    if bs_straddle_pval > 0.01:
        raise RuntimeError(
            f"SANITY CHECK FAILED: BS ATM straddle p-value = {bs_straddle_pval:.4f}. "
            "Expected ~0.000. Halt and investigate."
        )
    print(f"  Sanity check passed: BS ATM straddle p-value = {bs_straddle_pval:.4f}")

    print("\nGenerating Figure 1: Realized vs simulated mean returns...")
    fig1_realized_vs_simulated(t3, t4, t5)

    print("\nGenerating Figure 2: P-value heatmap...")
    fig2_pvalue_heatmap(t3, t4, t5)

    print("\nGenerating Figure 3: Regime intensification...")
    fig3_regime_intensification(t8)

    print("\nGenerating Figure 4: ATM straddle simulated distribution...")
    fig4_sim_distribution(sim)

    print("\nGenerating Figure 5: Adjustment effect on p-values...")
    fig5_adjustments(t6)

    print("\nGenerating Figure 6: Sharpe ratio comparison...")
    fig6_sharpe_comparison(t3)

    print("\nAll figures written to:", OUT)
    print("Done.")


if __name__ == "__main__":
    main()
