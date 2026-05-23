"""
make_p3_figures.py
==================
Publication-quality figures for Paper P3:
  "Trading the Volatility Risk Premium on Nifty 50: Backtest with Realistic Frictions"
  Sole author: Sumin Pillai

Reproducibility
---------------
  python papers/P3/figures/make_p3_figures.py

Inputs (read-only, never modified)
-----------------------------------
  papers/P3/tables/strategy_results.csv
  data/p3_monthly_pnl.csv

Outputs
-------
  papers/P3/figures/figure1.png / figure1.pdf  -- Cost attribution bar
  papers/P3/figures/figure2.png / figure2.pdf  -- Net annualised return bar
  papers/P3/figures/figure3.png / figure3.pdf  -- Win-rate vs annual return scatter
  papers/P3/figures/figure4.png / figure4.pdf  -- Heatmap: ann_return x Sharpe
  papers/P3/figures/figure5.png / figure5.pdf  -- Monthly P&L time series (put-write k=0.94)

CRITICAL: the cumulative_return column in p3_monthly_pnl.csv has floating-point overflow
from mid-2023 onward and is NEVER used. All cumulative series are recomputed from net_pnl.

Random seed (required by project rules, no stochastic code here but set for reproducibility):
  np.random.seed(20260509)
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from matplotlib.patches import Patch

np.random.seed(20260509)

# ---------------------------------------------------------------------------
# Paths -- resolve relative to this script's location so the script can be
# run from any working directory.
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT   = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
FIGURES_DIR = SCRIPT_DIR

STRATEGY_CSV = os.path.join(REPO_ROOT, "papers", "P3", "tables", "strategy_results.csv")
MONTHLY_CSV  = os.path.join(REPO_ROOT, "data", "p3_monthly_pnl.csv")

DPI = 300
SEABORN_STYLE = "whitegrid"

# Colour-blind-safe palette (Paul Tol's vibrant, 7 colours)
# Order: blue, orange, red, teal, magenta, grey, dark-blue
CB_PALETTE = [
    "#0077BB",  # blue
    "#EE7733",  # orange
    "#CC3311",  # red
    "#009988",  # teal
    "#AA3377",  # magenta
    "#BBBBBB",  # grey
    "#004488",  # dark-blue
]

# Highlight colour for put-write k=0.94 (the "least adverse" strategy)
HIGHLIGHT_COLOUR = "#EE7733"   # orange
DEFAULT_COLOUR   = "#0077BB"   # blue

# Labels for strategy-moneyness combinations (no em-dashes per project rules)
LABEL_MAP = {
    ("straddle_htm",        "ATM"):    "Straddle HTM (ATM)",
    ("crash_neutral",       "ATM_CN"): "Crash-Neutral (ATM)",
    ("putwrite",            "k094"):   "Put-Write k=0.94",
    ("putwrite",            "k096"):   "Put-Write k=0.96",
    ("putwrite",            "k098"):   "Put-Write k=0.98",
    ("putwrite",            "k100"):   "Put-Write k=1.00",
    ("delta_hedged_straddle","ATM"):   "Delta-Hedged Straddle (ATM)",
}

# Preferred row order for tables and axes
ROW_ORDER = [
    "Straddle HTM (ATM)",
    "Crash-Neutral (ATM)",
    "Delta-Hedged Straddle (ATM)",
    "Put-Write k=1.00",
    "Put-Write k=0.98",
    "Put-Write k=0.96",
    "Put-Write k=0.94",
]


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def load_strategy_results() -> pd.DataFrame:
    """Load strategy_results.csv and attach display labels."""
    df = pd.read_csv(STRATEGY_CSV)
    df["label"] = df.apply(
        lambda r: LABEL_MAP.get((r["strategy"], r["moneyness"]), f"{r['strategy']} {r['moneyness']}"),
        axis=1,
    )
    df["label"] = pd.Categorical(df["label"], categories=ROW_ORDER, ordered=True)
    df = df.sort_values("label").reset_index(drop=True)
    # Derived: gross monthly P&L = net_monthly_pnl + mean_monthly_cost
    df["gross_monthly_pnl"] = df["mean_monthly_pnl"] + df["mean_monthly_cost"]
    # net_monthly_pnl is already mean_monthly_pnl
    df["net_monthly_pnl"]   = df["mean_monthly_pnl"]
    return df


def load_monthly_pnl() -> pd.DataFrame:
    """Load p3_monthly_pnl.csv. NEVER uses cumulative_return column."""
    df = pd.read_csv(MONTHLY_CSV)
    df["month_dt"] = pd.to_datetime(df["month"], format="%Y-%m")
    df["label"] = df.apply(
        lambda r: LABEL_MAP.get((r["strategy"], r["moneyness"]), f"{r['strategy']} {r['moneyness']}"),
        axis=1,
    )
    # Compute safe cumulative P&L from net_pnl (cumsum, in INR)
    df = df.sort_values(["strategy", "moneyness", "month_dt"]).reset_index(drop=True)
    df["cum_net_pnl"] = df.groupby(["strategy", "moneyness"])["net_pnl"].cumsum()
    return df


def save_figure(fig: plt.Figure, basename: str) -> None:
    """Save as both PNG (300 dpi) and PDF."""
    png_path = os.path.join(FIGURES_DIR, f"{basename}.png")
    pdf_path = os.path.join(FIGURES_DIR, f"{basename}.pdf")
    fig.savefig(png_path, dpi=DPI, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"  Saved: {png_path}")
    print(f"  Saved: {pdf_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 1: Cost Attribution Stacked Bar
# ---------------------------------------------------------------------------

def figure1_cost_attribution(df: pd.DataFrame) -> None:
    """
    Grouped bar per strategy-moneyness showing:
      - Gross mean monthly P&L (includes tail losses)
      - Mean monthly cost (secondary drag)
      - Net mean monthly P&L (result after cost)

    Visual argument: costs are a secondary drag; the gross component
    (tail losses) drives the negative result.
    """
    sns.set_style(SEABORN_STYLE)
    fig, ax = plt.subplots(figsize=(12, 6))

    labels      = df["label"].astype(str).tolist()
    x           = np.arange(len(labels))
    width       = 0.26
    offsets     = [-width, 0, width]

    # Colours: gross=blue (driven by tail), cost=grey, net=orange
    colours = {
        "Gross P&L (before costs)": "#0077BB",
        "Mean monthly cost":        "#BBBBBB",
        "Net P&L (after costs)":    "#EE7733",
    }

    bars_gross = ax.bar(x + offsets[0], df["gross_monthly_pnl"], width,
                        label="Gross P&L (before costs)",
                        color=colours["Gross P&L (before costs)"], alpha=0.9, zorder=3)
    bars_cost  = ax.bar(x + offsets[1], -df["mean_monthly_cost"], width,
                        label="Cost drag (negative sign)",
                        color=colours["Mean monthly cost"],  alpha=0.9, zorder=3)
    bars_net   = ax.bar(x + offsets[2], df["net_monthly_pnl"], width,
                        label="Net P&L (after costs)",
                        color=colours["Net P&L (after costs)"], alpha=0.9, zorder=3)

    ax.axhline(0, color="black", linewidth=0.8, zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Mean Monthly P&L (INR)", fontsize=10)
    ax.set_title(
        "Figure 1: Cost Attribution by Strategy and Moneyness\n"
        "Losses are driven by the gross component (tail events), not transaction costs",
        fontsize=11, pad=10,
    )
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(axis="y", alpha=0.4, zorder=0)
    fig.tight_layout()
    save_figure(fig, "figure1")


# ---------------------------------------------------------------------------
# Figure 2: Net Annualised Return Bar
# ---------------------------------------------------------------------------

def figure2_ann_return(df: pd.DataFrame) -> None:
    """
    Horizontal bar chart of ann_return per strategy-moneyness.
    Put-write k=0.94 highlighted as the least adverse.
    All bars are negative per the paper's confirmed key facts.
    """
    sns.set_style(SEABORN_STYLE)
    fig, ax = plt.subplots(figsize=(9, 5))

    labels = df["label"].astype(str).tolist()
    ann    = df["ann_return"].values
    y      = np.arange(len(labels))

    bar_colours = [
        HIGHLIGHT_COLOUR if lbl == "Put-Write k=0.94" else DEFAULT_COLOUR
        for lbl in labels
    ]

    bars = ax.barh(y, ann, color=bar_colours, alpha=0.9, height=0.6, zorder=3)

    # Annotate values
    for bar, val in zip(bars, ann):
        ax.text(
            min(val - 0.8, -0.8), bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}%", va="center", ha="right", fontsize=8.5, color="black",
        )

    ax.axvline(0, color="black", linewidth=0.8, zorder=4)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Annualised Net Return (%)", fontsize=10)
    ax.set_title(
        "Figure 2: Net Annualised Return by Strategy and Moneyness\n"
        "All seven strategy-moneyness combinations are net-negative after realistic costs",
        fontsize=11, pad=10,
    )
    ax.grid(axis="x", alpha=0.4, zorder=0)

    # Legend patch for highlight
    legend_elements = [
        Patch(facecolor=HIGHLIGHT_COLOUR, label="Put-Write k=0.94 (least adverse)"),
        Patch(facecolor=DEFAULT_COLOUR,   label="Other strategies"),
    ]
    ax.legend(handles=legend_elements, fontsize=9, loc="lower right")
    fig.tight_layout()
    save_figure(fig, "figure2")


# ---------------------------------------------------------------------------
# Figure 3: The Paradox Scatter (win rate vs annual return)
# ---------------------------------------------------------------------------

def figure3_paradox_scatter(df: pd.DataFrame) -> None:
    """
    X: pct_profitable_months (monthly win rate)
    Y: ann_return (%)

    Shows that high win rates (put-writes) still pair with negative annual returns,
    because rare tail losses dominate the P&L.
    """
    sns.set_style(SEABORN_STYLE)
    fig, ax = plt.subplots(figsize=(8, 6))

    win_rate = df["pct_profitable_months"].values * 100   # convert to %
    ann_ret  = df["ann_return"].values
    labels   = df["label"].astype(str).tolist()

    scatter_colours = [
        HIGHLIGHT_COLOUR if lbl == "Put-Write k=0.94" else DEFAULT_COLOUR
        for lbl in labels
    ]
    scatter_sizes = [
        120 if lbl == "Put-Write k=0.94" else 80
        for lbl in labels
    ]

    ax.scatter(win_rate, ann_ret, c=scatter_colours, s=scatter_sizes,
               zorder=4, alpha=0.92, edgecolors="white", linewidths=0.6)

    # Annotate each point
    label_offsets = {
        "Put-Write k=0.94":          ( 1.2,  0.5),
        "Put-Write k=0.96":          ( 1.2,  0.5),
        "Put-Write k=0.98":          ( 1.2,  0.5),
        "Put-Write k=1.00":          ( 1.2,  0.5),
        "Straddle HTM (ATM)":        (-1.5, -2.0),
        "Crash-Neutral (ATM)":       ( 1.2,  0.5),
        "Delta-Hedged Straddle (ATM)":( 1.2,  0.5),
    }
    for x_val, y_val, lbl in zip(win_rate, ann_ret, labels):
        dx, dy = label_offsets.get(lbl, (1.2, 0.5))
        ax.annotate(
            lbl, xy=(x_val, y_val),
            xytext=(x_val + dx, y_val + dy),
            fontsize=7.5, ha="left",
            arrowprops=dict(arrowstyle="-", lw=0.5, color="grey"),
        )

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5, zorder=3)
    ax.set_xlabel("Monthly Win Rate (%)", fontsize=10)
    ax.set_ylabel("Annualised Net Return (%)", fontsize=10)
    ax.set_title(
        "Figure 3: The Put-Write Paradox, Win Rate vs Annual Return\n"
        "High monthly win rates mask catastrophic tail losses that dominate annual P&L",
        fontsize=11, pad=10,
    )
    ax.grid(alpha=0.4, zorder=0)
    legend_elements = [
        Patch(facecolor=HIGHLIGHT_COLOUR, label="Put-Write k=0.94 (least adverse)"),
        Patch(facecolor=DEFAULT_COLOUR,   label="Other strategies"),
    ]
    ax.legend(handles=legend_elements, fontsize=9)
    fig.tight_layout()
    save_figure(fig, "figure3")


# ---------------------------------------------------------------------------
# Figure 4: Heatmap (strategy x moneyness for ann_return, annotated with Sharpe)
# ---------------------------------------------------------------------------

def figure4_heatmap(df: pd.DataFrame) -> None:
    """
    Heatmap with strategy as rows and moneyness as columns.
    Cell fill = ann_return (%), cell annotation = Sharpe ratio.
    Diverging colormap centred at 0 (all values are negative here).
    """
    sns.set_style("white")   # heatmaps look cleaner without grid lines

    # Build a 2-D pivot with strategy names on rows, moneyness on columns
    # For strategies with a single moneyness, the pivot will have NaN in others
    strategy_order = [
        "straddle_htm",
        "crash_neutral",
        "delta_hedged_straddle",
        "putwrite",
    ]
    moneyness_order = ["ATM", "ATM_CN", "k100", "k098", "k096", "k094"]

    # Pivot ann_return and sharpe
    pivot_ret    = df.pivot_table(index="strategy", columns="moneyness",
                                   values="ann_return",   aggfunc="first")
    pivot_sharpe = df.pivot_table(index="strategy", columns="moneyness",
                                   values="sharpe",       aggfunc="first")

    # Reorder
    pivot_ret    = pivot_ret.reindex(index=strategy_order, columns=moneyness_order)
    pivot_sharpe = pivot_sharpe.reindex(index=strategy_order, columns=moneyness_order)

    # Human-readable labels
    row_labels = {
        "straddle_htm":          "Straddle HTM",
        "crash_neutral":         "Crash-Neutral",
        "delta_hedged_straddle": "Delta-Hedged Straddle",
        "putwrite":              "Put-Write",
    }
    col_labels = {
        "ATM":    "ATM",
        "ATM_CN": "ATM (CN)",
        "k100":   "k=1.00",
        "k098":   "k=0.98",
        "k096":   "k=0.96",
        "k094":   "k=0.94",
    }
    pivot_ret.index    = [row_labels[r] for r in pivot_ret.index]
    pivot_ret.columns  = [col_labels[c] for c in pivot_ret.columns]
    pivot_sharpe.index   = pivot_ret.index
    pivot_sharpe.columns = pivot_ret.columns

    # Build annotation strings: "ret%\n(SR=x.xx)" for non-NaN cells
    annot_arr = pivot_ret.copy().astype(object)
    for row in pivot_ret.index:
        for col in pivot_ret.columns:
            rv = pivot_ret.loc[row, col]
            sv = pivot_sharpe.loc[row, col]
            if pd.isna(rv):
                annot_arr.loc[row, col] = ""
            else:
                annot_arr.loc[row, col] = f"{rv:.1f}%\n(SR={sv:.2f})"

    fig, ax = plt.subplots(figsize=(10, 5))

    # Use a sequential colormap that emphasises all-negative values:
    # RdYlGn reversed so deep red = most negative, light yellow = least negative
    cmap = matplotlib.cm.RdYlGn
    # Determine vmin/vmax to anchor the colourbar
    finite_vals = pivot_ret.values[~np.isnan(pivot_ret.values.astype(float))]
    vmin = np.nanmin(finite_vals)
    vmax = 0   # anchor at 0 even though no strategy has positive return

    sns.heatmap(
        pivot_ret.astype(float),
        annot=annot_arr,
        fmt="",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Annualised Net Return (%)"},
        ax=ax,
        annot_kws={"size": 9},
        mask=pivot_ret.isna(),
    )
    ax.set_title(
        "Figure 4: Heatmap of Annualised Net Return (%) with Sharpe Ratio\n"
        "All cells are net-negative; deeper red indicates greater loss",
        fontsize=11, pad=10,
    )
    ax.set_xlabel("Moneyness", fontsize=10)
    ax.set_ylabel("Strategy", fontsize=10)
    ax.tick_params(axis="x", rotation=0, labelsize=9)
    ax.tick_params(axis="y", rotation=0, labelsize=9)
    fig.tight_layout()
    save_figure(fig, "figure4")
    sns.set_style(SEABORN_STYLE)   # restore for subsequent figures


# ---------------------------------------------------------------------------
# Figure 5: Monthly Net P&L Over Time (Put-Write k=0.94)
# ---------------------------------------------------------------------------

def figure5_monthly_pnl_timeseries(df_monthly: pd.DataFrame) -> None:
    """
    Monthly net P&L for Put-Write k=0.94 as a bar chart,
    with a cumulative P&L overlay (recomputed from net_pnl, not the overflowed column).
    Large loss months (COVID March/Sep 2020) are annotated.
    """
    sns.set_style(SEABORN_STYLE)
    subset = df_monthly[
        (df_monthly["strategy"] == "putwrite") &
        (df_monthly["moneyness"] == "k094")
    ].sort_values("month_dt").reset_index(drop=True)

    months   = subset["month_dt"]
    net_pnl  = subset["net_pnl"].values
    cum_pnl  = subset["cum_net_pnl"].values   # recomputed cumsum, safe

    fig, ax1 = plt.subplots(figsize=(14, 6))

    bar_colours = [
        "#CC3311" if v < 0 else "#009988"
        for v in net_pnl
    ]
    ax1.bar(months, net_pnl, color=bar_colours, width=20, alpha=0.85, zorder=3, label="Monthly Net P&L (INR)")
    ax1.axhline(0, color="black", linewidth=0.8, zorder=4)
    ax1.set_ylabel("Monthly Net P&L (INR)", fontsize=10)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))

    # Cumulative P&L on secondary axis
    ax2 = ax1.twinx()
    ax2.plot(months, cum_pnl, color="#004488", linewidth=1.8, zorder=5, label="Cumulative Net P&L (INR)")
    ax2.axhline(0, color="#004488", linewidth=0.4, linestyle=":", zorder=4)
    ax2.set_ylabel("Cumulative Net P&L (INR)", fontsize=10, color="#004488")
    ax2.tick_params(axis="y", colors="#004488")
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))

    # Annotate large loss months
    annotate_months = {
        "2020-03": "Mar 2020\n(COVID crash)",
        "2020-09": "Sep 2020\n(post-COVID spike)",
    }
    for m_str, label in annotate_months.items():
        row = subset[subset["month_dt"] == pd.Timestamp(m_str)]
        if not row.empty:
            idx_val = row["month_dt"].values[0]
            y_val   = row["net_pnl"].values[0]
            ax1.annotate(
                label,
                xy=(idx_val, y_val),
                xytext=(idx_val, y_val - 1200),
                fontsize=8, ha="center", color="#CC3311",
                arrowprops=dict(arrowstyle="->", lw=0.8, color="#CC3311"),
            )

    ax1.set_title(
        "Figure 5: Monthly Net P&L, Put-Write k=0.94 (January 2015 to April 2025)\n"
        "Tail events in 2020 dominate; 91.6% of months are profitable but two months wipe accumulated gains",
        fontsize=11, pad=10,
    )

    # Combined legend
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    profit_patch = Patch(facecolor="#009988", label="Profitable month")
    loss_patch   = Patch(facecolor="#CC3311", label="Loss month")
    ax1.legend(handles=[profit_patch, loss_patch] + h2, fontsize=9, loc="upper left")

    ax1.set_xlabel("Month", fontsize=10)
    fig.tight_layout()
    save_figure(fig, "figure5")


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------

def run_sanity_checks(df_strat: pd.DataFrame) -> None:
    """
    Verify that key facts from the paper are consistent with the CSV data.
    Hard-stops with an error message if a critical inconsistency is found.
    """
    errors = []
    warnings_list = []

    # All 7 strategy-moneyness combos should have negative ann_return
    positive = df_strat[df_strat["ann_return"] > 0]
    if not positive.empty:
        errors.append(
            f"CRITICAL: {len(positive)} strategy-moneyness combinations have POSITIVE "
            f"ann_return -- contradicts paper's confirmed finding:\n{positive[['label','ann_return']]}"
        )

    # Put-write k=0.94 should have the highest (least negative) ann_return
    pw94 = df_strat[df_strat["label"].astype(str) == "Put-Write k=0.94"]
    if not pw94.empty:
        pw94_ret = pw94["ann_return"].values[0]
        others   = df_strat[df_strat["label"].astype(str) != "Put-Write k=0.94"]["ann_return"]
        if not (others < pw94_ret).all():
            warnings_list.append(
                f"WARNING: Put-Write k=0.94 (ann_return={pw94_ret:.2f}%) is NOT the "
                f"least adverse. Some other strategy has a higher (less negative) ann_return."
            )
        # Check approx -0.9%
        if abs(pw94_ret - (-0.933529)) > 0.01:
            warnings_list.append(
                f"WARNING: Put-write k=0.94 ann_return={pw94_ret:.4f}% vs paper claim ~-0.9%"
            )
        # Check win rate ~91.6%
        pw94_win = pw94["pct_profitable_months"].values[0] * 100
        if abs(pw94_win - 91.5966) > 0.1:
            warnings_list.append(
                f"WARNING: put-write k=0.94 win rate={pw94_win:.2f}% vs paper claim ~91.6%"
            )
    else:
        errors.append("CRITICAL: Put-Write k=0.94 not found in strategy_results.csv")

    # Straddle variants should be around -44 to -45%
    straddle = df_strat[df_strat["strategy"] == "straddle_htm"]
    if not straddle.empty:
        st_ret = straddle["ann_return"].values[0]
        if not (-46 <= st_ret <= -43):
            warnings_list.append(
                f"WARNING: Straddle HTM ann_return={st_ret:.2f}%, expected ~-44 to -45%"
            )

    if errors:
        print("\n" + "=" * 70)
        for e in errors:
            print("SANITY CHECK FAILED:", e)
        print("=" * 70)
        raise RuntimeError(
            "Sanity checks failed. Halting. Investigate before reporting any result."
        )

    print("\nSanity checks passed.")
    for w in warnings_list:
        print(w)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 70)
    print("make_p3_figures.py -- Paper P3 Figure Generator")
    print("=" * 70)

    # Verify source files exist
    for path in (STRATEGY_CSV, MONTHLY_CSV):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Source file not found: {path}\n"
                "Run from the repo root or ensure data files are present."
            )

    print("\nLoading data...")
    df_strat   = load_strategy_results()
    df_monthly = load_monthly_pnl()

    print(f"  strategy_results: {len(df_strat)} rows")
    print(f"  p3_monthly_pnl:   {len(df_monthly)} rows across "
          f"{df_monthly[['strategy','moneyness']].drop_duplicates().shape[0]} strategy-moneyness combos")
    print()

    # Sanity checks before any figure is produced
    run_sanity_checks(df_strat)

    # Generate figures
    print("Generating figures...")

    print("\nFigure 1: Cost attribution bar...")
    figure1_cost_attribution(df_strat)

    print("\nFigure 2: Net annualised return bar...")
    figure2_ann_return(df_strat)

    print("\nFigure 3: Win-rate vs annual return scatter...")
    figure3_paradox_scatter(df_strat)

    print("\nFigure 4: Heatmap (ann_return + Sharpe)...")
    figure4_heatmap(df_strat)

    print("\nFigure 5: Monthly P&L time series (put-write k=0.94)...")
    figure5_monthly_pnl_timeseries(df_monthly)

    print("\n" + "=" * 70)
    print("All figures written to:", FIGURES_DIR)
    print("=" * 70)


if __name__ == "__main__":
    main()
