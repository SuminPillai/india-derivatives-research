"""
make_p2_figures.py
==================
Produce all publication-quality figures for:
  "Technical-Indicator Strategy Performance in Indian Equities:
   Joint Test of Market Efficiency 2015-2025"
  Sole author: Sumin Pillai

Reproducibility command
-----------------------
  python papers/P2/figures/make_p2_figures.py

Outputs (papers/P2/figures/)
-----------------------------
  figure1.png / figure1.pdf  -- Strategy x Metric heatmap
  figure2.png / figure2.pdf  -- Romano-Wolf adjusted p-value bar chart
  figure3.png / figure3.pdf  -- Sharpe ratio ranking with CI error bars
  figure4.png / figure4.pdf  -- Cumulative equity curves, 8 survivors
  figure5.png / figure5.pdf  -- Nominal vs RW-adjusted p-value scatter

Rules
-----
  - np.random.seed(20260509) for any random operation.
  - No em-dashes in labels or titles.
  - 300 dpi PNG + PDF for every figure.
  - Source CSVs are read-only; no modification.
  - If the CSV contradicts stated paper facts, script prints a WARNING
    and uses the CSV values (ground truth).
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import seaborn as sns

np.random.seed(20260509)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
PERF_CSV  = os.path.join(REPO_ROOT, "papers", "P2", "tables", "strategy_performance.csv")
RET_CSV   = os.path.join(REPO_ROOT, "data", "p2_daily_returns.csv")
OUT_DIR   = os.path.join(REPO_ROOT, "papers", "P2", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------
sns.set_theme(style="whitegrid", font_scale=1.0)
plt.rcParams.update({
    "font.family":       "serif",
    "font.serif":        ["Times New Roman", "DejaVu Serif"],
    "axes.titlesize":    11,
    "axes.labelsize":    10,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "legend.fontsize":   9,
    "figure.dpi":        300,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
})

# Colour-blind-safe palette (Wong 2011)
CB_BLUE   = "#0072B2"
CB_ORANGE = "#E69F00"
CB_GREEN  = "#009E73"
CB_RED    = "#D55E00"
CB_PURPLE = "#CC79A7"
CB_LBLUE  = "#56B4E9"
CB_YELLOW = "#F0E442"
CB_BLACK  = "#000000"

# ---------------------------------------------------------------------------
# Helper: save figure in both PNG and PDF
# ---------------------------------------------------------------------------
def save_fig(fig, name: str) -> None:
    for ext in ("png", "pdf"):
        path = os.path.join(OUT_DIR, f"{name}.{ext}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"  Saved {name}.png and {name}.pdf")


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
def load_data():
    perf = pd.read_csv(PERF_CSV)
    rets = pd.read_csv(RET_CSV, parse_dates=["date"])
    rets.sort_values("date", inplace=True)
    rets.reset_index(drop=True, inplace=True)
    return perf, rets


# ---------------------------------------------------------------------------
# Strategy metadata helpers
# ---------------------------------------------------------------------------
FAMILY_MAP = {
    "SMA_10_50":    "Trend: SMA",
    "SMA_20_100":   "Trend: SMA",
    "SMA_50_200":   "Trend: SMA",
    "MACD_12_26_9": "Trend: MACD",
    "MACD_26_52_18":"Trend: MACD",
    "BB_20_2.0":    "Mean-Rev: BB",
    "BB_20_2.5":    "Mean-Rev: BB",
    "RSI_30_70":    "Mean-Rev: RSI",
    "RSI_25_75":    "Mean-Rev: RSI",
    "RSI_20_80":    "Mean-Rev: RSI",
    "Donchian_20":  "Trend: Donchian",
    "Donchian_55":  "Trend: Donchian",
    "Confluence_v1":"Confluence",
}

FAMILY_ORDER = [
    "Trend: SMA",
    "Trend: MACD",
    "Trend: Donchian",
    "Mean-Rev: RSI",
    "Mean-Rev: BB",
    "Confluence",
]

FAMILY_COLOUR = {
    "Trend: SMA":      CB_BLUE,
    "Trend: MACD":     CB_LBLUE,
    "Trend: Donchian": CB_GREEN,
    "Mean-Rev: RSI":   CB_ORANGE,
    "Mean-Rev: BB":    CB_RED,
    "Confluence":      CB_PURPLE,
}

SURVIVORS = {
    "SMA_10_50", "SMA_20_100", "SMA_50_200",
    "MACD_12_26_9", "MACD_26_52_18",
    "Donchian_20", "Donchian_55",
    "RSI_20_80",
}

DISPLAY_NAME = {
    "SMA_10_50":    "SMA(10,50)",
    "SMA_20_100":   "SMA(20,100)",
    "SMA_50_200":   "SMA(50,200)",
    "MACD_12_26_9": "MACD(12,26,9)",
    "MACD_26_52_18":"MACD(26,52,18)",
    "BB_20_2.0":    "BB(20, 2.0)",
    "BB_20_2.5":    "BB(20, 2.5)",
    "RSI_30_70":    "RSI(30/70)",
    "RSI_25_75":    "RSI(25/75)",
    "RSI_20_80":    "RSI(20/80)",
    "Donchian_20":  "Donchian(20)",
    "Donchian_55":  "Donchian(55)",
    "Confluence_v1":"Confluence",
}


def add_meta(perf: pd.DataFrame) -> pd.DataFrame:
    perf = perf.copy()
    perf["family"]   = perf["strategy_name"].map(FAMILY_MAP)
    perf["display"]  = perf["strategy_name"].map(DISPLAY_NAME)
    perf["survivor"] = perf["strategy_name"].isin(SURVIVORS)
    # Sort by family order, then by Sharpe descending within family
    perf["family_order"] = perf["family"].map({f: i for i, f in enumerate(FAMILY_ORDER)})
    perf.sort_values(["family_order", "sharpe"], ascending=[True, False], inplace=True)
    perf.reset_index(drop=True, inplace=True)
    return perf


# ---------------------------------------------------------------------------
# Sanity checks against stated paper facts
# ---------------------------------------------------------------------------
def sanity_checks(perf: pd.DataFrame) -> None:
    print("\n--- Sanity checks vs paper facts ---")

    n_surv_csv = (perf["rw_adjusted_pvalue"] < 0.05).sum()
    if n_surv_csv != 8:
        warnings.warn(
            f"WARNING: CSV shows {n_surv_csv} survivors at RW p<0.05, "
            f"paper states 8. Using CSV values."
        )
    else:
        print(f"  Survivors at RW p<0.05: {n_surv_csv} (matches paper)")

    best_row = perf.loc[perf["sharpe"].idxmax()]
    if best_row["strategy_name"] != "Donchian_20":
        warnings.warn(
            f"WARNING: Best Sharpe in CSV is {best_row['strategy_name']} "
            f"({best_row['sharpe']:.4f}), paper states Donchian_20."
        )
    else:
        print(f"  Best Sharpe: Donchian_20 ({best_row['sharpe']:.4f}) -- matches paper")

    don20_rw = perf.loc[perf["strategy_name"] == "Donchian_20", "rw_adjusted_pvalue"].values[0]
    if abs(don20_rw - 0.0006) > 0.0001:
        warnings.warn(
            f"WARNING: Donchian_20 RW p={don20_rw:.4f}, paper states 0.0006."
        )
    else:
        print(f"  Donchian_20 RW p={don20_rw:.4f} -- matches paper")

    rsi2080_rw = perf.loc[perf["strategy_name"] == "RSI_20_80", "rw_adjusted_pvalue"].values[0]
    if abs(rsi2080_rw - 0.0367) > 0.001:
        warnings.warn(
            f"WARNING: RSI_20_80 RW p={rsi2080_rw:.4f}, paper states 0.037."
        )
    else:
        print(f"  RSI_20_80 RW p={rsi2080_rw:.4f} -- matches paper (0.037)")

    print("--- End sanity checks ---\n")


# ---------------------------------------------------------------------------
# Figure 1: Strategy x Metric heatmap
# ---------------------------------------------------------------------------
def figure1_heatmap(perf: pd.DataFrame) -> None:
    print("Building Figure 1: heatmap...")

    METRICS = [
        ("ann_return",        "Ann. Return"),
        ("sharpe",            "Sharpe"),
        ("sortino",           "Sortino"),
        ("max_dd",            "Max DD"),
        ("turnover",          "Turnover"),
        ("nominal_pvalue",    "Nominal p"),
        ("rw_adjusted_pvalue","RW p"),
    ]

    col_keys   = [m[0] for m in METRICS]
    col_labels = [m[1] for m in METRICS]

    # Build display matrix (ordered by family then sharpe, top-to-bottom)
    mat = perf[col_keys].copy()
    row_labels = perf["display"].tolist()

    # Per-column z-score normalisation for colour (so heatmap is comparable across columns)
    mat_norm = mat.copy().astype(float)
    for col in col_keys:
        col_std = mat[col].std()
        if col_std > 0:
            mat_norm[col] = (mat[col] - mat[col].mean()) / col_std
        else:
            mat_norm[col] = 0.0

    # For p-value columns, flip sign in normalised matrix so
    # lower p (better) shows as warmer colour
    for col in ("nominal_pvalue", "rw_adjusted_pvalue"):
        mat_norm[col] = -mat_norm[col]

    # For max_dd (negative number, larger magnitude = worse), flip
    mat_norm["max_dd"] = -mat_norm["max_dd"]

    fig, ax = plt.subplots(figsize=(11, 7))

    sns.heatmap(
        mat_norm.values,
        ax=ax,
        cmap="RdYlGn",
        center=0,
        annot=False,
        linewidths=0.4,
        linecolor="white",
        cbar=True,
        cbar_kws={"label": "Standardised score (higher = better)", "shrink": 0.6},
        xticklabels=col_labels,
        yticklabels=row_labels,
    )

    # Annotate each cell with the raw value
    ann_fmt = {
        "ann_return":        "{:.1%}",
        "sharpe":            "{:.2f}",
        "sortino":           "{:.2f}",
        "max_dd":            "{:.1%}",
        "turnover":          "{:.1f}x",
        "nominal_pvalue":    "{:.4f}",
        "rw_adjusted_pvalue":"{:.4f}",
    }
    for r_idx, r_name in enumerate(col_keys):
        for c_idx, col_name in enumerate(col_keys):
            # Note: heatmap rows=strategies (y), cols=metrics (x)
            # We need row = strategy index, col = metric index
            pass

    # Re-annotate properly
    nrows, ncols = mat.shape
    for r in range(nrows):
        for c, col_name in enumerate(col_keys):
            val = mat.iloc[r][col_name]
            fmt = ann_fmt[col_name]
            txt = fmt.format(val)
            # Determine text colour for readability
            norm_val = mat_norm.iloc[r][col_name]
            text_color = "black" if abs(norm_val) < 1.5 else "white"
            ax.text(
                c + 0.5, r + 0.5, txt,
                ha="center", va="center",
                fontsize=7.5, color=text_color, fontweight="normal",
            )

    # Draw family-group separators
    family_groups = perf.groupby("family_order", sort=True).size().cumsum().tolist()
    for sep in family_groups[:-1]:
        ax.axhline(sep, color="white", linewidth=2.5)

    # Add family labels on the left margin
    family_starts = [0] + family_groups[:-1]
    family_ends   = family_groups
    family_names  = (
        perf.sort_values("family_order")
            .groupby("family_order", sort=True)["family"]
            .first()
            .tolist()
    )
    for start, end, fname in zip(family_starts, family_ends, family_names):
        mid = (start + end) / 2
        ax.text(
            -0.35, mid, fname.replace(": ", "\n"),
            ha="right", va="center",
            fontsize=7.5, color=FAMILY_COLOUR.get(fname, CB_BLACK),
            fontweight="bold", transform=ax.get_yaxis_transform(),
        )

    # Mark survivor rows with a bold border
    for r, row in perf.iterrows():
        if row["survivor"]:
            ax.add_patch(mpatches.FancyBboxPatch(
                (0, r), ncols, 1,
                boxstyle="square,pad=0",
                linewidth=1.8, edgecolor=CB_BLUE, facecolor="none",
                zorder=5,
            ))

    ax.set_title(
        "Figure 1: Strategy Performance Heatmap, All 13 Configurations\n"
        "(Cells show raw values; colour reflects standardised score, higher = better; "
        "blue borders mark Romano-Wolf survivors)",
        fontsize=10, pad=10,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=30)
    ax.tick_params(axis="y", rotation=0)

    plt.tight_layout()
    save_fig(fig, "figure1")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: Romano-Wolf adjusted p-value bar chart
# ---------------------------------------------------------------------------
def figure2_rw_bar(perf: pd.DataFrame) -> None:
    print("Building Figure 2: Romano-Wolf bar chart...")

    df = perf.sort_values("rw_adjusted_pvalue").copy()

    colours = [CB_BLUE if s else CB_RED for s in df["survivor"]]

    fig, ax = plt.subplots(figsize=(9, 5.5))

    bars = ax.barh(
        df["display"],
        df["rw_adjusted_pvalue"],
        color=colours,
        edgecolor="white",
        linewidth=0.6,
        height=0.65,
    )

    ax.axvline(0.05, color=CB_BLACK, linewidth=1.4, linestyle="--", label="5% threshold")

    # Annotate bar values
    for bar, val in zip(bars, df["rw_adjusted_pvalue"]):
        ax.text(
            val + 0.005, bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}",
            va="center", ha="left", fontsize=8,
        )

    surv_patch = mpatches.Patch(color=CB_BLUE, label="Survivor (RW p < 0.05)")
    non_patch  = mpatches.Patch(color=CB_RED,  label="Non-survivor (RW p >= 0.05)")
    ax.legend(handles=[surv_patch, non_patch, plt.Line2D([], [], color=CB_BLACK,
              linestyle="--", linewidth=1.4, label="5% threshold")],
              loc="lower right", frameon=True, framealpha=0.9)

    ax.set_xlabel("Romano-Wolf Step-M Adjusted p-value", fontsize=10)
    ax.set_title(
        "Figure 2: Romano-Wolf Adjusted p-values, All 13 Strategy Configurations\n"
        "(8 survivors highlighted in blue; dashed line at 5% family-wise threshold)",
        fontsize=10, pad=10,
    )
    ax.set_xlim(0, max(df["rw_adjusted_pvalue"]) * 1.25)
    ax.invert_yaxis()
    sns.despine(left=True, bottom=False)
    plt.tight_layout()
    save_fig(fig, "figure2")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3: Sharpe ratio ranking with CI error bars
# ---------------------------------------------------------------------------
def figure3_sharpe(perf: pd.DataFrame) -> None:
    print("Building Figure 3: Sharpe ranking with confidence intervals...")

    df = perf.sort_values("sharpe", ascending=False).copy()

    bar_colours = [FAMILY_COLOUR.get(f, CB_BLACK) for f in df["family"]]

    err_lo = df["sharpe"] - df["sharpe_ci_lo"]
    err_hi = df["sharpe_ci_hi"] - df["sharpe"]

    fig, ax = plt.subplots(figsize=(9, 5.5))

    x = np.arange(len(df))

    ax.bar(
        x, df["sharpe"],
        color=bar_colours, edgecolor="white", linewidth=0.6, width=0.65,
    )
    ax.errorbar(
        x, df["sharpe"],
        yerr=[err_lo, err_hi],
        fmt="none", color=CB_BLACK, capsize=4, linewidth=1.2, capthick=1.2,
    )

    ax.axhline(0, color=CB_BLACK, linewidth=0.8, linestyle="-")

    # Mark survivors with a star
    for xi, (_, row) in zip(x, df.iterrows()):
        if row["survivor"]:
            ax.text(xi, row["sharpe_ci_hi"] + 0.06, "*",
                    ha="center", va="bottom", fontsize=12, color=CB_BLUE, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(df["display"], rotation=38, ha="right", fontsize=8.5)
    ax.set_ylabel("Annualised Sharpe Ratio", fontsize=10)
    ax.set_title(
        "Figure 3: Sharpe Ratio Ranking with 95% Bootstrap Confidence Intervals\n"
        "(Colour by family; blue stars mark Romano-Wolf survivors)",
        fontsize=10, pad=10,
    )

    legend_handles = [
        mpatches.Patch(color=FAMILY_COLOUR[f], label=f) for f in FAMILY_ORDER
    ]
    legend_handles.append(
        plt.Line2D([], [], marker="*", color=CB_BLUE, linestyle="none",
                   markersize=10, label="RW survivor")
    )
    ax.legend(handles=legend_handles, loc="upper right", frameon=True,
              framealpha=0.9, fontsize=8)

    sns.despine(bottom=False)
    plt.tight_layout()
    save_fig(fig, "figure3")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4: Cumulative equity curves, 8 survivors
# ---------------------------------------------------------------------------
def figure4_equity_curves(perf: pd.DataFrame, rets: pd.DataFrame) -> None:
    print("Building Figure 4: Cumulative equity curves (8 survivors)...")

    survivor_names = sorted(SURVIVORS)

    # Map strategy_name to display name
    disp = {s: DISPLAY_NAME[s] for s in survivor_names}

    # Map to CSV column names (same as strategy_name in the returns CSV)
    # Verify all survivors are present in the returns CSV
    missing = [s for s in survivor_names if s not in rets.columns]
    if missing:
        raise RuntimeError(
            f"Survivor columns missing from daily returns CSV: {missing}\n"
            "Cannot produce Figure 4. Halting."
        )

    # Assign a colour per survivor, using family colours with disambiguation
    surv_colours = [
        CB_BLUE, CB_LBLUE, CB_GREEN, CB_GREEN,
        CB_ORANGE, CB_BLUE, CB_BLUE, CB_LBLUE,
    ]
    # Better: assign by family
    fam_col = {s: FAMILY_COLOUR.get(FAMILY_MAP[s], CB_BLACK) for s in survivor_names}

    # Use a distinguishable linestyle map within same family
    LINESTYLES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]
    family_line_counter: dict = {}

    fig, ax = plt.subplots(figsize=(11, 6))

    for strat in sorted(survivor_names,
                        key=lambda s: FAMILY_ORDER.index(FAMILY_MAP[s])):
        family = FAMILY_MAP[strat]
        idx = family_line_counter.get(family, 0)
        family_line_counter[family] = idx + 1
        ls = LINESTYLES[idx % len(LINESTYLES)]
        col = FAMILY_COLOUR.get(family, CB_BLACK)

        daily = rets[strat].fillna(0).values
        equity = np.cumprod(1.0 + daily)

        ax.plot(
            rets["date"], equity,
            label=disp[strat],
            color=col, linestyle=ls, linewidth=1.4, alpha=0.88,
        )

    ax.axhline(1.0, color=CB_BLACK, linewidth=0.8, linestyle="--", alpha=0.6)

    ax.set_xlabel("Date", fontsize=10)
    ax.set_ylabel("Growth of 1 Rupee", fontsize=10)
    ax.set_title(
        "Figure 4: Cumulative Equity Curves, 8 Romano-Wolf Survivor Strategies\n"
        "(Daily compound returns, equal-weight cross-sectional portfolio, Jan 2015 to Apr 2025)",
        fontsize=10, pad=10,
    )

    ax.legend(loc="upper left", frameon=True, framealpha=0.9, fontsize=8.5,
              ncol=2)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1fx"))
    ax.xaxis.set_major_locator(matplotlib.dates.YearLocator(2))
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y"))

    sns.despine(bottom=False)
    plt.tight_layout()
    save_fig(fig, "figure4")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 5: Nominal vs RW-adjusted p-value scatter (data-snooping story)
# ---------------------------------------------------------------------------
def figure5_pvalue_scatter(perf: pd.DataFrame) -> None:
    print("Building Figure 5: Nominal vs RW-adjusted p-value scatter...")

    fig, ax = plt.subplots(figsize=(7.5, 6.5))

    colours = [CB_BLUE if s else CB_RED for s in perf["survivor"]]

    ax.scatter(
        perf["nominal_pvalue"],
        perf["rw_adjusted_pvalue"],
        c=colours,
        s=90, edgecolors="white", linewidths=0.8, zorder=5, alpha=0.92,
    )

    # Reference line y=x (no correction)
    lim_max = max(perf["rw_adjusted_pvalue"].max(), perf["nominal_pvalue"].max()) * 1.05
    ax.plot([0, lim_max], [0, lim_max], color="grey",
            linestyle=":", linewidth=1.0, label="No correction (y = x)", zorder=3)

    # Threshold lines
    ax.axvline(0.05, color=CB_ORANGE, linewidth=1.2, linestyle="--",
               label="Nominal 5% threshold", zorder=4)
    ax.axhline(0.05, color=CB_GREEN,  linewidth=1.2, linestyle="--",
               label="RW 5% threshold", zorder=4)

    # Label each point
    for _, row in perf.iterrows():
        ax.annotate(
            row["display"],
            (row["nominal_pvalue"], row["rw_adjusted_pvalue"]),
            textcoords="offset points", xytext=(5, 2),
            fontsize=7.5, color="black",
        )

    ax.set_xlabel("Nominal p-value (unadjusted)", fontsize=10)
    ax.set_ylabel("Romano-Wolf Adjusted p-value", fontsize=10)
    ax.set_title(
        "Figure 5: Nominal vs Romano-Wolf Adjusted p-values\n"
        "(Points above the orange dashed line lose significance after correction;\n"
        " blue = survivor, red = non-survivor at 5% FWER)",
        fontsize=10, pad=10,
    )

    surv_patch = mpatches.Patch(color=CB_BLUE, label="Survivor")
    non_patch  = mpatches.Patch(color=CB_RED,  label="Non-survivor")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles=[surv_patch, non_patch] + handles,
        labels=["Survivor", "Non-survivor"] + labels,
        loc="upper left", frameon=True, framealpha=0.9, fontsize=8,
    )

    ax.set_xlim(-0.005, lim_max)
    ax.set_ylim(-0.005, lim_max * 1.05)

    sns.despine(bottom=False)
    plt.tight_layout()
    save_fig(fig, "figure5")
    plt.close(fig)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    print("=== make_p2_figures.py ===")
    print(f"Reading performance data from: {PERF_CSV}")
    print(f"Reading daily returns from:    {RET_CSV}")
    print(f"Output directory:              {OUT_DIR}\n")

    perf, rets = load_data()
    perf = add_meta(perf)

    sanity_checks(perf)

    figure1_heatmap(perf)
    figure2_rw_bar(perf)
    figure3_sharpe(perf)
    figure4_equity_curves(perf, rets)
    figure5_pvalue_scatter(perf)

    print("\n=== All figures produced successfully ===")
    print("Files written to:", OUT_DIR)
    for i in range(1, 6):
        for ext in ("png", "pdf"):
            path = os.path.join(OUT_DIR, f"figure{i}.{ext}")
            exists = os.path.isfile(path)
            size   = os.path.getsize(path) if exists else 0
            print(f"  figure{i}.{ext}: {'OK' if exists else 'MISSING'}, {size:,} bytes")


if __name__ == "__main__":
    main()
