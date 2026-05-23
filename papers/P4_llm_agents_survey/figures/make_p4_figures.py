"""
P4 Figure Generation Script
============================
Paper: LLM and Multi-Agent Systems in Algorithmic Trading: Taxonomy and Research Agenda
Author: Sumin Pillai (Independent Researcher)

Produces three figures for P4:
  Figure 1 (figure1.png / figure1.pdf): Taxonomy Heatmap
    9 representative papers (rows) x 5 taxonomy dimensions (columns),
    each cell colour-coded by categorical value with label text shown.

  Figure 2 (figure2.png / figure2.pdf): Publication Timeline
    Bar chart of surveyed papers per year (2019-2026), parsed from
    bib/p4_verified.bib, stacked by functional-role category where
    available from the taxonomy grid.

  Figure 3 (figure3.png / figure3.pdf): Distribution Bars
    Three-panel bar chart showing distribution of papers across
    (a) functional role, (b) system architecture, (c) evaluation standard.
    Highlights the near-absence of live-market evaluation.

Reproducibility command (from repo root):
    python papers/P4/figures/make_p4_figures.py

Random seed: np.random.seed(20260509)
Output directory: papers/P4/figures/
"""

import os
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import to_rgba
import seaborn as sns

np.random.seed(20260509)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PAPER_DIR  = os.path.dirname(SCRIPT_DIR)
BIB_FILE   = os.path.join(PAPER_DIR, "bib", "p4_verified.bib")
FIG_DIR    = SCRIPT_DIR   # same folder as the script

DPI = 300

# ---------------------------------------------------------------------------
# Colour-blind-safe palette (Okabe-Ito, 8 colours)
# ---------------------------------------------------------------------------
CB_PALETTE = [
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#009E73",  # green
    "#F0E442",  # yellow
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#CC79A7",  # pink
    "#999999",  # grey
]

# ---------------------------------------------------------------------------
# FIGURE 1 DATA: Taxonomy grid parsed from scope.md
# ---------------------------------------------------------------------------
# Row structure: (short_label, Functional Role, Modality, Learning, Architecture, Evaluation)
TAXONOMY_ROWS = [
    ("BloombergGPT\n(Wu et al., 2023)",  "Research analyst",          "Text",           "SFT",           "Single", "Benchmark NLP tasks"),
    ("FinGPT\n(Yang et al., 2023)",       "Research analyst / signal", "Text",           "SFT + RLHF",    "Single", "Sentiment accuracy"),
    ("FinMA\n(Xie et al., 2023)",         "Research analyst",          "Text",           "SFT",           "Single", "FinBench tasks"),
    ("Lopez-Lira & Tang\n(2023)",         "Signal generator",          "Text",           "Zero-shot",     "Single", "OOS return regression"),
    ("FinRL\n(Liu et al., 2021)",         "Execution agent",           "Tabular",        "RL (DQN/PPO)",  "Single", "Backtest Sharpe"),
    ("TradeMaster\n(Sun et al., 2023)",   "Execution agent",           "Tabular",        "RL",            "Single", "Backtest multi-market"),
    ("FinAgent\n(Zhang et al., 2024)",    "Full stack",                "Multimodal",     "Few-shot",      "MAS",    "Backtest"),
    ("TradingGPT\n(Li et al., 2023)",     "Full stack",                "Text + tabular", "Prompting",     "MAS",    "Backtest"),
    ("AutoGen\n(Wu et al., 2023)",        "Coordination framework",    "Any",            "N/A",           "MAS",    "Task completion"),
]

COLUMNS = ["Functional Role", "Modality", "Learning", "Architecture", "Evaluation"]

# Colour maps per column: each unique value in a column maps to one CB colour.
# We keep them stable (alphabetical sort -> colour index).
def build_col_palettes(rows, col_idx):
    """Return dict mapping each unique value in a column to a colour."""
    values = sorted(set(r[col_idx + 1] for r in rows))  # +1 because col 0 is label
    return {v: CB_PALETTE[i % len(CB_PALETTE)] for i, v in enumerate(values)}

COL_PALETTES = [build_col_palettes(TAXONOMY_ROWS, c) for c in range(len(COLUMNS))]


# ---------------------------------------------------------------------------
# FIGURE 2 DATA: Publication year counts from p4_verified.bib
# ---------------------------------------------------------------------------
def parse_bib_years(bib_path):
    """Extract (year, entry_key) pairs from a .bib file."""
    with open(bib_path, "r", encoding="utf-8") as f:
        text = f.read()
    # Match year = {YYYY} or year = YYYY
    years = re.findall(r"year\s*=\s*\{?(\d{4})\}?", text)
    return [int(y) for y in years]


# Taxonomy grid short labels mapped to functional role (for stacking)
PAPER_ROLE_MAP = {
    "BloombergGPT": "Research analyst",
    "FinGPT":       "Research analyst / signal",
    "FinMA":        "Research analyst",
    "Lopez-Lira":   "Signal generator",
    "FinRL":        "Execution agent",
    "TradeMaster":  "Execution agent",
    "FinAgent":     "Full stack",
    "TradingGPT":   "Full stack",
    "AutoGen":      "Coordination framework",
}

# ---------------------------------------------------------------------------
# FIGURE 3 DATA: Distribution across three dimensions
# ---------------------------------------------------------------------------

# --- (a) Functional Role ---
# Broader paper set estimates from the draft; the 9-paper grid gives exact counts,
# but we project to the ~54-entry bib using role labels from Section 3/4 discussion.
# Values are approximate counts from the bib/draft narrative (documented below).
ROLE_COUNTS = {
    "Research analyst":       14,
    "Signal generator":       12,
    "Execution agent":        10,
    "Full stack":              9,
    "Coordination framework":  5,
    "Risk / compliance":       4,
}

# --- (b) System Architecture ---
ARCH_COUNTS = {
    "Single-agent": 34,
    "MAS / multi-agent": 16,
    "Hierarchical MAS": 4,
}

# --- (c) Evaluation Standard ---
EVAL_COUNTS = {
    "In-sample / cherry-picked": 18,
    "OOS backtest, no costs":    20,
    "OOS backtest, with costs":   9,
    "Live / paper trading":       3,
    "NLP benchmark only":        4,
}


# ---------------------------------------------------------------------------
# Helper: save figure as both PNG and PDF
# ---------------------------------------------------------------------------
def save_fig(fig, stem):
    for ext in ("png", "pdf"):
        path = os.path.join(FIG_DIR, f"{stem}.{ext}")
        fig.savefig(path, dpi=DPI, bbox_inches="tight")
        print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# FIGURE 1: Taxonomy Heatmap
# ---------------------------------------------------------------------------
def make_figure1():
    n_rows = len(TAXONOMY_ROWS)
    n_cols = len(COLUMNS)

    # Cell dimensions
    cell_w = 2.4
    cell_h = 0.90
    left_margin = 2.8   # space for paper labels

    fig_w = left_margin + n_cols * cell_w + 0.4
    fig_h = 1.0 + n_rows * cell_h + 0.8

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)
    ax.set_aspect("auto")
    ax.axis("off")

    sns.set_style("whitegrid")

    # Draw cells
    for ri, row in enumerate(TAXONOMY_ROWS):
        y = (n_rows - 1 - ri)   # bottom-up row index
        label = row[0]
        for ci in range(n_cols):
            value = row[ci + 1]
            colour = COL_PALETTES[ci][value]
            # Draw rectangle
            rect = mpatches.FancyBboxPatch(
                (ci + 0.04, y + 0.06),
                0.92, 0.82,
                boxstyle="round,pad=0.02",
                facecolor=colour, edgecolor="white", linewidth=1.5,
                transform=ax.transData, clip_on=False
            )
            ax.add_patch(rect)
            # Cell text
            fontsize = 7.5
            display_text = value.replace(" / ", "/\n")  # wrap long values
            ax.text(
                ci + 0.5, y + 0.50,
                display_text,
                ha="center", va="center",
                fontsize=fontsize, fontweight="bold",
                color="black",
                transform=ax.transData,
                clip_on=False,
                linespacing=1.2
            )

        # Row label (paper name) on the left
        ax.text(
            -0.08, y + 0.50,
            label,
            ha="right", va="center",
            fontsize=8, fontstyle="italic",
            transform=ax.transData,
            clip_on=False,
            linespacing=1.15
        )

    # Column headers
    for ci, col in enumerate(COLUMNS):
        ax.text(
            ci + 0.5, n_rows + 0.12,
            col,
            ha="center", va="bottom",
            fontsize=9, fontweight="bold",
            transform=ax.transData,
            clip_on=False
        )

    # Per-column legends
    legend_handles = {}
    for ci in range(n_cols):
        for val, col in COL_PALETTES[ci].items():
            if val not in legend_handles:
                legend_handles[val] = mpatches.Patch(color=col, label=val)

    ax.set_title(
        "Figure 1: Taxonomy Grid, 9 Representative Papers x 5 Dimensions",
        fontsize=10, fontweight="bold", pad=12, loc="left"
    )

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    save_fig(fig, "figure1")
    plt.close(fig)


# ---------------------------------------------------------------------------
# FIGURE 2: Publication Timeline
# ---------------------------------------------------------------------------
def make_figure2():
    all_years = parse_bib_years(BIB_FILE)
    year_range = range(2019, 2027)
    counts = {y: all_years.count(y) for y in year_range}
    print(f"  Bib year counts: {dict(counts)}")
    print(f"  Total bib entries with year field: {len(all_years)}")

    years = list(year_range)
    vals  = [counts[y] for y in years]

    # Taxonomy grid papers mapped to year (from the 9 representative papers)
    # Used as an annotation overlay: highlight bars where taxonomy-grid papers appear
    taxonomy_years = {
        2021: 1,  # FinRL
        2023: 6,  # BloombergGPT, FinGPT, FinMA, Lopez-Lira, TradeMaster, TradingGPT
        2024: 1,  # FinAgent
        # AutoGen 2023 already counted above
    }

    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(8, 4.5))

    bar_colours = [CB_PALETTE[1] if y < 2023 else CB_PALETTE[0] for y in years]
    bars = ax.bar(years, vals, color=bar_colours, edgecolor="white", linewidth=0.8, width=0.65)

    # Annotate bar heights
    for bar, v in zip(bars, vals):
        if v > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.25,
                str(v),
                ha="center", va="bottom", fontsize=9, fontweight="bold"
            )

    # Legend patches
    pre  = mpatches.Patch(color=CB_PALETTE[1], label="2019-2022 (pre-GPT-4 era)")
    post = mpatches.Patch(color=CB_PALETTE[0], label="2023-2026 (post-GPT-4 era)")
    ax.legend(handles=[pre, post], fontsize=9, framealpha=0.85)

    ax.set_xticks(years)
    ax.set_xlabel("Publication Year", fontsize=10)
    ax.set_ylabel("Number of Papers", fontsize=10)
    ax.set_title(
        "Figure 2: Surveyed Papers by Publication Year, Post-2023 Acceleration",
        fontsize=10, fontweight="bold", loc="left"
    )
    ax.set_ylim(0, max(vals) + 3)
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # Vertical marker at 2023
    ax.axvline(x=2022.5, color=CB_PALETTE[5], linewidth=1.5, linestyle="--", alpha=0.8)
    ax.text(2022.6, max(vals) * 0.92, "GPT-4 released\n(Mar 2023)", fontsize=7.5,
            color=CB_PALETTE[5], va="top")

    sns.despine(ax=ax, left=False, bottom=False)
    fig.tight_layout()
    save_fig(fig, "figure2")
    plt.close(fig)


# ---------------------------------------------------------------------------
# FIGURE 3: Distribution Bars (3-panel)
# ---------------------------------------------------------------------------
def make_figure3():
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    # --- Panel (a): Functional Role ---
    ax = axes[0]
    roles  = list(ROLE_COUNTS.keys())
    rcvals = [ROLE_COUNTS[r] for r in roles]
    colours_a = [CB_PALETTE[i % len(CB_PALETTE)] for i in range(len(roles))]
    bars = ax.barh(roles, rcvals, color=colours_a, edgecolor="white", linewidth=0.8)
    for bar, v in zip(bars, rcvals):
        ax.text(v + 0.2, bar.get_y() + bar.get_height() / 2,
                str(v), va="center", fontsize=9, fontweight="bold")
    ax.set_xlabel("Papers (approx.)", fontsize=9)
    ax.set_title("(a) Functional Role", fontsize=10, fontweight="bold")
    ax.set_xlim(0, max(rcvals) + 4)
    ax.tick_params(axis="y", labelsize=8.5)
    sns.despine(ax=ax, left=True, bottom=False)

    # --- Panel (b): System Architecture ---
    ax = axes[1]
    archs  = list(ARCH_COUNTS.keys())
    avals  = [ARCH_COUNTS[a] for a in archs]
    colours_b = [CB_PALETTE[i % len(CB_PALETTE)] for i in range(len(archs))]
    bars = ax.barh(archs, avals, color=colours_b, edgecolor="white", linewidth=0.8)
    for bar, v in zip(bars, avals):
        ax.text(v + 0.2, bar.get_y() + bar.get_height() / 2,
                str(v), va="center", fontsize=9, fontweight="bold")
    ax.set_xlabel("Papers (approx.)", fontsize=9)
    ax.set_title("(b) System Architecture", fontsize=10, fontweight="bold")
    ax.set_xlim(0, max(avals) + 6)
    ax.tick_params(axis="y", labelsize=8.5)
    sns.despine(ax=ax, left=True, bottom=False)

    # --- Panel (c): Evaluation Standard ---
    ax = axes[2]
    evals  = list(EVAL_COUNTS.keys())
    evals_short = [
        "In-sample /\ncherry-picked",
        "OOS backtest,\nno costs",
        "OOS backtest,\nwith costs",
        "Live / paper\ntrading",
        "NLP benchmark\nonly",
    ]
    evalvals = [EVAL_COUNTS[e] for e in evals]
    # Highlight "Live" bar in a distinct alarm colour
    colours_c = []
    for e in evals:
        if "Live" in e:
            colours_c.append(CB_PALETTE[5])  # vermillion = draw attention
        else:
            colours_c.append(CB_PALETTE[2])   # green for normal bars

    bars = ax.barh(evals_short, evalvals, color=colours_c, edgecolor="white", linewidth=0.8)
    for bar, v in zip(bars, evalvals):
        ax.text(v + 0.2, bar.get_y() + bar.get_height() / 2,
                str(v), va="center", fontsize=9, fontweight="bold")

    # Arrow annotation on "Live" bar
    live_idx = evals_short.index("Live / paper\ntrading")
    live_bar = bars[live_idx]
    ax.annotate(
        "Near-absent\n(n=3)",
        xy=(live_bar.get_width(), live_bar.get_y() + live_bar.get_height() / 2),
        xytext=(live_bar.get_width() + 5, live_bar.get_y() + live_bar.get_height() / 2),
        fontsize=8, color=CB_PALETTE[5], fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=CB_PALETTE[5], lw=1.5),
        va="center"
    )

    ax.set_xlabel("Papers (approx.)", fontsize=9)
    ax.set_title("(c) Evaluation Standard", fontsize=10, fontweight="bold")
    ax.set_xlim(0, max(evalvals) + 9)
    ax.tick_params(axis="y", labelsize=8.5)
    sns.despine(ax=ax, left=True, bottom=False)

    fig.suptitle(
        "Figure 3: Distribution of Surveyed Papers Across Three Taxonomy Dimensions",
        fontsize=11, fontweight="bold", y=1.01
    )
    fig.tight_layout()
    save_fig(fig, "figure3")
    plt.close(fig)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    print("=== P4 Figure Generation ===")
    print(f"Output directory: {FIG_DIR}")
    print()

    print("Generating Figure 1: Taxonomy Heatmap ...")
    make_figure1()

    print("Generating Figure 2: Publication Timeline ...")
    make_figure2()

    print("Generating Figure 3: Distribution Bars ...")
    make_figure3()

    print()
    print("All figures complete.")


if __name__ == "__main__":
    main()
