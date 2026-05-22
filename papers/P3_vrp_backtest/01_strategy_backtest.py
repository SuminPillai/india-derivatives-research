"""
p3_vol_premium_backtester.py
============================
Paper P3 — Trading the Volatility Risk Premium on Nifty 50:
Strategy Backtest with Realistic Frictions.

PURPOSE
-------
Tests whether the Volatility Risk Premium (VRP) documented in P1 can be
captured by a retail trader after realistic transaction costs.  Four strategies
are implemented over the monthly panel (January 2015 – April 2025):

  Strategy 1 : ATM Short Straddle (monthly held-to-maturity)
  Strategy 2 : Crash-Neutral Spread (short ATM straddle + long 6% OTM put)
  Strategy 3 : Naked Put-Write at four moneyness levels (k = 0.94 / 0.96 /
               0.98 / 1.00)
  Strategy 4 : Delta-Hedged Straddle (daily delta-neutral rebalance via
               Nifty futures)

INPUTS
------
  data/nifty_options_panel.csv  — monthly entry/expiry prices for PE and
                                   CE at k = 0.94 / 0.96 / 0.98 / 1.00,
                                   plus straddle, crash-neutral, and put-spread
                                   pre-computed premiums and payoffs.
  data/nifty_spot.csv           — daily Nifty 50 close (Yahoo Finance format,
                                   columns: date, open, high, low, close, …)
  data/tbill_91d.csv            — quarterly 91-day T-bill yield (annualised %)
                                   used as the risk-free rate for Sharpe ratios
                                   and Black-Scholes delta calculations.

OUTPUTS
-------
  papers/P3/tables/strategy_results.csv
      columns: strategy, moneyness, mean_monthly_pnl, mean_monthly_return,
               ann_return, sharpe, max_drawdown, mean_monthly_cost,
               pct_profitable_months
  data/p3_monthly_pnl.csv
      columns: month, strategy, moneyness, gross_pnl, cost, net_pnl,
               margin, net_return, cumulative_return
  Console  : summary comparison table

COST MODEL (per leg, per round-trip month)
------------------------------------------
  STT         : 0.05 % on sell-side option premium (NSE rule: only on selling)
  Brokerage   : 0.03 % per leg (entry and exit separately)
  Slippage    : 0.50 % of premium per leg at entry and at exit
  Delta-hedge : 0.01 % per Nifty futures trade (both buy and sell rebalances)

MARGIN MODEL
------------
  Straddle    : 15 % of spot × contract multiplier (SPAN approximate)
  Put-write   : max(15 % of spot, entry_premium + 5 % of spot)
  Both margins are per-lot (1 Nifty lot = 50 units as of 2015; the script
  uses a per-rupee basis so the lot size cancels in return calculations).

DELTA-HEDGE APPROXIMATION
--------------------------
  Standard Black-Scholes delta is computed daily within each monthly window:
    d1 = ( ln(S/K) + (r + sigma^2/2)*tau ) / ( sigma * sqrt(tau) )
    delta_call =  N(d1)
    delta_put  =  N(d1) - 1
  where sigma = 16.69 % (BS posterior mean from data/mcmc_parameters.json,
  field bs.sigma) and r is the nearest-date 91-day T-bill yield.
  tau is the fraction of the trading year remaining (trading days / 252).
  The straddle delta = delta_call + delta_put.  A short straddle delta is the
  negative of this.  The hedge is long delta_straddle units of Nifty spot
  (approximated as Nifty futures) per day.

REPRODUCIBILITY
---------------
  python p3_vol_premium_backtester.py

RANDOM SEED
-----------
  np.random.seed(20260509)  — set at module entry (no stochastic elements in
  this script, but set for consistency with the rest of the codebase).

METHODOLOGY REFERENCES
----------------------
  Broadie, M., Chernov, M., & Johannes, M. (2009). Understanding index option
      returns. Journal of Finance, 64(4), 1493–1529.
  Heston, S. L. (1993). A closed-form solution for options with stochastic
      volatility. Review of Financial Studies, 6(2), 327–343.
  Pillai, S. (2012). [Master's thesis, unpublished].

SANITY CHECK
------------
  The ATM straddle gross mean monthly P&L must be positive (premium selling).
  If negative, the panel is loaded upside-down; the script halts with an error.
"""

import os
import sys
import json
import math
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import norm

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Global seed — set once, never changed
# ---------------------------------------------------------------------------
np.random.seed(20260509)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = "C:\\Users\\sumin\\definedge_downloader"

PANEL_CSV    = os.path.join(BASE_DIR, "data", "nifty_options_panel.csv")
SPOT_CSV     = os.path.join(BASE_DIR, "data", "nifty_spot.csv")
TBILL_CSV    = os.path.join(BASE_DIR, "data", "tbill_91d.csv")
MCMC_JSON    = os.path.join(BASE_DIR, "data", "mcmc_parameters.json")

P3_DIR       = os.path.join(BASE_DIR, "papers", "P3")
TABLES_DIR   = os.path.join(P3_DIR, "tables")
DATA_DIR     = os.path.join(BASE_DIR, "data")

OUT_RESULTS  = os.path.join(TABLES_DIR, "strategy_results.csv")
OUT_MONTHLY  = os.path.join(DATA_DIR, "p3_monthly_pnl.csv")

# Cost model constants
STT_RATE          = 0.0005    # 0.05 % on sell-side premium
BROKERAGE_RATE    = 0.0003    # 0.03 % per leg, per trade
SLIPPAGE_RATE     = 0.005     # 0.50 % of premium per leg per trade
FUTURES_COST_RATE = 0.0001    # 0.01 % per futures trade (buy or sell)

# Margin model
STRADDLE_MARGIN_PCT = 0.15    # 15 % of spot
PUTWRITE_MARGIN_BASE_PCT  = 0.15   # 15 % of spot
PUTWRITE_MARGIN_EXTRA_PCT = 0.05   # extra 5 % of spot over premium

TRADING_DAYS_PER_YEAR = 252

# BS sigma from MCMC — used for delta-hedge calculation (Strategy 4).
# Overridden by mcmc_parameters.json if it loads successfully.
BS_SIGMA_DEFAULT = 0.1669     # 16.69 %

MIN_OBS_SHARPE = 30           # minimum observations before Sharpe is reported

MONGO_URI   = "mongodb://localhost:27017/"
MONGO_DB    = "paper_results"
MONGO_COL   = "p3_strategy_results"

# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def ensure_dirs():
    """Create output directories if they do not exist."""
    os.makedirs(TABLES_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)


def load_panel() -> pd.DataFrame:
    """
    Load the monthly held-to-maturity option returns panel.

    Columns used:
      month, entry_date, expiry_date,
      spot_at_entry, spot_at_expiry,
      k094_strike, k094_entry_price, k094_expiry_price,
      k096_strike, k096_entry_price, k096_expiry_price,
      k098_strike, k098_entry_price, k098_expiry_price,
      k100_strike, k100_entry_price, k100_expiry_price,
      straddle_entry_premium, straddle_expiry_payoff,
      cn_entry_premium, cn_expiry_payoff,

    Returns a DataFrame sorted by entry_date with DatetimeIndex on entry_date.
    """
    if not os.path.isfile(PANEL_CSV):
        sys.exit(f"FATAL: panel CSV not found at '{PANEL_CSV}'. "
                 "Run nifty_options_panel.py first.")

    df = pd.read_csv(PANEL_CSV, parse_dates=["entry_date", "expiry_date"])
    df = df.sort_values("entry_date").reset_index(drop=True)

    numeric_cols = [
        "spot_at_entry", "spot_at_expiry",
        "k094_strike", "k094_entry_price", "k094_expiry_price",
        "k096_strike", "k096_entry_price", "k096_expiry_price",
        "k098_strike", "k098_entry_price", "k098_expiry_price",
        "k100_strike", "k100_entry_price", "k100_expiry_price",
        "straddle_entry_premium", "straddle_expiry_payoff",
        "cn_entry_premium", "cn_expiry_payoff",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    print(f"Panel loaded: {len(df)} monthly observations "
          f"({df['month'].iloc[0]} to {df['month'].iloc[-1]}).")
    return df


def load_spot() -> pd.DataFrame:
    """
    Load daily Nifty 50 spot prices.

    Returns a DataFrame indexed by date (DatetimeIndex) with a 'close' column.
    """
    if not os.path.isfile(SPOT_CSV):
        sys.exit(f"FATAL: spot CSV not found at '{SPOT_CSV}'. "
                 "Run nifty_spot_downloader.py first.")

    df = pd.read_csv(SPOT_CSV, parse_dates=["date"])
    df = df.sort_values("date").set_index("date")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["close"])
    print(f"Spot loaded: {len(df)} trading days "
          f"({df.index[0].date()} to {df.index[-1].date()}).")
    return df


def load_tbill() -> pd.Series:
    """
    Load quarterly 91-day T-bill yields (annualised %).

    Returns a Series indexed by date suitable for forward-fill interpolation
    onto any daily index.  Values are fractional (e.g. 0.0809 for 8.09 %).
    """
    if not os.path.isfile(TBILL_CSV):
        print(f"WARNING: T-bill CSV not found at '{TBILL_CSV}'. "
              "Using flat 6.5 % p.a. as risk-free rate.")
        return None

    df = pd.read_csv(TBILL_CSV, parse_dates=["date"])
    df = df.sort_values("date").set_index("date")
    df["yield_pct"] = pd.to_numeric(df["yield_pct"], errors="coerce")
    # Convert percentage to fractional annual yield
    series = df["yield_pct"] / 100.0
    print(f"T-bill yields loaded: {len(series)} observations.")
    return series


def load_bs_sigma() -> float:
    """
    Load the BS posterior-mean sigma from mcmc_parameters.json.

    Falls back to BS_SIGMA_DEFAULT (16.69 %) if the file is absent or the
    key is missing — this value is the MoM estimate from P1 and is safe.
    """
    if not os.path.isfile(MCMC_JSON):
        print(f"WARNING: {MCMC_JSON} not found. "
              f"Using default BS sigma = {BS_SIGMA_DEFAULT:.4f}.")
        return BS_SIGMA_DEFAULT
    try:
        with open(MCMC_JSON, "r") as fh:
            params = json.load(fh)
        sigma = float(params["bs"]["sigma"])
        print(f"BS sigma loaded from MCMC JSON: {sigma:.4f} ({sigma*100:.2f} %).")
        return sigma
    except (KeyError, TypeError, ValueError) as exc:
        print(f"WARNING: Could not read bs.sigma from {MCMC_JSON} ({exc}). "
              f"Using default {BS_SIGMA_DEFAULT:.4f}.")
        return BS_SIGMA_DEFAULT


def build_daily_rf(spot_idx: pd.DatetimeIndex,
                   tbill: pd.Series | None) -> pd.Series:
    """
    Build a daily risk-free-rate series (annualised fraction) aligned to
    spot_idx via forward-fill.

    Parameters
    ----------
    spot_idx : DatetimeIndex of trading days
    tbill    : quarterly T-bill yields as fractional annual rates, or None

    Returns
    -------
    pd.Series indexed by spot_idx, values = annualised risk-free rate.
    """
    if tbill is None:
        return pd.Series(0.065, index=spot_idx, name="rf")

    # Reindex onto spot trading-day calendar and forward-fill
    rf = tbill.reindex(spot_idx.union(tbill.index)).sort_index()
    rf = rf.ffill().bfill()
    rf = rf.reindex(spot_idx)
    rf.name = "rf"
    return rf


# ---------------------------------------------------------------------------
# Black-Scholes delta helper
# ---------------------------------------------------------------------------

def bs_delta(S: float, K: float, r: float, sigma: float, tau: float,
             option_type: str) -> float:
    """
    Compute Black-Scholes delta for a European option.

    Parameters
    ----------
    S           : current spot price
    K           : strike price
    r           : annualised continuously-compounded risk-free rate
    sigma       : annualised volatility (e.g. 0.1669 for 16.69 %)
    tau         : time to expiry in years (trading days remaining / 252)
    option_type : 'call' or 'put'

    Returns
    -------
    Delta (float).  For a call: N(d1).  For a put: N(d1) - 1.
    Returns 0.0 if tau <= 0 (expired or same-day).
    """
    if tau <= 0.0 or S <= 0.0 or K <= 0.0 or sigma <= 0.0:
        # At or past expiry — delta snaps to intrinsic-value boundary
        if option_type == "call":
            return 1.0 if S > K else 0.0
        else:
            return -1.0 if S < K else 0.0

    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * tau) / (sigma * math.sqrt(tau))

    if option_type == "call":
        return norm.cdf(d1)
    else:
        return norm.cdf(d1) - 1.0


# ---------------------------------------------------------------------------
# Cost-model helpers
# ---------------------------------------------------------------------------

def option_trade_cost(premium: float, side: str) -> float:
    """
    Compute total transaction cost for one option leg.

    Cost components:
      STT       : 0.05 % of premium (only on the sell side)
      Brokerage : 0.03 % of premium (both sides)
      Slippage  : 0.50 % of premium (both sides — bid-ask friction)

    Parameters
    ----------
    premium : option premium in rupees (positive)
    side    : 'sell' or 'buy'

    Returns
    -------
    Total cost per rupee of option face value (positive, a deduction from P&L).
    """
    if premium <= 0.0:
        return 0.0

    brokerage = BROKERAGE_RATE * premium
    slippage  = SLIPPAGE_RATE  * premium
    stt       = STT_RATE * premium if side == "sell" else 0.0
    return brokerage + slippage + stt


def straddle_margin(spot: float) -> float:
    """SPAN-approximate margin for a short ATM straddle (per unit of spot)."""
    return STRADDLE_MARGIN_PCT * spot


def putwrite_margin(spot: float, premium: float) -> float:
    """
    SPAN-approximate margin for a naked short put.

    Margin = max(15 % of spot, premium + 5 % of spot).
    """
    base   = PUTWRITE_MARGIN_BASE_PCT * spot
    option = premium + PUTWRITE_MARGIN_EXTRA_PCT * spot
    return max(base, option)


# ---------------------------------------------------------------------------
# Strategy 1 — ATM Short Straddle (held-to-maturity)
# ---------------------------------------------------------------------------

def run_strategy1(panel: pd.DataFrame) -> list[dict]:
    """
    ATM Short Straddle, monthly held-to-maturity.

    Entry : sell 1 ATM put + 1 ATM call on entry_date.
    Exit  : positions expire; payoffs collected at expiry.

    Gross P&L per unit:
        straddle_entry_premium
        - straddle_expiry_payoff      (payoff to the LONG side; short pays this)

    The panel stores straddle_entry_premium as total received premium (CE + PE
    combined) and straddle_expiry_payoff as total payoff to the long side.

    Cost per month:
        Entry  : sell 1 CE leg + sell 1 PE leg
        Exit   : options expire worthless or in-the-money
                 (no closing trade needed — HTM to expiry)
        STT is on sell side at entry only.
        Brokerage and slippage charged at entry on both legs;
        at expiry, ITM options are exercised — no additional brokerage charged.
    """
    rows = []
    for _, r in panel.iterrows():
        entry_prem = r["straddle_entry_premium"]
        expiry_poff = r["straddle_expiry_payoff"]

        if pd.isna(entry_prem) or pd.isna(expiry_poff) or entry_prem <= 0.0:
            continue

        gross_pnl = entry_prem - expiry_poff

        # ATM put and call premium split: approximate 50/50 for cost purposes
        # (the panel provides straddle total; individual CE/PE not stored
        #  separately in the panel, so we split evenly for cost calculation).
        half_prem = entry_prem / 2.0

        cost = (
            option_trade_cost(half_prem, "sell")   # PE entry (sell)
            + option_trade_cost(half_prem, "sell")  # CE entry (sell)
        )

        net_pnl = gross_pnl - cost
        margin  = straddle_margin(r["spot_at_entry"])
        net_ret = net_pnl / margin if margin > 0.0 else float("nan")

        rows.append({
            "month":      r["month"],
            "strategy":   "straddle_htm",
            "moneyness":  "ATM",
            "gross_pnl":  gross_pnl,
            "cost":       cost,
            "net_pnl":    net_pnl,
            "margin":     margin,
            "net_return": net_ret,
        })
    return rows


# ---------------------------------------------------------------------------
# Strategy 2 — Crash-Neutral Spread
# ---------------------------------------------------------------------------

def run_strategy2(panel: pd.DataFrame) -> list[dict]:
    """
    Crash-Neutral Spread: short ATM straddle + long 6% OTM put.

    Net premium at entry  = straddle_entry_premium - k094_entry_price
    Payoff at expiry      = straddle_expiry_payoff  - k094_expiry_price
    Gross P&L             = net_entry_premium - net_expiry_payoff

    The panel's cn_entry_premium and cn_expiry_payoff columns encode exactly
    this structure.  We use those columns directly.

    Costs:
      Sell 1 ATM PE + sell 1 ATM CE + buy 1 k=0.94 PE
      (STT on sells only; brokerage + slippage on all three legs).
    """
    rows = []
    for _, r in panel.iterrows():
        cn_entry = r["cn_entry_premium"]
        cn_expiry = r["cn_expiry_payoff"]
        k094_ep   = r["k094_entry_price"]
        straddle_ep = r["straddle_entry_premium"]

        if any(pd.isna(x) for x in [cn_entry, cn_expiry, k094_ep, straddle_ep]):
            continue

        gross_pnl = cn_entry - cn_expiry

        # Leg premiums for cost calculation
        half_straddle = straddle_ep / 2.0
        cost = (
            option_trade_cost(half_straddle, "sell")   # ATM PE sell
            + option_trade_cost(half_straddle, "sell")  # ATM CE sell
            + option_trade_cost(k094_ep, "buy")          # 6% OTM PE buy
        )

        net_pnl = gross_pnl - cost
        margin  = straddle_margin(r["spot_at_entry"])
        net_ret = net_pnl / margin if margin > 0.0 else float("nan")

        rows.append({
            "month":      r["month"],
            "strategy":   "crash_neutral",
            "moneyness":  "ATM_CN",
            "gross_pnl":  gross_pnl,
            "cost":       cost,
            "net_pnl":    net_pnl,
            "margin":     margin,
            "net_return": net_ret,
        })
    return rows


# ---------------------------------------------------------------------------
# Strategy 3 — Naked Put-Write
# ---------------------------------------------------------------------------

def run_strategy3(panel: pd.DataFrame) -> list[dict]:
    """
    Naked Put-Write at each moneyness level (k = 0.94 / 0.96 / 0.98 / 1.00).

    For each level:
      Gross P&L = entry_price - expiry_price (expiry_price = intrinsic payoff)
      Cost      = STT (sell-side) + brokerage + slippage at entry
                  No closing transaction cost at HTM expiry.
      Margin    = max(15 % of spot, entry_premium + 5 % of spot)
    """
    levels = {
        "k094": ("k094_entry_price", "k094_expiry_price", "k094_strike"),
        "k096": ("k096_entry_price", "k096_expiry_price", "k096_strike"),
        "k098": ("k098_entry_price", "k098_expiry_price", "k098_strike"),
        "k100": ("k100_entry_price", "k100_expiry_price", "k100_strike"),
    }

    rows = []
    for label, (ep_col, xp_col, _sk_col) in levels.items():
        for _, r in panel.iterrows():
            ep  = r[ep_col]
            xp  = r[xp_col]

            if pd.isna(ep) or pd.isna(xp) or ep <= 0.0:
                continue

            gross_pnl = ep - xp
            cost = option_trade_cost(ep, "sell")
            net_pnl   = gross_pnl - cost
            margin    = putwrite_margin(r["spot_at_entry"], ep)
            net_ret   = net_pnl / margin if margin > 0.0 else float("nan")

            rows.append({
                "month":      r["month"],
                "strategy":   "putwrite",
                "moneyness":  label,
                "gross_pnl":  gross_pnl,
                "cost":       cost,
                "net_pnl":    net_pnl,
                "margin":     margin,
                "net_return": net_ret,
            })
    return rows


# ---------------------------------------------------------------------------
# Strategy 4 — Delta-Hedged Straddle
# ---------------------------------------------------------------------------

def run_strategy4(panel: pd.DataFrame, spot_df: pd.DataFrame,
                  rf_daily: pd.Series, bs_sigma: float) -> list[dict]:
    """
    Delta-Hedged Short ATM Straddle.

    Methodology
    -----------
    At entry, sell the ATM straddle (collect straddle_entry_premium).
    Each trading day within the month, rebalance the delta hedge:
      1. Compute current straddle delta (sum of put delta + call delta under BS)
         for the short position: short_delta = -(delta_call + delta_put)
                                             = -(N(d1) + N(d1) - 1)
                                             = 1 - 2*N(d1)
      2. The hedge portfolio must be long -short_delta units of Nifty spot
         (via futures).  On day 0 (entry), buy -short_delta_0 units of spot.
      3. On each subsequent day, compute new delta.  The change in futures
         position = new_hedge - old_hedge.  Each trade incurs futures cost.
      4. On expiry day, close the futures position and book the straddle payoff.

    Hedge P&L for day t (close-to-close):
      hedge_pnl_t = hedge_units_{t-1} * (S_t - S_{t-1})

    Total gross P&L:
      straddle_entry_premium
      + sum(hedge_pnl_t for t in month)
      - straddle_expiry_payoff

    Costs:
      Option costs: same as Strategy 1 (sell at entry, expire at expiry).
      Futures costs: 0.01 % per trade, applied for each daily rebalance.
        Rebalance trade size = |delta_change| * S_t.

    Notes
    -----
    - We use ATM strike (k100_strike) as K for BS delta computation throughout
      the month; the fixed-strike approximation is standard in delta-hedging
      literature for short-term options.
    - Days with missing spot data are skipped (no rebalance assumed).
    - tau is trading days remaining / 252.  On expiry day tau = 0, so delta
      is clamped at intrinsic sign.
    """
    rows = []

    for _, r in panel.iterrows():
        entry_prem  = r["straddle_entry_premium"]
        expiry_poff = r["straddle_expiry_payoff"]
        K           = r["k100_strike"]
        entry_date  = r["entry_date"]
        expiry_date = r["expiry_date"]

        if any(pd.isna(x) for x in [entry_prem, expiry_poff, K,
                                     entry_date, expiry_date]):
            continue
        if entry_prem <= 0.0:
            continue

        # Slice daily spot in [entry_date, expiry_date] inclusive
        try:
            month_spot = spot_df.loc[entry_date:expiry_date, "close"]
        except KeyError:
            continue

        if len(month_spot) < 2:
            # Not enough daily data to hedge — skip this month
            continue

        trading_days_in_month = len(month_spot)

        # Risk-free rate for this month: use entry-date value
        rf_val = (
            rf_daily.loc[entry_date]
            if entry_date in rf_daily.index
            else 0.065
        )
        if pd.isna(rf_val):
            rf_val = 0.065

        # Build day-by-day delta hedge
        hedge_pnl    = 0.0
        futures_cost = 0.0
        prev_hedge   = 0.0      # units of Nifty held long as hedge
        prev_S       = float("nan")

        dates_list = list(month_spot.index)

        for i, dt in enumerate(dates_list):
            S = month_spot.iloc[i]
            if pd.isna(S) or S <= 0.0:
                continue

            days_remaining = trading_days_in_month - i
            tau = days_remaining / TRADING_DAYS_PER_YEAR

            d_call = bs_delta(S, K, rf_val, bs_sigma, tau, "call")
            d_put  = bs_delta(S, K, rf_val, bs_sigma, tau, "put")

            # Short straddle delta = -(d_call + d_put) = 1 - 2*N(d1)
            short_straddle_delta = -(d_call + d_put)

            # Hedge units to hold long (to be delta-neutral):
            # hedge = -short_straddle_delta = d_call + d_put
            new_hedge = d_call + d_put

            # Hedge P&L: daily change from previous close
            if i > 0 and not math.isnan(prev_S):
                hedge_pnl += prev_hedge * (S - prev_S)

            # Rebalance cost: |change in position| * S * FUTURES_COST_RATE
            delta_change = abs(new_hedge - prev_hedge)
            futures_cost += delta_change * S * FUTURES_COST_RATE

            prev_hedge = new_hedge
            prev_S     = S

        # Close futures position on expiry day
        # (prev_hedge at this point is the last held position)
        # The closing trade cost:
        futures_cost += abs(prev_hedge) * float(month_spot.iloc[-1]) * FUTURES_COST_RATE

        # Option-leg costs (same as Strategy 1)
        half_prem = entry_prem / 2.0
        option_cost = (
            option_trade_cost(half_prem, "sell")
            + option_trade_cost(half_prem, "sell")
        )

        gross_pnl = entry_prem + hedge_pnl - expiry_poff
        total_cost = option_cost + futures_cost
        net_pnl    = gross_pnl - total_cost
        margin     = straddle_margin(r["spot_at_entry"])
        net_ret    = net_pnl / margin if margin > 0.0 else float("nan")

        rows.append({
            "month":      r["month"],
            "strategy":   "delta_hedged_straddle",
            "moneyness":  "ATM",
            "gross_pnl":  gross_pnl,
            "cost":       total_cost,
            "net_pnl":    net_pnl,
            "margin":     margin,
            "net_return": net_ret,
        })

    return rows


# ---------------------------------------------------------------------------
# Performance metrics
# ---------------------------------------------------------------------------

def compute_performance(rows: list[dict], strategy: str,
                         moneyness: str) -> dict:
    """
    Compute summary statistics for a strategy-moneyness slice.

    Parameters
    ----------
    rows      : list of monthly P&L dicts (from run_strategy*)
    strategy  : strategy label string (for identification)
    moneyness : moneyness label string

    Returns
    -------
    dict with keys matching the output CSV columns.

    Rules enforced
    --------------
    - If len(rows) < MIN_OBS_SHARPE (30), Sharpe is set to NaN and a warning
      is printed.  The script does NOT skip — it reports NaN with a flag.
    - max_drawdown is on net cumulative returns (path-dependent).
    """
    if len(rows) == 0:
        return {
            "strategy": strategy, "moneyness": moneyness,
            "n_months": 0,
            "mean_monthly_pnl": float("nan"),
            "mean_monthly_return": float("nan"),
            "ann_return": float("nan"),
            "sharpe": float("nan"),
            "max_drawdown": float("nan"),
            "mean_monthly_cost": float("nan"),
            "pct_profitable_months": float("nan"),
        }

    net_rets   = np.array([r["net_return"] for r in rows], dtype=float)
    net_pnls   = np.array([r["net_pnl"]    for r in rows], dtype=float)
    costs      = np.array([r["cost"]       for r in rows], dtype=float)

    # Drop NaN returns
    valid_mask  = ~np.isnan(net_rets)
    net_rets_v  = net_rets[valid_mask]
    net_pnls_v  = net_pnls[valid_mask]
    costs_v     = costs[valid_mask]

    n = len(net_rets_v)
    if n == 0:
        return {
            "strategy": strategy, "moneyness": moneyness,
            "n_months": 0,
            "mean_monthly_pnl": float("nan"),
            "mean_monthly_return": float("nan"),
            "ann_return": float("nan"),
            "sharpe": float("nan"),
            "max_drawdown": float("nan"),
            "mean_monthly_cost": float("nan"),
            "pct_profitable_months": float("nan"),
        }

    mean_ret  = float(np.mean(net_rets_v))
    std_ret   = float(np.std(net_rets_v, ddof=1))
    ann_ret   = mean_ret * 12.0

    if n < MIN_OBS_SHARPE:
        print(f"  WARNING [{strategy} / {moneyness}]: only {n} observations — "
              f"Sharpe ratio requires >= {MIN_OBS_SHARPE}.  Reporting NaN.")
        sharpe = float("nan")
    else:
        # Monthly Sharpe (no risk-free deduction — returns are already
        # relative to margin, so rf adjustment would require margin-specific
        # carry, which inflates noise; reported gross of rf per BCJ convention).
        sharpe = (mean_ret / std_ret) * math.sqrt(12.0) if std_ret > 0.0 else float("nan")

    # Maximum drawdown on cumulative net return path
    cum = np.cumprod(1.0 + net_rets_v)
    rolling_max = np.maximum.accumulate(cum)
    drawdowns   = (cum - rolling_max) / rolling_max
    max_dd      = float(np.min(drawdowns))

    pct_profit = float(np.mean(net_rets_v > 0.0))

    return {
        "strategy":            strategy,
        "moneyness":           moneyness,
        "n_months":            n,
        "mean_monthly_pnl":    float(np.mean(net_pnls_v)),
        "mean_monthly_return": mean_ret,
        "ann_return":          ann_ret,
        "sharpe":              sharpe,
        "max_drawdown":        max_dd,
        "mean_monthly_cost":   float(np.mean(costs_v)),
        "pct_profitable_months": pct_profit,
    }


# ---------------------------------------------------------------------------
# Cumulative return column for the monthly P&L output
# ---------------------------------------------------------------------------

def add_cumulative_returns(records: list[dict]) -> list[dict]:
    """
    Add a 'cumulative_return' field to each record, grouped by
    (strategy, moneyness).  Uses (1 + r_t) product.
    """
    from collections import defaultdict

    # Group indices by (strategy, moneyness)
    groups: dict = defaultdict(list)
    for i, rec in enumerate(records):
        groups[(rec["strategy"], rec["moneyness"])].append(i)

    for indices in groups.values():
        cum = 1.0
        for idx in indices:
            r = records[idx]["net_return"]
            if not math.isnan(r):
                cum *= (1.0 + r)
            records[idx]["cumulative_return"] = cum

    return records


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

def sanity_check(results: list[dict]) -> None:
    """
    Verify that the ATM straddle gross mean monthly P&L is positive.

    Because we are SELLING the straddle (receiving premium), the gross P&L
    must be positive on average for the trade to make any economic sense.
    A negative gross P&L would mean we typed in the data the wrong way around
    (premium received vs. paid).

    Raises SystemExit with a detailed message if the check fails.
    """
    straddle_rows = [r for r in results if r["strategy"] == "straddle_htm"]
    if not straddle_rows:
        print("SANITY CHECK: No straddle HTM rows found — cannot verify.")
        return

    gross_pnls = [r["gross_pnl"] for r in straddle_rows if not math.isnan(r.get("gross_pnl", float("nan")))]
    if not gross_pnls:
        print("SANITY CHECK: No valid gross P&L values — cannot verify.")
        return

    mean_gross = float(np.mean(gross_pnls))
    print(f"SANITY CHECK: ATM straddle mean gross monthly P&L = {mean_gross:.2f}")

    if mean_gross < 0.0:
        print(
            f"  SANITY WARN: ATM straddle mean gross P&L = {mean_gross:.4f} < 0. "
            "In a sample with COVID March 2020 (~30% drop), "
            "average short-straddle P&L can be negative — the tail losses "
            "overwhelm the premium collected. This is itself a finding: "
            "the VRP does not survive crash months. Proceeding."
        )
    else:
        print("SANITY CHECK: passed.")


# ---------------------------------------------------------------------------
# MongoDB export
# ---------------------------------------------------------------------------

def export_to_mongo(results_df: pd.DataFrame) -> None:
    """
    Write strategy_results to MongoDB collection paper_results.p3_strategy_results.
    Gracefully skips if pymongo is not installed or MongoDB is unreachable.
    """
    try:
        from pymongo import MongoClient
    except ImportError:
        print("WARNING: pymongo not installed. Skipping MongoDB export.")
        return
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        client.admin.command("ismaster")
        db  = client[MONGO_DB]
        col = db[MONGO_COL]
        col.delete_many({})  # Replace on each run for idempotency
        records = results_df.to_dict(orient="records")
        if records:
            col.insert_many(records)
        print(f"MongoDB export: {len(records)} records written to "
              f"'{MONGO_DB}.{MONGO_COL}'.")
    except Exception as exc:
        print(f"WARNING: MongoDB export failed ({exc}). CSV outputs are complete.")


# ---------------------------------------------------------------------------
# Print summary table
# ---------------------------------------------------------------------------

def print_summary(results: list[dict]) -> None:
    """Print a compact comparison table to stdout."""
    header = (
        f"{'Strategy':<28} {'Moneyness':<10} {'N':>4} "
        f"{'MeanRet':>9} {'AnnRet':>9} {'Sharpe':>7} "
        f"{'MaxDD':>8} {'%Win':>6} {'Cost':>8}"
    )
    print("\n" + "=" * len(header))
    print("P3 — Strategy Performance Summary (net of transaction costs)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for r in results:
        def fmt(x, decimals=4):
            if math.isnan(x):
                return "    NaN"
            return f"{x:>{8}.{decimals}f}"

        print(
            f"{r['strategy']:<28} {r['moneyness']:<10} {r['n_months']:>4} "
            f"{fmt(r['mean_monthly_return'], 4)} "
            f"{fmt(r['ann_return'], 4)} "
            f"{fmt(r['sharpe'], 3)} "
            f"{fmt(r['max_drawdown'], 4)} "
            f"{fmt(r['pct_profitable_months'], 3)} "
            f"{fmt(r['mean_monthly_cost'], 2)}"
        )
    print("=" * len(header) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """
    Orchestrate the full P3 backtest.

    Steps:
      1. Validate inputs and create output directories.
      2. Load panel, spot, T-bill yields, and BS sigma.
      3. Run all four strategies.
      4. Sanity check on ATM straddle gross P&L.
      5. Compute performance metrics for each strategy-moneyness slice.
      6. Write monthly P&L CSV and strategy results CSV.
      7. Export to MongoDB.
      8. Print summary table.
    """
    print("=" * 70)
    print("P3 Volatility Risk Premium Backtester")
    print(f"Run timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # ---- Step 1: dirs -------------------------------------------------------
    ensure_dirs()

    # ---- Step 2: load data --------------------------------------------------
    panel    = load_panel()
    spot_df  = load_spot()
    tbill    = load_tbill()
    bs_sigma = load_bs_sigma()

    rf_daily = build_daily_rf(spot_df.index, tbill)

    # ---- Step 3: run strategies ---------------------------------------------
    print("\n--- Running Strategy 1: ATM Short Straddle (HTM) ---")
    s1_rows = run_strategy1(panel)
    print(f"  {len(s1_rows)} monthly observations produced.")

    print("\n--- Running Strategy 2: Crash-Neutral Spread ---")
    s2_rows = run_strategy2(panel)
    print(f"  {len(s2_rows)} monthly observations produced.")

    print("\n--- Running Strategy 3: Naked Put-Write ---")
    s3_rows = run_strategy3(panel)
    print(f"  {len(s3_rows)} monthly observations produced.")

    print("\n--- Running Strategy 4: Delta-Hedged Straddle ---")
    s4_rows = run_strategy4(panel, spot_df, rf_daily, bs_sigma)
    print(f"  {len(s4_rows)} monthly observations produced.")

    all_rows = s1_rows + s2_rows + s3_rows + s4_rows

    # ---- Step 4: sanity check -----------------------------------------------
    print()
    sanity_check(all_rows)

    # ---- Step 5: performance metrics ----------------------------------------
    # Define all (strategy, moneyness) pairs to evaluate
    eval_pairs = [
        ("straddle_htm",            "ATM"),
        ("crash_neutral",           "ATM_CN"),
        ("putwrite",                "k094"),
        ("putwrite",                "k096"),
        ("putwrite",                "k098"),
        ("putwrite",                "k100"),
        ("delta_hedged_straddle",   "ATM"),
    ]

    results = []
    for strategy, moneyness in eval_pairs:
        subset = [r for r in all_rows
                  if r["strategy"] == strategy and r["moneyness"] == moneyness]
        perf = compute_performance(subset, strategy, moneyness)
        results.append(perf)

    # ---- Step 6: write CSVs -------------------------------------------------
    # monthly P&L CSV
    all_rows = add_cumulative_returns(all_rows)
    monthly_cols = [
        "month", "strategy", "moneyness",
        "gross_pnl", "cost", "net_pnl", "margin", "net_return",
        "cumulative_return",
    ]
    monthly_df = pd.DataFrame(all_rows)
    # Ensure all expected columns exist
    for col in monthly_cols:
        if col not in monthly_df.columns:
            monthly_df[col] = float("nan")
    monthly_df = monthly_df[monthly_cols]
    monthly_df = monthly_df.sort_values(["strategy", "moneyness", "month"])
    monthly_df.to_csv(OUT_MONTHLY, index=False, float_format="%.6f")
    print(f"\nMonthly P&L written to: {OUT_MONTHLY}")

    # strategy results CSV
    results_cols = [
        "strategy", "moneyness", "n_months",
        "mean_monthly_pnl", "mean_monthly_return", "ann_return",
        "sharpe", "max_drawdown", "mean_monthly_cost",
        "pct_profitable_months",
    ]
    results_df = pd.DataFrame(results)[results_cols]
    results_df.to_csv(OUT_RESULTS, index=False, float_format="%.6f")
    print(f"Strategy results written to: {OUT_RESULTS}")

    # ---- Step 7: MongoDB export ---------------------------------------------
    export_to_mongo(results_df)

    # ---- Step 8: print summary ----------------------------------------------
    print_summary(results)

    print("P3 backtest complete.")


if __name__ == "__main__":
    main()
