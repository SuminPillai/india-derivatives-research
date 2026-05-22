"""
p2_strategy_backtester.py
=========================
Paper P2 — Technical-Indicator Strategy Performance in Indian Equities:
Joint Test of Market Efficiency 2015-2025.

PURPOSE
-------
Runs 6 strategy families (~22 configurations) over 78 F&O-eligible NSE tickers
using daily OHLCV data stored in MongoDB (price_data database).  For each
strategy x ticker pair it produces a daily return series, then aggregates into
equal-weighted portfolio returns and reports:

  - Annualized return, alpha vs Nifty 50 buy-and-hold
  - Sharpe ratio with 10,000-bootstrap 95 % CI
  - Sortino ratio, maximum drawdown, turnover
  - Romano-Wolf StepM adjusted p-values for the joint null Sharpe <= 0

OUTPUTS
-------
  papers/P2/tables/strategy_performance.csv   — headline statistics table
  data/p2_daily_returns.csv                   — daily portfolio returns

REPRODUCIBILITY
---------------
  python p2_strategy_backtester.py

Random seed: np.random.seed(20260509)
Chain / bootstrap samples: B = 10,000

METHODOLOGY
-----------
  - Backtests use vectorised pandas (no backtrader) for speed over 78 tickers.
  - Transaction cost: 0.05 % round-trip applied on every trade entry and exit.
  - Long-only, fully-invested (one position at a time per ticker).
  - Universe: tradingtickers.txt (78 F&O tickers).
  - Backtest window: 2015-01-01 to 2025-04-30.
  - Nifty 50 benchmark: ticker "NIFTY50" collection or index_data collection.
  - Romano-Wolf StepM follows Romano & Wolf (2005, JASA) and the
    implementation in Lehmann & Romano (2005).  The max-statistic bootstrap
    distribution is built from stationary block bootstrap (block length ~5
    trading days for daily returns).

REFERENCES
----------
  Romano, J. P. and Wolf, M. (2005).  "Stepwise Multiple Testing as
    Formalized Data Snooping." Econometrica 73(4): 1237-1282.
  White, H. (2000).  "A Reality Check for Data Snooping." Econometrica
    68(5): 1097-1126.
  Pillai (2012, Master's thesis, unpublished).
"""

import os
import sys
import math
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from pymongo import MongoClient

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

np.random.seed(20260509)

BASE_DIR         = "C:\\Users\\sumin\\definedge_downloader"
TICKER_FILE      = os.path.join(BASE_DIR, "tradingtickers.txt")
PAPERS_P2_DIR    = os.path.join(BASE_DIR, "papers", "P2")
TABLES_DIR       = os.path.join(PAPERS_P2_DIR, "tables")
DATA_DIR         = os.path.join(BASE_DIR, "data")

MONGO_URI        = "mongodb://localhost:27017/"
MONGO_DB_NAME    = "price_data"
NIFTY_TICKER     = "NIFTY50"           # collection name for benchmark

START_DATE       = "2015-01-01"
END_DATE         = "2025-04-30"

TRANSACTION_COST = 0.0005              # 0.05 % one-way; applied twice (round trip)
MIN_OBS_SHARPE   = 30                  # minimum monthly obs before Sharpe is computed
BOOTSTRAP_B      = 10_000             # number of bootstrap draws
BLOCK_LENGTH     = 5                   # trading days for stationary block bootstrap
ALPHA_LEVEL      = 0.05                # significance level for Romano-Wolf

TRADING_DAYS_PER_YEAR = 252

# ---------------------------------------------------------------------------
# Strategy catalogue
# ---------------------------------------------------------------------------
# Each entry is a dict with keys:
#   name         str  — human-readable strategy name
#   family       str  — family tag
#   params       dict — strategy-specific parameters
# ---------------------------------------------------------------------------

STRATEGY_CONFIGS = [
    # --- Strategy 1: SMA Crossover ---
    {"name": "SMA_10_50",    "family": "SMA_Crossover",       "params": {"fast": 10,  "slow": 50}},
    {"name": "SMA_20_100",   "family": "SMA_Crossover",       "params": {"fast": 20,  "slow": 100}},
    {"name": "SMA_50_200",   "family": "SMA_Crossover",       "params": {"fast": 50,  "slow": 200}},
    # --- Strategy 2: MACD Signal ---
    {"name": "MACD_12_26_9",  "family": "MACD_Signal",        "params": {"fast": 12, "slow": 26, "signal": 9}},
    {"name": "MACD_26_52_18", "family": "MACD_Signal",        "params": {"fast": 26, "slow": 52, "signal": 18}},
    # --- Strategy 3: Bollinger Band Squeeze ---
    {"name": "BB_20_2.0",    "family": "BB_Squeeze",          "params": {"period": 20, "width": 2.0, "squeeze_threshold": 0.03}},
    {"name": "BB_20_2.5",    "family": "BB_Squeeze",          "params": {"period": 20, "width": 2.5, "squeeze_threshold": 0.03}},
    # --- Strategy 4: RSI Mean Reversion ---
    {"name": "RSI_30_70",    "family": "RSI_MeanReversion",   "params": {"period": 14, "buy_thresh": 30, "sell_thresh": 70}},
    {"name": "RSI_25_75",    "family": "RSI_MeanReversion",   "params": {"period": 14, "buy_thresh": 25, "sell_thresh": 75}},
    {"name": "RSI_20_80",    "family": "RSI_MeanReversion",   "params": {"period": 14, "buy_thresh": 20, "sell_thresh": 80}},
    # --- Strategy 5: Donchian Breakout ---
    {"name": "Donchian_20",  "family": "Breakout_Momentum",   "params": {"channel": 20}},
    {"name": "Donchian_55",  "family": "Breakout_Momentum",   "params": {"channel": 55}},
    # --- Strategy 6: Multi-Indicator Confluence (mirrors smamacd1.py logic) ---
    {"name": "Confluence_v1","family": "Multi_Confluence",    "params": {
        "sma_period": 200, "ema_fast": 12, "ema_slow": 26,
        "atr_period": 14,  "atr_ma_period": 20
    }},
]

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def ensure_dirs():
    """Create output directories if they do not exist."""
    os.makedirs(TABLES_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)


def load_tickers(path: str) -> list[str]:
    """Load ticker list from a text file, one symbol per line."""
    if not os.path.exists(path):
        sys.exit(f"FATAL: ticker file not found: {path}")
    with open(path, "r") as fh:
        tickers = [ln.strip() for ln in fh if ln.strip()]
    if not tickers:
        sys.exit(f"FATAL: ticker file is empty: {path}")
    print(f"Loaded {len(tickers)} tickers from {path}")
    return tickers


def get_mongo_client() -> MongoClient:
    """Connect to MongoDB; exit with a clear message on failure."""
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        client.admin.command("ismaster")
        print("Connected to MongoDB.")
        return client
    except Exception as exc:
        sys.exit(f"FATAL: MongoDB connection failed — {exc}")


def load_price_data(db, ticker: str) -> pd.DataFrame | None:
    """
    Load OHLCV data for one ticker from MongoDB.
    Returns a DataFrame indexed by date with columns:
      open, high, low, close, volume
    or None if the data is insufficient.
    """
    try:
        # MongoDB collections have -EQ suffix (e.g., RELIANCE-EQ)
        col_name = ticker
        if col_name not in db.list_collection_names():
            col_name = f"{ticker}-EQ"
        col = db[col_name]
        raw = pd.DataFrame(list(col.find({})))
        if raw.empty:
            return None

        # Column name varies: 'Date', 'convertedDate', or 'date'
        date_col = next(
            (c for c in ("convertedDate", "Date", "date") if c in raw.columns),
            None,
        )
        if date_col is None:
            return None

        raw = raw.rename(columns={date_col: "date"})
        raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
        raw = raw.dropna(subset=["date"])
        raw = raw.sort_values("date").set_index("date")

        raw.columns = [c.lower() for c in raw.columns]

        for req in ("open", "high", "low", "close"):
            if req not in raw.columns:
                return None

        raw = raw[[c for c in ("open", "high", "low", "close", "volume") if c in raw.columns]]
        raw = raw.loc[START_DATE:END_DATE]
        raw = raw.dropna(subset=["close"])

        if len(raw) < 252:          # need at least one year of history
            return None

        return raw

    except Exception as exc:
        print(f"  Warning: could not load {ticker} — {exc}")
        return None


# ---------------------------------------------------------------------------
# Technical indicator computation
# ---------------------------------------------------------------------------

def compute_ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def compute_sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period).mean()


def compute_macd(close: pd.Series, fast: int, slow: int, signal: int):
    ema_f = compute_ema(close, fast)
    ema_s = compute_ema(close, slow)
    macd_line = ema_f - ema_s
    signal_line = compute_ema(macd_line, signal)
    return macd_line, signal_line


def compute_rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_bollinger(close: pd.Series, period: int, num_std: float):
    mid   = compute_sma(close, period)
    std   = close.rolling(period).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    bw    = (upper - lower) / mid          # bandwidth (normalised)
    return upper, mid, lower, bw


def compute_atr(df: pd.DataFrame, period: int) -> pd.Series:
    high  = df["high"]
    low   = df["low"]
    close = df["close"]
    tr    = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def compute_donchian(close: pd.Series, high: pd.Series, low: pd.Series, channel: int):
    """Return (channel_high, channel_low) based on prior `channel` bars."""
    ch_high = high.shift(1).rolling(channel).max()
    ch_low  = low.shift(1).rolling(channel).min()
    return ch_high, ch_low


# ---------------------------------------------------------------------------
# Strategy signal generators
# Each returns a pd.Series of positions: 1 = long, 0 = flat.
# ---------------------------------------------------------------------------

def signal_sma_crossover(df: pd.DataFrame, fast: int, slow: int) -> pd.Series:
    """Long when fast SMA > slow SMA."""
    close = df["close"]
    fast_ma = compute_sma(close, fast)
    slow_ma = compute_sma(close, slow)
    raw_signal = (fast_ma > slow_ma).astype(int)
    # positions are known at end-of-day; trade enters on next open
    return raw_signal.shift(1).fillna(0)


def signal_macd(df: pd.DataFrame, fast: int, slow: int, signal: int) -> pd.Series:
    """Long when MACD line > signal line."""
    close = df["close"]
    macd_line, signal_line = compute_macd(close, fast, slow, signal)
    raw_signal = (macd_line > signal_line).astype(int)
    return raw_signal.shift(1).fillna(0)


def signal_bb_squeeze(df: pd.DataFrame, period: int, width: float,
                       squeeze_threshold: float) -> pd.Series:
    """
    Entry: price closes above upper band AFTER the bandwidth was in a squeeze
    (bandwidth < squeeze_threshold).  Exit: price returns to middle band.
    """
    close = df["close"]
    upper, mid, lower, bw = compute_bollinger(close, period, width)

    squeeze = (bw < squeeze_threshold)
    # A breakout is a close above upper band in the bar after a squeeze
    prev_squeeze = squeeze.shift(1).fillna(False)
    breakout     = (close > upper) & prev_squeeze

    # Build position series explicitly
    n   = len(close)
    pos = np.zeros(n)
    in_trade = False

    for i in range(1, n):
        if not in_trade:
            if breakout.iloc[i]:
                in_trade = True
                pos[i] = 1
        else:
            pos[i] = 1
            if close.iloc[i] <= mid.iloc[i]:   # exit at middle band
                in_trade = False
                pos[i] = 0

    raw_signal = pd.Series(pos, index=close.index)
    return raw_signal.shift(1).fillna(0)


def signal_rsi_mean_reversion(df: pd.DataFrame, period: int,
                               buy_thresh: float, sell_thresh: float) -> pd.Series:
    """Buy when RSI < buy_thresh, sell when RSI > sell_thresh."""
    close = df["close"]
    rsi   = compute_rsi(close, period)

    n   = len(close)
    pos = np.zeros(n)
    in_trade = False

    for i in range(1, n):
        if not in_trade:
            if rsi.iloc[i] < buy_thresh:
                in_trade = True
                pos[i] = 1
        else:
            pos[i] = 1
            if rsi.iloc[i] > sell_thresh:
                in_trade = False
                pos[i] = 0

    raw_signal = pd.Series(pos, index=close.index)
    return raw_signal.shift(1).fillna(0)


def signal_donchian_breakout(df: pd.DataFrame, channel: int) -> pd.Series:
    """
    Long on close above channel high; trailing stop at channel low.
    """
    close    = df["close"]
    high     = df["high"]
    low      = df["low"]
    ch_high, ch_low = compute_donchian(close, high, low, channel)

    n   = len(close)
    pos = np.zeros(n)
    in_trade = False

    for i in range(channel + 1, n):
        if not in_trade:
            if close.iloc[i] > ch_high.iloc[i]:
                in_trade = True
                pos[i] = 1
        else:
            pos[i] = 1
            if close.iloc[i] < ch_low.iloc[i]:
                in_trade = False
                pos[i] = 0

    raw_signal = pd.Series(pos, index=close.index)
    return raw_signal.shift(1).fillna(0)


def signal_confluence(df: pd.DataFrame, sma_period: int, ema_fast: int,
                       ema_slow: int, atr_period: int, atr_ma_period: int) -> pd.Series:
    """
    Mirrors smamacd1.py TrendMomentumVolumeStrategy (pure pandas version).
    Long when: close > SMA-200, EMA-fast > EMA-slow, MACD hist > 0, ATR > ATR-MA.
    OBV confirmation is dropped here (OBV already stored in Mongo but we
    recompute to avoid dependency on stored indicator accuracy).
    """
    close  = df["close"]
    volume = df.get("volume", pd.Series(np.ones(len(df)), index=df.index))

    sma200  = compute_sma(close, sma_period)
    ema_f   = compute_ema(close, ema_fast)
    ema_s   = compute_ema(close, ema_slow)
    macd_l, sig_l = compute_macd(close, ema_fast, ema_slow, 9)
    macd_hist     = macd_l - sig_l
    atr     = compute_atr(df, atr_period)
    atr_ma  = compute_sma(atr, atr_ma_period)

    # OBV recomputed
    obv_delta = np.where(close.diff() > 0, volume, np.where(close.diff() < 0, -volume, 0))
    obv = pd.Series(np.cumsum(obv_delta), index=close.index)

    raw_signal = (
        (close > sma200) &
        (ema_f > ema_s) &
        (macd_hist > 0) &
        (atr > atr_ma) &
        (obv > obv.shift(1))
    ).astype(int)
    return raw_signal.shift(1).fillna(0)


def generate_signal(df: pd.DataFrame, config: dict) -> pd.Series:
    """Dispatch to the correct signal function based on family tag."""
    family = config["family"]
    p      = config["params"]

    if family == "SMA_Crossover":
        return signal_sma_crossover(df, p["fast"], p["slow"])
    elif family == "MACD_Signal":
        return signal_macd(df, p["fast"], p["slow"], p["signal"])
    elif family == "BB_Squeeze":
        return signal_bb_squeeze(df, p["period"], p["width"], p["squeeze_threshold"])
    elif family == "RSI_MeanReversion":
        return signal_rsi_mean_reversion(df, p["period"], p["buy_thresh"], p["sell_thresh"])
    elif family == "Breakout_Momentum":
        return signal_donchian_breakout(df, p["channel"])
    elif family == "Multi_Confluence":
        return signal_confluence(df, p["sma_period"], p["ema_fast"], p["ema_slow"],
                                  p["atr_period"], p["atr_ma_period"])
    else:
        raise ValueError(f"Unknown strategy family: {family}")


# ---------------------------------------------------------------------------
# Return computation for a single ticker x strategy
# ---------------------------------------------------------------------------

def compute_ticker_returns(df: pd.DataFrame, position: pd.Series) -> pd.Series:
    """
    Compute daily net returns for a long-only strategy on one ticker.

    Daily gross return = close[t] / close[t-1] - 1  when position == 1.
    Transaction cost TRANSACTION_COST is deducted on entry and exit days
    (i.e., the day the position changes).

    Returns a pd.Series of daily net returns aligned to df.index.
    """
    gross_ret = df["close"].pct_change().fillna(0.0)

    # Identify trade entry and exit days
    pos_change  = position.diff().fillna(position.iloc[0] if len(position) > 0 else 0)
    entry_days  = (pos_change > 0)   # 0->1 transition
    exit_days   = (pos_change < 0)   # 1->0 transition

    # Cost applied on the day the trade is entered or exited
    tc_series   = (entry_days | exit_days).astype(float) * TRANSACTION_COST

    net_ret     = position * gross_ret - tc_series
    return net_ret


def compute_turnover(position: pd.Series) -> float:
    """
    Annualised turnover = (number of round-trip trades / total days held) * TRADING_DAYS.
    Expressed as a fraction.
    """
    trades = (position.diff().abs() / 2).sum()   # each round trip counts as 1
    n_days = len(position)
    if n_days == 0:
        return np.nan
    return float(trades / n_days * TRADING_DAYS_PER_YEAR)


# ---------------------------------------------------------------------------
# Portfolio aggregation
# ---------------------------------------------------------------------------

def equal_weight_portfolio(ticker_returns: dict[str, pd.Series]) -> pd.Series:
    """
    Combine per-ticker daily return series into an equal-weighted portfolio.
    On any given day, only tickers with data contribute; their weights are
    1 / n_tickers_with_data_that_day.
    """
    if not ticker_returns:
        return pd.Series(dtype=float)
    df = pd.DataFrame(ticker_returns)
    # Replace NaN with 0 (ticker not trading that day)
    portfolio = df.mean(axis=1)   # equal-weight average
    return portfolio


# ---------------------------------------------------------------------------
# Performance statistics
# ---------------------------------------------------------------------------

def annualised_return(daily_ret: pd.Series) -> float:
    """Compound annualised return."""
    if daily_ret.empty:
        return np.nan
    total = (1 + daily_ret).prod()
    n_years = len(daily_ret) / TRADING_DAYS_PER_YEAR
    if n_years <= 0:
        return np.nan
    return float(total ** (1.0 / n_years) - 1.0)


def sharpe_ratio(daily_ret: pd.Series, risk_free_daily: float = 0.0) -> float:
    """Annualised Sharpe ratio (excess returns over risk-free)."""
    excess = daily_ret - risk_free_daily
    if excess.std() == 0 or len(excess) < 2:
        return np.nan
    return float(excess.mean() / excess.std() * math.sqrt(TRADING_DAYS_PER_YEAR))


def sortino_ratio(daily_ret: pd.Series, risk_free_daily: float = 0.0) -> float:
    """Annualised Sortino ratio using downside deviation."""
    excess      = daily_ret - risk_free_daily
    downside    = excess[excess < 0]
    if len(downside) < 2 or downside.std() == 0:
        return np.nan
    dd          = math.sqrt((downside ** 2).mean()) * math.sqrt(TRADING_DAYS_PER_YEAR)
    ann_excess  = excess.mean() * TRADING_DAYS_PER_YEAR
    return float(ann_excess / dd)


def max_drawdown(daily_ret: pd.Series) -> float:
    """Maximum peak-to-trough drawdown (negative fraction)."""
    if daily_ret.empty:
        return np.nan
    wealth  = (1 + daily_ret).cumprod()
    rolling_max = wealth.cummax()
    drawdown    = (wealth - rolling_max) / rolling_max
    return float(drawdown.min())


def alpha_vs_benchmark(strat_ret: pd.Series, bench_ret: pd.Series) -> float:
    """
    Jensen's alpha: regress daily strategy excess returns on benchmark excess
    returns (risk-free assumed zero for simplicity).
    Returns annualised alpha.
    """
    if strat_ret.empty or bench_ret.empty:
        return np.nan
    aligned = pd.concat([strat_ret, bench_ret], axis=1, join="inner").dropna()
    if len(aligned) < 2:
        return np.nan
    y  = aligned.iloc[:, 0].values
    x  = aligned.iloc[:, 1].values
    xm = x - x.mean()
    if xm @ xm == 0:
        return np.nan
    beta  = float((xm @ y) / (xm @ xm))
    alpha_daily = float(y.mean() - beta * x.mean())
    return alpha_daily * TRADING_DAYS_PER_YEAR


# ---------------------------------------------------------------------------
# Bootstrap Sharpe CI  (stationary block bootstrap)
# ---------------------------------------------------------------------------

def stationary_block_bootstrap_sharpe(daily_ret: pd.Series,
                                       B: int = BOOTSTRAP_B,
                                       block_length: int = BLOCK_LENGTH) -> tuple[float, float]:
    """
    Compute a 95 % bootstrap confidence interval for the annualised Sharpe ratio
    using a stationary block bootstrap (Politis & Romano 1994).

    Returns (ci_lo, ci_hi).
    Raises RuntimeError if fewer than MIN_OBS_SHARPE monthly observations.
    """
    monthly = daily_ret.resample("ME").apply(lambda x: (1 + x).prod() - 1)
    n_monthly = monthly.dropna()
    if len(n_monthly) < MIN_OBS_SHARPE:
        raise RuntimeError(
            f"Only {len(n_monthly)} monthly observations — minimum {MIN_OBS_SHARPE} required. "
            "Skipping Sharpe CI."
        )

    arr      = daily_ret.dropna().values
    n        = len(arr)
    p        = 1.0 / block_length    # geometric distribution parameter

    boot_sharpes = np.empty(B)
    for b in range(B):
        boot_sample = []
        while len(boot_sample) < n:
            start     = np.random.randint(0, n)
            length    = np.random.geometric(p)
            indices   = [(start + k) % n for k in range(length)]
            boot_sample.extend(arr[indices])
        boot_arr  = np.array(boot_sample[:n])
        std_b     = boot_arr.std()
        if std_b == 0:
            boot_sharpes[b] = 0.0
        else:
            boot_sharpes[b] = float(boot_arr.mean() / std_b * math.sqrt(TRADING_DAYS_PER_YEAR))

    ci_lo = float(np.percentile(boot_sharpes, 2.5))
    ci_hi = float(np.percentile(boot_sharpes, 97.5))
    return ci_lo, ci_hi


# ---------------------------------------------------------------------------
# Romano-Wolf StepM multiple-testing correction
# ---------------------------------------------------------------------------

def romano_wolf_stepm(sharpe_values: np.ndarray,
                       daily_return_matrix: np.ndarray,
                       B: int = BOOTSTRAP_B,
                       block_length: int = BLOCK_LENGTH,
                       alpha: float = ALPHA_LEVEL) -> np.ndarray:
    """
    Romano-Wolf StepM procedure (Romano & Wolf 2005, Econometrica).

    Tests the joint null H0_k: Sharpe_k <= 0 for k = 1, ..., K strategies.

    Parameters
    ----------
    sharpe_values       : shape (K,)  — observed Sharpe ratios
    daily_return_matrix : shape (T, K) — daily portfolio returns per strategy
    B                   : bootstrap replications
    block_length        : block length for stationary block bootstrap
    alpha               : familywise error rate

    Returns
    -------
    rw_pvalues : shape (K,)  — Romano-Wolf adjusted p-values (one per strategy)
    """
    T, K = daily_return_matrix.shape
    p    = 1.0 / block_length

    # --- Step 1: build bootstrap distribution of max Sharpe under H0 ---
    # Centre each strategy's returns at zero (impose null)
    centred = daily_return_matrix - daily_return_matrix.mean(axis=0, keepdims=True)

    boot_max_sharpe = np.empty(B)
    for b in range(B):
        boot_sample = np.empty((T, K))
        pos = 0
        while pos < T:
            start  = np.random.randint(0, T)
            length = np.random.geometric(p)
            idx    = [(start + k) % T for k in range(length)]
            chunk  = centred[idx, :]
            end    = min(pos + len(idx), T)
            boot_sample[pos:end, :] = chunk[:end - pos, :]
            pos = end

        boot_sharpes = np.empty(K)
        for k in range(K):
            col = boot_sample[:, k]
            s   = col.std()
            boot_sharpes[k] = col.mean() / s * math.sqrt(TRADING_DAYS_PER_YEAR) if s > 0 else 0.0

        boot_max_sharpe[b] = boot_sharpes.max()

    # --- Step 2: compute nominal p-values ---
    nominal_pvalues = np.empty(K)
    for k in range(K):
        nominal_pvalues[k] = float(np.mean(boot_max_sharpe >= sharpe_values[k]))

    # --- Step 3: StepM stepdown adjustment ---
    # Sort strategies by descending Sharpe
    order      = np.argsort(-sharpe_values)           # indices: largest Sharpe first
    rw_pvalues = np.ones(K)
    rejected   = np.zeros(K, dtype=bool)

    remaining = list(range(K))                        # indices still in consideration

    while remaining:
        sub_returns = daily_return_matrix[:, remaining]
        sub_centred = sub_returns - sub_returns.mean(axis=0, keepdims=True)

        # Bootstrap max over remaining strategies
        sub_boot_max = np.empty(B)
        for b in range(B):
            boot_sample = np.empty((T, len(remaining)))
            pos = 0
            while pos < T:
                start  = np.random.randint(0, T)
                length = np.random.geometric(p)
                idx    = [(start + k) % T for k in range(length)]
                chunk  = sub_centred[idx, :]
                end    = min(pos + len(idx), T)
                boot_sample[pos:end, :] = chunk[:end - pos, :]
                pos = end

            sub_sharpes = np.empty(len(remaining))
            for j, k in enumerate(remaining):
                col = boot_sample[:, j]
                s   = col.std()
                sub_sharpes[j] = col.mean() / s * math.sqrt(TRADING_DAYS_PER_YEAR) if s > 0 else 0.0

            sub_boot_max[b] = sub_sharpes.max()

        threshold = float(np.percentile(sub_boot_max, (1 - alpha) * 100))

        new_rejections = []
        for k in remaining:
            pval = float(np.mean(sub_boot_max >= sharpe_values[k]))
            # Enforce monotonicity: RW p-value >= previous step's p-value for same strategy
            rw_pvalues[k] = pval
            if sharpe_values[k] > threshold:
                new_rejections.append(k)
                rejected[k] = True

        if not new_rejections:
            break
        remaining = [k for k in remaining if not rejected[k]]

    # Enforce monotonicity of adjusted p-values (Romano & Wolf requirement)
    # Sort by descending test stat; p-values must be non-decreasing in that order
    sorted_idx = np.argsort(-sharpe_values)
    for i in range(1, len(sorted_idx)):
        prev = sorted_idx[i - 1]
        curr = sorted_idx[i]
        rw_pvalues[curr] = max(rw_pvalues[curr], rw_pvalues[prev])

    return rw_pvalues


# ---------------------------------------------------------------------------
# Nominal p-value for individual Sharpe test (bootstrap)
# ---------------------------------------------------------------------------

def nominal_pvalue_sharpe(daily_ret: np.ndarray,
                           observed_sharpe: float,
                           B: int = BOOTSTRAP_B,
                           block_length: int = BLOCK_LENGTH) -> float:
    """
    One-sided bootstrap p-value for H0: Sharpe <= 0.
    Resamples under the null (centred returns) B times.
    """
    T   = len(daily_ret)
    p   = 1.0 / block_length
    arr = daily_ret - daily_ret.mean()   # impose null: mean = 0

    boot_sharpes = np.empty(B)
    for b in range(B):
        boot_sample = []
        pos = 0
        while pos < T:
            start  = np.random.randint(0, T)
            length = np.random.geometric(p)
            idx    = [(start + k) % T for k in range(length)]
            boot_sample.extend(arr[idx])
            pos += length
        bsamp = np.array(boot_sample[:T])
        s = bsamp.std()
        boot_sharpes[b] = bsamp.mean() / s * math.sqrt(TRADING_DAYS_PER_YEAR) if s > 0 else 0.0

    return float(np.mean(boot_sharpes >= observed_sharpe))


# ---------------------------------------------------------------------------
# Benchmark (Nifty 50) loading
# ---------------------------------------------------------------------------

def load_benchmark(db) -> pd.Series:
    """
    Load Nifty 50 daily returns from MongoDB.
    Tries the NIFTY50 collection first, then falls back to NIFTY_50.
    Returns a pd.Series of daily returns indexed by date.
    """
    for name in (NIFTY_TICKER, "NIFTY_50", "NIFTY50INDEX", "Nifty50"):
        try:
            col = db[name]
            raw = pd.DataFrame(list(col.find({})))
            if raw.empty:
                continue
            date_col = "convertedDate" if "convertedDate" in raw.columns else "date"
            if date_col not in raw.columns:
                continue
            raw = raw.rename(columns={date_col: "date"})
            raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
            raw = raw.dropna(subset=["date"]).sort_values("date").set_index("date")
            raw.columns = [c.lower() for c in raw.columns]
            if "close" not in raw.columns:
                continue
            bench = raw.loc[START_DATE:END_DATE, "close"].pct_change().fillna(0.0)
            if len(bench) > 100:
                print(f"  Benchmark loaded from collection '{name}' ({len(bench)} rows).")
                return bench
        except Exception:
            continue

    # Fallback: load from nifty_spot.csv
    spot_path = os.path.join(BASE_DIR, "data", "nifty_spot.csv")
    if os.path.exists(spot_path):
        spot = pd.read_csv(spot_path, parse_dates=["date"])
        spot = spot.sort_values("date").set_index("date")
        spot.columns = [c.lower() for c in spot.columns]
        bench = spot.loc[START_DATE:END_DATE, "close"].pct_change().fillna(0.0)
        if len(bench) > 100:
            print(f"  Benchmark loaded from {spot_path} ({len(bench)} rows).")
            return bench

    print("  WARNING: Nifty 50 benchmark not found. Alpha will be NaN.")
    return pd.Series(dtype=float)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_strategy_on_universe(config: dict, db, tickers: list[str]) -> dict:
    """
    Run one strategy configuration across all tickers.

    Returns a dict with:
        portfolio_returns  — pd.Series  (daily equal-weighted)
        ticker_results     — list of per-ticker dicts
        turnover           — float (avg annualised turnover)
        n_tickers_used     — int
    """
    ticker_return_series = {}
    turnovers            = []
    n_skipped            = 0

    for ticker in tickers:
        df = load_price_data(db, ticker)
        if df is None:
            n_skipped += 1
            continue

        try:
            position = generate_signal(df, config)
            net_ret  = compute_ticker_returns(df, position)
            to_val   = compute_turnover(position)
        except Exception as exc:
            print(f"    Warning: {ticker} failed for {config['name']} — {exc}")
            n_skipped += 1
            continue

        ticker_return_series[ticker] = net_ret
        if not np.isnan(to_val):
            turnovers.append(to_val)

    portfolio = equal_weight_portfolio(ticker_return_series)
    avg_to    = float(np.mean(turnovers)) if turnovers else np.nan
    n_used    = len(ticker_return_series)

    return {
        "portfolio_returns": portfolio,
        "n_tickers_used":    n_used,
        "turnover":          avg_to,
    }


def summarise_strategy(config: dict,
                        result: dict,
                        bench_ret: pd.Series) -> dict:
    """
    Compute all headline statistics for one strategy configuration.
    Returns a dict matching the output CSV columns.
    """
    port_ret = result["portfolio_returns"].dropna()

    if port_ret.empty or len(port_ret) < 2:
        return {
            "strategy_name": config["name"],
            "params":        str(config["params"]),
            "n_tickers":     result["n_tickers_used"],
            "ann_return":    np.nan,
            "alpha_vs_nifty": np.nan,
            "sharpe":        np.nan,
            "sharpe_ci_lo":  np.nan,
            "sharpe_ci_hi":  np.nan,
            "sortino":       np.nan,
            "max_dd":        np.nan,
            "turnover":      result["turnover"],
            "nominal_pvalue":    np.nan,
            "rw_adjusted_pvalue": np.nan,
        }

    ann_ret = annualised_return(port_ret)
    sr      = sharpe_ratio(port_ret)
    so      = sortino_ratio(port_ret)
    mdd     = max_drawdown(port_ret)
    alpha   = alpha_vs_benchmark(port_ret, bench_ret)

    # Bootstrap Sharpe CI
    try:
        ci_lo, ci_hi = stationary_block_bootstrap_sharpe(port_ret)
    except RuntimeError as exc:
        print(f"    Sharpe CI skipped: {exc}")
        ci_lo, ci_hi = np.nan, np.nan

    # Nominal p-value (placeholder; overwritten by RW in final pass)
    nom_pval = nominal_pvalue_sharpe(port_ret.values,
                                     sr if not np.isnan(sr) else 0.0)

    return {
        "strategy_name": config["name"],
        "params":        str(config["params"]),
        "n_tickers":     result["n_tickers_used"],
        "ann_return":    round(ann_ret, 6) if not np.isnan(ann_ret) else np.nan,
        "alpha_vs_nifty": round(alpha, 6) if not np.isnan(alpha) else np.nan,
        "sharpe":        round(sr, 4) if not np.isnan(sr) else np.nan,
        "sharpe_ci_lo":  round(ci_lo, 4) if not np.isnan(ci_lo) else np.nan,
        "sharpe_ci_hi":  round(ci_hi, 4) if not np.isnan(ci_hi) else np.nan,
        "sortino":       round(so, 4) if not np.isnan(so) else np.nan,
        "max_dd":        round(mdd, 4) if not np.isnan(mdd) else np.nan,
        "turnover":      round(result["turnover"], 4) if not np.isnan(result["turnover"]) else np.nan,
        "nominal_pvalue":    round(nom_pval, 4),
        "rw_adjusted_pvalue": np.nan,   # filled in later
    }


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

def sanity_check_buy_and_hold(db, tickers: list[str]) -> None:
    """
    Sanity check: the simplest possible long-only buy-and-hold on the first
    5 tickers should produce a positive Sharpe ratio over 2015-2025 (Indian
    equity bull run).  Halt if this is violated — it likely signals a data
    or return-calculation bug.
    """
    print("\n--- Sanity check: buy-and-hold Sharpe ---")
    results = []
    for ticker in tickers[:5]:
        df = load_price_data(db, ticker)
        if df is None:
            continue
        ret = df["close"].pct_change().dropna()
        sr  = sharpe_ratio(ret)
        results.append((ticker, sr))
        print(f"  {ticker}: Sharpe = {sr:.3f}")

    valid = [(t, s) for t, s in results if not np.isnan(s)]
    if not valid:
        sys.exit("SANITY FAIL: could not compute buy-and-hold Sharpe for any ticker. "
                 "Check MongoDB connection and data.")

    avg_bh_sharpe = np.mean([s for _, s in valid])
    if avg_bh_sharpe < 0.0:
        sys.exit(
            f"SANITY FAIL: average buy-and-hold Sharpe = {avg_bh_sharpe:.3f} (expected > 0 "
            "for Indian equities 2015-2025). Possible data error. Halting."
        )
    print(f"  Average buy-and-hold Sharpe = {avg_bh_sharpe:.3f}  [PASS]\n")


# ---------------------------------------------------------------------------
# Write outputs
# ---------------------------------------------------------------------------

def write_outputs(perf_rows: list[dict],
                   all_port_returns: dict[str, pd.Series]) -> None:
    """
    Write:
      papers/P2/tables/strategy_performance.csv
      data/p2_daily_returns.csv
    """
    # Performance table
    perf_df  = pd.DataFrame(perf_rows)
    perf_path = os.path.join(TABLES_DIR, "strategy_performance.csv")
    perf_df.to_csv(perf_path, index=False)
    print(f"\nWrote performance table: {perf_path}")

    # Daily returns matrix
    ret_df   = pd.DataFrame(all_port_returns)
    ret_path = os.path.join(DATA_DIR, "p2_daily_returns.csv")
    ret_df.to_csv(ret_path)
    print(f"Wrote daily returns:      {ret_path}")


def write_mongo_results(perf_rows: list[dict]) -> None:
    """
    Mirror the performance table to MongoDB collection paper_results.
    """
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        db_out = client["paper_results"]
        col    = db_out["p2_strategy_performance"]
        col.delete_many({})    # replace previous run
        records = []
        for row in perf_rows:
            rec = dict(row)
            rec["run_timestamp"] = datetime.utcnow().isoformat()
            records.append(rec)
        if records:
            col.insert_many(records)
        print(f"Wrote {len(records)} records to MongoDB paper_results.p2_strategy_performance")
        client.close()
    except Exception as exc:
        print(f"WARNING: MongoDB write to paper_results failed — {exc}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    """
    Main execution pipeline.

    Steps:
      1. Load tickers and connect to MongoDB.
      2. Sanity check buy-and-hold.
      3. For each strategy config, run backtest across all 78 tickers.
      4. Compute headline statistics per strategy.
      5. Run Romano-Wolf StepM multiple-testing correction jointly.
      6. Write outputs to CSV and MongoDB.
    """
    ensure_dirs()
    tickers = load_tickers(TICKER_FILE)

    client = get_mongo_client()
    db     = client[MONGO_DB_NAME]

    bench_ret = load_benchmark(db)

    sanity_check_buy_and_hold(db, tickers)

    all_port_returns: dict[str, pd.Series] = {}
    strategy_results: list[dict]           = []   # raw result dicts from run_strategy_on_universe
    perf_rows: list[dict]                  = []

    total = len(STRATEGY_CONFIGS)
    for idx, config in enumerate(STRATEGY_CONFIGS, start=1):
        print(f"\n[{idx}/{total}] Strategy: {config['name']}  ({config['family']})")

        result = run_strategy_on_universe(config, db, tickers)

        if result["portfolio_returns"].empty:
            print(f"  No portfolio returns — all tickers skipped for {config['name']}.")
            continue

        all_port_returns[config["name"]] = result["portfolio_returns"]
        strategy_results.append((config, result))

        row = summarise_strategy(config, result, bench_ret)
        perf_rows.append(row)

        sr_display = row["sharpe"] if not (isinstance(row["sharpe"], float) and np.isnan(row["sharpe"])) else "NaN"
        print(f"  n_tickers={row['n_tickers']}  ann_ret={row['ann_return']:.2%}  "
              f"Sharpe={sr_display}  max_dd={row['max_dd']:.2%}  "
              f"nom_p={row['nominal_pvalue']:.4f}")

    # -------------------------------------------------------------------
    # Romano-Wolf StepM correction across all strategies with valid Sharpe
    # -------------------------------------------------------------------
    print("\n--- Romano-Wolf StepM multiple-testing correction ---")

    valid_idx    = [i for i, r in enumerate(perf_rows)
                    if not (isinstance(r["sharpe"], float) and np.isnan(r["sharpe"]))]

    if len(valid_idx) >= 2:
        sharpe_arr = np.array([perf_rows[i]["sharpe"] for i in valid_idx])

        # Align all return series to a common date index
        valid_names = [perf_rows[i]["strategy_name"] for i in valid_idx]
        ret_df      = pd.DataFrame({n: all_port_returns[n] for n in valid_names}).dropna()
        ret_matrix  = ret_df.values   # shape (T, K)

        if ret_matrix.shape[0] < MIN_OBS_SHARPE * TRADING_DAYS_PER_YEAR / 12:
            print("  WARNING: insufficient aligned observations for RW correction. "
                  "Nominal p-values retained.")
        else:
            rw_pvals = romano_wolf_stepm(sharpe_arr, ret_matrix)
            for j, i in enumerate(valid_idx):
                perf_rows[i]["rw_adjusted_pvalue"] = round(float(rw_pvals[j]), 4)
            print("  Romano-Wolf p-values computed.")
    else:
        print("  Fewer than 2 valid strategies — Romano-Wolf skipped.")

    # Copy nominal p-values to rw column where RW was skipped
    for row in perf_rows:
        if np.isnan(row.get("rw_adjusted_pvalue", np.nan)):
            row["rw_adjusted_pvalue"] = row["nominal_pvalue"]

    client.close()

    # -------------------------------------------------------------------
    # Write outputs
    # -------------------------------------------------------------------
    write_outputs(perf_rows, all_port_returns)
    write_mongo_results(perf_rows)

    # -------------------------------------------------------------------
    # Print final summary
    # -------------------------------------------------------------------
    print("\n=== FINAL RESULTS SUMMARY ===")
    print(f"{'Strategy':<20} {'Sharpe':>8} {'Ann Ret':>9} {'RW p-val':>10}")
    print("-" * 55)
    for row in sorted(perf_rows, key=lambda r: r.get("sharpe") or -99, reverse=True):
        sr  = row["sharpe"]
        ar  = row["ann_return"]
        rw  = row["rw_adjusted_pvalue"]
        sr_s = f"{sr:.4f}" if isinstance(sr, float) and not np.isnan(sr) else "   NaN"
        ar_s = f"{ar:.2%}"  if isinstance(ar, float) and not np.isnan(ar) else "    NaN"
        rw_s = f"{rw:.4f}" if isinstance(rw, float) and not np.isnan(rw) else "   NaN"
        print(f"{row['strategy_name']:<20} {sr_s:>8} {ar_s:>9} {rw_s:>10}")

    print("\nDone. Check papers/P2/tables/strategy_performance.csv for the full table.")


if __name__ == "__main__":
    main()
