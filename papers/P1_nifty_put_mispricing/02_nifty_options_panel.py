"""
nifty_options_panel.py
======================
Monthly Held-to-Maturity Option Panel Builder — Paper P1.

Follows the Broadie, Chernov and Johannes (2009) [BCJ] methodology for
constructing monthly held-to-maturity option returns on the Nifty 50 index.

INPUT FILES (produced by bhavcopy_downloader.py and nifty_spot_downloader.py):
  data/nifty_options/YYYY-MM-DD.csv  — one file per trading day, OPTIDX rows
  data/nifty_spot.csv                — Nifty 50 daily close (Yahoo Finance format)

OUTPUT:
  data/nifty_options_panel.csv       — one row per monthly expiry cycle
  MongoDB: optionsdatabase.nifty_options_panel  (keyed on `month`)

PANEL CONSTRUCTION LOGIC:
  For each monthly expiry (last Thursday of each calendar month in the sample):
    1. Entry date  : first available trading day on or after the previous expiry + 1.
    2. Expiry date : last Thursday of the current month (or nearest prior trading day).
    3. Spot at entry : Nifty 50 close on the entry date.
    4. Select puts by moneyness k = Strike / Spot_at_entry:
         k ≈ 0.94  (6% OTM put)
         k ≈ 0.96  (4% OTM put)
         k ≈ 0.98  (2% OTM put)
         k ≈ 1.00  (ATM put)
       Closest available strike used; marked missing if no strike within 1% of target.
    5. HTM put return: settle_price_at_expiry / settle_price_at_entry - 1
       Expiry settle falls back to intrinsic value max(K - S_expiry, 0) if the
       recorded settle_price is zero or NaN.
    6. Strategy returns:
         ATM Short Straddle   : sell ATM CE + ATM PE at entry, hold to expiry.
         Crash-Neutral (CN)   : short ATM straddle + long 6% OTM put.
         Put Spread (PSP)     : long k=0.96 put, short k=1.00 put.

MONTHLY vs WEEKLY EXPIRY DISAMBIGUATION:
  NSE listed weekly Nifty options from May 2019 onwards. The bhavcopy contains
  both. A monthly expiry is identified as the LAST Thursday of a calendar month.
  When loading available expiry dates from the bhavcopy, only dates that are the
  last Thursday of their respective month are treated as monthly expiries.

IDEMPOTENCY:
  If data/nifty_options_panel.csv already exists, months already present are
  skipped. Only new months (or months with now-available data) are processed.

RESUMABILITY:
  Month iteration is fully sorted. If the script dies on month N, rerunning it
  will skip months 1..N-1 (already in CSV) and restart from month N.

Dependencies:
    pip install pandas python-dotenv pymongo

Usage:
    python nifty_options_panel.py

Author: Sumin Pillai (Independent Researcher)
"""

import os
import time
from datetime import datetime, timedelta, date

import pandas as pd
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# CONFIGURATION — change these constants to adjust the panel window.
# ---------------------------------------------------------------------------
START_DATE = "2015-01-01"
END_DATE   = "2026-05-12"
# Moneyness targets (k = Strike / Spot_at_entry)
K_TARGETS = {
    "k094": 0.94,
    "k096": 0.96,
    "k098": 0.98,
    "k100": 1.00,
}
# Maximum allowed deviation from target moneyness before marking as missing.
K_TOLERANCE = 0.01   # 1%
# ---------------------------------------------------------------------------

BASE_DIR         = "C:\\Users\\sumin\\definedge_downloader"
OPTIONS_FOLDER   = os.path.join(BASE_DIR, "data", "nifty_options")
SPOT_CSV_PATH    = os.path.join(BASE_DIR, "data", "nifty_spot.csv")
OUTPUT_CSV_PATH  = os.path.join(BASE_DIR, "data", "nifty_options_panel.csv")

MONGO_CONNECTION_STRING = "mongodb://localhost:27017/"
MONGO_DATABASE_NAME     = "optionsdatabase"
MONGO_COLLECTION_NAME   = "nifty_options_panel"


# ---------------------------------------------------------------------------
# setup_environment
# ---------------------------------------------------------------------------

def setup_environment():
    """Creates output folders and confirms input paths exist."""
    data_dir = os.path.join(BASE_DIR, "data")
    for folder in (data_dir, OPTIONS_FOLDER):
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f"Created output folder: '{folder}'")
        else:
            print(f"Output folder exists: '{folder}'")

    if not os.path.isfile(SPOT_CSV_PATH):
        print(
            f"WARNING: Spot CSV not found at '{SPOT_CSV_PATH}'. "
            "Run nifty_spot_downloader.py first."
        )
    else:
        print(f"Spot CSV found: '{SPOT_CSV_PATH}'")

    options_files = [
        f for f in os.listdir(OPTIONS_FOLDER) if f.endswith(".csv")
    ] if os.path.isdir(OPTIONS_FOLDER) else []
    print(f"Options bhavcopy files available: {len(options_files)}")
    return len(options_files)


# ---------------------------------------------------------------------------
# load_spot_from_file
# ---------------------------------------------------------------------------

def load_spot_from_file() -> pd.DataFrame | None:
    """
    Loads Nifty 50 daily close prices from data/nifty_spot.csv.

    Expects columns: date, open, high, low, close  (nifty_spot_downloader.py
    format).  Returns a DataFrame indexed by date (datetime64, tz-naive) with
    a 'close' column.  Returns None and logs an error if the file is missing
    or malformed.
    """
    if not os.path.isfile(SPOT_CSV_PATH):
        print(f"ERROR: Spot CSV missing at '{SPOT_CSV_PATH}'.")
        return None

    try:
        df = pd.read_csv(SPOT_CSV_PATH, parse_dates=["date"])
    except Exception as exc:
        print(f"ERROR: Could not read spot CSV: {exc}")
        return None

    if "date" not in df.columns or "close" not in df.columns:
        print(
            f"ERROR: Spot CSV must contain 'date' and 'close' columns. "
            f"Found: {list(df.columns)}"
        )
        return None

    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    df = df.sort_values("date").set_index("date")
    print(
        f"Spot data loaded: {len(df):,} rows "
        f"({df.index.min().date()} to {df.index.max().date()})."
    )
    return df


# ---------------------------------------------------------------------------
# load_options_for_date
# ---------------------------------------------------------------------------

def load_options_for_date(trade_date: datetime) -> pd.DataFrame | None:
    """
    Loads the options bhavcopy CSV for a single trading date from
    data/nifty_options/YYYY-MM-DD.csv.

    Returns a DataFrame with the canonical bhavcopy column schema, or None
    if the file does not exist (holiday / date not yet downloaded).
    """
    csv_path = os.path.join(
        OPTIONS_FOLDER, trade_date.strftime("%Y-%m-%d") + ".csv"
    )
    if not os.path.isfile(csv_path):
        return None

    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        print(f"    WARNING: Could not read options CSV {csv_path}: {exc}")
        return None

    if df.empty:
        return None

    # Force datetime conversion for date columns
    for dcol in ("expiry_date", "timestamp"):
        if dcol in df.columns:
            df[dcol] = pd.to_datetime(df[dcol], errors="coerce")

    # Ensure string columns are clean.
    for col in ("option_type", "symbol", "instrument"):
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # Normalise numeric columns that may have been stored as strings.
    for col in ("strike_price", "settle_price", "close", "open"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# ---------------------------------------------------------------------------
# get_mongo_collection
# ---------------------------------------------------------------------------

def get_mongo_collection():
    """
    Connects to MongoDB and returns the target collection object from
    `optionsdatabase`.  Returns None if pymongo is not installed or the
    server is unreachable — CSV output continues regardless.
    """
    load_dotenv()
    try:
        from pymongo import MongoClient
    except ImportError:
        print(
            "WARNING: pymongo is not installed. "
            "Install with:  pip install pymongo\n"
            "CSV output will still be written; MongoDB upsert will be skipped."
        )
        return None

    try:
        client = MongoClient(MONGO_CONNECTION_STRING, serverSelectionTimeoutMS=5000)
        client.admin.command("ismaster")
        db         = client[MONGO_DATABASE_NAME]
        collection = db[MONGO_COLLECTION_NAME]
        print(
            f"Connected to MongoDB — database '{MONGO_DATABASE_NAME}', "
            f"collection '{MONGO_COLLECTION_NAME}'."
        )
        return collection
    except Exception as exc:
        print(
            f"WARNING: Could not connect to MongoDB ({exc}). "
            "CSV output will still be written; MongoDB upsert will be skipped."
        )
        return None


# ---------------------------------------------------------------------------
# calendar helpers
# ---------------------------------------------------------------------------

def last_thursday_of_month(year: int, month: int) -> date:
    """Returns the last Thursday of the given calendar month."""
    # Find the last day of the month.
    if month == 12:
        last_day = date(year + 1, 1, 1) - timedelta(days=1)
    else:
        last_day = date(year, month + 1, 1) - timedelta(days=1)

    # Walk backwards from last_day until we hit Thursday (weekday == 3).
    offset = (last_day.weekday() - 3) % 7
    return last_day - timedelta(days=offset)


def is_last_thursday_of_month(d: date) -> bool:
    """Returns True if d is the last Thursday of its calendar month."""
    return d.weekday() == 3 and d == last_thursday_of_month(d.year, d.month)


def nearest_prior_trading_day(target: date, trading_days: set) -> date | None:
    """
    Returns target itself if it is a trading day, else walks backwards until a
    trading day is found (up to 10 calendar days back).  Returns None if none
    found.
    """
    for offset in range(10):
        candidate = target - timedelta(days=offset)
        if candidate in trading_days:
            return candidate
    return None


def nearest_following_trading_day(target: date, trading_days: set) -> date | None:
    """
    Returns target itself if it is a trading day, else walks forward until a
    trading day is found (up to 10 calendar days ahead).  Returns None if none
    found.
    """
    for offset in range(10):
        candidate = target + timedelta(days=offset)
        if candidate in trading_days:
            return candidate
    return None


# ---------------------------------------------------------------------------
# build_monthly_expiry_schedule
# ---------------------------------------------------------------------------

def build_monthly_expiry_schedule(
    start_dt: datetime,
    end_dt: datetime,
    spot_df: pd.DataFrame,
) -> list[dict]:
    """
    Builds a list of monthly expiry cycles within [start_dt, end_dt].

    Each cycle is a dict:
        month        : "YYYY-MM"  (the expiry month)
        expiry_date  : date — last Thursday of the month (adjusted for holidays)
        entry_date   : date — first available trading day after the previous expiry
        prev_expiry  : date — previous month's expiry (used only for entry-date calc)

    Only includes cycles where:
      - The expiry falls within the requested date range.
      - Both entry_date and expiry_date have spot data available.

    Trading days are inferred from the spot DataFrame index (days that have a
    close price).  This is the most reliable signal for whether a day is
    actually a trading day — it is consistent with real NSE market days.
    """
    trading_days: set[date] = set(
        d.date() for d in spot_df.index
    )

    schedule = []
    year  = start_dt.year
    month = start_dt.month

    # We need to iterate from one month before start_dt to capture the entry
    # date correctly for the first cycle.
    prev_expiry_date: date | None = None

    # Seed: find last-Thursday of the month preceding start_dt.
    seed_month = month - 1 if month > 1 else 12
    seed_year  = year if month > 1 else year - 1
    prev_expiry_raw = last_thursday_of_month(seed_year, seed_month)
    prev_expiry_date = nearest_prior_trading_day(prev_expiry_raw, trading_days)

    while True:
        # Compute expiry for this month.
        expiry_raw  = last_thursday_of_month(year, month)
        expiry_date = nearest_prior_trading_day(expiry_raw, trading_days)

        if expiry_date is None:
            # Expiry falls outside our data range — stop
            break

        expiry_as_dt = datetime(expiry_date.year, expiry_date.month, expiry_date.day)
        if expiry_as_dt > end_dt:
            break

        # Entry date: first trading day on or after (prev_expiry + 1 calendar day).
        if prev_expiry_date is None:
            # Seed month before data range — use first available trading day
            entry_start_raw = min(trading_days)
        else:
            entry_start_raw = prev_expiry_date + timedelta(days=1)
        entry_date = nearest_following_trading_day(entry_start_raw, trading_days)

        entry_as_dt = datetime(entry_date.year, entry_date.month, entry_date.day)
        if entry_as_dt >= datetime(start_dt.year, start_dt.month, start_dt.day):
            schedule.append({
                "month":       f"{year:04d}-{month:02d}",
                "expiry_date": expiry_date,
                "entry_date":  entry_date,
                "prev_expiry": prev_expiry_date,
            })

        prev_expiry_date = expiry_date

        # Advance month.
        month += 1
        if month > 12:
            month = 1
            year += 1

    print(f"Monthly expiry schedule built: {len(schedule)} cycles.")
    return schedule


# ---------------------------------------------------------------------------
# identify_monthly_expiry_from_bhavcopy
# ---------------------------------------------------------------------------

def identify_monthly_expiry_dates_in_bhavcopy(
    options_df: pd.DataFrame,
) -> set[date]:
    """
    Given a bhavcopy DataFrame for one day, returns the set of expiry dates
    present that are last-Thursday-of-month (i.e., monthly expiries, not
    weekly expiries introduced from May 2019).
    """
    if options_df is None or options_df.empty:
        return set()
    expiry_dates = options_df["expiry_date"].dropna().dt.date.unique()
    return {d for d in expiry_dates if is_last_thursday_of_month(d)}


# ---------------------------------------------------------------------------
# select_strike_for_k
# ---------------------------------------------------------------------------

def select_strike_for_k(
    options_df: pd.DataFrame,
    spot: float,
    k_target: float,
    option_type: str,   # "PE" or "CE"
    expiry_date: date,
    label: str,         # e.g. "k094" — used only for logging
) -> dict | None:
    """
    From options_df (all options for one trading day), selects the contract
    closest to k_target * spot among those with the specified option_type
    and expiry_date.

    Returns a dict with keys: strike, settle_price.
    Returns None if no strike is within K_TOLERANCE of the target or the
    settle_price is missing/zero for every candidate.
    """
    target_strike = k_target * spot

    # Filter to the target expiry and option type.
    mask = (
        (options_df["option_type"] == option_type)
        & (options_df["expiry_date"].dt.date == expiry_date)
    )
    subset = options_df[mask].copy()

    if subset.empty:
        return None

    # Select the strike closest to the target.
    subset = subset.dropna(subset=["strike_price"])
    if subset.empty:
        return None

    subset["_dist"] = (subset["strike_price"] - target_strike).abs()
    best_row = subset.loc[subset["_dist"].idxmin()]

    actual_k   = best_row["strike_price"] / spot
    k_dev      = abs(actual_k - k_target)

    if k_dev > K_TOLERANCE:
        return None

    settle = best_row["settle_price"]
    if pd.isna(settle) or settle <= 0:
        # Fall back to close price if settle is missing.
        settle = best_row.get("close", float("nan"))
        if pd.isna(settle) or settle <= 0:
            print(
                f"      WARNING [{label} {option_type}]: strike {best_row['strike_price']:.0f} "
                f"has zero/NaN settle and zero/NaN close on {expiry_date} — skipped."
            )
            return None

    return {
        "strike":       float(best_row["strike_price"]),
        "settle_price": float(settle),
    }


# ---------------------------------------------------------------------------
# get_expiry_settle
# ---------------------------------------------------------------------------

def get_expiry_settle(
    expiry_date: date,
    strike: float,
    option_type: str,
    spot_at_expiry: float,
    options_df_expiry: pd.DataFrame | None,
    label: str,
) -> float:
    """
    Returns the settle price of a contract on its expiry date.

    Priority order:
      1. settle_price from the bhavcopy on expiry_date (if > 0 and not NaN).
      2. close price from the bhavcopy on expiry_date (if > 0 and not NaN).
      3. Intrinsic value:
           PE: max(strike - spot_at_expiry, 0)
           CE: max(spot_at_expiry - strike, 0)

    The intrinsic-value fallback is correct at expiry by definition and is
    equivalent to the payoff used in BCJ.
    """
    if options_df_expiry is not None and not options_df_expiry.empty:
        mask = (
            (options_df_expiry["option_type"] == option_type)
            & (options_df_expiry["expiry_date"].dt.date == expiry_date)
            & (options_df_expiry["strike_price"].round(1) == round(strike, 1))
        )
        match = options_df_expiry[mask]
        if not match.empty:
            sp = match.iloc[0]["settle_price"]
            cl = match.iloc[0]["close"]

            # NSE puts the underlying index settlement level in settle_price on
            # expiry day for all option rows in the bhavcopy. Detect this:
            # 1. For a PE, a valid option settle must be <= strike.
            # 2. For a CE, a valid option settle must be <= strike * 2.
            # 3. If close is available and settle/close ratio > 10, settle is
            #    the index level (catches ITM puts where index < strike).
            sp_valid = (not pd.isna(sp)) and sp > 0
            if sp_valid and option_type == "PE" and sp > strike:
                sp_valid = False  # index settlement leaked into settle_price column
            if sp_valid and option_type == "CE" and sp > strike * 2:
                sp_valid = False
            if sp_valid and (not pd.isna(cl)) and cl > 0 and sp / cl > 10:
                sp_valid = False  # settle is index level, close is real option price

            if sp_valid:
                return float(sp)
            # Try close.
            if not pd.isna(cl) and cl > 0:
                return float(cl)

    # Intrinsic value fallback.
    if option_type == "PE":
        intrinsic = max(strike - spot_at_expiry, 0.0)
    else:
        intrinsic = max(spot_at_expiry - strike, 0.0)

    print(
        f"      INFO [{label} {option_type} K={strike:.0f}]: using intrinsic value "
        f"({intrinsic:.2f}) at expiry (bhavcopy settle not available)."
    )
    return intrinsic


# ---------------------------------------------------------------------------
# calculate_strategy_returns
# ---------------------------------------------------------------------------

def calculate_strategy_returns(
    put_data: dict,  # keyed by k-label ("k094" etc) -> {"strike", "settle_price", "expiry_settle"}
    call_atm: dict | None,   # {"entry", "expiry_settle"}  for ATM CE
) -> dict:
    """
    Computes three BCJ strategy returns from per-contract entry and expiry
    settle prices:

      straddle  : Sell ATM CE + ATM PE at entry.
                  Return = (CE_entry + PE_entry - CE_expiry - PE_expiry)
                           / (CE_entry + PE_entry)

      cn        : Crash-Neutral — short straddle + long 6% OTM put.
                  Net_entry  = CE_entry + PE_ATM_entry - PE_k094_entry
                  Net_expiry = CE_expiry + PE_ATM_expiry - PE_k094_expiry
                  Return     = (Net_entry - Net_expiry) / abs(Net_entry)

      psp       : Put-Spread Portfolio — long k=0.96 PE, short k=1.00 PE.
                  Net_entry  = PE_k096_entry - PE_k100_entry
                  Net_expiry = PE_k096_expiry - PE_k100_expiry
                  Return     = (Net_entry - Net_expiry) / abs(Net_entry)
                  (net_entry is typically negative: we pay net premium)

    All returns are NaN when required data is missing or entry premium is zero.
    """
    nan = float("nan")

    def _safe(d, key):
        return d.get(key, nan) if d else nan

    # put_data dicts use "settle_price" for the entry premium (set by select_strike_for_k)
    # and "expiry_settle" for the expiry value (set in fetch_panel_row).
    # call_atm dict uses "entry" and "expiry_settle" (assembled in fetch_panel_row).
    atm_pe_entry  = _safe(put_data.get("k100"), "settle_price")
    atm_pe_expiry = _safe(put_data.get("k100"), "expiry_settle")
    atm_ce_entry  = _safe(call_atm, "entry")
    atm_ce_expiry = _safe(call_atm, "expiry_settle")
    k094_pe_entry  = _safe(put_data.get("k094"), "settle_price")
    k094_pe_expiry = _safe(put_data.get("k094"), "expiry_settle")
    k096_pe_entry  = _safe(put_data.get("k096"), "settle_price")
    k096_pe_expiry = _safe(put_data.get("k096"), "expiry_settle")

    results = {
        "straddle_entry_premium":  nan,
        "straddle_expiry_payoff":  nan,
        "straddle_return":         nan,
        "cn_entry_premium":        nan,
        "cn_expiry_payoff":        nan,
        "cn_return":               nan,
        "psp_entry_premium":       nan,
        "psp_expiry_payoff":       nan,
        "psp_return":              nan,
    }

    # --- ATM Short Straddle ---
    if all(not _is_nan(v) for v in [atm_ce_entry, atm_pe_entry,
                                     atm_ce_expiry, atm_pe_expiry]):
        straddle_entry  = atm_ce_entry + atm_pe_entry
        straddle_expiry = atm_ce_expiry + atm_pe_expiry
        if straddle_entry > 0:
            results["straddle_entry_premium"] = straddle_entry
            results["straddle_expiry_payoff"]  = straddle_expiry
            results["straddle_return"] = (
                (straddle_entry - straddle_expiry) / straddle_entry
            )

    # --- Crash-Neutral (CN) ---
    if all(not _is_nan(v) for v in [atm_ce_entry, atm_pe_entry, k094_pe_entry,
                                     atm_ce_expiry, atm_pe_expiry, k094_pe_expiry]):
        cn_entry  = atm_ce_entry + atm_pe_entry - k094_pe_entry
        cn_expiry = atm_ce_expiry + atm_pe_expiry - k094_pe_expiry
        if abs(cn_entry) > 0:
            results["cn_entry_premium"] = cn_entry
            results["cn_expiry_payoff"]  = cn_expiry
            results["cn_return"] = (cn_entry - cn_expiry) / abs(cn_entry)

    # --- Put Spread Portfolio (PSP) ---
    if all(not _is_nan(v) for v in [k096_pe_entry, atm_pe_entry,
                                     k096_pe_expiry, atm_pe_expiry]):
        psp_entry  = k096_pe_entry - atm_pe_entry   # typically negative (net cost)
        psp_expiry = k096_pe_expiry - atm_pe_expiry
        if abs(psp_entry) > 0:
            results["psp_entry_premium"] = psp_entry
            results["psp_expiry_payoff"]  = psp_expiry
            results["psp_return"] = (psp_entry - psp_expiry) / abs(psp_entry)

    return results


def _is_nan(v) -> bool:
    """Returns True for Python float nan or pandas NA."""
    try:
        return v != v   # IEEE 754: NaN != NaN
    except Exception:
        return True


# ---------------------------------------------------------------------------
# fetch_panel_row  (one monthly cycle)
# ---------------------------------------------------------------------------

def fetch_panel_row(
    cycle: dict,
    spot_df: pd.DataFrame,
) -> dict | None:
    """
    Builds one panel row for the given monthly expiry cycle.

    Returns a dict of all output columns or None if the cycle cannot be
    built (e.g. entry or expiry date has no bhavcopy data).

    cycle keys: month, expiry_date, entry_date, prev_expiry
    """
    month_str   = cycle["month"]
    entry_dt    = cycle["entry_date"]      # date
    expiry_dt   = cycle["expiry_date"]     # date

    entry_ts  = datetime(entry_dt.year,  entry_dt.month,  entry_dt.day)
    expiry_ts = datetime(expiry_dt.year, expiry_dt.month, expiry_dt.day)

    # ---- Spot at entry ----
    spot_ts_entry = pd.Timestamp(entry_ts)
    if spot_ts_entry not in spot_df.index:
        print(f"  [{month_str}] SKIP — entry date {entry_dt} not in spot data.")
        return None
    spot_entry = float(spot_df.loc[spot_ts_entry, "close"])

    # ---- Spot at expiry ----
    spot_ts_expiry = pd.Timestamp(expiry_ts)
    if spot_ts_expiry not in spot_df.index:
        print(f"  [{month_str}] SKIP — expiry date {expiry_dt} not in spot data.")
        return None
    spot_expiry = float(spot_df.loc[spot_ts_expiry, "close"])

    # ---- Load bhavcopy for entry and expiry dates ----
    opt_entry  = load_options_for_date(entry_ts)
    opt_expiry = load_options_for_date(expiry_ts)

    if opt_entry is None:
        print(
            f"  [{month_str}] SKIP — no bhavcopy CSV for entry date {entry_dt}. "
            "(Run bhavcopy_downloader.py to populate data/nifty_options/.)"
        )
        return None

    # Validate that this entry-day CSV actually contains the monthly expiry.
    monthly_expiries_on_entry_day = identify_monthly_expiry_dates_in_bhavcopy(opt_entry)
    if expiry_dt not in monthly_expiries_on_entry_day:
        # The monthly expiry might not be in the entry-day file if this is the
        # very first listed day — search the next few trading days for it.
        found = False
        for offset in range(1, 6):
            alt_dt  = datetime(
                entry_dt.year, entry_dt.month, entry_dt.day
            ) + timedelta(days=offset)
            alt_df  = load_options_for_date(alt_dt)
            if alt_df is not None and expiry_dt in identify_monthly_expiry_dates_in_bhavcopy(alt_df):
                print(
                    f"  [{month_str}] INFO — monthly expiry {expiry_dt} not in entry-day "
                    f"bhavcopy; using {alt_dt.date()} for entry prices."
                )
                opt_entry  = alt_df
                entry_ts   = alt_dt
                spot_ts_entry_new = pd.Timestamp(alt_dt)
                if spot_ts_entry_new in spot_df.index:
                    spot_entry = float(spot_df.loc[spot_ts_entry_new, "close"])
                found = True
                break
        if not found:
            print(
                f"  [{month_str}] SKIP — monthly expiry {expiry_dt} not found in "
                f"entry-week bhavcopy data."
            )
            return None

    # ---- Select puts at each moneyness target ----
    put_data: dict[str, dict | None] = {}
    for label, k_target in K_TARGETS.items():
        result = select_strike_for_k(
            opt_entry, spot_entry, k_target, "PE", expiry_dt, label
        )
        put_data[label] = result
        if result:
            print(
                f"    [{month_str}] {label} PE: strike={result['strike']:.0f} "
                f"k_actual={result['strike']/spot_entry:.4f} "
                f"entry_settle={result['settle_price']:.2f}"
            )
        else:
            print(f"    [{month_str}] {label} PE: NOT FOUND (no strike within {K_TOLERANCE*100:.0f}% of target)")

    # ---- Select ATM call for straddle/CN strategies ----
    call_atm_raw = select_strike_for_k(
        opt_entry, spot_entry, 1.00, "CE", expiry_dt, "k100CE"
    )
    if call_atm_raw:
        print(
            f"    [{month_str}] k100 CE: strike={call_atm_raw['strike']:.0f} "
            f"entry_settle={call_atm_raw['settle_price']:.2f}"
        )
    else:
        print(f"    [{month_str}] k100 CE: NOT FOUND")

    # ---- Expiry settles for puts ----
    for label in K_TARGETS:
        if put_data[label] is None:
            continue
        strike = put_data[label]["strike"]
        expiry_settle = get_expiry_settle(
            expiry_dt, strike, "PE", spot_expiry, opt_expiry, label
        )
        put_data[label]["expiry_settle"] = expiry_settle

    # ---- Expiry settle for ATM call ----
    if call_atm_raw is not None:
        call_expiry_settle = get_expiry_settle(
            expiry_dt, call_atm_raw["strike"], "CE", spot_expiry, opt_expiry, "k100CE"
        )
        call_atm = {
            "entry":        call_atm_raw["settle_price"],
            "expiry_settle": call_expiry_settle,
        }
    else:
        call_atm = None

    # ---- Per-put HTM returns ----
    def _put_return(label: str) -> float:
        d = put_data.get(label)
        if d is None:
            return float("nan")
        entry_p  = d.get("settle_price", float("nan"))
        expiry_p = d.get("expiry_settle", float("nan"))
        if _is_nan(entry_p) or _is_nan(expiry_p):
            return float("nan")
        if entry_p <= 0:
            return float("nan")
        return expiry_p / entry_p - 1.0

    # ---- Strategy returns ----
    strategy = calculate_strategy_returns(put_data, call_atm)

    # ---- Assemble panel row ----
    def _sf(label, key):
        d = put_data.get(label)
        if d is None:
            return float("nan")
        return d.get(key, float("nan"))

    row = {
        "month":               month_str,
        "entry_date":          entry_dt.strftime("%Y-%m-%d"),
        "expiry_date":         expiry_dt.strftime("%Y-%m-%d"),
        "spot_at_entry":       round(spot_entry,  2),
        "spot_at_expiry":      round(spot_expiry, 2),
        # k=0.94
        "k094_strike":         _sf("k094", "strike"),
        "k094_entry_price":    _sf("k094", "settle_price"),
        "k094_expiry_price":   _sf("k094", "expiry_settle"),
        "k094_return":         _put_return("k094"),
        # k=0.96
        "k096_strike":         _sf("k096", "strike"),
        "k096_entry_price":    _sf("k096", "settle_price"),
        "k096_expiry_price":   _sf("k096", "expiry_settle"),
        "k096_return":         _put_return("k096"),
        # k=0.98
        "k098_strike":         _sf("k098", "strike"),
        "k098_entry_price":    _sf("k098", "settle_price"),
        "k098_expiry_price":   _sf("k098", "expiry_settle"),
        "k098_return":         _put_return("k098"),
        # k=1.00
        "k100_strike":         _sf("k100", "strike"),
        "k100_entry_price":    _sf("k100", "settle_price"),
        "k100_expiry_price":   _sf("k100", "expiry_settle"),
        "k100_return":         _put_return("k100"),
        # Strategies
        **strategy,
    }

    return row


# ---------------------------------------------------------------------------
# load_existing_panel
# ---------------------------------------------------------------------------

def load_existing_panel() -> set[str]:
    """
    Returns the set of 'month' values already present in the output CSV,
    so they can be skipped (idempotency / resume support).
    """
    if not os.path.isfile(OUTPUT_CSV_PATH):
        return set()

    try:
        df = pd.read_csv(OUTPUT_CSV_PATH, usecols=["month"])
        months = set(df["month"].dropna().astype(str).tolist())
        print(
            f"Existing panel CSV found: {len(months)} months already processed."
        )
        return months
    except Exception as exc:
        print(
            f"WARNING: Could not read existing panel CSV ({exc}). "
            "Starting fresh."
        )
        return set()


# ---------------------------------------------------------------------------
# upsert_panel_to_mongo
# ---------------------------------------------------------------------------

def upsert_panel_to_mongo(rows: list[dict], collection) -> int:
    """
    Upserts panel rows keyed on `month`.  Running twice does not duplicate.
    Returns count of documents written.
    """
    if not rows:
        return 0

    from pymongo import UpdateOne

    operations = []
    for row in rows:
        filter_doc  = {"month": row["month"]}
        # Convert NaN floats to None for Mongo compatibility.
        update_doc  = {
            k: (None if (isinstance(v, float) and v != v) else v)
            for k, v in row.items()
        }
        operations.append(
            UpdateOne(filter_doc, {"$set": update_doc}, upsert=True)
        )

    result   = collection.bulk_write(operations, ordered=False)
    inserted = result.upserted_count
    modified = result.modified_count
    print(f"  Mongo upsert (panel): {inserted} new, {modified} updated.")
    return inserted + modified


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    """
    Orchestrates the monthly options panel construction pipeline.

    Steps:
      1. setup_environment() — confirm folders and input files.
      2. load_spot_from_file() — Nifty 50 daily close data.
      3. build_monthly_expiry_schedule() — list of monthly cycles.
      4. load_existing_panel() — skip already-processed months.
      5. For each unprocessed month:
           a. fetch_panel_row() — select strikes, compute prices & returns.
           b. Append to growing list of rows.
           c. sleep(1) between months (two CSV reads per month — no API calls,
              but keeps I/O paced and mirrors reference script conventions).
      6. Write final CSV (append to existing, or write fresh).
      7. Upsert to MongoDB.
      8. Print summary.
    """
    print("=" * 65)
    print("  Nifty 50 Monthly Options Panel Builder — BCJ Methodology")
    print("  Sumin Pillai (Independent Researcher) — Paper P1")
    print("=" * 65)

    n_option_files = setup_environment()
    if n_option_files == 0:
        print(
            "\nWARNING: No bhavcopy CSV files found in data/nifty_options/. "
            "The panel cannot be built until bhavcopy_downloader.py has run.\n"
            "Continuing anyway to validate structure and spot data."
        )

    # ---- Spot data ----
    spot_df = load_spot_from_file()
    if spot_df is None:
        print(
            "\nERROR: Cannot build panel without spot data. "
            "Run nifty_spot_downloader.py first."
        )
        raise SystemExit(1)

    # ---- Date range ----
    start_dt = datetime.strptime(START_DATE, "%Y-%m-%d")
    end_dt   = datetime.strptime(END_DATE,   "%Y-%m-%d")
    print(f"Panel date range: {START_DATE} to {END_DATE}")

    # ---- Build monthly cycle schedule ----
    schedule = build_monthly_expiry_schedule(start_dt, end_dt, spot_df)
    if not schedule:
        print("ERROR: No monthly expiry cycles found in the requested date range.")
        raise SystemExit(1)

    # ---- Connect to MongoDB (optional) ----
    collection = get_mongo_collection()

    # ---- Load already-processed months (resume support) ----
    already_done = load_existing_panel()

    # ---- Process each monthly cycle ----
    new_rows   = []
    skipped    = 0
    processed  = 0
    missing    = 0
    anomalies  = []

    for cycle in schedule:
        month_str = cycle["month"]

        if month_str in already_done:
            print(f"  [{month_str}] SKIP — already in panel CSV.")
            skipped += 1
            continue

        print(f"\n----- Processing month: {month_str} "
              f"(entry: {cycle['entry_date']}  expiry: {cycle['expiry_date']}) -----")

        row = fetch_panel_row(cycle, spot_df)

        if row is None:
            print(f"  [{month_str}] SKIP — insufficient data (see above).")
            missing += 1
            # Rate-limit pacing.
            time.sleep(1)
            continue

        # Anomaly checks.
        for label in K_TARGETS:
            ret_col = f"{label}_return"
            ret_val = row.get(ret_col)
            if ret_val is not None and not _is_nan(ret_val):
                if ret_val < -1.0:
                    anomalies.append(
                        f"{month_str} {ret_col}={ret_val:.4f}: return below -100%"
                    )
                elif ret_val > 50.0:
                    anomalies.append(
                        f"{month_str} {ret_col}={ret_val:.4f}: implausibly large return (>5000%)"
                    )

        entry_px = row.get("k100_entry_price")
        if not _is_nan(entry_px) and entry_px is not None and entry_px == 0:
            anomalies.append(f"{month_str}: ATM put entry price is zero")

        new_rows.append(row)
        processed += 1

        print(
            f"  [{month_str}] DONE — "
            f"spot_entry={row['spot_at_entry']:.1f}  "
            f"spot_expiry={row['spot_at_expiry']:.1f}  "
            f"straddle_return={row.get('straddle_return', float('nan')):.4f}"
            if not _is_nan(row.get("straddle_return", float("nan")))
            else f"  [{month_str}] DONE — spot_entry={row['spot_at_entry']:.1f}  "
                 f"straddle_return=NaN (insufficient data)"
        )

        # Rate-limit pacing (mirrors alltickersdatadownloader.py convention).
        time.sleep(1)

    # ---- Write CSV ----
    if new_rows:
        new_df = pd.DataFrame(new_rows)

        # Column order matches the spec exactly.
        ordered_cols = [
            "month", "entry_date", "expiry_date", "spot_at_entry", "spot_at_expiry",
            "k094_strike", "k094_entry_price", "k094_expiry_price", "k094_return",
            "k096_strike", "k096_entry_price", "k096_expiry_price", "k096_return",
            "k098_strike", "k098_entry_price", "k098_expiry_price", "k098_return",
            "k100_strike", "k100_entry_price", "k100_expiry_price", "k100_return",
            "straddle_entry_premium", "straddle_expiry_payoff", "straddle_return",
            "cn_entry_premium", "cn_expiry_payoff", "cn_return",
            "psp_entry_premium", "psp_expiry_payoff", "psp_return",
        ]
        # Guard against any column being absent (should not happen, but defensive).
        final_cols = [c for c in ordered_cols if c in new_df.columns]
        new_df = new_df[final_cols]

        if os.path.isfile(OUTPUT_CSV_PATH) and already_done:
            # Append mode: append without repeating the header.
            new_df.to_csv(OUTPUT_CSV_PATH, mode="a", index=False, header=False)
            print(f"\n  CSV appended: {OUTPUT_CSV_PATH}  (+{len(new_df)} rows)")
        else:
            new_df.to_csv(OUTPUT_CSV_PATH, mode="w", index=False, header=True)
            print(f"\n  CSV written:  {OUTPUT_CSV_PATH}  ({len(new_df)} rows)")

        # Upsert to MongoDB.
        mongo_rows = 0
        if collection is not None:
            time.sleep(1)
            mongo_rows = upsert_panel_to_mongo(new_rows, collection)
    else:
        print("\n  No new rows to write.")
        mongo_rows = 0

    # ---- Summary ----
    total_rows = len(already_done) + processed
    print("\n" + "=" * 65)
    print("  PANEL BUILD COMPLETE")
    print("=" * 65)
    print(f"  Configured range          : {START_DATE} to {END_DATE}")
    print(f"  Monthly cycles in range   : {len(schedule)}")
    print(f"  Already processed (skip)  : {skipped}")
    print(f"  Processed this run        : {processed}")
    print(f"  Skipped (missing data)    : {missing}")
    print(f"  Total panel rows on disk  : {total_rows}")
    print(f"  Output CSV                : {OUTPUT_CSV_PATH}")
    print(f"  MongoDB collection        : {MONGO_DATABASE_NAME}.{MONGO_COLLECTION_NAME}")
    print(f"  Mongo rows written        : {mongo_rows:,}")
    if anomalies:
        print(f"\n  ANOMALIES ({len(anomalies)}):")
        for a in anomalies:
            print(f"    - {a}")
    else:
        print("\n  No anomalies detected.")
    print("=" * 65)
    print(
        "\nNext step: run mcmc_estimation.py to fit BS / SV / SVJ parameter "
        "models to the Nifty 50 log-return series (nifty_spot.csv), then "
        "option_simulation.py to generate the 25k-path simulated option returns "
        "needed for the BCJ hypothesis tests in Paper P1."
    )


if __name__ == "__main__":
    main()
