"""
bhavcopy_downloader.py
======================
NSE F&O Bhavcopy Downloader — Paper P1.

Downloads daily F&O bhavcopy ZIP files from the NSE archives, extracts the
embedded CSV, splits records into two instrument types, and writes each day's
records to:

  OPTIONS (INSTRUMENT == "OPTIDX", SYMBOL == "NIFTY"):
    - MongoDB  : database `optionsdatabase`, collection `nifty_options`
    - CSV      : data/nifty_options/YYYY-MM-DD.csv   (one file per trading day)

  FUTURES (INSTRUMENT == "FUTIDX", SYMBOL == "NIFTY"):
    - MongoDB  : database `optionsdatabase`, collection `nifty_futures`
    - CSV      : data/nifty_futures/YYYY-MM-DD.csv   (one file per trading day)

Only ONE HTTP request is made per trading day — the ZIP is downloaded once and
split in memory.  A day is considered "done" only when BOTH the options CSV and
the futures CSV already exist on disk.

Idempotent: options are keyed on (timestamp, expiry_date, strike_price,
option_type); futures are keyed on (timestamp, expiry_date).  Running the
script twice does not duplicate rows.

Resumable: date iteration checks whether both CSVs for that date already exist
before hitting the network, so a mid-run crash restarts from where it left off.

Dependencies (add to requriments.txt if missing):
    requests, pandas, python-dotenv, pymongo

Usage:
    python bhavcopy_downloader.py

Author: Sumin Pillai (Independent Researcher)
"""

import os
import io
import time
import zipfile
from datetime import datetime, timedelta

import pandas as pd
import requests
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# CONFIGURATION — edit these two constants to change the download window.
# ---------------------------------------------------------------------------
START_DATE = "2015-01-01"
END_DATE   = "2025-04-30"
# ---------------------------------------------------------------------------

BASE_DIR = "C:\\Users\\sumin\\definedge_downloader"

OPTIONS_FOLDER = os.path.join(BASE_DIR, "data", "nifty_options")
FUTURES_FOLDER = os.path.join(BASE_DIR, "data", "nifty_futures")

MONGO_CONNECTION_STRING      = "mongodb://localhost:27017/"
MONGO_DATABASE_NAME          = "optionsdatabase"
OPTIONS_COLLECTION_NAME      = "nifty_options"
FUTURES_COLLECTION_NAME      = "nifty_futures"

# NSE archive URL pattern for F&O bhavcopy.
# Example: https://nsearchives.nseindia.com/content/historical/DERIVATIVES/2024/JAN/fo01JAN2024bhav.csv.zip
NSE_URL_TEMPLATE = (
    "https://nsearchives.nseindia.com/content/historical/DERIVATIVES"
    "/{year}/{month_upper}/fo{dd}{month_upper}{yyyy}bhav.csv.zip"
)

# NSE blocks automated clients that omit a realistic User-Agent.
HTTP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Referer": "https://www.nseindia.com/",
}

# Raw CSV columns present in OPTIDX rows (all others are dropped for options).
OPTIONS_COLUMN_MAP = {
    "INSTRUMENT": "instrument",
    "SYMBOL":     "symbol",
    "EXPIRY_DT":  "expiry_date",
    "STRIKE_PR":  "strike_price",
    "OPTION_TYP": "option_type",
    "OPEN":       "open",
    "HIGH":       "high",
    "LOW":        "low",
    "CLOSE":      "close",
    "SETTLE_PR":  "settle_price",
    "CONTRACTS":  "contracts",
    "VAL_INLAKH": "value_in_lakh",
    "OPEN_INT":   "open_interest",
    "CHG_IN_OI":  "change_in_oi",
    "TIMESTAMP":  "timestamp",
}

# Raw CSV columns present in FUTIDX rows.
# STRIKE_PR and OPTION_TYP are absent in futures rows — they are simply not
# included in this map and will not appear in the futures DataFrame.
FUTURES_COLUMN_MAP = {
    "INSTRUMENT": "instrument",
    "SYMBOL":     "symbol",
    "EXPIRY_DT":  "expiry_date",
    "OPEN":       "open",
    "HIGH":       "high",
    "LOW":        "low",
    "CLOSE":      "close",
    "SETTLE_PR":  "settle_price",
    "CONTRACTS":  "contracts",
    "VAL_INLAKH": "value_in_lakh",
    "OPEN_INT":   "open_interest",
    "CHG_IN_OI":  "change_in_oi",
    "TIMESTAMP":  "timestamp",
}


# ---------------------------------------------------------------------------
# setup_environment
# ---------------------------------------------------------------------------

def setup_environment():
    """Creates output folders if they do not already exist."""
    for folder in (OPTIONS_FOLDER, FUTURES_FOLDER):
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f"Created output folder: '{folder}'")
        else:
            print(f"Output folder exists: '{folder}'")


# ---------------------------------------------------------------------------
# load_date_range_from_config
# ---------------------------------------------------------------------------

def load_date_range_from_config():
    """
    Returns (start_dt, end_dt) as datetime objects from the module-level
    START_DATE / END_DATE constants.  Dates are never hardcoded inside
    function bodies — they come from the constants block at the top.
    """
    start_dt = datetime.strptime(START_DATE, "%Y-%m-%d")
    end_dt   = datetime.strptime(END_DATE,   "%Y-%m-%d")
    print(f"Date range: {START_DATE} to {END_DATE}")
    return start_dt, end_dt


# ---------------------------------------------------------------------------
# get_mongo_collection
# ---------------------------------------------------------------------------

def get_mongo_collection(collection_name: str):
    """
    Connects to MongoDB and returns the named collection object from
    `optionsdatabase`.  Returns None if pymongo is not installed or the
    server is unreachable — CSV output continues regardless.
    """
    load_dotenv()
    try:
        from pymongo import MongoClient
    except ImportError:
        print(
            "WARNING: pymongo is not installed. "
            "Install it with:  pip install pymongo\n"
            "CSV output will still be written; MongoDB upsert will be skipped."
        )
        return None

    try:
        client = MongoClient(MONGO_CONNECTION_STRING, serverSelectionTimeoutMS=5000)
        client.admin.command("ismaster")
        db         = client[MONGO_DATABASE_NAME]
        collection = db[collection_name]
        print(
            f"Connected to MongoDB — database '{MONGO_DATABASE_NAME}', "
            f"collection '{collection_name}'."
        )
        return collection
    except Exception as exc:
        print(
            f"WARNING: Could not connect to MongoDB ({exc}). "
            "CSV output will still be written; MongoDB upsert will be skipped."
        )
        return None


# ---------------------------------------------------------------------------
# fetch_bhavcopy
# ---------------------------------------------------------------------------

def fetch_bhavcopy(trade_date: datetime) -> bytes | None:
    """
    Downloads the F&O bhavcopy ZIP for a single trading date from NSE archives.

    Returns the raw ZIP bytes on success, or None on HTTP 404 (weekend /
    holiday — expected and logged at INFO level).  Any other HTTP error is
    logged as ERROR and causes a loud failure via SystemExit.
    """
    dd          = trade_date.strftime("%d")
    month_upper = trade_date.strftime("%b").upper()   # e.g. JAN
    yyyy        = trade_date.strftime("%Y")

    url = NSE_URL_TEMPLATE.format(
        year=yyyy,
        month_upper=month_upper,
        dd=dd,
        yyyy=yyyy,
    )

    print(f"  - GET {url}")

    for attempt in range(1, 4):
        try:
            response = requests.get(url, headers=HTTP_HEADERS, timeout=30)
            break
        except requests.exceptions.RequestException as exc:
            print(f"    WARN: Network error (attempt {attempt}/3): {exc}")
            if attempt == 3:
                print(f"    ERROR: All 3 attempts failed for {url}. Skipping date.")
                return None
            time.sleep(5 * attempt)

    if response.status_code == 200:
        print(f"    OK — {len(response.content):,} bytes received.")
        return response.content

    if response.status_code == 404:
        print(f"    SKIP — 404 (likely weekend or holiday): {url}")
        return None

    # Any other HTTP error is unexpected — fail loudly.
    print(
        f"    ERROR: HTTP {response.status_code} for {url}. "
        "Aborting to avoid silent data gaps."
    )
    raise SystemExit(1)


# ---------------------------------------------------------------------------
# parse_bhavcopy_zip  (returns raw full DataFrame before instrument split)
# ---------------------------------------------------------------------------

def parse_bhavcopy_zip(zip_bytes: bytes, trade_date: datetime) -> pd.DataFrame | None:
    """
    Extracts the CSV from the ZIP bytes and returns the full raw DataFrame
    for NIFTY rows (both OPTIDX and FUTIDX).  Column stripping and type
    coercions happen downstream in normalise_columns().

    Returns None if the ZIP is empty or contains no NIFTY rows at all.
    """
    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
            if not csv_names:
                print(f"    ERROR: No CSV found inside ZIP for {trade_date.date()}.")
                return None
            csv_name = csv_names[0]
            print(f"    Extracting '{csv_name}' from ZIP.")
            with zf.open(csv_name) as csv_file:
                df = pd.read_csv(csv_file)
    except zipfile.BadZipFile as exc:
        print(f"    ERROR: Bad ZIP file for {trade_date.date()}: {exc}")
        return None

    # Strip whitespace from column names (NSE sometimes has trailing spaces).
    df.columns = [c.strip() for c in df.columns]

    # Check that expected columns are present before filtering.
    required_cols = {"INSTRUMENT", "SYMBOL"}
    missing = required_cols - set(df.columns)
    if missing:
        print(f"    ERROR: Expected columns missing from bhavcopy: {missing}")
        raise SystemExit(1)

    # Strip whitespace from the key filter columns.
    df["INSTRUMENT"] = df["INSTRUMENT"].str.strip()
    df["SYMBOL"]     = df["SYMBOL"].str.strip()

    # Keep only NIFTY rows (both instrument types).
    nifty_mask = df["SYMBOL"] == "NIFTY"
    df = df[nifty_mask].copy()

    if df.empty:
        print(f"    WARNING: No NIFTY rows at all found for {trade_date.date()}.")
        return None

    opt_count = (df["INSTRUMENT"] == "OPTIDX").sum()
    fut_count = (df["INSTRUMENT"] == "FUTIDX").sum()
    print(f"    {opt_count} NIFTY option rows, {fut_count} NIFTY futures rows found.")
    return df


# ---------------------------------------------------------------------------
# normalise_columns
# ---------------------------------------------------------------------------

def normalise_columns(df: pd.DataFrame, column_map: dict) -> pd.DataFrame:
    """
    Renames columns to the canonical schema using `column_map`, coerces data
    types, and drops any columns not in the target schema.

    Works for both options (OPTIONS_COLUMN_MAP) and futures (FUTURES_COLUMN_MAP).
    Futures rows naturally lack STRIKE_PR and OPTION_TYP; those simply don't
    appear in the futures DataFrame — no NaN-filling needed.

    CSV column names match Mongo field names exactly (per the column map).
    """
    # Keep only recognised raw columns present in this particular DataFrame.
    keep = [c for c in column_map if c in df.columns]
    df   = df[keep].rename(columns=column_map)

    # --- Type coercions ---
    for date_col in ("expiry_date", "timestamp"):
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col], format="%d-%b-%Y", errors="coerce")
            bad = df[date_col].isna().sum()
            if bad:
                print(f"    WARNING: {bad} unparseable values in column '{date_col}' — set to NaT.")

    for float_col in ("strike_price", "open", "high", "low", "close", "settle_price", "value_in_lakh"):
        if float_col in df.columns:
            df[float_col] = pd.to_numeric(df[float_col], errors="coerce")

    for int_col in ("contracts", "open_interest", "change_in_oi"):
        if int_col in df.columns:
            df[int_col] = pd.to_numeric(df[int_col], errors="coerce").astype("Int64")

    for str_col in ("instrument", "symbol", "option_type"):
        if str_col in df.columns:
            df[str_col] = df[str_col].str.strip()

    return df


# ---------------------------------------------------------------------------
# upsert_options_to_mongo
# ---------------------------------------------------------------------------

def upsert_options_to_mongo(df: pd.DataFrame, collection) -> int:
    """
    Upserts option records.  Compound key: (timestamp, expiry_date,
    strike_price, option_type) — identical to the original behaviour.

    Returns count of rows upserted/inserted.
    """
    from pymongo import UpdateOne

    operations = []
    for _, row in df.iterrows():
        filter_doc = {
            "timestamp":    row["timestamp"].to_pydatetime() if pd.notna(row.get("timestamp")) else None,
            "expiry_date":  row["expiry_date"].to_pydatetime() if pd.notna(row.get("expiry_date")) else None,
            "strike_price": float(row["strike_price"]) if pd.notna(row.get("strike_price")) else None,
            "option_type":  row["option_type"],
        }
        update_doc = _build_update_doc(df, row)
        operations.append(UpdateOne(filter_doc, {"$set": update_doc}, upsert=True))

    if not operations:
        return 0

    result   = collection.bulk_write(operations, ordered=False)
    inserted = result.upserted_count
    modified = result.modified_count
    print(f"    Mongo upsert (options): {inserted} new, {modified} updated.")
    return inserted + modified


# ---------------------------------------------------------------------------
# upsert_futures_to_mongo
# ---------------------------------------------------------------------------

def upsert_futures_to_mongo(df: pd.DataFrame, collection) -> int:
    """
    Upserts futures records.  Compound key: (timestamp, expiry_date).
    strike_price and option_type are intentionally absent from the filter
    because FUTIDX rows carry neither field.

    Returns count of rows upserted/inserted.
    """
    from pymongo import UpdateOne

    operations = []
    for _, row in df.iterrows():
        filter_doc = {
            "timestamp":   row["timestamp"].to_pydatetime() if pd.notna(row.get("timestamp")) else None,
            "expiry_date": row["expiry_date"].to_pydatetime() if pd.notna(row.get("expiry_date")) else None,
        }
        update_doc = _build_update_doc(df, row)
        operations.append(UpdateOne(filter_doc, {"$set": update_doc}, upsert=True))

    if not operations:
        return 0

    result   = collection.bulk_write(operations, ordered=False)
    inserted = result.upserted_count
    modified = result.modified_count
    print(f"    Mongo upsert (futures): {inserted} new, {modified} updated.")
    return inserted + modified


# ---------------------------------------------------------------------------
# _build_update_doc  (shared helper)
# ---------------------------------------------------------------------------

def _build_update_doc(df: pd.DataFrame, row) -> dict:
    """
    Converts a DataFrame row to a Mongo-safe update document, coercing numpy
    scalars to Python native types and Timestamps to Python datetimes.
    """
    update_doc = {}
    for col in df.columns:
        val = row[col]
        if pd.isna(val) if not hasattr(val, "__iter__") else False:
            update_doc[col] = None
        elif hasattr(val, "to_pydatetime"):
            update_doc[col] = val.to_pydatetime()
        elif hasattr(val, "item"):          # numpy scalars
            update_doc[col] = val.item()
        else:
            update_doc[col] = val
    return update_doc


# ---------------------------------------------------------------------------
# write_csv
# ---------------------------------------------------------------------------

def write_csv(df: pd.DataFrame, trade_date: datetime, folder: str) -> str:
    """
    Writes the day's DataFrame to a CSV file under `folder`.
    File name format: YYYY-MM-DD.csv

    Returns the path written.
    """
    csv_path = os.path.join(folder, f"{trade_date.strftime('%Y-%m-%d')}.csv")
    df.to_csv(csv_path, index=False)
    print(f"    CSV written: {csv_path}  ({len(df)} rows)")
    return csv_path


# ---------------------------------------------------------------------------
# both_csvs_exist  (resume helper — replaces the old csv_already_exists)
# ---------------------------------------------------------------------------

def both_csvs_exist(trade_date: datetime) -> bool:
    """
    Returns True only when BOTH the options CSV and the futures CSV for this
    date already exist on disk.  If either is missing the day is re-processed.
    """
    opt_path = os.path.join(OPTIONS_FOLDER, f"{trade_date.strftime('%Y-%m-%d')}.csv")
    fut_path = os.path.join(FUTURES_FOLDER, f"{trade_date.strftime('%Y-%m-%d')}.csv")
    return os.path.isfile(opt_path) and os.path.isfile(fut_path)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    """
    Orchestrates the full bhavcopy download pipeline for the configured
    date range.  One HTTP request per trading day; the ZIP is split in memory
    into options and futures before writing.

    Steps per trading day:
      1. Skip if BOTH CSVs already on disk (resume support).
      2. Download ZIP from NSE archives (single request).
      3. Parse full NIFTY rows from CSV inside ZIP.
      4. Split into options (OPTIDX) and futures (FUTIDX) DataFrames.
      5. Normalise column names and types for each instrument type.
      6. Write CSVs to data/nifty_options/ and data/nifty_futures/.
      7. Upsert to MongoDB optionsdatabase.nifty_options and .nifty_futures.
      8. Sleep 1 second before the next request (rate limiting).
    """
    print("=" * 65)
    print("  NSE F&O Bhavcopy Downloader — P1")
    print("  (Options + Futures dual output)")
    print("=" * 65)

    setup_environment()
    start_dt, end_dt    = load_date_range_from_config()
    options_col         = get_mongo_collection(OPTIONS_COLLECTION_NAME)
    futures_col         = get_mongo_collection(FUTURES_COLLECTION_NAME)

    total_opt_rows  = 0
    total_fut_rows  = 0
    days_downloaded = 0
    days_skipped    = 0
    days_holiday    = 0
    anomalies       = []

    current_dt = start_dt
    while current_dt <= end_dt:
        date_str = current_dt.strftime("%Y-%m-%d")
        print(f"\n----- {date_str} -----")

        # --- Resume: skip only when BOTH CSVs are present ---
        if both_csvs_exist(current_dt):
            print(f"  SKIP — both CSVs already exist for {date_str}.")
            days_skipped += 1
            current_dt += timedelta(days=1)
            continue

        # --- Download (single request) ---
        zip_bytes = fetch_bhavcopy(current_dt)

        if zip_bytes is None:
            # Weekend or holiday — not an error.
            days_holiday += 1
            current_dt += timedelta(days=1)
            time.sleep(1)
            continue

        # --- Parse full NIFTY rows ---
        raw_df = parse_bhavcopy_zip(zip_bytes, current_dt)
        if raw_df is None:
            anomalies.append(f"{date_str}: no NIFTY rows in bhavcopy at all")
            current_dt += timedelta(days=1)
            time.sleep(1)
            continue

        # --- Split by INSTRUMENT ---
        opt_raw = raw_df[raw_df["INSTRUMENT"] == "OPTIDX"].copy()
        fut_raw = raw_df[raw_df["INSTRUMENT"] == "FUTIDX"].copy()

        # ---- OPTIONS branch ----
        if opt_raw.empty:
            anomalies.append(f"{date_str}: no OPTIDX NIFTY rows")
        else:
            opt_df = normalise_columns(opt_raw, OPTIONS_COLUMN_MAP)

            zero_vol = (opt_df["contracts"] == 0).sum()
            if zero_vol:
                anomalies.append(f"{date_str}: {zero_vol} option contracts with zero volume")

            write_csv(opt_df, current_dt, OPTIONS_FOLDER)
            total_opt_rows += len(opt_df)

            if options_col is not None:
                upsert_options_to_mongo(opt_df, options_col)

        # ---- FUTURES branch ----
        if fut_raw.empty:
            anomalies.append(f"{date_str}: no FUTIDX NIFTY rows")
        else:
            fut_df = normalise_columns(fut_raw, FUTURES_COLUMN_MAP)

            zero_vol = (fut_df["contracts"] == 0).sum()
            if zero_vol:
                anomalies.append(f"{date_str}: {zero_vol} futures contracts with zero volume")

            write_csv(fut_df, current_dt, FUTURES_FOLDER)
            total_fut_rows += len(fut_df)

            if futures_col is not None:
                upsert_futures_to_mongo(fut_df, futures_col)

        days_downloaded += 1

        # Rate-limit: 1 second between HTTP requests.
        time.sleep(1)
        current_dt += timedelta(days=1)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("  DOWNLOAD COMPLETE")
    print("=" * 65)
    print(f"  Date range                   : {START_DATE} to {END_DATE}")
    print(f"  Trading days downloaded      : {days_downloaded}")
    print(f"  Holidays / weekends skipped  : {days_holiday}")
    print(f"  Already-on-disk skipped      : {days_skipped}")
    print(f"  Total option rows (CSV)      : {total_opt_rows:,}")
    print(f"  Total futures rows (CSV)     : {total_fut_rows:,}")
    print(f"  Options MongoDB collection   : {MONGO_DATABASE_NAME}.{OPTIONS_COLLECTION_NAME}")
    print(f"  Futures MongoDB collection   : {MONGO_DATABASE_NAME}.{FUTURES_COLLECTION_NAME}")
    print(f"  Options CSV folder           : {OPTIONS_FOLDER}")
    print(f"  Futures CSV folder           : {FUTURES_FOLDER}")
    if anomalies:
        print(f"\n  ANOMALIES ({len(anomalies)}):")
        for a in anomalies:
            print(f"    - {a}")
    else:
        print("\n  No anomalies detected.")
    print("=" * 65)
    print(
        "\nNext step: run nifty_futures_continuous.py to build the front-month "
        "futures series, then nifty_options_panel.py for the monthly HTM panel."
    )


if __name__ == "__main__":
    main()
