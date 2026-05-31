import sqlite3
import requests
import pandas as pd
import numpy as np
from pathlib import Path

DB_PATH  = "smard_cache.db"
BASE_URL = "https://www.smard.de/app/chart_data"
HEADERS  = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept":  "application/json",
    "Referer": "https://www.smard.de/",
}


# DB helpers

def get_connection():
    """Get SQLite connection."""
    return sqlite3.connect(DB_PATH)


def init_db():
    """Create tables and indexes if they don't exist."""
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS smard_data (
                timestamp   TEXT NOT NULL,
                feature     TEXT NOT NULL,
                value       REAL,
                PRIMARY KEY (timestamp, feature)
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_feature_ts
            ON smard_data (feature, timestamp)
        """)
    print(f"DB initialized → {DB_PATH}")


def get_last_cached_timestamp(feature: str) -> pd.Timestamp | None:
    """Return the latest timestamp we have cached for a feature."""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT MAX(timestamp) FROM smard_data WHERE feature = ?",
            (feature,)
        ).fetchone()
    val = row[0] if row else None
    if val:
        return pd.Timestamp(val, tz="Europe/Berlin")
    return None


def save_series_to_db(name: str, s: pd.Series):
    """Save a Series to the DB, ignoring rows that already exist."""
    if s.empty:
        return
    rows = [
        (ts.isoformat(), name, float(val) if pd.notna(val) else None)
        for ts, val in s.items()
    ]
    with get_connection() as conn:
        conn.executemany(
            "INSERT OR IGNORE INTO smard_data (timestamp, feature, value) "
            "VALUES (?,?,?)",
            rows
        )


def load_series_from_db(name: str) -> pd.Series:
    """Load a full Series for a feature from the DB."""
    with get_connection() as conn:
        df = pd.read_sql(
            "SELECT timestamp, value FROM smard_data "
            "WHERE feature = ? ORDER BY timestamp",
            conn,
            params=(name,),
        )
    if df.empty:
        return pd.Series(dtype=float, name=name)
    df["timestamp"] = (
        pd.to_datetime(df["timestamp"], utc=True)
        .dt.tz_convert("Europe/Berlin")
    )
    return df.set_index("timestamp")["value"].rename(name)


def get_db_stats():
    """Print summary of what's in the DB."""
    try:
        with get_connection() as conn:
            rows = conn.execute("""
                SELECT feature, COUNT(*) as n_rows,
                       MIN(timestamp) as first_ts,
                       MAX(timestamp) as last_ts
                FROM smard_data
                GROUP BY feature
                ORDER BY feature
            """).fetchall()
    except Exception:
        print(f"\nDB not initialized yet: {DB_PATH}")
        return

    if not rows:
        print(f"\nDB is empty: {DB_PATH}")
        return

    size_kb = Path(DB_PATH).stat().st_size / 1024
    print(f"\nDB: {DB_PATH}  ({size_kb:.0f} KB)")
    print(f"  {'Feature':<33} {'Rows':>6}  {'From':<12}  {'To':<12}")
    print("  " + "-" * 66)
    for feat, n, first, last in rows:
        print(f"  {feat:<33} {n:>6}  {first[:10]}  {last[:10]}")

# SMARD availability check (cheap — only fetches the index)

def get_latest_smard_timestamp(fid: int, region: str) -> pd.Timestamp | None:
    """
    Check the latest data timestamp available on SMARD.
    Only fetches the bucket index (tiny JSON) — not the actual data.
    Returns the start of the most recent weekly bucket.
    """
    try:
        r = requests.get(
            f"{BASE_URL}/{fid}/{region}/index_hour.json",
            headers=HEADERS, timeout=10
        )
        r.raise_for_status()
        timestamps = r.json().get("timestamps", [])
        if not timestamps:
            return None
        latest_bucket_ms = max(timestamps)
        return pd.Timestamp(latest_bucket_ms, unit="ms", tz="UTC").tz_convert(
            "Europe/Berlin"
        )
    except Exception:
        return None


# Smart fetch — only download what's missing

def fetch_feature_cached(name: str, fid: int, region: str,
                          fetch_fn, n_weeks: int) -> pd.Series:
    """
    Fetch a SMARD feature with timestamp-based caching:

    1. Load existing data from DB
    2. Ask SMARD for the latest available bucket timestamp (cheap)
    3. If our cache covers that timestamp → return cache immediately
    4. Otherwise fetch fresh data, save new rows, return merged result
    """
    cached      = load_series_from_db(name)
    last_cached = get_last_cached_timestamp(name)

    # Ask SMARD: what is the latest bucket you have?
    latest_on_smard = get_latest_smard_timestamp(fid, region)

    if last_cached is not None and latest_on_smard is not None:
        if last_cached >= latest_on_smard:
            # Our cache already covers the latest SMARD bucket — nothing to do
            print(f"  CACHE  {name:<33} "
                  f"(up to date: {last_cached.strftime('%Y-%m-%d %H:%M')})")
            return cached
        else:
            print(f"  UPDATE {name:<33} "
                  f"cache={last_cached.strftime('%m-%d %H:%M')}  "
                  f"SMARD={latest_on_smard.strftime('%m-%d %H:%M')}")
    else:
        print(f"  FETCH  {name:<33} (not in cache)")

    # Download fresh data from SMARD
    try:
        fresh = fetch_fn(name, fid, region)
    except Exception as e:
        print(f"  ERROR  {name}: {e} — returning cached data")
        return cached

    if fresh.empty:
        return cached

    # Save only genuinely new rows
    new_rows = fresh[fresh.index > last_cached] if last_cached is not None else fresh
    if not new_rows.empty:
        save_series_to_db(name, new_rows)
        print(f"           → saved {len(new_rows)} new rows to DB")

    # Return merged result (cache + fresh, deduped)
    if not cached.empty:
        combined = pd.concat([cached, fresh])
        combined = combined[~combined.index.duplicated(keep="last")]
        return combined.sort_index()

    return fresh
