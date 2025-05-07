import argparse
import asyncio
from datetime import datetime, timedelta
import polars as pl
import asyncpg
from strategy_lab.config import EOD_DIR, INTRADAY_DIR, SPLITS_DIR, DB_CONFIG
from strategy_lab.utils.trading_calendar import TradingCalendar
from tqdm.asyncio import tqdm

async def fetch_eod(conn, start_date: str, end_date: str) -> pl.DataFrame:
    start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
    end_date = datetime.strptime(end_date, "%Y-%m-%d").date()
    rows = await conn.fetch("""
        SELECT stk, dt, o, hi, lo, c, v
        FROM eods
        WHERE dt BETWEEN $1 AND $2
    """, start_date, end_date)
    if not rows:
        print("No EOD data found.")
        return pl.DataFrame([])
    records = [dict(row) for row in rows]
    df = pl.DataFrame(records)
    return df.rename({"stk": "ticker", "dt": "date", "o": "open", "hi": "high", "lo": "low", "c": "close", "v": "volume"})

async def fetch_intraday(conn, start_date: str, end_date: str) -> pl.DataFrame:
    start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
    end_date = datetime.strptime(end_date, "%Y-%m-%d").date()
    rows = await conn.fetch("""
        SELECT stk, dt, o, hi, lo, c, v
        FROM intraday
        WHERE dt::date BETWEEN $1 AND $2
    """, start_date, end_date)
    if not rows:
        print("No intraday data found.")
        return pl.DataFrame([])
    records = [dict(row) for row in rows]
    df = pl.DataFrame(records)
    return df.rename({"stk": "ticker", "dt": "timestamp", "o": "open", "hi": "high", "lo": "low", "c": "close", "v": "volume"})

async def fetch_splits(conn: asyncpg.Connection, start_date: str, end_date: str) -> pl.DataFrame:
    # Fetch splits data from the database
    # and filter it based on the provided date range.
    query = f"""
    SELECT stk, dt, ratio
    FROM dividends
    WHERE ratio IS NOT NULL AND dt BETWEEN '{start_date}' AND '{end_date}'
    """
    rows = await conn.fetch(query)
    if not rows:
        print("No splits data found for the given date range.")
        return pl.DataFrame([])
    records = [dict(row) for row in rows]
    df = pl.DataFrame(records).rename({"stk": "ticker", "dt": "date"})
    return df.sort(["ticker", "date"])

def save_eod(df: pl.DataFrame) -> None:
    df = df.with_columns([
        (pl.col("open") / 100).alias("open"),
        (pl.col("high") / 100).alias("high"),
        (pl.col("low") / 100).alias("low"),
        (pl.col("close") / 100).alias("close"),
        (pl.col("volume") * 1000).alias("volume"),
    ])
    # Note: using .to_list() here because we're iterating over a single column
    tickers = df.select("ticker").unique().get_column("ticker").to_list()
    for ticker in tickers:
        ticker_df = df.filter(pl.col("ticker") == ticker).drop("ticker")
        filepath = EOD_DIR / f"{ticker}.parquet"
        if filepath.exists():
            existing = pl.read_parquet(filepath)
            # Note: using vertical_relaxed to avoid strict schema checks
            combined = pl.concat([existing, ticker_df], how="vertical_relaxed").unique(subset=["date"], keep="last")
        else:
            combined = ticker_df
        combined = combined.sort("date")
        combined.write_parquet(filepath)

def save_intraday(df: pl.DataFrame):
    df = df.with_columns([
        (pl.col("open") / 100).alias("open"),
        (pl.col("high") / 100).alias("high"),
        (pl.col("low") / 100).alias("low"),
        (pl.col("close") / 100).alias("close"),
        pl.col("timestamp").dt.strftime("%Y-%m-%d").alias("date"),
        pl.col("timestamp").dt.strftime("%H:%M:%S").alias("time")
    ])
    # Note: iterating over unique (ticker, date) pairs using iter_rows()
    # avoids nested loops and unnecessary filtering, and is cleaner and more efficient.
    for row in df.select(["ticker", "date"]).unique().iter_rows():
        ticker, dt = row
        date_str = str(dt)
        # Note: we drop both ticker and date to make the parquet file smaller and avoid redundancy.
        # The filename already encodes the date, and the folder encodes the ticker.
        ticker_df = df.filter((pl.col("ticker") == ticker) & (pl.col("date") == dt)).drop(["ticker", "date"])
        ticker_df = ticker_df.sort("time")
        path = INTRADAY_DIR / ticker / f"{date_str}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        ticker_df.write_parquet(path)

def save_splits(df: pl.DataFrame):
    output_file = SPLITS_DIR / "splits.parquet"

    if output_file.exists():
        existing = pl.read_parquet(output_file)
        combined = pl.concat([existing, df], how="vertical_relaxed").unique(subset=["ticker", "date"])
    else:
        combined = df

    combined = combined.sort(["ticker", "date"])
    combined.write_parquet(output_file)
    print(f"Updated {output_file} with {len(df)} new records")


async def update_splits_parquet(start_date: str, end_date: str):
    conn = await asyncpg.connect(**DB_CONFIG)
    try:
        df = await fetch_splits(conn, start_date, end_date)
        if df.height > 0:
            save_splits(df)
    finally:
        await conn.close()

async def update_eod_parquet(start_date: str, end_date: str):
    conn = await asyncpg.connect(**DB_CONFIG)
    try:
        df = await fetch_eod(conn, start_date, end_date)
        if df.height > 0:
            save_eod(df)
    finally:
        await conn.close()

async def update_intraday_parquet(start_date: str, end_date: str):
    conn = await asyncpg.connect(**DB_CONFIG)
    try:
        df = await fetch_intraday(conn, start_date, end_date)
        if df.height > 0:
            save_intraday(df)
    finally:
        await conn.close()

async def run_batch_updates(start_date: str, end_date: str):
    calendar = TradingCalendar()
    dates = calendar.date_range(start_date, end_date)

    # EOD: one batch per month
    current = datetime.strptime(start_date, "%Y-%m-%d").date()
    end_dt = datetime.strptime(end_date, "%Y-%m-%d").date()
    while current <= end_dt:
        month_end = (current.replace(day=1) + timedelta(days=32)).replace(day=1) - timedelta(days=1)
        month_end = min(month_end, end_dt)
        print(f"Updating EOD data from {current} to {month_end}...")
        await update_eod_parquet(str(current), str(month_end))
        current = month_end + timedelta(days=1)

    # Splits: full period
    print(f"Updating splits data from {start_date} to {end_date}...")
    await update_splits_parquet(start_date, end_date)

    # Intraday: one batch per day (parallel)
    print("Updating intraday data in parallel...")

    sem = asyncio.Semaphore(10)  # Limit concurrent intraday tasks to 10

    async def limited_update(day):
        async with sem:
            await update_intraday_parquet(day, day)
            intraday_progress.update(1)
            intraday_progress.set_postfix_str(f"Updated {day}")

    print("Updating intraday data with concurrency limit (10)...")
    intraday_progress = tqdm(total=len(dates), desc="Intraday Updates", unit="day")

    await asyncio.gather(*[
        limited_update(day) for day in dates
    ])
    print("Intraday data update completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch convert EOD and intraday data to Parquet.")
    parser.add_argument("--start-date", type=str, required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, required=True, help="End date (YYYY-MM-DD)")
    args = parser.parse_args()

    asyncio.run(run_batch_updates(args.start_date, args.end_date))
    print(f"Batch update completed from {args.start_date} to {args.end_date}.")
