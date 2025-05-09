import argparse
import asyncio
import asyncpg
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
import polars as pl
from strategy_lab.config import EOD_DIR, INTRADAY_DIR, SPLITS_DIR, DB_CONFIG
from strategy_lab.utils.trading_calendar import TradingCalendar
from strategy_lab.utils.logger import get_logger
from tqdm.asyncio import tqdm

logger = get_logger(__name__)

INTRADAY_CONCURRENCY = 10 # Number of concurrent intraday updates

async def fetch_eod(conn: asyncpg.Connection, start_date: str, end_date: str) -> pl.DataFrame:
    # Fetch EOD data from the database
    start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
    end_date = datetime.strptime(end_date, "%Y-%m-%d").date()
    rows = await conn.fetch("""
        SELECT stk, dt, o, hi, lo, c, v
        FROM eods
        WHERE dt BETWEEN $1 AND $2
    """, start_date, end_date)
    if not rows:
        # If no rows are found, log a warning and return an empty DataFrame
        logger.warning(f"No EOD data found between {start_date} and {end_date}.")
        return pl.DataFrame([])
    records = [dict(row) for row in rows]
    df = pl.DataFrame(records)
    return df.rename({"stk": "ticker", "dt": "date", "o": "open", "hi": "high", "lo": "low", "c": "close", "v": "volume"})

async def fetch_intraday(conn: asyncpg.Connection, start_date: str, end_date: str) -> pl.DataFrame:
    start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
    end_date = datetime.strptime(end_date, "%Y-%m-%d").date()
    rows = await conn.fetch("""
        SELECT stk, dt, o, hi, lo, c, v
        FROM intraday
        WHERE dt::date BETWEEN $1 AND $2
    """, start_date, end_date)
    if not rows:
        logger.warning(f"No intraday data found between {start_date} and {end_date}.")
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
        logger.warning("No splits data found for the given date range.")
        return pl.DataFrame([])
    records = [dict(row) for row in rows]
    df = pl.DataFrame(records).rename({"stk": "ticker", "dt": "date"})
    return df.sort(["ticker", "date"])

def save_eod(df: pl.DataFrame) -> None:
    df = df.with_columns([
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
        pl.col("timestamp").dt.strftime("%Y-%m-%d").alias("date"),
        pl.col("timestamp").dt.strftime("%H:%M:%S").alias("time")
    ])
    # Note: iterating over unique (ticker, date) pairs using iter_rows()
    # avoids nested loops and unnecessary filtering, and is cleaner and more efficient.
    for row in df.select(["ticker", "date"]).unique().iter_rows():
        ticker, dt = row
        date_str = str(dt)
        # Note: we drop ticker, timestamp, and date to make the parquet file smaller and avoid redundancy.
        # The filename already encodes the date, and the folder encodes the ticker.
        ticker_df = df.filter((pl.col("ticker") == ticker) & (pl.col("date") == dt)).drop(["ticker", "timestamp", "date"])
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
    logger.info(f"Updated {output_file} with {len(df)} new records")

@asynccontextmanager
async def get_connection_pool(pool: asyncpg.Pool):
    conn = await pool.acquire()
    try:
        yield conn
    finally:
        await pool.release(conn)
        logger.info("Connection released back to the pool.")

async def update_splits_parquet(pool: asyncpg.Pool, start_date: str, end_date: str):
    async with get_connection_pool(pool) as conn:
        try:
            df = await fetch_splits(conn, start_date, end_date)
            if df.height > 0:
                save_splits(df)
        except Exception as e:
            logger.error(f"Error fetching splits data between {start_date} and {end_date}: {e}")

async def update_eod_parquet(pool: asyncpg.Pool, start_date: str, end_date: str):
    async with get_connection_pool(pool) as conn:
        try:
            df = await fetch_eod(conn, start_date, end_date)
            if df.height > 0:
                save_eod(df)
        except Exception as e:
            logger.error(f"Error fetching EOD data between {start_date} and {end_date}: {e}")

async def update_intraday_parquet(pool: asyncpg.Pool, start_date: str, end_date: str):
    async with get_connection_pool(pool) as conn:
        try:
            df = await fetch_intraday(conn, start_date, end_date)
            if df.height > 0:
                save_intraday(df)
        except Exception as e:
            logger.error(f"Error fetching intraday data between {start_date} and {end_date}: {e}")

async def run_batch_updates(pool: asyncpg.Pool, start_date: str, end_date: str, intraday_only: bool, eod_only: bool):
    calendar = TradingCalendar()
    dates = calendar.date_range(start_date, end_date)
    try:
        # EOD and splits update
        if not intraday_only:
            logger.info(f"Updating EOD and splits data from {start_date} to {end_date}...")
            # EOD: one batch per month
            current = datetime.strptime(start_date, "%Y-%m-%d").date()
            end_dt = datetime.strptime(end_date, "%Y-%m-%d").date()
            while current <= end_dt:
                month_end = (current.replace(day=1) + timedelta(days=32)).replace(day=1) - timedelta(days=1)
                month_end = min(month_end, end_dt)
                logger.info(f"Updating EOD data from {current} to {month_end}...")
                await update_eod_parquet(pool, str(current), str(month_end))
                # Move to the next month
                current = month_end + timedelta(days=1)
                logger.info(f"EOD data update completed for {current} to {month_end}.")

            # Splits: full period
            logger.info(f"Updating splits data from {start_date} to {end_date}...")
            await update_splits_parquet(pool, start_date, end_date)
            logger.info(f"Splits data update completed for {start_date} to {end_date}.")

        # Intraday update
        if not eod_only:
            # Intraday: one batch per day (INTRADAY_CONCURENCY = 10 batches in parallel)
            logger.info(f"Updating intraday data with concurrency limit ({INTRADAY_CONCURRENCY})...")
            intraday_progress = tqdm(total=len(dates), desc="Intraday Updates", unit="day")

            sem = asyncio.Semaphore(INTRADAY_CONCURRENCY)  # Limit concurrent intraday tasks to INTRADAY_CONCURENCY

            async def limited_update(day):
                async with sem:
                    # Update intraday data for the specific day
                    logger.info(f"Updating intraday data for {day}...")
                    await update_intraday_parquet(pool, day, day)
                    # Update progress bar
                    intraday_progress.update(1)
                    intraday_progress.set_postfix_str(f"Updated {day}")

            await asyncio.gather(*[
                limited_update(day) for day in dates
            ])
            logger.info("Intraday data update completed.")
            intraday_progress.close()
    except Exception as e:
        logger.error(f"Failed to connect to the database: {e}")
    finally:
        await pool.close()
        logger.info("Database connection pool closed.")
        logger.info("Batch update completed.")

async def main():
    parser = argparse.ArgumentParser(description="Batch convert EOD and intraday data to Parquet.")
    parser.add_argument("--start-date", type=str, required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--intraday-only", action="store_true", help="Only update intraday data")
    parser.add_argument("--eod-only", action="store_true", help="Only update EOD and splits data")
    args = parser.parse_args()

    pool: asyncpg.Pool = await asyncpg.create_pool(**DB_CONFIG, min_size=1, max_size=INTRADAY_CONCURRENCY)
    await run_batch_updates(pool, args.start_date, args.end_date, args.intraday_only, args.eod_only)
    logger.info("Database connection pool closed.")
    logger.info(f"Batch update completed from {args.start_date} to {args.end_date}.")

if __name__ == "__main__":
    asyncio.run(main())
