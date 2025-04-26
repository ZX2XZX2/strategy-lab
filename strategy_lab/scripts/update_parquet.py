import argparse
import asyncio
import polars as pl
import os
import asyncpg
from typing import List
from pathlib import Path
from strategy_lab.utils.trading_calendar import TradingCalendar

DB_CONFIG = {
    "user": os.getenv("POSTGRES_USER"),
    "password": os.getenv("POSTGRES_PASSWORD"),
    "port": os.getenv("POSTGRES_PORT"),
    "database": os.getenv("POSTGRES_DB"),
    "host": os.getenv("POSTGRES_HOST")
}

PARQUET_DIR = os.path.join(os.getenv("HOME"), "parquet")
if not os.path.exists(PARQUET_DIR):
    os.makedirs(PARQUET_DIR)
EOD_DIR = os.path.join(PARQUET_DIR, "eod")
INTRADAY_DIR = os.path.join(PARQUET_DIR, "intraday")
if not os.path.exists(EOD_DIR):
    os.makedirs(EOD_DIR)
if not os.path.exists(INTRADAY_DIR):
    os.makedirs(INTRADAY_DIR)


EOD_OUTPUT_DIR = Path(EOD_DIR)
INTRADAY_OUTPUT_DIR = Path(INTRADAY_DIR)
CALENDAR_PATH = Path(os.path.join(PARQUET_DIR, "calendar.parquet"))

async def fetch_eod(conn: asyncpg.Connection, start_date: str, end_date: str) -> List[asyncpg.Record]:
    query = f"""
    SELECT stk, dt, o, hi, lo, c, v
    FROM eod_table
    WHERE dt BETWEEN '{start_date}' AND '{end_date}'
    """
    return await conn.fetch(query)

async def fetch_intraday(conn: asyncpg.Connection, start_date: str, end_date: str) -> List[asyncpg.Record]:
    query = f"""
    SELECT stk, dt, o, hi, lo, c, v
    FROM intraday_table
    WHERE dt::date BETWEEN '{start_date}' AND '{end_date}'
    """
    return await conn.fetch(query)

async def save_eod(records: List[asyncpg.Record]) -> None:
    if not records:
        return
    df = pl.DataFrame(records).rename({
        "stk": "ticker", "dt": "date", "o": "open", "hi": "high",
        "lo": "low", "c": "close", "v": "volume"
    })
    df = df.with_columns([
        (pl.col("open") / 100).alias("open"),
        (pl.col("high") / 100).alias("high"),
        (pl.col("low") / 100).alias("low"),
        (pl.col("close") / 100).alias("close"),
    ])
    for ticker in df.get_column("ticker").unique():
        sub_df = df.filter(pl.col("ticker") == ticker)
        path = EOD_OUTPUT_DIR / f"{ticker}.parquet"
        if path.exists():
            existing = pl.read_parquet(path)
            updated = pl.concat([existing, sub_df]).unique(subset=["date"])
        else:
            updated = sub_df
        updated.sort("date").write_parquet(path)

async def save_intraday(records: List[asyncpg.Record]) -> None:
    if not records:
        return
    df = pl.DataFrame(records).rename({
        "stk": "ticker", "dt": "timestamp", "o": "open", "hi": "high",
        "lo": "low", "c": "close", "v": "volume", "oi": "open_interest"
    })
    df = df.with_columns([
        (pl.col("open") / 100).alias("open"),
        (pl.col("high") / 100).alias("high"),
        (pl.col("low") / 100).alias("low"),
        (pl.col("close") / 100).alias("close"),
        pl.col("timestamp").dt.strftime("%Y-%m-%d").alias("date"),
        pl.col("timestamp").dt.strftime("%H:%M:%S").alias("time")
    ])
    for ticker in df.get_column("ticker").unique():
        ticker_df = df.filter(pl.col("ticker") == ticker)
        for date in ticker_df.get_column("date").unique():
            day_df = ticker_df.filter(pl.col("date") == date)
            ticker_dir = INTRADAY_OUTPUT_DIR / ticker
            ticker_dir.mkdir(parents=True, exist_ok=True)
            path = ticker_dir / f"{date}.parquet"
            day_df.drop("ticker").write_parquet(path)

async def main(start_date: str, end_date: str) -> None:
    conn = await asyncpg.connect(**DB_CONFIG)
    eod_records = await fetch_eod(conn, start_date, end_date)
    intraday_records = await fetch_intraday(conn, start_date, end_date)
    await save_eod(eod_records)
    await save_intraday(intraday_records)
    await conn.close()

if __name__ == "__main__":
    calendar = TradingCalendar.from_parquet(CALENDAR_PATH)
    default_date = calendar.current_business_date(hour=20)

    parser = argparse.ArgumentParser(description="Update parquet files from DB.")
    parser.add_argument("--start_date", type=str, default=default_date, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end_date", type=str, default=default_date, help="End date (YYYY-MM-DD)")
    args = parser.parse_args()

    asyncio.run(main(args.start_date, args.end_date))
