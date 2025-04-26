import asyncio
import polars as pl
import os
import asyncpg
import sys
from pathlib import Path

DB_CONFIG = {
    "user": os.getenv("POSTGRES_USER"),
    "password": os.getenv("POSTGRES_PASSWORD"),
    "port": os.getenv("POSTGRES_PORT"),
    "database": os.getenv("POSTGRES_DB"),
    "host": os.getenv("POSTGRES_HOST")
}

EOD_OUTPUT_DIR = Path("path/to/eod")
INTRADAY_OUTPUT_DIR = Path("path/to/intraday")

async def fetch_eod(conn, start_date, end_date):
    query = f"""
    SELECT stk, dt, o, hi, lo, c, v
    FROM eod_table
    WHERE dt BETWEEN '{start_date}' AND '{end_date}'
    """
    return await conn.fetch(query)

async def fetch_intraday(conn, start_date, end_date):
    query = f"""
    SELECT stk, dt, o, hi, lo, c, v
    FROM intraday_table
    WHERE dt::date BETWEEN '{start_date}' AND '{end_date}'
    """
    return await conn.fetch(query)

async def save_eod(records):
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

async def save_intraday(records):
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

async def main(start_date, end_date):
    conn = await asyncpg.connect(**DB_CONFIG)
    eod_records = await fetch_eod(conn, start_date, end_date)
    intraday_records = await fetch_intraday(conn, start_date, end_date)
    await save_eod(eod_records)
    await save_intraday(intraday_records)
    await conn.close()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python update_parquet.py START_DATE END_DATE (format YYYY-MM-DD)")
        sys.exit(1)
    start_date = sys.argv[1]
    end_date = sys.argv[2]
    asyncio.run(main(start_date, end_date))