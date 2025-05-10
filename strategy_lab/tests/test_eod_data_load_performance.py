import time
import asyncio
import polars as pl
import asyncpg
import os
import strategy_lab.config as cfg
from strategy_lab.utils.logger import get_logger
logger = get_logger(__name__)


async def download_eod_data(db_url, table_name, start_date, end_date):
    conn = await asyncpg.connect(db_url)
    start_time = time.time()
    query = f"SELECT * FROM {table_name} WHERE dt BETWEEN '{start_date}' AND '{end_date}'"
    rows = await conn.fetch(query)
    await conn.close()
    if not rows:
        # If no rows are found, log a warning and return an empty DataFrame
        logger.warning(f"No EOD data found between {start_date} and {end_date}.")
        return pl.DataFrame([])
    records = [dict(row) for row in rows]
    df = pl.DataFrame(records)
    end_time = time.time()
    print(f"Downloaded EOD data from database in {end_time - start_time:.4f} seconds")
    return df.rename({"stk": "ticker", "dt": "date", "o": "open", "hi": "high", "lo": "low", "c": "close", "v": "volume"})


def save_to_parquet(df, parquet_path):
    start_time = time.time()
    df.write_parquet(parquet_path)
    end_time = time.time()
    print(f"Saved EOD data to Parquet in {end_time - start_time:.4f} seconds")


def load_eod_data(parquet_path):
    return pl.scan_parquet(parquet_path)


def calculate_activity(df, lazy=False):
    start_time = time.time()
    wap = (pl.col("open") + pl.col("high") + pl.col("low") + pl.col("close")) / 4
    if lazy:
        df = df.with_columns((pl.col("volume") * wap).alias("activity"), wap.alias("weighted_avg_price"))
        df = df.select(*[col for col in df.columns if col not in ["weighted_avg_price", "activity"]], "weighted_avg_price", "activity").collect()
    else:
        df = df.with_columns((pl.col("volume") * wap).alias("activity"), wap.alias("weighted_avg_price"))
        df = df.select(*[col for col in df.columns if col not in ["weighted_avg_price", "activity"]], "weighted_avg_price", "activity")
    end_time = time.time()
    if df.is_empty():
        print("[No Data]")
    else:
        print(df)
    print(f"Calculated 20-day average activity in {end_time - start_time:.4f} seconds")
    df.write_parquet(cfg.ROOT_DIR / "activity.parquet")


async def main():
    db_url = os.getenv("POSTGRES_CNX")
    table_name = 'eods'
    start_date = '2023-01-01'
    end_date = '2023-12-31'
    parquet_path = cfg.ROOT_DIR / "2023.parquet"
    save_to_parquet_flag = True

    df = await download_eod_data(db_url, table_name, start_date, end_date)
    if save_to_parquet_flag:
        save_to_parquet(df, parquet_path)
        df = load_eod_data(parquet_path)
        lazy = True
    else:
        lazy = False

    calculate_activity(df, lazy=lazy)


if __name__ == '__main__':
    asyncio.run(main())
