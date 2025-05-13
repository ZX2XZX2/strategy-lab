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


def calculate_activity(df, window=20, lazy=False):
    start_time = time.time()
    wap = (pl.col("open") + pl.col("high") + pl.col("low") + pl.col("close")) / 4
    activity = pl.col("volume") * wap
    avg_activity = activity.rolling_mean(window)
    if lazy:
        df = df.with_columns(activity.alias("activity"), wap.alias("weighted_avg_price"), avg_activity.alias(f"activity_{window}")).collect()
    else:
        df = df.with_columns(activity.alias("activity"), wap.alias("weighted_avg_price"), avg_activity.alias(f"activity_{window}"))
    end_time = time.time()
    if df.is_empty():
        print("[No Data]")
    else:
        print(df)
    print(f"Calculated {window}-day average activity in {end_time - start_time:.4f} seconds")
    return df


def calculate_relative_strength(df, window=252, lazy=False):
    w1, w2, w3 = window // 4, window // 2, window
    rs = 40 * (pl.col("close") / pl.col("close").shift(w1)) + 30 * (pl.col("close") / pl.col("close").shift(w2)) + 30 * (pl.col("close") / pl.col("close").shift(w3))
    if lazy:
        df = df.with_columns(rs.alias(f"relative_strength_{window}")).collect()
    else:
        df = df.with_columns(rs.alias(f"relative_strength_{window}"))
    print(f"Calculated {window}-day relative strength:")
    return df


def calculate_intraday_volatility(df, window=20, lazy=False):
    volatility = 100 * (pl.col("high") - pl.col("low")) / pl.col("weighted_avg_price")
    avg_volatility = volatility.rolling_mean(window)
    if lazy:
        df = df.with_columns(volatility.alias("intraday_volatility"), avg_volatility.alias(f"intraday_volatility_{window}")).collect()
    else:
        df = df.with_columns(volatility.alias("intraday_volatility"), avg_volatility.alias(f"intraday_volatility_{window}"))
    print(f"Calculated {window}-day intraday volatility:")
    return df


async def main():
    db_url = os.getenv("POSTGRES_CNX")
    table_name = 'eods'
    start_date = '2024-01-01'
    end_date = '2025-12-31'
    parquet_path = cfg.ROOT_DIR / "eod.parquet"
    save_to_parquet_flag = False

    df = await download_eod_data(db_url, table_name, start_date, end_date)
    if save_to_parquet_flag:
        save_to_parquet(df, parquet_path)
        df = load_eod_data(parquet_path)
        lazy = True
    else:
        lazy = False

    df = calculate_activity(df, window=20, lazy=lazy)
    df = calculate_activity(df, window=5, lazy=lazy)
    df = calculate_intraday_volatility(df, window=20, lazy=lazy)
    df = calculate_relative_strength(df, window=252, lazy=lazy)
    df = calculate_relative_strength(df, window=45, lazy=lazy)
    df = calculate_relative_strength(df, window=10, lazy=lazy)
    df = calculate_relative_strength(df, window=4, lazy=lazy)

    calculate_activity(df, lazy=lazy)
    df.write_parquet(cfg.ROOT_DIR / "indicators.parquet")


if __name__ == '__main__':
    asyncio.run(main())
