import argparse
import asyncio
import json
import polars as pl
from strategy_lab.data.loader import DataLoader
from strategy_lab.utils.logger import get_logger

logger = get_logger(__name__)

def calculate_activity(df: pl.DataFrame, window: int = 20, lazy: bool = False) -> pl.DataFrame:
    """
    Calculate activity for a given rolling window.

    Args:
        df (pl.DataFrame): The EOD data containing 'open', 'high', 'low', 'close', and 'volume'.
        window (int): The rolling window size (e.g., 5, 20).
        lazy(bool): If True, use lazy evaluation. Defaults to False.

    Returns:
        pl.DataFrame: DataFrame with activity indicators.
    """
    wap = (pl.col("open") + pl.col("high") + pl.col("low") + pl.col("close")) / 4
    activity = pl.col("volume") * wap
    avg_activity = activity.rolling_mean(window)
    if lazy:
        df = df.with_columns(activity.alias("activity"), wap.alias("weighted_avg_price"), avg_activity.alias(f"activity_{window}")).collect()
    else:
        df = df.with_columns(activity.alias("activity"), wap.alias("weighted_avg_price"), avg_activity.alias(f"activity_{window}"))
    if df.is_empty():
        print("[No Data]")
    else:
        print(df)
    return df


def calculate_relative_strength(df: pl.DataFrame, window: int = 252, lazy: bool = False) -> pl.DataFrame:
    """
    Calculate relative strength for a given rolling window.

    Args:
        df (pl.DataFrame): The EOD data containing 'open', 'high', 'low', 'close', and 'volume'.
        window (int): The rolling window size (e.g., 5, 20).
        lazy(bool): If True, use lazy evaluation. Defaults to False.

    Returns:
        pl.DataFrame: DataFrame with relative strength indicators.
    """
    w1, w2, w3 = window // 4, window // 2, window
    rs = 40 * (pl.col("close") / pl.col("close").shift(w1)) + 30 * (pl.col("close") / pl.col("close").shift(w2)) + 30 * (pl.col("close") / pl.col("close").shift(w3))
    if lazy:
        df = df.with_columns(rs.alias(f"relative_strength_{window}")).collect()
    else:
        df = df.with_columns(rs.alias(f"relative_strength_{window}"))
    print(f"Calculated {window}-day relative strength:")
    return df


def calculate_intraday_volatility(df: pl.DataFrame, window: int = 20, lazy: bool = False)-> pl.DataFrame:
    """
    Calculate intraday volatility for a given rolling window.
    Args:
        df (pl.DataFrame): The EOD data containing 'open', 'high', 'low', 'close', and 'volume'.
        window (int): The rolling window size (e.g., 5, 20).
        lazy(bool): If True, use lazy evaluation. Defaults to False.
    Returns:
        pl.DataFrame: DataFrame with intraday volatility indicators.
    """
    volatility = 100 * (pl.col("high") - pl.col("low")) / pl.col("weighted_avg_price")
    avg_volatility = volatility.rolling_mean(window)
    if lazy:
        df = df.with_columns(volatility.alias("intraday_volatility"), avg_volatility.alias(f"intraday_volatility_{window}")).collect()
    else:
        df = df.with_columns(volatility.alias("intraday_volatility"), avg_volatility.alias(f"intraday_volatility_{window}"))
    print(f"Calculated {window}-day intraday volatility:")
    return df


def save_to_parquet(df, parquet_path):
    df.write_parquet(parquet_path)


def load_eod_data(parquet_path):
    return pl.scan_parquet(parquet_path)


def weighted_average(df, prefix, weights):
    bucket_cols = [col for col in df.columns if col.startswith(f"bucket_{prefix}")]
    if len(bucket_cols) != len(weights):
        print(f"Error: Number of columns with prefix '{prefix}' does not match the number of weights.")
        return df
    weighted_cols = [pl.col(col) * weight for col, weight in zip(bucket_cols, weights)]
    weighted_sum = sum(weighted_cols)
    total_weight = sum(weights)
    weighted_avg = weighted_sum / total_weight
    weighted_col_name = f"bucket_{prefix}"
    df = df.with_columns(weighted_avg.alias(weighted_col_name))
    print(f"Calculated weighted average for {prefix} bucket as '{weighted_col_name}'.")
    print(df.select(weighted_col_name).head(1))
    return df


def calculate_overall_rank(df, weights):
    bucket_cols = [col for col in df.columns if col.startswith("bucket_")]
    if len(bucket_cols) != len(weights):
        print("Error: Number of bucket columns does not match the number of weights.")
        return df
    weighted_cols = [pl.col(col) * weight for col, weight in zip(bucket_cols, weights)]
    weighted_sum = sum(weighted_cols)
    total_weight = sum(weights)
    overall_rank = weighted_sum / total_weight
    df = df.with_columns(overall_rank.alias("overall_rank"))
    print("Calculated overall rank.")
    print(df.select("overall_rank").head(1))
    return df


def bucketize(df, column_name):
    bucket_col = f"bucket_{column_name}"
    df = df.with_columns(((pl.col(column_name).rank(method="max") - 1) * 99 / (pl.col(column_name).count() - 1)).cast(pl.Int64).alias(bucket_col))
    print(f"Calculated bucket for {column_name}.")
    print(df.select(column_name, bucket_col).head(1))
    return df


def rankize(df, column_name):
    rank_col = f"rank_{column_name}"
    df = df.with_columns(pl.col(column_name).rank(method="dense").over("date").alias(rank_col))
    print(f"Calculated rank for {column_name} per date.")
    print(df.select("date", column_name, rank_col).head(1))
    return df


def filter_initial_rows(df, n=252):
    print(f"Filtering out the first {n} rows.")
    df = df.slice(n, df.height - n)
    print("Data after filtering:")
    print(df.head(1))
    return df


def calculate_weighted_average(df, columns, weights, result_name):
    weighted_cols = [pl.col(col) * weight for col, weight in zip(columns, weights)]
    weighted_sum = sum(weighted_cols)
    total_weight = sum(weights)
    weighted_avg = weighted_sum / total_weight
    df = df.with_columns(weighted_avg.alias(result_name))
    print(f"Calculated weighted average for {result_name}.")
    print(df.select(result_name).head(1))
    return df


def calculate_aggregated_rank(df, config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    for layer in config['layers']:
        name = layer['name']
        columns = layer['columns']
        weights = layer['weights']
        df = calculate_weighted_average(df, columns, weights, name)
    print("Completed hierarchical weighted average calculation.")
    return df


# async def main():
#     db_url = 'postgresql://user:password@localhost:5432/eod'
#     table_name = 'eod_data'
#     start_date = '2023-01-01'
#     end_date = '2023-12-31'
#     parquet_path = 'data/eod_data.parquet'
#     config_path = 'config/weighted_average.json'
#     save_to_parquet_flag = True

#     df = await download_eod_data(db_url, table_name, start_date, end_date)
#     if save_to_parquet_flag:
#         save_to_parquet(df, parquet_path)
#         df = load_eod_data(parquet_path)
#     else:
#         pass

#     # Calculate hierarchical weighted averages
#     df = calculate_aggregated_rank(df, config_path)

#     df = filter_initial_rows(df, n=252)

async def download_eod_data(start_date, end_date, tickers=None):
    logger.info(f"Downloading EOD data from {start_date} to {end_date} for tickers: {tickers}")
    data_loader = DataLoader()
    dfs = await data_loader.load_all_eod_data(start_date, end_date, tickers)
    if len(dfs)< 10:
        for df in dfs:
            print(df)
    else:
        logger.info(f"Data loaded successfully for {len(dfs)} tickers.")

async def main():
    parser = argparse.ArgumentParser(description="Batch convert EOD and intraday data to Parquet.")
    parser.add_argument("--start-date", type=str, required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--intraday-only", action="store_true", help="Only update intraday data")
    parser.add_argument("--eod-only", action="store_true", help="Only update EOD and splits data")
    parser.parse_args()


if __name__ == '__main__':
    # asyncio.run(main())
    asyncio.run(download_eod_data("2023-01-01", "2024-12-31"))
    # asyncio.run(download_eod_data("2024-08-01", "2024-08-09", ["COIN", "MSTR", "SMCI"]))
