import argparse
import asyncio
from datetime import datetime
import json
import os
import polars as pl
import strategy_lab.config as cfg
from strategy_lab.data.loader import DataLoader
from strategy_lab.utils.logger import get_logger
from strategy_lab.utils.trading_calendar import TradingCalendar
from tqdm.asyncio import tqdm_asyncio
from tqdm import tqdm
from typing import List

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
    # else:
    #     print(df)
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
    # print(f"Calculated {window}-day relative strength:")
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
    # print(f"Calculated {window}-day intraday volatility:")
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
    # print(f"Calculated weighted average for {prefix} bucket as '{weighted_col_name}'.")
    # print(df.select(weighted_col_name).head(1))
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


async def download_eod_data(start_date: str, end_date: str, max_window: int = 0, tickers: List[str] = None) -> List[pl.DataFrame]:
    """
    Download EOD data for a list of tickers from the database.
    Args:
        start_date (str): Start date in YYYY-MM-DD format.
        end_date (str): End date in YYYY-MM-DD format.
        max_window (int): Maximum window size for calculations.
        tickers (List[str]): List of tickers to download data for.
    Returns:
        List[pl.DataFrame]: List of DataFrames containing EOD data for each ticker.
    """
    data_loader = DataLoader()
    start_date = data_loader.calendar.previous(start_date, max_window)
    if tickers is None:
        logger.info(f"Downloading EOD data from {start_date} to {end_date} for all tickers")
    else:
        logger.info(f"Downloading EOD data from {start_date} to {end_date} for tickers: {tickers}")
    dfs = await data_loader.load_all_eod_data(start_date, end_date, tickers)
    if len(dfs)< 10:
        for df in dfs:
            print(df)
    else:
        logger.info(f"Data loaded successfully for {len(dfs)} tickers.")
    return dfs

async def calculate_indicators(dfs: List[pl.DataFrame], config: dict, start_date: str) -> pl.DataFrame:

    async def process_indicator(df, indicator, window):
        func = globals().get(indicator)
        if func:
            # print(f"Calculating {indicator} with window {window}")
            updated_df = await asyncio.to_thread(func, df, window)
            # Combine the result with the original dataframe
            df = df.hstack(updated_df.select([col for col in updated_df.columns if col not in df.columns]))
            return df
        else:
            logger.warning(f"Warning: Function {indicator} not found!")
            return df

    async def process_all_indicators(df):
        for indicator, windows in config['indicators'].items():
            for window in windows:
                df = await process_indicator(df, indicator, window)
        return df

    # Launch processing for all dataframes concurrently
    tasks = [process_all_indicators(df) for df in dfs]
    results = []
    with tqdm_asyncio(desc="Calculating indicators", total=len(tasks)) as progress:
        for task in tqdm_asyncio.as_completed(tasks):
            result = await task
            results.append(result)
            progress.update(1)
        progress.close()
    return results

def fill_null_values(df: pl.DataFrame) -> pl.DataFrame:
    # Get a list of all relative strength columns sorted by window size (ascending)
    rel_strength_cols = sorted(
        [col for col in df.columns if col.startswith("relative_strength_")],
        key=lambda x: int(x.split("_")[-1])
    )

    # Start with the smallest window and propagate values to the larger ones
    for i in range(1, len(rel_strength_cols)):
        small_col = rel_strength_cols[i - 1]
        large_col = rel_strength_cols[i]
        # Fill nulls in the larger window column with the values from the smaller window column
        df = df.with_columns(
            pl.col(large_col).fill_null(pl.col(small_col)).alias(large_col)
        )
        print(f"Filled nulls in {large_col} with values from {small_col}")

    return df

async def process_and_save_indicators(dfs: list, start_date: str, output_path: str):

    def filter_dataframe(df: pl.DataFrame, start_date) -> pl.DataFrame:

        # Step 1: Filter rows by date
        df = df.filter(pl.col("date") >= start_date)
        # Step 2: Drop unnecessary columns
        df = df.drop(["open", "high", "low", "close", "volume"])
        return df

    start_date = datetime.strptime(start_date, "%Y-%m-%d").date()

    # Parallel filtering with progress bar
    tasks = [asyncio.to_thread(filter_dataframe, df, start_date) for df in dfs]
    filtered_dfs = []
    with tqdm_asyncio(desc="Filtering DataFrames", total=len(tasks)) as progress:
        for task in asyncio.as_completed(tasks):
            result = await task
            filtered_dfs.append(result)
            progress.update(1)
        progress.close()

    # Concatenate all dataframes
    combined_df = pl.concat(filtered_dfs)
    sorted_df = combined_df.sort("date")
    # Fill null values in relative strength columns
    sorted_df = fill_null_values(sorted_df)
    len_1 = len(sorted_df)
    # Drop nulls first
    sorted_df = sorted_df.drop_nulls()

    # Identify float columns
    float_cols = [col for col, dtype in zip(sorted_df.columns, sorted_df.dtypes) if dtype in (pl.Float32, pl.Float64)]

    # Remove rows with NaNs in any float column
    for col in float_cols:
        sorted_df = sorted_df.filter(~(pl.col(col).is_nan() | (pl.col(col) == float("inf")) | (pl.col(col) == float("-inf"))))

    len_2 = len(sorted_df)
    logger.info(f"Filtered DataFrame length before drop: {len_1}, after drop: {len_2}")
    sorted_df.write_parquet(output_path, compression="zstd", row_group_size=12000)


def rank_and_bucket_indicators(start_date: str, end_date: str, config: dict):
    parquet_path = os.path.join(cfg.INDICATORS_DIR, f"{start_date}_{end_date}.parquet")
    date_range = TradingCalendar().date_range(start_date, end_date)
    if not date_range:
        logger.warning(f"No trading days found between {start_date} and {end_date}.")
        return pl.DataFrame()
    logger.info(f"Ranking and bucketizing indicators for {len(date_range)} dates, between {date_range[0]} and {date_range[-1]}")

    # Get all indicator columns dynamically
    indicator_cols = []
    for indicator, windows in config['indicators'].items():
        for window in windows:
            indicator_cols.append(f"{indicator[10:]}_{window}")

    def load_date(date: str) -> pl.DataFrame:
        date = datetime.strptime(date, "%Y-%m-%d").date()
        # Lazy load data for a specific date
        lazy_df = pl.scan_parquet(parquet_path).filter(pl.col("date") == date)
        df = lazy_df.collect()
        return df

    def rank_and_bucketize(df: pl.DataFrame, indicator_cols: list) -> pl.DataFrame:
        # Rank and bucketize all indicators in one shot
        rank_exprs = [pl.col(col).rank("dense").alias(f"rank_{col}") for col in indicator_cols]
        bucket_exprs = [((pl.col(col) / pl.col(col).max()) * 99).cast(pl.Int32).alias(f"bucket_{col}") for col in indicator_cols]

        # Apply the ranking and bucketizing in one go
        df = df.with_columns(rank_exprs + bucket_exprs)
        return df

    processed_dfs = []
    for date in tqdm(date_range, desc="Loading and processing dates"):
        df = load_date(date)
        df = rank_and_bucketize(df, indicator_cols)
        processed_dfs.append(df)

    # Concatenate all processed dataframes
    combined_df = pl.concat(processed_dfs, how="vertical")

    # Save the final DataFrame to Parquet
    output_path = os.path.join(cfg.INDICATORS_DIR, f"ranked_{start_date}_{end_date}.parquet")
    combined_df.write_parquet(output_path, compression="zstd", row_group_size=12000)
    logger.info(f"Ranked indicators saved to {output_path}")


async def main():

    parser = argparse.ArgumentParser(description="Batch convert EOD and intraday data to Parquet.")
    parser.add_argument("--start-date", type=str, required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--config-file", type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json"), help="Path to the config file")
    parser.add_argument("--skip-indicators", action="store_true", help="Skip indicator calculation")
    parser.add_argument("--skip-ranking", action="store_true", help="Skip ranking and bucketization")
    args = parser.parse_args()
    indicator_start_date = args.start_date
    end_date = args.end_date
    config_file = args.config_file
    logger.info(f"Start date: {indicator_start_date}, End date: {end_date}, Config file: {config_file}")

    with open(config_file, 'r') as f:
        config = json.load(f)

    max_window = 0
    # Find the largest window size
    for indicator, windows in config['indicators'].items():
        max_window = max(max_window, max(windows))
    logger.info(f"Largest window found: {max_window}")

    if not args.skip_indicators:
        ticker_dfs = await download_eod_data(indicator_start_date, end_date, max_window=max_window)
        ticker_dfs = await calculate_indicators(ticker_dfs, config, indicator_start_date)
        output_path = os.path.join(cfg.INDICATORS_DIR, f"{indicator_start_date}_{end_date}.parquet")
        await process_and_save_indicators(ticker_dfs, indicator_start_date, output_path)
        logger.info(f"Indicators calculated and saved to {output_path}")

    if not args.skip_ranking:
        # Load the saved indicators
        rank_and_bucket_indicators(indicator_start_date, end_date, config)

if __name__ == '__main__':
    asyncio.run(main())
    # asyncio.run(download_eod_data("2023-01-01", "2024-12-31"))
    # asyncio.run(download_eod_data("2024-08-01", "2024-08-09", ["COIN", "MSTR", "SMCI"]))
