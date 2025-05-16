import asyncio
import asyncpg
from datetime import datetime
import polars as pl
from strategy_lab.config import EOD_DIR, INTRADAY_DIR, SPLITS_DIR, DB_CONFIG
from strategy_lab.utils.adjuster import Adjuster
from strategy_lab.utils.trading_calendar import TradingCalendar
from strategy_lab.utils.logger import get_logger
from tqdm import tqdm
from typing import List

logger = get_logger(__name__)

# This module provides a DataLoader class to load end-of-day (EOD) and intraday data
# from parquet files. It includes methods to load EOD data, intraday data, and splits data.
# The DataLoader class is initialized with a TradingCalendar instance to manage trading days.
# It also includes methods to apply splits to the loaded data using the Adjuster class.


class DataLoader:
    def __init__(self, calendar: TradingCalendar = TradingCalendar()):
        self.eod_path = EOD_DIR
        self.intraday_path = INTRADAY_DIR
        self.splits_path = SPLITS_DIR
        self.calendar = calendar

    async def load_all_eod_data(self, start_date: str, end_date: str, tickers: list = None) -> List[pl.DataFrame]:
        """
        Load all EOD data for the specified date range and tickers.

        Args:
            start_date (str): The start date in "YYYY-MM-DD" format.
            end_date (str): The end date in "YYYY-MM-DD" format.
            tickers (list, optional): List of tickers to load. If None, load all available tickers.

        Returns:
            List[pl.DataFrame]: A list of polars dataframes containing adjusted for splits ticker EOD data.
        """
        start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
        end_date = datetime.strptime(end_date, "%Y-%m-%d").date()
        if start_date > end_date:
            raise ValueError("start_date must be less than or equal to end_date")
        data_query = f"SELECT stk, dt, o, hi, lo, c, v FROM eods WHERE dt BETWEEN '{start_date}' AND '{end_date}'"
        splits_query = f"SELECT stk, dt, ratio FROM dividends WHERE dt BETWEEN '{start_date}' AND '{end_date}'"
        if tickers:
            tickers_str = "', '".join(tickers)
            data_query += f" AND stk IN ('{tickers_str}')"
            splits_query += f" AND stk IN ('{tickers_str}')"
        conn = None
        try:
            conn = await asyncpg.connect(**DB_CONFIG)
            data = await conn.fetch(data_query)
            splits = await conn.fetch(splits_query)
        except Exception as e:
            logger.error(f"Error loading data from PostgreSQL: {e}")
            return []
        finally:
            if conn:
                await conn.close()
        data_records = [dict(row) for row in data]
        split_records = [dict(row) for row in splits]
        logger.info(f"Loaded {len(data_records)} records from eods and {len(split_records)} records from dividends.")
        df = pl.DataFrame(data_records)
        df = df.rename({"stk": "ticker", "dt": "date", "o": "open", "hi": "high", "lo": "low", "c": "close", "v": "volume"})

        split_df = pl.DataFrame(split_records)
        split_df = split_df.rename({"stk": "ticker", "dt": "date"})

        # Split data by ticker
        tickers = df['ticker'].unique().to_list()
        # Invoke process_ticker for each ticker in parallel
        def process_ticker(ticker, df, split_df):
            ticker_df = df.filter(pl.col('ticker') == ticker)
            ticker_df = Adjuster.apply_splits(ticker_df, split_df.filter(pl.col('ticker') == ticker), date_col='date')
            return ticker_df
        # Process tickers asynchronously using asyncio.to_thread
        async def process_ticker_async(ticker):
            return await asyncio.to_thread(process_ticker, ticker, df, split_df)

        tasks = [process_ticker_async(ticker) for ticker in tickers]
        results = []
        for task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Processing tickers"):
            result = await task
            results.append(result)
        return results

    def load_eod(self, ticker: str, as_of_date: str = None, start_date: str = None, end_date: str = None) -> pl.DataFrame:
        path = self.eod_path / f"{ticker}.parquet"
        if not path.exists():
            logger.warning(f"EOD data for {ticker} not found at {path}.")
            return pl.DataFrame()

        if start_date or end_date:
            # Efficiently load only rows where the date >= start_date and date <= end_date
            query = pl.scan_parquet(str(path))
            if start_date:
                start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
                query = query.filter(pl.col("date") >= start_date)
            if end_date:
                end_date = datetime.strptime(end_date, "%Y-%m-%d").date()
                query = query.filter(pl.col("date") <= end_date)
            df = query.collect()
        else:
            df = pl.read_parquet(str(path))

        splits = self._load_splits(ticker)
        if as_of_date:
            as_of_date = datetime.strptime(as_of_date, "%Y-%m-%d").date()
            splits = splits.filter(pl.col("date") <= as_of_date)

        return Adjuster.apply_splits(df, splits, date_col="date")

    def load_intraday(self, ticker: str, start_date: str, end_date: str) -> pl.DataFrame:
        frames = []
        for date in self.calendar.date_range(start_date, end_date):
            path = self.intraday_path / ticker / f"{date}.parquet"
            if path.exists():
                df = pl.read_parquet(path)
                date_str = path.stem
                df = df.with_columns(
                    pl.concat_str([pl.lit(date_str), pl.col("time")], separator=" ").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S").alias("timestamp")
                )
                frames.append(df)
        if not frames:
            return pl.DataFrame()
        df = pl.concat(frames)
        splits = self._load_splits(ticker)
        return Adjuster.apply_splits(df, splits, date_col="timestamp")

    def _load_splits(self, ticker: str) -> pl.DataFrame:
        df = pl.read_parquet(self.splits_path)
        return df.filter(pl.col("ticker") == ticker).sort("date")
