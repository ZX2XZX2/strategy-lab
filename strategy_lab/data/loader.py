import asyncio
import asyncpg
from datetime import datetime, timedelta
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
    def __init__(self, calendar: TradingCalendar | None = None):
        if calendar is None:
            calendar = TradingCalendar()
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

    def load_eod(
        self,
        ticker: str,
        as_of_date: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        data_source: str = "db",
        records: list[dict] | None = None,
    ) -> pl.DataFrame:
        if data_source == "db":
            df = asyncio.run(
                self._fetch_eod_from_db(ticker, start_date or as_of_date, end_date or as_of_date)
            )
        elif data_source == "dict":
            if not records:
                return pl.DataFrame()
            df = pl.DataFrame(records).rename(
                {
                    "stk": "ticker",
                    "dt": "date",
                    "o": "open",
                    "hi": "high",
                    "lo": "low",
                    "c": "close",
                    "v": "volume",
                    "oi": "open_interest",
                }
            )
            if df.get_column("date").dtype != pl.Date:
                df = df.with_columns(pl.col("date").str.strptime(pl.Date, "%Y-%m-%d"))
        else:
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

        split_source = "db" if data_source == "db" else "file"
        splits = self._load_splits(ticker, data_source=split_source)
        if as_of_date:
            as_of_date = datetime.strptime(as_of_date, "%Y-%m-%d").date()
            splits = splits.filter(pl.col("date") <= as_of_date)

        return Adjuster.apply_splits(df, splits, date_col="date")

    def add_eod(
        self,
        df: pl.DataFrame,
        ticker: str,
        as_of_date: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        data_source: str = "db",
        records: list[dict] | None = None,
    ) -> pl.DataFrame:
        """Append additional EOD records to an existing DataFrame."""
        new_df = self.load_eod(
            ticker,
            as_of_date=as_of_date,
            start_date=start_date,
            end_date=end_date,
            data_source=data_source,
            records=records,
        )
        if df.is_empty():
            return new_df
        if new_df.is_empty():
            return df

        last_date = df.get_column("date").max()
        split_source = "db" if data_source == "db" else "file"
        splits = self._load_splits(ticker, data_source=split_source)
        if as_of_date:
            as_dt = datetime.strptime(as_of_date, "%Y-%m-%d").date()
            splits = splits.filter(pl.col("date") <= as_dt)
        new_splits = splits.filter(pl.col("date") > last_date)
        if not new_splits.is_empty():
            df = Adjuster.apply_splits(df, new_splits, date_col="date")

        return pl.concat([df, new_df]).sort("date")

    def append_eod_records(
        self,
        df: pl.DataFrame,
        ticker: str,
        as_of_date: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        data_source: str = "db",
        records: list[dict] | None = None,
    ) -> pl.DataFrame:
        """Wrapper for backwards compatibility."""
        return self.add_eod(
            df,
            ticker,
            as_of_date=as_of_date,
            start_date=start_date,
            end_date=end_date,
            data_source=data_source,
            records=records,
        )

    def load_intraday(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        data_source: str = "db",
        records: list[dict] | None = None,
        as_of_date: str | None = None,
    ) -> pl.DataFrame:
        if data_source == "db":
            df = asyncio.run(self._fetch_intraday_from_db(ticker, start_date, end_date))
        elif data_source == "dict":
            if not records:
                return pl.DataFrame()
            df = pl.DataFrame(records).rename(
                {
                    "stk": "ticker",
                    "dt": "timestamp",
                    "o": "open",
                    "hi": "high",
                    "lo": "low",
                    "c": "close",
                    "v": "volume",
                }
            )
            if df.get_column("timestamp").dtype != pl.Datetime:
                df = df.with_columns(pl.col("timestamp").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S"))
            if "time" not in df.columns:
                df = df.with_columns(pl.col("timestamp").dt.strftime("%H:%M:%S").alias("time"))
        else:
            frames = []
            for date in self.calendar.date_range(start_date, end_date):
                path = self.intraday_path / ticker / f"{date}.parquet"
                if path.exists():
                    parquet_df = pl.read_parquet(path)
                    date_str = path.stem
                    parquet_df = parquet_df.with_columns(
                        pl.concat_str([pl.lit(date_str), pl.col("time")], separator=" ")
                        .str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S")
                        .alias("timestamp")
                    )
                    frames.append(parquet_df)
            df = pl.concat(frames) if frames else pl.DataFrame()

            if df.is_empty():
                try:
                    df = asyncio.run(
                        self._fetch_intraday_from_db(ticker, start_date, end_date)
                    )
                except Exception as e:
                    logger.error(f"Error loading intraday data from database: {e}")

        if df.is_empty():
            return df

        split_source = "db" if data_source == "db" else "file"
        splits = self._load_splits(ticker, data_source=split_source)
        if as_of_date:
            as_dt = datetime.strptime(as_of_date, "%Y-%m-%d").date()
            splits = splits.filter(pl.col("date") <= as_dt)
        return Adjuster.apply_splits(df, splits, date_col="timestamp")

    def add_intraday(
        self,
        df: pl.DataFrame,
        ticker: str,
        start_date: str,
        end_date: str,
        data_source: str = "db",
        records: list[dict] | None = None,
    ) -> pl.DataFrame:
        """Append additional intraday records to an existing DataFrame."""
        new_df = self.load_intraday(
            ticker,
            start_date=start_date,
            end_date=end_date,
            data_source=data_source,
            records=records,
            as_of_date=end_date,
        )

        if df.is_empty():
            return new_df
        if new_df.is_empty():
            return df

        last_ts = df.get_column("timestamp").max()
        last_date = pl.Series([last_ts]).dt.date()[0]

        split_source = "db" if data_source == "db" else "file"
        splits = self._load_splits(ticker, data_source=split_source)
        new_splits = splits.filter(pl.col("date") > last_date)
        if not new_splits.is_empty():
            df = Adjuster.apply_splits(df, new_splits, date_col="timestamp")

        return pl.concat([df, new_df]).sort("timestamp")

    def append_intraday_records(
        self,
        df: pl.DataFrame,
        ticker: str,
        start_date: str,
        end_date: str,
        data_source: str = "db",
        records: list[dict] | None = None,
    ) -> pl.DataFrame:
        """Wrapper for backwards compatibility."""
        return self.add_intraday(
            df,
            ticker,
            start_date,
            end_date,
            data_source=data_source,
            records=records,
        )

    async def _fetch_intraday_from_db(self, ticker: str, start_date: str, end_date: str) -> pl.DataFrame:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1) - timedelta(seconds=1)
        query = (
            "SELECT dt, o, hi, lo, c, v FROM intraday "
            "WHERE stk = $1 AND dt BETWEEN $2 AND $3 ORDER BY dt"
        )
        conn = await asyncpg.connect(**DB_CONFIG)
        rows = await conn.fetch(query, ticker, start_dt, end_dt)
        await conn.close()
        if not rows:
            return pl.DataFrame()
        df = pl.DataFrame([dict(r) for r in rows]).rename(
            {
                "dt": "timestamp",
                "o": "open",
                "hi": "high",
                "lo": "low",
                "c": "close",
                "v": "volume",
            }
        )
        df = df.with_columns(pl.col("timestamp").dt.strftime("%H:%M:%S").alias("time"))
        return df

    async def _fetch_eod_from_db(
        self,
        ticker: str,
        start_date: str | None,
        end_date: str | None,
    ) -> pl.DataFrame:
        if start_date:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d").date()
        else:
            start_dt = datetime(1970, 1, 1).date()
        if end_date:
            end_dt = datetime.strptime(end_date, "%Y-%m-%d").date()
        else:
            end_dt = datetime.now().date()
        query = (
            "SELECT dt, o, hi, lo, c, v, oi FROM eods "
            "WHERE stk = $1 AND dt BETWEEN $2 AND $3 ORDER BY dt"
        )
        conn = await asyncpg.connect(**DB_CONFIG)
        rows = await conn.fetch(query, ticker, start_dt, end_dt)
        await conn.close()
        if not rows:
            return pl.DataFrame()
        df = pl.DataFrame([dict(r) for r in rows]).rename(
            {
                "dt": "date",
                "o": "open",
                "hi": "high",
                "lo": "low",
                "c": "close",
                "v": "volume",
                "oi": "open_interest",
            }
        )
        return df

    async def _fetch_splits_from_db(self, ticker: str) -> pl.DataFrame:
        query = (
            "SELECT stk, dt, ratio FROM dividends "
            "WHERE stk = $1 ORDER BY dt"
        )
        conn = await asyncpg.connect(**DB_CONFIG)
        rows = await conn.fetch(query, ticker)
        await conn.close()
        if not rows:
            return pl.DataFrame()
        df = pl.DataFrame([dict(r) for r in rows]).rename(
            {"stk": "ticker", "dt": "date"}
        )
        return df

    def _load_splits(self, ticker: str, data_source: str = "db") -> pl.DataFrame:
        if data_source == "db":
            df = asyncio.run(self._fetch_splits_from_db(ticker))
        else:
            df = pl.read_parquet(self.splits_path)
            if df.is_empty():
                return df
            if df.dtypes[1] != pl.Date:
                df = df.with_columns(
                    pl.col("date").str.strptime(pl.Date, "%Y-%m-%d")
                )
        return df.filter(pl.col("ticker") == ticker).sort("date")
