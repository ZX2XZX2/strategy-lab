from datetime import datetime
import polars as pl
from strategy_lab.config import EOD_DIR, INTRADAY_DIR, SPLITS_DIR
from strategy_lab.utils.adjuster import Adjuster
from strategy_lab.utils.trading_calendar import TradingCalendar
from strategy_lab.utils.logger import get_logger

logger = get_logger(__name__)

# This module provides a DataLoader class to load end-of-day (EOD) and intraday data
# from parquet files. It includes methods to load EOD data, intraday data, and splits data.
# The DataLoader class is initialized with a TradingCalendar instance to manage trading days.
# It also includes methods to apply splits to the loaded data using the Adjuster class.

class DataLoader:
    def __init__(self, calendar: TradingCalendar):
        self.eod_path = EOD_DIR
        self.intraday_path = INTRADAY_DIR
        self.splits_path = SPLITS_DIR
        self.calendar = calendar

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
                frames.append(df)
        if not frames:
            return pl.DataFrame()
        df = pl.concat(frames)
        splits = self._load_splits(ticker)
        return Adjuster.apply_splits(df, splits, date_col="date")

    def _load_splits(self, ticker: str) -> pl.DataFrame:
        df = pl.read_parquet(self.splits_path)
        return df.filter(pl.col("ticker") == ticker).sort("date")
