import polars as pl
from strategy_lab.config import EOD_DIR, INTRADAY_DIR, SPLITS_DIR
from strategy_lab.utils.adjuster import Adjuster
from strategy_lab.utils.trading_calendar import TradingCalendar

class DataLoader:
    def __init__(self, calendar: TradingCalendar):
        self.eod_path = EOD_DIR
        self.intraday_path = INTRADAY_DIR
        self.splits_path = SPLITS_DIR
        self.calendar = calendar

    def load_eod(self, ticker: str, as_of_date: str = None) -> pl.DataFrame:
        df = pl.read_parquet(self.eod_path / f"{ticker}.parquet")
        splits = self._load_splits(ticker)
        if as_of_date:
            splits = splits.filter(pl.col("split_date") <= as_of_date)
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
