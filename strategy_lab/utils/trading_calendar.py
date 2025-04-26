from datetime import datetime
from typing import List
import polars as pl

class TradingCalendar:
    def __init__(self, trading_days: List[str]):
        """
        Args:
            trading_days (List[str]): List of trading days in 'YYYY-MM-DD' format, sorted ascending.
        """
        self.trading_days = trading_days
        self._day_to_index = {day: idx for idx, day in enumerate(trading_days)}

    @classmethod
    def from_parquet(cls, calendar_path: str) -> "TradingCalendar":
        """
        Instantiate a TradingCalendar from a parquet file.
        """
        df = pl.read_parquet(calendar_path)
        trading_days = df.sort("date").get_column("date").to_list()
        return cls(trading_days)

    @staticmethod
    def build_calendar_from_dates(dates: List[str], output_path: str) -> None:
        """
        Build and save a trading calendar parquet file from a list of dates.
        """
        df = pl.DataFrame({"date": sorted(dates)})
        df.write_parquet(output_path)

    def date_range(self, start_date: str, end_date: str) -> List[str]:
        """
        Returns a list of trading days between start_date and end_date (inclusive).
        """
        if start_date not in self._day_to_index or end_date not in self._day_to_index:
            raise ValueError("Start or end date not in trading calendar.")
        start_idx = self._day_to_index[start_date]
        end_idx = self._day_to_index[end_date]
        return self.trading_days[start_idx:end_idx+1]

    def next(self, date: str, steps: int = 1) -> str:
        """
        Returns the trading date 'steps' days after the given date.
        """
        idx = self._day_to_index.get(date)
        if idx is None:
            raise ValueError(f"Date {date} not in trading calendar.")
        new_idx = idx + steps
        if new_idx < 0 or new_idx >= len(self.trading_days):
            raise IndexError("Date navigation out of bounds.")
        return self.trading_days[new_idx]

    def previous(self, date: str, steps: int = 1) -> str:
        """
        Returns the trading date 'steps' days before the given date.
        """
        return self.next(date, steps=-steps)

    def contains(self, date: str) -> bool:
        """
        Returns True if the date is a trading day.
        """
        return date in self._day_to_index
