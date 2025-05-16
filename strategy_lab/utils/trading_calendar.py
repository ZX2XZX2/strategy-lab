import bisect
from datetime import datetime
from typing import List
import polars as pl
from strategy_lab.config import CALENDAR_PATH

# This module provides a TradingCalendar class to manage trading days and their operations.
# It includes methods to generate a trading calendar from a parquet file, and
# to return a range of trading days between two dates.

class TradingCalendar:
    def __init__(self, calendar_path: str = CALENDAR_PATH):
        """
        Instantiate a TradingCalendar from a parquet file.
        """
        self.trading_days = pl.read_parquet(calendar_path).get_column("date").to_list()

    def date_range(self, start_date: str, end_date: str) -> List[str]:
        """
        Returns a list of trading days between start_date and end_date (inclusive).
        """
        # Adjust start to the next available business date if it's not in the calendar
        start = bisect.bisect_left(self.trading_days, start_date)
        if start >= len(self.trading_days):
            return []

        # Adjust end to the previous available business date if it's not in the calendar
        end = bisect.bisect_right(self.trading_days, end_date) - 1
        if end < 0:
            return []

        return self.trading_days[start : end + 1]

    def next(self, date: str, steps: int = 1) -> str:
        """
        Returns the trading date 'steps' days after the given date.
        """
        idx = bisect.bisect_left(self.trading_days, date)
        new_idx = idx + steps
        if new_idx < 0 or new_idx >= len(self.trading_days):
            raise IndexError("Date navigation out of bounds.")
        return self.trading_days[new_idx]

    def previous(self, date: str, steps: int = 1) -> str:
        """
        Returns the trading date 'steps' days before the given date.
        """
        idx = bisect.bisect_left(self.trading_days, date)
        new_idx = idx - steps
        if new_idx < 0 or new_idx >= len(self.trading_days):
            raise IndexError("Date navigation out of bounds.")
        return self.trading_days[new_idx]

    def current_business_date(self, hour: int = 0) -> str:
        """
        Returns today's business date based on current time.
        If before specified hour and today is a business date, returns previous business date.
        """
        today = datetime.now().strftime("%Y-%m-%d")
        current_hour = datetime.now().hour

        # Find the insertion point for today
        idx = bisect.bisect_left(self.trading_days, today)

        # Check if today is a business day
        if idx >= 0 and self.trading_days[idx] == today:
            # If it's before the specified hour, return the previous business date
            if current_hour < hour:
                return self.trading_days[max(0, idx - 1)]
            else:
                return today

        # If today is not a business day, return the previous business day
        if idx >= 0:
            return self.trading_days[idx]

        # If the list is empty or today is before the first trading day
        raise ValueError("No valid business date found.")
