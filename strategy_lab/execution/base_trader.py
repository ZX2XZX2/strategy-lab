from abc import ABC, abstractmethod
from typing import List

class BaseTrader(ABC):
    def __init__(self, calendar, loader):
        self.calendar = calendar
        self.loader = loader

    @abstractmethod
    def trade(self, ticker: str, as_of_date: str) -> List[dict]:
        """
        Defines intraday trades for a single ticker on a given day.

        Returns:
            List[dict]: Each dict contains keys like {"timestamp", "side", "quantity", "price"}.
        """
        pass

    def load_intraday_data(self, ticker: str, start_date: str, end_date: str):
        """
        Utility to load intraday data for a ticker between two dates.
        """
        return self.loader.load_intraday(ticker, start_date, end_date)