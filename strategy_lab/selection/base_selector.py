from abc import ABC, abstractmethod
from typing import List, Set

class BaseSelector(ABC):
    def __init__(self, calendar, loader, etfs: Set[str] = None):
        self.calendar = calendar
        self.loader = loader
        self.etfs = etfs if etfs else set()

    @abstractmethod
    def select(self, as_of_date: str) -> List[str]:
        pass

    def load_eod_data(
        self, tickers: List[str], as_of_date: str, data_source: str = "db"
    ) -> dict:
        data = {}
        for ticker in tickers:
            if ticker in self.etfs:
                continue
            try:
                df = self.loader.load_eod(
                    ticker, as_of_date=as_of_date, data_source=data_source
                )
            except TypeError:
                df = self.loader.load_eod(ticker, as_of_date)
            data[ticker] = df
        return data
