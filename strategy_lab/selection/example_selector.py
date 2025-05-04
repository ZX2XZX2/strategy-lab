from .base_selector import BaseSelector
from typing import List
import polars as pl

class ExampleSelector(BaseSelector):
    def select(self, as_of_date: str) -> List[str]:
        """
        Simple selector: Pick top 10 stocks by volume from previous EOD close.
        """
        # Assume you already have a list of all tickers somehow
        all_tickers = self.loader.get_all_tickers()

        eod_data = self.load_eod_data(all_tickers, as_of_date)

        volumes = []
        for ticker, df in eod_data.items():
            latest = df.filter(pl.col("date") == as_of_date)
            if latest.height > 0:
                volumes.append((ticker, latest["volume"].item()))

        # Sort by volume descending
        top_tickers = sorted(volumes, key=lambda x: -x[1])[:10]

        return [ticker for ticker, _ in top_tickers]
