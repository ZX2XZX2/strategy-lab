from .base_trader import BaseTrader
from typing import List
import polars as pl

class ExampleTrader(BaseTrader):
    def trade(self, ticker: str, as_of_date: str) -> List[dict]:
        """
        Simple trading strategy:
        - Buy 100 shares at open price
        - Sell 100 shares at close price
        """
        intraday = self.load_intraday_data(ticker, as_of_date, as_of_date)

        if intraday.height == 0:
            return []

        first = intraday.sort("time").head(1)
        last = intraday.sort("time").tail(1)

        trades = []

        if first.height > 0:
            trades.append({
                "timestamp": f"{as_of_date} {first[0, 'time']}",
                "side": "buy",
                "quantity": 100,
                "price": first[0, "open"]
            })

        if last.height > 0:
            trades.append({
                "timestamp": f"{as_of_date} {last[0, 'time']}",
                "side": "sell",
                "quantity": 100,
                "price": last[0, "close"]
            })

        return trades