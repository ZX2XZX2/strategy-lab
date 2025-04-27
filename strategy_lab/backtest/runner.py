from typing import List, Dict
import polars as pl

class BacktestRunner:
    def __init__(self, calendar, loader, selector, trader):
        self.calendar = calendar
        self.loader = loader
        self.selector = selector
        self.trader = trader

    def run_day(self, as_of_date: str) -> Dict[str, List[dict]]:
        """
        Run backtest for a single day.

        Returns:
            Dict[ticker, List[trades]]
        """
        selected = self.selector.select(as_of_date)
        results = {}
        for ticker in selected:
            trades = self.trader.trade(ticker, as_of_date)
            results[ticker] = trades
        return results

    def run_range(self, start_date: str, end_date: str) -> Dict[str, List[dict]]:
        """
        Run backtest over a date range.

        Returns:
            Dict[ticker, List[trades]]
        """
        all_results = {}
        dates = self.calendar.date_range(start_date, end_date)
        for date in dates:
            daily_results = self.run_day(date)
            for ticker, trades in daily_results.items():
                all_results.setdefault(ticker, []).extend(trades)
        return all_results