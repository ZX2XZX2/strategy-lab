import pytest
from strategy_lab.backtest.runner import BacktestRunner

class DummyCalendar:
    def date_range(self, start_date, end_date):
        return [start_date]

class DummyLoader:
    def load_eod(self, ticker, as_of_date):
        return None
    def load_intraday(self, ticker, start_date, end_date):
        return None
    def get_all_tickers(self):
        return []

class DummySelector:
    def select(self, as_of_date):
        return []

class DummyTrader:
    def trade(self, ticker, as_of_date):
        return []

def test_runner_runs():
    runner = BacktestRunner(
        calendar=DummyCalendar(),
        loader=DummyLoader(),
        selector=DummySelector(),
        trader=DummyTrader()
    )

    result = runner.run_day("2024-01-01")
    assert isinstance(result, dict)