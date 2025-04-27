import pytest
from strategy_lab.execution.example_trader import ExampleTrader

class DummyLoader:
    def load_intraday(self, ticker, start_date, end_date):
        import polars as pl
        return pl.DataFrame({"time": ["09:30", "16:00"], "open": [100], "close": [110]})

class DummyCalendar:
    pass

def test_example_trader():
    trader = ExampleTrader(calendar=DummyCalendar(), loader=DummyLoader())
    trades = trader.trade("AAPL", "2024-01-01")
    assert isinstance(trades, list)
    assert len(trades) == 2