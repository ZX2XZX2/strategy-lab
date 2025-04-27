import pytest
from strategy_lab.selection.example_selector import ExampleSelector

class DummyLoader:
    def get_all_tickers(self):
        return ["AAPL", "MSFT"]
    def load_eod(self, ticker, as_of_date):
        import polars as pl
        return pl.DataFrame({"date": [as_of_date], "volume": [1000000]})

class DummyCalendar:
    pass

def test_example_selector():
    selector = ExampleSelector(calendar=DummyCalendar(), loader=DummyLoader())
    tickers = selector.select("2024-01-01")
    assert isinstance(tickers, list)
    assert len(tickers) <= 10