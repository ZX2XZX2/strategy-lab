import pytest
from strategy_lab.data.loader import DataLoader

class DummyAdjuster:
    def adjust(self, df, as_of_date):
        return df

class DummyCalendar:
    def current_business_date(self, hour=20):
        return "2024-01-01"
    def date_range(self, start_date: str, end_date: str):
        return [start_date]  # simple fake range (one date)

@pytest.fixture
def dummy_loader(tmp_path):
    # Provide a DataLoader that uses the dummy calendar and temporary directories
    loader = DataLoader(calendar=DummyCalendar())
    loader.eod_path = tmp_path / "eod"
    loader.intraday_path = tmp_path / "intraday"
    loader.splits_path = tmp_path / "splits.parquet"
    import polars as pl
    pl.DataFrame({"ticker": ["AAPL"], "date": ["2024-01-02"], "ratio": [0.5]}).write_parquet(loader.splits_path)
    return loader

def test_load_eod(dummy_loader):
    records = [
        {"stk": "AAPL", "dt": "2024-01-01", "o": 1, "hi": 2, "lo": 0.5, "c": 1.5, "v": 10, "oi": 0}
    ]
    df = dummy_loader.load_eod(
        "AAPL",
        "2024-01-01",
        data_source="dict",
        records=records,
    )
    assert not df.is_empty()
    assert set(df.columns) == {
        "ticker",
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "open_interest",
    }

def test_load_intraday(dummy_loader):
    records = [
        {
            "stk": "AAPL",
            "dt": "2024-01-01 10:00:00",
            "o": 1,
            "hi": 2,
            "lo": 0.5,
            "c": 1.5,
            "v": 10,
        }
    ]
    df = dummy_loader.load_intraday(
        "AAPL",
        "2024-01-01",
        "2024-01-01",
        data_source="dict",
        records=records,
    )
    assert not df.is_empty()
    assert "timestamp" in df.columns

def test_add_functions(dummy_loader):
    import polars as pl

    eod_first = [
        {"stk": "AAPL", "dt": "2024-01-01", "o": 10, "hi": 10, "lo": 10, "c": 10, "v": 100, "oi": 0}
    ]
    df = dummy_loader.add_eod(pl.DataFrame(), "AAPL", "2024-01-01", data_source="dict", records=eod_first)

    eod_second = [
        {"stk": "AAPL", "dt": "2024-01-03", "o": 20, "hi": 20, "lo": 20, "c": 20, "v": 200, "oi": 0}
    ]
    df = dummy_loader.add_eod(df, "AAPL", "2024-01-03", start_date="2024-01-03", end_date="2024-01-03", data_source="dict", records=eod_second)
    assert df.sort("date").get_column("open").to_list() == [5, 20]

    intraday_first = [
        {"stk": "AAPL", "dt": "2024-01-01 10:00:00", "o": 10, "hi": 10, "lo": 10, "c": 10, "v": 100}
    ]
    df2 = dummy_loader.add_intraday(pl.DataFrame(), "AAPL", "2024-01-01", "2024-01-01", data_source="dict", records=intraday_first)

    intraday_second = [
        {"stk": "AAPL", "dt": "2024-01-03 10:00:00", "o": 20, "hi": 20, "lo": 20, "c": 20, "v": 200}
    ]
    df2 = dummy_loader.add_intraday(df2, "AAPL", "2024-01-03", "2024-01-03", data_source="dict", records=intraday_second)
    assert df2.sort("timestamp").get_column("open").to_list() == [5, 20]
