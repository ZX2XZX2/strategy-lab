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
    # Assume dummy folders for EOD and intraday exist
    return DataLoader(
        eod_path=tmp_path / "eod",
        intraday_path=tmp_path / "intraday",
        splits_path=tmp_path / "splits",
        calendar=DummyCalendar(),
    )

def test_load_eod(dummy_loader):
    # Currently will fail gracefully because no file exists
    try:
        dummy_loader.load_eod("AAPL", "2024-01-01")
    except FileNotFoundError:
        pass

def test_load_intraday(dummy_loader):
    try:
        dummy_loader.load_intraday("AAPL", "2024-01-01", "2024-01-01")
    except FileNotFoundError:
        pass