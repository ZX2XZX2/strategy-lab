from datetime import datetime, date

import polars as pl
from strategy_lab.scripts.plot_stock import plot_stock
from strategy_lab.utils.trading_calendar import adjust_calc_dt, adjust_start_end


class DummyCalendar:
    def __init__(self):
        self.trading_days = ["2024-01-02", "2024-01-03"]

    def previous(self, date: str) -> str:
        for d in reversed(self.trading_days):
            if d < date:
                return d
        return self.trading_days[0]


class DummyLoader:
    def load_eod(self, ticker, start_date=None, end_date=None, **kwargs):
        return pl.DataFrame(
            {
                "date": [date(2024, 1, 2), date(2024, 1, 3)],
                "open": [1, 2],
                "high": [1.5, 2.5],
                "low": [0.5, 1.5],
                "close": [1.2, 2.0],
                "volume": [100, 200],
            }
        )

    def load_intraday(self, ticker, start_date, end_date, **kwargs):
        return pl.DataFrame(
            {
                "timestamp": [
                    datetime(2024, 1, 2, 9, 30, 0),
                    datetime(2024, 1, 2, 15, 55, 0),
                ],
                "open": [1, 2],
                "high": [1.5, 2.5],
                "low": [0.5, 1.5],
                "close": [1.2, 2.0],
                "volume": [100, 200],
            }
        )


def test_adjust_dates_intraday():
    cal = DummyCalendar()
    start, end = adjust_start_end(cal, "2024-01-01", "2024-01-01", "intraday")
    assert start == "2024-01-02 09:30:00"
    assert end == "2024-01-02 15:55:00"


def test_adjust_calc_dt_intraday():
    cal = DummyCalendar()
    dt = adjust_calc_dt(cal, "2024-01-01", "intraday")
    assert dt == "2024-01-02 15:55:00"


def test_adjust_calc_dt_eod():
    cal = DummyCalendar()
    dt = adjust_calc_dt(cal, "2024-01-01", "eod")
    assert dt == "2024-01-02"


def test_plot_stock_returns_figure():
    cal = DummyCalendar()
    loader = DummyLoader()
    fig = plot_stock(
        "AAPL",
        "eod",
        "2024-01-01",
        "2024-01-03",
        loader=loader,
        calendar=cal,
        show=False,
    )
    assert fig is not None
