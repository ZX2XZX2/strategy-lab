from datetime import datetime

import polars as pl

from strategy_lab.selection.support_resistance import (
    group_pivots,
    detect_intraday_high_close_signals,
    PivotArea,
)
from strategy_lab.selection.jl_pivotal_points import StxJL, JLPivot
from strategy_lab.data.loader import DataLoader


def test_group_pivots_simple():
    pivots = [
        JLPivot("2024-01-01", StxJL.NRe, 100, 1),
        JLPivot("2024-01-02", StxJL.NRe, 101, 1),
        JLPivot("2024-01-03", StxJL.NRa, 102, 1),
        JLPivot("2024-01-04", StxJL.UT, 103, 1),
        JLPivot("2024-01-05", StxJL.NRa, 150, 1),
    ]
    areas = group_pivots(pivots, threshold=3, buffer=1)
    assert len(areas) == 1
    area = areas[0]
    assert len(area.pivots) == 4
    assert area.lower == 99
    assert area.upper == 104


def test_detect_intraday_high_close_signals(monkeypatch):
    # Create simple intraday dataset for two days
    data = [
        {"timestamp": datetime(2024, 1, 1, 9, 30), "open": 100, "high": 101, "low": 99, "close": 100, "volume": 100, "time": "09:30:00"},
        {"timestamp": datetime(2024, 1, 2, 9, 30), "open": 100, "high": 101, "low": 99, "close": 100, "volume": 100, "time": "09:30:00"},
        {"timestamp": datetime(2024, 1, 2, 9, 35), "open": 101, "high": 102, "low": 100, "close": 101.8, "volume": 100, "time": "09:35:00"},
        {"timestamp": datetime(2024, 1, 2, 9, 40), "open": 102, "high": 103, "low": 101, "close": 102.8, "volume": 100, "time": "09:40:00"},
    ]
    df = pl.DataFrame(data)

    def mock_load_intraday(self, ticker, start_date, end_date):
        return df

    monkeypatch.setattr(DataLoader, "load_intraday", mock_load_intraday)

    pivot = JLPivot("2024-01-02 09:30:00", StxJL.NRe, 100, 1)

    def mock_extract_pivots(_jl):
        return [pivot]

    def mock_group_pivots(_pivots, threshold, buffer):
        return [PivotArea(99, 101, _pivots)]

    monkeypatch.setattr("strategy_lab.selection.support_resistance.extract_pivots", mock_extract_pivots)
    monkeypatch.setattr("strategy_lab.selection.support_resistance.group_pivots", mock_group_pivots)

    class DummyCal:
        trading_days = ["2024-01-01", "2024-01-02"]

    signals = detect_intraday_high_close_signals(
        "XYZ", "2024-01-01", 2, 1.0, 1, 1, calendar=DummyCal()
    )
    assert signals == ["2024-01-02 09:40:00"]

