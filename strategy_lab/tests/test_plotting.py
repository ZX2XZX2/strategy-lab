from datetime import datetime, date

import matplotlib.pyplot as plt
import mplfinance as mpf

import polars as pl
from strategy_lab.plotting.charts import plot_candlestick


def test_plot_daily_chart(monkeypatch):
    captured = {}

    def fake_plot(data, *args, **kwargs):
        captured["data"] = data
        captured["kwargs"] = kwargs
        return plt.figure(), []

    monkeypatch.setattr(mpf, "plot", fake_plot)

    df = pl.DataFrame(
        {
            "date": [date(2024, 1, 1), date(2024, 1, 2)],
            "open": [1, 2],
            "high": [1.5, 2.5],
            "low": [0.5, 1.5],
            "close": [1.2, 2.0],
            "volume": [100, 200],
        }
    )
    fig = plot_candlestick(df, "2024-01-01", "2024-01-02")
    assert fig is not None
    pd_df = captured["data"]
    assert pd_df["open"].iloc[0] == 0.01
    assert captured["kwargs"]["warn_too_much_data"] == 2000


def test_plot_intraday_chart(monkeypatch):
    captured = {}

    def fake_plot(data, *args, **kwargs):
        captured["data"] = data
        captured["kwargs"] = kwargs
        return plt.figure(), []

    monkeypatch.setattr(mpf, "plot", fake_plot)

    df = pl.DataFrame(
        {
            "timestamp": [
                datetime(2024, 1, 1, 10, 0, 0),
                datetime(2024, 1, 1, 15, 0, 0),
            ],
            "open": [1, 2],
            "high": [1.5, 2.5],
            "low": [0.5, 1.5],
            "close": [1.2, 2],
            "volume": [100, 200],
        }
    )
    fig = plot_candlestick(
        df, "2024-01-01 09:00:00", "2024-01-01 16:00:00"
    )
    assert fig is not None
    pd_df = captured["data"]
    assert pd_df["open"].iloc[0] == 0.01
    assert captured["kwargs"]["warn_too_much_data"] == 2000
    