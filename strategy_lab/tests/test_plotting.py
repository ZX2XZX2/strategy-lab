import polars as pl
from strategy_lab.plotting.charts import plot_candlestick


def test_plot_daily_chart():
    df = pl.DataFrame({
        "date": ["2024-01-01", "2024-01-02"],
        "open": [1, 2],
        "high": [1.5, 2.5],
        "low": [0.5, 1.5],
        "close": [1.2, 2.0],
        "volume": [100, 200],
    })
    fig = plot_candlestick(df, "2024-01-01", "2024-01-02")
    assert fig is not None


def test_plot_intraday_chart():
    df = pl.DataFrame({
        "timestamp": ["2024-01-01 10:00:00", "2024-01-01 15:00:00"],
        "open": [1, 2],
        "high": [1.5, 2.5],
        "low": [0.5, 1.5],
        "close": [1.2, 2],
        "volume": [100, 200],
    })
    fig = plot_candlestick(df, "2024-01-01 09:00:00", "2024-01-01 16:00:00")
    assert fig is not None
