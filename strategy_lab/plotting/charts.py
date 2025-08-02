"""Plotting utilities for stock data."""

from __future__ import annotations

from datetime import datetime

import matplotlib.pyplot as plt
import mplfinance as mpf
import polars as pl


def plot_candlestick(df: pl.DataFrame, start_date: str, end_date: str) -> plt.Figure:
    """Plot OHLC candlestick chart between two dates.

    The dataframe must contain ``open``, ``high``, ``low``, ``close`` and
    ``volume`` columns. A ``date`` column indicates daily data while a
    ``timestamp`` column indicates intraday data.
    """
    if "timestamp" in df.columns:
        date_col = "timestamp"
        if df.get_column(date_col).dtype != pl.Datetime:
            df = df.with_columns(
                pl.col(date_col).str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S")
            )
        start = datetime.strptime(start_date, "%Y-%m-%d %H:%M:%S")
        end = datetime.strptime(end_date, "%Y-%m-%d %H:%M:%S")
    elif "date" in df.columns:
        date_col = "date"
        if df.get_column(date_col).dtype != pl.Date:
            df = df.with_columns(
                pl.col(date_col).str.strptime(pl.Date, "%Y-%m-%d")
            )
        start = datetime.strptime(start_date, "%Y-%m-%d").date()
        end = datetime.strptime(end_date, "%Y-%m-%d").date()
    else:
        raise ValueError("DataFrame must contain 'date' or 'timestamp' column")

    df = (
        df.filter((pl.col(date_col) >= start) & (pl.col(date_col) <= end))
        .sort(date_col)
        .rename({date_col: "Date"})
    )
    if df.is_empty():
        raise ValueError("No data available in the specified range")

    pd_df = (
        df.select(["Date", "open", "high", "low", "close", "volume"])
        .to_pandas()
        .set_index("Date")
    )

    pd_df[["open", "high", "low", "close"]] = pd_df[["open", "high", "low", "close"]] / 100

    fig, _ = mpf.plot(
        pd_df,
        type="candle",
        volume=True,
        style="yahoo",
        warn_too_much_data=2000,
        returnfig=True,
    )

    return fig
