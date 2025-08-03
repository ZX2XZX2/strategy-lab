import argparse
from datetime import datetime
from typing import List

import matplotlib.pyplot as plt
import polars as pl

from strategy_lab.data.loader import DataLoader
from strategy_lab.plotting import plot_candlestick
from strategy_lab.selection.support_resistance import PivotArea, detect_areas
from strategy_lab.utils.trading_calendar import TradingCalendar


def _adjust_dates(calendar: TradingCalendar, start: str, end: str, data_type: str) -> tuple[str, str]:
    """Adjust provided start and end to valid business days.

    For intraday data, missing time components are filled with default
    start and end times.
    """
    if data_type == "intraday":
        start_parts = start.split()
        start_date = start_parts[0]
        start_time = start_parts[1] if len(start_parts) > 1 else "09:30:00"
        if start_date not in calendar.trading_days:
            start_date = calendar.previous(start_date)
        start = f"{start_date} {start_time}"

        end_parts = end.split()
        end_date = end_parts[0]
        end_time = end_parts[1] if len(end_parts) > 1 else "15:55:00"
        if end_date not in calendar.trading_days:
            end_date = calendar.previous(end_date)
        end = f"{end_date} {end_time}"
    else:
        if start not in calendar.trading_days:
            start = calendar.previous(start)
        if end not in calendar.trading_days:
            end = calendar.previous(end)
    return start, end


def _load_data(loader: DataLoader, data_type: str, ticker: str, start: str, end: str) -> pl.DataFrame:
    """Load data for the ticker between start and end."""
    if data_type == "intraday":
        df = loader.load_intraday(ticker, start.split()[0], end.split()[0])
        start_dt = datetime.strptime(start, "%Y-%m-%d %H:%M:%S")
        end_dt = datetime.strptime(end, "%Y-%m-%d %H:%M:%S")
        return df.filter((pl.col("timestamp") >= start_dt) & (pl.col("timestamp") <= end_dt))
    return loader.load_eod(ticker, start_date=start, end_date=end)


def _draw_sr_areas(fig: plt.Figure, areas: List[PivotArea]) -> None:
    """Overlay support/resistance areas on an existing figure."""
    ax = fig.axes[0]
    for area in areas:
        ax.axhspan(area.lower / 100, area.upper / 100, color="gold", alpha=0.2)


def plot_stock(
    ticker: str,
    data_type: str,
    start: str,
    end: str,
    loader: DataLoader | None = None,
    calendar: TradingCalendar | None = None,
    sr_areas: List[PivotArea] | None = None,
    show: bool = True,
) -> plt.Figure:
    """Plot candlestick chart for a ticker.

    When ``sr_areas`` is provided, the zones are drawn on top of the
    candlestick chart.
    """
    if calendar is None:
        calendar = TradingCalendar()
    start, end = _adjust_dates(calendar, start, end, data_type)

    if loader is None:
        loader = DataLoader(calendar=calendar)

    df = _load_data(loader, data_type, ticker, start, end)
    fig = plot_candlestick(df, start, end)
    if sr_areas:
        _draw_sr_areas(fig, sr_areas)
    if show:
        plt.show()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot candlestick charts.")
    parser.add_argument("ticker", type=str, help="Stock ticker symbol")
    parser.add_argument("data_type", choices=["eod", "intraday"], help="Data type")
    parser.add_argument("start", type=str, help="Start date or datetime")
    parser.add_argument("end", type=str, help="End date or datetime")
    parser.add_argument(
        "--sr",
        action="store_true",
        help="Overlay support/resistance areas",
    )
    parser.add_argument(
        "--sr-factor",
        type=float,
        default=3.0,
        help="JL penetration factor",
    )
    parser.add_argument(
        "--sr-threshold",
        type=float,
        default=1.0,
        help="Clustering threshold in dollars",
    )
    parser.add_argument(
        "--sr-buffer",
        type=float,
        default=0.5,
        help="Buffer above/below pivot price in dollars",
    )
    args = parser.parse_args()

    sr_areas = None
    if args.sr:
        threshold = int(args.sr_threshold * 100)
        buffer = int(args.sr_buffer * 100)
        sr_areas = detect_areas(
            args.ticker,
            args.start.split()[0],
            args.end.split()[0],
            args.end,
            args.sr_factor,
            threshold,
            buffer,
            data_type=args.data_type,
        )

    plot_stock(
        args.ticker,
        args.data_type,
        args.start,
        args.end,
        sr_areas=sr_areas,
    )


if __name__ == "__main__":
    main()
