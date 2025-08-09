from __future__ import annotations

from dataclasses import dataclass
from typing import List

from datetime import datetime, timedelta

import polars as pl

from strategy_lab.data.loader import DataLoader
from strategy_lab.selection.jl_pivotal_points import StxJL, JLPivot
from strategy_lab.utils.trading_calendar import TradingCalendar, adjust_dt


@dataclass
class PivotArea:
    lower: float
    upper: float
    pivots: List[JLPivot]


def extract_pivots(jl: StxJL) -> List[JLPivot]:
    """Return all pivotal points from a ``StxJL`` instance."""
    pivs: List[JLPivot] = []
    for rec in jl.jl_recs[1:]:
        if rec[jl.col_ix["pivot2"]] == 1:
            pivs.append(
                JLPivot(
                    str(rec[jl.col_ix["dt"]]),
                    rec[jl.col_ix["state2"]],
                    rec[jl.col_ix["price2"]],
                    rec[jl.col_ix["rg"]],
                )
            )
        if rec[jl.col_ix["pivot"]] == 1:
            pivs.append(
                JLPivot(
                    str(rec[jl.col_ix["dt"]]),
                    rec[jl.col_ix["state"]],
                    rec[jl.col_ix["price"]],
                    rec[jl.col_ix["rg"]],
                )
            )
    return pivs


def group_pivots(pivots: List[JLPivot], threshold: float, buffer: float) -> List[PivotArea]:
    """Group pivots into combined support/resistance areas.

    Pivots of both types are clustered together when they fall within
    ``threshold`` of the running average price of an existing cluster. Clusters
    containing only a single pivot are ignored.
    """

    clusters: List[List[JLPivot]] = []
    for p in pivots:
        for cl in clusters:
            avg = sum(x.price for x in cl) / len(cl)
            if abs(p.price - avg) <= threshold:
                cl.append(p)
                break
        else:
            clusters.append([p])

    areas: List[PivotArea] = []
    for cl in clusters:
        if len(cl) <= 1:
            continue
        prices = [p.price for p in cl]
        areas.append(PivotArea(min(prices) - buffer, max(prices) + buffer, cl))
    return areas


def detect_areas(
    ticker: str,
    start_date: str,
    end_date: str,
    calc_dt: str,
    factor: float,
    threshold: float,
    buffer: float,
    data_type: str = "eod",
    calendar: TradingCalendar | None = None,
) -> List[PivotArea]:
    if calendar is None:
        calendar = TradingCalendar()
    start_date = adjust_dt(calendar, start_date, is_start=True)
    end_date = adjust_dt(calendar, end_date)
    calc_dt = adjust_dt(calendar, calc_dt, data_type == "intraday")
    loader = DataLoader(calendar=calendar)
    start_dt = datetime.fromisoformat(start_date).date()
    end_dt = datetime.fromisoformat(end_date).date()

    if data_type == "eod":
        df = loader.load_eod(
            ticker,
            start_date=start_dt.isoformat(),
            end_date=end_dt.isoformat(),
            as_of_date=end_dt.isoformat(),
        ).rename({"date": "dt", "high": "hi", "low": "lo", "close": "c"})
    else:
        df = loader.load_intraday(
            ticker,
            start_dt.isoformat(),
            end_dt.isoformat(),
            as_of_date=end_dt.isoformat(),
        ).rename({"timestamp": "dt", "high": "hi", "low": "lo", "close": "c"})

    jl = StxJL(df, factor)
    if df.get_column("dt").dtype == pl.Date:
        dt_val = datetime.fromisoformat(calc_dt).date()
    elif df.get_column("dt").dtype == pl.Datetime:
        dt_val = datetime.fromisoformat(calc_dt)
    else:
        dt_val = calc_dt

    jl.jl(dt_val)
    pivots = extract_pivots(jl)
    return group_pivots(pivots, threshold, buffer)


def _pivots_and_areas_from_df(
    df: pl.DataFrame,
    factor: float,
    threshold: float,
    buffer: float,
) -> tuple[List[JLPivot], List[PivotArea]]:
    """Helper that derives pivots and areas from a DataFrame.

    The JL algorithm requires a minimum lookback window.  For small intraday
    samples we cap the window to the available data length to avoid index
    errors.
    """

    jl = StxJL(df, factor, w=min(20, len(df)))
    dt_val = df.get_column("dt")[-1]
    jl.jl(dt_val)
    pivots = extract_pivots(jl)
    areas = group_pivots(pivots, threshold, buffer)
    return pivots, areas


def detect_intraday_high_close_signals(
    ticker: str,
    start_date: str,
    n_days: int,
    factor: float,
    threshold: float,
    buffer: float,
    calendar: TradingCalendar | None = None,
) -> List[str]:
    """Scan intraday data for high-close signals around support/resistance.

    The first ``n_days - 1`` days are used to establish support/resistance
    areas.  The intraday bars of the final day are processed sequentially and
    the areas are recalculated after each bar.  When the most recent pivot of
    type ``DT``/``NRe``/``SRe`` falls inside a support/resistance area and at
    least two high-close bars have printed since that pivot, a signal is
    emitted.

    Parameters are deliberately lightweight to keep the function suitable for
    testing and experimentation.  It returns a list of ISO formatted timestamps
    at which signals were triggered.
    """

    if calendar is None:
        calendar = TradingCalendar()

    start_dt = datetime.fromisoformat(start_date)
    end_dt = start_dt + timedelta(days=n_days - 1)

    loader = DataLoader(calendar=calendar)
    df = loader.load_intraday(
        ticker,
        start_dt.date().isoformat(),
        end_dt.date().isoformat(),
    ).rename({"timestamp": "dt", "high": "hi", "low": "lo", "close": "c"})

    if df.is_empty():
        return []

    if df.get_column("dt").dtype != pl.Datetime:
        df = df.with_columns(pl.col("dt").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S"))

    last_day = end_dt.date()
    df_prev = df.filter(pl.col("dt").dt.date() < last_day)
    df_curr = df.filter(pl.col("dt").dt.date() == last_day)

    signals: List[str] = []
    last_signal_pivot: str | None = None

    for i in range(len(df_curr)):
        running = pl.concat([df_prev, df_curr.slice(0, i + 1)])
        pivots, areas = _pivots_and_areas_from_df(running, factor, threshold, buffer)

        trg_pivots = [p for p in pivots if p.state in (StxJL.DT, StxJL.NRe, StxJL.SRe)]
        if not trg_pivots:
            continue
        last_pivot = trg_pivots[-1]

        if last_signal_pivot == last_pivot.dt:
            continue

        in_area = any(area.lower <= last_pivot.price <= area.upper for area in areas)
        if not in_area:
            continue

        pivot_dt = datetime.fromisoformat(last_pivot.dt)
        bars_since = running.filter(pl.col("dt") > pivot_dt)
        if bars_since.is_empty():
            continue

        high_close_count = (
            (bars_since["c"] - bars_since["lo"] >= 0.75 * (bars_since["hi"] - bars_since["lo"]))
            .sum()
        )
        if int(high_close_count) >= 2:
            signals.append(str(df_curr[i, "dt"]))
            last_signal_pivot = last_pivot.dt

    return signals


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Identify support/resistance areas from JL pivots",
    )
    parser.add_argument("--stk", required=True)
    parser.add_argument("--start_date", required=True)
    parser.add_argument("--end_date", required=True)
    parser.add_argument("--data_type", choices=["eod", "intraday"], required=True)
    parser.add_argument("--dt", required=True)
    parser.add_argument("--factor", type=float, required=True)
    parser.add_argument("--threshold", type=float, required=True, help="Clustering threshold in dollars")
    parser.add_argument("--area", type=float, required=True, help="Area buffer above/below pivot price in dollars")
    args = parser.parse_args()

    cal = TradingCalendar()
    start = adjust_dt(cal, args.start_date, args.data_type == "intraday", is_start=True)
    end = adjust_dt(cal, args.end_date, args.data_type == "intraday")

    threshold = int(args.threshold * 100)
    buffer = int(args.area * 100)
    areas = detect_areas(
        args.stk,
        start,
        end,
        adjust_dt(cal, args.dt, args.data_type == "intraday"),
        args.factor,
        threshold,
        buffer,
        data_type=args.data_type,
    )

    for area in areas:
        print(f"Area {area.lower/100:.2f} - {area.upper/100:.2f}")
        for p in area.pivots:
            print(f"  {p.dt} state:{p.state} price:{p.price/100:.2f}")
