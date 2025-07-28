from __future__ import annotations

from dataclasses import dataclass
from typing import List

from datetime import datetime

import polars as pl

from strategy_lab.data.loader import DataLoader
from .jl_pivotal_points import StxJL, JLPivot


@dataclass
class PivotArea:
    lower: float
    upper: float
    pivots: List[JLPivot]
    area_type: str  # 'support' or 'resistance'


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
    """Group pivots into support/resistance areas."""
    clusters: List[dict] = []
    for p in pivots:
        added = False
        for cl in clusters:
            if abs(p.price - cl["avg"] ) <= threshold:
                cl["pivots"].append(p)
                cl["avg"] = sum(x.price for x in cl["pivots"]) / len(cl["pivots"])
                added = True
                break
        if not added:
            clusters.append({"avg": p.price, "pivots": [p]})

    areas: List[PivotArea] = []
    for cl in clusters:
        prices = [p.price for p in cl["pivots"]]
        upper = max(prices) + buffer
        lower = min(prices) - buffer
        up_states = [StxJL.NRa, StxJL.UT]
        dn_states = [StxJL.NRe, StxJL.DT]
        up_count = sum(1 for p in cl["pivots"] if p.state in up_states)
        dn_count = sum(1 for p in cl["pivots"] if p.state in dn_states)
        area_type = "resistance" if up_count >= dn_count else "support"
        areas.append(PivotArea(lower, upper, cl["pivots"], area_type))
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
) -> List[PivotArea]:
    loader = DataLoader()
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

    threshold = int(args.threshold * 100)
    buffer = int(args.area * 100)
    areas = detect_areas(
        args.stk,
        args.start_date,
        args.end_date,
        args.dt,
        args.factor,
        threshold,
        buffer,
        data_type=args.data_type,
    )

    for area in areas:
        print(f"{area.area_type.capitalize()} {area.lower/100:.2f} - {area.upper/100:.2f}")
        for p in area.pivots:
            print(f"  {p.dt} state:{p.state} price:{p.price/100:.2f}")
