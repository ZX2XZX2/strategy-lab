from strategy_lab.selection.support_resistance import group_pivots
from strategy_lab.selection.jl_pivotal_points import StxJL, JLPivot


def test_group_pivots_simple():
    pivots = [
        JLPivot("2024-01-01", StxJL.NRa, 100, 1),
        JLPivot("2024-01-02", StxJL.NRe, 102, 1),
    ]
    areas = group_pivots(pivots, threshold=5, buffer=2)
    assert len(areas) == 1
    area = areas[0]
    assert area.lower == 98
    assert area.upper == 104
    assert len(area.pivots) == 2
