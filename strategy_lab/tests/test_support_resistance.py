from strategy_lab.selection.support_resistance import group_pivots
from strategy_lab.selection.jl_pivotal_points import StxJL, JLPivot


def test_group_pivots_simple():
    pivots = [
        JLPivot("2024-01-01", StxJL.NRe, 100, 1),
        JLPivot("2024-01-02", StxJL.NRe, 101, 1),
        JLPivot("2024-01-03", StxJL.NRa, 102, 1),
        JLPivot("2024-01-04", StxJL.UT, 103, 1),
        JLPivot("2024-01-05", StxJL.NRa, 150, 1),
    ]
    areas = group_pivots(pivots, threshold=3, buffer=1)
    assert len(areas) == 2
    types = {area.area_type for area in areas}
    assert {"support", "resistance"} == types
    for area in areas:
        if area.area_type == "support":
            assert len(area.pivots) == 2
            assert area.lower == 99
            assert area.upper == 102
        elif area.area_type == "resistance":
            assert len(area.pivots) == 2

