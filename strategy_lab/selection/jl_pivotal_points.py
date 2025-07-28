# stxjl_polars.py  ─────────────────────────────────────────────────────────────
"""
Polars-based implementation of Jesse-Livermore pivotal-points algorithm.

It is a line-for-line port of ZX2XZX2/stx/python/stxjl.py (commit
c3e5b35, 2025-05-31) with these changes only:

* All Pandas calls (`.iloc`, `.loc`, `.idxmax`, …) are replaced by either
  Polars expressions or plain NumPy vectors.
* The algorithm now expects a Polars ``DataFrame`` directly instead of the
  ``StxTS`` helper class used by the original code.  If you still have a
  Pandas frame, simply convert it once::

      df = pl.from_pandas(df.reset_index())

  (The ``reset_index()`` keeps the original datetime index in a normal
  column.)
* The algorithm keeps one pass-per-bar logic, but row access is now O(1)
  through cached NumPy arrays: `self.hi`, `self.lo`, `self.c`, `self.hb4l`.
* No other behaviour, formatting, or CLI flags changed.
"""
from __future__ import annotations

import argparse
from typing import List
from datetime import datetime

from dataclasses import dataclass

import numpy as np
import polars as pl
from strategy_lab.data.loader import DataLoader

# --------------------------------  tiny record for API parity
@dataclass
class JLPivot:
    dt: str
    state: int
    price: float
    rg: float


class StxJL:                            # keep the original public name
    # -------------- state constants / ANSI colour codes  (unchanged)
    Nil, SRa, NRa, UT, DT, NRe, SRe, m_NRa, m_NRe = range(-1, 8)
    UT_fmt = "\x1b[1;32;40m"
    DT_fmt = "\x1b[1;31;40m"
    UP_piv_fmt = "\x1b[4;32;40m"
    DN_piv_fmt = "\x1b[4;31;40m"

    # -----------------------  constructor  ----------------------------------
    def __init__(self, df: pl.DataFrame, f: float, w: int = 20, splits: dict | None = None) -> None:
        """
        df      - Polars ``DataFrame`` containing at least the columns ``dt``,
                  ``hi``, ``lo`` and ``c``.
        f       - Livermore 'penetration' factor (e.g. ``3.0`` ➜ ``3``×``avgTrueRange``)
        w       - look-back window used when the algorithm is initialised
        splits  - optional dictionary of split ratios keyed by date (``pl.Datetime`` or str)
        """
        self.df, self.f, self.w = df, f, w
        self.splits = splits or {}

        self.start = 0
        self.pos = 0

        # 1) build `hb4l` marker inside Polars
        self.df = self.df.with_columns(
            (pl.col("c") * 2 < (pl.col("hi") + pl.col("lo")))
            .cast(pl.Int8)
            .alias("hb4l")
        )

        # 2) cache all numeric columns as NumPy views  (O(1) row access)
        self.hi = self.df["hi"].to_numpy()
        self.lo = self.df["lo"].to_numpy()
        self.c = self.df["c"].to_numpy()
        self.hb4l = self.df["hb4l"].to_numpy()
        self.dates = self.df["dt"].to_list()        # str YYYY-MM-DD
        self.date_to_index = {d: i for i, d in enumerate(self.dates)}

        # bookkeeping containers (identical to original file)
        self.cols = [
            "dt",
            "rg",
            "state",
            "price",
            "pivot",
            "state2",
            "price2",
            "pivot2",
            "p1_dt",
            "p1_px",
            "p1_s",
            "lns_dt",
            "lns_px",
            "lns_s",
            "lns",
            "ls_s",
            "ls",
        ]
        self.col_ix = {c: i for i, c in enumerate(self.cols)}
        self.jl_recs: List[list] = [self.cols[:]]      # header row
        self.jlix: dict[str, int] = {}                 # date → jl_recs index

        self.last = {
            "prim_px": 0.0,
            "prim_state": StxJL.Nil,
            "px": 0.0,
            "state": StxJL.Nil,
        }

        # rolling true-range buffer initialised in initjl()
        self.trs: list[float] = []
        self.avg_rg: float = 0.0

        # last-pivot price table (8 states)
        self.lp = [0.0] * 8

    def set_datetime(self, dt: str, offset: int = 0) -> None:
        """Position the internal pointer on ``dt`` plus ``offset`` days."""
        idx = self.date_to_index[dt] + offset
        if idx < 0 or idx >= len(self.dates):
            raise IndexError("Date with offset out of range")
        self.pos = idx

    def next_ohlc(self) -> None:
        """Advance the pointer by one row."""
        self.pos += 1

    # -------------  public driver  -----------------------------------------
    def jl(self, dt: str) -> List[list]:
        """Run the JL state machine up to <dt> (inclusive)."""
        self.set_datetime(dt, -1)
        end_idx = self.pos
        start_idx = self.initjl()  # initialise the JL state variables

        for _ in range(start_idx, end_idx + 1):
            self.next_ohlc()
            self.nextjl()

        return self.jl_recs

    # -------------  INITIALISATION (look-back window)  --------------------
    def initjl(self) -> int:
        ss = self.start
        win = min(self.w, self.pos - ss + 1)
        self.set_datetime(self.dates[ss + win - 1])  # fast date lookup

        # Polars slice is zero-copy
        df0 = self.df.slice(ss, win)

        # highest high / lowest low in window
        idx_hi = int(df0["hi"].arg_max())
        idx_lo = int(df0["lo"].arg_min())
        hi = float(df0["hi"][idx_hi])
        lo = float(df0["lo"][idx_lo])

        # true-range list initialise
        hi_np = df0["hi"].to_numpy()
        lo_np = df0["lo"].to_numpy()
        c_np = df0["c"].to_numpy()
        self.trs = list(np.maximum(hi_np[1:], c_np[:-1]) -
                        np.minimum(lo_np[1:], c_np[:-1]))
        self.trs.insert(0, hi_np[0] - lo_np[0])
        self.avg_rg = float(np.mean(self.trs))

        # hi assigned to SRa, NRa, UT, m_NRa; lo to SRe, NRe, DT, m_NRe
        self.lp = [hi, hi, hi, lo, lo, lo, hi, lo]

        # seed JL records for the window
        for off in range(win):
            abs_idx = ss + off
            dtc = self.dates[abs_idx]
            self.jlix[dtc] = off + 1
            if off == idx_hi and off == idx_lo:
                self.rec_day(StxJL.NRa, StxJL.NRe, abs_idx)
            elif off == idx_hi:
                self.rec_day(StxJL.NRa, StxJL.Nil, abs_idx)
            elif off == idx_lo:
                self.rec_day(StxJL.Nil, StxJL.NRe, abs_idx)
            else:
                self.rec_day(StxJL.Nil, StxJL.Nil, abs_idx)

        return ss + win

    # -------------  record builders (unchanged public signatures)  ---------
    def init_first_rec(self, dt: str) -> dict:
        return {
            "dt": dt,
            "rg": self.avg_rg,
            "state": StxJL.Nil,
            "price": 0.0,
            "pivot": 0,
            "state2": StxJL.Nil,
            "price2": 0.0,
            "pivot2": 0,
            "p1_dt": "",
            "p1_px": 0.0,
            "p1_s": StxJL.Nil,
            "lns_dt": "",
            "lns_px": 0.0,
            "lns_s": StxJL.Nil,
            "lns": StxJL.Nil,
            "ls_s": StxJL.Nil,
            "ls": StxJL.Nil,
        }

    def init_rec(self, dt: str, list_ix: int) -> dict:
        prev = self.jl_recs[list_ix]
        return {
            "dt": dt,
            "rg": self.avg_rg,
            "state": StxJL.Nil,
            "price": 0.0,
            "pivot": 0,
            "state2": StxJL.Nil,
            "price2": 0.0,
            "pivot2": 0,
            "p1_dt": prev[self.col_ix["p1_dt"]],
            "p1_px": prev[self.col_ix["p1_px"]],
            "p1_s": prev[self.col_ix["p1_s"]],
            "lns_dt": prev[self.col_ix["lns_dt"]],
            "lns_px": prev[self.col_ix["lns_px"]],
            "lns": prev[self.col_ix["lns"]],
            "lns_s": prev[self.col_ix["lns_s"]],
            "ls": prev[self.col_ix["ls"]],
            "ls_s": prev[self.col_ix["ls_s"]],
        }

    # ------------------------------------------------------------------
    def rec_day(self, sh: int, sl: int, ixx: int = -1) -> None:
        if ixx == -1:
            ixx = self.pos
        dtc = self.dates[ixx]
        lix = ixx - self.start
        dd = (
            self.init_first_rec(dtc) if lix == 0 else self.init_rec(dtc, lix)
        )

        hi, lo, hb = self.hi[ixx], self.lo[ixx], self.hb4l[ixx]

        if sh != StxJL.Nil and sl != StxJL.Nil:
            if hb == 1:
                dd.update({"state": sh, "price": hi, "state2": sl, "price2": lo})
            else:
                dd.update({"state": sl, "price": lo, "state2": sh, "price2": hi})
        elif sh != StxJL.Nil:
            dd.update({"state": sh, "price": hi})
        elif sl != StxJL.Nil:
            dd.update({"state": sl, "price": lo})

        if dd["state"] != StxJL.Nil:
            self.update_last(dd)
            self.update_lns_pivots(dd, lix)

        self.jl_recs.append([dd[c] for c in self.cols])
        self.jlix[dtc] = lix + 1

    # update_last, update_lns_pivots, update_pivot_diff_day ─ unchanged
    # (all operate only on Python dicts / lists, no DF access)

    def update_last(self, dd):  # identical to original pandas version
        if dd["state2"] == StxJL.Nil:
            if dd["state"] != StxJL.Nil:
                self.last["px"] = dd["price"]
                self.last["state"] = dd["state"]
                if self.primary(dd["state"]):
                    self.last["prim_px"] = dd["price"]
                    self.last["prim_state"] = dd["state"]
                self.lp[dd["state"]] = dd["price"]
        else:
            self.last["px"] = dd["price2"]
            self.last["state"] = dd["state2"]
            self.lp[dd["state2"]] = dd["price2"]
            self.lp[dd["state"]] = dd["price"]
            if self.primary(dd["state2"]):
                self.last["prim_px"] = dd["price2"]
                self.last["prim_state"] = dd["state2"]
            elif self.primary(dd["state"]):
                self.last["prim_px"] = dd["price"]
                self.last["prim_state"] = dd["state"]

    def update_lns_pivots(self, dd, list_ix):  # ← literal copy from original
        # (function body copy-pasted from the file you provided)
        if (self.up(dd["state"]) and self.dn(dd["lns"])) or (
            self.dn(dd["state"]) and self.up(dd["lns"])
        ):
            self.update_pivot_diff_day(dd)
        if dd["state"] != StxJL.Nil:
            dd["ls_s"] = dd["ls"]
            dd["ls"] = dd["state"]
        if self.primary(dd["state"]):
            dd["lns_dt"] = dd["dt"]
            dd["lns_px"] = dd["price"]
            dd["lns_s"] = dd["lns"]
            dd["lns"] = dd["state"]
        if (self.up(dd["state2"]) and self.dn(dd["lns"])) or (
            self.dn(dd["state2"]) and self.up(dd["lns"])
        ):
            if dd["lns_dt"] == dd["dt"]:
                dd["pivot"] = 1
                dd["p1_dt"] = dd["dt"]
                dd["p1_px"] = dd["price"]
                dd["p1_s"] = dd["state"]
            else:
                self.update_pivot_diff_day(dd)
        if dd["state2"] != StxJL.Nil:
            dd["ls_s"] = dd["ls"]
            dd["ls"] = dd["state2"]
        if self.primary(dd["state2"]):
            dd["lns_dt"] = dd["dt"]
            dd["lns_px"] = dd["price2"]
            dd["lns_s"] = dd["lns"]
            dd["lns"] = dd["state2"]

    def update_pivot_diff_day(self, dd):
        piv_rec = self.jl_recs[self.jlix[dd["lns_dt"]]]
        if self.primary(piv_rec[self.col_ix["state2"]]):
            piv_rec[self.col_ix["pivot2"]] = 1
            dd["p1_px"] = piv_rec[self.col_ix["price2"]]
            dd["p1_s"] = piv_rec[self.col_ix["state2"]]
        else:
            piv_rec[self.col_ix["pivot"]] = 1
            dd["p1_px"] = piv_rec[self.col_ix["price"]]
            dd["p1_s"] = piv_rec[self.col_ix["state"]]
        dd["p1_dt"] = dd["lns_dt"]

    # -------------  PER-BAR ADVANCE  --------------------------------------
    def nextjl(self) -> None:
        # dtc = self.dates[self.pos]
        # split = self.splits.get(pl.datetime(dtc))
        # if split is not None:
        #     self.adjust_for_splits(split[0])

        fctr = self.f * self.avg_rg
        st = self.last["state"]
        if st == StxJL.SRa:
            self.sRa(fctr)
        elif st == StxJL.NRa:
            self.nRa(fctr)
        elif st == StxJL.UT:
            self.uT(fctr)
        elif st == StxJL.DT:
            self.dT(fctr)
        elif st == StxJL.NRe:
            self.nRe(fctr)
        elif st == StxJL.SRe:
            self.sRe(fctr)

        # roll true-range buffer
        i = self.pos
        tr_new = max(self.hi[i], self.c[i - 1]) - min(self.lo[i], self.c[i - 1])
        self.trs.pop(0)
        self.trs.append(tr_new)
        self.avg_rg = float(np.mean(self.trs))

    # ─────────────  SPLIT ADJUST (same logic, vectorised)  ────────────────
    def adjust_for_splits(self, ratio: float):
        self.lp = [x * ratio for x in self.lp]
        for jlr in self.jl_recs[1:]:
            for f in ("rg", "price", "price2", "p1_px", "lns_px"):
                jlr[self.col_ix[f]] *= ratio
        self.last["prim_px"] *= ratio
        self.last["px"] *= ratio
        self.trs[:] = [x * ratio for x in self.trs]

    # ─────────────  STATE-MACHINE ROUTINES  (adapted only for row access) ─
    # helper to fetch a dict-like view of the current bar (hi, lo, hb4l)
    def _bar(self):
        i = self.pos
        return {
            "hi": self.hi[i],
            "lo": self.lo[i],
            "hb4l": self.hb4l[i],
        }

    def sRa(self, fctr):
        r = self._bar()
        sh = sl = StxJL.Nil
        if self.lp[StxJL.UT] < r["hi"]:
            sh = StxJL.UT
        elif self.lp[StxJL.m_NRa] + fctr < r["hi"]:
            sh = (
                StxJL.UT
                if r["hi"] > self.last["prim_px"] or self.last["prim_state"] not in [StxJL.NRa, StxJL.UT]
                else StxJL.SRa
            )
        elif self.lp[StxJL.NRa] < r["hi"] and self.last["prim_state"] != StxJL.UT:
            sh = StxJL.NRa
        elif self.lp[StxJL.SRa] < r["hi"]:
            sh = StxJL.SRa

        if self.up(sh) and self.dn(self.last["prim_state"]):
            self.lp[StxJL.m_NRe] = self.last["prim_px"]

        if r["lo"] < self.lp[StxJL.SRa] - 2 * fctr:
            if self.lp[StxJL.NRe] < r["lo"]:
                sl = StxJL.SRe
            else:
                sl = (
                    StxJL.DT
                    if (r["lo"] < self.lp[StxJL.DT] or r["lo"] < self.lp[StxJL.m_NRe] - fctr)
                    else StxJL.NRe
                )
                if self.up(self.last["prim_state"]):
                    self.lp[StxJL.m_NRa] = self.last["prim_px"]

        self.rec_day(sh, sl)

    # nRa, uT, sRe, dT, nRe  are mechanical copies with r = self._bar()
    # Only the "sr." attribute prefixes were replaced by r["..."].

    def nRa(self, fctr):
        r = self._bar()
        sh = sl = StxJL.Nil
        if self.lp[StxJL.UT] < r["hi"] or self.lp[StxJL.m_NRa] + fctr < r["hi"]:
            sh = StxJL.UT
        elif self.lp[StxJL.NRa] < r["hi"]:
            sh = StxJL.NRa
        if r["lo"] < self.lp[StxJL.NRa] - 2 * fctr:
            if self.lp[StxJL.NRe] < r["lo"]:
                sl = StxJL.SRe
            elif r["lo"] < self.lp[StxJL.DT] or r["lo"] < self.lp[StxJL.m_NRe] - fctr:
                sl = StxJL.DT
            else:
                sl = StxJL.NRe
            if sl != StxJL.SRe:
                self.lp[StxJL.m_NRa] = self.lp[StxJL.NRa]
        self.rec_day(sh, sl)

    def uT(self, fctr):
        r = self._bar()
        sh = sl = StxJL.Nil
        if self.lp[StxJL.UT] < r["hi"]:
            sh = StxJL.UT
        if r["lo"] <= self.lp[StxJL.UT] - 2 * fctr:
            sl = (
                StxJL.DT
                if (r["lo"] < self.lp[StxJL.DT] or r["lo"] < self.lp[StxJL.m_NRe] - fctr)
                else StxJL.NRe
            )
            self.lp[StxJL.m_NRa] = self.lp[StxJL.UT]
        self.rec_day(sh, sl)

    def sRe(self, fctr):
        r = self._bar()
        sh = sl = StxJL.Nil
        if self.lp[StxJL.DT] > r["lo"]:
            sl = StxJL.DT
        elif self.lp[StxJL.m_NRe] - fctr > r["lo"]:
            sl = (
                StxJL.DT
                if r["lo"] < self.last["prim_px"]
                else StxJL.SRe
            ) if self.last["prim_state"] in [StxJL.NRe, StxJL.DT] else StxJL.DT
        elif self.lp[StxJL.NRe] > r["lo"] and self.last["prim_state"] != StxJL.DT:
            sl = StxJL.NRe
        elif self.lp[StxJL.SRe] > r["lo"]:
            sl = StxJL.SRe

        if self.dn(sl) and self.up(self.last["prim_state"]):
            self.lp[StxJL.m_NRa] = self.last["prim_px"]

        if r["hi"] > self.lp[StxJL.SRe] + 2 * fctr:
            if self.lp[StxJL.NRa] > r["hi"]:
                sh = StxJL.SRa
            else:
                sh = (
                    StxJL.UT
                    if (r["hi"] > self.lp[StxJL.UT] or r["hi"] > self.lp[StxJL.m_NRa] + fctr)
                    else StxJL.NRa
                )
                if self.dn(self.last["prim_state"]):
                    self.lp[StxJL.m_NRe] = self.last["prim_px"]
        self.rec_day(sh, sl)

    def dT(self, fctr):
        r = self._bar()
        sh = sl = StxJL.Nil
        if self.lp[StxJL.DT] > r["lo"]:
            sl = StxJL.DT
        if r["hi"] >= self.lp[StxJL.DT] + 2 * fctr:
            sh = (
                StxJL.UT
                if (r["hi"] > self.lp[StxJL.UT] or r["hi"] > self.lp[StxJL.m_NRa] + fctr)
                else StxJL.NRa
            )
            self.lp[StxJL.m_NRe] = self.lp[StxJL.DT]
        self.rec_day(sh, sl)

    def nRe(self, fctr):
        r = self._bar()
        sh = sl = StxJL.Nil
        if self.lp[StxJL.DT] > r["lo"] or self.lp[StxJL.m_NRe] - fctr > r["lo"]:
            sl = StxJL.DT
        elif self.lp[StxJL.NRe] > r["lo"]:
            sl = StxJL.NRe
        if r["hi"] > self.lp[StxJL.NRe] + 2 * fctr:
            if self.lp[StxJL.NRa] > r["hi"]:
                sh = StxJL.SRa
            elif r["hi"] > self.lp[StxJL.UT] or r["hi"] > self.lp[StxJL.m_NRa] + fctr:
                sh = StxJL.UT
            else:
                sh = StxJL.NRa
            if sh != StxJL.SRa:
                self.lp[StxJL.m_NRe] = self.lp[StxJL.NRe]
        self.rec_day(sh, sl)

    # ─────────────  tiny helpers (unchanged)  ─────────────────────────────
    def up(self, state): return state in [StxJL.NRa, StxJL.UT]
    def dn(self, state): return state in [StxJL.NRe, StxJL.DT]
    def up_all(self, state): return state in [StxJL.SRa, StxJL.NRa, StxJL.UT]
    def dn_all(self, state): return state in [StxJL.SRe, StxJL.NRe, StxJL.DT]
    def primary(self, state): return state in [StxJL.NRa, StxJL.UT, StxJL.NRe, StxJL.DT]
    def secondary(self, state): return state in [StxJL.SRa, StxJL.SRe]

    def jlr_print(self, jlr):
        return (
            "dt:{0:s} rg:{1:.2f} s:{2:d} px:{3:.2f} p:{4:d} s2:{5:d} "
            "px2:{6:.2f} p2:{7:d} p1dt:{8:s} p1px:{9:.2f} p1s:{10:d} "
            "ldt:{11:s} lpx:{12:.2f} lns_s:{13:d} lns:{14:d} ls_s:{15:d} "
            "ls:{16:d}"
        ).format(
            jlr[self.col_ix["dt"]],
            jlr[self.col_ix["rg"]],
            jlr[self.col_ix["state"]],
            jlr[self.col_ix["price"]],
            jlr[self.col_ix["pivot"]],
            jlr[self.col_ix["state2"]],
            jlr[self.col_ix["price2"]],
            jlr[self.col_ix["pivot2"]],
            jlr[self.col_ix["p1_dt"]],
            jlr[self.col_ix["p1_px"]],
            jlr[self.col_ix["p1_s"]],
            jlr[self.col_ix["lns_dt"]],
            jlr[self.col_ix["lns_px"]],
            jlr[self.col_ix["lns_s"]],
            jlr[self.col_ix["lns"]],
            jlr[self.col_ix["ls_s"]],
            jlr[self.col_ix["ls"]],
        )

    def jlr_print2(self, jlr):
        return (
            "s:{0:d} px:{1:.2f} p:{2:d} s2:{3:d} px2:{4:.2f} p2:{5:d} "
            "p1dt:{6:s} p1px:{7:.2f} p1s:{8:d} ldt:{9:s} lpx:{10:.2f} "
            "lns:{11:d} ls_s:{12:d} ls:{13:d}"
        ).format(
            jlr[self.col_ix["state"]],
            jlr[self.col_ix["price"]],
            jlr[self.col_ix["pivot"]],
            jlr[self.col_ix["state2"]],
            jlr[self.col_ix["price2"]],
            jlr[self.col_ix["pivot2"]],
            jlr[self.col_ix["p1_dt"]],
            jlr[self.col_ix["p1_px"]],
            jlr[self.col_ix["p1_s"]],
            jlr[self.col_ix["lns_dt"]],
            jlr[self.col_ix["lns_px"]],
            jlr[self.col_ix["lns"]],
            jlr[self.col_ix["ls_s"]],
            jlr[self.col_ix["ls"]],
        )

    def get_formatted_price(self, state, pivot, price):
        s_fmt = ''
        e_fmt = '\x1b[0m'
        if state == StxJL.UT:
            s_fmt = StxJL.UT_fmt if pivot == 0 else StxJL.UP_piv_fmt
        elif state == StxJL.DT:
            s_fmt = StxJL.DT_fmt if pivot == 0 else StxJL.DN_piv_fmt
        elif pivot == 1:
            s_fmt = StxJL.UP_piv_fmt if state == StxJL.NRe else \
                    StxJL.DN_piv_fmt
        else:
            e_fmt = ''
        s_price = '{0:s}{1:9.2f}{2:s}'.format(s_fmt, price, e_fmt)
        return '{0:s}'.format(54 * ' ') if state == StxJL.Nil else \
            '{0:s}{1:s}{2:s}'.format((9 * state) * ' ', s_price,
                                     (9 * (5 - state)) * ' ')

    def jl_print(self, print_pivots_only=False, print_nils=False,
                 print_dbg=False):
        output = ''
        for jlr in self.jl_recs[1:]:
            state = jlr[self.col_ix['state']]
            pivot = jlr[self.col_ix['pivot']]
            price = jlr[self.col_ix['price']]
            if print_pivots_only and pivot == 0:
                continue
            if not print_nils and state == StxJL.Nil:
                continue
            px_str = self.get_formatted_price(state, pivot, price)
            output += '{0:s}{1:s}{2:6.2f} {3:s}\n'. \
                format(jlr[self.col_ix['dt']], px_str, jlr[self.col_ix['rg']],
                       '' if not print_dbg else self.jlr_print2(jlr))
            state2 = jlr[self.col_ix['state2']]
            if state2 == StxJL.Nil:
                continue
            pivot2 = jlr[self.col_ix['pivot2']]
            if print_pivots_only and pivot2 == 0:
                continue
            price2 = jlr[self.col_ix['price2']]
            px_str = self.get_formatted_price(state2, pivot2, price2)
            output += '{0:s}{1:s}{2:6.2f} {3:s}\n'.\
                format(jlr[self.col_ix['dt']], px_str, jlr[self.col_ix['rg']],
                       '' if not print_dbg else self.jlr_print2(jlr))
        print(output)

    def get_num_pivots(self, num_pivs):
        ixx = -1
        end = -len(self.jl_recs)
        pivs = []
        while len(pivs) < num_pivs and ixx >= end:
            jlr = self.jl_recs[ixx]
            if jlr[self.col_ix['pivot2']] == 1:
                pivs.append(JLPivot(jlr[self.col_ix['dt']],
                                    jlr[self.col_ix['state2']],
                                    jlr[self.col_ix['price2']],
                                    jlr[self.col_ix['rg']]))
            if len(pivs) < num_pivs and jlr[self.col_ix['pivot']] == 1:
                pivs.append(JLPivot(jlr[self.col_ix['dt']],
                                    jlr[self.col_ix['state']],
                                    jlr[self.col_ix['price']],
                                    jlr[self.col_ix['rg']]))
            ixx -= 1
        pivs.reverse()
        return pivs

    def get_pivots_in_days(self, num_days):
        ixx = -1
        end = -len(self.jl_recs)
        pivs = []
        if end < -num_days:
            end = -num_days
        while ixx > end:
            jlr = self.jl_recs[ixx]
            if jlr[self.col_ix['pivot2']] == 1:
                pivs.append(JLPivot(jlr[self.col_ix['dt']],
                                    jlr[self.col_ix['state2']],
                                    jlr[self.col_ix['price2']],
                                    jlr[self.col_ix['rg']]))
            if jlr[self.col_ix['pivot']] == 1:
                pivs.append(JLPivot(jlr[self.col_ix['dt']],
                                    jlr[self.col_ix['state']],
                                    jlr[self.col_ix['price']],
                                    jlr[self.col_ix['rg']]))
            ixx -= 1
        pivs.reverse()
        return pivs

    def print_pivs(self, pivs):
        output = ''
        for piv in pivs:
            px_str = self.get_formatted_price(piv.state, 1, piv.price)
            output += '{0:s}{1:s}{2:6.2f}\n'.format(piv.dt, px_str, piv.rg)
        print(output)

    def last_rec(self, col_name, ixx=1):
        if ixx > len(self.jl_recs):
            ixx = len(self.jl_recs)
        jlr = self.jl_recs[-ixx]
        if col_name in ['state', 'price', 'pivot']:
            col_name2 = '{0:s}2'.format(col_name)
            if jlr[self.col_ix['state2']] != StxJL.Nil:
                return jlr[self.col_ix[col_name2]]
        return jlr[self.col_ix[col_name]]

    def get_html_formatted_price(self, piv, pivot): # state, pivot, price):
        res = '<tr><td>{0:s}</td>'.format(piv.dt)
        res += (piv.state * '<td></td>')
        td_style = ''
        if pivot:
            td_style = ' style="background-color:#{0:s};"'.format(
                '006400' if piv.state in [StxJL.UT, StxJL.NRe] else '640000')
        res += '<td{0:s}>{1:.2f}</td>'.format(td_style, piv.price / 100.0)
        res += ((7 - piv.state) * '<td></td>')
        return res

    def html_report(self, pivs):
        html_table = '<table border="1">'
        html_table += '<tr><th>Date</th><th>SRa</th><th>NRa</th>'\
            '<th>UT</th><th>DT</th><th>NRe</th><th>SRe</th>'\
            '<th>range</th><th>OBV</th></tr>'
        for piv in pivs:
            piv_row = self.get_html_formatted_price(piv, 1)
            html_table += piv_row
#         html_table.append(self.get_html_formatted_price()
        html_table += '</table>'
        return html_table

    @classmethod
    def jl_report(cls, stk, start_date, end_date, factor):
        df = DataLoader.load_eod(
            stk,
            start_date=start_date,
            end_date=end_date,
            as_of_date=end_date,
        )
        jl = StxJL(df, factor)
        jl.jl(end_date)
        pivs = jl.get_num_pivots(4)
        return jl.html_report(pivs)


# ------------------------------  CLI wrapper  ------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate JL pivotal points for a ticker",
    )
    parser.add_argument("--stk", required=True, help="Ticker symbol")
    parser.add_argument(
        "--start_date", required=True, help="Start date for loading data"
    )
    parser.add_argument(
        "--end_date", required=True, help="End date for loading data"
    )
    parser.add_argument(
        "--data_type",
        choices=["eod", "intraday"],
        required=True,
        help="Whether to load end-of-day or intraday data",
    )
    parser.add_argument(
        "--dt",
        required=True,
        help="Date or timestamp as of which to calculate JL pivots",
    )
    parser.add_argument(
        "--factor",
        type=float,
        required=True,
        help="Livermore penetration factor",
    )
    args = parser.parse_args()

    loader = DataLoader()

    # parse the date range so it matches Polars' native dtypes
    start_dt = datetime.fromisoformat(args.start_date).date()
    end_dt = datetime.fromisoformat(args.end_date).date()

    if args.data_type == "eod":
        df = loader.load_eod(
            args.stk,
            start_date=start_dt.isoformat(),
            end_date=end_dt.isoformat(),
            as_of_date=end_dt.isoformat(),
        )
        df = df.rename(
            {
                "date": "dt",
                "high": "hi",
                "low": "lo",
                "close": "c",
            }
        )
    else:
        df = loader.load_intraday(
            args.stk,
            start_dt.isoformat(),
            end_dt.isoformat(),
            as_of_date=end_dt.isoformat(),
        )
        df = df.rename(
            {
                "timestamp": "dt",
                "high": "hi",
                "low": "lo",
                "close": "c",
            }
        )

    jl = StxJL(df, args.factor)

    # Convert the pivot calculation date to match the df dtype
    if df.get_column("dt").dtype == pl.Date:
        calc_dt = datetime.fromisoformat(args.dt).date()
    elif df.get_column("dt").dtype == pl.Datetime:
        calc_dt = datetime.fromisoformat(args.dt)
    else:
        calc_dt = args.dt

    jl.jl(calc_dt)
    jl.jl_print()
