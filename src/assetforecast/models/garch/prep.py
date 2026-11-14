"""Turning a Panel view into model inputs on the 21-day scale.

The target is the variance of 21-trading-day sums. Daily (and, less so,
5-day) variance depends on how a source is priced (smoothed NAVs: too low;
ETFs of foreign/illiquid assets: too high), and that convention changes at
splice dates. Before any model sees daily or weekly returns they are
rescaled by a CAUSAL, segment-local estimate of the variance ratio

    VR_k[i] = EW-mean(s21^2 / 21) / EW-mean(s_k^2 / k)

computed over rows <= i of the current source segment (s_m = overlapping
m-row sums; exponential weights, half-life ``VR_HALF_LIFE_DAYS``), shrunk in
logs toward 1:

    log VR* = w log VR,   w = n_eff / (n_eff + n0),   n0 = 2 * 21 / prior_sd^2

``n_eff`` is the Kish effective number of 21-day sums in the EW window;
``2 * 21 / n`` is the iid sampling variance of a log variance ratio from n
overlapping 21-day sums; ``prior_sd`` is the cross-segment spread of the log
variance ratio in ``reports/data/horizon_evidence.md`` (k = 1: 1-day/21-day
vol ratios 0.69..1.20, i.e. log variance ratios about +-0.3 sd; k = 5:
0.84..1.08, about +-0.15). Segments are runs of constant data tier in
``view.tiers``: the EW estimate restarts at every tier change, so no
pre-splice scale is carried across it. Source changes that keep the tier
(e.g. VGSIX -> VNQ) are absorbed by the 3-year half-life instead.

Return at row i is multiplied by sqrt(VR*[i-1]) (the ratio known the day
before), so the adjusted series is causal and the same whatever origin it is
computed from.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import lfilter

VR_HALF_LIFE_DAYS = 756          # 3 years: noise SE of log VR ~0.14 at full weight; adapts within a few years
VR_PRIOR_SD = {1: 0.30, 5: 0.15}
VR_CLIP = (np.log(1 / 3), np.log(3))
HORIZON = 21


def segment_ids(dates: pd.DatetimeIndex, tiers: pd.Series) -> np.ndarray:
    """Integer id per row, incremented whenever the asset's monthly tier changes."""
    me = dates + pd.offsets.MonthEnd(0)
    t = tiers.reindex(me).astype(object).fillna("?").to_numpy()
    chg = np.r_[True, t[1:] != t[:-1]]
    return np.cumsum(chg) - 1


def _ew(x, lam):
    return lfilter([1.0], [1.0, -lam], x)


def vr_log(r: np.ndarray, seg: np.ndarray, k: int = 1) -> np.ndarray:
    """Shrunk log variance ratio (21-day vs k-day) known at the close of each row."""
    n = len(r)
    out = np.zeros(n)
    lam = 0.5 ** (1.0 / VR_HALF_LIFE_DAYS)
    n0 = 2.0 * HORIZON / VR_PRIOR_SD[k] ** 2
    x_all = np.nan_to_num(r, nan=0.0)
    starts = np.flatnonzero(np.r_[True, seg[1:] != seg[:-1]])
    ends = np.r_[starts[1:], n]
    for i0, i1 in zip(starts, ends):
        m = i1 - i0
        if m <= HORIZON:
            continue
        c = np.r_[0.0, np.cumsum(x_all[i0:i1])]
        S21 = np.zeros(m); I21 = np.zeros(m)
        S21[HORIZON - 1:] = (c[HORIZON:] - c[:-HORIZON]) ** 2 / HORIZON
        I21[HORIZON - 1:] = 1.0
        Sk = np.zeros(m); Ik = np.zeros(m)
        Sk[k - 1:] = (c[k:] - c[:-k]) ** 2 / k
        Ik[k - 1:] = 1.0
        n21, d21, nk, dk = _ew(S21, lam), _ew(I21, lam), _ew(Sk, lam), _ew(Ik, lam)
        sw2 = lfilter([1.0], [1.0, -lam * lam], I21)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = (n21 / d21) / (nk / dk)
            neff = d21 ** 2 / sw2
            lr = np.clip(np.log(ratio), *VR_CLIP)
            w = neff / (neff + n0)
            v = w * lr
        out[i0:i1] = np.where(np.isfinite(v), v, 0.0)
    return out


def _lag_within_segment(v: np.ndarray, seg: np.ndarray, lag: int = 1) -> np.ndarray:
    out = np.zeros_like(v)
    if len(v) > lag:
        out[lag:] = np.where(seg[lag:] == seg[:-lag], v[:-lag], 0.0)
    return out


def asset_rows(ex_daily: pd.Series) -> pd.Series:
    """The asset's daily rows from its first to its last observation (NaN = no trade)."""
    s = ex_daily
    fv, lv = s.first_valid_index(), s.last_valid_index()
    if fv is None:
        return s.iloc[:0]
    return s.loc[fv:lv]


def daily_adjusted(ex_daily: pd.Series, tiers: pd.Series) -> pd.Series:
    """VR-adjusted daily log excess returns on the asset's own trading days."""
    s = asset_rows(ex_daily)
    if s.empty:
        return s
    r = s.to_numpy(float)
    seg = segment_ids(s.index, tiers)
    lv = _lag_within_segment(vr_log(r, seg, 1), seg, 1)
    adj = r * np.exp(0.5 * lv)
    ok = np.isfinite(r)
    return pd.Series(adj[ok], index=s.index[ok])


def block_adjusted(ex_daily: pd.Series, tiers: pd.Series, end, k: int = 5) -> pd.Series:
    """Non-overlapping k-row sums of daily log excess returns ending exactly at
    the last row <= ``end`` (anchored at the end, so the newest block is
    complete), each scaled by sqrt(VR_k) known before the block starts.
    Indexed by block end date."""
    s = asset_rows(ex_daily)
    s = s.loc[:pd.Timestamp(end)]
    if len(s) < k:
        return s.iloc[:0]
    r = s.to_numpy(float)
    seg = segment_ids(s.index, tiers)
    lv = vr_log(r, seg, k)
    x = np.nan_to_num(r, nan=0.0)
    c = np.r_[0.0, np.cumsum(x)]
    n = len(r)
    ends = np.arange(n, k - 1, -k)[::-1]          # exclusive end positions in c
    sums = c[ends] - c[ends - k]
    first = ends - k                              # row index of the block's first row
    prev = first - 1
    fac = np.where((prev >= 0) & (seg[np.maximum(prev, 0)] == seg[first]), lv[np.maximum(prev, 0)], 0.0)
    return pd.Series(sums * np.exp(0.5 * fac), index=s.index[ends - 1])


def rows_last_year(index: pd.DatetimeIndex, t) -> int:
    t = pd.Timestamp(t)
    lo = t - pd.DateOffset(years=1)
    return int(((index > lo) & (index <= t)).sum())


def est_date(t) -> pd.Timestamp:
    """Parameters are re-estimated once a year, on data through the last
    December 31 on or before the origin."""
    t = pd.Timestamp(t)
    return t.normalize() if (t.month == 12 and t.day == 31) else pd.Timestamp(year=t.year - 1, month=12, day=31)
