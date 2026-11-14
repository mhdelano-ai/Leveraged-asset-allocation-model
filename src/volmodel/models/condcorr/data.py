"""Inputs for the condcorr family, all derived from the view.

realized_corr_series   trailing 12-month realized correlations at every
                       month-end, with the same estimator as the target
                       (``volmodel.eval.targets``): overlapping 21-trading-day
                       sums of daily log excess returns, uncentered, when both
                       assets pass the daily coverage rules in the window;
                       otherwise the uncentered correlation of the 12 monthly
                       log excess returns (monthly-only eras). The value at
                       month-end ``e`` covers ``(e - 12 months, e]``. It is
                       the TRAINING TARGET for origin ``e - 12 months`` and the
                       "recent 12m" PREDICTOR at origin ``e``.
short_corr_series      trailing 63-trading-day correlation of 1-day returns
                       (uncentered; both assets need >= 50 returns) - the
                       "short recent window" predictor. Only where daily data
                       exist.
ref_pair_z             the pair value of ``ref_corr_last120m`` (half last-120m,
                       half all-history uncentered monthly correlation) at
                       every month-end: the baseline every model starts from.
predictors_at          macro predictors as known at every month-end (value of
                       the latest observation published by then; stale
                       series -> NaN).

Everything is vectorized with prefix sums (no per-month pandas slicing), so
a cold call recomputes the whole history in tens of milliseconds.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ...eval.targets import HORIZON_DAYS, MIN_DAILY_FRAC, MIN_DAYS_PER_MONTH, kday_sums

H = 12              # months in a target window
Z_CLIP = 0.98       # |corr| clip before the Fisher transform
SHORT_DAYS = 63     # "short recent window": one quarter of trading days
SHORT_MIN_OBS = 50


def fisher_z(c):
    return np.arctanh(np.clip(c, -Z_CLIP, Z_CLIP))


def _prefix(a: np.ndarray) -> np.ndarray:
    out = np.zeros((a.shape[0] + 1,) + a.shape[1:], dtype=float)
    np.cumsum(a, axis=0, out=out[1:])
    return out


def month_grid(view) -> pd.DatetimeIndex:
    idx = view.monthly.index
    return pd.date_range(pd.Timestamp(idx.min()) + pd.offsets.MonthEnd(0),
                         pd.Timestamp(idx.max()) + pd.offsets.MonthEnd(0), freq="ME").as_unit("ns")


def realized_corr_series(view, pairs: list[tuple[str, str]], grid: pd.DatetimeIndex | None = None,
                         k: int = HORIZON_DAYS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Trailing 12-month realized correlation of each pair at each month-end.

    Returns ``(corr, daily_based)``: two frames indexed by month-end with one
    column per pair (``"a~b"``); corr is NaN where neither estimator applies.
    """
    grid = month_grid(view) if grid is None else grid
    assets = sorted({a for p in pairs for a in p}, key=list(view.monthly.columns).index)
    pos = {a: i for i, a in enumerate(assets)}
    G = len(grid)
    cols = [f"{a}~{b}" for a, b in pairs]
    out = np.full((G, len(pairs)), np.nan)
    dflag = np.zeros((G, len(pairs)), dtype=bool)
    if G <= H:
        return pd.DataFrame(out, index=grid, columns=cols), pd.DataFrame(dflag, index=grid, columns=cols)

    # ---------------- monthly estimator: 12 monthly returns, all present
    xm = view.excess_monthly.reindex(index=grid, columns=assets).to_numpy(float)
    okm = np.isfinite(xm)
    xm0 = np.where(okm, xm, 0.0)
    Pc = _prefix(okm.astype(float))
    # window ending at grid row g holds rows (g-H, g]  ->  prefix rows [g-H+1, g+1)
    g = np.arange(H - 1, G)
    lo, hi = g - H + 1, g + 1
    cnt = Pc[hi] - Pc[lo]
    for p, (a, b) in enumerate(pairs):
        ia, ib = pos[a], pos[b]
        Pab = _prefix(xm0[:, ia] * xm0[:, ib])
        Paa = _prefix(xm0[:, ia] ** 2)
        Pbb = _prefix(xm0[:, ib] ** 2)
        full = (cnt[:, ia] >= H - 0.5) & (cnt[:, ib] >= H - 0.5)
        with np.errstate(invalid="ignore", divide="ignore"):
            c = (Pab[hi] - Pab[lo]) / np.sqrt((Paa[hi] - Paa[lo]) * (Pbb[hi] - Pbb[lo]))
        out[g, p] = np.where(full, c, np.nan)

    # ---------------- daily estimator: overlapping k-day sums (target definition)
    d = view.excess_daily.reindex(columns=assets)
    if len(d) > k:
        didx = d.index
        xd = d.to_numpy(float)
        okd = np.isfinite(xd)
        xd0 = np.where(okd, xd, 0.0)
        trade = okd.any(axis=1)
        e_row = didx.searchsorted(grid, side="right")                  # rows [.., e) up to month-end
        s_row = np.r_[np.zeros(H, dtype=int), e_row[:-H]]              # window start row for end g
        Pd_cnt, Pd_tr = _prefix(okd.astype(float)), _prefix(trade.astype(float))
        month_cnt = pd.DataFrame(okd.astype(float), index=didx).groupby(
            didx + pd.offsets.MonthEnd(0)).sum().reindex(grid, fill_value=0.0).to_numpy()
        # min over the 12 months ending at g
        mc = np.full((G, len(assets)), -1.0)
        if G >= H:
            mc[H - 1:] = np.lib.stride_tricks.sliding_window_view(month_cnt, H, axis=0).min(axis=2)
        Y = kday_sums(xd0, k)
        gg = np.arange(H, G)                                            # need a full year before
        s, e = s_row[gg], e_row[gg]
        n_obs = Pd_cnt[e] - Pd_cnt[s]
        n_days = Pd_tr[e] - Pd_tr[s]
        hi_y = np.maximum(e - k + 1, s)
        n_sums = hi_y - s
        okw = (n_days[:, None] > 0) & (n_sums[:, None] > 0) \
            & (n_obs >= MIN_DAILY_FRAC * n_days[:, None]) & (mc[gg] >= MIN_DAYS_PER_MONTH)
        for p, (a, b) in enumerate(pairs):
            ia, ib = pos[a], pos[b]
            Pab = _prefix(Y[:, ia] * Y[:, ib])
            Paa = _prefix(Y[:, ia] ** 2)
            Pbb = _prefix(Y[:, ib] ** 2)
            hc = np.minimum(hi_y, len(Pab) - 1)
            sc = np.minimum(s, hc)
            with np.errstate(invalid="ignore", divide="ignore"):
                c = (Pab[hc] - Pab[sc]) / np.sqrt((Paa[hc] - Paa[sc]) * (Pbb[hc] - Pbb[sc]))
            use = okw[:, ia] & okw[:, ib] & np.isfinite(c)
            out[gg[use], p] = c[use]
            dflag[gg[use], p] = True
    return (pd.DataFrame(np.clip(out, -1, 1), index=grid, columns=cols),
            pd.DataFrame(dflag, index=grid, columns=cols))


def short_corr_series(view, pairs: list[tuple[str, str]], grid: pd.DatetimeIndex | None = None,
                      days: int = SHORT_DAYS, min_obs: int = SHORT_MIN_OBS) -> pd.DataFrame:
    """Trailing ``days``-row uncentered correlation of 1-day log excess returns."""
    grid = month_grid(view) if grid is None else grid
    cols = [f"{a}~{b}" for a, b in pairs]
    out = np.full((len(grid), len(pairs)), np.nan)
    assets = sorted({a for p in pairs for a in p}, key=list(view.monthly.columns).index)
    d = view.excess_daily.reindex(columns=assets)
    if len(d) >= days:
        x = d.to_numpy(float)
        ok = np.isfinite(x)
        x0 = np.where(ok, x, 0.0)
        e = d.index.searchsorted(grid, side="right")
        s = np.maximum(e - days, 0)
        valid_rows = e >= days
        pos = {a: i for i, a in enumerate(assets)}
        for p, (a, b) in enumerate(pairs):
            ia, ib = pos[a], pos[b]
            both = ok[:, ia] & ok[:, ib]
            Pn = _prefix(both.astype(float))
            Pab = _prefix(np.where(both, x0[:, ia] * x0[:, ib], 0.0))
            Paa = _prefix(np.where(both, x0[:, ia] ** 2, 0.0))
            Pbb = _prefix(np.where(both, x0[:, ib] ** 2, 0.0))
            n = Pn[e] - Pn[s]
            with np.errstate(invalid="ignore", divide="ignore"):
                c = (Pab[e] - Pab[s]) / np.sqrt((Paa[e] - Paa[s]) * (Pbb[e] - Pbb[s]))
            out[:, p] = np.where(valid_rows & (n >= min_obs) & np.isfinite(c), c, np.nan)
    return pd.DataFrame(np.clip(out, -1, 1), index=grid, columns=cols)


def ref_pair_corr(view, pairs: list[tuple[str, str]], grid: pd.DatetimeIndex | None = None,
                  months: int = 120, weight: float = 0.5, min_overlap: int = 24) -> pd.DataFrame:
    """Pair value of ``ref_corr_last120m`` (before its PD repair) at every month-end:
    ``weight * corr(last months) + (1 - weight) * corr(all history)``, uncentered
    monthly, pairwise common sample; a thin window (< ``min_overlap`` common
    months) falls back to the long run; a thin long run gives 0."""
    grid = month_grid(view) if grid is None else grid
    cols = [f"{a}~{b}" for a, b in pairs]
    out = np.full((len(grid), len(pairs)), np.nan)
    xm = view.excess_monthly.reindex(index=grid)
    G = len(grid)
    hi = np.arange(1, G + 1)
    lo = np.maximum(hi - months, 0)
    for p, (a, b) in enumerate(pairs):
        xa, xb = xm[a].to_numpy(float), xm[b].to_numpy(float)
        both = np.isfinite(xa) & np.isfinite(xb)
        xa0, xb0 = np.where(both, xa, 0.0), np.where(both, xb, 0.0)
        Pn, Pab, Paa, Pbb = (_prefix(both.astype(float)), _prefix(xa0 * xb0),
                             _prefix(xa0 ** 2), _prefix(xb0 ** 2))
        with np.errstate(invalid="ignore", divide="ignore"):
            long = Pab[hi] / np.sqrt(Paa[hi] * Pbb[hi])
            rec = (Pab[hi] - Pab[lo]) / np.sqrt((Paa[hi] - Paa[lo]) * (Pbb[hi] - Pbb[lo]))
        n_long, n_rec = Pn[hi], Pn[hi] - Pn[lo]
        long = np.where((n_long >= min_overlap) & np.isfinite(long), long, 0.0)
        rec = np.where((n_rec >= min_overlap) & np.isfinite(rec), rec, long)
        out[:, p] = weight * rec + (1 - weight) * long
    return pd.DataFrame(np.clip(out, -1, 1), index=grid, columns=cols)


def predictors_at(view, series: list[str], grid: pd.DatetimeIndex | None = None,
                  max_age_days: int = 190) -> pd.DataFrame:
    """Value of each predictor as KNOWN at each month-end of ``grid``.

    For month-end ``e``: the observation with the latest ``obs_date`` among
    those published by ``e`` (``available_date <= e``). NaN if none, or if
    that observation is older than ``max_age_days`` (dead or interrupted series).
    """
    grid = month_grid(view) if grid is None else grid
    out = pd.DataFrame(np.nan, index=grid, columns=series)
    p = view.predictors
    if p is None or len(p) == 0:
        return out
    p = p[p["series"].isin(series)]
    if len(p) == 0:
        return out
    p = p.drop_duplicates(["series", "obs_date"], keep="first")
    left = pd.DataFrame({"t": grid.astype("datetime64[ns]")})
    for s_name, g in p.groupby("series", sort=False):
        g = g.sort_values("available_date", kind="stable")
        g = g.assign(obs_max=g["obs_date"].cummax())
        g = g[g["obs_date"] == g["obs_max"]]          # ignore late-published older obs
        r = pd.DataFrame({"t": g["available_date"].astype("datetime64[ns]").to_numpy(),
                          "obs": g["obs_date"].astype("datetime64[ns]").to_numpy(),
                          "v": g["value"].to_numpy(float)})
        m = pd.merge_asof(left, r, on="t", direction="backward")
        age = (m["t"] - m["obs"]).dt.days.to_numpy(float)
        v = np.where(age <= max_age_days, m["v"].to_numpy(float), np.nan)
        out[s_name] = v
    return out
