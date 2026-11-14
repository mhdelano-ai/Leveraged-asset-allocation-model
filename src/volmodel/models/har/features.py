"""Regressors of the HAR family, for every month-end of a Panel view at once.

Everything is a VARIANCE on the target's scale (annualized variance of
21-trading-day log excess returns) measured over a trailing window that ends
at the month-end, using only data dated at or before it. Computing all
month-ends in one vectorized pass serves both uses: the training rows at a
yearly estimation date and the single forecast row at the origin, from the
same code (so a training row at ``s`` is exactly what a forecast at ``s``
would have seen).

Monthly-measured (any era, immune to daily pricing conventions)
    LR      12 * mean(r_m^2) over the asset's whole history to date
    M60     12 * mean(r_m^2) over the last 60 months (what exists if fewer)
    M12     12 * mean(r_m^2) over the last 12 months
    M3      12 * mean(r_m^2) over the last 3 months
No one-month monthly regressor: a single squared return carries almost no
information about the level (log chi^2_1 has variance 4.9).

Daily-measured, mapped to the 21-day scale (D1, D3, D12 for 1, 3, 12 months)
    Dh = D_yr * v1(h) * q
    v1(h)  per-row mean of squared daily log excess returns over the window
           (a missing day counts 0 and the row still counts, exactly as in the
           target, so a return spanning a holiday is not lost or doubled)
    D_yr   panel daily rows in the last 12 months (about 252; about 295
           before 1952), the target's own annualization
    q      the asset's trailing 21-day / 1-day variance ratio:
           q_raw = [mean of squared overlapping 21-row sums / 21] / v1 over
           the window (t - 36 months, t] cut at the start of the asset's
           current data-tier run (``view.tiers``), so a pre-splice scale is
           never carried across a tier change; shrunk toward 1 in logs,
           q = q_raw ** (n / (n + 18)) with n the window length in months.
           The 18 is the ratio of sampling variance to prior variance:
           se(log q_raw)^2 ~ 0.09 * 12 / n (a 21-day variance over one year
           has ~34% relative error, partly offset in the ratio by the
           correlated 1-day variance) and the cross-segment spread of log q
           in reports/data/horizon_evidence.md is sd ~0.25 (vol ratios
           0.69-1.20), so n0 = 0.09 * 12 / 0.0625 ~ 18 months.
    daily_ok  the daily regressors exist only if the asset's tier at t is A
           or B (daily sources), the current tier run is at least 12 months
           long (so all three windows lie inside one run), and daily returns
           cover at least 90% of the rows of the last 12 months and of the q
           window (the target's own coverage rule).
Within-tier source changes (e.g. hy_credit VWEHX -> HYG in 2007, both tier
A) are not visible in the view; there q adapts over the 36-month window.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

Q_WINDOW = 36          # months of trailing data for the 21-day / 1-day ratio
Q_PRIOR_MONTHS = 18.0  # shrinkage of log q toward 0, see module docstring
MIN_RUN = 12           # months of the current tier run before daily regressors are used
MIN_COVER = 0.90       # share of rows with a daily return (the target's rule)
K = 21                 # the target's horizon in trading days

MONTHLY_REGS = ("M3", "M12", "M60")
DAILY_REGS = ("D1", "D3", "D12", "M60")


def _prefix(a: np.ndarray) -> np.ndarray:
    out = np.zeros((a.shape[0] + 1,) + a.shape[1:], dtype=float)
    np.cumsum(a, axis=0, out=out[1:])
    return out


def _monthly_window(P: np.ndarray, C: np.ndarray, h: int, min_cnt: int) -> np.ndarray:
    """12 * mean of squares over the last h rows (inclusive), NaN if < min_cnt obs."""
    G = P.shape[0] - 1
    hi = np.arange(1, G + 1)
    lo = np.maximum(hi - h, 0)
    s = P[hi] - P[lo]
    c = C[hi] - C[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        v = 12.0 * s / c
    return np.where(c >= min_cnt, v, np.nan)


def har_features(view) -> dict:
    """All regressors at every month-end of ``view`` (G x N arrays, assets = columns)."""
    xm_df = view.excess_monthly
    grid = xm_df.index
    assets = list(xm_df.columns)
    G, N = len(grid), len(assets)
    xm = xm_df.to_numpy(float)
    okm = np.isfinite(xm)
    xm0 = np.where(okm, xm, 0.0)
    Pm, Cm = _prefix(xm0 ** 2), _prefix(okm.astype(float))
    n_obs = Cm[1:]
    with np.errstate(invalid="ignore", divide="ignore"):
        LR = np.where(n_obs > 0, 12.0 * Pm[1:] / n_obs, np.nan)
    M60 = _monthly_window(Pm, Cm, 60, 1)
    M12 = _monthly_window(Pm, Cm, 12, 12)
    M3 = _monthly_window(Pm, Cm, 3, 3)
    # a missing short window falls back to the next longer one
    M60 = np.where(np.isfinite(M60), M60, LR)
    M12 = np.where(np.isfinite(M12), M12, M60)
    M3 = np.where(np.isfinite(M3), M3, M12)

    out = {"grid": grid, "assets": assets, "n_obs": n_obs, "LR": LR,
           "M60": M60, "M12": M12, "M3": M3}

    nanG = np.full((G, N), np.nan)
    D1, D3, D12, q = nanG.copy(), nanG.copy(), nanG.copy(), nanG.copy()
    daily_ok = np.zeros((G, N), dtype=bool)
    d = view.daily
    if len(d) > K + 1:
        xd = view.excess_daily.reindex(columns=assets).to_numpy(float)
        dates = d.index
        okd = np.isfinite(xd)
        x0 = np.where(okd, xd, 0.0)
        T = x0.shape[0]
        P2, Pok = _prefix(x0 ** 2), _prefix(okd.astype(float))
        Cx = _prefix(x0)
        s21 = np.zeros_like(x0)
        s21[K - 1:] = Cx[K:] - Cx[:T - K + 1]          # sum of rows j-20..j, at row j
        Q = _prefix(s21 ** 2)

        # row boundary after each month-end: rows [0, end[i]) are dated <= grid[i]
        end = dates.searchsorted(grid, side="right")

        def start_of(back):                              # rows after grid[i - back]
            idx = np.arange(G) - back
            return np.where(idx >= 0, end[np.maximum(idx, 0)], 0)

        e = end[:, None]
        # tier runs
        tiers = view.tiers.reindex(index=grid, columns=assets)
        tv = tiers.to_numpy(dtype=object)
        tv = np.where(pd.isna(tv), "", tv).astype(str)
        change = np.ones((G, N), dtype=bool)
        change[1:] = tv[1:] != tv[:-1]
        run_first = np.maximum.accumulate(np.where(change, np.arange(G)[:, None], 0), axis=0)
        daily_tier = np.isin(tv, ["A", "B"])

        # q window: months (grid[a], grid[i]], a = max(i - 36, run_first - 1)
        a_idx = np.maximum(np.arange(G)[:, None] - Q_WINDOW, run_first - 1)
        n_months = np.arange(G)[:, None] - a_idx
        qs = np.where(a_idx >= 0, end[np.maximum(a_idx, 0)], 0)
        rows = e - qs
        with np.errstate(invalid="ignore", divide="ignore"):
            sum1 = _gather(P2, e) - _gather(P2, qs)
            cov_q = (_gather(Pok, e) - _gather(Pok, qs)) / rows
            m_sums = rows - (K - 1)
            lo21 = np.minimum(qs + K - 1, e)
            sum21 = _gather(Q, e) - _gather(Q, lo21)
            v21 = sum21 / m_sums / K
            v1q = sum1 / rows
            q_raw = v21 / v1q
            w = n_months / (n_months + Q_PRIOR_MONTHS)
            qq = np.exp(w * np.log(q_raw))

        s12 = start_of(12)[:, None]
        rows12 = e - s12
        with np.errstate(invalid="ignore", divide="ignore"):
            cov12 = (_gather(Pok, e) - _gather(Pok, s12)) / rows12
        Dyr = rows12.astype(float)

        def v1(back):
            s = start_of(back)[:, None]
            r = e - s
            with np.errstate(invalid="ignore", divide="ignore"):
                return (_gather(P2, e) - _gather(P2, s)) / r

        ok = daily_tier & (n_months >= MIN_RUN) & (cov12 >= MIN_COVER) & (cov_q >= MIN_COVER) \
            & (m_sums >= 2 * K) & np.isfinite(qq) & (qq > 0)
        vals = {}
        for name, back in (("D1", 1), ("D3", 3), ("D12", 12)):
            vv = v1(back) * qq * Dyr
            ok &= np.isfinite(vv) & (vv > 0)
            vals[name] = vv
        daily_ok = ok
        D1, D3, D12 = (np.where(ok, vals[n], np.nan) for n in ("D1", "D3", "D12"))
        q = np.where(ok, qq, np.nan)
    out.update({"D1": D1, "D3": D3, "D12": D12, "q": q, "daily_ok": daily_ok})
    return out


def _gather(P: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """P[idx[i, j], j] for a (T+1) x N prefix array and a G x N row index."""
    return np.take_along_axis(P, np.broadcast_to(idx, (idx.shape[0], P.shape[1])), 0)
