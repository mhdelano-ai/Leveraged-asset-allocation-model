"""Features and training targets for the pooled ML vol models.

Everything is computed from ONE Panel view on its month-end grid with
trailing (causal) operations only, so the value for month ``s`` is the same
whatever later data the view holds. All variances are annualized variances
of 21-day log excess returns (the target's scale) and enter in logs RELATIVE
to the asset's own long-run level

    LR_{a,s} = log( 12 * mean_{u <= s} r_{a,u}^2 )      (expanding, monthly data)

so that assets are comparable and a learner cannot simply memorize asset
identity from the level.

Per (month-end s, asset a) row:

    lr          LR_{a,s} itself (log long-run variance)
    hist        log(years of monthly history)
    m3,m12,m60  log(12 * mean r^2 over the last 3/12/60 months) - LR
                (m60 needs >= 36 months)                      [monthly, all eras]
    d1,d3,d12   log of the 21-day-scale variance from DAILY data over the last
                21/63/252 trading days - LR; NaN without daily data  [see below]
    g_m3,g_m12  cross-sectional mean of m3 / m12 over the assets eligible at s
    g_d3        cross-sectional mean of d3 over the assets with daily data
    grp_*       asset-group dummies (rates, credit, real; equity = baseline)
    x_*         optional macro/implied state (MACRO_SERIES), NaN when the
                series is not yet published or stale

Daily features and the 21-day scale. A 1-day variance is NOT the target's
quantity (smoothed NAVs understate it, ETFs overstate it; ARCHITECTURE.md).
The daily features therefore use OVERLAPPING 5-day sums (k = 5 removes most
of the convention effect; ``reports/data/horizon_evidence.md``) and are then
rescaled by a trailing estimate of the 21-day / 5-day variance ratio,

    log rho = log[ (mean s21^2 / 21) / (mean s5^2 / 5) ]

measured on the same asset's daily data over the last <= 60 months but never
before the asset's last data-tier change (``view.tiers``; a splice), so a
pre-splice scale is never carried across a splice. The log ratio is shrunk
toward 0 by n / (n + 36) (n = months in its window), clipped to +-0.5, and
set to 0 with less than 12 months of same-tier daily data. Annualization is
by the actual number of daily rows in the trailing 12 months (~295 a year
before 1952, ~252 after), exactly as the target.

Targets (``training_rows``): for every origin s with s + 12 months <= the
view's observed-through date, the realized 12-month variance with the same
estimator and coverage rules as ``assetforecast.eval.targets`` (21-day
overlapping sums, n_rows/21 annualization, daily coverage >= 90% of trading
days and >= 10 returns per month; else the monthly proxy 12 * mean r_m^2),
expressed as the RATIO y = var / exp(LR_{a,s}).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ...data.panel import Panel
from ...eval.targets import (HORIZON_DAYS, MIN_DAILY_FRAC, MIN_DAYS_PER_MONTH, kday_sums,
                             observed_through)
from ...universe import BY_ID, HORIZON_MONTHS

MIN_MONTHS = 36           # eligibility, as the harness
K_SHORT = 5               # daily sums used for daily-based features
VAR_FLOOR = 1e-6          # annualized variance floor (0.1% vol) before logs
RATIO_MAX_MONTHS = 60
RATIO_MIN_MONTHS = 12
RATIO_SHRINK_MONTHS = 36
DAILY_MIN_FRAC = 0.8      # share of window rows an asset must have traded

MONTHLY_FEATURES = ["m3", "m12", "m60"]
DAILY_FEATURES = ["d1", "d3", "d12"]
GLOBAL_FEATURES = ["g_m3", "g_m12", "g_d3"]
GROUP_FEATURES = ["grp_rates", "grp_credit", "grp_real"]
CORE_FEATURES = ["lr", "hist"] + MONTHLY_FEATURES + DAILY_FEATURES + GLOBAL_FEATURES + GROUP_FEATURES
# forecast variance must be non-decreasing in every trailing-variance feature
MONOTONE_FEATURES = set(MONTHLY_FEATURES + DAILY_FEATURES + GLOBAL_FEATURES)

# name -> (series, transform, max age in days); chosen a priori: the market's
# implied vol, the credit (default) spread, the term spread, inflation
# uncertainty and a recession signal - the standard macro vol predictors
# (Schwert 1989; Paye 2012; Christiansen, Schmeling and Schrimpf 2012).
MACRO_SERIES = {
    "x_vix": ("iv_vix_vxo_spliced", "log", 10),
    "x_baa_aaa": ("credit_moodys_baa_aaa_spread_m", "log", 130),
    "x_term": ("macro_term_spread", "raw", 130),
    "x_infl_vol": ("macro_infl_vol_36m", "log", 130),
    "x_sahm": ("macro_sahm_gap", "raw", 130),
}
MACRO_FEATURES = list(MACRO_SERIES)


def _prefix(a: np.ndarray) -> np.ndarray:
    out = np.zeros((a.shape[0] + 1,) + a.shape[1:], dtype=float)
    np.cumsum(a, axis=0, out=out[1:])
    return out


def _safe_log(v):
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.log(np.maximum(v, VAR_FLOOR))


class FeatureState:
    """All per-month arrays (G months x N assets) for one view, built once."""

    def __init__(self, view: Panel, macro: bool = True):
        self.view = view
        m = view.excess_monthly
        self.assets = list(m.columns)
        self.grid = pd.DatetimeIndex(m.index)
        G, N = len(self.grid), len(self.assets)
        r = m.to_numpy(float)
        ok = np.isfinite(r)
        r0 = np.where(ok, r, 0.0)
        cnt_p, sq_p = _prefix(ok.astype(float)), _prefix(r0 ** 2)
        cnt = cnt_p[1:]
        self.cnt = cnt
        with np.errstate(invalid="ignore", divide="ignore"):
            self.lr = np.where(cnt > 0, _safe_log(12.0 * sq_p[1:] / np.maximum(cnt, 1)), np.nan)
        self.r0, self.okm = r0, ok
        self.sq_p, self.cnt_p = sq_p, cnt_p

        F = {}
        F["lr"] = self.lr
        with np.errstate(divide="ignore"):
            F["hist"] = np.where(cnt > 0, np.log(np.maximum(cnt, 1) / 12.0), np.nan)
        idx = np.arange(1, G + 1)
        for name, k, need in (("m3", 3, 3), ("m12", 12, 12), ("m60", 60, 36)):
            lo = np.maximum(idx - k, 0)
            c = cnt_p[idx] - cnt_p[lo]
            s = sq_p[idx] - sq_p[lo]
            with np.errstate(invalid="ignore", divide="ignore"):
                v = 12.0 * s / np.maximum(c, 1)
            F[name] = np.where(c >= need, _safe_log(v) - self.lr, np.nan)
        F.update(self._daily_features())
        elig = cnt >= MIN_MONTHS
        with np.errstate(invalid="ignore"):
            for g, src, mask in (("g_m3", "m3", elig), ("g_m12", "m12", elig),
                                 ("g_d3", "d3", elig & np.isfinite(F["d3"]))):
                x = np.where(mask & np.isfinite(F[src]), F[src], 0.0)
                n = (mask & np.isfinite(F[src])).sum(axis=1)
                F[g] = np.repeat(np.where(n > 0, x.sum(axis=1) / np.maximum(n, 1), np.nan)[:, None], N, axis=1)
        groups = np.array([BY_ID[a].group if a in BY_ID else "other" for a in self.assets])
        for gname in ("rates", "credit", "real"):
            F[f"grp_{gname}"] = np.repeat((groups == gname).astype(float)[None, :], G, axis=0)
        if macro:
            for name, arr in self._macro_features().items():
                F[name] = np.repeat(arr[:, None], N, axis=1)
        self.F = F
        self.elig = elig

    # ------------------------------------------------------------ daily
    def _daily_features(self) -> dict:
        view, grid = self.view, self.grid
        G, N = len(grid), len(self.assets)
        out = {k: np.full((G, N), np.nan) for k in DAILY_FEATURES}
        d = view.excess_daily.reindex(columns=self.assets)
        if len(d) < 300:
            return out
        didx = pd.DatetimeIndex(d.index)
        x = d.to_numpy(float)
        okd = np.isfinite(x)
        x0 = np.where(okd, x, 0.0)
        Pc = _prefix(okd.astype(float))
        P5 = _prefix(kday_sums(x0, K_SHORT) ** 2)
        P21 = _prefix(kday_sums(x0, HORIZON_DAYS) ** 2)
        e = didx.searchsorted(grid, side="right")                       # rows [.., e)
        e12 = didx.searchsorted(grid - pd.offsets.MonthEnd(12), side="right")
        rpy = (e - e12).astype(float)                                   # rows in trailing 12 months
        # tier segments: start (month index) of the current same-tier run
        tiers = view.tiers.reindex(index=grid, columns=self.assets)
        tv = tiers.to_numpy(dtype=object)
        seg_start = np.zeros((G, N), dtype=int)
        for j in range(N):
            col = tv[:, j]
            st = 0
            for g in range(G):
                if g > 0 and not (pd.isna(col[g]) or pd.isna(col[g - 1])) and col[g] != col[g - 1]:
                    st = g
                seg_start[g, j] = st
        # first daily row of the current segment = first row after the previous month-end
        row_after = didx.searchsorted(grid, side="right")                # first row after each month-end
        seg_row = np.where(seg_start > 0, row_after[np.maximum(seg_start - 1, 0)], 0)
        ratio_lo_row = didx.searchsorted(grid - pd.offsets.MonthEnd(RATIO_MAX_MONTHS), side="right")
        first_row = np.array([np.argmax(okd[:, j]) if okd[:, j].any() else len(didx) for j in range(N)])

        # 21-day / 5-day scale ratio, trailing within the current tier segment
        w0 = np.maximum(np.maximum(ratio_lo_row[:, None], seg_row), first_row[None, :])
        ee = e[:, None] * np.ones((1, N), dtype=int)
        n21 = ee - HORIZON_DAYS + 1 - w0
        n5 = ee - K_SHORT + 1 - w0
        lrho = np.zeros((G, N))
        good = (n21 >= RATIO_MIN_MONTHS * 21)
        cols = np.arange(N)[None, :]
        with np.errstate(invalid="ignore", divide="ignore"):
            a21 = (P21[np.clip(ee - HORIZON_DAYS + 1, 0, len(P21) - 1), cols]
                   - P21[np.clip(w0, 0, len(P21) - 1), cols]) / np.maximum(n21, 1) / HORIZON_DAYS
            a5 = (P5[np.clip(ee - K_SHORT + 1, 0, len(P5) - 1), cols]
                  - P5[np.clip(w0, 0, len(P5) - 1), cols]) / np.maximum(n5, 1) / K_SHORT
            raw = np.log(a21 / a5)
        good &= np.isfinite(raw) & (a5 > 0)
        months = np.maximum(ee - w0, 0) / 21.0
        lrho = np.where(good, np.clip(raw, -0.5, 0.5) * months / (months + RATIO_SHRINK_MONTHS), 0.0)
        self.lrho = lrho

        for name, L in (("d1", 21), ("d3", 63), ("d12", 252)):
            lo = ee - L
            valid = (lo >= first_row[None, :]) & (lo >= 0)
            lo_c = np.clip(lo, 0, None)
            nobs = Pc[ee, cols] - Pc[lo_c, cols]
            hi5 = np.clip(ee - K_SHORT + 1, 0, len(P5) - 1)
            s = P5[hi5, cols] - P5[np.minimum(lo_c, hi5), cols]
            with np.errstate(invalid="ignore", divide="ignore"):
                v = rpy[:, None] / K_SHORT * s / (L - K_SHORT + 1)
            valid &= (nobs >= DAILY_MIN_FRAC * L) & (rpy[:, None] > 100)
            out[name] = np.where(valid, _safe_log(v) + lrho - self.lr, np.nan)
        return out

    # ------------------------------------------------------------ macro
    def _macro_features(self) -> dict:
        p = self.view.predictors
        grid = self.grid
        out = {}
        for name, (series, tf, max_age) in MACRO_SERIES.items():
            arr = np.full(len(grid), np.nan)
            sub = p[p["series"] == series] if len(p) else p
            if len(sub):
                av = pd.DatetimeIndex(sub["available_date"])
                ob = pd.DatetimeIndex(sub["obs_date"])
                val = sub["value"].to_numpy(float)
                k = av.searchsorted(grid, side="right") - 1          # last published by month-end
                has = k >= 0
                kk = np.clip(k, 0, None)
                age = (grid - ob[kk]).days
                v = np.where(has & (age <= max_age), val[kk], np.nan)
                if tf == "log":
                    with np.errstate(invalid="ignore", divide="ignore"):
                        v = np.where(v > 0, np.log(v), np.nan)
                arr = v
            out[name] = arr
        return out

    # ------------------------------------------------------------ outputs
    def frame(self, rows: np.ndarray, cols: np.ndarray, features: list[str]) -> pd.DataFrame:
        return pd.DataFrame({f: self.F[f][rows, cols] for f in features})

    def row_at(self, t) -> int:
        return int(self.grid.searchsorted(pd.Timestamp(t), side="right")) - 1


def realized_targets(view: Panel) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Realized 12-month variances from the view, ``targets``' estimator and rules.

    Returns (origin grid positions P, asset positions A, variances V) for every
    (origin, asset) whose window ends on or before ``observed_through(view)``
    and that has a daily-based or a monthly-proxy target.
    """
    grid = pd.DatetimeIndex(view.monthly.index)
    H, k = HORIZON_MONTHS, HORIZON_DAYS
    through = observed_through(view)
    ends = grid + pd.offsets.MonthEnd(H)
    n_orig = int((ends <= through).sum())
    if n_orig <= 0:
        return np.array([], int), np.array([], int), np.array([])
    origins = grid[:n_orig]
    r = view.excess_monthly.to_numpy(float)
    ok = np.isfinite(r)
    Pc, Ps = _prefix(ok.astype(float)), _prefix(np.where(ok, r, 0.0) ** 2)
    a, b = np.arange(1, n_orig + 1), np.arange(1 + H, n_orig + 1 + H)
    m_cnt = Pc[b] - Pc[a]
    with np.errstate(invalid="ignore", divide="ignore"):
        proxy = np.where(m_cnt >= H - 0.5, (Ps[b] - Ps[a]) * H / np.maximum(m_cnt, 1), np.nan)
    var = proxy
    d = view.excess_daily
    d = d.loc[d.index <= through]
    if len(d) > k:
        didx = pd.DatetimeIndex(d.index)
        x = d.to_numpy(float)
        okd = np.isfinite(x)
        x0 = np.where(okd, x, 0.0)
        trade = okd.any(axis=1)
        s = didx.searchsorted(origins, side="right")
        e = didx.searchsorted(ends[:n_orig], side="right")
        Pdc, Pdt = _prefix(okd.astype(float)), _prefix(trade.astype(float))
        n_obs = Pdc[e] - Pdc[s]
        n_days = Pdt[e] - Pdt[s]
        mc = pd.DataFrame(okd.astype(float), index=didx).groupby(
            didx + pd.offsets.MonthEnd(0)).sum().reindex(grid, fill_value=0.0).to_numpy()
        # min over the window months p+1..p+H
        mcw = np.lib.stride_tricks.sliding_window_view(
            np.vstack([mc, np.zeros((H, mc.shape[1]))]), H, axis=0)[1:n_orig + 1]
        min_month = mcw.min(axis=2)
        P21 = _prefix(kday_sums(x0, k) ** 2)
        hi = np.maximum(e - k + 1, s)
        n_sums = (hi - s).astype(float)
        hi_c = np.minimum(hi, len(P21) - 1)
        s_c = np.minimum(s, hi_c)
        with np.errstate(invalid="ignore", divide="ignore"):
            dv = (P21[hi_c] - P21[s_c]) * ((e - s) / k / n_sums)[:, None]
        daily_ok = (n_days[:, None] > 0) & (n_sums[:, None] > 0) \
            & (n_obs >= MIN_DAILY_FRAC * n_days[:, None]) & (min_month >= MIN_DAYS_PER_MONTH) \
            & np.isfinite(dv) & (dv > 0)
        var = np.where(daily_ok, dv, proxy)
    P, A = np.nonzero(np.isfinite(var) & (var > 0))
    return P, A, var[P, A]
