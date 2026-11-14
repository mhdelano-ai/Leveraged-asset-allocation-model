"""Regime-switching and structural-break volatility forecasters.

Every model forecasts the annualized variance of 21-day log excess returns
over the next 12 months and returns its square root. Shared rules:

* Parameters are re-estimated at most once a year, on the data up to the
  latest December 31 on or before the origin (the "estimation date"), and
  cached under a fingerprint of exactly the data they were fitted on; the
  state (filtered regime probabilities, run-length posterior) is updated
  every month from the view. A cold call therefore gives the same answer as
  a warm one, and data after the origin cannot reach a forecast.
* Only FILTERED probabilities enter a forecast. Smoothed probabilities are
  used inside EM on data up to the estimation date (that is estimation, not
  forecasting).
* Every eligible asset gets a finite forecast: an asset too short for its
  own fit borrows the pooled transition matrix and regime multipliers; if
  nothing can be fitted, or anything fails, the asset falls back to its
  full-history mean squared monthly return (= ``ref_longrun``).
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from ..base import VolForecaster
from ..ref import nearest_corr, pairwise_corr, window_var
from .breaks import BOCPDState, bocpd_forecast, bocpd_update, icss_kappa2
from .hmm import MSFit, expected_path, filter_probs, fit_common_ms_t, fit_ms_t, stationary

NU = 5.0            # Student-t degrees of freedom of every emission (fixed a priori)
HORIZON_M = 12      # months in the forecast window
VAR_FLOOR = 1e-8    # annualized variance floor (0.01% vol)


# ------------------------------------------------------------------ helpers
def estimation_date(view) -> pd.Timestamp:
    t = pd.Timestamp(view.asof_date if view.asof_date is not None else view.monthly.index[-1])
    dec = pd.Timestamp(year=t.year, month=12, day=31)
    return dec if t >= dec else pd.Timestamp(year=t.year - 1, month=12, day=31)


def fingerprint(*arrays) -> str:
    h = hashlib.blake2b(digest_size=16)
    for a in arrays:
        a = np.ascontiguousarray(np.asarray(a, dtype=float))
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def longrun_var(view, assets) -> pd.Series:
    """Annualized full-history mean squared monthly excess return (the universal fallback)."""
    return window_var(view.excess_monthly[assets], None)


def asset_series(X: pd.DataFrame, a: str) -> pd.Series:
    s = X[a]
    first = s.first_valid_index()
    return s.iloc[0:0] if first is None else s.loc[first:]


class _RegimeBase(VolForecaster):
    """Wraps ``_variance`` so that the model never raises or returns a bad value."""

    def __init__(self):
        self._cache: dict = {}
        self.last_error: str | None = None

    def _variance(self, view, assets) -> dict[str, float]:
        raise NotImplementedError

    def forecast_vol(self, view, assets):
        fb = longrun_var(view, assets)
        try:
            v = self._variance(view, assets)
            self.last_error = None
        except Exception as e:  # never let an estimation failure stop a backtest
            v = {}
            self.last_error = f"{type(e).__name__}: {e}"
        out = []
        for a in assets:
            x = v.get(a, np.nan)
            if not (np.isfinite(x) and x > 0):
                x = fb.get(a, np.nan)
            if not (np.isfinite(x) and x > 0):
                x = 0.04
            out.append(max(float(x), VAR_FLOOR))
        return pd.Series(np.sqrt(out), index=assets)


# ------------------------------------------------- per-asset monthly MS
class MSMonthly(_RegimeBase):
    """Per-asset K-state Markov-switching Student-t model on monthly log excess returns.

    x_t | s_t = k ~ t_nu(0, scale_k^2), s_t Markov(P). Forecast of the annual
    variance at origin T:  12 * (1/12) sum_{h=1..12} Pr(s_T | x_1..x_T) P^h v,
    v_k = E[x^2 | regime k].
    """

    def __init__(self, K: int = 2, min_fit: int | None = None):
        super().__init__()
        self.K = K
        # enough months that every regime is visited several times: 10 years for 2 states, 20 for 3
        self.min_fit = min_fit or {2: 120, 3: 240}.get(K, 120 * (K - 1))

    def fit_asset(self, a: str, xe: np.ndarray, est) -> MSFit | None:
        if np.isfinite(xe).sum() < self.min_fit:
            return None
        key = ("ms", a, est, fingerprint(xe))
        if key not in self._cache:
            try:
                self._cache[key] = fit_ms_t(xe, self.K, NU)
            except Exception:
                self._cache[key] = None
        return self._cache[key]

    def fits(self, view):
        X = view.excess_monthly
        est = estimation_date(view)
        out = {}
        for a in X.columns:
            s = asset_series(X, a)
            f = self.fit_asset(a, s.loc[:est].to_numpy(), est)
            if f is not None:
                out[a] = f
        return out

    def pooled(self, fits: dict[str, MSFit]):
        """Pooled transition matrix and regime multipliers (geometric mean across fitted assets)."""
        if not fits:
            return None
        Ps = np.array([f.P for f in fits.values()])
        rel = []
        for f in fits.values():
            pi = stationary(f.P)
            rel.append(np.log(f.var / float(pi @ f.var)))
        P = Ps.mean(axis=0)
        P /= P.sum(axis=1, keepdims=True)
        m = np.exp(np.mean(rel, axis=0))
        m /= float(stationary(P) @ m)
        return P, m

    def borrowed_fit(self, x: np.ndarray, pooled) -> MSFit:
        P, m = pooled
        ok = np.isfinite(x)
        lev = float(np.mean(x[ok] ** 2))
        var = lev * m
        return MSFit(P, var * (NU - 2) / NU, var, NU, np.nan, int(ok.sum()), 0, True)

    def _variance(self, view, assets):
        X = view.excess_monthly
        fits = self.fits(view)
        pooled = None
        out = {}
        for a in assets:
            x = asset_series(X, a).to_numpy()
            f = fits.get(a)
            if f is None:
                if pooled is None:
                    pooled = self.pooled(fits) or False
                if not pooled or np.isfinite(x).sum() < 12:
                    continue
                f = self.borrowed_fit(x, pooled)
            out[a] = 12.0 * f.forecast(x, HORIZON_M)[0]
        return out


# --------------------------------------------- per-asset weekly (5-day) MS
def daily_blocks(view, a: str, k: int = 5):
    """Non-overlapping k-day sums of daily log excess returns (NaN day = 0), anchored
    at row 0 of the panel's daily calendar, complete blocks inside the asset's daily
    span only. Returns (sums, month-end of each block's last day)."""
    D = view.excess_daily[a].to_numpy()
    ok = np.isfinite(D)
    if ok.sum() < k:
        return np.array([]), pd.DatetimeIndex([])
    first, last = int(np.argmax(ok)), len(ok) - 1 - int(np.argmax(ok[::-1]))
    b0, b1 = -(-first // k), (last + 1) // k
    if b1 <= b0:
        return np.array([]), pd.DatetimeIndex([])
    Z = np.where(ok, D, 0.0)[b0 * k:b1 * k].reshape(-1, k).sum(axis=1)
    ends = view.daily.index[np.arange(b0, b1) * k + k - 1]
    return Z, ends.to_period("M").to_timestamp("M")


def scale_ratios(view, a: str, blocks, block_month, cutoff, k: int = 5, prior_months: float = 60.0):
    """Per-block factor kappa mapping k-day variance to the 21-day (calendar-month) scale.

    For each run of constant quality tier (a splice changes the tier), kappa =
    [sum r_m^2 / sum trading days_m] / [sum s_w^2 / (k n_w)] over the complete
    daily months of that run up to ``cutoff``, shrunk towards 1 in logs with
    weight n/(n + prior_months) (prior worth 5 years: where the ratio's sampling
    error is about the size of the true departures, +-15-20% in variance).
    """
    Xm = view.excess_monthly[a]
    tiers = view.tiers[a]
    D = view.excess_daily[a]
    okd = D.notna()
    rows = view.daily.index.to_period("M").to_timestamp("M")
    n_rows = pd.Series(1, index=rows).groupby(level=0).sum()
    first_m = okd.idxmax().to_period("M").to_timestamp("M")
    last_m = okd[::-1].idxmax().to_period("M").to_timestamp("M")
    # complete daily months: after the (possibly partial) first one, and no later
    # than the last month the view has completed
    hi = min(pd.Timestamp(cutoff), view.monthly.index[-1], last_m)
    months = n_rows.index[(n_rows.index > first_m) & (n_rows.index <= hi)]
    tier_m = tiers.reindex(block_month).to_numpy()
    seg = np.zeros(len(blocks), int)
    if len(blocks):
        chg = np.r_[False, tier_m[1:] != tier_m[:-1]]
        seg = np.cumsum(chg)
    kappa_seg = {}
    bm = pd.Series(blocks ** 2, index=block_month)
    for sgi in np.unique(seg):
        sel = seg == sgi
        mset = pd.DatetimeIndex(np.unique(block_month[sel]))
        mset = mset[mset.isin(months)]
        n = len(mset)
        if n == 0:
            kappa_seg[sgi] = 1.0
            continue
        pm = float((Xm.reindex(mset) ** 2).sum() / n_rows.reindex(mset).sum())
        wk = bm[bm.index.isin(mset)]
        pw = float(wk.sum() / (k * len(wk))) if len(wk) else np.nan
        if not (np.isfinite(pm) and np.isfinite(pw) and pm > 0 and pw > 0):
            kappa_seg[sgi] = 1.0
            continue
        w = n / (n + prior_months)
        kappa_seg[sgi] = float(np.exp(w * np.log(pm / pw)))
    return np.array([kappa_seg[s] for s in seg]) if len(blocks) else np.array([])


def weekly_y(view, a: str, est, k: int = 5):
    """k-day sums mapped to the 21-day scale with the splice-segment ratios known at ``est``.

    Blocks after ``est`` keep the ratio of their segment as of ``est`` (or 1 if a
    new segment began after it), so the series is fixed within a year.
    Returns (y, block months, number of blocks up to est).
    """
    Z, bmon = daily_blocks(view, a, k)
    if len(Z) == 0:
        return Z, bmon, 0
    sel = np.asarray(bmon <= est)
    n_fit = int(sel.sum())
    kap = np.ones(len(Z))
    if n_fit:
        kap[:n_fit] = scale_ratios(view, a, Z[:n_fit], bmon[:n_fit], est, k)
        if n_fit < len(Z):
            tiers = view.tiers[a].reindex(bmon).to_numpy()
            same = np.cumprod(tiers[n_fit:] == tiers[n_fit - 1]).astype(bool)
            kap[n_fit:] = np.where(same, kap[n_fit - 1], 1.0)
    return Z * np.sqrt(kap), bmon, n_fit


def year_days(view) -> int:
    """Trading days in the last 12 months of the view's calendar (about 252; 295 before 1952)."""
    idx = view.daily.index
    if len(idx) == 0:
        return 252
    return int((idx > idx[-1] - pd.DateOffset(years=1)).sum())


class MSWeekly(_RegimeBase):
    """Per-asset K-state MS Student-t model on 5-day sums of daily returns, mapped to
    the 21-day scale per splice segment; assets with under 5 years of daily data use
    the monthly model of the same K.

    Annual variance = N * mean_{h=1..N/5} E[y^2_{T+h}] / 5, N = trading days in a year.
    """

    def __init__(self, K: int = 2, k: int = 5, min_years: float = 5.0):
        super().__init__()
        self.K, self.k = K, k
        self.min_blocks = int(min_years * 252 / k)
        self.monthly = MSMonthly(K)

    def fit_asset(self, view, a):
        est = estimation_date(view)
        y, bmon, n_fit = weekly_y(view, a, est, self.k)
        if n_fit < self.min_blocks:
            return None, y
        key = ("wk", a, est, fingerprint(y[:n_fit]))
        if key not in self._cache:
            try:
                self._cache[key] = fit_ms_t(y[:n_fit], self.K, NU, smooth_window=26)
            except Exception:
                self._cache[key] = None
        return self._cache[key], y

    def _variance(self, view, assets):
        out = self.monthly._variance(view, assets)
        n_days = year_days(view)
        horizon = max(int(round(n_days / self.k)), 1)
        for a in assets:
            f, y = self.fit_asset(view, a)
            if f is not None:
                out[a] = n_days * f.forecast(y, horizon)[0] / self.k
        return out


# ------------------------------------------------------- common regime
class CommonMonthly(_RegimeBase):
    """One K-state regime shared by all assets (a global risk-on / risk-off state),
    asset-specific variances per regime, estimated jointly on monthly returns."""

    def __init__(self, K: int = 3, min_months: int = 120, min_asset_months: int = 24,
                 proportional: bool = False):
        super().__init__()
        self.K, self.min_months, self.min_asset_months = K, min_months, min_asset_months
        self.proportional = proportional

    def fit(self, view):
        X = view.excess_monthly
        est = estimation_date(view)
        Xe = X.loc[:est]
        cols = [a for a in X.columns if Xe[a].notna().sum() >= self.min_asset_months]
        if not cols:
            return None, cols
        Xe = Xe[cols]
        Xe = Xe.loc[Xe.notna().any(axis=1).idxmax():]
        if len(Xe) < self.min_months:
            return None, cols
        key = ("common", est, tuple(cols), fingerprint(Xe.to_numpy()))
        if key not in self._cache:
            try:
                R = nearest_corr(pairwise_corr(Xe), floor=1e-2)
                self._cache[key] = fit_common_ms_t(Xe.to_numpy(), R, self.K, NU, assets=cols,
                                                  proportional=self.proportional)
            except Exception:
                self._cache[key] = None
        return self._cache[key], cols

    def _variance(self, view, assets):
        f, cols = self.fit(view)
        if f is None:
            return {}
        X = view.excess_monthly[cols]
        X = X.loc[X.notna().any(axis=1).idxmax():]
        prob = f.filter(X.to_numpy())[-1]
        out = {}
        for a in assets:
            if a in f.assets:
                out[a] = 12.0 * expected_path(prob, f.P, f.var[f.assets.index(a)], HORIZON_M)
        return out


# --------------------------------------------------------------- BOCPD
class BOCPDMonthly(_RegimeBase):
    """Bayesian online change-point detection on each asset's monthly returns.

    Segments: x_t ~ N(0, sigma^2), sigma^2 ~ InvGamma(alpha0, (alpha0-1) v0),
    v0 = the asset's mean squared return up to the estimation date; constant
    hazard H. Forecast: average over the next 12 months of the posterior mean
    segment variance, mixing in v0 for months after a possible future break.
    """

    def __init__(self, hazard: float = 1 / 240, alpha0: float = 3.0):
        super().__init__()
        self.hazard, self.alpha0 = hazard, alpha0

    def run(self, tag, a, est, x, n_est, hz):
        """Run-length posterior after all of ``x``; prior mean from x[:n_est].

        The state is cached per (series, asset, estimation date) and extended
        incrementally; a cold call recomputes exactly the same steps."""
        xe = x[:n_est]
        base = xe if np.isfinite(xe).sum() >= 12 else x
        if len(base) == 0 or not np.isfinite(base).any():
            return None, np.nan
        v0 = float(np.nanmean(base ** 2))
        if not (np.isfinite(v0) and v0 > 0):
            return None, v0
        b0 = (self.alpha0 - 1.0) * v0
        key = (tag, a, est, v0, hz)
        st = None
        hit = self._cache.get(key)
        if hit is not None:
            n_done, fp, st0 = hit
            if n_done <= len(x) and fingerprint(x[:n_done]) == fp:
                st = st0.copy()
        if st is None:
            st = BOCPDState(self.alpha0, b0)
        st = bocpd_update(st, x[st.n:], hz, self.alpha0, b0)
        self._cache[key] = (len(x), fingerprint(x), st.copy())
        return st, v0

    def state(self, view, a, hazard=None):
        est = estimation_date(view)
        s = asset_series(view.excess_monthly, a)
        return self.run("m", a, est, s.to_numpy(), int((s.index <= est).sum()), hazard or self.hazard)

    def _variance(self, view, assets):
        out = {}
        for a in assets:
            st, v0 = self.state(view, a)
            if st is not None:
                out[a] = 12.0 * bocpd_forecast(st, self.hazard, v0, HORIZON_M)
        return out


class BOCPDEmpiricalBayes(BOCPDMonthly):
    """BOCPD whose hazard is estimated inside the model: each year, the hazard on a
    grid of round expected segment lengths (1, 3, 5, 10, 20, 40 years) that
    maximizes the marginal likelihood of all assets' monthly returns up to the
    estimation date (one hazard shared by all assets)."""

    GRID_YEARS = (1, 3, 5, 10, 20, 40)

    def __init__(self, alpha0: float = 3.0, min_obs: int = 60):
        super().__init__(1 / 240, alpha0)
        self.min_obs = min_obs

    def chosen_hazard(self, view) -> float:
        est = estimation_date(view)
        X = view.excess_monthly.loc[:est]
        cols = [a for a in X.columns if X[a].notna().sum() >= self.min_obs]
        if not cols:
            return self.hazard
        key = ("eb", est, tuple(cols), fingerprint(X[cols].to_numpy()))
        if key not in self._cache:
            ev = {}
            for yrs in self.GRID_YEARS:
                hz = 1.0 / (12 * yrs)
                tot = 0.0
                for a in cols:
                    xe = asset_series(X, a).to_numpy()
                    v0 = float(np.nanmean(xe ** 2))
                    st = bocpd_update(BOCPDState(self.alpha0, (self.alpha0 - 1) * v0), xe, hz,
                                      self.alpha0, (self.alpha0 - 1) * v0)
                    tot += st.loglik
                ev[hz] = tot
            self._cache[key] = max(ev, key=ev.get)
        return self._cache[key]

    def _variance(self, view, assets):
        hz = self.chosen_hazard(view)
        out = {}
        for a in assets:
            st, v0 = self.state(view, a, hz)
            if st is not None:
                out[a] = 12.0 * bocpd_forecast(st, hz, v0, HORIZON_M)
        return out


class BOCPDWeekly(BOCPDMonthly):
    """BOCPD on 5-day sums mapped to the 21-day scale (as in MSWeekly), with the
    same expected segment length in calendar time (hazard per block = H * 12 * 5
    / 252); assets with under 5 years of daily data use the monthly version."""

    def __init__(self, hazard: float = 1 / 240, alpha0: float = 3.0, k: int = 5, min_years: float = 5.0):
        super().__init__(hazard, alpha0)
        self.k = k
        self.min_blocks = int(min_years * 252 / k)
        self.hazard_block = hazard * 12.0 * k / 252.0

    def state_weekly(self, view, a):
        est = estimation_date(view)
        y, _, n_fit = weekly_y(view, a, est, self.k)
        if n_fit < self.min_blocks:
            return None, np.nan
        return self.run("w", a, est, y, n_fit, self.hazard_block)

    def _variance(self, view, assets):
        out = super()._variance(view, assets)
        n_days = year_days(view)
        horizon = max(int(round(n_days / self.k)), 1)
        for a in assets:
            st, v0 = self.state_weekly(view, a)
            if st is not None:
                out[a] = n_days * bocpd_forecast(st, self.hazard_block, v0, horizon) / self.k
        return out


# ---------------------------------------------------- ICSS level + blend
class BreakBlend(_RegimeBase):
    """``ref_blend`` with the long-run level measured only since the last detected break.

    Breaks: binary segmentation with the kappa-2 test (HAC, valid under
    volatility clustering) at the 1% level, segments of at least 24 months,
    on the asset's monthly squared returns up to the estimation date.
    Forecast variance = sqrt(level_since_break * last_60m), both annualized.
    """

    def __init__(self, level: float = 0.01, min_seg: int = 24, recent: int = 60):
        super().__init__()
        self.level, self.min_seg, self.recent = level, min_seg, recent

    def breaks(self, view, a) -> list[pd.Timestamp]:
        est = estimation_date(view)
        s = view.excess_monthly[a].loc[:est].dropna()
        key = ("icss", a, est, fingerprint(s.to_numpy()))
        if key not in self._cache:
            pos = icss_kappa2(s.to_numpy(), self.level, self.min_seg) if len(s) >= 2 * self.min_seg else []
            self._cache[key] = [s.index[p] for p in pos]
        return self._cache[key]

    def _variance(self, view, assets):
        X = view.excess_monthly[assets]
        recent = window_var(X, self.recent)
        out = {}
        for a in assets:
            br = self.breaks(view, a)
            s = X[a].dropna()
            if br:
                s = s.loc[br[-1]:]
            level = 12.0 * float((s ** 2).mean())
            out[a] = float(np.sqrt(level * recent[a]))
        return out
