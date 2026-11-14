"""Forecasters of the GARCH family (see the package docstring)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..base import VolForecaster
from . import core, prep
from .core import CGarch, Garch11, GJR, Midas, RM2006, estimate, fingerprint

WINDOW_YEARS = 20     # daily/weekly estimation and filtering window (round number; several vol cycles,
                      # but not 1930s data in a 1990s fit)
MIN_YEARS = 3         # minimum daily/weekly history in the estimation window before a model is fitted
MIN_MONTHLY_POOL = 120  # pooled monthly fit needs at least 10 asset-years in total
STALE_DAYS = 31       # daily data must reach within a month of the origin
DAYS_PER_YEAR = 252.0


def _safe(x: float) -> bool:
    return x is not None and np.isfinite(x) and x > 0


# ====================================================================== monthly
class MonthlyPooled(VolForecaster):
    """Monthly GARCH-type model with parameters common to all assets (pooled
    Student-t likelihood) and asset-specific variance targets."""

    def __init__(self, spec_cls=Garch11):
        self.spec = spec_cls(12.0)
        self._fits: dict = {}
        self.last_fit: core.Fit | None = None

    def _series(self, ex_m: pd.DataFrame, a: str, end=None) -> np.ndarray:
        s = ex_m[a]
        if end is not None:
            s = s.loc[:end]
        return s.dropna().to_numpy(float)

    def _fit(self, ex_m: pd.DataFrame, E) -> core.Fit:
        data = [self._series(ex_m, a, E) for a in ex_m.columns]
        data = [d for d in data if len(d) >= 36]
        key = (E, fingerprint(*data) if data else "none")
        if key in self._fits:
            return self._fits[key]
        if sum(len(d) for d in data) < MIN_MONTHLY_POOL:
            f = core.Fit(np.asarray(self.spec.starts[0], float), float("nan"), False, "too little data")
        else:
            vb = [float(np.mean(d * d)) for d in data]
            ntot = sum(len(d) for d in data)
            spec = self.spec

            def nll(th):
                return sum(spec.nll(th, d, v) for d, v in zip(data, vb)) / ntot

            f = estimate(spec, nll)
        self._fits[key] = f
        return f

    def annual_var(self, view, assets) -> dict:
        ex_m = view.excess_monthly
        t = view.asof_date if view.asof_date is not None else ex_m.index[-1]
        f = self._fit(ex_m, prep.est_date(t))
        self.last_fit = f
        prm = self.spec.unpack(f.theta)
        out = {}
        for a in assets:
            try:
                e = self._series(ex_m, a)
                vbar = float(np.mean(e * e))
                _, state = self.spec.filter(e, vbar, prm)
                out[a] = float(self.spec.agg_forecast(state, vbar, prm, 12.0))
            except Exception:                      # noqa: BLE001
                out[a] = np.nan
        return out

    def forecast_vol(self, view, assets):
        v = self.annual_var(view, assets)
        return _finish(view, assets, v)


def _last_resort(view, a) -> float:
    """12 x mean squared monthly return over the last 5 years (all history if
    thinner), else 0.15^2."""
    try:
        x = view.excess_monthly[a].dropna()
        w = x.iloc[-60:] if len(x) >= 12 else x
        v = 12.0 * float(np.mean(w.to_numpy() ** 2))
        return v if _safe(v) else 0.0225
    except Exception:                              # noqa: BLE001
        return 0.0225


def _finish(view, assets, var: dict, fallback: dict | None = None) -> pd.Series:
    vals = []
    for a in assets:
        v = var.get(a, np.nan)
        if not _safe(v) and fallback is not None:
            v = fallback.get(a, np.nan)
        if not _safe(v):
            v = _last_resort(view, a)
        vals.append(np.sqrt(v))
    return pd.Series(vals, index=list(assets), dtype=float)


# ============================================================= daily / weekly
class HighFreq(VolForecaster):
    """GARCH-type model fitted per asset on VR-adjusted daily (k=1) or
    non-overlapping 5-day (k=5) log excess returns, over the last 20 years,
    re-estimated each December 31 and filtered with fixed parameters up to the
    origin. Assets without enough daily data use the pooled monthly GARCH(1,1).
    """

    def __init__(self, spec_cls=Garch11, k: int = 1):
        self.k = k
        self.steps_per_year = DAYS_PER_YEAR / k
        self.spec = spec_cls(self.steps_per_year) if spec_cls is not RM2006 else None
        self.rm = RM2006(step_days=k) if spec_cls is RM2006 else None
        self.fallback = MonthlyPooled(Garch11)
        self._fits: dict = {}
        self.fits_log: dict = {}       # (asset, E) -> Fit, for reporting

    # ---------------------------------------------------------------- inputs
    def _series(self, ex_d: pd.Series, tiers: pd.Series, end) -> pd.Series:
        if self.k == 1:
            return prep.daily_adjusted(ex_d, tiers).loc[:end]
        return prep.block_adjusted(ex_d, tiers, end, self.k)

    def _extra(self, s: pd.Series, lo):
        """MIDAS inputs for the steps of ``s`` after ``lo``: month index of each
        step and the per-step realized variance of every month from 36 months
        before ``lo`` on."""
        m = s.index.to_period("M")
        rv = (s ** 2).groupby(m).mean()
        months = pd.period_range(m.min(), m.max(), freq="M")
        rv = rv.reindex(months)
        w = s.loc[s.index > lo]
        first = w.index[0].to_period("M") - core.MIDAS_K
        keep = months >= first
        months_k, rv_k = months[keep], rv.to_numpy(float)[keep]
        pos = pd.Index(months_k).get_indexer(w.index.to_period("M"))
        return w, pos, rv_k

    # ------------------------------------------------------------ estimation
    def _fit(self, a, s_est: pd.Series, E) -> core.Fit | None:
        lo = E - pd.DateOffset(years=WINDOW_YEARS)
        w = s_est.loc[s_est.index > lo]
        if len(w) < MIN_YEARS * self.steps_per_year * 0.95:
            return None
        key = (a, E, fingerprint(s_est.to_numpy()) if self.spec.name == "midas" else fingerprint(w.to_numpy()))
        if key in self._fits:
            return self._fits[key]
        e = w.to_numpy(float)
        vbar = float(np.mean(e * e))
        spec = self.spec
        if spec.name == "midas":
            w2, pos, rv = self._extra(s_est, lo)
            e = w2.to_numpy(float)
            extra = (pos, rv, np.log(vbar))
        else:
            extra = None
        n = len(e)

        def nll(th):
            return spec.nll(th, e, vbar, extra) / n

        f = estimate(spec, nll)
        self._fits[key] = f
        self.fits_log[(a, E)] = f
        return f

    # -------------------------------------------------------------- forecast
    def _one(self, a, ex_d, tiers, t) -> float:
        s_all = self._series(ex_d, tiers, t)
        if len(s_all) == 0 or (t - s_all.index[-1]).days > STALE_DAYS:
            return np.nan
        rows = prep.asset_rows(ex_d).loc[:t]
        H = (prep.rows_last_year(s_all.index, t) if self.k == 1
             else prep.rows_last_year(rows.index, t) / self.k)
        if H < 0.5 * self.steps_per_year:
            return np.nan
        lo = t - pd.DateOffset(years=WINDOW_YEARS)
        w = s_all.loc[s_all.index > lo]
        if self.rm is not None:
            if len(w) < MIN_YEARS * self.steps_per_year * 0.95:
                return np.nan
            e2 = w.to_numpy(float) ** 2
            x0 = float(np.mean(e2[: int(self.steps_per_year)]))
            x1 = self.rm.components(e2, x0)
            return self.rm.agg_forecast(x1, H)
        E = prep.est_date(t)
        s_est = s_all.loc[:E] if self.k == 1 else self._series(ex_d, tiers, E)
        f = self._fit(a, s_est, E)
        if f is None:
            return np.nan
        prm = self.spec.unpack(f.theta)
        e = w.to_numpy(float)
        vbar = float(np.mean(e * e))
        extra = None
        if self.spec.name == "midas":
            w2, pos, rv = self._extra(s_all, lo)
            e = w2.to_numpy(float)
            extra = (pos, rv, np.log(vbar))
        _, state = self.spec.filter(e, vbar, prm, extra)
        return float(self.spec.agg_forecast(state, vbar, prm, H))

    def forecast_vol(self, view, assets):
        t = view.asof_date if view.asof_date is not None else view.monthly.index[-1]
        t = pd.Timestamp(t)
        ex_d = view.excess_daily
        out = {}
        for a in assets:
            try:
                out[a] = self._one(a, ex_d[a], view.tiers[a], t) if a in ex_d.columns else np.nan
            except Exception:                      # noqa: BLE001 - never raise
                out[a] = np.nan
        need = [a for a in assets if not _safe(out.get(a))]
        fb = self.fallback.annual_var(view, need) if need else {}
        return _finish(view, assets, out, fb)
