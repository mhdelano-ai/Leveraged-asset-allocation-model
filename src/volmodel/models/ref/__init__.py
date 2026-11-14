"""Reference benchmarks: the simplest forecasts that deserve to be beaten.

All of them use only MONTHLY log excess returns, so they treat every asset
and every era alike (monthly data always exists; daily data does not) and are
immune to how a daily source happens to be priced. A squared monthly return
is one draw of the 21-day variance the target measures, so
``12 * mean(r_month ** 2)`` estimates the target directly.

    ref_longrun    all history, equally weighted
    ref_last12m    the last 12 months (a noisy random walk)
    ref_last60m    the last 5 years
    ref_blend      geometric mean of ref_longrun and ref_last60m variances

    ref_corr_longrun   all history, pairwise, repaired to positive definite
    ref_corr_last120m  the last 10 years, shrunk halfway to the long run
    ref_corr_zero      identity (how much do correlations matter at all?)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..base import CorrForecaster, VolForecaster, register

MIN_OBS = 12  # months before a window estimate is trusted on its own


def _monthly(view, assets):
    return view.excess_monthly[assets]


def window_var(x: pd.DataFrame, months: int | None) -> pd.Series:
    """Annualized uncentered variance from the last ``months`` monthly returns
    (all history if None). Falls back to all history when the window is thin."""
    full = 12.0 * (x ** 2).mean()
    if months is None:
        return full
    w = x.iloc[-months:]
    v = 12.0 * (w ** 2).mean()
    thin = w.notna().sum() < min(MIN_OBS, months)
    return v.where(~thin, full)


def nearest_corr(c: np.ndarray, floor: float = 1e-3) -> np.ndarray:
    """Symmetrize, clip eigenvalues at ``floor`` and rescale to unit diagonal."""
    c = (c + c.T) / 2.0
    w, v = np.linalg.eigh(c)
    c = (v * np.maximum(w, floor)) @ v.T
    d = np.sqrt(np.diag(c))
    c = c / np.outer(d, d)
    np.fill_diagonal(c, 1.0)
    return (c + c.T) / 2.0


def pairwise_corr(x: pd.DataFrame, min_overlap: int = 24) -> np.ndarray:
    """Uncentered correlation of monthly returns on each pair's common sample;
    pairs with fewer than ``min_overlap`` common months get 0."""
    a = x.to_numpy(float)
    ok = np.isfinite(a)
    a0 = np.where(ok, a, 0.0)
    okf = ok.astype(float)
    sxy = a0.T @ a0
    sxx = (a0 ** 2).T @ okf          # sum of x_i^2 over months where j is present
    n = okf.T @ okf
    with np.errstate(invalid="ignore", divide="ignore"):
        c = sxy / np.sqrt(sxx * sxx.T)
    c = np.where((n >= min_overlap) & np.isfinite(c), c, 0.0)
    np.fill_diagonal(c, 1.0)
    return np.clip(c, -1.0, 1.0)


class _WindowVol(VolForecaster):
    def __init__(self, months: int | None):
        self.months = months

    def forecast_vol(self, view, assets):
        return np.sqrt(window_var(_monthly(view, assets), self.months)).reindex(assets)


class _BlendVol(VolForecaster):
    def forecast_vol(self, view, assets):
        x = _monthly(view, assets)
        v = np.exp(0.5 * (np.log(window_var(x, None)) + np.log(window_var(x, 60))))
        return np.sqrt(v).reindex(assets)


class _CorrLongrun(CorrForecaster):
    def forecast_corr(self, view, assets):
        c = nearest_corr(pairwise_corr(_monthly(view, assets)))
        return pd.DataFrame(c, index=assets, columns=assets)


class _CorrRecent(CorrForecaster):
    def __init__(self, months: int = 120, weight: float = 0.5):
        self.months, self.weight = months, weight

    def forecast_corr(self, view, assets):
        x = _monthly(view, assets)
        long = pairwise_corr(x)
        recent = pairwise_corr(x.iloc[-self.months:])
        n = x.iloc[-self.months:].notna().to_numpy(float)
        thin = (n.T @ n) < 24
        recent = np.where(thin, long, recent)
        c = nearest_corr(self.weight * recent + (1 - self.weight) * long)
        return pd.DataFrame(c, index=assets, columns=assets)


class _CorrZero(CorrForecaster):
    def forecast_corr(self, view, assets):
        return pd.DataFrame(np.eye(len(assets)), index=assets, columns=assets)


register("vol", "ref_longrun")(lambda: _WindowVol(None))
register("vol", "ref_last12m")(lambda: _WindowVol(12))
register("vol", "ref_last60m")(lambda: _WindowVol(60))
register("vol", "ref_blend")(_BlendVol)
register("corr", "ref_corr_longrun")(_CorrLongrun)
register("corr", "ref_corr_last120m")(_CorrRecent)
register("corr", "ref_corr_zero")(_CorrZero)
