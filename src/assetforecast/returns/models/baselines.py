"""Benchmarks. Every model is scored against ``hist_mean``; distribution models
are also scored against ``vol_ewma`` so that merely tracking volatility does not
count as predicting returns.

All baselines use only each asset's own excess-return history through the
origin (``train.xr``), refit every month (``refit_every = 1``), and ignore
features.

Definitions at origin t for asset a, horizon h (``xr`` monthly excess log
returns through t, ``Y`` the realized h-month targets, i.e. rows s <= t-h):

* ``hist_mean``   mean = h * mean(xr[:t]); sd = sd(Y) * sqrt(1 + h/T) (the
                  factor adds the estimation error of the mean; T = months of
                  history); Gaussian.
                  THE benchmark (Campbell-Thompson / Goyal-Welch). Using the
                  monthly mean times h rather than the mean of the overlapping
                  realized h-month targets uses the last h-1 months too; the two
                  agree for h = 1.
* ``hist_quantile`` mean as ``hist_mean``; quantiles = mean + empirical
                  quantiles of (Y - mean(Y)), both widened by sqrt(1 + h/T).
* ``zero``        mean = 0; sd = sqrt(mean(Y^2)); Gaussian. ("Excess returns
                  are unpredictable and have no premium.")
* ``vol_ewma``    mean as ``hist_mean``; scale from an EWMA of squared monthly
                  excess returns (lambda = 0.94), projected over h months with
                  the variance mean-reverting to its expanding average at rate
                  phi = 0.9 per month, times the variance ratio
                  var(Y) / (h * var(xr)) to carry any serial correlation in
                  monthly returns into the h-month scale, times sqrt(1 + h/T);
                  Gaussian.
* ``vol_ewma_fhs`` as ``vol_ewma`` but with the shape from empirical quantiles
                  of past standardized h-month returns (filtered historical
                  simulation).

lambda and phi are fixed a priori (RiskMetrics-style monthly values), not tuned
on any sample.
"""
from __future__ import annotations

from functools import partial

import numpy as np
import pandas as pd

from .base import (Forecaster, InsufficientDataError, TrainingSet, TAUS,
                   empirical_z_quantiles, gaussian_frame, location_scale_frame,
                   distribution_frame)
from .registry import register

BENCHMARK = "hist_mean"
VOL_BENCHMARK = "vol_ewma"
BASELINES = ("hist_mean", "hist_quantile", "zero", "vol_ewma", "vol_ewma_fhs")


class _Baseline(Forecaster):
    refit_every = 1
    uses_features = False

    def _assets(self, X: pd.DataFrame) -> pd.Index:
        return X.index.get_level_values("asset")

    def _lookup(self, table: dict, X: pd.DataFrame) -> np.ndarray:
        try:
            return np.array([table[a] for a in self._assets(X)], dtype=float)
        except KeyError as e:
            raise KeyError(f"{type(self).__name__}: asset {e} not seen at fit") from None


def _mean_uncertainty(train: TrainingSet, xr: pd.DataFrame) -> pd.Series:
    """sqrt(1 + h / T): widens the predictive sd for the estimation error of the
    mean (Var(h * mean(xr)) = h^2 sigma^2 / T = Var(Y) h / T)."""
    return np.sqrt(1.0 + train.horizon / xr.notna().sum().clip(lower=1))


def _history(train: TrainingSet):
    xr = train.xr.dropna(axis=1, how="all")
    if xr.empty:
        raise InsufficientDataError("no return history")
    Y = train.realized_targets()[xr.columns]
    return xr, Y


class HistoricalMean(_Baseline):
    """Expanding per-asset historical mean, Gaussian predictive distribution."""

    def fit(self, train: TrainingSet):
        xr, Y = _history(train)
        self._mean = (train.horizon * xr.mean()).to_dict()
        self._sd = (Y.std(ddof=1) * _mean_uncertainty(train, xr)).to_dict()
        return self

    def predict(self, X):
        return gaussian_frame(self._lookup(self._mean, X), self._lookup(self._sd, X), X.index)


class HistoricalQuantile(_Baseline):
    """Historical mean with empirical quantiles of past realized h-month returns."""

    def fit(self, train: TrainingSet):
        xr, Y = _history(train)
        self._mean = (train.horizon * xr.mean()).to_dict()
        infl = _mean_uncertainty(train, xr)
        self._sd = (Y.std(ddof=1) * infl).to_dict()
        self._dq = {}
        for a in Y.columns:
            v = Y[a].dropna().to_numpy()
            self._dq[a] = (np.quantile(v - v.mean(), TAUS) * infl[a] if len(v) >= 2
                           else np.full(len(TAUS), np.nan))
        return self

    def predict(self, X):
        m = self._lookup(self._mean, X)
        dq = np.vstack([self._dq[a] for a in self._assets(X)])
        return distribution_frame(m, self._lookup(self._sd, X), m[:, None] + dq, X.index)


class ZeroMean(_Baseline):
    """Zero excess return; sd is the root mean square of past h-month returns."""

    def fit(self, train: TrainingSet):
        _, Y = _history(train)
        self._sd = np.sqrt((Y ** 2).mean()).to_dict()
        return self

    def predict(self, X):
        return gaussian_frame(np.zeros(len(X)), self._lookup(self._sd, X), X.index)


class VolEWMA(_Baseline):
    """Historical mean with an EWMA-volatility scale (see module docstring)."""

    def __init__(self, lam: float = 0.94, phi: float = 0.9, shape: str = "gaussian"):
        self.lam = lam
        self.phi = phi
        self.shape = shape

    def _h_var(self, ewma_var, lr_var, vr, h):
        k = np.arange(1, h + 1)
        decay = (self.phi ** k).sum()
        return vr * (h * lr_var + decay * (ewma_var - lr_var))

    def fit(self, train: TrainingSet):
        xr, Y = _history(train)
        h = train.horizon
        mu = xr.mean()
        dev2 = (xr - mu) ** 2
        ewma = dev2.ewm(alpha=1 - self.lam, adjust=False, ignore_na=True).mean()
        lr_var = xr.var(ddof=1)
        vr = (Y.var(ddof=1) / (h * lr_var)).clip(0.25, 4.0).fillna(1.0)
        self._mean = (h * mu).to_dict()
        hv = self._h_var(ewma.iloc[-1], lr_var, vr, h).clip(lower=0.05 * h * lr_var)
        self._sd = (np.sqrt(hv) * _mean_uncertainty(train, xr)).to_dict()
        if self.shape == "gaussian":
            self._zq = None
        else:
            # standardized past h-month returns using the scale known at each row date
            hv_path = self._h_var(ewma, lr_var, vr, h).clip(lower=0.05 * h * lr_var, axis=1)
            z = (Y - h * mu) / np.sqrt(hv_path)
            self._zq = empirical_z_quantiles(z.to_numpy().ravel())
        return self

    def predict(self, X):
        m, s = self._lookup(self._mean, X), self._lookup(self._sd, X)
        if self._zq is None:
            return gaussian_frame(m, s, X.index)
        return location_scale_frame(m, s, self._zq, X.index)


register("hist_mean", HistoricalMean)
register("hist_quantile", HistoricalQuantile)
register("zero", ZeroMean)
register("vol_ewma", VolEWMA)
register("vol_ewma_fhs", partial(VolEWMA, shape="empirical"))
