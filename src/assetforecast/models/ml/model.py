"""The walk-forward wrapper shared by every ML vol model.

At origin t:
  1. tau = the last December month-end <= t (re-estimation at most yearly).
  2. Training set, built from ``view.asof(tau)`` only: every (origin s, asset)
     with s + 12 months <= tau (the target window is complete by tau; this is
     the 12-month purge), the asset eligible at s (>= 36 months), features at
     s and the realized target ratio y = var / exp(LR_s).
  3. If the training set spans fewer than MIN_TRAIN_YEARS distinct years of
     origins, fall back to ref_blend (geometric mean of the long-run and
     last-60-month variances). Otherwise fit the learner (cached per tau and
     per data fingerprint, so a cold call gives the identical model).
  4. Forecast variance = exp(LR_t) * yhat(features at t), clipped to
     [0.05, 20] x the long-run variance; vol = sqrt.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from ..base import VolForecaster
from ..ref import window_var
from .features import FeatureState, realized_targets

MIN_TRAIN_YEARS = 20
RATIO_BOUNDS = (0.05, 20.0)


def training_date(t) -> pd.Timestamp:
    t = pd.Timestamp(t)
    return t if t.month == 12 and t.is_month_end else pd.Timestamp(year=t.year - 1, month=12, day=31)


def _fingerprint(view) -> str:
    h = hashlib.sha1()
    for obj in (view.monthly, view.daily, view.cash_monthly, view.cash_daily):
        h.update(np.ascontiguousarray(obj.to_numpy(float)).tobytes())
        h.update(str(obj.shape).encode())
    p = view.predictors
    h.update(str(len(p)).encode())
    if len(p):
        h.update(np.ascontiguousarray(p["value"].to_numpy(float)).tobytes())
    return h.hexdigest()


def ref_blend_var(view, assets) -> pd.Series:
    x = view.excess_monthly[assets]
    return np.exp(0.5 * (np.log(window_var(x, None)) + np.log(window_var(x, 60))))


class MLVol(VolForecaster):
    def __init__(self, make_learner, features: list[str], min_years: int = MIN_TRAIN_YEARS):
        self.make_learner = make_learner
        self.features = list(features)
        self.min_years = min_years
        self.uses_macro = any(f.startswith("x_") for f in self.features)
        self._cache: dict = {}
        self.log: dict = {}       # tau -> diagnostics (rows, years, importance, alpha)

    # ------------------------------------------------------------ training
    def training_set(self, view_tau):
        fs = FeatureState(view_tau, macro=self.uses_macro)
        P, A, V = realized_targets(view_tau)
        keep = fs.elig[P, A] & np.isfinite(fs.lr[P, A])
        P, A, V = P[keep], A[keep], V[keep]
        X = fs.frame(P, A, self.features)
        y = V / np.exp(fs.lr[P, A])
        origin = fs.grid[P].values
        return X, y, origin

    def trained(self, view, tau):
        vt = view if (view.asof_date is not None and pd.Timestamp(view.asof_date) <= tau) else view.asof(tau)
        key = (tau, _fingerprint(vt))
        if key in self._cache:
            return self._cache[key]
        X, y, origin = self.training_set(vt)
        n_years = len(np.unique(pd.DatetimeIndex(origin).year)) if len(y) else 0
        learner = None
        if n_years >= self.min_years and len(y) >= 200:
            learner = self.make_learner(self.features).fit(X, y, origin)
            self.log[tau] = {"rows": len(y), "years": n_years,
                             "alpha": getattr(learner, "alpha_", None),
                             "importance": learner.importance()}
        self._cache[key] = learner
        return learner

    # ------------------------------------------------------------ forecast
    def forecast_vol(self, view, assets):
        t = pd.Timestamp(view.asof_date) if view.asof_date is not None else pd.Timestamp(view.monthly.index[-1])
        tau = training_date(t)
        learner = self.trained(view, tau)
        fb = ref_blend_var(view, assets)
        if learner is None:
            return np.sqrt(fb).reindex(assets)
        fs = FeatureState(view, macro=self.uses_macro)
        g = fs.row_at(t)
        cols = np.array([fs.assets.index(a) for a in assets])
        X = fs.frame(np.full(len(cols), g), cols, self.features)
        ratio = np.clip(np.asarray(learner.predict(X), float), *RATIO_BOUNDS)
        lr = fs.lr[g, cols]
        var = np.exp(lr) * ratio
        var = np.where(np.isfinite(var) & (var > 0), var, fb.reindex(assets).to_numpy(float))
        return pd.Series(np.sqrt(var), index=assets)
