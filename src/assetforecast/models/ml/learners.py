"""Learners for the pooled vol panel. All predict the variance RATIO
y = realized 12-month variance / exp(LR) and are fitted under losses whose
expectation is minimized by the conditional MEAN of y (gamma deviance =
2 x QLIKE, Poisson deviance), so no log-retransformation bias arises and the
monthly-proxy targets (noisier but unbiased) are used consistently.

Hyperparameters are fixed a priori (see the report for reasons) except the
GLM's L2 strength, which is chosen inside the model by blocked time-series
cross-validation (whole calendar blocks across ALL assets, 12-month embargo).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .features import DAILY_FEATURES, MONOTONE_FEATURES

SEED = 0
LEAF_YEARS = 5.0          # a leaf must hold >= 5 independent years' worth of rows
CV_FOLDS = 5
EMBARGO_MONTHS = 12
GLM_ALPHAS = (1e-4, 1e-3, 1e-2, 1e-1)
Y_CLIP = (1e-3, 1e3)
DAILY_FALLBACK = {"d1": "m3", "d3": "m3", "d12": "m12"}


def gamma_deviance(y, mu):
    r = y / mu
    return float(np.mean(r - np.log(r) - 1.0))


def min_leaf(n_rows: int, n_years: float) -> int:
    return int(max(20, np.ceil(n_rows / max(n_years, 1.0) * LEAF_YEARS)))


# ------------------------------------------------------------------ GLM
class GLM:
    """Gamma GLM with log link and L2 penalty: log E[y|x] = b0 + b'x.

    Missing daily features are replaced by their monthly analogue (d1, d3 ->
    m3; d12 -> m12) plus a has-daily indicator; missing macro features by the
    TRAINING mean plus a missing indicator. Columns are standardized with
    training moments. Nothing is fitted on data after the training date.
    """

    def __init__(self, features: list[str], alpha: float | None = None):
        self.features = list(features)
        self.alpha = alpha

    def _prep(self, X: pd.DataFrame, fit: bool) -> np.ndarray:
        X = X[self.features].copy()
        cols = {}
        has_daily = None
        for f in self.features:
            v = X[f].to_numpy(float)
            if f in DAILY_FEATURES:
                fb = X[DAILY_FALLBACK[f]].to_numpy(float) if DAILY_FALLBACK[f] in X else np.zeros(len(X))
                if has_daily is None:
                    has_daily = np.isfinite(v).astype(float)
                v = np.where(np.isfinite(v), v, np.where(np.isfinite(fb), fb, 0.0))
            elif f.startswith("x_") or f.startswith("g_") or f in ("m60",):
                miss = ~np.isfinite(v)
                if fit:
                    self.fill_ = getattr(self, "fill_", {})
                    self.fill_[f] = float(np.nanmean(v)) if np.isfinite(v).any() else 0.0
                v = np.where(miss, self.fill_[f], v)
                if f.startswith("x_"):
                    cols[f + "_na"] = miss.astype(float)
            else:
                v = np.where(np.isfinite(v), v, 0.0)
            cols[f] = v
        if has_daily is not None:
            cols["has_daily"] = has_daily
        M = np.column_stack([cols[c] for c in cols])
        if fit:
            self.names_ = list(cols)
            self.mu_ = M.mean(axis=0)
            sd = M.std(axis=0)
            self.sd_ = np.where(sd > 1e-12, sd, 1.0)
        return (M - self.mu_) / self.sd_

    def _fit_alpha(self, X, y, alpha):
        from sklearn.linear_model import GammaRegressor
        M = self._prep(X, fit=True)
        m = GammaRegressor(alpha=alpha, max_iter=1000, tol=1e-6)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.fit(M, y)
        return m

    def fit(self, X: pd.DataFrame, y: np.ndarray, origin: np.ndarray):
        if self.alpha is None:
            self.alpha_ = choose_by_blocked_cv(
                lambda a, Xt, yt: GLM(self.features, alpha=a).fit(Xt, yt, None),
                GLM_ALPHAS, X, y, origin)
        else:
            self.alpha_ = self.alpha
        self.model_ = self._fit_alpha(X, y, self.alpha_)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model_.predict(self._prep(X, fit=False))

    def importance(self) -> dict:
        return dict(zip(self.names_, self.model_.coef_))


def blocked_folds(origin: np.ndarray, n_folds: int = CV_FOLDS, embargo: int = EMBARGO_MONTHS):
    """Contiguous calendar blocks of origins (all assets together); training
    excludes origins within ``embargo`` months of the block on either side
    (targets overlap by 11 months)."""
    o = pd.DatetimeIndex(origin)
    u = np.unique(o.values)
    edges = np.array_split(u, n_folds)
    for blk in edges:
        if len(blk) == 0:
            continue
        lo, hi = pd.Timestamp(blk[0]), pd.Timestamp(blk[-1])
        val = (o >= lo) & (o <= hi)
        tr = (o < lo - pd.DateOffset(months=embargo)) | (o > hi + pd.DateOffset(months=embargo))
        if tr.sum() > 50 and val.sum() > 0:
            yield np.asarray(tr), np.asarray(val)


def choose_by_blocked_cv(make, grid, X, y, origin):
    scores = []
    for a in grid:
        num = den = 0.0
        for tr, va in blocked_folds(origin):
            m = make(a, X.loc[tr].reset_index(drop=True), y[tr])
            mu = np.clip(m.predict(X.loc[va].reset_index(drop=True)), *Y_CLIP)
            num += gamma_deviance(y[va], mu) * va.sum()
            den += va.sum()
        scores.append(num / den if den else np.inf)
    return grid[int(np.argmin(scores))]


# ------------------------------------------------------------------ GBM
class GBM:
    """LightGBM, gamma objective (log link; deviance = 2 x QLIKE), depth-2
    trees, 150 rounds at learning rate 0.05, leaves of >= 5 years' rows, L2 10,
    70% row subsampling; optional monotone (non-decreasing) constraints on the
    trailing-variance features. NaN handled natively."""

    params = dict(objective="gamma", n_estimators=150, learning_rate=0.05, num_leaves=4, max_depth=2,
                  reg_lambda=10.0, subsample=0.7, subsample_freq=1, colsample_bytree=1.0,
                  max_bin=63, min_child_weight=1e-3)

    def __init__(self, features: list[str], monotone: bool = True):
        self.features = list(features)
        self.monotone = monotone

    def fit(self, X, y, origin):
        import lightgbm as lgb
        n_years = len(np.unique(origin)) / 12.0
        p = dict(self.params)
        p.update(min_child_samples=min_leaf(len(y), n_years), random_state=SEED, n_jobs=1,
                 deterministic=True, force_row_wise=True, verbose=-1)
        if self.monotone:
            p["monotone_constraints"] = [1 if f in MONOTONE_FEATURES else 0 for f in self.features]
            p["monotone_constraints_method"] = "advanced"
        self.model_ = lgb.LGBMRegressor(**p)
        self.model_.fit(X[self.features].to_numpy(float), y)
        return self

    def predict(self, X):
        return self.model_.predict(X[self.features].to_numpy(float))

    def importance(self) -> dict:
        g = self.model_.booster_.feature_importance(importance_type="gain")
        g = g / max(g.sum(), 1e-12)
        return dict(zip(self.features, g))


# ------------------------------------------------------------------ RF
class RF:
    """Random forest, Poisson-deviance splits (consistent for the mean of a
    positive target, less dominated by crisis outliers than squared error),
    200 trees, half-sample subsampling, half the features per split, leaves
    of >= 5 years' rows. NaN handled natively (scikit-learn >= 1.4)."""

    def __init__(self, features: list[str]):
        self.features = list(features)

    def fit(self, X, y, origin):
        from sklearn.ensemble import RandomForestRegressor
        n_years = len(np.unique(origin)) / 12.0
        self.model_ = RandomForestRegressor(
            n_estimators=200, criterion="poisson", min_samples_leaf=min_leaf(len(y), n_years),
            max_features=0.5, max_samples=0.5, bootstrap=True, random_state=SEED, n_jobs=1)
        self.model_.fit(X[self.features].to_numpy(float), y)
        return self

    def predict(self, X):
        return self.model_.predict(X[self.features].to_numpy(float))

    def importance(self) -> dict:
        return dict(zip(self.features, self.model_.feature_importances_))


# ------------------------------------------------------------------ average
class Average:
    """Arithmetic mean of the members' variance forecasts (equal weights,
    fixed: estimating combination weights on ~90 noisy years rarely beats
    equal weights; Smith and Wallis 2009)."""

    def __init__(self, members):
        self.members = members

    def fit(self, X, y, origin):
        for m in self.members:
            m.fit(X, y, origin)
        return self

    def predict(self, X):
        return np.mean([m.predict(X) for m in self.members], axis=0)

    def importance(self) -> dict:
        return {}
