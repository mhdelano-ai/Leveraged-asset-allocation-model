"""Conditional-mean models: standardized-deviation target, pooled across assets.

What every model here predicts
------------------------------
The benchmark is each asset's own expanding mean, so a pooled model that
predicts raw h-month returns spends its capacity relearning asset means and is
dominated by the high-volatility assets. Every model here instead predicts the
*standardized deviation* of the target from the asset's own mean::

    z[s, a] = (y[s, a] - m_a) / (sqrt(h) * sigma_a)

with ``m_a`` the asset's mean h-month target over the training window and
``sigma_a`` the sd of its monthly excess log returns through the origin. The
forecast is mapped back as::

    mean[t, a] = h * mean(xr_a through t) + b * sqrt(h) * sigma_a * zhat[t, a]

whose first term is exactly ``hist_mean``. So a model with no signal (zhat = 0
or b = 0) reproduces the benchmark, and anything it adds is a bet on deviations.

Calibration shrinkage ``b``
---------------------------
``PointForecaster`` already fits the model on purged k-fold splits of the
training window to get out-of-fold residuals. We reuse those out-of-fold
predictions to estimate a Mincer-Zarnowitz slope of the realized z on the
predicted z, pooled across assets, clipped to [0, 1]. The final forecast is
shrunk by it: a model whose deviations do not line up with outcomes inside the
training window falls back toward the benchmark (b = 0 is hist_mean exactly).
With fewer than two usable folds, b = 0. This is the only data-driven setting;
everything else is fixed a priori (see "Hyperparameters").

Learners (combined by simple averaging of zhat when more than one is given)
---------------------------------------------------------------------------
* ``ridge``  linear, common slopes; median-imputed, standardized features,
  penalty ``alpha * n_rows`` with alpha fixed. ``ridge_g`` adds slopes that
  differ by asset group (features x group dummies, same penalty).
* ``combo``  forecast combination (Rapach-Strauss-Zhou): one univariate OLS
  slope per feature *per asset group*; zhat is the mean over features.
* ``gbm``    LightGBM, depth-2 trees, learning rate 0.01, 200 rounds, large
  leaves (100*sqrt(h) rows), feature/row subsampling, L2 penalty, averaged
  over 3 seeds; native NaN handling; asset-group dummies as extra inputs.
* ``rf``     random forest (scikit-learn), depth <= 4, leaves >= 150*sqrt(h)
  rows, 1/3 of features per split, half-samples, 100 trees; native NaN.

Asset groups: equity (us_eq, dev_eq, em_eq, reits), rates (ust_int, ust_long,
tips), credit (ig_credit, hy_credit), real (gold, commodities).

Linear learners (ridge, combo) median-impute and standardize on the training
window, winsorize at +-5 sd, and ignore a column until it has 24 distinct
observed values in the window (a series that has just started, like VIX in
1990, otherwise has sd ~ 0 and its standardized value explodes).

Hyperparameters
---------------
Fixed a priori, not tuned on the walk-forward sample, because with ~25-300
independent observations per asset purged CV cannot reliably distinguish
settings whose true R^2 differ by a fraction of a percent (``gbm_cv`` is the
one variant that tunes the number of boosting rounds by purged expanding CV,
including the option of zero rounds, as a test of exactly that question).

Feature sets
------------
``feature_set`` is a string: ``"all"`` (``select.columns()``, 82 columns),
``"all-nohind"`` (drops the hindsight features), or families joined by ``+``
(``"price"``, ``"macro+sentiment"``), optionally with ``-nohind``; ``None``
uses every column handed in. If none of the named columns are in the panel
(e.g. on ``evaluation.synthetic`` data) all columns are used.
"""
from __future__ import annotations

import copy
import logging
import warnings
from functools import partial

import numpy as np
import pandas as pd

from ..evaluation.cv import purged_splits
from .base import InsufficientDataError, PointForecaster, TrainingSet
from .registry import register

log = logging.getLogger(__name__)

ASSET_GROUP = {
    "us_eq": "equity", "dev_eq": "equity", "em_eq": "equity", "reits": "equity",
    "ust_int": "rates", "ust_long": "rates", "tips": "rates",
    "ig_credit": "credit", "hy_credit": "credit",
    "gold": "real", "commodities": "real",
}
GROUPS = ("equity", "rates", "credit", "real")
MIN_DISTINCT = 24   # distinct observed values a column needs before linear learners use it
CLIP_Z = 5.0        # standardized features are winsorized at +-CLIP_Z


def _group_codes(assets) -> np.ndarray:
    return np.array([GROUPS.index(ASSET_GROUP.get(a, "equity")) for a in assets], dtype=int)


def resolve_columns(feature_set, available) -> list:
    """Columns for ``feature_set`` (see module docstring), in panel order."""
    available = list(available)
    if feature_set is None:
        return available
    spec = str(feature_set)
    hind = True
    if spec.endswith("-nohind"):
        spec, hind = spec[: -len("-nohind")], False
    fams = None if spec == "all" else tuple(spec.split("+"))
    try:
        from ..features import select
        wanted = select.columns(families=fams, hindsight=hind)
    except (FileNotFoundError, OSError):
        wanted = []
    cols = [c for c in available if c in set(wanted)]
    if not cols:
        log.info("feature_set %r matches no columns; using all %d", feature_set, len(available))
        return available
    return cols


# --------------------------------------------------------------------------
# Learners: fit(Xm, z, g, h, seed) on numpy arrays; predict(Xm, g) -> zhat
# --------------------------------------------------------------------------
def _impute_standardize(Xm):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN columns early on
        med = np.nanmedian(Xm, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    Z = np.where(np.isfinite(Xm), Xm, med)
    mu, sd = Z.mean(axis=0), Z.std(axis=0)
    # A column needs MIN_DISTINCT distinct observed values in the window to be used;
    # otherwise (a series that has just started, e.g. VIX in 1990) its sd is ~0 and
    # standardizing explodes. Unused columns are zeroed via use=0.
    use = np.array([len(np.unique(c[np.isfinite(c)])) >= MIN_DISTINCT for c in Xm.T], dtype=bool)
    use &= sd > 1e-12
    sd = np.where(use, sd, 1.0)
    prep = (med, mu, sd, use.astype(float))
    return _apply_prep(Xm, prep), prep


def _apply_prep(Xm, prep):
    med, mu, sd, use = prep
    Z = (np.where(np.isfinite(Xm), Xm, med) - mu) / sd
    return np.clip(Z, -CLIP_Z, CLIP_Z) * use


class RidgeLearner:
    def __init__(self, alpha: float = 2.0, by_group: bool = False):
        self.alpha, self.by_group = alpha, by_group

    def _design(self, Z, g):
        if not self.by_group:
            return Z
        D = np.zeros((len(g), len(GROUPS)))
        D[np.arange(len(g)), g] = 1.0
        return np.hstack([Z] + [Z * D[:, [k]] for k in range(len(GROUPS))])

    def fit(self, Xm, z, g, h, seed):
        Z, self._prep = _impute_standardize(Xm)
        A = self._design(Z, g)
        p = A.shape[1]
        self._beta = np.linalg.solve(A.T @ A + self.alpha * len(z) * np.eye(p), A.T @ z)
        return self

    def predict(self, Xm, g):
        return self._design(_apply_prep(Xm, self._prep), g) @ self._beta


class ComboLearner:
    """Mean of univariate per-group OLS forecasts (one per feature)."""

    def fit(self, Xm, z, g, h, seed):
        Z, self._prep = _impute_standardize(Xm)
        self._slope = np.zeros((len(GROUPS), Z.shape[1]))
        self._center = np.zeros((len(GROUPS), Z.shape[1]))
        for k in range(len(GROUPS)):
            m = g == k
            if m.sum() < 24:
                continue
            Zk = Z[m]
            c = Zk.mean(axis=0)
            Zc = Zk - c
            den = (Zc ** 2).sum(axis=0)
            self._slope[k] = np.where(den > 0, Zc.T @ (z[m] - z[m].mean()) / np.where(den > 0, den, 1), 0.0)
            self._center[k] = c
        return self

    def predict(self, Xm, g):
        Z = _apply_prep(Xm, self._prep)
        return ((Z - self._center[g]) * self._slope[g]).mean(axis=1)


def _with_groups(Xm, g):
    D = np.zeros((len(g), len(GROUPS)))
    D[np.arange(len(g)), g] = 1.0
    return np.hstack([Xm, D])


class GBMLearner:
    def __init__(self, n_estimators: int = 200, learning_rate: float = 0.01, num_leaves: int = 4,
                 max_depth: int = 2, leaf_rows: int = 100, feature_fraction: float = 0.5,
                 bagging_fraction: float = 0.5, reg_lambda: float = 10.0, seeds: int = 3,
                 cv_rounds: bool = False):
        self.n_estimators, self.learning_rate = n_estimators, learning_rate
        self.num_leaves, self.max_depth, self.leaf_rows = num_leaves, max_depth, leaf_rows
        self.feature_fraction, self.bagging_fraction = feature_fraction, bagging_fraction
        self.reg_lambda, self.seeds, self.cv_rounds = reg_lambda, seeds, cv_rounds

    def _model(self, h, seed, n):
        import lightgbm as lgb
        return lgb.LGBMRegressor(
            n_estimators=n, learning_rate=self.learning_rate, num_leaves=self.num_leaves,
            max_depth=self.max_depth, min_child_samples=int(self.leaf_rows * np.sqrt(h)),
            colsample_bytree=self.feature_fraction, subsample=self.bagging_fraction,
            subsample_freq=1, reg_lambda=self.reg_lambda, random_state=seed % (2**31 - 1), n_jobs=1,
            verbose=-1)

    def _choose_rounds(self, A, z, h, seed, dates):
        """Purged expanding CV over the number of rounds (0 allowed)."""
        grid = np.array([0, 25, 50, 100, 200, 400])
        folds = purged_splits(dates, h, 4, mode="expanding")
        if not folds:
            return 0
        loss = np.zeros(len(grid))
        for tr, va in folds:
            m = self._model(h, seed, int(grid.max())).fit(A[tr], z[tr])
            base = z[tr].mean()
            for j, n in enumerate(grid):
                pred = np.full(len(va), base) if n == 0 else m.predict(A[va], num_iteration=int(n))
                loss[j] += ((z[va] - pred) ** 2).sum()
        return int(grid[int(np.argmin(loss))])

    def fit(self, Xm, z, g, h, seed, dates=None):
        A = _with_groups(Xm, g)
        n = self.n_estimators
        if self.cv_rounds:
            n = self._choose_rounds(A, z, h, seed, dates)
        self.rounds_ = n
        self._models = [] if n == 0 else [self._model(h, seed + k, n).fit(A, z) for k in range(self.seeds)]
        self._const = float(z.mean())
        return self

    def predict(self, Xm, g):
        if not self._models:
            return np.full(len(Xm), self._const)
        A = _with_groups(Xm, g)
        return np.mean([m.predict(A) for m in self._models], axis=0)


class RFLearner:
    def __init__(self, n_estimators: int = 100, max_depth: int = 4, leaf_rows: int = 150,
                 max_features: float = 0.33, max_samples: float = 0.5):
        self.n_estimators, self.max_depth, self.leaf_rows = n_estimators, max_depth, leaf_rows
        self.max_features, self.max_samples = max_features, max_samples

    def fit(self, Xm, z, g, h, seed):
        from sklearn.ensemble import RandomForestRegressor
        A = _with_groups(Xm, g)
        self._m = RandomForestRegressor(
            n_estimators=self.n_estimators, max_depth=self.max_depth,
            min_samples_leaf=int(self.leaf_rows * np.sqrt(h)), max_features=self.max_features,
            max_samples=self.max_samples, bootstrap=True, n_jobs=1, random_state=seed % (2**31))
        self._m.fit(A, z)
        return self

    def predict(self, Xm, g):
        return self._m.predict(_with_groups(Xm, g))


LEARNERS = {
    "ridge": RidgeLearner,
    "ridge_g": partial(RidgeLearner, by_group=True),
    "combo": ComboLearner,
    "gbm": GBMLearner,
    "gbm_cv": partial(GBMLearner, cv_rounds=True),
    "rf": RFLearner,
}


# --------------------------------------------------------------------------
# The forecaster
# --------------------------------------------------------------------------
class ZScoreMean(PointForecaster):
    """Pooled point model of standardized deviations from each asset's mean.

    ``learners``: names from ``LEARNERS``; their zhat are averaged.
    ``calibrate``: shrink zhat by the clipped out-of-fold Mincer-Zarnowitz slope.
    ``feature_set``: see module docstring.
    """

    def __init__(self, learners=("gbm",), feature_set="all", calibrate: bool = True):
        self.learners = tuple(learners)
        self.feature_set = feature_set
        self.calibrate = calibrate
        unknown = set(self.learners) - set(LEARNERS)
        if unknown:
            raise ValueError(f"unknown learners {sorted(unknown)}")

    # -- z-target plumbing -------------------------------------------------
    def _arrays(self, X):
        if getattr(self, "_cols", None) is None:
            self._cols = resolve_columns(self.feature_set, X.columns)
        assets = X.index.get_level_values("asset")
        return X[self._cols].to_numpy(dtype=float), _group_codes(assets), np.asarray(assets)

    def _scale_of(self, assets) -> np.ndarray:
        return np.array([self._zscale.get(a, self._zscale_default) for a in assets], dtype=float)

    def fit_point(self, X, y, horizon):
        if len(y) < 24:
            raise InsufficientDataError("too few rows")
        Xm, g, assets = self._arrays(X)
        ya = y.to_numpy(dtype=float)
        s = pd.Series(ya).groupby(assets)
        self._fold_anchor = s.mean().to_dict()
        self._fold_default = float(ya.mean())
        if not hasattr(self, "_zscale"):  # fit_point used on its own
            sd = s.std(ddof=1)
            self._zscale = sd[sd > 0].to_dict()
            self._zscale_default = float(np.nanstd(ya)) or 1.0
        anchor = np.array([self._fold_anchor[a] for a in assets])
        z = (ya - anchor) / self._scale_of(assets)
        dates = X.index.get_level_values("date")
        seed = getattr(self, "_seed", 0)
        self._fitted = []
        for name in self.learners:
            L = LEARNERS[name]()
            if isinstance(L, GBMLearner):
                L.fit(Xm, z, g, horizon, seed, dates=dates)
            else:
                L.fit(Xm, z, g, horizon, seed)
            self._fitted.append(L)

    def predict_z(self, X) -> np.ndarray:
        """Uncalibrated standardized deviation zhat for each row of X."""
        Xm, g, _ = self._arrays(X)
        return np.mean([L.predict(Xm, g) for L in self._fitted], axis=0)

    def _anchor_of(self, assets, final: bool) -> np.ndarray:
        if final and getattr(self, "_anchor_final", None) is not None:
            return np.array([self._anchor_final.get(a, self._fold_default) for a in assets])
        return np.array([self._fold_anchor.get(a, self._fold_default) for a in assets])

    def predict_point(self, X):
        assets = X.index.get_level_values("asset")
        b = getattr(self, "_b", 1.0)
        return (self._anchor_of(assets, final=True)
                + b * self._scale_of(assets) * self.predict_z(X))

    # -- out-of-fold residuals + calibration slope --------------------------
    def _oof_residuals(self, train: TrainingSet) -> pd.Series:
        X, y = train.X, train.y
        splits = purged_splits(X, train.horizon, self.resid_splits, mode="kfold", min_train_dates=24)
        if len(splits) < 2:
            self._b = 0.0 if self.calibrate else 1.0
            self.oof_ = None
            return pd.Series(dtype=float)
        assets = np.asarray(X.index.get_level_values("asset"))
        ya = y.to_numpy(dtype=float)
        scale = self._scale_of(assets)
        zd = np.full(len(y), np.nan)
        anc = np.full(len(y), np.nan)
        for tr, va in splits:
            m = copy.deepcopy(self)
            m.fit_point(X.iloc[tr], y.iloc[tr], train.horizon)
            zd[va] = m.predict_z(X.iloc[va])
            anc[va] = m._anchor_of(assets[va], final=False)
        ok = np.isfinite(zd)
        zy = (ya - anc) / scale
        if self.calibrate:
            den = float((zd[ok] ** 2).sum())
            b = float((zd[ok] * zy[ok]).sum() / den) if den > 0 else 0.0
            self._b = float(np.clip(b, 0.0, 1.0))
        else:
            self._b = 1.0
        self.oof_ = pd.DataFrame({"zhat": zd, "z": zy}, index=y.index)[ok]
        pred = anc + self._b * scale * zd
        return pd.Series(ya - pred, index=y.index)[ok]

    def fit(self, train: TrainingSet) -> "ZScoreMean":
        xr = train.xr.dropna(axis=1, how="all")
        h = train.horizon
        sd = xr.std(ddof=1) * np.sqrt(h)
        self._zscale = sd[sd > 0].dropna().to_dict()
        self._zscale_default = float(np.nanmedian(list(self._zscale.values()))) if self._zscale else 1.0
        self._seed = int(train.seed)
        self._anchor_final = None
        super().fit(train)  # OOF residuals (sets _b), final fit_point, residual scales
        self._anchor_final = (h * xr.mean()).to_dict()   # identical to hist_mean's mean
        self.b_ = self._b
        return self


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------
def _reg(name, **kw):
    register(name, partial(ZScoreMean, **kw))


# main configurations
_reg("ridge_z", learners=("ridge",))
_reg("ridge_zg", learners=("ridge_g",))
_reg("combo_z", learners=("combo",))
_reg("gbm_z", learners=("gbm",))
_reg("rf_z", learners=("rf",))
_reg("ens_z", learners=("ridge", "combo", "gbm", "rf"))
# diagnostics
_reg("gbm_z_nocal", learners=("gbm",), calibrate=False)
_reg("gbm_z_cv", learners=("gbm_cv",))
# feature-family ablations and hindsight robustness
for _base_name, _learn in (("gbm_z", ("gbm",)), ("combo_z", ("combo",))):
    for _fs in ("price", "valuation_carry", "macro", "sentiment", "all-nohind"):
        _reg(f"{_base_name}.{_fs.replace('all-', '')}", learners=_learn, feature_set=_fs)
