"""The forecaster interface every model implements.

How the walk-forward engine drives a model
------------------------------------------
For each horizon h the engine (``assetforecast.returns.evaluation.walkforward``) steps
through monthly forecast origins t. At a refit origin it builds a fresh model
from its registry factory and calls ``model.fit(train)`` with a
:class:`TrainingSet`; then, at t and at every origin until the next refit, it
calls ``model.predict(X_t)`` where ``X_t`` holds the feature rows dated t for
the assets eligible at t. ``predict`` returns one row per input row with the
columns ``mean``, ``sd`` and ``q01 .. q99`` (``PREDICTION_COLUMNS``): the
predictive distribution of the h-month excess log return
``y[t] = xr[t+1] + ... + xr[t+h]``.

What the engine guarantees (so a model cannot leak even if it tries)
--------------------------------------------------------------------
* ``train.X`` / ``train.y`` contain only rows with ``date + h <= t`` whose
  target is fully realized at t (the purge), and no NaN targets.
* ``train.xr`` (monthly excess log returns, wide) ends at t, so any target a
  model computes itself from it is NaN unless realized.
* A fresh model object is built at each refit; no state survives from a
  previous refit.
* Predictions are validated: right index, finite, ``sd > 0``, quantiles
  non-decreasing. A violation raises :class:`ForecastContractError`.

What the engine cannot guarantee: that the *features* obey the contract (a row
dated t built from data published after t). That is the feature stage's job;
``assetforecast.returns.evaluation.leakage`` has an audit that catches gross violations.

Writing a model
---------------
Full-distribution model: subclass :class:`Forecaster`, implement ``fit`` and
``predict``; use :func:`distribution_frame` / :func:`gaussian_frame` to build
the output and :func:`sort_quantiles` if your quantiles can cross.

Point model: subclass :class:`PointForecaster`, implement ``fit_point(X, y,
horizon)`` and ``predict_point(X) -> Series``. The base class turns it into a
distribution from out-of-fold residuals on the training window (see its
docstring), so it is scored on the same footing as distribution models.

Hyperparameters: tune inside ``fit`` with
``assetforecast.returns.evaluation.cv.purged_splits`` (or ``train.cv_splits()``). Never
tune on the out-of-sample period.

Then register it (``assetforecast.returns.models.registry``)::

    from assetforecast.returns.models.registry import register

    @register("my_gbm")
    def _make():
        return MyGBM(n_estimators=300)

Any module inside ``assetforecast/returns/models/`` is imported automatically when the
registry is first used, so nothing in the harness needs editing.
"""
from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
import pandas as pd
from scipy import stats

from .. import config, schema
from ..evaluation.cv import month_ordinal, purged_splits

TAUS = np.asarray(config.QUANTILES, dtype=float)
QUANTILE_COLUMNS = list(schema.QUANTILE_COLUMNS)
PREDICTION_COLUMNS = ["mean", "sd", *QUANTILE_COLUMNS]
_Z = stats.norm.ppf(TAUS)


class LeakageError(RuntimeError):
    """Training data would contain information not available at the origin."""


class ForecastContractError(ValueError):
    """A model's predictions violate the forecast contract."""


class InsufficientDataError(RuntimeError):
    """Raise from ``fit`` when the training set is too small to fit at all.

    The engine then skips the origins until the next refit (no forecast is
    written, and those rows drop out of every comparison for that horizon).
    """


# --------------------------------------------------------------------------
# Training data handed to fit
# --------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class TrainingSet:
    """Everything a model may learn from at forecast origin ``origin``.

    Attributes
    ----------
    horizon : months ahead, h.
    origin : the forecast origin t (month end). Forecasts are made at its close.
    xr : wide monthly excess log returns, index <= origin, columns = assets.
    assets : assets eligible for forecasting at ``origin`` (enough history).
    seed : deterministic seed for this (horizon, origin); use it for any RNG.
    X : feature rows (date, asset) with ``date + h <= origin`` and a realized,
        non-missing target (computed lazily, so models that ignore features
        pay nothing).
    y : the matching targets (Series named ``y``), same index as ``X``.
    """

    horizon: int
    origin: pd.Timestamp
    xr: pd.DataFrame
    assets: tuple
    seed: int
    _features: pd.DataFrame = field(repr=False)
    _targets: pd.Series = field(repr=False)   # long, aligned to _features rows

    def __post_init__(self):
        if self.xr.index.max() > self.origin:
            raise LeakageError("TrainingSet.xr extends past the origin")
        if len(self._targets) and self._targets.index.get_level_values("date").max() > self.cutoff:
            raise LeakageError("TrainingSet targets include rows whose target is not realized at the origin")

    @property
    def cutoff_ordinal(self) -> int:
        """Last usable row date, as a month ordinal: origin - h months."""
        return int(month_ordinal([self.origin])[0]) - self.horizon

    @property
    def cutoff(self) -> pd.Timestamp:
        """Last row date whose target is realized at the origin."""
        return self.origin - pd.offsets.MonthEnd(self.horizon)

    @cached_property
    def y(self) -> pd.Series:
        y = self._targets.dropna()
        y.name = "y"
        return y

    @cached_property
    def X(self) -> pd.DataFrame:
        return self._features.loc[self.y.index]

    def realized_targets(self) -> pd.DataFrame:
        """Wide h-month targets computed from ``xr``; NaN wherever unrealized."""
        return schema.make_targets(self.xr, self.horizon)

    def cv_splits(self, n_splits: int = 5, mode: str = "expanding", embargo: int = 0,
                  min_train_dates: int = 24):
        """Purged CV folds over ``self.X`` rows (see ``evaluation.cv``)."""
        return purged_splits(self.X, self.horizon, n_splits, mode, embargo, min_train_dates)


# --------------------------------------------------------------------------
# Output helpers
# --------------------------------------------------------------------------
def gaussian_frame(mean, sd, index: pd.Index) -> pd.DataFrame:
    """Predictive frame for N(mean, sd^2), quantiles on ``config.QUANTILES``."""
    mean = np.asarray(mean, dtype=float).reshape(-1)
    sd = np.asarray(sd, dtype=float).reshape(-1)
    q = mean[:, None] + sd[:, None] * _Z[None, :]
    return distribution_frame(mean, sd, q, index)


def location_scale_frame(mean, sd, z_quantiles, index: pd.Index) -> pd.DataFrame:
    """Predictive frame ``mean + sd * z`` for a standardized shape given by
    its quantiles ``z_quantiles`` on the grid (e.g. empirical standardized
    residual quantiles)."""
    mean = np.asarray(mean, dtype=float).reshape(-1)
    sd = np.asarray(sd, dtype=float).reshape(-1)
    z = np.asarray(z_quantiles, dtype=float).reshape(1, -1)
    q = mean[:, None] + sd[:, None] * z
    return distribution_frame(mean, sd, q, index)


def distribution_frame(mean, sd, quantiles, index: pd.Index) -> pd.DataFrame:
    """Assemble ``PREDICTION_COLUMNS`` from arrays (quantiles: n x len(TAUS))."""
    q = np.asarray(quantiles, dtype=float).reshape(len(index), len(TAUS))
    df = pd.DataFrame(q, index=index, columns=QUANTILE_COLUMNS)
    df.insert(0, "sd", np.asarray(sd, dtype=float).reshape(-1))
    df.insert(0, "mean", np.asarray(mean, dtype=float).reshape(-1))
    return df


def sort_quantiles(df: pd.DataFrame) -> pd.DataFrame:
    """Monotone rearrangement (Chernozhukov et al. 2010): sort each row's
    quantiles. Never worsens pinball loss; use it for quantile regressions."""
    out = df.copy()
    out[QUANTILE_COLUMNS] = np.sort(out[QUANTILE_COLUMNS].to_numpy(dtype=float), axis=1)
    return out


def empirical_z_quantiles(z) -> np.ndarray:
    """Quantiles of standardized draws on the grid, recentred to mean 0."""
    z = np.asarray(z, dtype=float)
    z = z[np.isfinite(z)]
    if len(z) < 20:
        return _Z.copy()
    z = (z - z.mean()) / z.std(ddof=1)
    return np.quantile(z, TAUS)


def validate_predictions(pred: pd.DataFrame, index: pd.Index, where: str = "") -> pd.DataFrame:
    """Check a model's output against the contract; return it aligned to ``index``."""
    tag = f" ({where})" if where else ""
    if not isinstance(pred, pd.DataFrame):
        raise ForecastContractError(f"predict must return a DataFrame{tag}")
    missing = [c for c in PREDICTION_COLUMNS if c not in pred.columns]
    if missing:
        raise ForecastContractError(f"predictions missing columns {missing}{tag}")
    if not pred.index.equals(index):
        if pred.index.has_duplicates or not index.isin(pred.index).all():
            raise ForecastContractError(f"predictions must have exactly one row per input row{tag}")
        pred = pred.loc[index]
    vals = pred[PREDICTION_COLUMNS].to_numpy(dtype=float)
    if not np.isfinite(vals).all():
        raise ForecastContractError(f"predictions contain NaN or inf{tag}")
    if (vals[:, 1] <= 0).any():
        raise ForecastContractError(f"predictive sd must be > 0{tag}")
    if (np.diff(vals[:, 2:], axis=1) < -1e-12).any():
        raise ForecastContractError(
            f"quantiles are not non-decreasing{tag}; apply models.base.sort_quantiles")
    return pred[PREDICTION_COLUMNS]


# --------------------------------------------------------------------------
# The interface
# --------------------------------------------------------------------------
class Forecaster(ABC):
    """A model producing a predictive distribution of the h-month excess log return.

    Class attributes a subclass may override:

    ``refit_every`` : months between refits for this model; ``None`` uses
        ``config.REFIT_EVERY``. Cheap models that should track the expanding
        history every month (the baselines) set 1.
    ``uses_features`` : informational; False for models that only use returns.
    """

    name: str = ""
    refit_every: int | None = None
    uses_features: bool = True

    @abstractmethod
    def fit(self, train: TrainingSet) -> "Forecaster":
        """Fit on the purged training window. Tune hyperparameters here."""

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Rows of X are (origin, asset). Return ``PREDICTION_COLUMNS`` per row."""

    def params(self) -> dict:
        """Settings that identify this model for the forecast cache. Default:
        the public, simple-typed instance attributes. Override if needed."""
        out = {}
        for k, v in sorted(vars(self).items()):
            if k.startswith("_") or k.endswith("_"):
                continue
            if isinstance(v, (int, float, str, bool, type(None), tuple, list)):
                out[k] = v
        return out

    def __repr__(self) -> str:
        args = ", ".join(f"{k}={v!r}" for k, v in self.params().items())
        return f"{type(self).__name__}({args})"


class PointForecaster(Forecaster):
    """Base for models that only produce a point forecast.

    Implement ``fit_point(X, y, horizon)`` and ``predict_point(X) -> Series``
    (or array) of mean forecasts. ``fit`` then:

    1. computes residuals on the training window. With ``residuals="oof"``
       (default) each row's residual comes from a copy of the model fit on the
       other folds of a purged k-fold split, so flexible models cannot shrink
       their own error bands by overfitting; ``"insample"`` uses the in-sample
       fit (only sensible for low-capacity models);
    2. sets each asset's predictive ``sd`` to the RMS of its residuals over
       the training window (assets with fewer than ``min_resid`` residuals fall
       back to the sd of their own realized targets);
    3. sets the shape from pooled residuals standardized by asset
       (``shape="empirical"``) or Gaussian (``shape="gaussian"``);
    4. refits the point model on the full training window.

    The predictive distribution is ``mean + sd_asset * z`` with ``z`` the
    standardized shape, so the scale is constant between refits. A point model
    therefore gets distribution credit only for its mean and for any
    asset-level scale its residuals imply.
    """

    residuals: str = "oof"
    resid_splits: int = 5
    shape: str = "empirical"
    min_resid: int = 24

    @abstractmethod
    def fit_point(self, X: pd.DataFrame, y: pd.Series, horizon: int) -> None:
        """Fit the point model on (X, y)."""

    @abstractmethod
    def predict_point(self, X: pd.DataFrame):
        """Mean forecast for each row of X."""

    def _oof_residuals(self, train: TrainingSet) -> pd.Series:
        X, y = train.X, train.y
        if self.residuals == "oof":
            splits = purged_splits(X, train.horizon, self.resid_splits, mode="kfold",
                                   min_train_dates=24)
            if len(splits) >= 2:
                pred = pd.Series(np.nan, index=y.index)
                for tr, va in splits:
                    m = copy.deepcopy(self)
                    m.fit_point(X.iloc[tr], y.iloc[tr], train.horizon)
                    pred.iloc[va] = np.asarray(m.predict_point(X.iloc[va]), dtype=float).reshape(-1)
                return (y - pred).dropna()
        elif self.residuals != "insample":
            raise ValueError("residuals must be 'oof' or 'insample'")
        return pd.Series(dtype=float)

    def fit(self, train: TrainingSet) -> "PointForecaster":
        X, y = train.X, train.y
        if len(y) == 0:
            raise InsufficientDataError("no training rows")
        resid = self._oof_residuals(train)
        self.fit_point(X, y, train.horizon)
        if resid.empty:  # 'insample' or too little data for folds
            resid = y - np.asarray(self.predict_point(X), dtype=float).reshape(-1)
        assets = resid.index.get_level_values("asset")
        rms = resid.pow(2).groupby(assets).mean().pow(0.5)
        counts = resid.groupby(assets).size()
        realized_sd = train.realized_targets().std()
        scale = {}
        for a in set(train.assets) | set(rms.index):
            if a in rms.index and counts[a] >= self.min_resid and rms[a] > 0:
                scale[a] = float(rms[a])
            elif a in realized_sd.index and np.isfinite(realized_sd[a]) and realized_sd[a] > 0:
                scale[a] = float(realized_sd[a])
        self._scale = scale
        if self.shape == "gaussian":
            self._zq = _Z.copy()
        else:
            s = pd.Series(scale)
            z = resid / s.reindex(assets).to_numpy()
            self._zq = empirical_z_quantiles(z.to_numpy())
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        mean = np.asarray(self.predict_point(X), dtype=float).reshape(-1)
        assets = X.index.get_level_values("asset")
        try:
            sd = np.array([self._scale[a] for a in assets], dtype=float)
        except KeyError as e:
            raise ForecastContractError(f"no residual scale for asset {e}") from None
        return location_scale_frame(mean, sd, self._zq, X.index)
