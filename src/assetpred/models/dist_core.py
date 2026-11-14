"""Framework for predictive-distribution models with a pluggable location.

A distribution model here is a :class:`Composite` of

* a **location** component (default :class:`HistMeanLocation`, the expanding
  historical mean ``h * mean(xr)`` used by every baseline), and
* one or more **scale/shape** components (:class:`DistComponent`), each of
  which learns the distribution of the residual ``e = y - h * mean(xr)`` and
  returns, per row, a predictive ``sd`` and quantile *offsets* from the
  location. Several components are combined by quantile averaging.

The forecast is ``mean = location(X)``, ``q_k = location(X) + offset_k``. With
the default location the mean equals ``hist_mean``'s, so the out-of-sample
R^2 is zero by construction and every bit of skill is distributional.

Separation from the mean model. Scale/shape components always fit on
residuals about the historical mean, never about the location they will be
paired with, so a location can be swapped without refitting them:
:class:`MeanPlusDist` (registered via :func:`register_combo`) puts any
registered model's ``mean`` at the centre of a registered distribution
model's offsets. If the paired mean forecasts better than the historical
mean, the offsets are slightly too wide (they still carry the variance the
mean model explains); if worse, too narrow. Both are second order for the
monthly-to-annual R^2 levels in play.

Refit schedule. The composite itself refits every month (``refit_every =
1``) so that cheap pieces (the historical mean, EWMA variance) are always
current, but each component is refit only on its own cadence
(``component.refit_every``): at origin t it is the component fitted at
``t0 = refit_origin(t)``, the latest month <= t on the grid of every
``refit_every`` months counted from ``config.OOS_START``, or later if the
eligible asset set changed since. Fitted components are cached per process
keyed by (component, horizon, t0, data hash), so this costs one fit per
scheduled refit; the cache is an optimisation only. A component fitted at t0
sees only the purged training data available at t0 (a truncated
:class:`TrainingSet`), so the schedule cannot leak.

Return-derived state. Each month the composite computes from ``train.xr``
causal per-asset series that the feature panel lacks (``DERIVED``: log EWMA
of squared monthly excess returns) and appends them to the feature rows
handed to components, both for training rows and at the origin. A component
refit annually therefore still sees this month's EWMA variance at predict.
"""
from __future__ import annotations

import hashlib
import inspect
from dataclasses import replace

import numpy as np
import pandas as pd

from .. import config
from ..evaluation.cv import month_ordinal
from .base import (Forecaster, InsufficientDataError, TrainingSet, TAUS,
                   distribution_frame)
from .registry import get_factory, register

K = len(TAUS)
_edges = np.concatenate([[0.0], TAUS, [1.0]])
GRID_W = (_edges[2:] - _edges[:-2]) / 2.0
GRID_W = GRID_W / GRID_W.sum()          # quantile-grid weights for an approximate mean

DERIVED_PREFIX = "_d_"
DERIVED = {  # name -> EWMA decay of squared monthly excess log returns
    "_d_lewma94": 0.94,
    "_d_lewma985": 0.985,
}

ASSET_GROUP = {
    "us_eq": "equity", "dev_eq": "equity", "em_eq": "equity", "reits": "equity",
    "ig_credit": "credit", "hy_credit": "credit",
    "ust_int": "rates", "ust_long": "rates", "tips": "rates",
    "gold": "real", "commodities": "real",
}


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def derived_long(xr: pd.DataFrame) -> pd.DataFrame:
    """Causal return-derived series, long (date, asset). Row (s, a) uses xr[:s, a] only."""
    cols = {}
    x2 = xr ** 2
    for name, lam in DERIVED.items():
        e = x2.ewm(alpha=1 - lam, adjust=False, ignore_na=True).mean()
        e = e.where(xr.notna().cumsum() >= 6)      # need a few months before it means anything
        cols[name] = np.log(e.clip(lower=1e-10))
    out = pd.DataFrame({k: v.stack() for k, v in cols.items()})
    out.index = out.index.set_names(["date", "asset"])
    return out


def grid_mean(Q: np.ndarray) -> np.ndarray:
    """Approximate mean of distributions given by quantiles on the grid (rows)."""
    return np.asarray(Q, float) @ GRID_W


def weighted_quantile(x, w, taus) -> np.ndarray:
    """Quantiles of x with sampling weights w (Hazen-type interpolation)."""
    o = np.argsort(x)
    x, w = np.asarray(x, float)[o], np.asarray(w, float)[o]
    c = (np.cumsum(w) - 0.5 * w) / w.sum()
    return np.interp(taus, c, x)


def centered_quantiles(z, w=None) -> np.ndarray:
    """Empirical quantiles of z minus its sample mean (scale kept, unlike
    ``base.empirical_z_quantiles`` which also rescales to unit sd); optional
    observation weights."""
    z = np.asarray(z, float)
    ok = np.isfinite(z)
    if w is None:
        z = z[ok]
        return np.quantile(z - z.mean(), TAUS)
    z, w = z[ok], np.asarray(w, float)[ok]
    return weighted_quantile(z - np.average(z, weights=w), w, TAUS)


def recency_weights(index: pd.Index, half_life: float | None) -> np.ndarray | None:
    """Exponential weights by row date, half-life in months (None: equal)."""
    if not half_life:
        return None
    m = month_ordinal(index.get_level_values("date"))
    return 0.5 ** ((m.max() - m) / float(half_life))


def shape_quantiles(z: pd.Series, horizon: int, mode: str = "pooled", n0: float = 150.0,
                    half_life: float | None = None) -> dict:
    """Standardized shape on the grid, per asset group.

    ``z``: standardized residuals indexed (date, asset). Returns a dict
    group -> quantile vector, with key ``None`` for the pooled shape.
    mode ``"pooled"``: one shape for all assets. ``"group"``: each asset
    group's shape shrunk to the pooled one with weight n_eff / (n_eff + n0),
    ``n_eff = n_group / horizon`` (overlapping targets). ``half_life``
    (months, pooled only): weight rows by recency.
    """
    z = z[np.isfinite(z.to_numpy())]
    pooled = centered_quantiles(z.to_numpy(), recency_weights(z.index, half_life))
    out = {None: pooled}
    if mode == "pooled":
        return out
    if half_life:
        raise ValueError("half_life is only implemented for the pooled shape")
    if mode != "group":
        raise ValueError("mode must be 'pooled' or 'group'")
    grp = z.index.get_level_values("asset").map(lambda a: ASSET_GROUP.get(a, "other"))
    zc = z - z.mean()
    for g in pd.unique(grp):
        v = zc[np.asarray(grp == g)].to_numpy()
        if len(v) < 20:
            continue
        w = (len(v) / horizon) / (len(v) / horizon + n0)
        out[g] = w * np.quantile(v, TAUS) + (1 - w) * pooled
    return out


def lookup_shape(shapes: dict, assets) -> np.ndarray:
    return np.vstack([shapes.get(ASSET_GROUP.get(a, "other"), shapes[None]) for a in assets])


def describe(obj) -> str:
    """Stable description of a component for cache keys and fingerprints."""
    if hasattr(obj, "describe"):
        return obj.describe()
    items = {k: v for k, v in sorted(vars(obj).items())
             if not k.startswith("_") and not k.endswith("_")
             and isinstance(v, (int, float, str, bool, type(None), tuple))}
    return f"{type(obj).__qualname__}({items})"


def source_hash(objs) -> str:
    h = hashlib.sha256()
    seen = set()
    for o in objs:
        try:
            f = inspect.getfile(o if inspect.isclass(o) else type(o))
        except (TypeError, OSError):
            continue
        if f not in seen:
            seen.add(f)
            h.update(open(f, "rb").read())
    return h.hexdigest()[:16]


# --------------------------------------------------------------------------
# training-set utilities
# --------------------------------------------------------------------------
def eligible_change_ordinal(xr: pd.DataFrame, min_history: int = config.MIN_ASSET_HISTORY) -> int:
    """Month ordinal of the last change of the eligible asset set (<= last xr date)."""
    elig = (xr.notna().cumsum() >= min_history).to_numpy()
    if len(elig) < 2:
        return int(month_ordinal(xr.index[:1])[0]) if len(elig) else 0
    changed = np.flatnonzero((elig[1:] != elig[:-1]).any(axis=1))
    i = changed[-1] + 1 if len(changed) else 0
    return int(month_ordinal(xr.index[i:i + 1])[0])


def refit_origin(train: TrainingSet, every: int) -> pd.Timestamp:
    """Origin at which a component with cadence ``every`` was (last) fitted."""
    t = train.origin
    if every <= 1:
        return t
    t_ord = int(month_ordinal([t])[0])
    anchor = int(month_ordinal([pd.Timestamp(config.OOS_START)])[0])
    sched = t_ord - ((t_ord - anchor) % every)
    t0 = max(sched, eligible_change_ordinal(train.xr))
    return t - pd.offsets.MonthEnd(t_ord - t0) if t0 < t_ord else t


def truncate(train: TrainingSet, t0: pd.Timestamp, features: pd.DataFrame | None = None) -> TrainingSet:
    """The TrainingSet the engine would have built at origin ``t0 <= train.origin``
    (optionally with substitute feature rows, e.g. augmented with derived columns)."""
    feats = train._features if features is None else features
    if t0 == train.origin and features is None:
        return train
    h = train.horizon
    t0_ord = int(month_ordinal([t0])[0])
    d_ord = month_ordinal(feats.index.get_level_values("date"))
    k = int(np.searchsorted(d_ord, t0_ord - h, side="right"))
    return TrainingSet(horizon=h, origin=t0, xr=train.xr.loc[:t0], assets=train.assets,
                       seed=int(config.SEED + 100_000 * h + t0_ord) % (2**32),
                       _features=feats.iloc[:k], _targets=train._targets.iloc[:k])


def hist_location(train: TrainingSet) -> dict:
    """Asset -> h * expanding mean of monthly excess log returns (the baselines' mean)."""
    return (train.horizon * train.xr.mean()).dropna().to_dict()


def residuals(train: TrainingSet) -> pd.Series:
    """y - historical-mean location, for the purged training rows."""
    loc = hist_location(train)
    y = train.y
    m = np.array([loc.get(a, np.nan) for a in y.index.get_level_values("asset")])
    return (y - m).dropna()


# --------------------------------------------------------------------------
# components
# --------------------------------------------------------------------------
class Location:
    """Location component interface: ``fit(train)``, ``predict_mean(X) -> array``."""

    refit_every: int = 1

    def fit(self, train: TrainingSet) -> "Location":
        raise NotImplementedError

    def predict_mean(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError


class HistMeanLocation(Location):
    """Expanding historical mean, refit monthly (identical to ``hist_mean``'s mean)."""

    refit_every = 1

    def fit(self, train):
        self._loc = hist_location(train)
        return self

    def predict_mean(self, X):
        try:
            return np.array([self._loc[a] for a in X.index.get_level_values("asset")], float)
        except KeyError as e:
            raise InsufficientDataError(f"no history for asset {e}") from None


class RegisteredLocation(Location):
    """Location = the ``mean`` column of a registered model's forecast.

    The model is fitted on its own cadence (its ``refit_every`` or
    ``config.REFIT_EVERY``) on the original feature panel (derived columns
    are stripped before it sees anything)."""

    def __init__(self, model: str):
        self.model = model
        probe = get_factory(model)()
        self.refit_every = int(probe.refit_every or config.REFIT_EVERY)

    def describe(self):
        m = get_factory(self.model)()
        return f"RegisteredLocation({self.model}: {m!r}, src={source_hash([m])})"

    def fit(self, train):
        from .registry import create
        self._m = create(self.model).fit(train)
        return self

    def predict_mean(self, X):
        cols = [c for c in X.columns if not str(c).startswith(DERIVED_PREFIX)]
        return self._m.predict(X[cols])["mean"].to_numpy(float)


class DistComponent:
    """Scale/shape component interface.

    ``fit(train)`` learns the residual distribution about the historical mean
    (use :func:`residuals`); ``train.X`` includes the ``DERIVED`` columns.
    ``predict_offsets(X)`` returns ``(sd, Q)`` with ``Q`` the n x K quantile
    offsets from the location (non-decreasing per row) and ``sd`` > 0.
    """

    refit_every: int = 12

    def fit(self, train: TrainingSet) -> "DistComponent":
        raise NotImplementedError

    def predict_offsets(self, X: pd.DataFrame):
        raise NotImplementedError


# --------------------------------------------------------------------------
# per-process cache of fitted components
# --------------------------------------------------------------------------
_CACHE: dict = {}


def _data_key(train: TrainingSet, t0: pd.Timestamp) -> int:
    xr = train.xr.loc[:t0]
    h1 = int(pd.util.hash_pandas_object(xr, index=True).sum())
    t0_ord = int(month_ordinal([t0])[0])
    feats = train._features
    d_ord = month_ordinal(feats.index.get_level_values("date"))
    k = int(np.searchsorted(d_ord, t0_ord - train.horizon, side="right"))
    h2 = int(pd.util.hash_pandas_object(feats.iloc[:k], index=True).sum()) if k else 0
    h3 = int(pd.util.hash_pandas_object(train._targets.iloc[:k].fillna(-9e9), index=True).sum()) if k else 0
    return hash((h1, h2, h3, k, tuple(train.assets)))


def fit_scheduled(comp, train: TrainingSet, augmented: pd.DataFrame | None, slot: str):
    """Fit ``comp`` (a fresh object) as of its refit origin, or reuse the cached fit."""
    every = int(getattr(comp, "refit_every", 1) or 1)
    t0 = refit_origin(train, every)
    key = (slot, describe(comp), train.horizon)
    full = (t0, _data_key(train, t0))
    hit = _CACHE.get(key)
    if hit is not None and hit[0] == full:
        return hit[1]
    sub = truncate(train, t0, augmented)
    comp.fit(sub)
    if every > 1:
        _CACHE[key] = (full, comp)
    return comp


def clear_cache():
    _CACHE.clear()


# --------------------------------------------------------------------------
# the forecaster
# --------------------------------------------------------------------------
class Composite(Forecaster):
    """Location + quantile-averaged scale/shape components (see module docstring).

    ``location``: a :class:`Location` (default :class:`HistMeanLocation`).
    ``components``: :class:`DistComponent` instances (fresh, unfitted).
    ``weights``: quantile-averaging weights (default equal).
    """

    refit_every = 1

    def __init__(self, components, location: Location | None = None, weights=None):
        self.components = list(components)
        self.location = location if location is not None else HistMeanLocation()
        w = np.ones(len(self.components)) if weights is None else np.asarray(weights, float)
        self.weights = tuple(float(x) for x in w / w.sum())

    def params(self) -> dict:
        objs = [self.location, *self.components]
        return {"location": describe(self.location),
                "components": [describe(c) for c in self.components],
                "weights": list(self.weights),
                "src": source_hash(objs)}

    def fit(self, train: TrainingSet):
        if len(train.y) == 0:
            raise InsufficientDataError("no training rows")
        self._derived = derived_long(train.xr)
        aug = train._features.join(self._derived, how="left")
        self._fitted = [fit_scheduled(c, train, aug, f"c{i}") for i, c in enumerate(self.components)]
        self._loc = fit_scheduled(self.location, train, None, "loc")
        return self

    def _augment(self, X: pd.DataFrame) -> pd.DataFrame:
        add = self._derived.reindex(X.index)
        return pd.concat([X, add], axis=1)

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        Xa = self._augment(X)
        mean = np.asarray(self._loc.predict_mean(X), float).reshape(-1)
        sd = np.zeros(len(X))
        Q = np.zeros((len(X), K))
        for w, c in zip(self.weights, self._fitted):
            s, q = c.predict_offsets(Xa)
            sd += w * np.asarray(s, float)
            Q += w * np.asarray(q, float)
        Q = np.sort(Q, axis=1)
        return distribution_frame(mean, sd, mean[:, None] + Q, X.index)


class MeanPlusDist(Composite):
    """The combinator: location from registered model ``mean_model``, offsets
    from registered distribution model ``dist_model`` (a :class:`Composite`)."""

    def __init__(self, mean_model: str, dist_model: str):
        d = get_factory(dist_model)()
        if not isinstance(d, Composite):
            raise TypeError(f"{dist_model!r} is not a dist_* Composite model")
        self.mean_model = mean_model
        self.dist_model = dist_model
        super().__init__(d.components, RegisteredLocation(mean_model), d.weights)

    def params(self) -> dict:
        p = super().params()
        p.update(mean_model=self.mean_model, dist_model=self.dist_model)
        return p


def register_combo(mean_model: str, dist_model: str, name: str | None = None) -> str:
    """Register ``MeanPlusDist(mean_model, dist_model)``; returns the registry name
    (default ``"<mean_model>__<dist_model>"``)."""
    from functools import partial
    name = name or f"{mean_model}__{dist_model}"
    register(name, partial(MeanPlusDist, mean_model, dist_model), overwrite=True)
    return name
