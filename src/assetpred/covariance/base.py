"""Covariance-forecaster interface, the per-origin information set, and a registry.

A :class:`CovForecaster` turns a :class:`CovTrainingSet` (everything knowable at
the close of origin month t) into an n x n matrix: the covariance of the
h-month excess log return vector ``y = xr[t+1] + ... + xr[t+h]`` of
``train.assets`` around its conditional mean. The engine (``engine.py``), not
the model, then

* adds the estimation error of the expanding mean that the joint forecast is
  centred on (``linalg.mean_uncertainty_factor``), so the diagonal agrees with
  the marginal baselines' ``sqrt(1 + h/T)`` inflation;
* symmetrizes, repairs tiny negative eigenvalues and checks PSD.

Interface::

    class MyCov(CovForecaster):
        refit_every = 12                # months between fit() calls (default 1)
        def fit(self, train):           # optional: estimate parameters
            ...
            return self
        def predict(self, train):       # required: n x n array, order = train.assets
            ...

``fit`` is called at the first origin, every ``refit_every`` origins, and when
the eligible universe changes; ``predict`` at every origin with that origin's
information set (so state such as an EWMA is always current even when
parameters are refit only yearly).

:class:`CovTrainingSet` refuses data dated after the origin.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from .. import schema


class CovLeakageError(RuntimeError):
    pass


class CovContractError(ValueError):
    pass


@dataclass(frozen=True)
class CovTrainingSet:
    """Information at the close of ``origin``.

    ``xr``: wide monthly excess log returns, rows dated <= origin, all columns
    (assets not in ``assets`` may be used as information, e.g. for long-run
    estimates; they need not be forecast). ``daily``: wide daily simple
    returns dated <= origin (calendar month end), NaN on days a source did not
    trade, or None. ``assets``: the eligible assets to forecast, in order.
    """
    horizon: int
    origin: pd.Timestamp
    xr: pd.DataFrame
    assets: tuple
    daily: pd.DataFrame | None = None
    features: pd.DataFrame | None = None
    _cache: dict = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self):
        if len(self.xr) and self.xr.index.max() > self.origin:
            raise CovLeakageError(f"monthly returns dated after origin {self.origin.date()}")
        if self.daily is not None and len(self.daily) and self.daily.index.max() > self.origin:
            raise CovLeakageError(f"daily returns dated after origin {self.origin.date()}")
        if self.features is not None and len(self.features):
            d = self.features.index.get_level_values("date").max()
            if d > self.origin:
                raise CovLeakageError(f"features dated after origin {self.origin.date()}")
        missing = [a for a in self.assets if a not in self.xr.columns]
        if missing:
            raise CovContractError(f"assets without returns: {missing}")

    @property
    def X(self) -> pd.DataFrame:
        """Monthly excess log returns of the eligible assets."""
        if "X" not in self._cache:
            self._cache["X"] = self.xr[list(self.assets)]
        return self._cache["X"]

    def targets(self) -> pd.DataFrame:
        """Realized h-month targets known at the origin: rows s <= t - h."""
        if "Y" not in self._cache:
            self._cache["Y"] = schema.make_targets(self.X, self.horizon)
        return self._cache["Y"]

    def counts(self) -> np.ndarray:
        """Pairwise joint month counts of the eligible assets."""
        m = self.X.notna().to_numpy(float)
        return m.T @ m


class CovForecaster:
    name: str = "cov"
    refit_every: int = 1
    uses_daily: bool = False
    uses_features: bool = False

    def fit(self, train: CovTrainingSet) -> "CovForecaster":
        return self

    def predict(self, train: CovTrainingSet) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def params(self) -> dict:
        out = {}
        for k, v in vars(self).items():
            if k.startswith("_") or k == "name":
                continue
            if isinstance(v, (int, float, str, bool, type(None), tuple)):
                out[k] = v
        return out


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------
_NAME = re.compile(r"^[A-Za-z0-9_.\-]+$")
_REGISTRY: dict[str, Callable[[], CovForecaster]] = {}


def register(name: str, factory: Callable[[], CovForecaster] | None = None, *,
             overwrite: bool = False):
    if not _NAME.match(name):
        raise ValueError(f"bad covariance model name {name!r}")

    def deco(f):
        if name in _REGISTRY and not overwrite:
            raise ValueError(f"covariance model {name!r} already registered")
        _REGISTRY[name] = f
        return f
    return deco(factory) if factory is not None else deco


def _discover() -> None:
    from . import models  # noqa: F401  (registers the built-in models)


def get_factory(name: str) -> Callable[[], CovForecaster]:
    _discover()
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown covariance model {name!r}; registered: {sorted(_REGISTRY)}") from None


def available() -> list[str]:
    _discover()
    return sorted(_REGISTRY)
