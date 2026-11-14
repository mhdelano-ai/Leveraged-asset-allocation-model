"""Forecaster interfaces and the model registry.

Every model answers one question, asked at a month-end ``t`` with a Panel
view truncated at ``t``: what will the annualized covariance of overlapping
21-TRADING-DAY LOG EXCESS returns be over the next 12 calendar months? (See
ARCHITECTURE.md for why the horizon is 21 days and not one: 1-day volatility
depends on how a source is priced. Daily returns are fine as inputs but a
1-day vol is not a forecast of this target.)

Three interfaces, so that volatility and correlation ideas can be developed
and scored separately and then paired:

    VolForecaster   -> per-asset annualized vol (pd.Series)
    CorrForecaster  -> correlation matrix (pd.DataFrame)
    CovForecaster   -> covariance matrix (pd.DataFrame)

``Composite(vol, corr)`` turns any vol model and any corr model into a
CovForecaster. The 12-month-return covariance (which differs from the above
when returns are serially correlated) is produced downstream by the bridge in
``assetforecast.bridge``; models do not produce it.

Rules every model must obey:
  * Use only ``view`` (already truncated at the origin). Never load data yourself.
  * If you train on realized targets, an origin ``s`` is usable only if
    ``s + 12 months <= view.asof_date`` (its target is fully observed).
  * Return exactly the requested ``assets``, in that order, all finite;
    vols > 0; correlation/covariance symmetric positive definite.
  * Be deterministic (seed anything random).
  * Models may cache state between calls but must give the same answer if
    called cold for a single origin.
"""
from __future__ import annotations

import importlib
import pkgutil
import warnings
from abc import ABC, abstractmethod
from typing import Callable

import numpy as np
import pandas as pd

from ..data.panel import Panel


class VolForecaster(ABC):
    name: str = "unnamed_vol"

    @abstractmethod
    def forecast_vol(self, view: Panel, assets: list[str]) -> pd.Series:
        """Annualized vol of 21-day log excess returns over the next 12 months."""


class CorrForecaster(ABC):
    name: str = "unnamed_corr"

    @abstractmethod
    def forecast_corr(self, view: Panel, assets: list[str]) -> pd.DataFrame:
        """Correlation matrix over the next 12 months (unit diagonal, PD)."""


class CovForecaster(ABC):
    name: str = "unnamed_cov"

    @abstractmethod
    def forecast_cov(self, view: Panel, assets: list[str]) -> pd.DataFrame:
        """Annualized covariance of 21-day log excess returns over the next 12 months."""

    # so a native covariance model can also be scored as a vol or corr model
    def forecast_vol(self, view: Panel, assets: list[str]) -> pd.Series:
        c = self.forecast_cov(view, assets)
        return pd.Series(np.sqrt(np.diag(c.values)), index=assets)

    def forecast_corr(self, view: Panel, assets: list[str]) -> pd.DataFrame:
        c = self.forecast_cov(view, assets)
        d = np.sqrt(np.diag(c.values))
        return pd.DataFrame(c.values / np.outer(d, d), index=assets, columns=assets)


class Composite(CovForecaster):
    """Covariance = diag(vol) @ corr @ diag(vol)."""

    def __init__(self, vol: VolForecaster, corr: CorrForecaster, name: str | None = None):
        self.vol, self.corr = vol, corr
        self.name = name or f"{vol.name}+{corr.name}"

    def forecast_cov(self, view: Panel, assets: list[str]) -> pd.DataFrame:
        v = self.vol.forecast_vol(view, assets).reindex(assets).values
        c = self.corr.forecast_corr(view, assets).reindex(index=assets, columns=assets).values
        return pd.DataFrame(c * np.outer(v, v), index=assets, columns=assets)


# ---------------------------------------------------------------- registry
_REGISTRY: dict[str, dict[str, Callable[[], object]]] = {"vol": {}, "corr": {}, "cov": {}}
_loaded = False
LOAD_ERRORS: dict[str, str] = {}  # family -> error, for families that failed to import


def register(kind: str, name: str):
    """Decorator: ``@register("vol", "ewma_hl126")`` on a zero-arg factory or class."""
    if kind not in _REGISTRY:
        raise ValueError(f"kind must be one of {list(_REGISTRY)}")

    def deco(factory):
        if name in _REGISTRY[kind]:
            raise ValueError(f"duplicate {kind} model name: {name}")
        _REGISTRY[kind][name] = factory
        return factory

    return deco


def _load_all():
    """Import every subpackage/module of assetforecast.models so registrations run."""
    global _loaded
    if _loaded:
        return
    _loaded = True
    pkg = importlib.import_module("assetforecast.models")
    for m in pkgutil.iter_modules(pkg.__path__):
        if m.name == "base":
            continue
        try:
            importlib.import_module(f"assetforecast.models.{m.name}")
        except Exception as e:  # one broken family must not take the others down
            LOAD_ERRORS[m.name] = repr(e)
            warnings.warn(f"model family '{m.name}' failed to import and is unavailable: {e!r}")


def list_models(kind: str) -> list[str]:
    _load_all()
    return sorted(_REGISTRY[kind])


def get_model(kind: str, name: str):
    _load_all()
    model = _REGISTRY[kind][name]()
    model.name = name
    return model
