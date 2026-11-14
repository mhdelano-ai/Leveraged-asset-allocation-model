"""Inputs, the feature record type, and point-in-time transforms shared by the
feature modules.

Every transform here uses only observations dated at or before t to produce the
value at t (rolling or expanding windows, no centring, no full-sample fits).
The truncation test in ``tests/test_features.py`` checks this end to end.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .. import config, schema

DAILY_RETURNS = config.PROCESSED / "asset_returns_daily.parquet"

# Minimum number of observations before an expanding standardisation is reported.
Z_MIN_OBS = 60


@dataclass
class Inputs:
    returns: pd.DataFrame          # monthly simple total returns, wide
    cash: pd.Series                # monthly simple cash return
    daily: pd.DataFrame            # daily simple total returns, wide
    predictors: pd.DataFrame       # raw predictors, already lagged to availability
    predictor_meta: pd.DataFrame   # one row per predictor

    @property
    def hindsight(self) -> set[str]:
        m = self.predictor_meta
        return set(m.loc[m["hindsight"].astype(str) == "True", "name"])

    def truncate(self, last: str | pd.Timestamp) -> "Inputs":
        """Inputs as they would have stood at the close of month ``last``."""
        t = pd.Timestamp(last) + pd.offsets.MonthEnd(0)
        return Inputs(
            returns=self.returns.loc[:t],
            cash=self.cash.loc[:t],
            daily=self.daily.loc[:t],
            predictors=self.predictors.loc[:t],
            predictor_meta=self.predictor_meta,
        )


def load_inputs() -> Inputs:
    daily = pd.read_parquet(DAILY_RETURNS)
    if not daily.index.is_monotonic_increasing or daily.index.has_duplicates:
        raise ValueError("daily returns: dates must be sorted and unique")
    return Inputs(
        returns=schema.load_asset_returns(),
        cash=schema.load_cash(),
        daily=daily,
        predictors=schema.load_predictors(),
        predictor_meta=pd.read_csv(schema.PREDICTOR_META, keep_default_na=False),
    )


@dataclass
class Feat:
    """One feature column.

    ``values`` is a Series indexed by month-end date for a global feature, or a
    wide DataFrame (date x asset) for an asset-scope feature. ``sources`` names
    the raw predictors it is built from; a feature is flagged hindsight when any
    source is.
    """
    name: str
    family: str
    values: pd.Series | pd.DataFrame
    description: str
    sources: tuple[str, ...] = field(default_factory=tuple)

    @property
    def scope(self) -> str:
        return "asset" if isinstance(self.values, pd.DataFrame) else "global"


# ------------------------------------------------------------------ transforms
def ez(x: pd.Series | pd.DataFrame, min_obs: int = Z_MIN_OBS):
    """Expanding z-score of x_t against all observations up to and including t."""
    e = x.expanding(min_periods=min_obs)
    return (x - e.mean()) / e.std()


def rz(x: pd.Series | pd.DataFrame, window: int, min_obs: int):
    """Rolling z-score over the trailing ``window`` months (including t)."""
    r = x.rolling(window, min_periods=min_obs)
    return (x - r.mean()) / r.std()


def chg(x, k: int):
    """k-month difference."""
    return x - x.shift(k)


def logchg(x, k: int):
    """k-month log change."""
    return np.log(x) - np.log(x.shift(k))


def to_month_end(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(index).normalize() + pd.offsets.MonthEnd(0)
