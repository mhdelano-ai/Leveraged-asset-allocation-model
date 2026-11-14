"""The data contract between pipeline stages, and loaders that enforce it.

Conventions
-----------
* Dates are calendar month-end timestamps (``pd.offsets.MonthEnd``), no timezone.
* A row dated t holds only what an investor could know at the close of month t.
  Anything published later (macro releases, revised data) must be lagged before
  it is written. Nothing downstream re-lags.
* ``asset_returns``: wide, index=date, columns=config.ASSETS ids, simple total
  return over the month. NaN before an asset's history begins, none after.
* ``cash``: Series named ``cash``; simple return on a 1-month T-bill held over
  the month.
* Excess return is in logs: ``xr = log1p(r) - log1p(cash)``.
* Target for horizon h at origin t: ``y[t] = xr[t+1] + ... + xr[t+h]`` (NaN if
  any term is missing). The forecast is made at the close of month t.
* ``predictors``: wide, index=date, one raw series per column, already lagged to
  availability. Series may end before the panel does (NaN tails as well as
  heads). ``predictor_meta``: one row per column; ``hindsight`` marks series
  estimated over the full sample, which embed future information even when
  lagged and are dropped in the robustness check.
* ``features``: long, MultiIndex (date, asset), float columns. Global features
  repeat across assets. ``feature_meta``: one row per column.
* Forecasts: long, columns FORECAST_COLUMNS. ``date`` is the origin t. ``mean``
  and ``sd`` describe the predictive distribution of ``y``; ``q01``..``q99`` are
  its quantiles and must be non-decreasing.
* Covariances: long, columns COV_COLUMNS, predictive covariance of the h-month
  excess log returns of asset_i and asset_j.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

ASSET_RETURNS = config.PROCESSED / "asset_returns.parquet"
CASH = config.PROCESSED / "cash.parquet"
ASSET_META = config.PROCESSED / "asset_meta.csv"        # asset, segment_start, segment_end, source, description
PREDICTORS = config.PROCESSED / "predictors.parquet"
PREDICTOR_META = config.PROCESSED / "predictor_meta.csv"  # name, family, scope, asset, source, lag_months, realtime, hindsight, first_date, last_date, description
FEATURES = config.PROCESSED / "features.parquet"
FEATURE_META = config.PROCESSED / "feature_meta.csv"    # name, family, scope, hindsight, first_date, description

QUANTILE_COLUMNS = [f"q{int(round(q * 100)):02d}" for q in config.QUANTILES]
FORECAST_COLUMNS = ["model", "horizon", "date", "asset", "mean", "sd", *QUANTILE_COLUMNS]
COV_COLUMNS = ["model", "horizon", "date", "asset_i", "asset_j", "cov"]


def _check_month_end(index: pd.DatetimeIndex, what: str) -> None:
    if not isinstance(index, pd.DatetimeIndex) or index.tz is not None:
        raise ValueError(f"{what}: index must be a naive DatetimeIndex")
    if not (index == index + pd.offsets.MonthEnd(0)).all():
        raise ValueError(f"{what}: dates must be calendar month ends")
    if not index.is_monotonic_increasing or index.has_duplicates:
        raise ValueError(f"{what}: dates must be sorted and unique")


def load_asset_returns() -> pd.DataFrame:
    df = pd.read_parquet(ASSET_RETURNS)
    _check_month_end(df.index, "asset_returns")
    unknown = set(df.columns) - set(config.ASSETS)
    if unknown:
        raise ValueError(f"asset_returns: unknown assets {sorted(unknown)}")
    return df


def load_cash() -> pd.Series:
    s = pd.read_parquet(CASH)["cash"]
    _check_month_end(s.index, "cash")
    return s


def excess_log_returns(returns: pd.DataFrame | None = None, cash: pd.Series | None = None) -> pd.DataFrame:
    returns = load_asset_returns() if returns is None else returns
    cash = load_cash() if cash is None else cash
    return np.log1p(returns).sub(np.log1p(cash.reindex(returns.index)), axis=0)


def make_targets(xr: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Forward h-month excess log return, dated at the forecast origin."""
    fwd = xr.rolling(horizon, min_periods=horizon).sum().shift(-horizon)
    return fwd


def load_predictors() -> pd.DataFrame:
    df = pd.read_parquet(PREDICTORS)
    _check_month_end(df.index, "predictors")
    return df


def load_features() -> pd.DataFrame:
    df = pd.read_parquet(FEATURES)
    if list(df.index.names) != ["date", "asset"]:
        raise ValueError("features: index must be (date, asset)")
    return df
