"""The data contract of the return side (ported from ``assetpred.schema``).

The file loaders moved to ``assetforecast.returns.inputs`` (returns and cash
now come from the package panel); what stays here is the contract itself.

Conventions
-----------
* Dates are calendar month-end timestamps (``pd.offsets.MonthEnd``), no timezone.
* A row dated t holds only what an investor could know at the close of month t.
  Anything published later (macro releases, revised data) must be lagged before
  it is written. Nothing downstream re-lags.
* Returns: wide, index=date, columns=``config.ASSETS`` ids, simple total
  return over the month. NaN before an asset's history begins.
* Cash: simple return on a 1-month T-bill held over the month.
* Excess return is in logs: ``xr = log1p(r) - log1p(cash)``.
* Target for horizon h at origin t: ``y[t] = xr[t+1] + ... + xr[t+h]`` (NaN if
  any term is missing). The forecast is made at the close of month t.
* ``features``: long, MultiIndex (date, asset), float columns, one row per month
  in which the asset has a return. Global features repeat across assets.
  Missing values stay NaN; models must cope. Any standardization uses past data
  only. ``feature_meta``: one row per column; ``hindsight`` marks features built
  from series estimated over the full sample (dropped in the robustness check).
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


def excess_log_returns(returns: pd.DataFrame, cash: pd.Series) -> pd.DataFrame:
    """``log1p(returns) - log1p(cash)``, cash aligned to the returns' dates.

    For panels held in memory (synthetic data, the archived assetpred inputs);
    the package panel's excess returns come from ``inputs.excess_returns``.
    """
    return np.log1p(returns).sub(np.log1p(cash.reindex(returns.index)), axis=0)


def make_targets(xr: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Forward h-month excess log return, dated at the forecast origin."""
    fwd = xr.rolling(horizon, min_periods=horizon).sum().shift(-horizon)
    return fwd


def make_targets_long(xr: pd.DataFrame, horizon: int) -> pd.Series:
    """`make_targets` as a Series indexed (date, asset), realized targets only."""
    y = make_targets(xr, horizon).stack().dropna()
    y.index = y.index.set_names(["date", "asset"])
    return y.rename("y")
