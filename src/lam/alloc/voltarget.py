"""Volatility targeting: turn a composition into a leverage figure.

Leverage is set so the *unlevered* portfolio's forecast volatility hits a target::

    L = clip(sigma_target / sigma_forecast, 0, L_max)

The forecast is deliberately pessimistic in one specific way. It is the **maximum**
of a normal-times covariance estimate and a crisis-covariance estimate, where the
latter uses correlations measured only on the benchmark's worst days::

    sigma_forecast = max( sqrt(u' S_normal u), sqrt(u' S_crisis u) )

Without that floor, the sizer prices in a diversification benefit computed from
calm-market correlations and levers up against it -- then those correlations
converge toward one precisely during the drawdown the leverage was supposed to
survive. The floor makes the book carry only as much leverage as it could
tolerate if diversification failed, which is the condition under which the
drawdown constraint is actually tested.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .vol import crisis_covariance, ewma_cov_matrix, shrink_to_constant_correlation

TRADING_DAYS = 252


def portfolio_vol_forecast(
    returns: pd.DataFrame,
    weights: pd.DataFrame,
    *,
    stress_reference: pd.Series | None = None,
    halflife: float = 60.0,
    shrinkage: float = 0.3,
    estimation_step: int = 21,
    min_history: int = 252,
    crisis_lookback: int = 2520,
) -> pd.Series:
    """Causal forecast of the unlevered portfolio's annualised volatility.

    The covariance matrix is re-estimated every ``estimation_step`` days and held
    between refreshes -- daily re-estimation is far more expensive and changes the
    forecast very little, since the EWMA halflife dominates.
    """
    index = returns.index
    n = len(index)
    out = np.full(n, np.nan)

    refresh_points = list(range(min_history, n, estimation_step))
    if not refresh_points:
        refresh_points = [min(min_history, n - 1)]

    cov_normal = None
    cov_crisis = None
    next_refresh = 0

    for t in range(n):
        if next_refresh < len(refresh_points) and t >= refresh_points[next_refresh]:
            window = returns.iloc[max(0, t - crisis_lookback) : t].dropna(how="all")
            usable = window.dropna(axis=1, how="all")
            if len(usable) >= 60 and usable.shape[1] > 0:
                cov_normal = shrink_to_constant_correlation(
                    ewma_cov_matrix(usable.fillna(0.0), halflife), shrinkage
                )
                if stress_reference is not None:
                    ref = stress_reference.reindex(usable.index)
                    cov_crisis = crisis_covariance(usable.fillna(0.0), ref)
                cols = list(usable.columns)
            next_refresh += 1

        if cov_normal is None:
            continue

        w = weights.iloc[t].reindex(cols).fillna(0.0).to_numpy(dtype=float)
        var_normal = float(w @ cov_normal @ w)
        var = max(var_normal, 0.0)
        if cov_crisis is not None:
            var = max(var, float(w @ cov_crisis @ w))
        out[t] = np.sqrt(max(var, 1e-12))

    return pd.Series(out, index=index, name="vol_forecast").ffill()


def target_leverage(
    vol_forecast: pd.Series,
    *,
    sigma_target: float = 0.11,
    max_leverage: float = 3.0,
) -> pd.Series:
    """Gross leverage implied by the volatility target."""
    lev = (sigma_target / vol_forecast.replace(0.0, np.nan)).clip(0.0, max_leverage)
    return lev.fillna(0.0).rename("target_leverage")
