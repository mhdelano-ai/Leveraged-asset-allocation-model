"""Volatility and covariance estimation. Every estimator here is causal.

The design choice that matters most is :func:`blended_vol`. A pure fast EWMA
produces its *lowest* readings during volatility-suppressed melt-ups -- 2006,
2017, late 2019 -- so a vol-targeting sizer fed that estimate takes its
*largest* leverage immediately before the regime breaks. Flooring the fast
estimate against a fraction of a three-year trailing vol removes most of that
behaviour at almost no cost in normal markets.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def ewma_vol(returns: pd.DataFrame | pd.Series, halflife: float = 40.0) -> pd.DataFrame | pd.Series:
    """Annualised EWMA volatility, right-aligned (uses no future data)."""
    return returns.ewm(halflife=halflife, min_periods=20).std() * np.sqrt(TRADING_DAYS)


def blended_vol(
    returns: pd.DataFrame,
    *,
    halflife: float = 40.0,
    slow_window: int = 756,
    slow_weight: float = 0.75,
) -> pd.DataFrame:
    """Fast EWMA vol floored against a fraction of trailing 3-year vol."""
    fast = ewma_vol(returns, halflife)
    slow = returns.rolling(slow_window, min_periods=126).std() * np.sqrt(TRADING_DAYS)
    return np.maximum(fast, slow_weight * slow.fillna(fast))


def ewma_cov_matrix(returns: pd.DataFrame, halflife: float = 60.0) -> np.ndarray:
    """Single EWMA covariance matrix from the whole sample passed in.

    Callers must slice ``returns`` to the data available at the decision date.
    """
    x = returns.dropna().to_numpy(dtype=float)
    n = len(x)
    if n < 20:
        return np.cov(x.T, ddof=1) * TRADING_DAYS if n > 2 else np.eye(returns.shape[1]) * 0.01
    lam = 0.5 ** (1.0 / halflife)
    weights = lam ** np.arange(n - 1, -1, -1)
    weights /= weights.sum()
    centred = x - np.average(x, axis=0, weights=weights)
    return (centred * weights[:, None]).T @ centred * TRADING_DAYS


def shrink_to_constant_correlation(cov: np.ndarray, delta: float = 0.3) -> np.ndarray:
    """Shrink a covariance matrix toward an equal-correlation target.

    Sample correlations over a short window are noisy, and the vol-target sizer
    divides by them -- noise there translates directly into leverage noise.
    """
    sd = np.sqrt(np.clip(np.diag(cov), 1e-12, None))
    corr = cov / np.outer(sd, sd)
    off = corr[~np.eye(len(corr), dtype=bool)]
    mean_corr = float(np.mean(off)) if off.size else 0.0
    target = np.full_like(corr, mean_corr)
    np.fill_diagonal(target, 1.0)
    blended = (1.0 - delta) * corr + delta * target
    return blended * np.outer(sd, sd)


def crisis_covariance(
    returns: pd.DataFrame,
    stress_reference: pd.Series,
    *,
    tail_quantile: float = 0.10,
    min_obs: int = 60,
) -> np.ndarray:
    """Covariance estimated only on the worst days of a reference asset.

    This is the single most valuable risk input in the stack. Diversification is
    priced from normal-times correlations, but those correlations converge
    toward one exactly when the diversification is needed. Using this as a
    *floor* on the volatility forecast stops the sizer from buying a
    diversification benefit that evaporates in the drawdown it is meant to
    protect against.
    """
    joined = pd.concat([returns, stress_reference.rename("__ref")], axis=1).dropna()
    if len(joined) < min_obs:
        return ewma_cov_matrix(returns)
    cutoff = joined["__ref"].quantile(tail_quantile)
    tail = joined[joined["__ref"] <= cutoff].drop(columns="__ref")
    if len(tail) < 20:
        return ewma_cov_matrix(returns)
    return np.cov(tail.to_numpy(dtype=float).T, ddof=1) * TRADING_DAYS


def rolling_correlation(a: pd.Series, b: pd.Series, window: int = 252) -> pd.Series:
    """Trailing correlation -- used to show the stock/bond regime flip."""
    joined = pd.concat([a.rename("a"), b.rename("b")], axis=1, join="inner").dropna()
    return joined["a"].rolling(window, min_periods=window // 2).corr(joined["b"])
