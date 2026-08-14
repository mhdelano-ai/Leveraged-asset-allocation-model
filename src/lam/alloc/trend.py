"""Time-series momentum overlay.

This layer, not volatility targeting, is what actually defends the drawdown
constraint in slow bear markets. Volatility targeting is **lagging by
construction**: realised volatility only rises *after* prices have fallen, so in
2000-02 and 2022 -- long grinding declines with unremarkable daily vol -- it stays
fully invested most of the way down. Trend is a price-*level* signal and turns
exposure off while the vol estimator still reports calm.

The converse also holds, which is why both layers exist: trend whipsaws in choppy
sideways markets and is too slow for a one-day event like 1987, where the vol
response is the effective brake. They fail in different places.

Momentum is measured **in excess of cash**. In the 1970s and early 1980s cash
yielded 8-15%, so raw price momentum rates a 5%-returning asset as a strong
uptrend when it was in fact a loss against the risk-free alternative -- exactly
backwards in the regime that matters most here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_LOOKBACKS = (63, 126, 252)


def excess_momentum_score(
    prices: pd.DataFrame,
    rf_annual: pd.Series,
    *,
    lookbacks: tuple[int, ...] = DEFAULT_LOOKBACKS,
) -> pd.DataFrame:
    """Mean of ``sign(excess return)`` across lookbacks. Range [-1, 1]."""
    rf = rf_annual.reindex(prices.index).ffill().fillna(0.0)
    # Compounded cash return over each lookback, approximated on trading days.
    scores = []
    for k in lookbacks:
        asset_ret = prices / prices.shift(k) - 1.0
        cash_ret = (1.0 + rf / 252.0) ** k - 1.0
        scores.append(np.sign(asset_ret.sub(cash_ret, axis=0)))
    stacked = sum(scores) / len(scores)
    return stacked


def fast_filter(prices: pd.DataFrame, window: int = 50) -> pd.DataFrame:
    """Binary above/below short moving average -- the crash-response leg."""
    sma = prices.rolling(window, min_periods=window // 2).mean()
    return (prices > sma).astype(float)


def trend_scaler(
    prices: pd.DataFrame,
    rf_annual: pd.Series,
    *,
    lookbacks: tuple[int, ...] = DEFAULT_LOOKBACKS,
    fast_weight: float = 0.30,
    fast_window: int = 50,
    floor: float = 0.0,
) -> pd.DataFrame:
    """Per-asset exposure multiplier in [floor, 1].

    A fully negative trend across every lookback scales the sleeve to ``floor``;
    a fully positive one leaves it untouched.
    """
    slow = excess_momentum_score(prices, rf_annual, lookbacks=lookbacks)
    slow_gate = (0.5 + 0.5 * slow).clip(0.0, 1.0)
    fast_gate = fast_filter(prices, fast_window)
    blended = (1.0 - fast_weight) * slow_gate + fast_weight * fast_gate
    return blended.clip(lower=floor, upper=1.0).fillna(1.0)


def prices_from_returns(returns: pd.DataFrame) -> pd.DataFrame:
    """Synthetic total-return price levels, for sleeves defined by returns."""
    return (1.0 + returns.fillna(0.0)).cumprod()
