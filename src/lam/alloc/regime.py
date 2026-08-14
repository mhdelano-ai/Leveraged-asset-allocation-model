"""Volatility-regime classification, and why it is not the same as volatility targeting.

Volatility targeting already scales leverage by ``1/sigma``, so a natural
objection to a regime layer is that it is redundant. It is not, and the
Nasdaq-100 says so numerically. Splitting the sample by lagged volatility
quintile and estimating the growth-optimal leverage inside each
(``metrics.growth.regime_kelly``):

    ==============  ======  =============
    state           vol     Kelly optimum
    ==============  ======  =============
    vol Q4          25.2%   3.83
    vol Q5          42.8%   0.82
    ==============  ======  =============

Volatility targeting would cut exposure between those two buckets by the ratio
of volatilities, ``25.2 / 42.8 = 0.59``. The growth-optimal cut is
``0.82 / 3.83 = 0.21``. Targeting under-reacts by roughly a factor of three in
the worst quintile, because the top of the volatility distribution is not merely
more volatile -- it is where returns are fattest-tailed, most negatively skewed,
and where the daily reset costs the most. That gap is what this layer closes.

The percentile is computed against a **trailing** window, never the full sample.
A full-sample quantile would tell the 1987 book that its volatility was in the
95th percentile of a distribution that includes 2008 and 2020 -- information that
did not exist. The cost of doing it honestly is a warm-up period during which no
regime opinion is available, and the layer stands neutral there rather than
guessing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def blended_vol(
    returns: pd.Series,
    *,
    halflife: float = 40.0,
    slow_window: int = 756,
    slow_weight: float = 0.75,
) -> pd.Series:
    """Annualised EWMA volatility floored against trailing 3-year volatility.

    The floor is the important half. A pure fast EWMA bottoms out during
    volatility-suppressed melt-ups -- 1999, 2017, late 2021 -- so a sizer reading
    it takes maximum leverage immediately before the regime breaks. Flooring
    against three-year volatility removes most of that at almost no cost in
    normal markets.
    """
    fast = returns.ewm(halflife=halflife, min_periods=20).std() * np.sqrt(TRADING_DAYS)
    slow = returns.rolling(slow_window, min_periods=126).std() * np.sqrt(TRADING_DAYS)
    return np.maximum(fast, slow_weight * slow.fillna(fast)).rename("vol")


def trailing_percentile(series: pd.Series, *, window: int = 1260, min_periods: int = 252) -> pd.Series:
    """Where today's value sits in its own trailing distribution, in [0, 1].

    ``window`` is five years: long enough to contain both a calm stretch and a
    scare, short enough that a 2020 reading is not judged against 1987.
    """
    return series.rolling(window, min_periods=min_periods).rank(pct=True).rename("vol_pct")


def stress_multiplier(
    vol: pd.Series,
    *,
    stress_percentile: float = 0.80,
    stress_cap: float = 0.35,
    window: int = 1260,
) -> pd.Series:
    """Exposure multiplier in ``[stress_cap, 1]`` driven by the volatility regime.

    Flat at 1 while volatility sits below ``stress_percentile`` of its trailing
    distribution, then ramping linearly down to ``stress_cap`` at the very top.
    The ramp is deliberate: a step function at a percentile boundary makes the
    book trade its full risk budget on a one-day wobble in an estimator, and pays
    for it twice when volatility oscillates across the threshold.

    Before enough history exists to rank against, the multiplier is 1 -- neutral,
    not cautious. Being cautious there would silently make the first five years of
    every backtest a different strategy from the rest.
    """
    pct = trailing_percentile(vol, window=window)
    span = max(1e-6, 1.0 - stress_percentile)
    ramp = ((pct - stress_percentile) / span).clip(0.0, 1.0)
    return (1.0 - (1.0 - stress_cap) * ramp).fillna(1.0).rename("stress_multiplier")


def regime_labels(
    vol: pd.Series, *, stress_percentile: float = 0.80, calm_percentile: float = 0.40
) -> pd.Series:
    """Discrete ``calm`` / ``normal`` / ``stressed`` labels, for reporting only.

    The strategy uses the continuous :func:`stress_multiplier`; these labels
    exist so a results table can say how much time was spent where.
    """
    pct = trailing_percentile(vol)
    out = pd.Series("normal", index=vol.index, dtype=object)
    out[pct < calm_percentile] = "calm"
    out[pct >= stress_percentile] = "stressed"
    out[pct.isna()] = "warmup"
    return out.rename("regime")
