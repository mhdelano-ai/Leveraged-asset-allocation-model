"""Leverage rule for a concentrated book whose objective is compound return.

The diversified stack in ``alloc.stack`` answers "how much leverage does this mix
support under a drawdown constraint". This one answers a different question:
**one asset, no diversification, maximise the compound growth rate.** The
mathematics is Kelly's, and it says the entire game is choosing ``L(t)``.

Four layers, in the order they bind:

1. **Volatility target** -- ``L = sigma_target / sigma_hat``. Sets the scale.
2. **Trend gate** -- multiplies exposure down when the asset is losing to cash.
   This is the layer that carries the strategy. Estimated on the S&P 500 since
   1960, the growth-optimal leverage above the 200-day average is **5.43**; below
   it, **-0.06**. There is no leverage that makes a downtrending index worth
   holding, and no volatility target that discovers this, because volatility says
   nothing about the sign of the drift.
3. **Volatility regime** -- a further, *non-proportional* cut in the top of the
   trailing volatility distribution. See ``alloc.regime`` for why ``1/sigma`` is
   not enough there.
4. **Drawdown throttle** -- applied inside the engine against realised equity.
   Trend and volatility are both forecasts and can both be wrong at once; this is
   the only layer that responds to money actually lost.

Layers 1-3 are computed here in one causal pass. Every estimator is right-aligned
and nothing is standardised on full-sample statistics, so precomputing is not a
lookahead shortcut -- the engine still applies row ``t`` to day ``t+1``'s return.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import regime, trend

# Fixed a priori rather than searched. With roughly seven independent bear
# markets in the Nasdaq-100 record, every free parameter is expensive.
TREND_LOOKBACKS = (63, 126, 252)
FAST_WINDOW = 50
STRESS_PERCENTILE = 0.80
REGIME_WINDOW = 1260

# Momentum score at which the gate reaches full risk. The score is the mean of
# sign(excess return) over the three lookbacks, so it lives on {-1, -1/3, 1/3, 1}
# and 1/3 means "two of three lookbacks positive".
TREND_FULL_RISK_SCORE = 1.0 / 3.0


def growth_gate(
    prices: pd.DataFrame,
    rf_annual: pd.Series,
    *,
    fast_weight: float = 0.30,
    floor: float = 0.0,
) -> pd.Series:
    """Trend participation for a growth book, in ``[floor, 1]``.

    This is deliberately *not* ``alloc.trend.trend_scaler``, which the
    constrained model uses. That mapping is ``0.5 + 0.5 * score``, so a neutral
    trend reads 0.5 and the long-run average gate lands near 0.7 -- a permanent
    30% haircut on exposure in all weather. For a book whose objective is
    compound return that is not a risk control, it is a 30% tax, and it costs
    more than every crash it avoids: with it the Nasdaq-100 book compounds at
    8.6%/yr against the index's 15.8%.

    So the gate reaches **full risk** as soon as two of the three lookbacks are
    positive, and ramps to ``floor`` only as momentum turns genuinely negative.
    The strategy runs unencumbered in ordinary markets and spends its risk budget
    on the states that actually destroy levered capital.
    """
    score = trend.excess_momentum_score(prices, rf_annual, lookbacks=TREND_LOOKBACKS)
    slow = ((score + 1.0) / (TREND_FULL_RISK_SCORE + 1.0)).clip(0.0, 1.0)
    fast = trend.fast_filter(prices, FAST_WINDOW)
    blended = (1.0 - fast_weight) * slow + fast_weight * fast
    gate = blended.mean(axis=1) if blended.shape[1] > 1 else blended.iloc[:, 0]
    return (floor + (1.0 - floor) * gate.clip(0.0, 1.0)).fillna(1.0).rename("trend_gate")


@dataclass
class GrowthParams:
    """Eight free parameters, matching the discipline of the constrained model.

    ``sigma_target`` and ``max_leverage`` are the return dials; the other six are
    risk controls. The search space in ``opt.growth`` deliberately allows
    ``max_leverage`` up to 4x so the optimiser is free to reject leverage on the
    evidence rather than by assumption.
    """

    sigma_target: float = 0.25
    max_leverage: float = 3.0
    vol_halflife: float = 40.0
    trend_fast_weight: float = 0.30
    # Exposure multiplier when every trend lookback is negative. Zero means the
    # book goes fully to cash rather than riding a downtrend at reduced size.
    trend_floor: float = 0.0
    # Leverage multiplier at the very top of the trailing volatility distribution.
    stress_cap: float = 0.35
    # Absolute drawdown at which the throttle takes exposure to zero. A
    # return-maximising book must tolerate deep drawdowns, so this is a
    # ruin-avoidance backstop, not a comfort setting.
    dd_budget: float = 0.50
    leverage_ratchet_up: float = 0.10

    # Fixed: a turnover control rather than a risk control, and tuned once.
    lev_band: float = 0.25


# A finite stand-in for "infinite leverage", used only when the volatility
# forecast is exactly zero. It is always clipped to ``max_leverage`` afterwards.
_UNBOUNDED = 1e6


def _vol_leverage(vol: pd.Series, sigma_target: float) -> pd.Series:
    """Leverage implied by the volatility target, with the two edges separated.

    A **missing** volatility forecast and a **zero** one mean opposite things and
    must not be collapsed together. Missing is the warm-up period, where the
    right exposure is none at all: no position may be taken on an unmeasured
    risk. Zero is a riskless asset, where the growth-optimal leverage is
    unbounded and the cap should bind. Mapping both to zero -- as dividing and
    then filling NaN with 0 does -- silently turns a riskless asset into a flat
    book.
    """
    lev = sigma_target / vol.replace(0.0, np.nan)
    return lev.where(vol.notna(), 0.0).fillna(_UNBOUNDED)


@dataclass
class GrowthSignals:
    base_weights: pd.DataFrame
    target_leverage: pd.Series
    vol_forecast: pd.Series
    trend_gate: pd.Series
    stress_multiplier: pd.Series
    vol_leverage: pd.Series

    def rescale(self, sigma_target: float, max_leverage: float) -> "GrowthSignals":
        """Re-derive the leverage target without recomputing any signal.

        Neither dial touches the volatility forecast, the trend gate or the
        regime multiplier, so the optimiser sweeps both for the cost of an array
        multiply instead of rebuilding the estimators on every candidate.
        """
        vol_lev = _vol_leverage(self.vol_forecast, sigma_target)
        lev = (vol_lev * self.trend_gate * self.stress_multiplier).clip(0.0, max_leverage)
        lev = lev.fillna(0.0)
        return GrowthSignals(
            base_weights=self.base_weights,
            target_leverage=lev.rename("target_leverage"),
            vol_forecast=self.vol_forecast,
            trend_gate=self.trend_gate,
            stress_multiplier=self.stress_multiplier,
            vol_leverage=vol_lev.rename("vol_leverage"),
        )


def build(
    returns: pd.DataFrame,
    rf_annual: pd.Series,
    *,
    params: GrowthParams | None = None,
    use_trend: bool = True,
    use_regime: bool = True,
) -> GrowthSignals:
    """Causal composition and leverage target for a concentrated panel.

    ``returns`` may hold more than one column -- the composition is then equal
    weight, which keeps the machinery usable for a two- or three-sleeve growth
    book -- but the design target is a single sleeve.
    """
    params = params or GrowthParams()

    asset = returns.mean(axis=1) if returns.shape[1] > 1 else returns.iloc[:, 0]
    asset = asset.fillna(0.0)

    vol = regime.blended_vol(asset, halflife=params.vol_halflife)
    vol_lev = _vol_leverage(vol, params.sigma_target)

    if use_trend:
        prices = trend.prices_from_returns(returns.fillna(0.0))
        gate = growth_gate(
            prices,
            rf_annual,
            fast_weight=params.trend_fast_weight,
            floor=params.trend_floor,
        )
    else:
        gate = pd.Series(1.0, index=returns.index)

    if use_regime:
        stress = regime.stress_multiplier(
            vol,
            stress_percentile=STRESS_PERCENTILE,
            stress_cap=params.stress_cap,
            window=REGIME_WINDOW,
        )
    else:
        stress = pd.Series(1.0, index=returns.index)

    gate = gate.reindex(returns.index).fillna(1.0)
    stress = stress.reindex(returns.index).fillna(1.0)
    lev = (vol_lev * gate * stress).clip(0.0, params.max_leverage).fillna(0.0)

    composition = pd.DataFrame(
        1.0 / returns.shape[1], index=returns.index, columns=returns.columns
    )

    return GrowthSignals(
        base_weights=composition,
        target_leverage=lev.rename("target_leverage"),
        vol_forecast=vol.rename("vol_forecast"),
        trend_gate=gate.rename("trend_gate"),
        stress_multiplier=stress.rename("stress_multiplier"),
        vol_leverage=vol_lev.rename("vol_leverage"),
    )
