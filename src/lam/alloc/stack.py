"""Compose the allocation layers into weights and a leverage target.

Four layers, each answering a different question:

1. **Base weights** -- what should the unlevered mix be? (inverse-vol with
   cluster risk budgets)
2. **Trend** -- should any sleeve be turned down right now? (time-series
   momentum, excess of cash)
3. **Volatility target** -- how much leverage does that mix support? (with a
   crisis-correlation floor on the risk forecast)
4. **Drawdown throttle** -- applied inside the engine, since it needs realised
   equity.

Layers 1-3 are computed here in one vectorised pass over the whole history.
They are all causal -- rolling and EWMA estimators are right-aligned, and nothing
is standardised using full-sample statistics -- so precomputing them is not a
lookahead shortcut, and the engine applies row ``t`` to day ``t+1``'s returns.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import base_weights, trend, voltarget
from .vol import blended_vol

# Assumed typical trend participation before enough history exists to measure it.
NEUTRAL_PARTICIPATION = 0.70


@dataclass
class StackParams:
    """Free parameters of the allocation stack.

    Deliberately capped at eight. Everything else -- momentum lookbacks,
    covariance shrinkage, the vol-blend slow window -- is fixed a priori at
    literature-standard values. With only ~7 independent drawdown events in the
    entire historical record, every additional free parameter buys in-sample fit
    at the direct expense of out-of-sample meaning.
    """

    sigma_target: float = 0.11
    max_leverage: float = 3.0
    vol_halflife: float = 40.0
    trend_fast_weight: float = 0.30
    growth_budget: float = 0.40
    dd_budget_margin: float = 0.80
    leverage_ratchet_up: float = 0.10
    # Wide by default. The leverage target moves a little every day as the vol
    # forecast updates, so a tight band chases noise: at 0.10 this book turned
    # over 15x a year and paid ~93bp in transaction costs for the privilege.
    lev_band: float = 0.25

    def cluster_budgets(self) -> dict[str, float]:
        """Cluster risk budgets, with the non-growth share split proportionally."""
        growth = float(np.clip(self.growth_budget, 0.10, 0.80))
        rest = 1.0 - growth
        base = {"rates": 0.30, "real": 0.20, "credit": 0.10}
        total = sum(base.values())
        out = {k: v / total * rest for k, v in base.items()}
        out["growth"] = growth
        return out


@dataclass
class StackOutput:
    base_weights: pd.DataFrame
    target_leverage: pd.Series
    vol_forecast: pd.Series
    trend_scaler: pd.DataFrame
    raw_weights: pd.DataFrame
    participation: pd.Series | None = None

    def rescale(self, sigma_target: float, max_leverage: float) -> "StackOutput":
        """Re-derive the leverage target without recomputing any signals.

        Only ``sigma_target`` and ``max_leverage`` change here, and neither
        touches the composition or the volatility forecast -- so the optimiser
        can sweep both for the cost of an array multiply instead of rebuilding
        the covariance path on every candidate.
        """
        lev = voltarget.target_leverage(
            self.vol_forecast, sigma_target=sigma_target, max_leverage=max_leverage
        )
        if self.participation is not None:
            lev = lev * self.participation
        return StackOutput(
            base_weights=self.base_weights,
            target_leverage=lev.rename("target_leverage"),
            vol_forecast=self.vol_forecast,
            trend_scaler=self.trend_scaler,
            raw_weights=self.raw_weights,
            participation=self.participation,
        )


def build(
    returns: pd.DataFrame,
    rf_annual: pd.Series,
    *,
    params: StackParams | None = None,
    eligible: pd.DataFrame | None = None,
    stress_reference: pd.Series | None = None,
    use_trend: bool = True,
    use_crisis_cov: bool = True,
) -> StackOutput:
    """Compute causal base weights and a leverage target for every date."""
    params = params or StackParams()

    filled = returns.fillna(0.0)
    vols = blended_vol(returns, halflife=params.vol_halflife)

    raw = base_weights.build(
        returns, vols, eligible=eligible, budgets=params.cluster_budgets()
    )

    if use_trend:
        prices = trend.prices_from_returns(filled)
        scaler = trend.trend_scaler(
            prices, rf_annual, fast_weight=params.trend_fast_weight
        )
        if eligible is not None:
            scaler = scaler.where(eligible.reindex_like(scaler).fillna(False), 0.0)
        gated = raw * scaler.reindex_like(raw).fillna(1.0)
    else:
        scaler = pd.DataFrame(1.0, index=raw.index, columns=raw.columns)
        gated = raw

    # Composition is renormalised so the vol target sizes the mix actually held.
    # Trend's de-risking is then applied once, as a *relative* participation
    # factor -- how much of the book is trending, measured against how much
    # normally is.
    #
    # Measuring participation on an absolute scale would be a permanent tax
    # rather than a risk control: the gate is 0.5 at neutral trend, and sleeves
    # like gold and commodities beat cash barely more than half the time, so
    # absolute gross averages ~0.6. Multiplying leverage by that every day cuts
    # exposure 40% in all weather, including bull markets, while the vol target
    # has *already* sized the gated composition. Normalising against a trailing
    # median leaves participation at ~1 in normal conditions and pulls it down
    # only when trend is genuinely worse than usual.
    gross = gated.sum(axis=1)
    composition = gated.div(gross.replace(0.0, np.nan), axis=0).fillna(0.0)
    neutral = gross.expanding(min_periods=504).median().fillna(NEUTRAL_PARTICIPATION)
    neutral = neutral.clip(lower=0.20)
    trend_participation = (gross / neutral).clip(0.0, 1.0)

    forecast = voltarget.portfolio_vol_forecast(
        returns,
        composition,
        stress_reference=stress_reference if use_crisis_cov else None,
        halflife=60.0,
    )
    lev = voltarget.target_leverage(
        forecast, sigma_target=params.sigma_target, max_leverage=params.max_leverage
    )
    lev = (lev * trend_participation).rename("target_leverage")

    return StackOutput(
        base_weights=composition,
        target_leverage=lev,
        vol_forecast=forecast,
        trend_scaler=scaler,
        raw_weights=raw,
        participation=trend_participation,
    )
