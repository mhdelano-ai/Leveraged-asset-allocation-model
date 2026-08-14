"""Margin vehicle: hold unlevered assets, borrow cash against them.

Returns are untouched -- this vehicle holds the real sleeves and simply borrows,
so unlike the LETF path it can express the target allocation exactly. Its cost
is elsewhere, in two places that a naive model omits:

**Financing is not infinitely elastic.** "Borrow at SOFR + 20bp via box spreads,
forever" assumes cheap financing is available exactly when it is not; box spreads
widened sharply in September 2019 and March 2020. A VIX-linked stress add-on
prices that in.

**You can be forced out at the bottom.** This is the decisive difference from
LETFs and the reason the margin model is worth building at all. When equity falls
below the maintenance requirement the broker liquidates -- crystallising the loss
and removing the position from the recovery. Two further details make it worse
than the textbook version, and both are modelled here: brokers *raise* house
requirements when volatility spikes (procyclical), and the requirement is tested
against intraday prices, not the close.

Portfolio margin did not exist for retail investors before 2007. Any pre-2007
result using the ``portfolio_margin`` schedule is counterfactual and is labelled
as such in the report; Reg-T at 2x is the honest pre-2007 assumption.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .base import VehicleContext

# Maintenance requirement per unit of position value.
REGT_MAINTENANCE = {
    "us_equity": 0.25, "intl_equity": 0.25, "em_equity": 0.30, "reit": 0.25,
    "hy_credit": 0.25, "ig_credit": 0.10, "commodities": 0.25, "gold": 0.25,
    "long_ust": 0.10, "interm_ust": 0.06,
}

PORTFOLIO_MARGIN_MAINTENANCE = {
    "us_equity": 0.15, "intl_equity": 0.15, "em_equity": 0.18, "reit": 0.15,
    "hy_credit": 0.15, "ig_credit": 0.07, "commodities": 0.15, "gold": 0.15,
    "long_ust": 0.06, "interm_ust": 0.04,
}

SCHEDULES = {
    "reg_t": {"maintenance": REGT_MAINTENANCE, "max_leverage": 2.0, "spread": 0.0075},
    "portfolio_margin": {
        "maintenance": PORTFOLIO_MARGIN_MAINTENANCE,
        "max_leverage": 3.0,
        "spread": 0.0100,
    },
    "box_spread": {
        "maintenance": PORTFOLIO_MARGIN_MAINTENANCE,
        "max_leverage": 3.0,
        "spread": 0.0020,
    },
}


@dataclass
class MarginAccount:
    """Cash borrowing against a portfolio of unlevered holdings."""

    schedule: str = "box_spread"
    vix_spread_beta: float = 2.0  # bp of extra spread per VIX point above 20
    maintenance_procyclicality: float = 0.5
    liquidation_buffer: float = 0.10
    base_slippage: float = 0.0050
    stressed_slippage_multiple: float = 3.0
    intraday_stress: float = 1.4
    name: str = "margin"

    _maintenance: np.ndarray | None = None
    _vix: np.ndarray | None = None
    _returns: np.ndarray | None = None
    _spread: float = 0.0
    max_leverage: float = 3.0
    _sleeves: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.schedule not in SCHEDULES:
            raise KeyError(f"unknown margin schedule {self.schedule!r}")
        spec = SCHEDULES[self.schedule]
        self._spread = spec["spread"]
        self.max_leverage = spec["max_leverage"]

    def prepare(self, ctx: VehicleContext) -> pd.DataFrame:
        spec = SCHEDULES[self.schedule]
        self._sleeves = tuple(ctx.returns.columns)
        self._maintenance = np.array(
            [spec["maintenance"].get(c, 0.25) for c in ctx.returns.columns], dtype=float
        )
        self._returns = ctx.returns.fillna(0.0).to_numpy(dtype=float)
        if ctx.vix is not None:
            # Forward-fill only. VIX starts in 1990, so back-filling would let
            # 1990s volatility set borrowing spreads and margin requirements in
            # the 1970s -- a small but real lookahead. Dates before the series
            # exists get the long-run median instead.
            aligned = ctx.vix.reindex(ctx.returns.index).ffill()
            self._vix = aligned.fillna(20.0).to_numpy(dtype=float)
        else:
            self._vix = np.full(len(ctx.returns), 20.0)
        # The margin account holds the real sleeves, unchanged.
        return ctx.returns

    def borrow_spread(self, t: int) -> float:
        vix = self._vix[t] if self._vix is not None else 20.0
        stress = self.vix_spread_beta * max(0.0, vix - 20.0) / 1e4
        return self._spread + stress

    def achievable_weights(self, target_notional: np.ndarray, t: int) -> np.ndarray:
        total = target_notional.sum()
        if total > self.max_leverage:
            return target_notional * (self.max_leverage / total)
        return target_notional

    def notional_leverage(self, held_weights: np.ndarray) -> float:
        return float(held_weights.sum())

    def check_margin(self, equity: float, weights: np.ndarray, t: int):
        """Test the maintenance requirement and force a deleverage if breached.

        The test uses an intraday stress applied to the day's move rather than
        the close. Checking the close only is a well-known source of fake
        survival: on 1987-10-19 and 2020-03-16 the intraday trough was far below
        it, and a book that "survived" on closing prices would in reality have
        been liquidated hours earlier.
        """
        if self._maintenance is None or equity <= 0:
            return None

        gross = float(np.abs(weights).sum())
        if gross <= 1.0:
            return None  # unlevered: nothing to call

        vix = self._vix[t] if self._vix is not None else 20.0
        hike = 1.0 + self.maintenance_procyclicality * max(0.0, vix / 20.0 - 1.0)

        # Everything below is expressed per unit of *current* equity: check_margin
        # runs after the book has been marked, so equity is 1.0 by construction
        # and `weights` are already fractions of it.
        #
        # Approximate the intraday trough by extending the day's adverse move --
        # synthetic sleeves have no intraday low. Both equity and position value
        # must be re-marked to that trough, or the comparison mixes a pre-move
        # requirement with a post-move equity.
        today = self._returns[t] if self._returns is not None else np.zeros_like(weights)
        adverse = np.minimum(today, 0.0) * (self.intraday_stress - 1.0)
        equity_low = 1.0 + float(weights @ adverse)
        position_low = np.abs(weights) * (1.0 + adverse)
        requirement = float(position_low @ self._maintenance) * hike

        if equity_low >= requirement:
            return None
        if equity_low <= 0.0:
            return 0.0, 0.0, "wiped out before liquidation could complete"

        # Sell down until equity covers the requirement with a buffer. Selling at
        # market does not change equity, so scaling positions by s scales the
        # requirement by s: solve equity >= (1 + buffer) * s * requirement.
        scale = float(
            np.clip(equity_low / ((1.0 + self.liquidation_buffer) * requirement), 0.0, 1.0)
        )
        slippage_rate = self.base_slippage * (
            self.stressed_slippage_multiple if vix > 40.0 else 1.0
        )
        sold = gross * (1.0 - scale)
        return (
            scale,
            slippage_rate * sold,
            f"equity {equity_low:.3f} below requirement {requirement:.3f}",
        )
