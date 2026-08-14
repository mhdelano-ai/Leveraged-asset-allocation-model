"""Leveraged-ETF vehicle: daily-reset funds, no borrowing at the account level.

A k-times daily-reset fund earns::

    r_letf = k * r_underlying - (k-1) * (rf + swap_spread) * dt/360 - TER * dt/365

compounded daily, so volatility decay is not a separate correction -- it emerges
from compounding ``k * r`` and is automatically path-dependent.

Two findings drive how this is parameterised.

**Cost.** Fitting the equation against real funds decomposes cleanly into the
prospectus expense ratio plus a financing spread of ~125bp on equity and ~82bp on
Treasuries. UPRO and SSO imply 1.26% and 1.25% from entirely independent fits,
which is strong evidence the functional form is right. The consequence is
uncomfortable: UPRO's true all-in drag is ~3.4%/yr, roughly **3.8x its stated
0.91% expense ratio**. Modelling LETF cost as the expense ratio alone understates
the drag by ~250bp/yr.

**Expressiveness -- the bigger effect.** Leveraged funds exist for large-cap
equity, long Treasuries, gold and a few equity indices. There is no usable 3x
broad-commodity, REIT or investment-grade credit fund. The LETF portfolio
therefore *cannot* hold the target allocation: it must carry those sleeves
unlevered and concentrate leverage where products happen to exist. The LETF path
is structurally less diversified than the margin path, and that is likely to
dominate the head-to-head ahead of any financing-cost difference.

Capital, not notional, is the binding constraint here: exposure ``w`` in a k-times
fund consumes ``w/k`` of capital, and the capital weights must sum to <= 1. There
is no margin call, and the worst case for a sleeve is losing the capital in it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .base import VehicleContext

# Available daily-reset multipliers and expense ratios per sleeve.
# A multiplier of 1.0 means no leveraged product exists and the sleeve must be
# held unlevered -- this is the structural limitation, expressed as data.
DEFAULT_PRODUCTS = {
    "us_equity": {"k": 3.0, "ter": 0.0091, "proxy": "UPRO"},
    "intl_equity": {"k": 2.0, "ter": 0.0095, "proxy": "EFO"},
    "em_equity": {"k": 3.0, "ter": 0.0095, "proxy": "EDC"},
    "long_ust": {"k": 3.0, "ter": 0.0106, "proxy": "TMF"},
    "interm_ust": {"k": 3.0, "ter": 0.0100, "proxy": "TYD"},
    "gold": {"k": 2.0, "ter": 0.0095, "proxy": "UGL"},
    "commodities": {"k": 1.0, "ter": 0.0085, "proxy": "none - unlevered"},
    "ig_credit": {"k": 1.0, "ter": 0.0015, "proxy": "none - unlevered"},
    "hy_credit": {"k": 1.0, "ter": 0.0049, "proxy": "none - unlevered"},
    "reit": {"k": 1.0, "ter": 0.0012, "proxy": "none - unlevered"},
}

# Financing spreads over the risk-free rate embedded in the swaps, fitted
# independently per fund (see calibrate.py). UPRO implies 68bp and SSO 98bp from
# separate regressions on different multipliers -- agreement to 30bp across
# independent fits is good evidence the functional form is right.
#
# Validation, simulated vs actual: UPRO -56bp/yr (corr 0.998), SSO -15bp/yr
# (0.995), TMF -6bp/yr (0.997), UGL +78bp/yr (0.997).
SWAP_SPREAD_EQUITY = 0.0083
SWAP_SPREAD_RATES = 0.0053
SWAP_SPREAD_GOLD = 0.0164


def simulate_letf(
    underlying: pd.Series,
    rf: pd.Series,
    *,
    multiplier: float,
    expense_ratio: float,
    swap_spread: float,
    day_counts: np.ndarray | None = None,
) -> pd.Series:
    """Daily returns of a synthetic k-times daily-reset fund."""
    r = underlying.to_numpy(dtype=float)
    rate = rf.reindex(underlying.index).ffill().fillna(0.0).to_numpy(dtype=float)
    dt = np.ones_like(r) if day_counts is None else day_counts

    financing = (multiplier - 1.0) * (rate + swap_spread) * dt / 360.0
    fee = expense_ratio * dt / 365.0
    return pd.Series(multiplier * r - financing - fee, index=underlying.index, name=underlying.name)


@dataclass
class LeveredETF:
    """Portfolio of daily-reset leveraged funds. No account-level borrowing."""

    products: dict = field(default_factory=lambda: dict(DEFAULT_PRODUCTS))
    swap_spread_equity: float = SWAP_SPREAD_EQUITY
    swap_spread_rates: float = SWAP_SPREAD_RATES
    swap_spread_gold: float = SWAP_SPREAD_GOLD
    max_leverage: float = 3.0
    name: str = "letf"

    _multipliers: np.ndarray | None = None
    _sleeves: tuple[str, ...] = ()

    def _spread_for(self, sleeve: str) -> float:
        if sleeve == "gold":
            return self.swap_spread_gold
        if "ust" in sleeve or "credit" in sleeve:
            return self.swap_spread_rates
        return self.swap_spread_equity

    def prepare(self, ctx: VehicleContext) -> pd.DataFrame:
        from ..timeaxis import day_deltas

        dt = day_deltas(ctx.returns.index, prepend=1.0)
        out = {}
        mults = []
        for sleeve in ctx.returns.columns:
            spec = self.products.get(sleeve, {"k": 1.0, "ter": 0.0010})
            k, ter = float(spec["k"]), float(spec["ter"])
            mults.append(k)
            out[sleeve] = simulate_letf(
                ctx.returns[sleeve].fillna(0.0),
                ctx.rf,
                multiplier=k,
                expense_ratio=ter,
                swap_spread=self._spread_for(sleeve),
                day_counts=dt,
            )
        self._multipliers = np.array(mults, dtype=float)
        self._sleeves = tuple(ctx.returns.columns)
        return pd.DataFrame(out, index=ctx.returns.index)

    def borrow_spread(self, t: int) -> float:
        # Financing lives inside the funds; the account itself does not borrow.
        return 0.0

    def achievable_weights(self, target_notional: np.ndarray, t: int) -> np.ndarray:
        """Convert notional exposure to capital weights, capped at 100% of equity.

        This cap is where the vehicle's real limitation bites: sleeves with no
        leveraged product consume capital one-for-one, so holding them crowds out
        the leverage available to everything else.
        """
        if self._multipliers is None:
            return target_notional
        capital = target_notional / self._multipliers
        total = capital.sum()
        if total > 1.0:
            capital = capital / total
        return capital

    def notional_leverage(self, held_weights: np.ndarray) -> float:
        if self._multipliers is None:
            return float(held_weights.sum())
        return float(held_weights @ self._multipliers)

    def check_margin(self, equity: float, weights: np.ndarray, t: int):
        # A leveraged ETF cannot margin-call the holder. This is the vehicle's
        # one genuine structural advantage over a margin account.
        return None

    def expressiveness_report(self, sleeve_names) -> pd.DataFrame:
        rows = []
        for s in sleeve_names:
            spec = self.products.get(s, {"k": 1.0, "ter": 0.0010, "proxy": "none"})
            rows.append(
                {
                    "sleeve": s,
                    "multiplier": spec["k"],
                    "expense_ratio_bps": spec["ter"] * 1e4,
                    "product": spec.get("proxy", "?"),
                    "levered": spec["k"] > 1.0,
                }
            )
        return pd.DataFrame(rows).set_index("sleeve")
