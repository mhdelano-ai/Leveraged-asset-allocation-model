"""The daily portfolio accounting identity -- single source of truth.

Every return the project reports flows through :func:`mark_to_market` and
:func:`apply_rebalance`. Keeping the arithmetic in one place is what makes the
identity tests meaningful: if these two functions are right, the engine is right.

Conventions, chosen once:

* ``w`` are fractions of **equity**, so ``L = sum(w)`` is gross leverage and
  ``k = 1 - L`` is the cash weight. ``k < 0`` means borrowing.
* Financing accrues on **calendar** days at **ACT/360** (money-market
  convention), so a position carried over a weekend pays three days of interest
  while earning two days of nothing. Coupon accrual inside the bond synthesiser
  uses ACT/365.25 instead -- the two conventions are genuinely different and are
  deliberately not unified.
* Transaction costs are charged one-way per unit of |weight change|.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DayResult:
    equity: float
    weights: np.ndarray
    gross_return: float
    financing_cost: float
    transaction_cost: float
    turnover: float


def mark_to_market(
    equity: float,
    weights: np.ndarray,
    asset_returns: np.ndarray,
    *,
    borrow_rate: float,
    lend_rate: float,
    day_count: float,
) -> tuple[float, np.ndarray, float, float]:
    """Advance one day before any rebalancing.

    Returns ``(equity_after, drifted_weights, asset_pnl, financing_cost)`` where
    both P&L terms are expressed as fractions of the *entering* equity.
    """
    leverage = float(weights.sum())
    cash_weight = 1.0 - leverage
    rate = borrow_rate if cash_weight < 0.0 else lend_rate
    financing = cash_weight * rate * day_count / 360.0

    asset_pnl = float(weights @ asset_returns)
    growth = 1.0 + asset_pnl + financing

    # A levered book can be wiped out. Report ruin rather than silently flooring
    # it -- a margin account genuinely can go to zero, and hiding that would make
    # the margin-vs-LETF comparison meaningless.
    if growth <= 0.0:
        return 0.0, np.zeros_like(weights), asset_pnl, financing

    equity_after = equity * growth
    drifted = weights * (1.0 + asset_returns) / growth
    return equity_after, drifted, asset_pnl, financing


def apply_rebalance(
    equity: float,
    drifted_weights: np.ndarray,
    target_weights: np.ndarray,
    cost_per_unit: np.ndarray,
) -> tuple[float, np.ndarray, float, float]:
    """Trade to target and charge costs.

    Returns ``(equity_after, new_weights, transaction_cost_fraction, turnover)``.
    """
    delta = np.abs(target_weights - drifted_weights)
    turnover = float(delta.sum())
    cost = float(delta @ cost_per_unit)
    return equity * (1.0 - cost), target_weights.copy(), cost, turnover


def needs_rebalance(
    drifted_weights: np.ndarray,
    target_weights: np.ndarray,
    *,
    calendar_trigger: bool,
    abs_band: float,
    rel_band: float,
    lev_band: float,
    risk_off_move: float,
    risk_off_threshold: float,
) -> bool:
    """Band logic. Any one trigger fires a full rebalance to target.

    Leverage error gets its own, tighter band and is checked daily: being levered
    2.6x when you meant 2.0x is far more dangerous than holding 4% too much gold.
    A large risk-off move overrides every band and calendar -- when the throttle
    says cut, waiting for month-end defeats the purpose.
    """
    # An empty book must always be established, whatever the bands say.
    # Otherwise wide bands leave the portfolio permanently in cash and every
    # downstream metric silently describes a position that was never taken.
    if not drifted_weights.any() and target_weights.any():
        return True
    if risk_off_move > risk_off_threshold:
        return True
    if abs(float(drifted_weights.sum()) - float(target_weights.sum())) > lev_band:
        return True
    if calendar_trigger:
        return True
    tol = np.maximum(abs_band, rel_band * np.abs(target_weights))
    return bool(np.any(np.abs(target_weights - drifted_weights) > tol))
