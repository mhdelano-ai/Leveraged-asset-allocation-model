"""What a fixed levered allocation does inside a real Reg-T brokerage account.

The Kelly solvers in :mod:`lam.kelly.solve` assume a frictionless account: any
leverage is available, borrowing costs one spread, and a bad month simply
shrinks the book. A retail margin account -- Robinhood's included -- violates
all three, and the violation that matters is the third:

    **A maintenance breach converts a drawdown into a realised loss.** The broker
    sells at the bottom, the position is not there for the recovery, and the
    investor's actual compound growth diverges permanently from the growth rate
    the Kelly calculation promised.

This module simulates that. It runs daily because a maintenance breach is a path
event inside the month: a book that ends March 2020 down 25% touched down 34%
on the way, and the broker acts on the touch. The only asymmetry Kelly's
continuous-rebalancing story cannot survive is being forced to sell.

Modelled explicitly:

* Reg-T initial leverage cap (2x) and per-asset maintenance requirements.
* House requirement hikes when the market is falling -- brokers raise
  requirements procyclically, which is to say exactly when you cannot meet them.
* An intraday stress factor, because the requirement is tested against the low,
  not the close.
* Slippage on forced sales, at a wider spread than voluntary trades.
* Interest accrued daily on the debit balance, on a calendar-day basis.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .panel import ASSETS, KellyPanel

# Reg-T maintenance per dollar of position value. Broad equity and bond funds at
# a retail broker; Robinhood's published range is 25%-100% depending on the
# security, with 25% the floor for a diversified fund.
MAINTENANCE = {
    "us_equity": 0.25,
    "intl_equity": 0.25,
    "us_bonds": 0.25,
    "intl_bonds": 0.25,
    "long_treasuries": 0.25,
    "reits": 0.25,
}

REG_T_MAX_LEVERAGE = 2.0


@dataclass(frozen=True)
class AccountResult:
    equity: pd.Series
    leverage: pd.Series
    calls: pd.DatetimeIndex
    ruined: bool
    ruin_date: pd.Timestamp | None
    target_leverage: float

    @property
    def summary(self) -> dict:
        eq = self.equity
        years = (eq.index[-1] - eq.index[0]).days / 365.25
        peak = eq.cummax()
        out = {
            "target_leverage": self.target_leverage,
            "margin_calls": len(self.calls),
            "first_call": self.calls[0] if len(self.calls) else None,
            "ruined": self.ruined,
            "ruin_date": self.ruin_date,
            "max_drawdown": float((eq / peak - 1.0).min()),
            "final_multiple": float(eq.iloc[-1] / eq.iloc[0]),
        }
        out["cagr"] = (
            float(out["final_multiple"] ** (1.0 / years) - 1.0)
            if out["final_multiple"] > 0
            else float("nan")
        )
        daily = eq.pct_change().dropna()
        out["vol"] = float(daily.std(ddof=1) * np.sqrt(252))
        return out


def simulate(
    panel: KellyPanel,
    weights: np.ndarray | pd.Series,
    *,
    target_leverage: float | None = None,
    financing_spread: float = 0.012,
    maintenance: dict[str, float] | None = None,
    procyclicality: float = 0.5,
    intraday_stress: float = 1.35,
    forced_slippage: float = 0.004,
    rebalance_slippage: float = 0.0005,
    liquidation_buffer: float = 0.10,
    relever_after_call: bool = True,
) -> AccountResult:
    """Hold ``weights`` in a Reg-T margin account, rebalanced monthly.

    ``weights`` is a notional allocation; if ``target_leverage`` is given the
    vector is rescaled to that gross exposure. ``relever_after_call=False`` is the
    behavioural case: the investor who has just been sold out at the bottom does
    not put the leverage back on.
    """
    if panel.daily is None or panel.daily_cash is None:
        raise ValueError("panel has no daily data; rebuild it with build_panel()")

    w = np.asarray(weights, dtype=float)
    if target_leverage is not None:
        gross = np.abs(w).sum()
        if gross <= 0:
            raise ValueError("cannot rescale an all-cash weight vector")
        w = w * (target_leverage / gross)
    target = float(np.abs(w).sum())

    maint = np.array([(maintenance or MAINTENANCE)[a] for a in ASSETS], dtype=float)
    returns = panel.daily.to_numpy()
    cash_rate = panel.daily_cash.to_numpy()
    index = panel.daily.index
    months = index.to_period("M")

    equity = 1.0
    positions = equity * w  # dollar notional per sleeve
    debt = positions.sum() - equity

    equity_path = np.empty(len(index))
    leverage_path = np.empty(len(index))
    calls: list[pd.Timestamp] = []
    ruin_date = None
    levered = True

    # A 60-day trailing drawdown drives the house-requirement hike: brokers raise
    # requirements after the market has already fallen, not before.
    market = pd.Series(returns[:, 0], index=index).add(1).cumprod()
    stress = (market / market.cummax() - 1.0).to_numpy()

    for t in range(len(index)):
        positions = positions * (1.0 + returns[t])
        # Interest on the debit balance: the day's bill accrual plus the broker's
        # spread over the same calendar days. A *credit* balance -- any book held
        # below 1x gross -- earns the bill rate instead, which is the whole reason
        # sub-1x allocations are worth considering at a 3.8% cash rate.
        if debt > 0:
            debt *= 1.0 + cash_rate[t] + financing_spread * _day_count(index, t) / 365.0
        elif debt < 0:
            debt *= 1.0 + cash_rate[t]
        value = positions.sum()
        equity = value - debt

        if equity <= 0:
            ruin_date = index[t]
            equity_path[t:] = 0.0
            leverage_path[t:] = np.nan
            return AccountResult(
                equity=pd.Series(equity_path, index=index),
                leverage=pd.Series(leverage_path, index=index),
                calls=pd.DatetimeIndex(calls),
                ruined=True,
                ruin_date=ruin_date,
                target_leverage=target,
            )

        if debt > 0:
            hike = 1.0 + procyclicality * max(0.0, -stress[t])
            required = float(np.abs(positions) @ maint) * hike * intraday_stress
            if equity < required:
                calls.append(index[t])
                # Sell pro rata until the requirement is met with a buffer, paying
                # a wider spread than a voluntary trade would.
                denom = float(np.abs(positions) @ maint) * hike * intraday_stress
                keep = min(1.0, max(0.0, equity * (1.0 - liquidation_buffer) / denom))
                sold = positions * (1.0 - keep)
                cost = np.abs(sold).sum() * forced_slippage
                positions = positions * keep
                debt = max(0.0, debt - np.abs(sold).sum() + cost)
                equity = positions.sum() - debt
                if not relever_after_call:
                    levered = False

        equity_path[t] = equity
        leverage_path[t] = positions.sum() / equity if equity > 0 else np.nan

        month_end = t + 1 == len(index) or months[t + 1] != months[t]
        if month_end and equity > 0:
            goal = w if levered else w / target
            desired = equity * goal
            turnover = np.abs(desired - positions).sum()
            equity -= turnover * rebalance_slippage
            positions = equity * goal
            debt = positions.sum() - equity

    return AccountResult(
        equity=pd.Series(equity_path, index=index),
        leverage=pd.Series(leverage_path, index=index),
        calls=pd.DatetimeIndex(calls),
        ruined=False,
        ruin_date=None,
        target_leverage=target,
    )


def _day_count(index: pd.DatetimeIndex, t: int) -> float:
    if t == 0:
        return 1.0
    return float(min((index[t] - index[t - 1]).days, 5))


def leverage_ladder(
    panel: KellyPanel,
    weights: np.ndarray | pd.Series,
    *,
    levels: tuple[float, ...] = (1.0, 1.25, 1.5, 1.75, 2.0),
    **kwargs,
) -> pd.DataFrame:
    """Run :func:`simulate` across target leverage levels and tabulate."""
    rows = []
    for level in levels:
        result = simulate(panel, weights, target_leverage=level, **kwargs)
        rows.append(result.summary)
    return pd.DataFrame(rows).set_index("target_leverage")


def call_threshold(
    target_leverage: float,
    *,
    maintenance: float = 0.25,
    intraday_stress: float = 1.0,
    procyclicality: float = 0.0,
) -> float:
    """How far the holding can fall from the last rebalance before a margin call.

    The single most useful number for sizing leverage, and one an account holder
    can check without a backtest: at ``L`` times leverage against a maintenance
    requirement ``m``, the call fires once the position has fallen by

        ``d = 1 - (L - 1) / (L * (1 - m))``

    Returns the decline as a positive decimal; 1.0 means no decline short of a
    total loss triggers it. ``intraday_stress`` and ``procyclicality`` inflate
    ``m`` the way :func:`simulate` does, so the number matches the simulation
    rather than the textbook.
    """
    if target_leverage <= 1.0:
        return 1.0
    effective = maintenance * intraday_stress
    if procyclicality:
        # Solved at the breach itself: the hike depends on the decline that
        # triggers it, so iterate the fixed point a few times.
        decline = 0.0
        for _ in range(50):
            hiked = min(effective * (1.0 + procyclicality * decline), 0.99)
            decline = 1.0 - (target_leverage - 1.0) / (target_leverage * (1.0 - hiked))
        return max(0.0, min(1.0, decline))
    if effective >= 1.0:
        return 0.0
    return max(0.0, min(1.0, 1.0 - (target_leverage - 1.0) / (target_leverage * (1.0 - effective))))
