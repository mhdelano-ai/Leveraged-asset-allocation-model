"""Simple state-machine rules, implemented exactly as stated so they can be beaten.

A published rule deserves to be tested on the same data, with the same execution
lag and the same costs as the model it is being compared against. Anything less
and the comparison measures the assumptions rather than the rules.

The rule implemented here is the widely-circulated four-state leveraged rotation:
trend decides *whether* to be levered, a short volatility filter decides whether
to be invested at all. It is a genuinely good design -- it needs no margin
account, so it cannot be margin-called and it is legal inside an IRA -- and its
weaknesses are specific rather than general.

Everything is causal: the state on day ``t`` is computed from data through day
``t-1``'s close and earns day ``t``'s return, matching the engine's one-day lag.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..financing.letf import simulate_letf
from ..timeaxis import day_deltas

TRADING_DAYS = 252

# Real-world frictions charged to whichever instrument a state actually holds.
QQQ_EXPENSE_RATIO = 0.0020
TQQQ_EXPENSE_RATIO = 0.0084
TQQQ_SWAP_SPREAD = 0.01044  # fitted in financing.calibrate
CASH_SPREAD = -0.0010  # brokerage cash yields slightly under the bill rate


@dataclass
class RuleParams:
    """The four-state rule's two free parameters, plus the instruments."""

    sma_window: int = 200
    vol_window: int = 20
    vol_threshold: float = 0.28
    levered_multiple: float = 3.0
    # Exposure held in each state, as a multiple of the underlying index.
    up_calm: float = 3.0     # above trend, calm -> leveraged fund
    up_wild: float = 0.0     # above trend, volatile -> cash
    down_calm: float = 1.0   # below trend, calm -> unlevered index
    down_wild: float = 0.0   # below trend, volatile -> cash
    # Hysteresis around the SMA crossing, as a fraction of the SMA: price must
    # close *outside* the band to flip trend, not merely cross the line. A flat,
    # choppy SMA (2011, 2015-16) otherwise flips the trend flag on noise alone.
    # 0.0 reproduces the plain `price > sma` rule this replaces. The volatility
    # leg has no band -- only the trend leg whipsaws on a flat average, and
    # widening the vol comparison was never asked for.
    sma_band: float = 0.01


def four_state_signal(
    index_tr: pd.Series,
    params: RuleParams | None = None,
    *,
    price_basis: pd.Series | None = None,
) -> pd.DataFrame:
    """The rule's state and target exposure for each day, lagged one day.

    Returns a frame with the raw inputs alongside the decision, so a state can
    always be explained rather than merely reported.

    ``price_basis``, if given, is a raw price *level* series (not returns) that
    the trend leg is tested against instead of the total-return curve compounded
    from ``index_tr``. This matters because a live dashboard has no dividend
    stream to reinvest and can only ever chart raw closes -- computing the SMA
    on total return while the page computes it on raw close lets the two drift
    apart by the dividend yield compounded over the window (~0.4% on 200 days
    for this index), which is enough to disagree about the state on a day near
    the line. Pass the same raw-close series the page would fetch so the baked
    backtest and the live signal share one definition. It is reindexed onto
    ``index_tr``'s calendar and forward-filled; ``None`` (the default) keeps the
    original total-return basis, so every caller that predates this parameter
    is unaffected.

    The volatility leg is left on ``index_tr`` regardless: the dividend accrual
    is a near-constant daily add-on, so it shifts a 200-day *level* average
    enough to matter but leaves a 20-day *standard deviation* untouched to many
    more decimal places than the 28% threshold could ever resolve.
    """
    p = params or RuleParams()

    price = (1.0 + index_tr.fillna(0.0)).cumprod()
    trend_price = price if price_basis is None else price_basis.reindex(index_tr.index).ffill()
    sma = trend_price.rolling(p.sma_window, min_periods=p.sma_window).mean()
    vol = index_tr.rolling(p.vol_window, min_periods=p.vol_window).std() * np.sqrt(TRADING_DAYS)

    # Hysteresis around the crossing: score +1/0 only on days that close outside
    # the band, leave every day inside it as NaN, then forward-fill so each of
    # those days inherits the last decisive crossing -- "hold the previous
    # state" without a Python loop. `sma_band=0.0` puts both thresholds on the
    # SMA itself, so the two masks partition every day the SMA exists and this
    # reduces exactly to the old `price > sma` (NaN-vs-NaN comparisons below are
    # what make that so: with no band, a day before the SMA exists satisfies
    # neither `>` nor `<` against a NaN threshold and falls through to the same
    # "not above" reading the plain comparison gave it).
    decisive = pd.Series(np.nan, index=index_tr.index)
    decisive[trend_price > sma * (1.0 + p.sma_band)] = 1.0
    decisive[trend_price < sma * (1.0 - p.sma_band)] = 0.0
    # Nothing exists to hold over from before the SMA itself does. Seed that
    # first day with a plain (unbanded) reading so the series starts decided
    # rather than defaulting the whole warmup to "cash" once ffill runs dry.
    first = sma.first_valid_index()
    if first is not None and pd.isna(decisive.loc[first]):
        decisive.loc[first] = 1.0 if trend_price.loc[first] > sma.loc[first] else 0.0
    above_raw = decisive.ffill()

    # Both inputs are shifted: the decision for day t uses only closes up to t-1.
    above = (above_raw == 1.0).shift(1)
    calm = (vol < p.vol_threshold).shift(1)

    state = pd.Series(pd.NA, index=index_tr.index, dtype=object)
    state[above.eq(True) & calm.eq(True)] = "levered"
    state[above.eq(True) & calm.eq(False)] = "cash (volatile uptrend)"
    state[above.eq(False) & calm.eq(True)] = "unlevered"
    state[above.eq(False) & calm.eq(False)] = "cash (volatile downtrend)"

    exposure = pd.Series(np.nan, index=index_tr.index)
    exposure[state == "levered"] = p.up_calm
    exposure[state == "cash (volatile uptrend)"] = p.up_wild
    exposure[state == "unlevered"] = p.down_calm
    exposure[state == "cash (volatile downtrend)"] = p.down_wild

    return pd.DataFrame(
        {
            "price": price,
            "trend_price": trend_price,
            "sma": sma,
            "vol": vol,
            "above_sma": above,
            "calm": calm,
            # Unlagged: today's own banded reading, using today's own close --
            # i.e. the value that *tomorrow*'s (lagged) above_sma will copy.
            # A dashboard asking "how far would price need to move to flip"
            # wants this one, not `above_sma`, which already describes today's
            # position decided from yesterday's close.
            "trend_held": above_raw == 1.0,
            "state": state.fillna("warmup"),
            "exposure": exposure.fillna(0.0),
        }
    )


def backtest_four_state(
    index_tr: pd.Series,
    rf_annual: pd.Series,
    params: RuleParams | None = None,
    *,
    cost_per_unit: float = 0.0002,
    cost_era_multipliers: tuple[tuple[str, float], ...] = (
        ("1990-01-01", 4.0), ("2000-01-01", 2.0), ("2100-01-01", 1.0),
    ),
    price_basis: pd.Series | None = None,
) -> tuple[pd.Series, pd.DataFrame]:
    """Daily returns of the rule, holding the instrument each state names.

    ``price_basis`` is forwarded to :func:`four_state_signal` unchanged -- see
    its docstring. Pass it when the caller wants the trend leg tested against
    raw closes rather than the compounded total-return curve.

    The leveraged state is priced through the calibrated daily-reset fund rather
    than as ``3 x index``: the volatility decay, the swap spread and the expense
    ratio are what a 3x fund actually costs, and treating the state as free
    leverage flatters it by roughly 3%/yr.
    """
    p = params or RuleParams()
    sig = four_state_signal(index_tr, p, price_basis=price_basis)
    rf = rf_annual.reindex(index_tr.index).ffill().fillna(0.0)
    days = day_deltas(index_tr.index, prepend=1.0)

    levered = simulate_letf(
        index_tr, rf,
        multiplier=p.levered_multiple,
        expense_ratio=TQQQ_EXPENSE_RATIO,
        swap_spread=TQQQ_SWAP_SPREAD,
        day_counts=days,
    )
    unlevered = index_tr - QQQ_EXPENSE_RATIO * days / 365.0
    cash = (rf + CASH_SPREAD) * days / 360.0

    exposure = sig["exposure"].to_numpy()
    out = np.where(
        exposure >= p.levered_multiple - 1e-9, levered.to_numpy(),
        np.where(exposure >= 1e-9, unlevered.to_numpy(), cash.to_numpy()),
    )

    # A state change is a full switch out of one instrument and into another,
    # so it costs two one-way trades, not one.
    mult = np.ones(len(index_tr))
    prev = pd.Timestamp("1800-01-01")
    for cutoff, factor in cost_era_multipliers:
        m = (index_tr.index >= prev) & (index_tr.index < pd.Timestamp(cutoff))
        mult[m] = factor
        prev = pd.Timestamp(cutoff)
    switched = sig["state"].ne(sig["state"].shift(1)).to_numpy()
    out = out - switched * 2.0 * cost_per_unit * mult

    return pd.Series(out, index=index_tr.index, name="four_state"), sig


def state_table(index_tr: pd.Series, returns: pd.Series, sig: pd.DataFrame) -> pd.DataFrame:
    """Time spent in each state, and what was earned there."""
    rows = []
    for state, grp in returns.groupby(sig["state"]):
        idx = grp.index
        rows.append(
            {
                "state": state,
                "days": len(grp),
                "share": len(grp) / len(returns),
                "ann_return": float(grp.mean()) * TRADING_DAYS,
                "ann_vol": float(grp.std()) * np.sqrt(TRADING_DAYS),
                "index_ann_return": float(index_tr.reindex(idx).mean()) * TRADING_DAYS,
            }
        )
    return pd.DataFrame(rows).set_index("state").sort_values("days", ascending=False)


def switch_count(sig: pd.DataFrame) -> float:
    """State changes per year -- the rule's turnover, and its whipsaw exposure."""
    changes = int(sig["state"].ne(sig["state"].shift(1)).sum())
    years = (sig.index[-1] - sig.index[0]).days / 365.25
    return changes / years
