"""Accounting identities and no-lookahead discipline.

These are the tests that make the engine trustworthy. Every identity is asserted
to near machine precision -- an approximate match would hide exactly the kind of
slow leak (a missing day-count, a dropped financing term) that this project has
already been bitten by once.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lam.alloc.dd_throttle import DrawdownThrottle
from lam.engine.accounting import apply_rebalance, mark_to_market, needs_rebalance
from lam.engine.backtest import EngineConfig, run

TOL = 1e-12


def _flat_inputs(n=500, n_assets=2, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2000-01-03", periods=n)
    rets = pd.DataFrame(
        rng.normal(0.0003, 0.008, (n, n_assets)),
        index=idx,
        columns=[f"a{i}" for i in range(n_assets)],
    )
    return idx, rets


# --------------------------------------------------------------------------
# Accounting identities
# --------------------------------------------------------------------------

def test_unlevered_zero_cost_reproduces_weighted_return():
    w = np.array([0.6, 0.4])
    r = np.array([0.01, -0.02])
    equity, drifted, pnl, fin = mark_to_market(
        1.0, w, r, borrow_rate=0.05, lend_rate=0.05, day_count=1.0
    )
    # Weights sum to 1.0 exactly, so there is neither cash nor borrowing.
    assert fin == pytest.approx(0.0, abs=TOL)
    assert pnl == pytest.approx(0.6 * 0.01 + 0.4 * -0.02, abs=TOL)
    assert equity == pytest.approx(1.0 + pnl, abs=TOL)
    assert drifted.sum() == pytest.approx(1.0, abs=TOL)


def test_two_times_leverage_with_zero_returns_pays_exactly_the_borrow_rate():
    w = np.array([2.0])
    equity, _, pnl, fin = mark_to_market(
        1.0, w, np.array([0.0]), borrow_rate=0.06, lend_rate=0.01, day_count=1.0
    )
    expected = -1.0 * 0.06 * 1.0 / 360.0
    assert fin == pytest.approx(expected, abs=TOL)
    assert pnl == pytest.approx(0.0, abs=TOL)
    assert equity == pytest.approx(1.0 + expected, abs=TOL)


def test_underlevered_book_earns_cash_interest():
    w = np.array([0.5])
    _, _, _, fin = mark_to_market(
        1.0, w, np.array([0.0]), borrow_rate=0.06, lend_rate=0.04, day_count=1.0
    )
    assert fin == pytest.approx(0.5 * 0.04 / 360.0, abs=TOL)


def test_financing_accrues_over_calendar_days_not_trading_days():
    w = np.array([2.0])
    _, _, _, one = mark_to_market(
        1.0, w, np.array([0.0]), borrow_rate=0.06, lend_rate=0.0, day_count=1.0
    )
    _, _, _, three = mark_to_market(
        1.0, w, np.array([0.0]), borrow_rate=0.06, lend_rate=0.0, day_count=3.0
    )
    assert three == pytest.approx(3.0 * one, abs=TOL)


def test_turnover_cost_identity():
    drifted = np.array([0.5, 0.5])
    target = np.array([0.8, 0.2])
    costs = np.array([0.001, 0.002])
    equity, w, tc, turnover = apply_rebalance(1.0, drifted, target, costs)
    assert turnover == pytest.approx(0.6, abs=TOL)
    assert tc == pytest.approx(0.3 * 0.001 + 0.3 * 0.002, abs=TOL)
    assert equity == pytest.approx(1.0 - tc, abs=TOL)
    assert np.allclose(w, target)


def test_ruin_is_reported_not_floored():
    w = np.array([3.0])
    equity, weights, _, _ = mark_to_market(
        1.0, w, np.array([-0.5]), borrow_rate=0.0, lend_rate=0.0, day_count=1.0
    )
    assert equity == 0.0
    assert np.all(weights == 0.0)


def _costless(**kwargs) -> EngineConfig:
    base = dict(
        default_cost=0.0, borrow_spread=0.0, lend_spread=0.0,
        cost_era_multipliers=(("2100-01-01", 0.0),), leverage_ratchet_up=10.0,
    )
    return EngineConfig(**(base | kwargs))


def test_buy_and_hold_identity_single_asset():
    """A fully invested single asset must compound exactly, with no drift error."""
    idx, rets = _flat_inputs(300, 1, seed=3)
    base = pd.DataFrame(1.0, index=idx, columns=rets.columns)
    lev = pd.Series(1.0, index=idx)
    cfg = _costless(rebalance_freq=None, lev_band=1e9, abs_band=1e9, rel_band=1e9)
    res = run(rets, base, lev, rf_annual=pd.Series(0.0, index=idx), config=cfg)
    # Position is established on day 0 and earns from day 1 onward.
    manual = float(np.prod(1.0 + rets.iloc[1:, 0]))
    assert res.equity.iloc[-1] == pytest.approx(manual, rel=1e-12)


def test_daily_rebalanced_identity_two_assets():
    """Rebalanced daily to fixed weights, equity is the exact product of the mix."""
    idx, rets = _flat_inputs(300, 2, seed=3)
    base = pd.DataFrame(0.5, index=idx, columns=rets.columns)
    lev = pd.Series(1.0, index=idx)
    # Zero bands force a rebalance to target every single day.
    cfg = _costless(rebalance_freq=None, lev_band=0.0, abs_band=0.0, rel_band=0.0)
    res = run(rets, base, lev, rf_annual=pd.Series(0.0, index=idx), config=cfg)
    manual = float(np.prod(1.0 + (rets * 0.5).sum(axis=1).iloc[1:]))
    assert res.equity.iloc[-1] == pytest.approx(manual, rel=1e-12)


def test_levered_identity_matches_closed_form():
    """2x on a single asset, constant rate: r_p = 2r - rate/360 exactly, daily."""
    n = 200
    idx = pd.bdate_range("2000-01-03", periods=n)
    rets = pd.DataFrame(0.001, index=idx, columns=["a"])
    base = pd.DataFrame(1.0, index=idx, columns=["a"])
    lev = pd.Series(2.0, index=idx)
    rate = 0.06
    cfg = _costless(rebalance_freq=None, lev_band=0.0, abs_band=0.0, rel_band=0.0)
    res = run(rets, base, lev, rf_annual=pd.Series(rate, index=idx), config=cfg)

    day_counts = np.diff(idx.to_numpy().astype("datetime64[s]").astype("int64")) / 86_400.0
    expected = 2.0 * 0.001 - rate * day_counts / 360.0
    np.testing.assert_allclose(res.returns.to_numpy()[1:], expected, rtol=1e-12, atol=1e-15)


def test_leverage_never_exceeds_configured_maximum():
    """The cap binds on the target; realised leverage may drift within the band."""
    idx, rets = _flat_inputs(400, 3, seed=5)
    base = pd.DataFrame(1 / 3, index=idx, columns=rets.columns)
    lev = pd.Series(10.0, index=idx)  # ask for far more than allowed
    cfg = EngineConfig(max_leverage=2.0, leverage_ratchet_up=10.0, default_cost=0.0,
                       lev_band=0.05)
    res = run(rets, base, lev, rf_annual=pd.Series(0.02, index=idx), config=cfg)
    # Drift between band checks is expected and bounded by the band itself.
    assert res.leverage.max() <= 2.0 + cfg.lev_band + 0.02
    # And rebalances must pull it straight back to the cap.
    assert res.leverage.median() == pytest.approx(2.0, abs=0.05)


def test_higher_costs_never_improve_returns():
    idx, rets = _flat_inputs(600, 3, seed=11)
    base = pd.DataFrame(1 / 3, index=idx, columns=rets.columns)
    lev = pd.Series(1.5, index=idx)
    rf = pd.Series(0.03, index=idx)
    out = []
    for cost in (0.0, 0.001, 0.01):
        cfg = EngineConfig(
            default_cost=cost, cost_era_multipliers=(("2100-01-01", 1.0),),
            rebalance_freq="M",
        )
        out.append(run(rets, base, lev, rf_annual=rf, config=cfg).equity.iloc[-1])
    assert out[0] >= out[1] >= out[2]


# --------------------------------------------------------------------------
# No-lookahead
# --------------------------------------------------------------------------

def test_corrupting_the_future_cannot_change_the_past():
    """The decisive lookahead test.

    Replace every return from a cutoff onward with noise. Equity and weights on
    all earlier dates must be bit-identical. This catches centred windows,
    full-sample standardisation, backfilling and any fit-on-all-data estimator
    in a single assertion.
    """
    idx, rets = _flat_inputs(800, 3, seed=17)
    base = pd.DataFrame(1 / 3, index=idx, columns=rets.columns)
    lev = pd.Series(1.8, index=idx)
    rf = pd.Series(0.03, index=idx)
    cfg = EngineConfig(rebalance_freq="M")
    throttle = DrawdownThrottle(budget=0.4)

    cut = 500
    clean = run(rets, base, lev, rf_annual=rf, config=cfg, throttle=throttle)

    corrupted = rets.copy()
    rng = np.random.default_rng(99)
    corrupted.iloc[cut:] = rng.normal(-0.02, 0.05, corrupted.iloc[cut:].shape)
    throttle.reset()
    dirty = run(corrupted, base, lev, rf_annual=rf, config=cfg, throttle=throttle)

    np.testing.assert_allclose(
        clean.equity.iloc[:cut].to_numpy(), dirty.equity.iloc[:cut].to_numpy(), rtol=0, atol=0
    )
    np.testing.assert_allclose(
        clean.weights.iloc[:cut].to_numpy(), dirty.weights.iloc[:cut].to_numpy(), rtol=0, atol=0
    )


def test_truncating_the_panel_gives_the_same_path():
    idx, rets = _flat_inputs(700, 2, seed=23)
    base = pd.DataFrame(0.5, index=idx, columns=rets.columns)
    lev = pd.Series(1.5, index=idx)
    rf = pd.Series(0.02, index=idx)
    cfg = EngineConfig(rebalance_freq="M")

    full = run(rets, base, lev, rf_annual=rf, config=cfg)
    cut = 400
    part = run(
        rets.iloc[:cut], base.iloc[:cut], lev.iloc[:cut],
        rf_annual=rf.iloc[:cut], config=cfg,
    )
    # Compare up to cut-1: the truncated panel's final row is necessarily a
    # month-end for the calendar trigger (it is the last data there is), so it
    # rebalances where the full run does not. That is correct backtest
    # behaviour, not lookahead -- the paths must agree everywhere before it.
    np.testing.assert_allclose(
        full.equity.iloc[: cut - 1].to_numpy(),
        part.equity.iloc[: cut - 1].to_numpy(),
        rtol=1e-12,
        atol=1e-15,
    )


def test_execution_lag_delays_the_signal():
    """A signal must never be traded on the day it is computed."""
    n = 60
    idx = pd.bdate_range("2000-01-03", periods=n)
    rets = pd.DataFrame(0.0, index=idx, columns=["a"])
    rets.iloc[30, 0] = -0.20  # a single crash day

    base = pd.DataFrame(1.0, index=idx, columns=["a"])
    lev = pd.Series(1.0, index=idx)
    lev.iloc[30] = 0.0  # "perfect" foresight, switching off on the crash day
    cfg = EngineConfig(default_cost=0.0, rebalance_freq=None, leverage_ratchet_up=10.0,
                       lev_band=0.001)
    res = run(rets, base, lev, rf_annual=pd.Series(0.0, index=idx), config=cfg)
    # With a one-day lag the position is still on when the crash lands.
    assert res.returns.iloc[30] == pytest.approx(-0.20, abs=1e-9)


# --------------------------------------------------------------------------
# Rebalance bands and throttle
# --------------------------------------------------------------------------

def test_wider_bands_reduce_turnover():
    idx, rets = _flat_inputs(1000, 4, seed=31)
    base = pd.DataFrame(0.25, index=idx, columns=rets.columns)
    lev = pd.Series(1.5, index=idx)
    rf = pd.Series(0.02, index=idx)
    turns = []
    for band in (0.001, 0.02, 0.20):
        cfg = EngineConfig(abs_band=band, rel_band=band * 10, lev_band=max(band, 0.05),
                           rebalance_freq=None)
        turns.append(run(rets, base, lev, rf_annual=rf, config=cfg).turnover.sum())
    assert turns[0] >= turns[1] >= turns[2]


def test_leverage_band_fires_on_leverage_error():
    assert needs_rebalance(
        np.array([1.0, 1.0]), np.array([1.2, 1.2]),
        calendar_trigger=False, abs_band=1.0, rel_band=1.0,
        lev_band=0.10, risk_off_move=0.0, risk_off_threshold=0.05,
    )


def test_risk_off_overrides_every_band():
    assert needs_rebalance(
        np.array([1.0]), np.array([1.0]),
        calendar_trigger=False, abs_band=1e9, rel_band=1e9,
        lev_band=1e9, risk_off_move=0.5, risk_off_threshold=0.05,
    )


def test_throttle_shape_and_hysteresis():
    t = DrawdownThrottle(budget=0.40, dead_zone=0.5, recovery_level=0.4, locked_cap=0.5)
    assert t(0.0) == pytest.approx(1.0)
    assert t(0.15) == pytest.approx(1.0)  # inside the dead zone
    assert t(0.30) == pytest.approx(0.5, abs=1e-9)  # halfway from 0.20 to 0.40
    assert t(0.45) == pytest.approx(0.0)  # past the budget -> locked
    # Locked: a partial recovery must not restore full risk.
    assert t(0.25) <= 0.5
    # A full recovery below 0.4 * budget releases the lock.
    assert t(0.10) == pytest.approx(1.0)


def test_throttle_reaches_zero_before_the_budget_is_breached():
    t = DrawdownThrottle(budget=0.35)
    assert t(0.35) == pytest.approx(0.0)
    assert t(0.34) < 0.1
