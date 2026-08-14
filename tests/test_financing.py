"""Financing vehicle behaviour: LETF mechanics and margin liquidation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lam.financing.base import VehicleContext
from lam.financing.letf import LeveredETF, simulate_letf
from lam.financing.margin import MarginAccount


def _ctx(n=300, cols=("us_equity", "commodities"), seed=0, ret=None):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2010-01-04", periods=n)
    data = (
        np.full((n, len(cols)), ret)
        if ret is not None
        else rng.normal(0.0004, 0.01, (n, len(cols)))
    )
    returns = pd.DataFrame(data, index=idx, columns=list(cols))
    return VehicleContext(returns=returns, rf=pd.Series(0.02, index=idx), sleeve_names=cols)


# --------------------------------------------------------------------------
# LETF
# --------------------------------------------------------------------------

def test_letf_multiplies_returns_and_charges_costs():
    idx = pd.bdate_range("2010-01-04", periods=10)
    u = pd.Series(0.01, index=idx)
    rf = pd.Series(0.03, index=idx)
    dt = np.ones(len(idx))
    out = simulate_letf(u, rf, multiplier=3.0, expense_ratio=0.0091,
                        swap_spread=0.0083, day_counts=dt)
    expected = 3 * 0.01 - 2 * (0.03 + 0.0083) / 360.0 - 0.0091 / 365.0
    assert out.iloc[0] == pytest.approx(expected, abs=1e-15)


def test_letf_volatility_decay_emerges_from_compounding():
    """A round trip loses money at 3x even though the underlying is flat."""
    idx = pd.bdate_range("2010-01-04", periods=2)
    u = pd.Series([0.10, -0.10 / 1.10], index=idx)  # up 10%, then exactly back
    assert (1 + u).prod() == pytest.approx(1.0, abs=1e-12)
    lev = simulate_letf(u, pd.Series(0.0, index=idx), multiplier=3.0,
                        expense_ratio=0.0, swap_spread=0.0, day_counts=np.zeros(2))
    assert (1 + lev).prod() < 1.0  # decay, with zero fees and zero financing


def test_letf_cannot_margin_call():
    v = LeveredETF()
    v.prepare(_ctx())
    assert v.check_margin(0.01, np.array([1.0, 1.0]), 5) is None


def test_letf_capital_weights_are_capped_at_full_investment():
    v = LeveredETF()
    v.prepare(_ctx())
    # Ask for 3x equity and 1x commodities in notional terms.
    got = v.achievable_weights(np.array([3.0, 1.0]), 0)
    assert got.sum() <= 1.0 + 1e-12


def test_unleverable_sleeve_crowds_out_leverage():
    """The structural limitation, asserted rather than asserted-in-prose.

    Commodities have no leveraged product, so holding them consumes capital
    one-for-one and reduces the notional exposure the book can carry.
    """
    v = LeveredETF()
    v.prepare(_ctx(cols=("us_equity", "commodities")))
    equity_only = v.achievable_weights(np.array([2.0, 0.0]), 0)
    with_commods = v.achievable_weights(np.array([2.0, 1.0]), 0)
    assert v.notional_leverage(equity_only) > v.notional_leverage(with_commods)


def test_letf_notional_exceeds_capital():
    v = LeveredETF()
    v.prepare(_ctx(cols=("us_equity",)))
    capital = v.achievable_weights(np.array([3.0]), 0)
    assert capital.sum() == pytest.approx(1.0)
    assert v.notional_leverage(capital) == pytest.approx(3.0)


# --------------------------------------------------------------------------
# Margin
# --------------------------------------------------------------------------

def test_no_margin_call_when_unlevered():
    v = MarginAccount(schedule="reg_t")
    v.prepare(_ctx())
    assert v.check_margin(1.0, np.array([0.5, 0.4]), 10) is None


def test_margin_call_fires_on_a_large_adverse_move():
    ctx = _ctx(cols=("us_equity",), ret=-0.30)
    v = MarginAccount(schedule="reg_t")
    v.prepare(ctx)
    action = v.check_margin(1.0, np.array([3.0]), 10)
    assert action is not None
    scale, slippage, _ = action
    assert 0.0 <= scale < 1.0  # forced to sell
    assert slippage > 0.0  # and it costs something


def test_zero_maintenance_never_calls():
    ctx = _ctx(cols=("us_equity",), ret=-0.30)
    v = MarginAccount(schedule="reg_t")
    v.prepare(ctx)
    v._maintenance = np.zeros(1)
    assert v.check_margin(1.0, np.array([3.0]), 10) is None


def test_borrow_spread_widens_with_volatility():
    ctx = _ctx()
    vix = pd.Series(60.0, index=ctx.returns.index)
    calm = MarginAccount(schedule="box_spread")
    calm.prepare(ctx)
    stressed = MarginAccount(schedule="box_spread")
    stressed.prepare(VehicleContext(returns=ctx.returns, rf=ctx.rf, vix=vix))
    assert stressed.borrow_spread(10) > calm.borrow_spread(10)


def test_maintenance_requirement_is_procyclical():
    """Brokers raise house requirements exactly when volatility spikes."""
    ctx = _ctx(cols=("us_equity",), ret=-0.12)
    calm = MarginAccount(schedule="portfolio_margin")
    calm.prepare(ctx)
    stressed = MarginAccount(schedule="portfolio_margin")
    stressed.prepare(
        VehicleContext(returns=ctx.returns, rf=ctx.rf,
                       vix=pd.Series(60.0, index=ctx.returns.index))
    )
    calm_action = calm.check_margin(1.0, np.array([3.0]), 10)
    stressed_action = stressed.check_margin(1.0, np.array([3.0]), 10)
    # The stressed book is called at least as hard as the calm one.
    calm_scale = calm_action[0] if calm_action else 1.0
    stressed_scale = stressed_action[0] if stressed_action else 1.0
    assert stressed_scale <= calm_scale


def test_reg_t_caps_leverage_lower_than_portfolio_margin():
    regt = MarginAccount(schedule="reg_t")
    pm = MarginAccount(schedule="portfolio_margin")
    assert regt.max_leverage < pm.max_leverage
    assert regt.achievable_weights(np.array([2.0, 2.0]), 0).sum() == pytest.approx(2.0)


def test_box_spread_is_cheaper_than_broker_margin():
    ctx = _ctx()
    box, broker = MarginAccount(schedule="box_spread"), MarginAccount(schedule="portfolio_margin")
    box.prepare(ctx)
    broker.prepare(ctx)
    assert box.borrow_spread(10) < broker.borrow_spread(10)


def test_intraday_check_is_stricter_than_close_only():
    """Checking the close alone is a known source of fake survival."""
    ctx = _ctx(cols=("us_equity",), ret=-0.18)
    strict = MarginAccount(schedule="reg_t", intraday_stress=1.6)
    strict.prepare(ctx)
    lenient = MarginAccount(schedule="reg_t", intraday_stress=1.0)
    lenient.prepare(ctx)
    strict_action = strict.check_margin(1.0, np.array([3.0]), 10)
    lenient_action = lenient.check_margin(1.0, np.array([3.0]), 10)
    strict_scale = strict_action[0] if strict_action else 1.0
    lenient_scale = lenient_action[0] if lenient_action else 1.0
    assert strict_scale <= lenient_scale
