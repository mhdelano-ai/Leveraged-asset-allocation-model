"""Tests for the return-maximising concentrated model.

Three things are worth testing here that the constrained model's suite does not
cover: the growth mathematics against a case with a known closed-form answer, the
leverage ratchet actually reaching its target, and the regime layer being causal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lam.alloc.growth import GrowthParams, build as build_growth, growth_gate
from lam.alloc.regime import blended_vol, stress_multiplier, trailing_percentile
from lam.alloc.trend import prices_from_returns
from lam.engine.backtest import EngineConfig, run
from lam.metrics import growth as mg


def _gbm(n=6000, mu=0.10, sigma=0.20, seed=0):
    """Daily lognormal returns with known drift and volatility."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("1990-01-01", periods=n)
    dt = 1.0 / 252.0
    logs = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * rng.normal(size=n)
    return pd.Series(np.expm1(logs), index=idx, name="asset")


# --------------------------------------------------------------------------
# Growth mathematics
# --------------------------------------------------------------------------

def test_unit_leverage_with_no_spread_reproduces_the_asset():
    asset = _gbm(500)
    rf = pd.Series(0.03, index=asset.index)
    out = mg.constant_leverage_returns(asset, rf, 1.0, borrow_spread=0.0, lend_spread=0.0)
    # At L=1 there is no cash, no borrowing and nothing to rebalance.
    np.testing.assert_allclose(out.to_numpy(), asset.to_numpy(), rtol=0, atol=1e-15)


def test_zero_leverage_earns_exactly_the_cash_rate():
    asset = _gbm(300)
    rf = pd.Series(0.05, index=asset.index)
    out = mg.constant_leverage_returns(asset, rf, 0.0, lend_spread=0.0, cost_per_unit=0.0)
    days = np.diff(asset.index.to_numpy().astype("datetime64[s]").astype("int64")) / 86_400.0
    np.testing.assert_allclose(out.to_numpy()[1:], 0.05 * days / 360.0, rtol=1e-12, atol=1e-15)


def test_kelly_optimum_matches_theory_on_lognormal_returns():
    """For GBM with no costs, ``L* = mu_excess / sigma^2`` exactly.

    This is the one case where the answer is known in closed form, so it is the
    only place the estimator can be checked rather than merely compared.

    The sample has to be enormous -- 120,000 days is about 460 years -- and that
    is the point rather than an inconvenience. At 20% volatility the standard
    error of an annual mean return over 40 years is 3.2%, so a Kelly estimate
    from a normal-length history carries a standard error of nearly a full turn
    of leverage. Forty years of real data cannot pin this down; only a synthetic
    sample can, which is exactly what ``kelly_bootstrap`` reports on real series.
    """
    mu, sigma, rf = 0.12, 0.20, 0.02
    asset = _gbm(120_000, mu=mu, sigma=sigma, seed=7)
    rate = pd.Series(rf, index=asset.index)

    theory = (mu - rf) / sigma**2  # = 2.5
    analytic = mg.kelly_analytic(asset, rate, spread=0.0)
    empirical, _ = mg.optimal_static_leverage(
        asset, rate, search=np.arange(0.5, 5.0, 0.05),
        borrow_spread=0.0, lend_spread=0.0, cost_per_unit=0.0,
    )
    assert analytic == pytest.approx(theory, rel=0.15)
    assert empirical == pytest.approx(theory, rel=0.15)


def test_growth_is_concave_in_leverage_and_turns_down_past_the_optimum():
    """The central fact of the whole exercise: more leverage stops paying."""
    asset = _gbm(8000, mu=0.12, sigma=0.20, seed=3)
    rf = pd.Series(0.02, index=asset.index)
    curve = mg.leverage_curve(
        asset, rf, levels=np.arange(0.5, 6.01, 0.5),
        borrow_spread=0.0, lend_spread=0.0, cost_per_unit=0.0,
    )
    best = curve["cagr"].idxmax()
    assert 1.0 < best < 5.0
    # Strictly increasing below the peak, strictly decreasing above it.
    below = curve.loc[:best, "cagr"]
    above = curve.loc[best:, "cagr"]
    assert below.is_monotonic_increasing
    assert above.is_monotonic_decreasing
    # Volatility, unlike growth, rises without limit.
    assert curve["vol"].is_monotonic_increasing


def test_ruin_is_absorbing():
    """A wipeout must end the series, not be followed by a recovery."""
    idx = pd.bdate_range("2000-01-03", periods=100)
    asset = pd.Series(0.001, index=idx)
    asset.iloc[50] = -0.40  # 3x this is -120%
    rf = pd.Series(0.0, index=idx)

    out = mg.constant_leverage_returns(asset, rf, 3.0, borrow_spread=0.0, cost_per_unit=0.0)
    assert out.iloc[50] == pytest.approx(-1.0)
    assert (out.iloc[51:] == 0.0).all()
    assert float(np.prod(1.0 + out.to_numpy())) == pytest.approx(0.0)


def test_leverage_curve_flags_ruin():
    idx = pd.bdate_range("2000-01-03", periods=200)
    asset = pd.Series(0.0005, index=idx)
    asset.iloc[100] = -0.30
    rf = pd.Series(0.0, index=idx)
    curve = mg.leverage_curve(asset, rf, levels=np.array([2.0, 4.0]), cost_per_unit=0.0)
    assert not curve.loc[2.0, "ruined"]
    assert curve.loc[4.0, "ruined"]


def test_growth_decomposition_reconciles_with_the_realised_path():
    """The lognormal decomposition should explain nearly all of a GBM path."""
    asset = _gbm(8000, mu=0.10, sigma=0.18, seed=5)
    rf = pd.Series(0.02, index=asset.index)
    d = mg.growth_decomposition(asset, rf, 2.0, borrow_spread=0.0, cost_per_unit=0.0)
    # Residual is skew, kurtosis and reset path dependence -- small for GBM.
    assert abs(d["residual_bps"]) < 25.0
    assert d["variance_drag_bps"] < 0
    # Above 1x the book is a net borrower, so carry is a cost.
    assert d["carry_bps"] < 0
    below = mg.growth_decomposition(asset, rf, 0.5, borrow_spread=0.0, cost_per_unit=0.0)
    assert below["carry_bps"] > 0


# --------------------------------------------------------------------------
# Regime layer
# --------------------------------------------------------------------------

def test_trailing_percentile_is_causal():
    """A percentile must never see the future.

    Corrupting the tail of the series must leave every earlier ranking
    bit-identical, which is the only way to be sure the window is right-aligned.
    """
    rng = np.random.default_rng(1)
    idx = pd.bdate_range("1990-01-01", periods=3000)
    series = pd.Series(np.abs(rng.normal(0.2, 0.05, 3000)), index=idx)

    clean = trailing_percentile(series, window=500, min_periods=100)
    corrupted = series.copy()
    corrupted.iloc[2000:] = 99.0
    dirty = trailing_percentile(corrupted, window=500, min_periods=100)

    np.testing.assert_allclose(
        clean.iloc[:2000].to_numpy(), dirty.iloc[:2000].to_numpy(), rtol=0, atol=0
    )


def test_stress_multiplier_bounds_and_neutral_warmup():
    rng = np.random.default_rng(2)
    idx = pd.bdate_range("1990-01-01", periods=3000)
    vol = pd.Series(np.abs(rng.normal(0.2, 0.08, 3000)), index=idx)
    m = stress_multiplier(vol, stress_percentile=0.80, stress_cap=0.30, window=1260)

    assert m.min() >= 0.30 - 1e-12
    assert m.max() <= 1.0 + 1e-12
    # Before any ranking exists the layer stands neutral rather than cautious.
    assert (m.iloc[:100] == 1.0).all()
    # The top of the distribution must actually be cut.
    assert m.min() == pytest.approx(0.30, abs=0.05)


def test_blended_vol_floor_binds_after_a_calm_stretch():
    """A fast estimator alone bottoms out in suppressed-volatility melt-ups."""
    idx = pd.bdate_range("1990-01-01", periods=1600)
    rng = np.random.default_rng(4)
    r = pd.Series(rng.normal(0, 0.02, 1600), index=idx)
    r.iloc[1200:] = rng.normal(0, 0.0005, 400)  # abrupt calm

    fast = r.ewm(halflife=40, min_periods=20).std() * np.sqrt(252)
    blended = blended_vol(r, halflife=40.0)
    tail = slice(1500, None)
    assert blended.iloc[tail].mean() > fast.iloc[tail].mean() * 2


# --------------------------------------------------------------------------
# Trend gate
# --------------------------------------------------------------------------

def test_growth_gate_reaches_full_risk_in_a_steady_uptrend():
    """The gate must not tax an ordinary bull market.

    The constrained model's ``0.5 + 0.5 * score`` mapping averages ~0.7 through
    all weather, which for a compounding objective is a permanent haircut rather
    than a risk control.
    """
    idx = pd.bdate_range("1990-01-01", periods=1500)
    prices = pd.DataFrame({"a": np.exp(np.linspace(0, 1.5, 1500))}, index=idx)
    rf = pd.Series(0.02, index=idx)
    gate = growth_gate(prices, rf, fast_weight=0.30, floor=0.0)
    assert gate.iloc[300:].mean() == pytest.approx(1.0, abs=1e-9)


def test_growth_gate_goes_to_the_floor_in_a_sustained_downtrend():
    idx = pd.bdate_range("1990-01-01", periods=1500)
    prices = pd.DataFrame({"a": np.exp(np.linspace(0, -1.5, 1500))}, index=idx)
    rf = pd.Series(0.02, index=idx)
    assert growth_gate(prices, rf, floor=0.0).iloc[400:].mean() == pytest.approx(0.0, abs=1e-9)
    assert growth_gate(prices, rf, floor=0.25).iloc[400:].mean() == pytest.approx(0.25, abs=1e-9)


# --------------------------------------------------------------------------
# The ratchet
# --------------------------------------------------------------------------

def test_ratchet_limits_the_daily_rise_but_still_reaches_the_target():
    """Regression test for a deadlock between the ratchet and the rebalance band.

    The ratchet used to be anchored to the leverage currently *held*, so it could
    never propose a book more than one step away from the one in place -- and the
    band only trades once the gap exceeds ``lev_band``. With
    ``ratchet < lev_band`` the two locked each other out and leverage could only
    advance on calendar triggers, one step per month. A 3x target took two and a
    half years to reach instead of thirty days.
    """
    n = 400
    idx = pd.bdate_range("2000-01-03", periods=n)
    rets = pd.DataFrame(0.0, index=idx, columns=["a"])
    base = pd.DataFrame(1.0, index=idx, columns=["a"])
    lev = pd.Series(3.0, index=idx)
    cfg = EngineConfig(
        default_cost=0.0, cost_era_multipliers=(("2100-01-01", 0.0),),
        borrow_spread=0.0, rebalance_freq=None,
        leverage_ratchet_up=0.10, lev_band=0.25, max_leverage=3.0,
    )
    res = run(rets, base, lev, rf_annual=pd.Series(0.0, index=idx), config=cfg)

    # Reached in roughly 3.0 / 0.10 = 30 days, plus band slack.
    assert res.leverage.iloc[60:].min() == pytest.approx(3.0, abs=0.26)
    assert res.leverage.iloc[-1] == pytest.approx(3.0, abs=0.26)
    # And it climbed rather than jumping: no single day adds more than a step
    # plus the one band-width the book is allowed to lag by.
    assert res.leverage.diff().max() <= 0.10 + 0.25 + 1e-9


def test_ratchet_slows_re_levering_after_a_crash():
    """The ratchet's purpose: no instant re-lever into a bear-market rally."""
    n = 300
    idx = pd.bdate_range("2000-01-03", periods=n)
    rets = pd.DataFrame(0.0, index=idx, columns=["a"])
    base = pd.DataFrame(1.0, index=idx, columns=["a"])
    lev = pd.Series(3.0, index=idx)
    lev.iloc[100:120] = 0.0  # signal switches off, then straight back on
    cfg = EngineConfig(
        default_cost=0.0, cost_era_multipliers=(("2100-01-01", 0.0),),
        borrow_spread=0.0, rebalance_freq=None,
        leverage_ratchet_up=0.10, lev_band=0.25, max_leverage=3.0,
    )
    res = run(rets, base, lev, rf_annual=pd.Series(0.0, index=idx), config=cfg)

    assert res.leverage.iloc[121] < 0.5  # cannot snap back
    assert res.leverage.iloc[125] < 1.5
    assert res.leverage.iloc[-1] == pytest.approx(3.0, abs=0.26)


# --------------------------------------------------------------------------
# The growth stack
# --------------------------------------------------------------------------

def test_growth_signals_are_causal():
    """Corrupting the future must not move a single earlier leverage target."""
    asset = _gbm(3000, seed=11)
    frame = asset.to_frame("a")
    rf = pd.Series(0.03, index=asset.index)
    params = GrowthParams()

    clean = build_growth(frame, rf, params=params)
    corrupted = frame.copy()
    rng = np.random.default_rng(77)
    corrupted.iloc[2000:] = rng.normal(-0.03, 0.06, corrupted.iloc[2000:].shape)
    dirty = build_growth(corrupted, rf, params=params)

    np.testing.assert_allclose(
        clean.target_leverage.iloc[:2000].to_numpy(),
        dirty.target_leverage.iloc[:2000].to_numpy(),
        rtol=0,
        atol=0,
    )


def test_rescale_matches_a_full_rebuild():
    """The optimiser's fast path must be exactly the slow path."""
    asset = _gbm(2000, seed=13)
    frame = asset.to_frame("a")
    rf = pd.Series(0.03, index=asset.index)

    base = build_growth(frame, rf, params=GrowthParams(sigma_target=0.20, max_leverage=2.0))
    rescaled = base.rescale(0.35, 3.0)
    rebuilt = build_growth(frame, rf, params=GrowthParams(sigma_target=0.35, max_leverage=3.0))

    np.testing.assert_allclose(
        rescaled.target_leverage.to_numpy(),
        rebuilt.target_leverage.to_numpy(),
        rtol=1e-12,
        atol=1e-15,
    )


def test_higher_sigma_target_never_lowers_target_leverage():
    asset = _gbm(2000, seed=17)
    frame = asset.to_frame("a")
    rf = pd.Series(0.03, index=asset.index)
    low = build_growth(frame, rf, params=GrowthParams(sigma_target=0.15, max_leverage=4.0))
    high = build_growth(frame, rf, params=GrowthParams(sigma_target=0.30, max_leverage=4.0))
    assert (high.target_leverage >= low.target_leverage - 1e-12).all()


def test_trend_gate_cuts_exposure_in_a_crash():
    """End to end: a sustained decline must take the book toward cash."""
    rng = np.random.default_rng(41)
    idx = pd.bdate_range("1990-01-01", periods=1600)
    drift = np.concatenate([np.full(1200, 0.0006), np.full(400, -0.004)])
    frame = pd.DataFrame({"a": drift + rng.normal(0, 0.008, 1600)}, index=idx)
    rf = pd.Series(0.02, index=idx)

    signals = build_growth(frame, rf, params=GrowthParams(trend_floor=0.0))
    assert signals.target_leverage.iloc[1100:1200].mean() > 0.5
    assert signals.target_leverage.iloc[-100:].mean() < 0.05


def test_riskless_asset_hits_the_leverage_cap_rather_than_zero():
    """A zero volatility forecast means unbounded Kelly, not a flat book.

    Dividing by the forecast and filling NaN with zero collapses "no risk" and
    "risk not yet measurable" into the same answer, and the two are opposites.
    """
    idx = pd.bdate_range("1990-01-01", periods=900)
    frame = pd.DataFrame({"a": 0.0002}, index=idx)  # constant return, zero vol
    rf = pd.Series(0.0, index=idx)
    signals = build_growth(frame, rf, params=GrowthParams(max_leverage=3.0))
    # Warm-up stands flat; once measurable, a riskless asset is held at the cap.
    assert signals.target_leverage.iloc[0] == pytest.approx(0.0)
    assert signals.target_leverage.iloc[-1] == pytest.approx(3.0)


def test_prices_from_returns_round_trips():
    asset = _gbm(500, seed=19)
    prices = prices_from_returns(asset.to_frame("a"))
    np.testing.assert_allclose(
        prices["a"].pct_change().dropna().to_numpy(),
        asset.to_numpy()[1:],
        rtol=1e-12,
        atol=1e-15,
    )
