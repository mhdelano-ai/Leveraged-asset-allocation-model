"""Kelly solver tests. All synthetic -- nothing here touches the network."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lam.kelly import (
    ASSETS,
    KellyPanel,
    constrained_shrunk_kelly,
    empirical_kelly,
    gaussian_kelly,
    growth_rate,
    score,
    summarise,
)
from lam.kelly.solve import portfolio_returns


def _panel(returns: np.ndarray, cash_rate: float = 0.0) -> KellyPanel:
    index = pd.date_range("2000-01-31", periods=len(returns), freq="ME")
    frame = pd.DataFrame(returns, columns=ASSETS, index=index)
    cash = pd.Series(cash_rate, index=index, name="cash")
    return KellyPanel(name="synthetic", returns=frame, cash=cash, tickers=dict.fromkeys(ASSETS, "X"))


def _lognormal_panel(mu: float, sigma: float, n: int = 4000, seed: int = 0) -> KellyPanel:
    """One asset with known drift and vol; the rest are zero-premium noise.

    The others are noise rather than literal zeros so the sample covariance stays
    invertible -- a degenerate column is a different test, below.
    """
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0, 0.01, size=(n, len(ASSETS)))
    returns[:, 0] = rng.normal(mu / 12.0, sigma / np.sqrt(12.0), size=n)
    return _panel(returns)


def test_gaussian_kelly_is_sigma_inverse_mu():
    rng = np.random.default_rng(7)
    returns = rng.normal(0.005, 0.03, size=(600, len(ASSETS)))
    panel = _panel(returns)
    mu = panel.excess.mean().to_numpy() * 12
    sigma = panel.excess.cov().to_numpy() * 12
    assert np.allclose(gaussian_kelly(panel).to_numpy(), np.linalg.solve(sigma, mu))


def test_single_asset_kelly_approaches_mu_over_sigma_squared():
    # The continuous-time optimum is f* = mu / sigma^2. Monthly rebalancing and
    # sampling noise keep it close rather than exact.
    mu, sigma = 0.08, 0.20
    panel = _lognormal_panel(mu, sigma, n=6000, seed=3)
    weights = empirical_kelly(panel, financing_spread=0.0).weights
    assert weights[ASSETS[0]] == pytest.approx(mu / sigma**2, rel=0.15)


def test_growth_rate_matches_the_definition():
    returns = np.tile(np.array([[0.02, 0.0, 0.0, 0.0], [-0.01, 0.0, 0.0, 0.0]]), (5, 1))
    panel = _panel(returns)
    w = np.array([1.0, 0.0, 0.0, 0.0])
    realised = portfolio_returns(w, panel.excess.to_numpy(), panel.cash.to_numpy())
    expected = 12 * np.mean(np.log1p(realised))
    assert growth_rate(w, panel.excess.to_numpy(), panel.cash.to_numpy()) == pytest.approx(expected)


def test_growth_rate_is_minus_infinity_when_a_month_wipes_the_account_out():
    returns = np.zeros((24, len(ASSETS)))
    returns[5, 0] = -0.30
    panel = _panel(returns)
    assert growth_rate(np.array([4.0, 0, 0, 0]), panel.excess.to_numpy(), panel.cash.to_numpy()) == -np.inf


def test_no_leverage_mode_never_borrows_or_shorts():
    panel = _lognormal_panel(0.20, 0.15, n=1200, seed=11)
    weights = empirical_kelly(panel, mode="long_only_unlevered").weights
    assert (weights >= -1e-9).all()
    assert weights.sum() <= 1.0 + 1e-6


def test_financing_cost_reduces_the_optimal_leverage():
    panel = _lognormal_panel(0.12, 0.15, n=3000, seed=5)
    cheap = empirical_kelly(panel, financing_spread=0.0).weights.sum()
    dear = empirical_kelly(panel, financing_spread=0.05).weights.sum()
    assert dear < cheap


def test_fraction_scales_the_full_kelly_weights():
    panel = _lognormal_panel(0.10, 0.18, n=1500, seed=2)
    full = empirical_kelly(panel).weights
    half = empirical_kelly(panel, fraction=0.5).weights
    assert np.allclose(half.to_numpy(), full.to_numpy() * 0.5)


def test_gross_cap_binds():
    panel = _lognormal_panel(0.25, 0.15, n=1500, seed=9)
    weights = empirical_kelly(panel, mode="long_only", max_gross=2.0).weights
    assert weights.abs().sum() <= 2.0 + 1e-6


def test_shrunk_kelly_pulls_a_negative_estimated_premium_back_toward_zero():
    # Asset 2 has a genuinely negative sample mean; unshrunk Kelly shorts it,
    # the equal-Sharpe prior should not.
    rng = np.random.default_rng(4)
    returns = rng.normal(0.006, 0.03, size=(400, len(ASSETS)))
    returns[:, 1] -= 0.010
    panel = _panel(returns)
    raw = gaussian_kelly(panel)
    shrunk = constrained_shrunk_kelly(panel, mode="long_only", mu_shrink=1.0)
    assert raw[ASSETS[1]] < 0
    assert shrunk[ASSETS[1]] >= -1e-9


def test_summarise_reports_ruin_as_a_date_not_a_return():
    index = pd.date_range("2000-01-31", periods=10, freq="ME")
    returns = pd.Series([0.01] * 4 + [-1.5] + [0.01] * 5, index=index)
    out = summarise(returns)
    assert out["ruined"]
    assert out["ruin_date"] == index[4]
    assert out["log_growth"] == -np.inf
    assert np.isnan(out["cagr"])


def test_score_reports_drawdown_of_a_flat_book_as_zero():
    panel = _panel(np.zeros((36, len(ASSETS))), cash_rate=0.002)
    out = score(np.zeros(len(ASSETS)), panel)
    assert out["max_drawdown"] == pytest.approx(0.0)
    assert out["growth"] == pytest.approx(12 * np.log1p(0.002))


# --- the Reg-T account simulation ------------------------------------------

from lam.kelly.account import MAINTENANCE, leverage_ladder, simulate  # noqa: E402


def _daily_panel(daily: np.ndarray, cash_rate: float = 0.0) -> KellyPanel:
    index = pd.date_range("2000-01-03", periods=len(daily), freq="B")
    frame = pd.DataFrame(daily, columns=ASSETS, index=index)
    cash = pd.Series(cash_rate, index=index, name="cash")
    monthly = frame.resample("ME").apply(lambda c: (1 + c).prod() - 1)
    return KellyPanel(
        name="synthetic",
        returns=monthly,
        cash=cash.resample("ME").apply(lambda c: (1 + c).prod() - 1),
        tickers=dict.fromkeys(ASSETS, "X"),
        daily=frame,
        daily_cash=cash,
    )


def test_unlevered_account_compounds_the_holding_and_is_never_called():
    daily = np.zeros((500, len(ASSETS)))
    daily[:, 0] = 0.0004
    panel = _daily_panel(daily)
    result = simulate(panel, np.array([1.0, 0, 0, 0]), financing_spread=0.05)
    assert len(result.calls) == 0
    assert not result.ruined
    # No debt, so the broker's spread must not touch the result.
    assert result.equity.iloc[-1] == pytest.approx(1.0004**500, rel=1e-6)


def test_leverage_costs_the_broker_spread_on_the_borrowed_part_only():
    panel = _daily_panel(np.zeros((252, len(ASSETS))))
    flat = simulate(panel, np.array([1.0, 0, 0, 0]), target_leverage=1.0, financing_spread=0.02)
    levered = simulate(panel, np.array([1.0, 0, 0, 0]), target_leverage=2.0, financing_spread=0.02)
    assert flat.equity.iloc[-1] == pytest.approx(1.0, rel=1e-9)
    # One turn of borrowed capital at 2% for a year.
    assert levered.equity.iloc[-1] == pytest.approx(1.0 - 0.02, abs=2e-3)


def test_a_crash_forces_a_sale_and_cuts_leverage():
    daily = np.zeros((60, len(ASSETS)))
    daily[30, 0] = -0.35
    panel = _daily_panel(daily)
    result = simulate(panel, np.array([1.0, 0, 0, 0]), target_leverage=2.0)
    assert len(result.calls) == 1
    assert result.calls[0] == panel.daily.index[30]
    # The crash alone takes leverage to 1.30/0.30 = 4.33x. The broker sells only
    # enough to meet the requirement, so the account is left levered ~2.3x -- not
    # back at its 2x target, and not deleveraged to safety.
    assert 2.0 < result.leverage.iloc[30] < 4.0


def test_no_call_when_the_same_crash_is_held_unlevered():
    daily = np.zeros((60, len(ASSETS)))
    daily[30, 0] = -0.35
    panel = _daily_panel(daily)
    result = simulate(panel, np.array([1.0, 0, 0, 0]), target_leverage=1.0)
    assert len(result.calls) == 0
    assert result.equity.iloc[-1] == pytest.approx(0.65, rel=1e-6)


def test_a_move_larger_than_the_equity_cushion_is_ruin():
    daily = np.zeros((30, len(ASSETS)))
    daily[10, 0] = -0.60
    panel = _daily_panel(daily)
    result = simulate(panel, np.array([1.0, 0, 0, 0]), target_leverage=2.0)
    assert result.ruined
    assert result.ruin_date == panel.daily.index[10]
    assert result.equity.iloc[-1] == 0.0


def test_declining_to_relever_leaves_the_account_unlevered_afterwards():
    daily = np.zeros((120, len(ASSETS)))
    daily[30, 0] = -0.35
    panel = _daily_panel(daily)
    result = simulate(
        panel, np.array([1.0, 0, 0, 0]), target_leverage=2.0, relever_after_call=False
    )
    assert result.leverage.iloc[-1] == pytest.approx(1.0, abs=1e-6)


def test_maintenance_covers_every_asset_in_the_universe():
    assert set(MAINTENANCE) == set(ASSETS)


def test_ladder_is_monotone_in_risk_when_nothing_is_ever_called():
    rng = np.random.default_rng(1)
    daily = np.zeros((2000, len(ASSETS)))
    daily[:, 0] = rng.normal(0.0004, 0.006, size=2000)
    panel = _daily_panel(daily)
    table = leverage_ladder(panel, np.array([1.0, 0, 0, 0]), levels=(1.0, 1.5, 2.0))
    assert (table["margin_calls"] == 0).all()
    assert table["vol"].is_monotonic_increasing
    assert table["max_drawdown"].is_monotonic_decreasing


def test_call_threshold_matches_the_textbook_reg_t_case():
    from lam.kelly.account import call_threshold

    # 2x against a 25% requirement: equity/value hits 25% after a 33.3% fall.
    assert call_threshold(2.0, maintenance=0.25) == pytest.approx(1 / 3, rel=1e-9)
    # 1x is never called, whatever the requirement.
    assert call_threshold(1.0, maintenance=0.5) == 1.0
    # Stress and procyclicality can only bring the call forward.
    assert call_threshold(2.0, intraday_stress=1.35) < call_threshold(2.0)
    assert call_threshold(2.0, intraday_stress=1.35, procyclicality=0.5) < call_threshold(
        2.0, intraday_stress=1.35
    )


def test_call_threshold_agrees_with_the_simulation():
    from lam.kelly.account import call_threshold

    edge = call_threshold(2.0, maintenance=0.25, intraday_stress=1.0)
    for delta, expect_call in ((-0.01, False), (+0.01, True)):
        daily = np.zeros((40, len(ASSETS)))
        daily[20, 0] = -(edge + delta)
        panel = _daily_panel(daily)
        result = simulate(
            panel,
            np.array([1.0, 0, 0, 0]),
            target_leverage=2.0,
            intraday_stress=1.0,
            procyclicality=0.0,
        )
        assert (len(result.calls) > 0) is expect_call


def test_uninvested_cash_earns_the_bill_rate():
    # A book held at 0.5x gross is half in cash; over a year that half must earn
    # the bill rate, not zero.
    # The synthetic panel has 252 rows for a year, so quote the rate per row --
    # a real panel carries the calendar day count inside daily_cash instead.
    daily_rate = 1.05 ** (1 / 252) - 1
    panel = _daily_panel(np.zeros((252, len(ASSETS))), cash_rate=daily_rate)
    result = simulate(panel, np.array([0.5, 0, 0, 0]), financing_spread=0.02)
    assert result.equity.iloc[-1] == pytest.approx(1.0 + 0.5 * 0.05, abs=3e-3)


# --- forward-looking assumptions -------------------------------------------

from lam.kelly import forward as fwd  # noqa: E402


def _market_inputs() -> fwd.MarketInputs:
    return fwd.MarketInputs(
        asof=pd.Timestamp("2026-08-20"),
        cash=0.038,
        treasury={"3m": 0.038, "5y": 0.0439, "10y": 0.0469, "30y": 0.0523},
        breakeven_inflation=0.0234,
        distribution_yield={
            "us_equity": 0.0117,
            "intl_equity": 0.0290,
            "us_bonds": 0.0403,
            "intl_bonds": 0.0457,
        },
        foreign_long={"euro": 0.0305, "japan": 0.0265, "uk": 0.0494},
        foreign_short={"euro": 0.0223, "japan": 0.0124, "uk": 0.0375},
    )


def _forward_panel() -> KellyPanel:
    rng = np.random.default_rng(12)
    daily = rng.normal(0.0003, 0.009, size=(1500, len(ASSETS)))
    daily[:, 2] *= 0.35
    daily[:, 3] *= 0.30
    return _daily_panel(daily, cash_rate=0.038 / 252)


def test_hedged_foreign_yield_adds_the_short_rate_differential():
    mi = _market_inputs()
    # Japan: a 2.65% JGB plus a 3.80% - 1.24% carry is 5.21% to a dollar investor.
    japan_leg = mi.foreign_long["japan"] + (mi.cash - mi.foreign_short["japan"])
    assert japan_leg == pytest.approx(0.0521, abs=1e-4)
    # And the blend must sit between the cheapest and dearest block.
    legs = [
        mi.foreign_long[k] + (mi.cash - mi.foreign_short[k]) for k in fwd.FOREIGN_BLOCKS
    ]
    assert min(legs) < mi.hedged_foreign_ytm() < max(legs)


def test_us_agg_yield_interpolates_the_curve_and_adds_the_index_spread():
    mi = _market_inputs()
    belly = np.interp(6.0, [0.25, 5.0, 10.0, 30.0], [0.038, 0.0439, 0.0469, 0.0523])
    assert mi.us_agg_ytm() == pytest.approx(belly + fwd.US_AGG_SPREAD)


def test_arithmetic_expected_return_is_geometric_plus_half_variance():
    cma = fwd.build_cma(_forward_panel(), inputs=_market_inputs())
    expected = cma.geometric + 0.5 * cma.vol**2
    assert np.allclose(cma.arithmetic.to_numpy(), expected.to_numpy())


def test_equity_build_up_sums_its_stated_components():
    cma = fwd.build_cma(_forward_panel(), inputs=_market_inputs())
    a = fwd.Assumptions()
    expected = 0.0117 + a.us_buyback + a.us_real_growth + 0.0234
    assert cma.geometric["us_equity"] == pytest.approx(expected)


def test_bond_expected_return_is_the_yield_less_credit_loss():
    mi = _market_inputs()
    cma = fwd.build_cma(_forward_panel(), inputs=mi)
    assert cma.geometric["us_bonds"] == pytest.approx(
        mi.us_agg_ytm() - fwd.Assumptions().us_bond_credit_loss
    )


def test_forward_kelly_without_financing_is_the_closed_form():
    cma = fwd.build_cma(_forward_panel(), inputs=_market_inputs())
    expected = np.linalg.solve(cma.cov.to_numpy(), cma.excess.to_numpy())
    assert np.allclose(fwd.kelly(cma).to_numpy(), expected)


def test_financing_cost_shrinks_forward_kelly():
    cma = fwd.build_cma(_forward_panel(), inputs=_market_inputs())
    assert fwd.kelly(cma, financing_spread=0.012).abs().sum() < fwd.kelly(cma).abs().sum()


def test_bundle_kelly_holds_the_split_it_was_given():
    cma = fwd.build_cma(_forward_panel(), inputs=_market_inputs())
    w = fwd.bundle_kelly(cma, equity_split=(0.6, 0.4), financing_spread=0.012)
    equity = w["us_equity"] + w["intl_equity"]
    if equity > 1e-6:
        assert w["us_equity"] / equity == pytest.approx(0.6, abs=1e-6)


def test_shifting_the_equity_premium_leaves_bonds_alone():
    cma = fwd.build_cma(_forward_panel(), inputs=_market_inputs())
    shifted = fwd.with_equity_premium(cma, 0.01)
    assert shifted.geometric["us_equity"] == pytest.approx(cma.geometric["us_equity"] + 0.01)
    assert shifted.geometric["us_bonds"] == pytest.approx(cma.geometric["us_bonds"])


def test_bootstrap_paths_carry_the_forward_drift_not_the_historical_one():
    panel = _forward_panel()
    cma = fwd.build_cma(panel, inputs=_market_inputs())
    sims = fwd.bootstrap_paths(panel, cma, years=5, paths=40, seed=1, window_years=5)
    assert len(sims) == 40
    assert len(sims[0].daily) == 5 * 252
    drift = np.mean([sim.daily.mean().to_numpy() * 252 for sim in sims], axis=0)
    assert np.allclose(drift, cma.arithmetic.to_numpy(), atol=0.02)
    vol = np.mean([sim.daily.std(ddof=1).to_numpy() * np.sqrt(252) for sim in sims], axis=0)
    assert np.allclose(vol, cma.vol.to_numpy(), rtol=0.15)
