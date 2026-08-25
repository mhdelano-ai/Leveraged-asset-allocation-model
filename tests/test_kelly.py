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


def _only(first: float, *rest: float) -> np.ndarray:
    """A weight vector over the whole universe, padded with zeros."""
    w = np.zeros(len(ASSETS))
    for i, v in enumerate((first, *rest)):
        w[i] = v
    return w


def _panel(returns: np.ndarray, cash_rate: float = 0.0) -> KellyPanel:
    index = pd.date_range("2000-01-31", periods=len(returns), freq="ME")
    frame = pd.DataFrame(returns, columns=ASSETS, index=index)
    cash = pd.Series(cash_rate, index=index, name="cash")
    return KellyPanel(name="synthetic", returns=frame, cash=cash, tickers=dict.fromkeys(ASSETS, "X"))


def _lognormal_panel(mu: float, sigma: float, n: int = 4000, seed: int = 0) -> KellyPanel:
    """One live asset with known drift and vol; every other sleeve is flat.

    Flat rather than noisy on purpose: with several zero-premium noise sleeves the
    optimiser finds spurious diversification and the single-asset Kelly identity
    stops being the thing under test. The singular covariance this creates is
    handled by the solver, which falls back to search when the closed form fails.
    """
    rng = np.random.default_rng(seed)
    returns = np.zeros((n, len(ASSETS)))
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
    returns = np.tile(np.vstack([_only(0.02), _only(-0.01)]), (5, 1))
    panel = _panel(returns)
    w = _only(1.0)
    realised = portfolio_returns(w, panel.excess.to_numpy(), panel.cash.to_numpy())
    expected = 12 * np.mean(np.log1p(realised))
    assert growth_rate(w, panel.excess.to_numpy(), panel.cash.to_numpy()) == pytest.approx(expected)


def test_growth_rate_is_minus_infinity_when_a_month_wipes_the_account_out():
    returns = np.zeros((24, len(ASSETS)))
    returns[5, 0] = -0.30
    panel = _panel(returns)
    assert growth_rate(_only(4.0), panel.excess.to_numpy(), panel.cash.to_numpy()) == -np.inf


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
    result = simulate(panel, _only(1.0), financing_spread=0.05)
    assert len(result.calls) == 0
    assert not result.ruined
    # No debt, so the broker's spread must not touch the result.
    assert result.equity.iloc[-1] == pytest.approx(1.0004**500, rel=1e-6)


def test_leverage_costs_the_broker_spread_on_the_borrowed_part_only():
    panel = _daily_panel(np.zeros((252, len(ASSETS))))
    flat = simulate(panel, _only(1.0), target_leverage=1.0, financing_spread=0.02)
    levered = simulate(panel, _only(1.0), target_leverage=2.0, financing_spread=0.02)
    assert flat.equity.iloc[-1] == pytest.approx(1.0, rel=1e-9)
    # One turn of borrowed capital at 2% for a year.
    assert levered.equity.iloc[-1] == pytest.approx(1.0 - 0.02, abs=2e-3)


def test_a_crash_forces_a_sale_and_cuts_leverage():
    daily = np.zeros((60, len(ASSETS)))
    daily[30, 0] = -0.35
    panel = _daily_panel(daily)
    result = simulate(panel, _only(1.0), target_leverage=2.0)
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
    result = simulate(panel, _only(1.0), target_leverage=1.0)
    assert len(result.calls) == 0
    assert result.equity.iloc[-1] == pytest.approx(0.65, rel=1e-6)


def test_a_move_larger_than_the_equity_cushion_is_ruin():
    daily = np.zeros((30, len(ASSETS)))
    daily[10, 0] = -0.60
    panel = _daily_panel(daily)
    result = simulate(panel, _only(1.0), target_leverage=2.0)
    assert result.ruined
    assert result.ruin_date == panel.daily.index[10]
    assert result.equity.iloc[-1] == 0.0


def test_declining_to_relever_leaves_the_account_unlevered_afterwards():
    daily = np.zeros((120, len(ASSETS)))
    daily[30, 0] = -0.35
    panel = _daily_panel(daily)
    result = simulate(
        panel, _only(1.0), target_leverage=2.0, relever_after_call=False
    )
    assert result.leverage.iloc[-1] == pytest.approx(1.0, abs=1e-6)


def test_maintenance_covers_every_asset_in_the_universe():
    assert set(MAINTENANCE) == set(ASSETS)


def test_ladder_is_monotone_in_risk_when_nothing_is_ever_called():
    rng = np.random.default_rng(1)
    daily = np.zeros((2000, len(ASSETS)))
    daily[:, 0] = rng.normal(0.0004, 0.006, size=2000)
    panel = _daily_panel(daily)
    table = leverage_ladder(panel, _only(1.0), levels=(1.0, 1.5, 2.0))
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
            _only(1.0),
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
    result = simulate(panel, _only(0.5), financing_spread=0.02)
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
            "long_treasuries": 0.0458,
            "reits": 0.0387,
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


def test_implied_eps_growth_is_aggregate_growth_plus_buyback():
    # The build-up's growth term is aggregate; adding the share-count term gives
    # the per-share figure that a historical EPS growth rate is comparable with.
    cma = fwd.build_cma(_forward_panel(), inputs=_market_inputs())
    a = fwd.Assumptions()
    assert cma.implied_eps_growth["us_equity"] == pytest.approx(
        a.us_real_growth + a.us_buyback
    )
    assert cma.implied_eps_growth["us_bonds"] == pytest.approx(0.0)


def test_growth_needed_for_inverts_the_build_up():
    cma = fwd.build_cma(_forward_panel(), inputs=_market_inputs())
    target = 0.08
    need = fwd.growth_needed_for(cma, target)
    row = cma.build_up.loc["us_equity"]
    rebuilt = row["income"] + row["inflation"] + row["valuation"] + row["currency"] + need
    assert rebuilt == pytest.approx(target)


def test_presets_are_coherent_growth_positions():
    # Each preset states aggregate growth and buyback separately; the per-share
    # figure they imply is the thing that can be compared with history.
    implied = {
        name: preset_growth
        for name, preset_growth in (
            (name, p.us_real_growth + p.us_buyback) for name, p in fwd.PRESETS.items()
        )
    }
    assert implied["historical"] == pytest.approx(0.0168)
    assert implied["base"] == pytest.approx(0.0280)
    assert implied["modern"] == pytest.approx(0.0386)
    assert implied["historical"] < implied["base"] < implied["modern"]


def test_more_assumed_growth_means_more_kelly_leverage():
    inputs = _market_inputs()
    panel = _forward_panel()
    leverage = {}
    for name, preset in fwd.PRESETS.items():
        cma = fwd.build_cma(panel, inputs=inputs, assumptions=preset)
        w = fwd.bundle_kelly(cma, financing_spread=0.012)
        leverage[name] = w["us_equity"] + w["intl_equity"]
    assert leverage["historical"] < leverage["base"] < leverage["modern"]


@pytest.mark.network
def test_growth_decomposition_is_additive_by_construction():
    # Deflating profits by the GDP deflator makes real profit growth identically
    # real GDP growth plus profit-share drift.
    parts = fwd.decompose_growth(eras=(("1985", "2023"),))
    row = parts.loc["1985-2023"]
    compounded = (1 + row["real_gdp"]) * (1 + row["profit_share_drift"]) - 1
    assert row["aggregate_real_profits"] == pytest.approx(compounded, abs=0.002)
    assert row["dilution"] == pytest.approx(
        row["aggregate_real_profits"] - row["real_eps_per_share"]
    )


def test_geometric_frontier_peaks_at_full_kelly():
    # f = 1 is the Kelly objective, so the frontier's maximum growth must sit there.
    cma = fwd.build_cma(_forward_panel(), inputs=_market_inputs())
    frontier = fwd.geometric_frontier(
        cma, fractions=np.round(np.arange(0.2, 1.61, 0.1), 3), financing_spread=0.012
    )
    assert frontier["geometric"].idxmax() == pytest.approx(1.0, abs=0.11)
    assert frontier["vol"].is_monotonic_increasing


def test_slope_and_scaling_agree_above_one_times_and_differ_below():
    cma = fwd.build_cma(_forward_panel(), inputs=_market_inputs())
    fractions = np.round(np.arange(0.2, 1.31, 0.1), 3)
    slope = fwd.geometric_frontier(cma, fractions=fractions, financing_spread=0.012)
    scaled = fwd.scaled_frontier(cma, fractions=fractions, financing_spread=0.012)
    gain = slope["geometric"] - scaled["geometric"]
    # Re-optimising can never do worse than diluting, and above the financing
    # kink two-fund separation holds so the two constructions coincide. The
    # tolerance is SLSQP convergence noise: 1e-7 is a hundred-thousandth of a
    # basis point of growth.
    assert (gain > -1e-7).all()
    levered = slope["gross"] > 1.001
    assert np.allclose(gain[levered].to_numpy(), 0.0, atol=1e-5)
    assert gain[~levered].max() > 0.001


def test_a_sleeve_is_levered_only_if_it_out_earns_the_borrowing_spread():
    """The financing kink sorts the universe, and the rule is exactly one number.

    Below 1x gross nothing is borrowed, so every sleeve competes against the bill
    rate. Above it, a sleeve has to clear the *spread* to be worth holding with
    borrowed money -- which is why the aggregate bond sleeves drop out at the kink
    while long Treasuries and REITs, whose premia are several times the spread,
    do not.
    """
    spread = 0.012
    cma = fwd.build_cma(_forward_panel(), inputs=_market_inputs())
    frontier = fwd.geometric_frontier(
        cma,
        fractions=np.round(np.arange(0.2, 1.31, 0.1), 3),
        financing_spread=spread,
        equity_split=fwd.GLOBAL_EQUITY_WEIGHTS,
    )
    levered = frontier["gross"] > 1.001
    for asset in ASSETS:
        if cma.excess[asset] < spread:
            assert frontier.loc[levered, asset].max() < 1e-6, asset

    # And below the kink the book is not all equity: something that yields more
    # than a bill earns a place.
    unlevered = frontier["gross"] <= 1.001
    diversifiers = frontier[[a for a in ASSETS if not a.endswith("equity")]].sum(axis=1)
    assert diversifiers[unlevered].max() > 0.05


def test_frontier_respects_the_cap_weighted_equity_split():
    cma = fwd.build_cma(_forward_panel(), inputs=_market_inputs())
    frontier = fwd.geometric_frontier(
        cma, fractions=np.array([0.4, 1.0]), equity_split=(0.6, 0.4)
    )
    for _, row in frontier.iterrows():
        equity = row["us_equity"] + row["intl_equity"]
        if equity > 1e-6:
            assert row["us_equity"] / equity == pytest.approx(0.6, abs=1e-6)
