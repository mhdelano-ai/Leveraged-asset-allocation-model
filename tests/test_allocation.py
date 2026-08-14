"""The allocation report must read the backtest, never re-derive it."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lam.data.panels import SLEEVE_COSTS, SLEEVE_TICKERS, Panel
from lam.engine.backtest import EngineConfig, run
from lam.report import allocation


def _panel(n=600, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2000-01-03", periods=n)
    cols = ["us_equity", "interm_ust", "gold"]
    rets = pd.DataFrame(rng.normal(0.0003, 0.008, (n, 3)), index=idx, columns=cols)
    return Panel(
        name="T",
        returns=rets,
        benchmark=rets["us_equity"].rename("bench"),
        rf=pd.Series(0.02, index=idx),
        eligible=pd.DataFrame(True, index=idx, columns=cols),
        description="synthetic",
    )


class _Out:
    """Minimal stand-in for pipeline.RunOutput."""

    def __init__(self, result, params):
        self.result = result
        self.params = params


def _run(panel, leverage=1.0, **cfg):
    from lam.alloc.stack import StackParams

    base = pd.DataFrame(1 / 3, index=panel.returns.index, columns=panel.returns.columns)
    lev = pd.Series(leverage, index=panel.returns.index)
    config = EngineConfig(default_cost=0.0, leverage_ratchet_up=10.0, **cfg)
    result = run(panel.returns, base, lev, rf_annual=panel.rf, config=config)
    return _Out(result, StackParams(max_leverage=max(leverage, 1.0)))


# --------------------------------------------------------------------------

def test_every_sleeve_in_every_panel_has_a_ticker():
    """Adding a sleeve without a ticker should fail here, not in a report."""
    missing = set(SLEEVE_COSTS) - set(SLEEVE_TICKERS)
    assert not missing, f"sleeves without a ticker: {sorted(missing)}"


def test_reported_weights_are_the_backtest_weights():
    panel = _panel()
    out = _run(panel)
    when = panel.returns.index[400]
    frame = allocation.current_allocation(out, panel, date=when)
    expected = out.result.weights.loc[when]
    for sleeve in panel.returns.columns:
        assert frame.loc[sleeve, "held"] == pytest.approx(float(expected[sleeve]), abs=1e-15)


def test_shares_of_book_sum_to_one():
    panel = _panel()
    out = _run(panel, leverage=1.5)
    frame = allocation.current_allocation(out, panel)
    assert frame["share_of_book"].sum() == pytest.approx(1.0, abs=1e-9)


def test_shares_are_zero_when_the_book_is_flat():
    panel = _panel()
    out = _run(panel, leverage=0.0)
    frame = allocation.current_allocation(out, panel)
    assert frame["share_of_book"].sum() == pytest.approx(0.0, abs=1e-12)
    assert frame.attrs["cash"] == pytest.approx(1.0, abs=1e-9)


def test_gross_and_cash_are_complementary():
    panel = _panel()
    out = _run(panel, leverage=1.5)
    frame = allocation.current_allocation(out, panel)
    assert frame.attrs["gross"] + frame.attrs["cash"] == pytest.approx(1.0, abs=1e-12)


def test_leverage_profile_on_a_constant_book():
    panel = _panel()
    out = _run(panel, leverage=2.0, lev_band=0.001, rebalance_freq=None)
    prof = allocation.leverage_profile(out)
    assert prof["avg_all"] == pytest.approx(2.0, abs=0.02)
    assert prof["median"] == pytest.approx(2.0, abs=0.02)
    assert prof["frac_above_1x"] > 0.99
    assert prof["avg_cash"] == pytest.approx(-1.0, abs=0.02)


def test_leverage_profile_flags_an_unlevered_book():
    panel = _panel()
    out = _run(panel, leverage=0.5)
    prof = allocation.leverage_profile(out)
    assert prof["frac_above_1x"] == pytest.approx(0.0, abs=1e-9)
    assert prof["frac_above_2x"] == pytest.approx(0.0, abs=1e-9)
    assert 0.0 < prof["avg_all"] < 1.0


def test_risk_book_shares_sum_to_one_and_are_sorted():
    panel = _panel()
    out = _run(panel, leverage=1.2)
    shares = allocation.risk_book_shares(out)
    assert shares.sum() == pytest.approx(1.0, abs=1e-9)
    assert list(shares) == sorted(shares, reverse=True)


def test_needs_trade_is_false_when_held_equals_target():
    panel = _panel()
    out = _run(panel)
    # Force the two to coincide, as they do on any rebalance day.
    out.result.target_weights.iloc[-1] = out.result.weights.iloc[-1]
    fired, drift = allocation.needs_trade(out, panel)
    assert not fired
    assert drift == pytest.approx(0.0, abs=1e-12)


def test_needs_trade_fires_on_a_large_divergence():
    panel = _panel()
    out = _run(panel)
    held = out.result.weights.iloc[-1]
    out.result.target_weights.iloc[-1] = held + 0.5  # far outside any band
    fired, drift = allocation.needs_trade(out, panel)
    assert fired
    assert drift == pytest.approx(1.5, abs=1e-9)


def test_target_weights_are_recorded_and_diverge_between_rebalances():
    """Held drifts away from the standing order; both must be available."""
    panel = _panel(800, seed=5)
    out = _run(panel, leverage=1.5, rebalance_freq="M", lev_band=0.5,
               abs_band=0.5, rel_band=0.5)
    gap = (out.result.weights - out.result.target_weights).abs().sum(axis=1)
    assert out.result.target_weights.shape == out.result.weights.shape
    assert gap.max() > 1e-6  # they genuinely differ somewhere


def test_allocation_history_decade_grouping():
    panel = _panel(3000, seed=9)
    out = _run(panel, leverage=1.0)
    hist = allocation.allocation_history(out, "decade")
    assert all(str(i).endswith("s") for i in hist.index)
    assert "GROSS" in hist.columns
    np.testing.assert_allclose(
        hist["GROSS"].to_numpy(),
        hist.drop(columns="GROSS").sum(axis=1).to_numpy(),
        rtol=1e-12,
    )


def test_describe_names_the_sleeves_and_reports_gross():
    panel = _panel()
    out = _run(panel, leverage=1.2)
    text = allocation.describe(out, panel)
    for sleeve in panel.returns.columns:
        assert sleeve in text
    assert "gross" in text and "cash" in text and "leverage" in text


def test_describe_flags_a_usually_unlevered_book():
    """The 'leveraged strategy that isn't' fact must be stated, not inferred."""
    panel = _panel()
    out = _run(panel, leverage=0.5)
    assert "usually de-levered" in allocation.describe(out, panel)


def test_panel_ticker_property_matches_column_order():
    panel = _panel()
    assert panel.tickers == [SLEEVE_TICKERS[c] for c in panel.returns.columns]
