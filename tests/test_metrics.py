"""Known-answer tests for the metrics layer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lam.metrics.constraint import evaluate
from lam.metrics.core import cagr, calmar, worst_rolling
from lam.metrics.drawdown import (
    depth,
    drawdown_series,
    max_drawdown,
    rolling_max_drawdown,
    ulcer_index,
    underwater_fraction,
)


def _returns_from_levels(levels, start="2000-01-03"):
    idx = pd.bdate_range(start, periods=len(levels))
    prices = pd.Series(levels, index=idx, dtype=float)
    return prices.pct_change().dropna()


def test_max_drawdown_exact():
    # 120 -> 60 is exactly -50%, and the later rise to 150 must not affect it.
    r = _returns_from_levels([100, 120, 60, 90, 150])
    assert max_drawdown(r) == pytest.approx(-0.5, abs=1e-12)
    assert depth(r) == pytest.approx(0.5, abs=1e-12)


def test_monotone_series_has_no_drawdown():
    r = _returns_from_levels([100, 101, 102, 103, 110])
    assert max_drawdown(r) == pytest.approx(0.0, abs=1e-12)
    assert underwater_fraction(r) == pytest.approx(0.0)


def test_cagr_doubling_over_two_years():
    # 2000-01-01 -> 2002-01-01 is 731 days (2000 is a leap year); wealth doubles.
    idx = pd.DatetimeIndex([pd.Timestamp("2000-01-01"), pd.Timestamp("2002-01-01")])
    r = pd.Series([0.0, 1.0], index=idx)
    assert cagr(r) == pytest.approx(2 ** (365.25 / 731.0) - 1, rel=1e-9)


def test_cagr_of_exact_annual_doubling_is_one_hundred_percent():
    idx = pd.DatetimeIndex([pd.Timestamp("2001-01-01"), pd.Timestamp("2002-01-01")])
    r = pd.Series([0.0, 1.0], index=idx)  # 365 days
    assert cagr(r) == pytest.approx(2 ** (365.25 / 365.0) - 1, rel=1e-9)


def test_cagr_of_constant_zero_is_zero():
    idx = pd.bdate_range("2000-01-03", periods=500)
    assert cagr(pd.Series(0.0, index=idx)) == pytest.approx(0.0, abs=1e-12)


def test_drawdown_series_matches_manual_path():
    r = _returns_from_levels([100, 110, 88, 99, 121])
    dd = drawdown_series(r)
    # Wealth relative to the entry level: 1.10, 0.88, 0.99, 1.21; peaks 1.10 then 1.21.
    assert dd.iloc[0] == pytest.approx(0.0, abs=1e-12)
    assert dd.iloc[1] == pytest.approx(0.88 / 1.10 - 1.0, abs=1e-12)
    assert dd.iloc[3] == pytest.approx(0.0, abs=1e-12)


def test_ulcer_index_zero_for_monotone():
    r = _returns_from_levels([100, 101, 102, 103])
    assert ulcer_index(r) == pytest.approx(0.0, abs=1e-9)


def test_rolling_window_isolates_the_drawdown():
    # A single -40% crash surrounded by flat periods. Windows containing the
    # crash must see it; windows entirely outside must not.
    levels = [100.0] * 200 + [60.0] * 200 + [60.0] * 200
    r = _returns_from_levels(levels)
    roll = rolling_max_drawdown(r, window=100, step=1)
    assert roll.min() == pytest.approx(-0.4, abs=1e-9)
    # The final window sits entirely in the flat tail after the crash.
    assert roll[-1] == pytest.approx(0.0, abs=1e-12)


def test_rolling_window_resets_peak_at_window_start():
    # Steady decline: any 100-day window should show only that window's decline,
    # never the cumulative fall from the very start.
    levels = list(np.linspace(100, 50, 400))
    r = _returns_from_levels(levels)
    full = depth(r)
    roll = -rolling_max_drawdown(r, window=100, step=1)
    assert roll.max() < full  # window depth strictly shallower than full sample


def test_rolling_step_subsampling_is_conservative_subset():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("1990-01-01", periods=3000)
    r = pd.Series(rng.normal(0.0003, 0.01, len(idx)), index=idx)
    fine = rolling_max_drawdown(r, 756, step=1)
    coarse = rolling_max_drawdown(r, 756, step=21)
    # Subsampled windows are a subset, so they cannot be worse than the full set.
    assert coarse.min() >= fine.min() - 1e-12


def test_calmar_is_cagr_over_depth():
    r = _returns_from_levels([100, 120, 60, 90, 150])
    assert calmar(r) == pytest.approx(cagr(r) / depth(r), rel=1e-9)


def test_worst_rolling_finds_the_worst_stretch():
    r = _returns_from_levels([100, 100, 50, 100, 100])
    assert worst_rolling(r, window=1) == pytest.approx(-0.5, abs=1e-12)


def test_constraint_identical_series_is_exactly_neutral():
    rng = np.random.default_rng(7)
    idx = pd.bdate_range("1990-01-01", periods=2000)
    r = pd.Series(rng.normal(0.0003, 0.01, len(idx)), index=idx)
    res = evaluate(r, r.copy(), step=21)
    assert res.passed
    assert res.worst_excess == pytest.approx(0.0, abs=1e-12)
    assert res.n_violating_windows == 0


def test_constraint_detects_a_deeper_strategy():
    idx = pd.bdate_range("1990-01-01", periods=2000)
    bench = pd.Series(0.0002, index=idx)
    strat = bench.copy()
    strat.iloc[1000] = -0.35  # one deep loss the benchmark never has
    res = evaluate(strat, bench, step=21)
    assert not res.passed
    assert not res.full_sample_pass
    assert res.worst_excess > 0.3


def test_constraint_rolling_catches_non_contemporaneous_drawdown():
    """Full-sample can pass while rolling fails -- the case the rolling test exists for."""
    idx = pd.bdate_range("1990-01-01", periods=4000)
    bench = pd.Series(0.0002, index=idx)
    strat = bench.copy()
    bench.iloc[500] = -0.40  # benchmark crashes early
    strat.iloc[3000] = -0.30  # strategy crashes later, but less deeply
    res = evaluate(strat, bench, step=21)
    assert res.full_sample_pass  # 30% < 40%
    assert not res.rolling_pass  # but not in the same window
    assert not res.passed
