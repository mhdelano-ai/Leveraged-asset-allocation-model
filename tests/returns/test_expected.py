"""Expected returns: the historical mean equals hist_mean, intervals use the
chosen long-run variance rule, nothing looks ahead, and the interval holds its
coverage on simulated serially correlated returns."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from scipy.signal import lfilter

from assetforecast.returns import config
from assetforecast.returns import expected as E


def _xr(T=240, seed=0, starts=(0, 0, 100, 200)):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("1990-01-31", periods=T, freq="ME")
    x = 0.004 + 0.04 * rng.standard_normal((T, len(starts)))
    for j, s in enumerate(starts):
        x[:s, j] = np.nan
    return pd.DataFrame(x, index=idx, columns=["us_large", "ust_int", "intl_dev", "tips"])


def test_mean_is_twelve_times_monthly_mean_and_eligibility():
    xr = _xr()
    t = xr.index[150]
    e = E.expected_from_xr(xr, t)
    # tips starts at row 200: not there; intl_dev has 51 months < 60: left out
    assert list(e.index) == ["us_large", "ust_int"]
    for a in e.index:
        assert e.at[a, "mean_12m"] == pytest.approx(12 * xr[a].loc[:t].mean(), rel=1e-12)
        assert e.at[a, "n_months"] == 151
    assert list(e.columns) == E.EXPECTED_COLUMNS
    assert e.at["us_large", "first_month"] == "1990-01"


def test_mean_equals_hist_mean_baseline():
    from assetforecast.returns.models.base import TrainingSet
    from assetforecast.returns.models.baselines import HistoricalMean
    xr = _xr()
    t = xr.index[-1]
    idx = pd.MultiIndex.from_arrays([pd.DatetimeIndex([]), pd.Index([], dtype=str)], names=["date", "asset"])
    train = TrainingSet(horizon=12, origin=t, xr=xr, assets=tuple(xr.columns), seed=0,
                        _features=pd.DataFrame(index=idx), _targets=pd.Series(index=idx, dtype=float))
    hm = HistoricalMean().fit(train)
    e = E.expected_from_xr(xr, t)
    for a in e.index:
        assert e.at[a, "mean_12m"] == pytest.approx(hm._mean[a], rel=1e-12)


def test_interval_is_student_t_with_ewc_df():
    xr = _xr()
    e = E.expected_from_xr(xr)
    for a in e.index:
        n = int(e.at[a, "n_months"])
        df = E.ewc_terms(n)
        half = stats.t.ppf(0.95, df) * e.at[a, "se_12m"]
        assert e.at[a, "ci90_hi"] - e.at[a, "mean_12m"] == pytest.approx(half, rel=1e-10)
        assert e.at[a, "mean_12m"] - e.at[a, "ci90_lo"] == pytest.approx(half, rel=1e-10)
    lo, hi = E._ci_names(0.8)
    e8 = E.expected_from_xr(xr, level=0.8)
    assert (lo, hi) == ("ci80_lo", "ci80_hi") and (e8[hi] - e8[lo] < e["ci90_hi"] - e["ci90_lo"]).all()


def test_ewc_rule():
    assert [E.ewc_terms(n) for n in (60, 357, 448, 622, 1204)] == [4, 7, 9, 12, 25]


def test_history_matches_single_origin_and_does_not_look_ahead():
    xr = _xr()
    h = E.history(xr)
    assert list(h.columns) == E.HISTORY_COLUMNS
    t = xr.index[180]
    trunc = E.history(xr.loc[:t])
    a = h[h.origin <= t].reset_index(drop=True)
    pd.testing.assert_frame_equal(a, trunc)
    one = E.expected_from_xr(xr, t)
    row = h[h.origin == t].set_index("asset")
    for c in E.EXPECTED_COLUMNS:
        assert (row.loc[one.index, c] == one[c]).all(), c
    # first row per asset at its 60th month
    first = h.groupby("asset").origin.min()
    assert first["intl_dev"] == xr.index[100 + config.MIN_ASSET_HISTORY - 1]


@pytest.mark.slow
def test_coverage_under_serial_correlation():
    """AR(1) 0.2 monthly returns, n = 357 (TIPS): the chosen interval keeps ~90%,
    the iid interval does not (the full Monte Carlo is in the module docstring)."""
    rng = np.random.default_rng(5)
    reps, n = 400, 357
    hit_ewc, hit_iid = [], []
    for _ in range(reps):
        x = lfilter([1], [1, -0.2], rng.standard_normal(n + 200))[200:]
        m, se, df, _ = E.mean_se(x)
        hit_ewc.append(abs(m) <= stats.t.ppf(0.95, df) * se)
        hit_iid.append(abs(m) <= 1.645 * x.std(ddof=1) / np.sqrt(n))
    assert 0.86 <= np.mean(hit_ewc) <= 0.95
    assert np.mean(hit_iid) < np.mean(hit_ewc) - 0.03


@pytest.mark.realdata
def test_real_panel_latest():
    from assetforecast.returns.inputs import excess_returns, load_return_panel
    panel = load_return_panel()
    e = E.expected_returns(panel)
    xr = excess_returns(panel)
    assert set(e.index) == set(config.ASSETS)
    assert e.at["tips", "first_month"] == "1997-02"
    np.testing.assert_allclose(e["mean_12m"], 12 * xr[e.index].mean().to_numpy(), rtol=1e-12)
    assert (e["ci90_lo"] < e["mean_12m"]).all() and (e["mean_12m"] < e["ci90_hi"]).all()
    # equal to the replication's hist_mean forecasts where they exist
    from assetforecast import paths
    p = paths.RESULTS_RETURNS / "forecasts" / "hist_mean" / "h12.parquet"
    if p.exists():
        f = pd.read_parquet(p)
        t = f.date.max()
        f = f[f.date == t].set_index("asset")["mean"]
        e2 = E.expected_returns(panel, t)
        np.testing.assert_allclose(e2.loc[f.index, "mean_12m"], f, rtol=1e-10)
