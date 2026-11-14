import numpy as np
import pandas as pd
import pytest
from dataclasses import replace

from assetforecast.data.synthetic import make_synthetic_panel
from assetforecast.eval.targets import realized_kday_cov
from assetforecast.models import get_model, list_models
from assetforecast.models.bench import (CORR_MODELS, VOL_MODELS, Moments, block_target, ew_corr, ew_mean,
                                   ew_series, mr_weight, shrink_intensity)

ORIGINS = ["1979-12-31", "1990-06-30", "2010-12-31"]   # early one: monthly-only assets


@pytest.fixture(scope="module")
def panel():
    return make_synthetic_panel(seed=3)


def _call(kind, m, view, assets):
    return m.forecast_vol(view, assets) if kind == "vol" else m.forecast_corr(view, assets)


def _check(kind, f, assets):
    if kind == "vol":
        assert list(f.index) == assets and np.isfinite(f).all() and (f > 0).all()
    else:
        c = f.to_numpy()
        assert list(f.index) == assets and list(f.columns) == assets
        assert np.isfinite(c).all() and np.allclose(c, c.T) and np.allclose(np.diag(c), 1.0)
        assert np.linalg.eigvalsh(c).min() > 0


ALL = [("vol", n) for n in VOL_MODELS] + [("corr", n) for n in CORR_MODELS]


def test_registered_within_budget():
    vol = [n for n in list_models("vol") if n.startswith("bench_")]
    corr = [n for n in list_models("corr") if n.startswith("bench_")]
    assert set(vol) == set(VOL_MODELS) and len(vol) <= 8
    assert set(corr) == set(CORR_MODELS) and len(corr) <= 6


@pytest.mark.parametrize("kind,name", ALL)
def test_valid_at_every_stage(panel, kind, name):
    m = get_model(kind, name)
    early = panel.asof(ORIGINS[0])
    assert not early.has_daily(), "early origin should be monthly-only"
    for t in ORIGINS:
        view = panel.asof(t)
        assets = view.eligible()
        _check(kind, _call(kind, m, view, assets), assets)
        sub = assets[::-1][:3]                                  # any subset, any order
        _check(kind, _call(kind, m, view, sub), sub)


@pytest.mark.parametrize("kind,name", ALL)
def test_cold_equals_warm_and_order_free(panel, kind, name):
    t = ORIGINS[1]
    view = panel.asof(t)
    assets = view.eligible()
    cold = _call(kind, get_model(kind, name), view, assets)
    warm_m = get_model(kind, name)
    for s in ["2010-12-31", "1979-12-31", "2003-03-31"]:
        v = panel.asof(s)
        _call(kind, warm_m, v, v.eligible())
    warm = _call(kind, warm_m, view, assets)
    assert np.allclose(np.asarray(cold), np.asarray(warm), rtol=0, atol=0)


@pytest.mark.parametrize("kind,name", ALL)
def test_no_lookahead(panel, kind, name):
    t = pd.Timestamp("2004-06-30")
    rng = np.random.default_rng(1)
    daily = panel.daily.copy()
    monthly = panel.monthly.copy()
    after_d, after_m = daily.index > t, monthly.index > t
    daily.loc[after_d] = daily.loc[after_d] * 5 + rng.normal(0, 0.05, daily.loc[after_d].shape)
    monthly.loc[after_m] = monthly.loc[after_m] * -3
    tiers = panel.tiers.copy()
    tiers.loc[tiers.index > t] = "A"
    altered = replace(panel, daily=daily, monthly=monthly, tiers=tiers, _shared={})
    a = panel.asof(t)
    assets = a.eligible()
    f1 = _call(kind, get_model(kind, name), a, assets)
    f2 = _call(kind, get_model(kind, name), altered.asof(t), assets)
    assert np.array_equal(np.asarray(f1), np.asarray(f2))


# ------------------------------------------------------------ building blocks
def test_daily_moment_matches_target_estimator(panel):
    """12-month sum of xd equals the target estimator on the same days (up to
    sums that straddle the window start)."""
    view = panel.asof("2008-12-31")
    a = ["us_large", "ust_int"]
    mo = Moments(view, a)
    rv_model = mo.xd[-12:].sum(0)
    X = view.excess_daily[a].loc["2008-01-01":].to_numpy()
    S, _ = realized_kday_cov(X, 21)
    assert np.allclose(rv_model, np.diag(S), rtol=0.15)


def test_ew_series_matches_ew_mean():
    rng = np.random.default_rng(0)
    x = rng.gamma(2.0, size=(80, 3))
    x[:10, 1] = np.nan
    s = ew_series(x, 12)
    for k in (20, 79):
        assert np.allclose(s[k], ew_mean(x[: k + 1], 12))
    assert np.allclose(ew_series(x, None)[-1], np.nanmean(x, 0))


def test_mr_weight_value():
    assert mr_weight(6) == pytest.approx(0.51, abs=0.01)
    assert mr_weight(1e6) == pytest.approx(1.0, abs=1e-3)


def test_corr_recovers_truth_and_unequal_history():
    """Pairwise EW correlation from monthly data with a late starter."""
    rng = np.random.default_rng(0)
    n = 400
    idx = pd.date_range("1950-01-31", periods=n, freq="ME")
    z = rng.standard_normal((n, 3))
    r = np.column_stack([z[:, 0], 0.6 * z[:, 0] + 0.8 * z[:, 1], z[:, 2]]) * 0.04
    r[:300, 2] = np.nan

    class V:  # minimal view: monthly only
        excess_monthly = pd.DataFrame(r, index=idx, columns=list("abc"))
        daily = pd.DataFrame()

    mo = Moments(V(), list("abc"), pairs=True, daily=False)
    c, neff = ew_corr(mo, None)
    assert c[0, 1] == pytest.approx(0.6, abs=0.08)
    assert abs(c[0, 2]) < 0.25 and neff[0, 2] == pytest.approx(100)


def test_shrinkage_intensity_behaves():
    rng = np.random.default_rng(0)
    groups = ["e", "e", "e", "r", "r"]
    F0 = block_target(np.full((5, 5), 0.3) + 0.4 * (np.add.outer(range(5), range(5)) < 0), groups)
    noisy = F0 + np.triu(rng.normal(0, 0.05, (5, 5)), 1)
    noisy = np.triu(noisy, 1) + np.triu(noisy, 1).T + np.eye(5)
    F = block_target(noisy, groups)
    assert shrink_intensity(noisy, F, np.full((5, 5), 30.0)) > shrink_intensity(noisy, F, np.full((5, 5), 3000.0))
    assert 0.0 <= shrink_intensity(noisy, F, np.full((5, 5), 30.0)) <= 1.0
