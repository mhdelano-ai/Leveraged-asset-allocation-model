from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from assetforecast.data.synthetic import make_synthetic_panel
from assetforecast.models import get_model, list_models
from assetforecast.models.regime.breaks import BOCPDState, bocpd_update, icss_kappa2, run_length_map
from assetforecast.models.regime.hmm import (expected_path, fit_common_ms_t, fit_ms_t, forward_backward,
                                        stationary)
from assetforecast.models.regime.models import estimation_date

MODELS = ["regime_ms2", "regime_ms3", "regime_ms2_weekly", "regime_common3",
          "regime_bocpd", "regime_bocpd_eb", "regime_bocpd_weekly", "regime_break_blend"]


@pytest.fixture(scope="module")
def panel():
    return make_synthetic_panel(seed=3)


@pytest.fixture(scope="module")
def small():
    # shorter panel for the repeated-call tests (fits are the expensive part)
    return make_synthetic_panel(seed=5, start="1984-01-01", end="2010-12-31", monthly_only_years=6)


def test_registered():
    assert set(MODELS) <= set(list_models("vol"))
    assert len([m for m in list_models("vol") if m.startswith("regime_")]) <= 8


@pytest.mark.parametrize("name", MODELS)
def test_valid_at_every_stage(panel, name):
    m = get_model("vol", name)
    for t in ["1979-12-31", "1990-06-30", "2010-12-31"]:
        view = panel.asof(t)
        assets = view.eligible()
        v = m.forecast_vol(view, assets)
        assert list(v.index) == assets
        assert np.isfinite(v).all() and (v > 0).all()
        assert m.last_error is None
        # synthetic long-run vols are 6%-23%: forecasts must be in a sane range
        assert (v > 0.02).all() and (v < 0.6).all()


@pytest.mark.parametrize("name", MODELS)
def test_cold_warm_and_order_invariance(small, name):
    t, others = "2007-03-31", ["2006-12-31", "2008-06-30", "2004-01-31"]
    view = small.asof(t)
    assets = view.eligible()
    cold = get_model("vol", name).forecast_vol(view, assets)
    m = get_model("vol", name)
    for o in others:
        v = small.asof(o)
        m.forecast_vol(v, v.eligible())
    warm = m.forecast_vol(view, assets)
    pd.testing.assert_series_equal(cold, warm, rtol=1e-12, atol=0)
    m2 = get_model("vol", name)
    for o in sorted(others):
        v = small.asof(o)
        m2.forecast_vol(v, v.eligible())
    pd.testing.assert_series_equal(cold, m2.forecast_vol(view, assets), rtol=1e-12, atol=0)


@pytest.mark.parametrize("name", MODELS)
def test_no_lookahead(small, name):
    t = pd.Timestamp("2006-09-30")
    rng = np.random.default_rng(1)
    d, mo = small.daily.copy(), small.monthly.copy()
    d.loc[d.index > t] = d.loc[d.index > t] * 4 + rng.normal(0, 0.01, d.loc[d.index > t].shape)
    mo.loc[mo.index > t] = mo.loc[mo.index > t] * 4
    altered = replace(small, daily=d, monthly=mo)
    a = small.asof(t).eligible()
    m = get_model("vol", name)
    m.forecast_vol(altered.asof("2009-12-31"), altered.asof("2009-12-31").eligible())   # warm on altered data
    f1 = m.forecast_vol(altered.asof(t), a)
    f0 = get_model("vol", name).forecast_vol(small.asof(t), a)
    pd.testing.assert_series_equal(f0, f1, rtol=1e-12, atol=0)


def test_estimation_date_is_last_december(panel):
    assert estimation_date(panel.asof("2000-11-30")) == pd.Timestamp("1999-12-31")
    assert estimation_date(panel.asof("2000-12-31")) == pd.Timestamp("2000-12-31")


# ----------------------------------------------------------------- machinery
def test_forward_backward_matches_loop():
    rng = np.random.default_rng(0)
    T, K = 300, 3
    P = np.array([[.9, .08, .02], [.1, .85, .05], [.05, .15, .8]])
    logb = rng.normal(size=(T, K)) * 3
    logb[7] = 0.0
    f, g, xi, ll = forward_backward(logb, P)
    b = np.exp(logb)
    a, c, prev = np.zeros((T, K)), np.zeros(T), stationary(P)
    for t in range(T):
        v = (prev @ P) * b[t]
        c[t] = v.sum()
        a[t] = prev = v / c[t]
    be = np.ones((T, K))
    for t in range(T - 2, -1, -1):
        v = P @ (b[t + 1] * be[t + 1])
        be[t] = v / v.sum()
    gg = a * be
    gg /= gg.sum(axis=1, keepdims=True)
    assert np.allclose(f, a, atol=1e-12) and np.allclose(g, gg, atol=1e-12)
    assert ll == pytest.approx(np.log(c).sum(), abs=1e-8)
    assert xi.sum() == pytest.approx(T - 1)


def test_expected_path_uses_transition_powers():
    P = np.array([[0.9, 0.1], [0.3, 0.7]])
    lev = np.array([1.0, 4.0])
    want = np.mean([(np.array([0.0, 1.0]) @ np.linalg.matrix_power(P, h)) @ lev for h in range(1, 13)])
    assert expected_path(np.array([0.0, 1.0]), P, lev, 12) == pytest.approx(want)
    # far horizon -> stationary mean, not the current regime
    assert expected_path(np.array([0.0, 1.0]), P, lev, 2000) == pytest.approx(stationary(P) @ lev, rel=1e-2)


def _simulate_ms(T, P, vols, seed, nu=5):
    rng = np.random.default_rng(seed)
    s = np.zeros(T, int)
    for t in range(1, T):
        s[t] = rng.choice(len(P), p=P[s[t - 1]])
    e = rng.standard_t(nu, size=T) * np.sqrt((nu - 2) / nu)
    return np.asarray(vols)[s] * e, s


def test_fit_ms_t_recovers_regimes():
    P = np.array([[0.97, 0.03], [0.08, 0.92]])
    x, s = _simulate_ms(1500, P, [0.03, 0.08], seed=2)
    x[200:205] = np.nan
    f = fit_ms_t(x, 2)
    assert np.sqrt(f.var) == pytest.approx([0.03, 0.08], rel=0.15)
    assert np.diag(f.P) == pytest.approx(np.diag(P), abs=0.04)


def test_t_emissions_ignore_single_outlier():
    rng = np.random.default_rng(4)
    x = rng.normal(0, 0.04, 600)
    x[300] = -0.40                                   # a 10-sigma month (1987-10-19 style)
    f = fit_ms_t(x, 2)
    # no state is spent on one month: the high state, if any, is not a single-month episode
    assert f.durations().min() > 3


def test_common_fit_shares_regime():
    rng = np.random.default_rng(6)
    P = np.array([[0.96, 0.04], [0.10, 0.90]])
    T = 800
    _, s = _simulate_ms(T, P, [1, 1], seed=7)
    mult = np.array([1.0, 2.5])[s]
    vols = np.array([0.04, 0.02, 0.06])
    C = np.array([[1, .5, .3], [.5, 1, .2], [.3, .2, 1]])
    Z = rng.standard_normal((T, 3)) @ np.linalg.cholesky(C).T
    X = Z * vols[None, :] * mult[:, None]
    X[:600, 2] = np.nan                              # short-history asset
    f = fit_common_ms_t(X, C, 2, assets=["a", "b", "c"])
    ratio = np.sqrt(f.var[:, 1] / f.var[:, 0])
    assert ratio[:2] == pytest.approx([2.5, 2.5], rel=0.2)
    assert 1.5 < ratio[2] < 3.5                      # the short asset inherits the regime
    acc = ((f.filter(X)[:, 1] > 0.5) == (s == 1)).mean()
    assert acc > 0.85


def test_bocpd_finds_level_shift():
    rng = np.random.default_rng(8)
    x = np.r_[rng.normal(0, 0.02, 300), rng.normal(0, 0.06, 120)]
    v0 = float(np.mean(x ** 2))
    st = bocpd_update(BOCPDState(3.0, 2.0 * v0), x, 1 / 240, 3.0, 2.0 * v0)
    rl, _ = run_length_map(st)
    assert 100 <= rl <= 130


def test_icss_breaks_and_false_alarms():
    rng = np.random.default_rng(9)
    x = np.r_[rng.normal(0, 0.02, 400), rng.normal(0, 0.05, 400)]
    br = icss_kappa2(x)
    assert len(br) >= 1 and min(abs(b - 400) for b in br) < 30
    # persistent volatility clusters without a level shift: the HAC kappa-2 should mostly stay quiet
    n_false = 0
    for seed in range(20):
        r = np.random.default_rng(100 + seed)
        lv = np.zeros(800)
        for t in range(1, 800):
            lv[t] = 0.95 * lv[t - 1] + 0.25 * r.standard_normal()
        n_false += len(icss_kappa2(np.exp(lv) * r.standard_normal(800))) > 0
    assert n_false <= 6
