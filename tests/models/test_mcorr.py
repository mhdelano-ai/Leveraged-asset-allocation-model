import dataclasses

import numpy as np
import pandas as pd
import pytest

from volmodel.data.synthetic import make_synthetic_panel
from volmodel.models import get_model, list_models
from volmodel.models.mcorr import McorrDCC, McorrStatic
from volmodel.models.mcorr import core, dcc

MODELS = ["mcorr_pairwise", "mcorr_stambaugh", "mcorr_block", "mcorr_shrink_block", "mcorr_nls",
          "mcorr_fa3", "mcorr_factor2", "mcorr_dcc"]
# early origin: monthly-only era (daily data start 1985) with only 3 assets eligible
ORIGINS = ["1979-12-31", "1992-06-30", "2012-12-31"]


@pytest.fixture(scope="module")
def panel():
    return make_synthetic_panel(seed=3)


def _valid(c: pd.DataFrame, assets):
    a = c.to_numpy()
    assert list(c.index) == assets and list(c.columns) == assets
    assert np.isfinite(a).all()
    assert np.allclose(a, a.T, atol=1e-12)
    assert np.allclose(np.diag(a), 1.0)
    assert np.linalg.eigvalsh(a).min() > 0
    assert (np.abs(a) <= 1 + 1e-12).all()


def test_registered_and_at_most_eight():
    names = [m for m in list_models("corr") if m.startswith("mcorr_")]
    assert sorted(names) == sorted(MODELS)
    assert len(names) <= 8


@pytest.mark.parametrize("name", MODELS)
def test_valid_at_every_stage(panel, name):
    m = get_model("corr", name)
    for t in ORIGINS:
        view = panel.asof(t)
        assets = view.eligible()
        assert len(assets) >= 2
        _valid(m.forecast_corr(view, assets), assets)
    # any requested subset and order, including a factor asset left out
    view = panel.asof(ORIGINS[-1])
    sub = ["gold", "tips", "em", "ig_credit"]
    _valid(m.forecast_corr(view, sub), sub)


@pytest.mark.parametrize("name", MODELS)
def test_cold_warm_and_order_independent(panel, name):
    cold = {}
    for t in ORIGINS:
        v = panel.asof(t)
        cold[t] = get_model("corr", name).forecast_corr(v, v.eligible())
    warm = get_model("corr", name)
    for order in (ORIGINS, ORIGINS[::-1], ORIGINS):
        for t in order:
            v = panel.asof(t)
            pd.testing.assert_frame_equal(warm.forecast_corr(v, v.eligible()), cold[t], atol=1e-12, rtol=0)


@pytest.mark.parametrize("name", MODELS)
def test_no_lookahead(panel, name):
    t = pd.Timestamp("1999-06-30")
    rng = np.random.default_rng(1)
    monthly = panel.monthly.copy()
    daily = panel.daily.copy()
    late_m, late_d = monthly.index > t, daily.index > t
    monthly.loc[late_m] = monthly.loc[late_m] * 3 + rng.normal(0, 0.05, monthly.loc[late_m].shape)
    daily.loc[late_d] = daily.loc[late_d] * 3
    altered = dataclasses.replace(panel, monthly=monthly, daily=daily, _shared={})
    v0, v1 = panel.asof(t), altered.asof(t)
    a = v0.eligible()
    pd.testing.assert_frame_equal(get_model("corr", name).forecast_corr(v0, a),
                                  get_model("corr", name).forecast_corr(v1, a), atol=0, rtol=0)


# ------------------------------------------------------------- unit tests
def _gauss(n, C, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, len(C))) @ np.linalg.cholesky(C).T


def test_stambaugh_equals_sample_without_missing_data():
    C = np.array([[1, .5, .2], [.5, 1, .3], [.2, .3, 1]])
    X = _gauss(400, C)
    w = core.ew_weights(400, 60)
    S = core.stambaugh_cov(X, w)
    direct = (X * w[:, None]).T @ X / w.sum()
    assert np.allclose(S, direct, atol=1e-10)


def test_stambaugh_monotone_missing_uses_long_history():
    C = np.array([[1, .6, .3], [.6, 1, .4], [.3, .4, 1]])
    X = _gauss(3000, C, seed=2)
    X[:2700, 2] = np.nan                 # short history for asset 2
    X[:1000, 1] = np.nan
    w = np.ones(len(X))
    R = core.stambaugh_corr(X, w)
    assert np.allclose(R, C, atol=0.08)
    assert np.linalg.eigvalsh(R).min() > 0
    # order of the columns does not matter
    R2 = core.stambaugh_corr(X[:, [2, 0, 1]], w)
    assert np.allclose(R2, R[np.ix_([2, 0, 1], [2, 0, 1])], atol=1e-10)


def test_block_average_is_pd_and_constant_in_cells():
    C = np.array([[1, .8, .1, .2], [.8, 1, .3, .0], [.1, .3, 1, .9], [.2, .0, .9, 1]])
    g = np.array([0, 0, 1, 1])
    B = core.block_average(C, g)
    assert B[0, 1] == pytest.approx(.8) and B[2, 3] == pytest.approx(.9)
    assert B[0, 2] == pytest.approx(np.mean([.1, .2, .3, .0]))
    assert np.linalg.eigvalsh(B).min() > 0


def test_shrinkage_intensity_ordering():
    R = np.array([[1, .7, .1], [.7, 1, .5], [.1, .5, 1]])
    Tg = np.full((3, 3), .4) + .6 * np.eye(3)
    n = np.array([[0, 1000, 40], [1000, 0, 40], [40, 40, 0]], float)
    out, info = core.shrink_to_target(R, Tg, n)
    # the well-measured pair moves (relatively) less than the short-history pairs
    move = np.abs(out - R) / np.maximum(np.abs(Tg - R), 1e-12)
    assert move[0, 1] < move[0, 2] and move[0, 1] < move[1, 2]
    assert np.linalg.eigvalsh(out).min() > 0


def test_nonlinear_shrinkage_pulls_eigenvalues_in():
    C = np.eye(10)
    X = _gauss(60, C, seed=4)
    R = core.cov_to_corr(X.T @ X / 60)
    lam = np.linalg.eigvalsh(R)
    S = core.nonlinear_shrink(R, 60)
    lam2 = np.linalg.eigvalsh(S)
    assert lam2.max() < lam.max() and lam2.min() > lam.min()
    assert np.allclose(np.diag(S), 1)


def test_factor_model_recovers_one_factor_structure():
    load = np.array([.9, .8, .7, .6, .5, .4, .3])
    C = np.outer(load, load)
    np.fill_diagonal(C, 1)
    F = core.pca_factor_corr(C, 1)
    assert np.allclose(F, C, atol=1e-4)
    assert core.ledermann_k(2, 3) == 0 and core.ledermann_k(3, 3) == 1 and core.ledermann_k(12, 3) == 3


def test_observed_factor_cov_exact_when_assets_are_factors():
    C = np.array([[1, .3, .5], [.3, 1, .1], [.5, .1, 1]])
    X = _gauss(5000, C, seed=5)
    w = np.ones(len(X))
    S = core.observed_factor_cov(X[:, [0, 1, 2]], X[:, [0, 1]], w,
                                 np.array([True, True, False]), np.array([0, 1, -1]))
    R = core.cov_to_corr(S)
    assert np.allclose(R, C, atol=0.05)


def test_dcc_mean_reversion_factor_and_static_limit(panel):
    m = McorrDCC()
    v = panel.asof("2012-12-31")
    a = v.eligible()
    c = m.forecast_corr(v, a)
    k = m.last_info["kappa"]
    p = m.last_info["a"] + m.last_info["b"]
    assert 0 <= p < 1
    assert k == pytest.approx(np.mean(p ** np.arange(12)) if p > 0 else 0.0)
    # with a = b = 0 the forecast is the target (the shrink-block estimate on standardized returns)
    m0 = McorrDCC()
    m0.params = lambda *args, **kw: (0.0, 0.0)
    c0 = m0.forecast_corr(v, a).to_numpy()
    E = dcc.standardize(v.excess_monthly[a].dropna(how="all").to_numpy(float))
    assert np.allclose(c0, dcc.target_corr(E, a), atol=1e-12)
    assert np.abs(c.to_numpy() - c0).max() < 0.3


def test_static_methods_rmt_and_lw_variants(panel):
    v = panel.asof("2012-12-31")
    a = v.eligible()
    for meth in ("rmt", "lw_common"):
        _valid(McorrStatic(meth).forecast_corr(v, a), a)
