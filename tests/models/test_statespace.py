import numpy as np
import pandas as pd
import pytest
from dataclasses import replace

from volmodel.data.synthetic import make_synthetic_panel
from volmodel.models.statespace import grid as G
from volmodel.models.statespace import models as M
from volmodel.models.statespace.data import KAPPA, build_obs
from volmodel.models.statespace.kalman import kf_multi, kf_univ

NAMES = ["statespace_ar1_asset", "statespace_ar1", "statespace_ar1_grid", "statespace_2c",
         "statespace_ar1_daily", "statespace_2c_daily", "statespace_factor", "statespace_factor_daily"]
ORIGINS = ["1980-12-31", "1995-06-30", "2014-12-31"]   # early: monthly data only


def _make(name):
    # built directly (not through the registry) so a broken sibling package
    # cannot fail these tests; test_registered checks the registry separately
    from volmodel.models import statespace as S  # noqa: F401  (registers)
    reg = {
        "statespace_ar1_asset": lambda: M.StateSpaceVol(M.Spec(1, False, False)),
        "statespace_ar1": lambda: M.StateSpaceVol(M.Spec(1, False, True)),
        "statespace_ar1_grid": M.GridVol,
        "statespace_2c": lambda: M.StateSpaceVol(M.Spec(2, False, True)),
        "statespace_ar1_daily": lambda: M.StateSpaceVol(M.Spec(1, True, True)),
        "statespace_2c_daily": lambda: M.StateSpaceVol(M.Spec(2, True, True)),
        "statespace_factor": lambda: M.FactorVol(M.FactorSpec(daily=False)),
        "statespace_factor_daily": lambda: M.FactorVol(M.FactorSpec(daily=True)),
    }
    m = reg[name]()
    m.name = name
    return m


@pytest.fixture(autouse=True)
def fast_fits(monkeypatch):
    """Cap optimizer iterations: the properties tested do not depend on convergence."""
    monkeypatch.setattr(M, "MAXITER", 4)
    M.clear_cache()
    yield
    M.clear_cache()


@pytest.fixture(scope="module")
def panel():
    return make_synthetic_panel(seed=3)


def test_registered():
    from volmodel.models.base import _REGISTRY
    import volmodel.models.statespace  # noqa: F401
    assert set(NAMES) <= set(_REGISTRY["vol"])
    assert len([n for n in _REGISTRY["vol"] if n.startswith("statespace")]) <= 8


@pytest.mark.parametrize("name", NAMES)
def test_valid_at_every_stage(panel, name):
    m = _make(name)
    for t in ORIGINS:
        view = panel.asof(t)
        assets = view.eligible()
        v = m.forecast_vol(view, assets)
        assert list(v.index) == assets
        assert np.isfinite(v).all() and (v > 0).all()
        assert (v > 0.01).all() and (v < 1.0).all()


@pytest.mark.parametrize("name", NAMES)
def test_cold_equals_warm_any_order(panel, name):
    t = "1995-06-30"
    view = panel.asof(t)
    assets = view.eligible()
    cold = _make(name).forecast_vol(view, assets)
    M.clear_cache()
    m = _make(name)
    for other in ["2003-12-31", "1990-01-31", "1994-12-31", "1995-03-31"]:
        o = panel.asof(other)
        m.forecast_vol(o, o.eligible())
    warm = m.forecast_vol(view, assets)
    pd.testing.assert_series_equal(cold, warm, rtol=1e-10, atol=0)


@pytest.mark.parametrize("name", NAMES)
def test_no_lookahead(panel, name):
    t = pd.Timestamp("1995-06-30")
    daily = panel.daily.copy()
    monthly = panel.monthly.copy()
    daily.loc[daily.index > t] *= 3.0
    monthly.loc[monthly.index > t] *= 3.0
    tiers = panel.tiers.copy()
    tiers.loc[tiers.index > t] = "A"
    altered = replace(panel, daily=daily, monthly=monthly, tiers=tiers, _shared={})
    view = panel.asof(t)
    a = _make(name).forecast_vol(view, view.eligible())
    M.clear_cache()
    b = _make(name).forecast_vol(altered.asof(t), view.eligible())
    pd.testing.assert_series_equal(a, b, rtol=1e-12, atol=0)


def test_estimation_date_is_last_december():
    idx = pd.date_range("1990-01-31", "2000-12-31", freq="ME")
    assert M.estimation_date("1995-06-30", idx, 1) == pd.Timestamp("1994-12-31")
    assert M.estimation_date("1995-12-31", idx, 1) == pd.Timestamp("1995-12-31")
    assert M.estimation_date("1995-06-30", idx, 5) == pd.Timestamp("1990-12-31")


def test_obs_measurement_is_unbiased_for_normal_returns():
    rng = np.random.default_rng(0)
    z2 = rng.standard_normal(400_000) ** 2
    assert np.mean(np.log(z2 + 0.005)) == pytest.approx(KAPPA, abs=0.01)


def test_daily_obs_and_splice_flags(panel):
    o = build_obs(panel.asof("2000-12-31"))
    j = o.assets.index("us_large")
    first_daily = np.argmax(np.isfinite(o.y2[:, j]))
    assert o.months[first_daily] >= pd.Timestamp("1985-01-31")      # synthetic daily era
    assert not np.isfinite(o.y2[: first_daily, j]).any()
    assert o.reset[:, j].sum() == 0          # tier C -> B happens before the offset is live


def test_kalman_matches_closed_form_local_level():
    """One AR(1) state, one observation: compare with a scalar recursion."""
    rng = np.random.default_rng(1)
    T, phi, q, H = 200, 0.9, 0.1, 2.0
    y = rng.standard_normal(T)
    y[[5, 50, 51]] = np.nan
    d = np.array([[1.0, phi]])
    qq = np.array([[0.0, q]])
    a0 = np.array([[0.3, 0.0]])
    P0 = np.diag([0.5, q / (1 - phi ** 2)])[None]
    Z = np.array([1.0, 1.0])
    ll, a, P, _ = kf_univ(d, qq, a0, P0, np.array([H]), np.array([1.0]), y[:, None], None,
                          np.zeros((T, 1), bool), np.zeros((T, 1), bool), 0.0, 0.0, Z, Z, None)
    # reference: generic Kalman filter with explicit matrices
    Tm = np.diag([1.0, phi]); Q = np.diag([0.0, q])
    m, S, ref = a0[0].copy(), P0[0].copy(), 0.0
    for t in range(T):
        m, S = Tm @ m, Tm @ S @ Tm.T + Q
        if np.isfinite(y[t]):
            F = Z @ S @ Z + H
            v = y[t] - Z @ m
            K = S @ Z / F
            m, S = m + K * v, S - np.outer(K, K) * F
            ref += -0.5 * (np.log(2 * np.pi) + np.log(F) + v * v / F)
    assert ll[0] == pytest.approx(ref, rel=1e-9)
    assert np.allclose(a[0], m) and np.allclose(P[0], S)
    # the multivariate filter gives the same answer
    ll2, a2, P2, _ = kf_multi(d, qq, a0, P0, Z[None, None, :], np.array([[[H]]]), y[:, None],
                              np.zeros(2), np.zeros(2), np.zeros((T, 2), bool),
                              np.zeros((T, 2), bool), Z[None, None, :])
    assert ll2[0] == pytest.approx(ref, rel=1e-9) and np.allclose(a2[0], m)


def test_grid_filter_recovers_level():
    rng = np.random.default_rng(2)
    T, mu, phi, sx = 600, np.log(0.01), 0.95, 0.5
    z = np.zeros(T)
    for t in range(1, T):
        z[t] = phi * z[t - 1] + np.sqrt(1 - phi ** 2) * rng.standard_normal()
    r2 = (np.exp(0.5 * (mu + sx * z)) * rng.standard_normal(T)) ** 2
    K = G.transition(phi)
    pi, pred = G.grid_filter(G.prior()[None, :], K, np.array([mu]), np.array([sx]), r2[:, None], store=True)
    assert np.mean(r2[100:] / pred[100:, 0]) == pytest.approx(1.0, abs=0.15)
    assert np.allclose(K.sum(1), 1.0)


def test_pooled_fit_recovers_persistence(panel):
    """On the synthetic panel (log-vol AR(1) daily phi = 0.995, i.e. ~0.9 a
    month) a converged pooled fit should find a persistent state."""
    import volmodel.models.statespace.models as MM
    MM.MAXITER = 100
    fit = MM.fit_univ(MM.Spec(1, True, True), panel.asof("2014-12-31"))
    th = dict(zip(fit["names"], fit["theta"][0]))
    assert 0.6 < th["phi1"] < 0.999
    assert 0.5 < fit["s"] < 2.0
