"""Tests of assetforecast.bridge: fast, synthetic, seeded."""
import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from assetforecast.bridge import (BRIDGES, NON_MARKET, Q, ScaleMixture, estimate_long_run, floor_relative,
                             get_bridge, market_mask, predictive_quantiles, standardized_returns,
                             variance_ratio, vol_uncertainty)
from assetforecast.data.panel import PREDICTOR_COLUMNS, Panel
from assetforecast.data.synthetic import make_synthetic_panel
from assetforecast.universe import ASSET_IDS


# ------------------------------------------------------------------ helpers
def true_vr(rho):
    """VR(12) from autocorrelations rho[0] = lag 1 ... rho[10] = lag 11."""
    return 1 + 2 * sum((1 - k / Q) * rho[k - 1] for k in range(1, Q))


def ar1(n, phi, rng, burn=200):
    e = rng.standard_normal((n + burn,) if np.ndim(phi) == 0 else (n + burn, len(phi)))
    x = np.zeros_like(e)
    for t in range(1, n + burn):
        x[t] = phi * x[t - 1] + e[t]
    return x[burn:]


def monthly_panel(R: np.ndarray, columns, start="1801-01-31", vol=0.04) -> Panel:
    """Monthly-only Panel whose monthly LOG excess returns are ``vol * R`` (cash = 0)."""
    idx = pd.date_range(start, periods=len(R), freq="ME")
    monthly = pd.DataFrame(np.expm1(vol * R), idx, columns)
    cash_m = pd.Series(0.0, idx, name="cash")
    daily = pd.DataFrame(columns=columns, index=pd.DatetimeIndex([], name="date"), dtype=float)
    return Panel(daily=daily, monthly=monthly, cash_daily=pd.Series([], index=daily.index, dtype=float, name="cash"),
                 cash_monthly=cash_m, tiers=pd.DataFrame("C", idx, columns, dtype=object),
                 predictors=pd.DataFrame(columns=PREDICTOR_COLUMNS))


def vr_monte_carlo(gen, n=720, reps=80):
    """Estimate VR on ``reps`` independent series; returns (estimates, stated se)."""
    X = gen(n + 12, reps)
    idx = pd.date_range("1801-01-31", periods=n + 12, freq="ME")
    U = standardized_returns(pd.DataFrame(0.04 * X + 0.003, idx))
    out = np.array([variance_ratio(U[c].dropna().to_numpy())[:2] for c in U.columns])
    return out[:, 0], out[:, 1]


# --------------------------------------------------- variance ratio recovery
CASES = {
    "ar_pos": (lambda n, r, rng: ar1(n, np.full(r, 0.3), rng), true_vr([0.3 ** k for k in range(1, Q)])),
    "ar_neg": (lambda n, r, rng: ar1(n, np.full(r, -0.3), rng), true_vr([(-0.3) ** k for k in range(1, Q)])),
    # smoothed price: observed return = e_t + 0.5 e_{t-1} (rho_1 = 0.4), like a stale NAV
    "smoothed": (lambda n, r, rng: (lambda e: e[1:] + 0.5 * e[:-1])(rng.standard_normal((n + 1, r))),
                 true_vr([0.4] + [0.0] * 10)),
    "iid": (lambda n, r, rng: rng.standard_normal((n, r)), 1.0),
}


@pytest.mark.parametrize("case", list(CASES))
def test_variance_ratio_recovers_truth_within_stated_error(case):
    gen, truth = CASES[case]
    rng = np.random.default_rng(11)
    est, se = vr_monte_carlo(lambda n, r: gen(n, r, rng))
    # unbiased to within half a stated standard error (60 years of data)
    assert abs(est.mean() - truth) < 0.5 * se.mean(), (est.mean(), truth, se.mean())
    # the stated standard error matches the actual dispersion (+-35%)
    assert 0.65 < est.std() / se.mean() < 1.35, (est.std(), se.mean())


def test_variance_ratio_robust_to_volatility_clustering():
    """With strong stochastic volatility the standardized estimator still finds VR = 1 for iid shocks."""
    rng = np.random.default_rng(5)
    n, r = 720, 80
    h = ar1(n + 12, np.full(r, 0.97), rng) * 0.25
    X = np.exp(h) * rng.standard_normal((n + 12, r))
    idx = pd.date_range("1801-01-31", periods=n + 12, freq="ME")
    U = standardized_returns(pd.DataFrame(0.04 * X, idx))
    est = np.array([variance_ratio(U[c].dropna().to_numpy())[0] for c in U.columns])
    assert abs(est.mean() - 1) < 0.05


def test_standardization_is_ex_ante():
    """The scale applied to month t uses months before t only."""
    rng = np.random.default_rng(0)
    idx = pd.date_range("1901-01-31", periods=100, freq="ME")
    x = pd.DataFrame({"a": rng.standard_normal(100) * 0.04}, idx)
    x2 = x.copy()
    x2.iloc[60, 0] *= 10          # a huge month 60
    u1, u2 = standardized_returns(x)["a"], standardized_returns(x2)["a"]
    sig = lambda xx, uu, i: (xx["a"].iloc[i] - xx["a"].mean()) / uu.iloc[i]
    assert abs(sig(x2, u2, 60) / sig(x, u1, 60) - 1) < 0.02     # month 60's own size does not enter its scale
    assert sig(x2, u2, 61) / sig(x, u1, 61) > 1.5                # but it enters the next month's


# ------------------------------------------------------- bridge calibration
def test_bridge_calibrates_z_scores_identity_does_not():
    """Independent AR(1) assets (phi = 0.25): train on 200 years, test on the next 200
    non-overlapping years with the TRUE monthly covariance as cov_rv (oracle vol)."""
    rng = np.random.default_rng(3)
    n_assets, n = 4, 4800
    X = ar1(n, np.full(n_assets, 0.25), rng)
    X = X / X[:2400].std(axis=0)                       # unit monthly variance in-sample
    cols = [ASSET_IDS[i] for i in range(n_assets)]
    panel = monthly_panel(X, cols)
    view = panel.asof(panel.monthly.index[2399])
    m_var = 0.04 ** 2 * (1 / (1 - 0.25 ** 2)) / (1 / (1 - 0.25 ** 2))   # = vol^2 (unit variance scaling)
    cov_rv = pd.DataFrame(np.eye(n_assets) * 12 * m_var, cols, cols)
    A_id = get_bridge("identity").cov_annual(cov_rv, view)
    A_b = get_bridge("vr_market").cov_annual(cov_rv, view)
    R = 0.04 * X[2400:].reshape(-1, 12, n_assets).sum(axis=1)          # 200 years x 4
    R = R - R.mean(axis=0)
    z_id = R / np.sqrt(np.diag(A_id))
    z_b = R / np.sqrt(np.diag(A_b))
    truth = true_vr([0.25 ** k for k in range(1, Q)])                   # ~1.57
    v_id, v_b = (z_id ** 2).mean(), (z_b ** 2).mean()                   # se of each ~ sqrt(2/800) = 0.05
    assert abs(v_id - truth) < 0.25 and v_id > 1.3
    assert abs(v_b - 1.0) < 0.15, (v_b, np.diag(A_b) / np.diag(A_id))


def test_iid_returns_keep_bridge_near_identity():
    rng = np.random.default_rng(8)
    X = rng.standard_normal((600, 12))
    panel = monthly_panel(X, ASSET_IDS)
    view = panel.asof(panel.monthly.index[-1])
    est = estimate_long_run(view, ASSET_IDS, era="all", cross=True)
    assert (est.vr_raw.sub(1).abs() < 0.6).all()
    assert (est.vr.sub(1).abs() < 0.1).all(), est.vr           # shrinkage pulls noise back to 1
    assert est.gap.abs().to_numpy().max() < 0.05
    cov_rv = pd.DataFrame(np.eye(12) * 0.02 + 0.01, ASSET_IDS, ASSET_IDS)
    for name in BRIDGES:
        A = get_bridge(name).cov_annual(cov_rv, view)
        assert np.allclose(A.to_numpy(), cov_rv.to_numpy(), rtol=0.12), name


def test_common_lead_lag_shrinkage_does_not_lower_correlations():
    """Two assets with the SAME autocorrelation and high correlation: annual correlation
    equals the monthly one, and the bridge keeps it (no artificial diversification)."""
    rng = np.random.default_rng(2)
    f = ar1(1200, 0.3, rng)
    X = np.column_stack([f + 0.3 * rng.standard_normal(1200), f + 0.3 * rng.standard_normal(1200)])
    cols = ASSET_IDS[:2]
    panel = monthly_panel(X, cols)
    view = panel.asof(panel.monthly.index[-1])
    c = 0.9
    cov_rv = pd.DataFrame([[1, c], [c, 1]], cols, cols) * 0.02
    for name in ("vr_market", "lr_market"):
        A = get_bridge(name).cov_annual(cov_rv, view).to_numpy()
        ca = A[0, 1] / np.sqrt(A[0, 0] * A[1, 1])
        assert abs(ca - c) < 0.05, (name, ca)
        assert A[0, 0] / cov_rv.iat[0, 0] > 1.3


# ------------------------------------------------------- structural checks
@pytest.fixture(scope="module")
def synth():
    return make_synthetic_panel(seed=4)


@pytest.mark.parametrize("name", list(BRIDGES))
def test_output_pd_and_assets_preserved(synth, name):
    rng = np.random.default_rng(1)
    b = get_bridge(name)
    for t in ["1979-12-31", "1990-06-30", "2010-12-31"]:
        view = synth.asof(t)
        assets = view.eligible(36)
        assets = list(rng.permutation(assets))          # arbitrary order must be preserved
        k = len(assets)
        L = rng.standard_normal((k, k + 2)) * 0.1
        for cov in (L @ L.T, L[:, :1] @ L[:, :1].T + 1e-8 * np.eye(k)):   # well- and ill-conditioned
            cov_rv = pd.DataFrame(cov, assets, assets)
            A = b.cov_annual(cov_rv, view)
            assert list(A.index) == assets and list(A.columns) == assets
            a = A.to_numpy()
            assert np.isfinite(a).all() and np.allclose(a, a.T)
            assert np.linalg.eigvalsh(a).min() > 0


def test_short_history_asset_heavily_shrunk():
    rng = np.random.default_rng(6)
    n = 600
    X = ar1(n, np.full(4, 0.25), rng)
    X[:-36, 3] = np.nan                                  # asset 3 has only 36 months
    cols = ASSET_IDS[:4]
    panel = monthly_panel(X, cols)
    view = panel.asof(panel.monthly.index[-1])
    est = estimate_long_run(view, cols, era="all")
    assert est.n[cols[3]] == 24                          # 12 months used to start the EWMA
    assert est.vr_se[cols[3]] > 0.4
    assert 0.8 < est.vr[cols[3]] < 1.4
    w = est.shrink_weight
    assert w[cols[3]] < 0.5 and w[cols[3]] < w[cols[:3]].min()
    A = get_bridge("lr_market").cov_annual(pd.DataFrame(np.eye(4) * 0.02, cols, cols), view)
    assert np.isfinite(A.to_numpy()).all()


def test_asset_with_too_little_history_gets_ratio_one():
    rng = np.random.default_rng(6)
    X = rng.standard_normal((300, 3))
    X[:-20, 2] = np.nan
    cols = ASSET_IDS[:3]
    view = monthly_panel(X, cols).asof("2100-01-01")
    est = estimate_long_run(view, cols)
    assert est.vr[cols[2]] == 1.0 and not np.isfinite(est.vr_se[cols[2]])


def test_no_look_ahead_and_cold_equals_warm(synth):
    t = pd.Timestamp("1995-03-31")
    view = synth.asof(t)
    assets = view.eligible(36)
    cov_rv = pd.DataFrame(np.diag(np.linspace(0.01, 0.05, len(assets))) + 0.002, assets, assets)
    # alter everything after t
    rng = np.random.default_rng(0)
    m2 = synth.monthly.copy()
    later = m2.index > t
    m2.loc[later] = m2.loc[later] * 3 + rng.normal(0, 0.05, m2.loc[later].shape)
    d2 = synth.daily.copy()
    d2.loc[d2.index > t] = d2.loc[d2.index > t] * -2
    altered = Panel(daily=d2, monthly=m2, cash_daily=synth.cash_daily, cash_monthly=synth.cash_monthly,
                    tiers=synth.tiers, predictors=synth.predictors)
    for name in BRIDGES:
        cold = get_bridge(name).cov_annual(cov_rv, view)
        alt = get_bridge(name).cov_annual(cov_rv, altered.asof(t))
        warm_b = get_bridge(name)
        for o in ["2005-12-31", "1985-06-30", "1999-01-31"]:
            v = synth.asof(o)
            a = v.eligible(36)
            warm_b.cov_annual(pd.DataFrame(np.eye(len(a)) * 0.02, a, a), v)
        warm = warm_b.cov_annual(cov_rv, view)
        assert np.allclose(cold.to_numpy(), alt.to_numpy()), name
        assert np.allclose(cold.to_numpy(), warm.to_numpy()), name


def test_parameters_reestimated_once_a_year(synth):
    b = get_bridge("vr_market")
    a = synth.asof("2000-12-31").eligible(36)
    cov = pd.DataFrame(np.eye(len(a)) * 0.02, a, a)
    A1 = b.cov_annual(cov, synth.asof("2001-01-31"))
    A2 = b.cov_annual(cov, synth.asof("2001-11-30"))
    A3 = b.cov_annual(cov, synth.asof("2001-12-31"))
    assert np.allclose(A1.to_numpy(), A2.to_numpy())
    assert not np.allclose(A1.to_numpy(), A3.to_numpy())


def test_market_mask_excludes_documented_segments():
    idx = pd.date_range("1900-01-31", "2014-12-31", freq="ME")
    m = market_mask(idx, ["us_large", "hy_credit", "gold"], "market")
    assert not m[idx <= "1926-06-30", 0].any() and m[idx >= "1926-07-31", 0].all()
    assert not m[(idx >= "1985-01-31") & (idx <= "2007-04-30"), 1].any() and m[idx >= "2007-05-31", 1].all()
    assert m[:, 2].all()
    assert market_mask(idx, ["us_small"], "all").all()
    assert not market_mask(idx, ["us_small"], "strict")[(idx >= "1926-07-31") & (idx < "2000-05-01")].any()
    assert {a for a, *_ in NON_MARKET} <= set(ASSET_IDS)


def test_floor_relative_bounds_portfolio_ratio():
    rng = np.random.default_rng(0)
    L = rng.standard_normal((6, 8))
    S = L @ L.T
    d = np.sqrt(np.diag(S))
    R = S / np.outer(d, d)
    G = rng.normal(0, 0.3, (6, 6))
    C = R + (G + G.T) / 2
    np.fill_diagonal(C, 1.0)
    F = floor_relative(C, R)
    assert np.allclose(np.diag(F), 1.0) and np.linalg.eigvalsh(F).min() > 0
    W = rng.standard_normal((500, 6))
    ratio = np.einsum("pi,ij,pj->p", W, F, W) / np.einsum("pi,ij,pj->p", W, R, W)
    assert ratio.min() > 0.3            # 0.5 before the unit diagonal is restored


# ------------------------------------------------- predictive distribution
def test_scale_mixture_moments_and_quantiles():
    rng = np.random.default_rng(0)
    m0 = ScaleMixture(0.0)
    assert abs(m0.quantile(0.05) - norm.ppf(0.05)) < 1e-6 and m0.kurtosis == 3.0
    m = ScaleMixture(0.5)
    eps = np.exp(rng.normal(-0.25, np.sqrt(0.5), 400_000))
    x = np.sqrt(eps) * rng.standard_normal(400_000)
    assert abs(x.var() - 1) < 0.02
    assert abs(m.quantile(0.01) - np.quantile(x, 0.01)) < 0.03
    assert abs(m.quantile(0.05) - np.quantile(x, 0.05)) < 0.02
    assert abs(m.share_beyond(2) - np.mean(np.abs(x) > 2)) < 0.003
    assert abs(np.mean(x ** 4) - m.kurtosis) / m.kurtosis < 0.1
    # same variance, fatter extreme tail, thinner moderate tail
    assert abs(m.quantile(0.01)) > abs(norm.ppf(0.01)) and abs(m.quantile(0.05)) < abs(norm.ppf(0.05))


def test_predictive_quantiles_shapes():
    A = pd.DataFrame([[0.04, 0.01], [0.01, 0.01]], ["a", "b"], ["a", "b"])
    q = predictive_quantiles(A, s2=0.4)
    assert list(q.index) == ["a", "b"] and np.allclose(q["sigma"], [0.2, 0.1])
    assert (q["q0.01_mixture"] < q["q0.01_normal"]).all()
    w = pd.Series({"a": 0.6, "b": 0.4})
    qp = predictive_quantiles(A, w, s2=0.0)
    assert np.isclose(qp["q0.05_normal"].iloc[0], qp["q0.05_mixture"].iloc[0])


def _vol_sim(s2, rng, n_years=150, n_assets=4):
    """Monthly returns with variance constant within calendar years, lognormal across years.

    Returns the panel and the TRUE dispersion target Var(log V / V_hat) over the same
    asset-origins ``vol_uncertainty`` uses (V = integrated variance of each 12-month window,
    V_hat = the ref_blend forecast computed exactly as in the module).
    """
    lv = rng.normal(-s2 / 2, np.sqrt(s2), (n_years, n_assets)) if s2 > 0 else np.zeros((n_years, n_assets))
    sig2 = np.repeat(np.exp(lv), 12, axis=0) * 0.04 ** 2
    X = np.sqrt(sig2) * rng.standard_normal((12 * n_years, n_assets))
    panel = monthly_panel(X / 0.04, ASSET_IDS[:n_assets], start="1820-12-31")
    r = panel.excess_monthly.to_numpy()
    E = []
    for j in range(n_assets):
        c2 = np.concatenate([[0.0], np.cumsum(r[:, j] ** 2)])
        cs = np.concatenate([[0.0], np.cumsum(sig2[:, j])])
        t = np.arange(60, len(r) - Q + 1)
        vhat = np.sqrt(12 * c2[t] / t * 12 * (c2[t] - c2[t - 60]) / 60)
        E.append(np.log((cs[t + Q] - cs[t]) / vhat))
    return panel, float(np.var(np.concatenate(E)))


def test_vol_uncertainty_recovers_dispersion():
    """s2 removes the measurement noise of a 12-month realized variance: it matches the
    dispersion of the TRUE integrated variance around the reference forecast."""
    rng = np.random.default_rng(4)
    for s2_true in (0.0, 0.4):
        panel, target = _vol_sim(s2_true, rng)
        got = vol_uncertainty(panel)
        assert abs(got["s2"] - target) < 0.08, (s2_true, target, got)
        assert abs(got["s2_gap"] - target) < 0.1, (s2_true, target, got)


# --------------------------------------------------------------- real data
@pytest.mark.realdata
def test_real_panel_estimates_are_sane():
    import os
    from assetforecast.data.panel import load_panel
    # development panel; passed explicitly so the setting does not leak into later tests
    p = load_panel(max_date=os.environ.get("ASSETFORECAST_MAX_DATE") or "2014-12-31")
    est = estimate_long_run(p, era="market", cross=True)
    assert est.vr.between(0.5, 2.5).all() and np.isfinite(est.vr_se[est.n >= 24]).all()
    assert np.isfinite(vol_uncertainty(p)["s2"])
