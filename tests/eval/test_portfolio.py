"""Tests for volmodel.eval.portfolio (fast, seeded)."""
import numpy as np
import pandas as pd
import pytest

from volmodel.data.panel import PREDICTOR_COLUMNS, Panel
from volmodel.data.synthetic import make_synthetic_panel
from volmodel.eval import portfolio as pf
from volmodel.universe import ASSET_IDS

VOLS = np.array([0.16, 0.21, 0.17, 0.23, 0.06, 0.12, 0.06, 0.07, 0.09, 0.20, 0.16, 0.18])


def true_cov(seed=0):
    """Two-factor covariance with realistic structure (incl. negative stock-bond corr)."""
    rng = np.random.default_rng(seed)
    g = np.array([1.0, 1.1, 0.9, 0.9, -0.2, -0.3, 0.0, 0.3, 0.7, 0.8, 0.0, 0.3])
    r = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.8, 0.8, 0.2, 0.1, 0.2, -0.1])
    L = np.column_stack([g, r]) + 0.05 * rng.standard_normal((12, 2))
    C = L @ L.T + np.diag(np.full(12, 0.6))
    d = np.sqrt(np.diag(C))
    C = C / np.outer(d, d)
    return C * np.outer(VOLS, VOLS)


def known_panel(Sigma, start="1990-01-01", end="2009-12-31", seed=1):
    """Panel of iid normal daily LOG excess returns with annualized covariance Sigma (cash = 0)."""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(start, end)
    X = rng.multivariate_normal(np.zeros(12), Sigma / 252, size=len(days))
    daily = pd.DataFrame(np.expm1(X), index=days, columns=ASSET_IDS)
    monthly = (1 + daily).groupby(pd.Grouper(freq="ME")).prod() - 1
    return Panel(daily=daily, monthly=monthly,
                 cash_daily=pd.Series(0.0, index=days, name="cash"),
                 cash_monthly=pd.Series(0.0, index=monthly.index, name="cash"),
                 tiers=pd.DataFrame("B", index=monthly.index, columns=ASSET_IDS),
                 predictors=pd.DataFrame(columns=PREDICTOR_COLUMNS))


def const_forecasts(origins, M, assets=ASSET_IDS):
    return {t: pd.DataFrame(M, index=assets, columns=assets) for t in origins}


def origins_of(panel, last_offset=12):
    m = panel.monthly.index
    return list(m[:-last_offset])


@pytest.fixture(scope="module")
def setup():
    S = true_cov()
    panel = known_panel(S)
    return S, panel, pf.RealizedCache(panel)


def _mean(res, model, port, metric):
    r = res[(res.model == model) & (res.portfolio == port) & (res.metric == metric)]
    return r.value.mean()


# ------------------------------------------------------------------ risk prediction
def test_true_matrix_unbiased_and_beats_wrong(setup):
    S, panel, cache = setup
    origins = origins_of(panel)
    d = np.sqrt(np.diag(S))
    C = S / np.outer(d, d)
    wrong_scale = S * 1.3 ** 2                                     # vols 30% too high
    wrong_corr = 0.2 * C + 0.8 * np.eye(12)                        # correlations shrunk to 0
    wrong_corr = wrong_corr * np.outer(d, d)
    res = pf.score_portfolios(panel, {"true": const_forecasts(origins, S),
                                      "scale": const_forecasts(origins, wrong_scale),
                                      "corr": const_forecasts(origins, wrong_corr)}, cache=cache)
    ports = [p for p in pf.PORTFOLIOS]
    assert set(res.portfolio.astype(str)) == set(ports)
    for p in ports:
        assert abs(_mean(res, "true", p, "log_ratio")) < 0.02, p
        for wrong in ("scale", "corr"):
            for metric in ("qlike", "sq_log_err"):
                assert _mean(res, "true", p, metric) < _mean(res, wrong, p, metric), (p, wrong, metric)
            r = res[(res.portfolio == p) & (res.metric == "log_ratio")]
            assert r[r.model == "true"].value.abs().mean() < r[r.model == wrong].value.abs().mean()
    # Engle-Colacito: true GMV realizes lower variance than GMV from wrong correlations
    for p in ("gmv", "gmv_long"):
        assert _mean(res, "true", p, "real_var") < _mean(res, "corr", p, "real_var")
    # vol targeting lands near 10% with the true matrix
    assert abs(_mean(res, "true", "erc_vt", "real_vol") - 0.10) < 0.003
    assert _mean(res, "true", "erc_vt", "vt_abs_err") < _mean(res, "scale", "erc_vt", "vt_abs_err")
    # constant forecasts -> zero turnover after the first origin
    t = res[(res.model == "true") & (res.metric == "turnover")]
    assert len(t) > 0 and t.value.max() < 1e-10


def test_gmv_true_beats_noisy_estimate(setup):
    S, panel, cache = setup
    origins = origins_of(panel)
    rng = np.random.default_rng(7)
    noisy = {}
    for t in origins:
        Y = rng.multivariate_normal(np.zeros(12), S / 252, size=60)
        noisy[t] = pd.DataFrame(Y.T @ Y * 252 / 60, index=ASSET_IDS, columns=ASSET_IDS)
    res = pf.score_portfolios(panel, {"true": const_forecasts(origins, S), "noisy": noisy},
                              cache=cache, n_random=5)
    for p in ("gmv", "gmv_long"):
        assert _mean(res, "true", p, "real_var") < _mean(res, "noisy", p, "real_var")
    # estimation error makes GMV look safer than it is: positive bias statistic
    assert _mean(res, "noisy", "gmv", "log_ratio") > 0.1
    assert _mean(res, "noisy", "gmv", "turnover") > _mean(res, "noisy", "invvol", "turnover")


def test_realized_cov_matches_target_definition(setup):
    S, panel, cache = setup
    t = pd.Timestamp("2000-06-30")
    rw = cache.get(t)
    X = panel.excess_daily.loc[(panel.excess_daily.index > t) & (panel.excess_daily.index <= rw.end)]
    assert rw.end == pd.Timestamp("2001-06-30") and rw.n_days == len(X)
    var = (X ** 2).sum() * 252 / len(X)
    S3 = X.rolling(3).sum().dropna()
    G = S3.T @ S3
    Cr = G / np.sqrt(np.outer(np.diag(G), np.diag(G)))
    expected = Cr.values * np.sqrt(np.outer(var, var))
    assert np.allclose(rw.cov, expected)
    assert np.allclose(rw.cov_direct, X.T @ X * 252 / len(X))
    # an annualized realized vol is near the truth
    assert np.allclose(np.sqrt(np.diag(rw.cov)), np.sqrt(np.diag(S)), rtol=0.15)
    # a window running past the panel end is not complete and is skipped
    assert not cache.get(pd.Timestamp("2009-06-30")).complete


# ------------------------------------------------------------------ optimizers
def _ill_conditioned(seed=3, cond=1e9):
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((12, 12)))
    ev = np.geomspace(1.0, 1.0 / cond, 12) * 0.04
    return (Q * ev) @ Q.T


@pytest.mark.parametrize("H,rtol", [(true_cov(), 1e-8), (true_cov(5), 1e-8), (_ill_conditioned(), 1e-6),
                                    (_ill_conditioned(4, 1e12), 1e-4)])
def test_erc_equalizes_risk_contributions(H, rtol):
    H = pf.regularize(H)                     # no-op unless cond > 1e10 (last case)
    w = pf.erc_weights(H)
    assert np.all(w > 0) and abs(w.sum() - 1) < 1e-12
    rc = pf.risk_contributions(w, H)
    assert np.allclose(rc, 1 / 12, rtol=rtol, atol=0)


@pytest.mark.parametrize("H", [true_cov(), true_cov(5), _ill_conditioned(), _ill_conditioned(4, 1e12)])
def test_long_only_gmv_constraints_and_optimality(H):
    from scipy.optimize import minimize
    H = pf.regularize(H)
    w = pf.gmv_long_only_weights(H)
    assert np.all(w >= 0) and abs(w.sum() - 1) < 1e-12
    # KKT: (Hw)_i = lambda on support, >= lambda off support
    g = H @ w
    lam = w @ g
    sup = w > 1e-12
    assert np.allclose(g[sup], lam, rtol=1e-6)
    assert np.all(g[~sup] >= lam * (1 - 1e-6))
    res = minimize(lambda x: x @ H @ x, np.full(12, 1 / 12), jac=lambda x: 2 * H @ x, method="SLSQP",
                   bounds=[(0, None)] * 12, constraints=[{"type": "eq", "fun": lambda x: x.sum() - 1}],
                   options={"ftol": 1e-16, "maxiter": 1000})
    assert w @ H @ w <= res.fun * (1 + 1e-6)
    # never better than unconstrained GMV
    wu = pf.gmv_weights(H)
    assert wu @ H @ wu <= w @ H @ w * (1 + 1e-9)


def test_long_only_equals_unconstrained_when_interior():
    H = np.diag(VOLS ** 2)                                         # GMV is long-only
    assert np.allclose(pf.gmv_long_only_weights(H), pf.gmv_weights(H))


def test_regularize_handles_singular_and_non_pd():
    H = true_cov()
    v = np.ones(12)
    singular = np.outer(v, v) * 0.02                               # rank one
    bad = H.copy()
    bad[0, 1] = bad[1, 0] = 2 * np.sqrt(H[0, 0] * H[1, 1])         # |corr| > 1
    for M in (singular, bad):
        W = pf.portfolio_weights(pd.DataFrame(M, index=ASSET_IDS, columns=ASSET_IDS), pd.Timestamp("2000-01-31"))
        assert np.isfinite(W.values).all()
        assert np.allclose(W.sum(axis=1), 1)
    with pytest.raises(ValueError):
        pf.regularize(np.full((3, 3), np.nan))


def test_portfolio_set_and_policies():
    t = pd.Timestamp("2000-01-31")
    W = pf.portfolio_weights(pd.DataFrame(true_cov(), index=ASSET_IDS, columns=ASSET_IDS), t, n_random=4)
    assert list(W.index[:5]) == ["gmv", "gmv_long", "erc", "invvol", "ew"]
    assert {"p6040", "p_balanced", "p_equity"} <= set(W.index)
    assert W.loc["p6040", "us_large"] == pytest.approx(0.6) and W.loc["p6040", "ust_int"] == pytest.approx(0.4)
    assert (W.filter(like="random", axis=0) >= 0).all().all()
    # policy needing a missing asset is skipped; balanced mix renormalized over present assets
    sub = ["us_large", "us_small", "ust_int", "gold"]
    W2 = pf.portfolio_weights(pd.DataFrame(true_cov()[np.ix_([0, 1, 4, 10], [0, 1, 4, 10])], index=sub,
                                           columns=sub), t, n_random=0)
    assert "p_equity" not in W2.index and W2.loc["p_balanced"].sum() == pytest.approx(1)


# ------------------------------------------------------------------ universes
def test_random_portfolios_identical_across_models_and_orders():
    t = pd.Timestamp("2003-03-31")
    a = pd.DataFrame(true_cov(), index=ASSET_IDS, columns=ASSET_IDS)
    rev = ASSET_IDS[::-1]
    b = pd.DataFrame(true_cov(9) * 2, index=ASSET_IDS, columns=ASSET_IDS).loc[rev, rev]
    Wa = pf.portfolio_weights(a, t).filter(like="random", axis=0)
    Wb = pf.portfolio_weights(b, t).filter(like="random", axis=0)[ASSET_IDS]
    assert np.allclose(Wa.values, Wb.values)
    assert not np.allclose(Wa.values, pf.portfolio_weights(a, t + pd.offsets.MonthEnd(1)).filter(
        like="random", axis=0).values)


def test_unequal_asset_sets_across_origins(setup):
    S, panel, cache = setup
    origins = origins_of(panel)
    small = ASSET_IDS[:6]
    fc = {}
    for t in origins:
        a = small if t < pd.Timestamp("1995-01-01") else ASSET_IDS
        ix = [ASSET_IDS.index(x) for x in a]
        fc[t] = pd.DataFrame(S[np.ix_(ix, ix)], index=a, columns=a)
    fc2 = {t: m.iloc[::-1, ::-1] * 1.1 for t, m in fc.items()}    # other model, reversed order
    res = pf.score_portfolios(panel, {"a": fc, "b": fc2}, cache=cache, n_random=6, random_detail=True)
    assert set(res.n_assets.unique()) == {6, 12}
    assert np.isfinite(res.value).all()
    # p_equity needs intl_dev (in first 6) -> present early; turnover at the switch counts entries
    tv = res[(res.model == "a") & (res.portfolio == "invvol") & (res.metric == "turnover")].set_index("origin").value
    assert tv.loc[pd.Timestamp("1995-01-31")] > 0.1
    # same random weights across models -> identical realized vol per random portfolio
    r = res[(res.metric == "real_vol") & res.portfolio.astype(str).str.startswith("random_")]
    piv = r.pivot_table(index=["origin", "portfolio"], columns="model", values="value", observed=True)
    assert np.allclose(piv["a"], piv["b"])


def test_asset_missing_daily_data_is_dropped(setup):
    S, _, _ = setup
    panel = known_panel(S, start="1995-01-01", end="2004-12-31", seed=2)
    daily = panel.daily.copy()
    daily.loc["2000-03-01":"2000-09-30", "em"] = np.nan              # 7-month hole
    daily.loc[:, "commodities"] = np.nan                            # monthly-only asset
    panel = Panel(daily=daily, monthly=panel.monthly, cash_daily=panel.cash_daily,
                  cash_monthly=panel.cash_monthly, tiers=panel.tiers, predictors=panel.predictors)
    origins = origins_of(panel)
    res = pf.score_portfolios(panel, {"true": const_forecasts(origins, S)}, n_random=5)
    by = res.drop_duplicates("origin").set_index("origin")
    assert by.loc[pd.Timestamp("1999-12-31"), "dropped"] == "em,commodities"
    assert by.loc[pd.Timestamp("1999-12-31"), "n_assets"] == 10
    assert by.loc[pd.Timestamp("2002-12-31"), "dropped"] == "commodities"
    assert by.loc[pd.Timestamp("2002-12-31"), "n_assets"] == 11
    assert np.isfinite(res.value).all()
    # the calibration test uses monthly data, so commodities stays in
    z = pf.annual_z_scores(panel, {"true": const_forecasts(origins, S)}, n_random=2)
    assert (z.n_assets == 12).all()


# ------------------------------------------------------------------ calibration
def monthly_panel(A, years=150, mu=None, seed=11):
    """Monthly iid normal log excess returns with annual covariance A (daily: tiny, empty)."""
    rng = np.random.default_rng(seed)
    months = pd.date_range("1800-01-31", periods=12 * years, freq="ME")
    mu = np.zeros(12) if mu is None else mu
    X = rng.multivariate_normal(mu / 12, A / 12, size=len(months))
    monthly = pd.DataFrame(np.expm1(X), index=months, columns=ASSET_IDS)
    days = pd.bdate_range("1800-01-01", periods=5)
    daily = pd.DataFrame(np.nan, index=days, columns=ASSET_IDS)
    return Panel(daily=daily, monthly=monthly, cash_daily=pd.Series(0.0, index=days),
                 cash_monthly=pd.Series(0.0, index=months), tiers=pd.DataFrame("C", index=months, columns=ASSET_IDS),
                 predictors=pd.DataFrame(columns=PREDICTOR_COLUMNS))


def test_annual_z_scores_calibrated_for_true_cov():
    A = true_cov()
    panel = monthly_panel(A)
    origins = list(panel.monthly.index[:-12])
    cache = pf.RealizedCache(panel)
    z = pf.annual_z_scores(panel, {"true": const_forecasts(origins, A),
                                   "too_wide": const_forecasts(origins, 2 * A)}, cache=cache, n_random=6)
    assert set(z.metric.astype(str)) == {"z", "ret", "pred_vol"}
    summ = pf.calibration_summary(z, n_boot=500).set_index(["model", "portfolio"])
    for p in ("p6040", "erc", "gmv", "ew", "random_pooled"):
        row = summ.loc[("true", p)]
        assert abs(row.var_z - 1) < 0.3, (p, row.var_z)
        assert row.ci_lo < 1 < row.ci_hi, p
        assert abs(row.var_z_nonoverlap - 1) < 0.3
        assert row.chi2_lo < 1 < row.chi2_hi
        assert 0.2 < row.share_gt1 < 0.45 and row.share_gt2 < 0.12
        bad = summ.loc[("too_wide", p)]
        assert bad.ci_hi < 1 and abs(bad.var_z - 0.5) < 0.15
    # the realized 12-month return is the sum of monthly log excess returns
    t = origins[0]
    rw = cache.get(t)
    expected = np.log1p(panel.monthly).iloc[1:13].sum().values
    assert np.allclose(rw.annual_ret, expected)


def test_demeaning_removes_premium():
    A = true_cov()
    mu = 0.5 * np.sqrt(np.diag(A))                                  # Sharpe 0.5 per asset
    panel = monthly_panel(A, mu=mu, seed=12)
    origins = list(panel.monthly.index[:-12])
    z = pf.annual_z_scores(panel, {"true": const_forecasts(origins, A)}, n_random=0)
    s = pf.calibration_summary(z, demean="sample", n_boot=200).set_index("portfolio")
    r = pf.calibration_summary(z, demean="none", n_boot=200).set_index("portfolio")
    assert abs(s.loc["ew", "var_z"] - 1) < 0.3
    assert r.loc["ew", "var_z"] > s.loc["ew", "var_z"] + 0.15


# ------------------------------------------------------------------ end-to-end / summary / io
def test_synthetic_panel_end_to_end_and_summary():
    panel = make_synthetic_panel(seed=0, start="1975-01-01", end="1998-12-31")
    xd = panel.excess_daily
    origins = pd.date_range("1987-12-31", "1997-12-31", freq="ME")
    fc = {}
    for t in origins:
        view = panel.asof(t)
        el = view.eligible()
        X = view.excess_daily[el].iloc[-252:].dropna(axis=1, thresh=200)
        a = list(X.columns)
        X0 = X.fillna(0.0)
        fc[t] = pd.DataFrame(X0.T @ X0 * 252 / len(X0), index=a, columns=a)
    ew = {t: pd.DataFrame(np.diag(np.diag(m.values)), index=m.index, columns=m.columns) for t, m in fc.items()}
    res = pf.score_portfolios(panel, {"sample": fc, "diag": ew}, n_random=8)
    assert set(res.columns) == {"origin", "model", "portfolio", "metric", "value", "n_assets", "dropped"}
    assert res.origin.max() <= pd.Timestamp("1997-12-31")
    assert res.n_assets.min() < 12 and res.n_assets.max() == 12
    piv = res.query("portfolio == 'gmv' and metric == 'real_var'").pivot(index="origin", columns="model",
                                                                        values="value")
    assert piv.notna().all().all() and len(piv) == 121
    summ = pf.summarize(res, eras="decade")
    assert {"model", "portfolio", "metric", "era", "mean", "n_origins"} == set(summ.columns)
    assert set(summ.era) == {"all", "1980s", "1990s"}
    s = summ[summ.era == "all"].set_index(["model", "portfolio", "metric"])["mean"]
    # ignoring correlations misstates risk of diversified portfolios
    assert s[("sample", "p_balanced", "qlike")] < s[("diag", "p_balanced", "qlike")]
    assert s[("sample", "gmv", "real_var")] < s[("diag", "gmv", "real_var")]
    eras = pf.summarize(res, eras={"early": (None, "1992-12-31"), "late": ("1993-01-31", None)})
    n = eras[(eras.metric == "qlike") & (eras.portfolio == "ew") & (eras.model == "sample")].set_index("era").n_origins
    assert n["early"] + n["late"] == n["all"]


def test_cov_forecasts_from_long_roundtrip(tmp_path):
    t = pd.Timestamp("2001-01-31")
    a = ["us_large", "ust_int", "gold"]
    M = true_cov()[np.ix_([0, 4, 10], [0, 4, 10])]
    rows = [(t, a[i], a[j], M[i, j]) for i in range(3) for j in range(i, 3)]      # upper triangle only
    df = pd.DataFrame(rows, columns=["origin", "asset_i", "asset_j", "cov"])
    (tmp_path / "m.parquet").parent.mkdir(exist_ok=True)
    df.to_parquet(tmp_path / "m.parquet")
    out = pf.load_cov_forecasts(["m"], root=tmp_path)["m"][t]
    assert list(out.index) == a and np.allclose(out.values, M)
