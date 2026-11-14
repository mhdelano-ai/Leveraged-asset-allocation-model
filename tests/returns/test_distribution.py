"""dist_hm_af: shape helpers (ported from assetpred's test_dist), residuals,
the pooled shape's timing and tail floor, portfolio consistency with
eval.portfolio, truncation invariance of the whole history, calibration on
simulated panels, and scoring."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from scipy.signal import lfilter

from assetforecast.eval.portfolio import POLICIES, policy_weights
from assetforecast.returns import config, schema
from assetforecast.returns import distribution as D
from assetforecast.returns import expected as E

IDS = ["us_large", "us_small", "intl_dev", "ust_int"]


# --------------------------------------------------------------------------
# shape helpers (ported)
def test_weighted_quantile_matches_unweighted():
    x = np.random.default_rng(1).standard_normal(5001)
    q = D.weighted_quantile(x, np.ones_like(x), np.array([0.05, 0.5, 0.95]))
    np.testing.assert_allclose(q, np.quantile(x, [0.05, 0.5, 0.95]), atol=0.01)


def test_centered_quantiles_removes_mean_keeps_scale():
    rng = np.random.default_rng(2)
    z = 0.3 + 2.0 * rng.standard_normal(20001)
    q = D.centered_quantiles(z)
    np.testing.assert_allclose(q, np.quantile(z - z.mean(), D.TAUS))
    assert abs(q[4]) < 0.05 and q[-1] - q[0] == pytest.approx(2 * 2 * 2.326, rel=0.05)
    qw = D.centered_quantiles(np.append(z, np.nan), np.ones(len(z) + 1))
    np.testing.assert_allclose(qw, q, atol=0.02)


def test_grid_weights_and_mean():
    assert D.GRID_W.sum() == pytest.approx(1.0)
    assert D.grid_mean(stats.norm.ppf(D.TAUS)) == pytest.approx(0.0, abs=1e-12)
    assert D.grid_mean(np.full(9, 3.0)) == pytest.approx(3.0)


# --------------------------------------------------------------------------
# synthetic panel and covariance history (point-in-time by construction)
def _panel(T=360, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("1960-01-31", periods=T, freq="ME")
    f = 0.035 * stats.t.rvs(5, size=T, random_state=rng) / np.sqrt(5 / 3)
    beta = np.array([1.0, 1.2, 0.8, -0.1])
    e = np.array([0.015, 0.025, 0.02, 0.012]) * rng.standard_normal((T, 4))
    x = 0.004 + beta * f[:, None] + e
    x[:80, 2] = np.nan                       # intl_dev enters later
    xr = pd.DataFrame(x, index=idx, columns=IDS)
    ew = (xr ** 2).ewm(alpha=0.05, adjust=False, ignore_na=True).mean()
    cc = xr.expanding(24).corr()
    cov = {}
    for t in idx[24:]:
        ok = xr.loc[:t].notna().sum() >= 24
        a = list(ok[ok].index)
        v = np.sqrt(12 * ew.loc[t, a].to_numpy())
        R = cc.loc[t].loc[a, a].to_numpy()
        cov[t] = pd.DataFrame(R * np.outer(v, v), index=a, columns=a)
    return xr, cov


@pytest.fixture(scope="module")
def synth():
    xr, cov = _panel()
    ex = E.history(xr)
    return xr, cov, ex, D.build(xr, cov, expected=ex)


def test_residuals_definition(synth):
    xr, cov, ex, _ = synth
    t = xr.index[200]
    f = D.residual_frame(xr, cov, [t], ex)
    y = schema.make_targets(xr, 12).loc[t]
    e = ex[ex.origin == t].set_index("asset")
    for a in f.loc[t].index:
        s2 = cov[t].at[a, a] + e.at[a, "se_12m"] ** 2
        assert f.at[(t, a), "z"] == pytest.approx((y[a] - e.at[a, "mean_12m"]) / np.sqrt(s2))
    z = D.standardized_residuals(xr, cov, None, ex)
    assert z.index.names == ["origin", "asset"] and np.isfinite(z).all()
    # no residual for an origin whose window is not closed
    assert z.index.get_level_values(0).max() == xr.index[-13]


def test_portfolios_follow_eval_portfolio(synth):
    xr, cov, ex, res = synth
    t = xr.index[100]
    f = D.portfolio_frame(xr, cov, [t], None, ex)
    e = ex[ex.origin == t].set_index("asset")
    U = [a for a in cov[t].index if a in e.index]
    assert "p_balanced" in f.loc[t].index and "p6040" in f.loc[t].index
    for p in f.loc[t].index:
        w = policy_weights(U, POLICIES[p])
        assert json.loads(f.at[(t, p), "weights"]) == pytest.approx(
            {a: round(x, 6) for a, x in zip(U, w) if x != 0})
        assert f.at[(t, p), "y"] == pytest.approx(w @ schema.make_targets(xr, 12).loc[t, U].to_numpy())
        assert f.at[(t, p), "mean"] == pytest.approx(w @ e.loc[U, "mean_12m"].to_numpy())
        S = cov[t].loc[U, U].to_numpy()
        assert f.at[(t, p), "scale"] == pytest.approx(np.sqrt(w @ S @ w))
        assert f.at[(t, p), "se"] == pytest.approx(np.abs(w) @ e.loc[U, "se_12m"].to_numpy())
    # p_equity needs intl_dev, which has no eligible mean until month 140
    early = D.portfolio_frame(xr, cov, [xr.index[100]], None, ex)
    assert "p_equity" not in early.loc[xr.index[100]].index


def test_shape_uses_only_closed_windows():
    rng = np.random.default_rng(3)
    dates = pd.date_range("1950-01-31", periods=200, freq="ME")
    idx = pd.MultiIndex.from_product([dates, ["a", "b"]], names=["origin", "asset"])
    z = pd.Series(rng.standard_normal(len(idx)), index=idx)
    t = dates[-1]
    base = D.pooled_shape(z, t)
    z2 = z.copy()
    z2.loc[(dates[-12], "a")] = 50.0          # s + 12 > t: window still open at t
    np.testing.assert_array_equal(D.pooled_shape(z2, t), base)
    z3 = z.copy()
    z3.loc[(dates[-13], "a")] = 50.0          # s + 12 = t: closed, must count
    assert D.pooled_shape(z3, t)[-1] > base[-1]
    info = D.shape_info(z, t)
    assert info["last_origin"] == dates[-13] and info["n_dates"] == 188


def test_shape_minimum_history_and_recentring():
    rng = np.random.default_rng(4)
    dates = pd.date_range("1950-01-31", periods=D.MIN_RESID_DATES + 11, freq="ME")
    z = pd.Series(rng.standard_normal(len(dates)),
                  index=pd.MultiIndex.from_arrays([dates, ["a"] * len(dates)]))
    assert np.isnan(D.pooled_shape(z, dates[-1])).all()          # 119 closed dates
    t = dates[-1] + pd.offsets.MonthEnd(1)
    assert np.isfinite(D.pooled_shape(z, t)).all()               # 120
    np.testing.assert_allclose(D.pooled_shape(z + 0.7, t), D.pooled_shape(z, t), atol=1e-12)


def test_tail_floor():
    rng = np.random.default_rng(5)
    dates = pd.date_range("1900-01-31", periods=600, freq="ME")
    idx = pd.MultiIndex.from_product([dates, ["a", "b", "c"]])
    t = dates[-1]
    # thin tails (uniform): every floored quantile sits on the Gaussian with the same sd
    u = pd.Series(rng.uniform(-1, 1, len(idx)), index=idx)
    s = D.shape_info(u, t, diagnostics=False)
    for c in D.TAIL_FLOOR:
        k = list(schema.QUANTILE_COLUMNS).index(c)
        assert s["q"][k] == pytest.approx(s["q_gauss"][k])
        assert abs(s["q"][k]) >= abs(s["q_raw"][k])
    np.testing.assert_array_equal(s["q"][2:7], s["q_raw"][2:7])
    # heavy tails (t3): the empirical tails are wider and are kept
    h = pd.Series(stats.t.rvs(3, size=len(idx), random_state=rng), index=idx)
    s = D.shape_info(h, t, diagnostics=False)
    assert s["q"][0] == s["q_raw"][0] < s["q_gauss"][0]
    assert np.all(np.diff(s["q"]) >= 0)


def test_history_contract_and_distribution_at(synth):
    xr, cov, ex, res = synth
    a, p = res["assets"], res["portfolios"]
    for f in (a, p):
        assert list(f.columns) == schema.FORECAST_COLUMNS
        Q = f[schema.QUANTILE_COLUMNS].to_numpy()
        assert np.isfinite(Q).all() and (np.diff(Q, axis=1) >= 0).all()
        assert (f["horizon"] == 12).all()
    assert set(a["model"]) == {D.MODEL} and set(p["model"]) == {D.MODEL, D.MODEL_ASSETSHAPE}
    # mean is the historical mean
    m = a.merge(ex, left_on=["date", "asset"], right_on=["origin", "asset"])
    np.testing.assert_allclose(m["mean"], m["mean_12m"], rtol=1e-12)
    # distribution_at reproduces the history row at an origin
    t = a["date"].max()
    za = res["residuals"].query("kind == 'assets'").set_index(["origin", "name"])["z"].dropna()
    zp = res["residuals"].query("kind == 'portfolios'").set_index(["origin", "name"])["z"].dropna()
    shapes = {"assets": D.shape_info(za, t), "portfolios": D.shape_info(zp, t)}
    ad, pd_ = D.distribution_at(t, xr, cov[t], shapes)
    h = a[a.date == t].set_index("asset")
    np.testing.assert_allclose(ad.set_index("asset").loc[h.index, schema.QUANTILE_COLUMNS],
                               h[schema.QUANTILE_COLUMNS], rtol=1e-12)
    assert list(ad.columns) == ["asset", "mean", "scale", "sd", *schema.QUANTILE_COLUMNS]
    assert list(pd_.columns) == ["portfolio", "mean", "scale", "sd", *schema.QUANTILE_COLUMNS, "weights"]
    np.testing.assert_allclose(ad.set_index("asset")["scale"] ** 2,
                               np.diag(cov[t].loc[ad.asset, ad.asset]), rtol=1e-12)
    hp = p[(p.date == t) & (p.model == D.MODEL)].set_index("asset")
    np.testing.assert_allclose(pd_.set_index("portfolio").loc[hp.index, "q05"], hp["q05"], rtol=1e-12)
    # sensitivity: no portfolio shape -> the asset shape
    _, ps = D.distribution_at(t, xr, cov[t], {"assets": shapes["assets"]})
    hs = p[(p.date == t) & (p.model == D.MODEL_ASSETSHAPE)].set_index("asset")
    np.testing.assert_allclose(ps.set_index("portfolio").loc[hs.index, "q05"], hs["q05"], rtol=1e-12)


@pytest.mark.slow
def test_history_is_truncation_invariant(synth):
    """Every forecast at an origin <= T0 is the same whether the data end at T0 or later."""
    xr, cov, ex, res = synth
    T0 = xr.index[290]
    short = D.build(xr.loc[:T0], {k: v for k, v in cov.items() if k <= T0})
    for key in ("assets", "portfolios"):
        full = res[key][res[key]["date"] <= T0].reset_index(drop=True)
        pd.testing.assert_frame_equal(full, short[key].reset_index(drop=True), check_exact=False,
                                      rtol=1e-12, atol=1e-14)


# --------------------------------------------------------------------------
# calibration on simulated panels (the design study, reduced)
def _sim(rng, T=600, starts=(0, 0, 0, 240, 360, 420)):
    N = len(starts)
    B = T + 300
    lv = lfilter([1], [1, -0.97], rng.standard_normal(B) * 0.22 * np.sqrt(1 - .97 ** 2))
    f = np.exp(lv) * 0.04 * stats.t.rvs(5, size=B, random_state=rng) / np.sqrt(5 / 3)
    f += np.where(rng.random(B) < 0.01, -0.12 * np.exp(lv), 0.0)      # crashes: negative skew
    beta = np.array([1.0, 1.2, 0.9, 1.1, 0.3, 0.0])
    e = (np.array([.02, .03, .025, .035, .015, .02]) * np.exp(lv)[:, None]
         * stats.t.rvs(5, size=(B, N), random_state=rng) / np.sqrt(5 / 3))
    x = lfilter([1], [1, -0.08], beta * f[:, None] + e, axis=0)[300:] + 0.004
    for j, s in enumerate(starts):
        x[:s, j] = np.nan
    xr = pd.DataFrame(x, index=pd.date_range("1950-01-31", periods=T, freq="ME"),
                      columns=[f"a{j}" for j in range(N)])
    ew = (xr ** 2).ewm(alpha=0.03, adjust=False, ignore_na=True).mean()
    ok = xr.notna().cumsum() >= 24
    cov = {}
    for t in xr.index:
        v = 12 * ew.loc[t][ok.loc[t]]
        if len(v):
            cov[t] = pd.DataFrame(np.diag(v.to_numpy()), index=v.index, columns=v.index)
    return xr, cov


@pytest.mark.slow
def test_calibrated_on_simulated_panels():
    """Feasible noisy EWMA scale, t(5) shocks, crashes, AR(1) 0.08, staggered entries:
    the 90% interval covers about 90% and the 1% tails are not badly exceeded.
    (The 40-replication study in the WP5 notes: 90.2% coverage, 1.12% / 1.23% beyond
    q01 / q99.) Single replications are dominated by one or two crash episodes
    (lower-tail rates of 0% to 6% per replication), hence six replications and
    loose tail bounds: this guards against a broken design, it does not measure it.)"""
    rng = np.random.default_rng(11)
    hits = []
    for _ in range(6):
        xr, cov = _sim(rng)
        res = D.build(xr, cov, policies={})
        f = res["assets"]
        y = schema.make_targets(xr, 12).stack()
        y.index = y.index.set_names(["date", "asset"])
        f = f.join(y.rename("y"), on=["date", "asset"]).dropna(subset=["y"])
        f = f[f.date >= f.date.max() - pd.offsets.MonthEnd(300)]
        hits.append(pd.DataFrame({"in90": (f.y >= f.q05) & (f.y <= f.q95), "lo1": f.y < f.q01,
                                  "hi1": f.y > f.q99}))
    h = pd.concat(hits)
    assert 0.85 <= h["in90"].mean() <= 0.95
    assert h["lo1"].mean() < 0.035 and h["hi1"].mean() < 0.035


@pytest.mark.slow
def test_score_runs_with_baselines(synth):
    xr, cov, ex, res = synth
    a = res["assets"]

    def gauss(model, widen):
        g = a[["date", "asset", "mean", "sd"]].copy()
        g["sd"] = g["sd"] * widen
        for c, zq in zip(schema.QUANTILE_COLUMNS, stats.norm.ppf(D.TAUS)):
            g[c] = g["mean"] + g["sd"] * zq
        g.insert(0, "horizon", 12)
        g.insert(0, "model", model)
        return g[schema.FORECAST_COLUMNS]

    base = pd.concat([gauss("hist_mean", 1.1), gauss("vol_ewma", 0.95)], ignore_index=True)
    sc = D.score(a, base, xr, res["portfolios"], res["residuals"], mt_reps=49)
    from assetforecast.returns.evaluation.scoring import SCORE_COLUMNS
    assert list(sc.columns) == [*SCORE_COLUMNS, "target"]
    assert sc.attrs["label"] == config.REPLICATION_LABEL
    s = sc[(sc.model == D.MODEL) & (sc.asset == "pooled") & (sc.period == "full")]
    assert {"crps_skill", "cover_90", "calib_p", "kupiec_p_q01"} <= set(s.metric)
    assert set(s[s.metric == "crps_skill"].benchmark) >= {"hist_mean", "vol_ewma"}
    sp = sc[sc.target == "portfolios"]
    assert set(sp.model) >= {D.MODEL, D.MODEL_ASSETSHAPE} and "p6040" in set(sp.asset)


@pytest.mark.realdata
def test_real_latest_origin():
    from assetforecast import paths
    p = paths.RESULTS_RETURNS / "distribution" / "history.parquet"
    if not p.exists():
        pytest.skip("distribution history not built")
    f = pd.read_parquet(p)
    last = f[f.date == f.date.max()]
    assert set(last.asset) == set(config.ASSETS)
    e = E.load_history()
    e = e[e.origin == f.date.max()].set_index("asset")
    np.testing.assert_allclose(last.set_index("asset").loc[e.index, "mean"], e["mean_12m"], rtol=1e-12)
