"""Strategy test: rules, accounting, timing (no look-ahead), and inference."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from assetpred import config
from assetpred.evaluation import strategy as stg
from assetpred.evaluation import strategy_rules as rules
from assetpred.evaluation import strategy_stats as st

A = stg.ASSETS
N = len(A)


def _cov(rng, n=N, scale=0.002):
    X = rng.standard_normal((n, n))
    C = X @ X.T / n + 0.5 * np.eye(n)
    d = np.sqrt(np.diag(C))
    vols = rng.uniform(0.01, 0.06, n)
    return C / np.outer(d, d) * np.outer(vols, vols)


def synthetic_inputs(seed=0, T=150, horizons=(1, 3), late=("gold", "tips"), drift=0.003):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("1995-01-31", periods=T + 15, freq="ME")
    C = _cov(rng)
    L = np.linalg.cholesky(C)
    x = rng.standard_normal((len(dates), N)) @ L.T + drift
    rf = pd.Series(0.002, index=dates, name="cash")
    R = pd.DataFrame(np.expm1(x + np.log1p(0.002)), index=dates, columns=A)
    origins, mean, cov = {}, {}, {}
    for h in horizons:
        o = dates[:T]
        origins[h] = o
        for m in ("hist_mean", "model"):
            mu = np.full((T, N), 0.003 * h) + (rng.normal(0, 0.004 * math.sqrt(h), (T, N)) if m == "model" else 0)
            for a in late:
                mu[: T // 2, A.index(a)] = np.nan
            mean[(m, h)] = mu
        for c in ("dyn", "sample"):
            k = np.exp(rng.normal(0, 0.2, T)) if c == "dyn" else np.ones(T)
            arr = np.repeat((C * h)[None], T, 0) * k[:, None, None]
            for a in late:
                i = A.index(a)
                arr[: T // 2, i, :] = np.nan
                arr[: T // 2, :, i] = np.nan
            cov[(c, h)] = arr
    return stg.Inputs(R, rf, origins, mean, cov)


# ----------------------------------------------------------------- rules
@pytest.mark.parametrize("seed", range(5))
def test_mean_variance_constraints_and_optimality(seed):
    rng = np.random.default_rng(seed)
    S = _cov(rng)
    mu = rng.normal(0.004, 0.006, N)
    s = rules.RuleSettings()
    w = rules.mean_variance(mu, S, s)
    assert (w >= -1e-9).all() and (w <= s.max_weight + 1e-9).all() and w.sum() <= s.max_gross + 1e-7
    u = lambda v: mu @ v - 0.5 * s.gamma * v @ S @ v
    # no random feasible point, nor a small feasible perturbation, does better
    for _ in range(2000):
        v = rng.uniform(0, 1, N) * (rng.random(N) < 0.5)
        if v.sum() > s.max_gross:
            v *= s.max_gross / v.sum()
        assert u(v) <= u(w) + 1e-9
        p = np.clip(w + rng.normal(0, 0.01, N), 0, s.max_weight)
        if p.sum() <= s.max_gross:
            assert u(p) <= u(w) + 1e-9


@pytest.mark.parametrize("seed", range(5))
def test_mv_voltarget_meets_risk_budget(seed):
    rng = np.random.default_rng(seed)
    S = _cov(rng)
    mu = rng.normal(0.004, 0.006, N)
    s = rules.RuleSettings()
    w = rules.mean_variance_voltarget(mu, S, s)
    v2 = s.vol_target ** 2 / 12
    assert w @ S @ w <= v2 * (1 + 1e-6)
    assert (w >= -1e-9).all() and (w <= s.max_weight + 1e-9).all() and w.sum() <= s.max_gross + 1e-7
    for _ in range(2000):
        v = rng.uniform(0, 1, N) * (rng.random(N) < 0.5)
        v *= min(1.0, s.max_gross / max(v.sum(), 1e-12), math.sqrt(v2 / max(v @ S @ v, 1e-18)))
        assert mu @ v <= mu @ w + 1e-9


def test_rules_ignore_ineligible_and_handle_negative_means():
    rng = np.random.default_rng(1)
    S = _cov(rng)
    mu = np.full(N, 0.004)
    mu[[2, 5]] = np.nan
    S[2, :] = S[:, 2] = np.nan
    for f in (rules.mean_variance, rules.mean_variance_voltarget, rules.timing, rules.vol_target_risk_parity):
        w = f(mu, S)
        assert w[2] == 0 and w[5] == 0 and np.isfinite(w).all()
    neg = np.full(N, -0.001)
    for f in (rules.mean_variance, rules.mean_variance_voltarget, rules.timing):
        assert (f(neg, _cov(rng)) == 0).all()


def test_timing_and_vol_target_formulas():
    s = rules.RuleSettings()
    S = np.diag(np.full(N, 0.03 ** 2))
    mu = np.linspace(-0.002, 0.01, N)
    w = rules.timing(mu, S, s)
    np.testing.assert_allclose(w, np.clip(mu / (s.gamma * 0.03 ** 2), 0, s.timing_cap) / N)
    S = 0.05 ** 2 * (0.5 * np.ones((N, N)) + 0.5 * np.eye(N))
    w = rules.vol_target_risk_parity(np.zeros(N), S, s)
    assert w.sum() < s.max_gross
    assert math.isclose(math.sqrt(w @ S @ w * 12), s.vol_target, rel_tol=1e-9)
    assert np.allclose(w, w[0])


# ----------------------------------------------------------------- accounting
def test_simulate_turnover_costs_and_alignment():
    f = pd.date_range("2000-01-31", periods=3, freq="ME")
    months = pd.date_range("2000-01-31", periods=4, freq="ME")
    R = pd.DataFrame(0.0, index=months, columns=A)
    R.loc["2000-02-29", "us_eq"] = 0.10
    R.loc["2000-03-31", "us_eq"] = -0.05
    rf = pd.Series(0.0, index=months)
    held = np.zeros((3, N))
    i = A.index("us_eq")
    j = A.index("ust_int")
    held[:, i] = [1.0, 1.0, 0.5]
    held[:, j] = [0.0, 0.5, 0.5]
    out = stg.simulate(held, f, R, rf, cost_bp={a: 10 for a in A}, spread=0.012)
    # the weight formed at end-Jan earns February's return
    assert out.date.tolist() == list(months[1:])
    np.testing.assert_allclose(out.gross, [0.10, -0.05, 0.0])
    # turnover: build 1.0; then drifted us_eq weight 1.1/1.1 = 1 -> +0.5 ust_int;
    # then drift: us_eq 0.95/0.95... portfolio return -0.05: us_eq 0.95/0.95=1.0, ust_int 0.5/0.95
    to2 = abs(0.5 - 1.0) + abs(0.5 - 0.5 / 0.95)
    np.testing.assert_allclose(out.turnover, [1.0, 0.5, to2])
    np.testing.assert_allclose(out.cost, 0.001 * out.turnover)
    np.testing.assert_allclose(out.financing, [0.0, 0.001 * 0.5, 0.0])
    np.testing.assert_allclose(out.net, out.gross - out.cost - out.financing)


def test_held_weights_staggered_average():
    rng = np.random.default_rng(0)
    W = rng.random((10, N))
    H, valid = stg.held_weights(W, 3)
    assert not valid[:2].any() and valid[2:].all()
    for k in range(2, 10):
        np.testing.assert_allclose(H[k], W[k - 2:k + 1].mean(0))


def test_eligible_set_grows():
    inp = synthetic_inputs()
    spec = stg.Spec("mv", "model", "dyn", "all", 1)
    W = stg.target_weights(spec, inp)
    g = A.index("gold")
    assert (W[:75, g] == 0).all() and (W[75:, g] > 0).any()
    W = stg.target_weights(stg.Spec("mv", "model", "dyn", "ex_hy", 1), inp)
    assert (W[:, A.index("hy_credit")] == 0).all()


# ----------------------------------------------------------------- no look-ahead
@pytest.mark.parametrize("rule,h", [("mv", 1), ("mv_vt", 1), ("timing", 3), ("vt_rp", 1), ("mv", 3)])
def test_no_weight_depends_on_later_data(rule, h):
    inp = synthetic_inputs(horizons=(h,))
    mean_model = "none" if rule == "vt_rp" else "model"
    spec = stg.Spec(rule, mean_model, "dyn", "all", h)
    ret0, w0 = stg.run_spec(spec, inp)
    cut = inp.origins[h][90]
    # scramble everything dated after the cut: returns, forecasts and covariances
    rng = np.random.default_rng(99)
    inp2 = synthetic_inputs(horizons=(h,))
    late_r = inp2.returns.index > cut
    inp2.returns.loc[late_r] = rng.normal(0, 0.2, (late_r.sum(), N))
    late_o = inp2.origins[h] > cut
    for k in list(inp2.mean):
        inp2.mean[k][late_o] = rng.normal(0, 0.05, (late_o.sum(), N))
    for k in list(inp2.cov):
        inp2.cov[k][late_o] *= rng.uniform(0.2, 5.0, late_o.sum())[:, None, None]
    ret1, w1 = stg.run_spec(spec, inp2)
    pd.testing.assert_frame_equal(w0.loc[:cut], w1.loc[:cut])
    assert not w0.loc[cut:].iloc[1:].equals(w1.loc[cut:].iloc[1:])
    # P&L through the month of the cut is unchanged too; the next month uses the cut's weights
    a = ret0[ret0.date <= cut].reset_index(drop=True)
    b = ret1[ret1.date <= cut].reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)


def test_oracle_forecast_profits_and_stale_one_does_not():
    """Forecast at t = realized excess return of t+1 must earn a large Sharpe
    ratio; forecast at t = return of month t (iid zero-mean data, so no
    information) must not."""
    inp = synthetic_inputs(seed=3, T=300, horizons=(1,), late=(), drift=0.0)
    o = inp.origins[1]
    xr = np.log1p(inp.returns).sub(np.log1p(inp.cash), axis=0)
    inp.mean[("model", 1)] = xr.shift(-1).reindex(o).to_numpy()
    inp.mean[("stale", 1)] = xr.reindex(o).to_numpy()
    ora, _ = stg.run_spec(stg.Spec("timing", "model", "sample", "all", 1), inp)
    sta, _ = stg.run_spec(stg.Spec("timing", "stale", "sample", "all", 1), inp)
    assert st.sharpe(ora.gross) > 2.0
    assert abs(st.sharpe(sta.gross)) < 1.0


# ----------------------------------------------------------------- inference
def test_diff_test_identical_series_is_zero():
    rng = np.random.default_rng(0)
    a = rng.normal(0.005, 0.03, 200)
    idx = st.stationary_bootstrap_indices(200, 199, 6, rng)
    d = st.diff_test(a, a.copy(), 5.0, 1, idx)
    assert d["d_sharpe"] == 0 and d["d_cer"] == 0


def test_diff_test_size_under_null():
    rng = np.random.default_rng(1)
    rej_s = rej_c = 0
    n_sim = 300
    for _ in range(n_sim):
        z = rng.standard_t(5, (300, 2)) * 0.03
        a = 0.004 + z[:, 0]
        b = 0.004 + 0.6 * z[:, 0] + 0.8 * z[:, 1]
        d = st.diff_test(a, b, 5.0, 1, None)
        rej_s += d["d_sharpe_p"] < 0.05
        # equal variances, so equal CER under the null
        rej_c += d["d_cer_p"] < 0.05
    assert 0.01 <= rej_s / n_sim <= 0.10
    assert 0.01 <= rej_c / n_sim <= 0.10


def test_diff_test_detects_large_difference():
    rng = np.random.default_rng(2)
    z = rng.normal(0, 0.03, (300, 2))
    d = st.diff_test(0.02 + z[:, 0], z[:, 1], 5.0, 1, None)
    assert d["d_sharpe"] > 0 and d["d_sharpe_p"] < 0.01 and d["d_cer_p"] < 0.01


def test_stationary_bootstrap_indices():
    rng = np.random.default_rng(0)
    idx = st.stationary_bootstrap_indices(100, 50, 5.0, rng)
    assert idx.shape == (50, 100) and idx.min() >= 0 and idx.max() < 100
    steps = (np.diff(idx, axis=1) % 100) == 1
    assert 0.7 < steps.mean() < 0.9      # mean block length 5 -> continue w.p. 0.8


def test_breakeven_cost_zeroes_cer_difference():
    rng = np.random.default_rng(3)
    ga = rng.normal(0.006, 0.03, 240)
    gb = rng.normal(0.004, 0.03, 240)
    ta, tb = rng.uniform(0.5, 1.5, 240), rng.uniform(0.0, 0.3, 240)
    fa = fb = np.zeros(240)
    be = st.breakeven(ga, gb, ta, tb, fa, fb, 5.0)
    if np.isfinite(be) and be > 0:
        c = be * 1e-4
        assert abs(st.cer(ga - c * ta, 5.0) - st.cer(gb - c * tb, 5.0)) < 1e-10
    assert st.breakeven(gb - 1, gb, ta, tb, fa, fb, 5.0) == 0.0


# ----------------------------------------------------------------- end to end on dev inputs
DEV = config.RESULTS / "dev"


@pytest.mark.skipif(not (DEV / "forecasts" / "hist_mean" / "h01.parquet").exists()
                    or not (DEV / "covariances" / "sample" / "h01.parquet").exists(),
                    reason="dev forecasts not present")
def test_dev_run_end_to_end(tmp_path):
    res = stg.run(DEV, ["ridge_z"], "vc_daily_ewma_m_lwb", ["ewma_cov"], [1], dev=True, jobs=1,
                  out_dir=tmp_path)
    for f in ("metrics.parquet", "summary.csv", "summary_by_period.csv", "returns.parquet", "weights.parquet", "meta.json"):
        assert (tmp_path / f).exists()
    r = res["returns"]
    assert r.date.max() <= pd.Timestamp(config.DEV_END)
    assert (r.formation < r.date).all()
    m = res["metrics"]
    assert {"mean", "risk", "vs_static"} <= set(m.family.astype(str))
    assert set(m.columns) >= {"strategy", "mean_model", "cov_model", "period", "basis", "metric", "value"}
    w = res["weights"]
    # gold and TIPS enter late: no weight before their first forecast origin
    assert w[(w.asset == "gold") & (w.formation < "2009-11-30")].empty
    assert w[(w.asset == "tips") & (w.formation < "2004-01-31")].empty


# ----------------------------------------------------------------- holdout periods (synthetic dates only)
@pytest.mark.parametrize("h,first_formation", [(1, "2015-01-31"), (3, "2015-03-31"), (12, "2015-12-31")])
def test_holdout_masks_assign_straddling_months_to_neither(h, first_formation):
    formation = pd.date_range("2013-01-31", "2017-12-31", freq="ME")
    dates = formation + pd.offsets.MonthEnd(1)
    pre, ho = stg.holdout_masks(dates, formation, h)
    assert not (pre & ho).any()
    assert dates[pre].max() == pd.Timestamp(config.DEV_END)
    assert formation[ho].min() == pd.Timestamp(first_formation)
    if h == 1:
        assert dates[ho].min() == pd.Timestamp("2015-02-28")
    # oldest sleeve of every holdout position is a holdout origin
    assert (formation[ho] - pd.offsets.MonthEnd(h - 1) >= pd.Timestamp(config.HOLDOUT_START)).all()
    # straddling months (h - 1 of them, plus 2015-01 at h = 1) are in neither
    assert (~pre & ~ho).sum() == h


def _synthetic_full_run(h):
    """Synthetic panel spanning the holdout boundary (no real post-2014 data)."""
    inp = synthetic_inputs(seed=5, T=120, horizons=(h,))
    shift = pd.DatetimeIndex(pd.date_range("2008-01-31", periods=len(inp.returns), freq="ME"))
    inp.returns.index = shift
    inp.cash.index = shift
    inp.origins[h] = shift[:120]
    return inp


@pytest.mark.parametrize("h", [1, 3])
def test_full_run_periods_metrics_and_tests(h):
    inp = _synthetic_full_run(h)
    s = rules.RuleSettings()
    a, _ = stg.run_spec(stg.Spec("mv_vt", "model", "dyn", "all", h), inp)
    b, _ = stg.run_spec(stg.Spec("mv_vt", "hist_mean", "dyn", "all", h), inp)
    lv = pd.DataFrame(stg.level_metrics(a, s, h, holdout=True), columns=["period", "basis", "metric", "value"])
    assert {"full", "pre_holdout", "holdout"} <= set(lv.period)
    # holdout metrics are computed on exactly the holdout months
    pre, ho = stg.holdout_masks(a.date, a.formation, h)
    n = lv[(lv.period == "holdout") & (lv.basis == "net") & (lv.metric == "n_months")].value.item()
    assert n == ho.sum()
    sr = lv[(lv.period == "holdout") & (lv.basis == "net") & (lv.metric == "sharpe")].value.item()
    assert math.isclose(sr, st.sharpe(a.net[ho]))
    cmp = pd.DataFrame(stg.compare(a, b, h, s, 1, {}, holdout=True), columns=["period", "basis", "metric", "value"])
    for per in ("pre_holdout", "holdout"):
        got = set(cmp[(cmp.period == per) & (cmp.basis == "net")].metric)
        assert {"d_sharpe", "d_sharpe_p", "d_cer_ci_lo", "d_cer_ci_hi", "d_cer_boot_p", "breakeven_cost_bp"} <= got
    # the default (dev) path has no holdout periods
    assert "holdout" not in {p for p, *_ in stg.level_metrics(a, s, h)}


def test_bh_family_is_frozen_models_only_and_per_period():
    rows = []
    for per in ("full", "holdout"):
        for m in (*stg.FROZEN_MEAN, *config.ROBUSTNESS, config.COMPOSITE):
            for r in ("mv", "timing"):
                rows.append(["x", "mean", r, m, "c", "all", 1, per, "net", "hist_mean", "d_cer_p",
                             0.01 if per == "holdout" else 0.5])
    met = pd.DataFrame(rows, columns=["strategy", "family", "rule", "mean_model", "cov_model", "universe",
                                      "horizon", "period", "basis", "benchmark", "metric", "value"])
    out = stg._add_fdr(met)
    q = out[out.metric == "d_cer_q_bh"]
    assert set(q.mean_model) == set(stg.FROZEN_MEAN)
    assert len(q) == 2 * 2 * len(stg.FROZEN_MEAN)
    assert np.allclose(q[q.period == "holdout"].value, 0.01) and np.allclose(q[q.period == "full"].value, 0.5)
    tab = stg.summary_by_period(out.assign(metric=out.metric))
    assert list(tab.index.get_level_values("period").unique()) == ["holdout", "full"]


def test_default_mean_models_read_from_config():
    assert set(stg.DEFAULT_MEAN) == {*config.FROZEN_MEAN, *config.ROBUSTNESS, config.COMPOSITE}
    assert stg.BH_FAMILY == tuple(config.FROZEN_MEAN)
