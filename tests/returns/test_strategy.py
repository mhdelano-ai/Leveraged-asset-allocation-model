"""Strategy test: rules, accounting, timing (no look-ahead), inference, periods,
the static ``sample`` comparator, and the reproduction of assetpred's archived
12-month results by the ported code."""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

from assetforecast import paths
from assetforecast.returns import config, schema
from assetforecast.returns.evaluation import static_cov
from assetforecast.returns.evaluation import strategy as stg
from assetforecast.returns.evaluation import strategy_rules as rules
from assetforecast.returns.evaluation import strategy_stats as st
from assetforecast.universe import ASSET_IDS

A = list(stg.ASSETS)
N = len(A)
H = stg.HORIZON
OLD_IDS = ("us_eq", "dev_eq", "em_eq", "ust_int", "ust_long", "tips", "ig_credit", "hy_credit", "reits",
           "gold", "commodities")
OPEN_START, OPEN_END = "1900-01-31", "2099-12-31"


def _cov(rng, n=N, scale=0.002):
    X = rng.standard_normal((n, n))
    C = X @ X.T / n + 0.5 * np.eye(n)
    d = np.sqrt(np.diag(C))
    vols = rng.uniform(0.01, 0.06, n)
    return C / np.outer(d, d) * np.outer(vols, vols)


def synthetic_inputs(seed=0, T=150, h=H, late=("gold", "tips"), drift=0.003, start="1995-01-31"):
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=T + h + 3, freq="ME")
    C = _cov(rng)
    L = np.linalg.cholesky(C)
    x = rng.standard_normal((len(dates), N)) @ L.T + drift
    rf = pd.Series(0.002, index=dates, name="cash")
    R = pd.DataFrame(np.expm1(x + np.log1p(0.002)), index=dates, columns=A)
    origins, mean, cov = {}, {}, {}
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
    return stg.Inputs(R, rf, origins, mean, cov, tuple(A))


# ----------------------------------------------------------------- universe and costs
def test_universe_and_costs_cover_the_package_ids():
    assert list(stg.ASSETS) == list(ASSET_IDS)
    assert set(stg.COST_BP) == set(ASSET_IDS)
    assert set(stg.COST_BP_AP) == set(OLD_IDS)
    carried = {"us_large": "us_eq", "intl_dev": "dev_eq", "em": "em_eq"}
    for a in ASSET_IDS:
        if a == "us_small":
            continue
        assert stg.COST_BP[a] == stg.COST_BP_AP[carried.get(a, a)], a
    assert stg.COST_BP["us_small"] == 10
    assert set(stg.COST_NEW_JUDGEMENTS) == {"us_small"}
    assert stg.SIXTY_FORTY == ("us_large", "ust_int")
    assert set(stg.SIXTY_FORTY) <= set(ASSET_IDS)
    assert set(stg.UNIVERSES["ex_hy"]) <= set(ASSET_IDS)


def test_defaults_read_from_config():
    assert set(stg.DEFAULT_MEAN) == {*config.FROZEN_MEAN, *config.ROBUSTNESS}
    assert stg.BH_FAMILY == tuple(config.FROZEN_MEAN)
    assert stg.PRIMARY_COV == "af_primary" and tuple(stg.DEFAULT_RISK) == ("af_primary",)
    assert stg.HORIZON == 12
    assert {k: tuple(v) for k, v in stg.PERIODS.items()} == {
        k: tuple(config.PERIODS[k]) for k in ("dev", "holdout", "holdout_ap")}
    assert stg.PERIODS["dev"][1] == "2013-12-31"
    assert stg.PERIODS["holdout"][0] == "2014-01-31"
    assert stg.PERIODS["holdout_ap"][0] == "2015-01-31"


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
    R.loc["2000-02-29", "us_large"] = 0.10
    R.loc["2000-03-31", "us_large"] = -0.05
    rf = pd.Series(0.0, index=months)
    held = np.zeros((3, N))
    i = A.index("us_large")
    j = A.index("ust_int")
    held[:, i] = [1.0, 1.0, 0.5]
    held[:, j] = [0.0, 0.5, 0.5]
    out = stg.simulate(held, f, R, rf, cost_bp={a: 10 for a in A}, spread=0.012, assets=A)
    # the weight formed at end-Jan earns February's return
    assert out.date.tolist() == list(months[1:])
    np.testing.assert_allclose(out.gross, [0.10, -0.05, 0.0])
    to2 = abs(0.5 - 1.0) + abs(0.5 - 0.5 / 0.95)
    np.testing.assert_allclose(out.turnover, [1.0, 0.5, to2])
    np.testing.assert_allclose(out.cost, 0.001 * out.turnover)
    np.testing.assert_allclose(out.financing, [0.0, 0.001 * 0.5, 0.0])
    np.testing.assert_allclose(out.net, out.gross - out.cost - out.financing)
    # the cost table is per asset: only us_large and ust_int traded
    out = stg.simulate(held, f, R, rf, cost_bp=stg.COST_BP, spread=0.012, assets=A)
    np.testing.assert_allclose(out.cost[0], 1.0 * stg.COST_BP["us_large"] * 1e-4)


def test_simulate_needs_a_cost_for_every_asset():
    f = pd.date_range("2000-01-31", periods=2, freq="ME")
    R = pd.DataFrame(0.0, index=pd.date_range("2000-01-31", periods=3, freq="ME"), columns=A)
    rf = pd.Series(0.0, index=R.index)
    with pytest.raises(KeyError):
        stg.simulate(np.zeros((2, N)), f, R, rf, cost_bp=stg.COST_BP_AP, assets=A)


@pytest.mark.parametrize("h", [3, 12])
def test_held_weights_staggered_average(h):
    rng = np.random.default_rng(0)
    W = rng.random((30, N))
    Hw, valid = stg.held_weights(W, h)
    assert not valid[:h - 1].any() and valid[h - 1:].all()
    for k in range(h - 1, 30):
        np.testing.assert_allclose(Hw[k], W[k - h + 1:k + 1].mean(0))


def test_eligible_set_grows():
    inp = synthetic_inputs()
    W = stg.target_weights(stg.Spec("mv", "model", "dyn", "all", H), inp)
    g = A.index("gold")
    assert (W[:75, g] == 0).all() and (W[75:, g] > 0).any()
    W = stg.target_weights(stg.Spec("mv", "model", "dyn", "ex_hy", H), inp)
    assert (W[:, A.index("hy_credit")] == 0).all()


def test_static_portfolios():
    inp = synthetic_inputs()
    ret, w = stg.run_spec(stg.Spec("sixty_forty", "none", "none", "all", H), inp)
    assert (w["us_large"] == 0.6).all() and (w["ust_int"] == 0.4).all()
    assert (w.drop(columns=["us_large", "ust_int"]) == 0).all().all()
    ew, w = stg.run_spec(stg.Spec("ew", "none", "none", "all", H), inp)
    # monthly target, no staggering: 1/N of the eligible assets at each formation
    n_elig = np.isfinite(inp.mean[("hist_mean", H)]).sum(1)[H - 1:]
    np.testing.assert_allclose(w.to_numpy(), (w.to_numpy() > 0) / n_elig[:, None])
    # same holding months as the staggered forecast rules
    mv, _ = stg.run_spec(stg.Spec("mv", "model", "dyn", "all", H), inp)
    assert ret.date.tolist() == mv.date.tolist() == ew.date.tolist()


def test_specs_and_comparisons_are_12_month_mean_and_risk_only():
    specs = stg.build_specs(["m1", "m2"], "af_primary", ["af_primary"])
    assert {sp.horizon for _, sp in specs} == {12}
    assert {f for f, _ in specs} == {"mean", "risk", "static"}
    pairs = stg.comparisons(specs, ["m1", "m2"], "af_primary", ["af_primary"])
    assert {f for f, *_ in pairs} == {"mean", "risk"}
    risk = [(a, b) for f, a, b in pairs if f == "risk"]
    # the pre-declared shape: three rules and vt_rp, package covariance vs static sample, both universes
    assert {(a.rule, a.universe) for a, _ in risk} == {(r, u) for r in ("mv", "mv_vt", "timing", "vt_rp")
                                                       for u in stg.UNIVERSES}
    assert all(a.cov_model == "af_primary" and b.cov_model == "sample" for a, b in risk)
    mean = [(a, b) for f, a, b in pairs if f == "mean"]
    assert len(mean) == 2 * 3 * len(stg.UNIVERSES)
    assert all(b.mean_model == "hist_mean" and b.cov_model == a.cov_model == "af_primary" for a, b in mean)


# ----------------------------------------------------------------- no look-ahead
@pytest.mark.parametrize("rule", ["mv", "mv_vt", "timing", "vt_rp"])
def test_no_weight_depends_on_later_data(rule):
    inp = synthetic_inputs()
    mean_model = "none" if rule == "vt_rp" else "model"
    spec = stg.Spec(rule, mean_model, "dyn", "all", H)
    ret0, w0 = stg.run_spec(spec, inp)
    cut = inp.origins[H][90]
    # scramble everything dated after the cut: returns, forecasts and covariances
    rng = np.random.default_rng(99)
    inp2 = synthetic_inputs()
    late_r = inp2.returns.index > cut
    inp2.returns.loc[late_r] = rng.normal(0, 0.2, (late_r.sum(), N))
    late_o = inp2.origins[H] > cut
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
    """Forecast at t = realized excess return over t+1..t+12 must earn a large
    Sharpe ratio; forecast at t = return over t-11..t (iid zero-mean data, so
    no information) must not."""
    inp = synthetic_inputs(seed=3, T=400, late=(), drift=0.0)
    o = inp.origins[H]
    xr = np.log1p(inp.returns).sub(np.log1p(inp.cash), axis=0)
    inp.mean[("model", H)] = schema.make_targets(xr, H).reindex(o).to_numpy()
    inp.mean[("stale", H)] = xr.rolling(H).sum().reindex(o).to_numpy()
    ora, _ = stg.run_spec(stg.Spec("timing", "model", "sample", "all", H), inp)
    sta, _ = stg.run_spec(stg.Spec("timing", "stale", "sample", "all", H), inp)
    assert st.sharpe(ora.gross) > 1.0
    assert abs(st.sharpe(sta.gross)) < 0.5


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


# ----------------------------------------------------------------- periods (synthetic dates only)
def test_period_masks_by_sleeve_formation_date():
    formation = pd.date_range("2012-01-31", "2017-12-31", freq="ME")
    m = stg.period_masks(formation, H)
    assert set(m) == {"dev", "holdout", "holdout_ap"}
    # dev: every sleeve formed by 2013-12-31; the holdouts: every sleeve formed from their start
    assert formation[m["dev"]].max() == pd.Timestamp("2013-12-31")
    assert formation[m["holdout"]].min() == pd.Timestamp("2014-12-31")
    assert formation[m["holdout_ap"]].min() == pd.Timestamp("2015-12-31")
    assert (formation[m["holdout"]] - pd.offsets.MonthEnd(H - 1) >= pd.Timestamp("2014-01-31")).all()
    assert not (m["dev"] & m["holdout"]).any()
    assert (m["holdout_ap"] <= m["holdout"]).all()
    # the 11 positions whose sleeves straddle the dev/holdout boundary are in neither
    assert (~m["dev"] & ~m["holdout"]).sum() == H - 1
    # the holdouts end at the last origin of config.PERIODS (2025-10-31)
    late = pd.date_range("2025-01-31", "2027-06-30", freq="ME")
    assert late[stg.period_masks(late, H)["holdout"]].max() == pd.Timestamp("2025-10-31")


@pytest.mark.parametrize("h", [1, 3, 12])
def test_period_masks_reproduce_assetpred_holdout_rule(h):
    """assetpred's rule (pre_holdout: holding month <= 2014-12-31; holdout:
    oldest sleeve formed >= 2015-01-31) is the formation-date rule with
    pre_holdout = origins up to 2014-11-30."""
    formation = pd.date_range("2013-01-31", "2017-12-31", freq="ME")
    dates = formation + pd.offsets.MonthEnd(1)
    oldest = formation - pd.offsets.MonthEnd(h - 1) if h > 1 else formation
    pre_old = np.asarray(dates <= pd.Timestamp("2014-12-31"))
    ho_old = np.asarray(oldest >= pd.Timestamp("2015-01-31"))
    m = stg.period_masks(formation, h, {"pre_holdout": (OPEN_START, "2014-11-30"),
                                        "holdout": ("2015-01-31", OPEN_END)})
    np.testing.assert_array_equal(m["pre_holdout"], pre_old)
    np.testing.assert_array_equal(m["holdout"], ho_old)


def _synthetic_full_run():
    """Synthetic panel spanning the dev/holdout boundaries (no real data)."""
    inp = synthetic_inputs(seed=5, T=160, start="2004-01-31")
    return inp


def test_full_run_periods_metrics_and_tests():
    inp = _synthetic_full_run()
    s = rules.RuleSettings()
    a, _ = stg.run_spec(stg.Spec("mv_vt", "model", "dyn", "all", H), inp)
    b, _ = stg.run_spec(stg.Spec("mv_vt", "hist_mean", "dyn", "all", H), inp)
    lv = pd.DataFrame(stg.level_metrics(a, s, H), columns=["period", "basis", "metric", "value"])
    assert {"full", "dev", "holdout", "holdout_ap"} <= set(lv.period)
    # period metrics are computed on exactly the period's months
    m = stg.period_masks(a.formation, H)
    for per in ("dev", "holdout", "holdout_ap"):
        n = lv[(lv.period == per) & (lv.basis == "net") & (lv.metric == "n_months")].value.item()
        assert n == m[per].sum()
        sr = lv[(lv.period == per) & (lv.basis == "net") & (lv.metric == "sharpe")].value.item()
        assert math.isclose(sr, st.sharpe(a.net[m[per]]))
    cmp = pd.DataFrame(stg.compare(a, b, H, s, 1, {}), columns=["period", "basis", "metric", "value"])
    for per in ("dev", "holdout", "holdout_ap"):
        got = set(cmp[(cmp.period == per) & (cmp.basis == "net")].metric)
        assert {"d_sharpe", "d_sharpe_p", "d_cer_ci_lo", "d_cer_ci_hi", "d_cer_boot_p", "breakeven_cost_bp"} <= got


def test_bh_family_is_frozen_models_only_and_per_period():
    rows = []
    for per in ("full", "holdout"):
        for m in (*stg.FROZEN_MEAN, *config.ROBUSTNESS):
            for r in ("mv", "timing"):
                rows.append(["x", "mean", r, m, "c", "all", 12, per, "net", "hist_mean", "d_cer_p",
                             0.01 if per == "holdout" else 0.5])
    met = pd.DataFrame(rows, columns=["strategy", "family", "rule", "mean_model", "cov_model", "universe",
                                      "horizon", "period", "basis", "benchmark", "metric", "value"])
    out = stg._add_fdr(met)
    q = out[out.metric == "d_cer_q_bh"]
    assert set(q.mean_model) == set(stg.FROZEN_MEAN)
    assert len(q) == 2 * 2 * len(stg.FROZEN_MEAN)
    assert np.allclose(q[q.period == "holdout"].value, 0.01) and np.allclose(q[q.period == "full"].value, 0.5)
    tab = stg.summary_by_period(out)
    assert list(tab.index.get_level_values("period").unique()) == ["holdout", "full"]


# ----------------------------------------------------------------- static sample covariance
def _xr(seed=0, T=240, late=("gold",)):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("1980-01-31", periods=T, freq="ME")
    x = pd.DataFrame(rng.normal(0.004, 0.04, (T, 4)) @ np.triu(np.ones((4, 4))) * 0.5, index=idx,
                     columns=["us_large", "ust_int", "gold", "em"])
    for a in late:
        x.loc[x.index[:150], a] = np.nan
    return x


def test_sample_cov_eligibility_psd_and_formula():
    xr = _xr()
    h = 12
    df = static_cov.sample_cov_history(xr, h, xr.index[60:])
    assert list(df.columns) == schema.COV_COLUMNS
    assert (df.model == "sample").all() and (df.horizon == h).all()
    for t, g in df.groupby("date"):
        assets = list(dict.fromkeys(g.asset_i))
        n_obs = xr.loc[:t].notna().sum()
        assert assets == [a for a in xr.columns if n_obs[a] >= 60]
        S = g["cov"].to_numpy().reshape(len(assets), len(assets))
        assert np.array_equal(S, S.T) and np.linalg.eigvalsh(S)[0] > 0
    # diagonal: variance of the realized 12-month sums (rows s <= t - 12) times (1 + h / T_i)
    t = xr.index[230]
    y = xr.loc[:t].rolling(h, min_periods=h).sum().shift(-h)
    g = df[df.date == t].set_index(["asset_i", "asset_j"])["cov"]
    for a in ("us_large", "gold"):
        T_a = xr.loc[:t, a].notna().sum()
        assert math.isclose(g[(a, a)], y[a].var(ddof=1) * (1 + h / T_a), rel_tol=1e-10)


def test_sample_cov_uses_no_later_data():
    xr = _xr()
    t = xr.index[180]
    a = static_cov.sample_cov_history(xr, 12, [t])
    xr2 = xr.copy()
    xr2.loc[xr2.index > t] = 9.0
    b = static_cov.sample_cov_history(xr2, 12, [t])
    pd.testing.assert_frame_equal(a, b)


def test_nearest_corr_and_make_psd():
    R = np.array([[1, 0.9, -0.9], [0.9, 1, 0.9], [-0.9, 0.9, 1.0]])
    C = static_cov.nearest_corr(R)
    assert np.allclose(np.diag(C), 1) and np.linalg.eigvalsh(C)[0] > 0
    S = np.diag([1.0, 2.0, 3.0])
    assert np.array_equal(static_cov.make_psd(S), S)
    M = static_cov.mean_uncertainty_factor(np.array([[100, 50], [50, 60]]), 12)
    np.testing.assert_allclose(np.diag(M), [1.12, 1.2])


def test_linalg_projections_on_random_matrices():
    """Kept from assetpred's test_covariance (the pieces static_cov ported)."""
    rng = np.random.default_rng(0)
    for _ in range(20):
        A_ = rng.uniform(-1, 1, (6, 6))
        A_ = (A_ + A_.T) / 2
        np.fill_diagonal(A_, 1)
        R = static_cov.nearest_corr(A_)
        assert np.allclose(np.diag(R), 1) and np.array_equal(R, R.T)
        assert np.linalg.eigvalsh(R)[0] > 0
        S = static_cov.make_psd(A_)
        assert static_cov.min_eig(S) > 0 and np.array_equal(S, S.T)
    good = static_cov.nearest_corr(np.array([[1, .3], [.3, 1.]]))
    assert np.allclose(good, [[1, .3], [.3, 1]])
    F = static_cov.mean_uncertainty_factor(np.array([[100, 40], [40, 40.]]), 12)
    assert np.allclose(np.diag(F), [1.12, 1.3]) and static_cov.min_eig(F - 1) >= -1e-12


def test_sample_cov_diagonal_equals_hist_mean_sd():
    """The static comparator's variances are the hist_mean baseline's sd^2 (kept from
    assetpred's test_covariance): the risk question changes only the covariance."""
    from assetforecast.returns.evaluation.synthetic import make_synthetic
    from assetforecast.returns.evaluation.walkforward import WalkForwardSettings, walk_forward
    data = make_synthetic(n_assets=3, start="1960-01-31", end="1995-12-31", seed=3)
    wf = walk_forward("hist_mean", data.features, data.xr, H,
                      WalkForwardSettings(oos_start="1985-01-31", min_history=60))
    cv = static_cov.sample_cov_history(data.xr, H, sorted(wf["date"].unique()))
    diag = cv[cv.asset_i == cv.asset_j].set_index(["date", "asset_i"])["cov"]
    sd = wf.set_index(["date", "asset"])["sd"]
    common = sd.index.intersection(diag.index)
    assert len(common) > 100
    np.testing.assert_allclose(np.sqrt(diag.loc[common].to_numpy()), sd.loc[common].to_numpy(), rtol=1e-8)


# ----------------------------------------------------------------- end to end (synthetic files)
def _write_synthetic_results(root, inp, h=H):
    o = inp.origins[h]
    for m in ("hist_mean", "model"):
        mu = pd.DataFrame(inp.mean[(m, h)], index=o, columns=A).rename_axis("date")
        f = mu.reset_index().melt("date", var_name="asset", value_name="mean").dropna()
        f.insert(0, "horizon", h)
        f.insert(0, "model", m)
        f["sd"] = 0.1
        for q in schema.QUANTILE_COLUMNS:
            f[q] = f["mean"]
        p = root / "forecasts" / m / f"h{h:02d}.parquet"
        p.parent.mkdir(parents=True)
        f[schema.FORECAST_COLUMNS].to_parquet(p)
    arr = inp.cov[("dyn", h)]
    rows = []
    for k, t in enumerate(o):
        ok = np.isfinite(np.diag(arr[k]))
        el = [a for a, e in zip(A, ok) if e]
        S = arr[k][np.ix_(ok, ok)]
        n = len(el)
        rows.append(pd.DataFrame({"model": "af_primary", "horizon": h, "date": t, "asset_i": np.repeat(el, n),
                                  "asset_j": np.tile(el, n), "cov": S.ravel()}))
    p = root / "covariances" / "af_primary" / f"h{h:02d}.parquet"
    p.parent.mkdir(parents=True)
    pd.concat(rows)[schema.COV_COLUMNS].to_parquet(p)


@pytest.mark.slow
def test_run_end_to_end_on_synthetic_files(tmp_path):
    inp = synthetic_inputs(seed=7, T=110, start="1998-01-31")
    # the static sample needs 60 months of history before the first origin
    pre = pd.DataFrame(np.random.default_rng(1).normal(0.004, 0.03, (80, N)),
                       index=pd.date_range("1991-05-31", periods=80, freq="ME"), columns=A)
    R = pd.concat([pre, inp.returns])
    rf = pd.Series(0.002, index=R.index)
    _write_synthetic_results(tmp_path, inp)
    out = tmp_path / "strategy"
    res = stg.run(tmp_path, ["model"], "af_primary", ["af_primary"], returns=R, cash=rf, jobs=1, out_dir=out,
                  max_date=None)
    for f in ("metrics.parquet", "summary.csv", "summary_by_period.csv", "returns.parquet", "weights.parquet",
              "meta.json"):
        assert (out / f).exists()
    assert (tmp_path / "covariances" / "sample" / "h12.parquet").exists()
    sample = pd.read_parquet(tmp_path / "covariances" / "sample" / "h12.parquet")
    assert set(pd.to_datetime(sample.date)) == set(inp.origins[H][inp.origins[H] + pd.offsets.MonthEnd(H)
                                                                  <= R.index.max()])
    r = res["returns"]
    assert (r.formation < r.date).all()
    m = res["metrics"]
    assert set(m.family.astype(str)) == {"mean", "risk", "static"}
    assert (m.horizon == 12).all()
    risk = m[(m.family == "risk") & (m.benchmark != "none")]
    assert set(risk.cov_model.astype(str)) == {"af_primary"}
    assert all("|sample|" in b for b in risk.benchmark)
    assert set(risk.rule.astype(str)) == {"mv", "mv_vt", "timing", "vt_rp"}
    w = res["weights"]
    assert w[(w.asset == "gold") & (w.formation < inp.origins[H][55])].empty
    meta = json.loads((out / "meta.json").read_text())
    assert meta["label"] == config.REPLICATION_LABEL
    assert meta["cost_bp_one_way"]["us_small"] == 10 and "us_small" in meta["cost_new_judgements"]
    assert meta["sixty_forty"] == ["us_large", "ust_int"] and meta["horizons"] == [12]
    assert meta["periods"]["holdout"] == list(config.PERIODS["holdout"])


def test_run_refuses_a_universe_without_costs(tmp_path):
    with pytest.raises(ValueError, match="no trading cost"):
        stg.run(tmp_path, ["model"], assets=OLD_IDS, cost_bp=stg.COST_BP, returns=pd.DataFrame(),
                cash=pd.Series(dtype=float))


def test_load_inputs_rejects_unknown_asset_ids(tmp_path):
    inp = synthetic_inputs(T=30)
    _write_synthetic_results(tmp_path, inp)
    with pytest.raises(ValueError, match="not in the universe"):
        stg.load_inputs(tmp_path, ["model"], ["af_primary"], inp.returns, inp.cash, assets=OLD_IDS)


# ----------------------------------------------------------------- reproduction of assetpred's results
AP = paths.ARCHIVE_AP
AP_COV = "vc_daily_ewma_m_lwb"                        # assetpred's PRIMARY_COV
AP_RISK = (AP_COV, "vc_ewma_ewma_m_lwb", "vc_daily_daily_m_lw", "ewma_cov", "lw")   # its FROZEN_COV
AP_MEAN = (*config.FROZEN_MEAN, *config.ROBUSTNESS, "ens_z__dist_lin_agg")      # + its COMPOSITE
# assetpred's DEV_END = 2014-12-31 (holding months) and HOLDOUT_START = 2015-01-31
AP_PERIODS = {"pre_holdout": (OPEN_START, "2014-11-30"), "holdout": ("2015-01-31", OPEN_END)}
_AP_FILES = [AP / "data" / "asset_returns.parquet", AP / "data" / "cash.parquet",
             AP / "strategy" / "metrics.parquet", AP / "strategy" / "summary_by_period.csv",
             AP / "covariances" / "sample" / "h12.parquet",
             *[AP / "forecasts" / m / "h12.parquet" for m in ("hist_mean", *AP_MEAN)],
             *[AP / "covariances" / c / "h12.parquet" for c in AP_RISK]]
needs_archive = pytest.mark.skipif(not all(p.exists() for p in _AP_FILES),
                                   reason="archived assetpred results not present")


def _archived_panel():
    R = pd.read_parquet(AP / "data" / "asset_returns.parquet")
    rf = pd.read_parquet(AP / "data" / "cash.parquet")["cash"]
    assert tuple(R.columns) == OLD_IDS
    return R, rf


@pytest.mark.slow
@pytest.mark.realdata
@needs_archive
def test_sample_cov_reproduces_archived_sample():
    R, rf = _archived_panel()
    xr = schema.excess_log_returns(R, rf)
    old = pd.read_parquet(AP / "covariances" / "sample" / "h12.parquet")
    new = static_cov.sample_cov_history(xr, 12, xr.index[xr.index >= "1990-01-31"])   # assetpred OOS_START
    assert len(new) == len(old)
    for c in ("model", "asset_i", "asset_j"):
        assert (new[c].astype(str).to_numpy() == old[c].astype(str).to_numpy()).all()
    assert (pd.to_datetime(new.date).to_numpy() == pd.to_datetime(old.date).to_numpy()).all()
    assert (new.horizon.to_numpy() == old.horizon.to_numpy()).all()
    np.testing.assert_array_equal(new["cov"].to_numpy(), old["cov"].to_numpy())


@pytest.fixture(scope="module")
def ap_run(tmp_path_factory):
    R, rf = _archived_panel()
    out = tmp_path_factory.mktemp("ap_strategy")
    res = stg.run(AP, list(AP_MEAN), AP_COV, list(AP_RISK), assets=OLD_IDS, cost_bp=stg.COST_BP_AP,
                  cost_notes={}, sixty_forty=("us_eq", "ust_int"), periods=AP_PERIODS, returns=R, cash=rf,
                  build_sample=False, max_date=None, jobs=2, out_dir=out)
    return res, out


@pytest.mark.slow
@pytest.mark.realdata
@needs_archive
def test_strategy_reproduces_archived_12_month_metrics(ap_run):
    """Every archived h=12 metric row, the FDR q-values and the bootstrap
    numbers included, is reproduced exactly by the ported code on the
    archived inputs with assetpred's ids, costs, risk model and periods."""
    res, _ = ap_run
    key = ["strategy", "family", "period", "basis", "benchmark", "metric"]
    old = pd.read_parquet(AP / "strategy" / "metrics.parquet")
    old = old[old.horizon == 12].astype({c: str for c in key}).set_index(key)["value"]
    new = res["metrics"].astype({c: str for c in key}).set_index(key)["value"]
    assert not old.index.duplicated().any() and not new.index.duplicated().any()
    assert len(old) == 69874
    missing = old.index.difference(new.index)
    assert len(missing) == 0, missing[:10]
    # the only additions are the static reference levels (assetpred ran them at h=1 only)
    extra = new.index.difference(old.index)
    assert set(extra.get_level_values("family")) == {"static"}
    got = new.reindex(old.index)
    both_nan = old.isna() & got.isna()
    assert (both_nan | (got == old)).all(), (got - old)[~(both_nan | (got == old))].head()


@pytest.mark.slow
@pytest.mark.realdata
@needs_archive
def test_strategy_reproduces_archived_summary(ap_run):
    res, _ = ap_run
    idx = ["period", "family", "horizon", "universe", "rule", "mean_model", "cov_model"]
    old = pd.read_csv(AP / "strategy" / "summary_by_period.csv")
    old = old[old.horizon == 12].set_index(idx).sort_index()
    met = res["metrics"]
    new = stg.summary_by_period(met[met.family != "static"], periods=("holdout", "pre_holdout", "full"))
    new = new.reset_index().set_index(idx).sort_index()
    assert old.index.equals(new.index)
    # the archive is written with float_format %.6g after rounding to 4 decimals
    np.testing.assert_allclose(new.to_numpy(float), old[new.columns].to_numpy(float), rtol=1e-5, atol=1e-12)
