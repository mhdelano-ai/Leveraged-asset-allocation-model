"""Tests for volmodel.eval.inference.

Fast tests (seconds, seeded) check mechanics and give coarse Monte Carlo
sanity bounds. ``slow`` tests reproduce the size / coverage results quoted in
the module docstring with more replications.

The simulated losses mimic the real backtest: a latent monthly log-variance
AR(1) with heavy-tailed (clipped t5) shocks, gamma measurement noise in monthly
realized variance, 12-month-ahead targets (11-month overlap between
consecutive origins), and forecasts whose log errors are persistent AR(1)
processes with a common component across models. Under the null two models
have identical error distributions, so E[loss differential] = 0 exactly.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from volmodel.eval import inference as inf

H = 12


def simulate_losses(T, sig_err, rng, *, phi=0.95, sd_x=0.7, nu=5, rho_e=0.9, common=0.5, loss="qlike", h=H, burn=200):
    sig_err = np.asarray(sig_err, float)
    m = sig_err.size
    N = burn + T + h
    shock_sd = sd_x * np.sqrt(1 - phi**2)
    if nu is None:
        z = rng.standard_normal(N)
    else:
        z = np.clip(rng.standard_t(nu, N) / np.sqrt(nu / (nu - 2)), -5.0, 5.0)
    x = np.empty(N)
    x[0] = 0.0
    for t in range(1, N):
        x[t] = phi * x[t - 1] + shock_sd * z[t]
    v = np.exp(x) * rng.gamma(4.0, 0.25, N)
    cs = np.concatenate([[0.0], np.cumsum(v)])
    idx = np.arange(burn, burn + T)
    Y = (cs[idx + h + 1] - cs[idx + 1]) / h
    k = np.arange(1, h + 1)
    F = np.mean(np.exp(phi ** k[None, :] * x[idx][:, None] + 0.5 * sd_x**2 * (1 - phi ** (2 * k))[None, :]), axis=1)

    def ar1(shape):
        e = rng.standard_normal(shape)
        out = np.empty(shape)
        out[0] = e[0]
        s = np.sqrt(1 - rho_e**2)
        for t in range(1, shape[0]):
            out[t] = rho_e * out[t - 1] + s * e[t]
        return out

    e = sig_err[None, :] * (np.sqrt(common) * ar1((T,))[:, None] + np.sqrt(1 - common) * ar1((T, m)))
    f = F[:, None] * np.exp(e)
    if loss == "qlike":
        r = Y[:, None] / f
        return r - np.log(r) - 1.0
    return (Y[:, None] - f) ** 2


def monthly(values, start="1950-01-31", columns=None):
    idx = pd.date_range(start, periods=len(values), freq="ME")
    if np.ndim(values) == 1:
        return pd.Series(values, index=idx)
    return pd.DataFrame(values, index=idx, columns=columns)


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", inf.InferenceWarning)
        yield


# ---------------------------------------------------------------------------
# defaults and helpers
# ---------------------------------------------------------------------------


def test_default_rules():
    assert inf.default_ewc_terms(480, 12) == 10
    assert inf.default_ewc_terms(1080, 12) == 22
    assert inf.default_ewc_terms(60, 12) == 4
    assert inf.default_ewc_terms(100, 1) == 8  # capped by 0.4 T^(2/3)
    assert inf.default_bandwidth(480, 12) == 48
    assert inf.default_bandwidth(100, 12) == 48
    assert inf.default_bandwidth(3000, 1) == 72
    assert inf.default_block_length(600, 12) == 36


def test_block_bootstrap_indices():
    rng = np.random.default_rng(0)
    for kind in ("circular", "stationary"):
        ib = inf.block_bootstrap_indices(100, 500, 10, kind=kind, rng=rng)
        assert ib.shape == (500, 100)
        assert ib.min() >= 0 and ib.max() < 100
        # every point roughly equally likely
        counts = np.bincount(ib.ravel(), minlength=100) / ib.size
        assert np.allclose(counts, 0.01, atol=0.003)
        # mean run length (consecutive indices) is about the block length
        cont = (np.diff(ib, axis=1) % 100) == 1
        mean_run = ib.shape[1] / (1 + (~cont).sum(axis=1)).mean()
        assert 7 < mean_run < 11
    circ = inf.block_bootstrap_indices(30, 2, 10, kind="circular", rng=1)
    assert np.all((np.diff(circ[:, :10], axis=1) % 30) == 1)


def test_lrv_known_ma_process():
    # MA(11) with equal weights: LRV = (sum b)^2 = 144 for unit innovations; gamma0 = 12
    rng = np.random.default_rng(1)
    e = rng.standard_normal(200_000 + 11)
    x = np.convolve(e, np.ones(12), mode="valid")
    for kernel in ("truncated", "bartlett", "ewc"):
        r = inf.long_run_variance(x, kernel=kernel, h=12, bandwidth=60 if kernel == "bartlett" else None, ewc_terms=200)
        assert r.lrv == pytest.approx(144, rel=0.15), kernel
    ess = inf.effective_sample_size(x[:1200], h=12)
    assert ess["n_nonoverlap"] == 100
    assert 40 < ess["n_eff"] < 250  # truth: 1200 * 12 / 144 = 100
    assert ess["acf1"] == pytest.approx(11 / 12, abs=0.05)


# ---------------------------------------------------------------------------
# Diebold-Mariano
# ---------------------------------------------------------------------------


def test_dm_iid_matches_simple_t():
    rng = np.random.default_rng(2)
    a, b = rng.standard_normal(500), rng.standard_normal(500) + 0.1
    r = inf.dm_test(a, b, kernel="bartlett", bandwidth=0, h=1)
    d = a - b
    assert r.mean_diff == pytest.approx(d.mean())
    assert r.se == pytest.approx(d.std() / np.sqrt(500))
    assert r.n_eff == pytest.approx(500, rel=1e-9)


def test_dm_antisymmetric_and_alternatives():
    rng = np.random.default_rng(3)
    L = monthly(simulate_losses(480, [0.3, 0.4], rng), columns=["a", "b"])
    r1, r2 = inf.dm_test(L.a, L.b), inf.dm_test(L.b, L.a)
    assert r1.stat == pytest.approx(-r2.stat) and r1.pvalue == pytest.approx(r2.pvalue)
    less = inf.dm_test(L.a, L.b, alternative="less")
    greater = inf.dm_test(L.a, L.b, alternative="greater")
    assert less.pvalue + greater.pvalue == pytest.approx(1.0)
    assert r1.df == inf.default_ewc_terms(480)
    assert r1.start == L.index[0] and r1.end == L.index[-1]
    s = r1.to_series()
    assert {"mean_diff", "se", "stat", "pvalue", "n_eff"} <= set(s.index)


def test_dm_classic_hln_and_negative_fallback():
    x = np.tile([1.0, -1.0], 60) + np.linspace(0, 0.01, 120)
    r = inf.dm_test(x, np.zeros(120), kernel="truncated", h=12)
    assert r.method == "dm_truncated_hln"
    assert np.isfinite(r.se) and r.se > 0
    assert any("negative" in n for n in r.notes)
    # HLN factor
    rng = np.random.default_rng(4)
    a, b = rng.standard_normal(240), rng.standard_normal(240)
    r0 = inf.dm_test(a, b, kernel="truncated", hln=False)
    r1 = inf.dm_test(a, b, kernel="truncated", hln=True)
    n = 240
    assert r1.stat / r0.stat == pytest.approx(np.sqrt((n + 1 - 24 + 12 * 11 / n) / n))
    assert r1.df == n - 1


def test_dm_handles_gaps_and_nan():
    rng = np.random.default_rng(5)
    L = monthly(simulate_losses(600, [0.3, 0.3], rng), columns=["a", "b"])
    L.iloc[:120, 0] = np.nan  # model a starts 10 years later
    L.iloc[300:303, 1] = np.nan  # short interior gap
    L = L.drop(L.index[400:402])  # missing index rows
    r = inf.dm_test(L.a, L.b)
    assert r.n_obs == 600 - 120 - 3 - 2
    assert r.start == L.index[120]
    rb = inf.dm_test_bootstrap(L.a, L.b, n_boot=199, seed=1)
    assert rb.n_obs == r.n_obs
    with pytest.warns(inf.InferenceWarning):
        warnings.simplefilter("always", inf.InferenceWarning)
        sparse = L.copy()
        sparse.iloc[::4, 0] = np.nan
        inf.dm_test(sparse.a, sparse.b)


def test_dm_warns_on_short_sample():
    rng = np.random.default_rng(6)
    L = simulate_losses(96, [0.3, 0.3], rng)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", inf.InferenceWarning)
        r = inf.dm_test(L[:, 0], L[:, 1])
    assert any(issubclass(x.category, inf.InferenceWarning) for x in w)
    assert any(n.startswith("WARNING") for n in r.notes)


def test_dm_bootstrap_variants_run_and_agree_roughly():
    rng = np.random.default_rng(7)
    L = simulate_losses(600, [0.3, 0.45], rng)
    hac = inf.dm_test(L[:, 0], L[:, 1])
    gk = inf.dm_test_bootstrap(L[:, 0], L[:, 1], n_boot=999, seed=1)
    st = inf.dm_test_bootstrap(L[:, 0], L[:, 1], n_boot=999, seed=1, bootstrap="stationary", studentize=False)
    assert gk.mean_diff == hac.mean_diff == st.mean_diff
    assert 0.4 < gk.se / hac.se < 2.5 and 0.4 < st.se / hac.se < 2.5
    assert np.sign(gk.stat) == np.sign(hac.stat)
    with pytest.raises(ValueError):
        inf.dm_test_bootstrap(L[:, 0], L[:, 1], bootstrap="stationary", studentize=True)
    # deterministic given seed
    assert inf.dm_test_bootstrap(L[:, 0], L[:, 1], n_boot=199, seed=3).pvalue == inf.dm_test_bootstrap(
        L[:, 0], L[:, 1], n_boot=199, seed=3
    ).pvalue


def test_ewc_is_exact_t_under_iid():
    rng = np.random.default_rng(8)
    T, B, reps = 240, 10, 1500
    X = rng.standard_normal((reps, T))
    p = np.array([inf.dm_test(x, np.zeros(T), ewc_terms=B, h=1).pvalue for x in X])
    assert abs(np.mean(p < 0.10) - 0.10) < 0.025
    assert abs(np.mean(p < 0.05) - 0.05) < 0.02


def test_dm_size_quick_monte_carlo():
    """Coarse, seeded size check of the default DM under the realistic DGP (200 reps)."""
    rng = np.random.default_rng(9)
    p_ewc, p_classic = [], []
    for _ in range(200):
        L = simulate_losses(480, [0.3, 0.3], rng)
        p_ewc.append(inf.dm_test(L[:, 0], L[:, 1], warn=False).pvalue)
        p_classic.append(inf.dm_test(L[:, 0], L[:, 1], kernel="truncated", warn=False).pvalue)
    rej = np.mean(np.array(p_ewc) < 0.10)
    assert 0.03 < rej < 0.18  # nominal 0.10; MC s.e. ~0.02
    assert np.mean(np.array(p_classic) < 0.10) >= rej - 0.05


# ---------------------------------------------------------------------------
# panel
# ---------------------------------------------------------------------------


def _panel_from(arrs: dict[str, np.ndarray], models=("a", "b")):
    rows = []
    for asset, X in arrs.items():
        idx = pd.date_range("1950-01-31", periods=X.shape[0], freq="ME")
        for j, m in enumerate(models):
            rows.append(pd.DataFrame({"origin": idx, "asset": asset, "model": m, "loss": X[:, j]}))
    return pd.concat(rows, ignore_index=True)


def test_panel_single_asset_equals_time_series():
    rng = np.random.default_rng(10)
    X = simulate_losses(480, [0.3, 0.35], rng)
    P = _panel_from({"us_large": X})
    r = inf.panel_dm_test(P, "a", "b")
    s = inf.dm_test(X[:, 0], X[:, 1])
    assert r.mean_diff == pytest.approx(s.mean_diff)
    assert r.se == pytest.approx(s.se, rel=1e-8)
    assert r.pvalue == pytest.approx(s.pvalue, rel=1e-8)


def test_panel_duplicated_assets_add_no_information():
    rng = np.random.default_rng(11)
    X = simulate_losses(480, [0.3, 0.35], rng)
    one = inf.panel_dm_test(_panel_from({"x": X}), "a", "b")
    five = inf.panel_dm_test(_panel_from({f"x{i}": X for i in range(5)}), "a", "b")
    assert five.se == pytest.approx(one.se, rel=1e-8)
    assert five.stat == pytest.approx(one.stat, rel=1e-8)


def test_panel_unbalanced_and_weights():
    rng = np.random.default_rng(12)
    arrs = {f"a{i}": simulate_losses(600, [0.3, 0.35], rng) for i in range(4)}
    arrs["late"] = simulate_losses(600, [0.3, 0.35], rng)
    arrs["late"][:400] = np.nan  # starts in the 1980s
    P = _panel_from(arrs).dropna()
    P = P[~((P.asset == "a0") & (P.model == "b") & (P.origin.dt.year == 1960))]  # a hole for one model
    for w in ("obs", "date", "asset"):
        r = inf.panel_dm_test(P, "a", "b", weights=w)
        assert np.isfinite(r.stat) and 0 <= r.pvalue <= 1
        assert r.n_obs == 600
    rb = inf.panel_dm_test(P, "a", "b", method="bootstrap", n_boot=299, seed=1)
    rh = inf.panel_dm_test(P, "a", "b")
    assert rb.mean_diff == pytest.approx(rh.mean_diff)
    assert 0.4 < rb.se / rh.se < 2.5
    with pytest.raises(KeyError):
        inf.panel_dm_test(P, "a", "zzz")


# ---------------------------------------------------------------------------
# MCS
# ---------------------------------------------------------------------------


def test_mcs_obvious_cases():
    rng = np.random.default_rng(13)
    X = simulate_losses(600, [0.3, 0.3, 0.3, 1.2, 1.5], rng)
    L = monthly(X, columns=list("abcde"))
    r = inf.model_confidence_set(L, 0.10, n_boot=500, seed=1)
    assert set(r.excluded) == {"d", "e"}
    assert r.elimination_order[:2] == ["e", "d"]
    # MCS p-values are non-decreasing along the elimination order and the last is 1
    pv = r.pvalues.loc[r.elimination_order].to_numpy()
    assert np.all(np.diff(pv) >= 0) and pv[-1] == 1.0
    summ = r.summary()
    assert summ["in_mcs"].sum() == 3
    rmax = inf.model_confidence_set(L, 0.10, statistic="max", n_boot=500, seed=1, bootstrap="circular")
    assert set(rmax.excluded) == {"d", "e"}


def test_mcs_matches_arch():
    arch_bs = pytest.importorskip("arch.bootstrap")
    rng = np.random.default_rng(14)
    X = simulate_losses(600, [0.3, 0.31, 0.33, 0.36, 0.4, 0.45], rng)
    L = monthly(X, columns=[f"m{i}" for i in range(6)])
    for stat in ("R", "max"):
        ours = inf.model_confidence_set(L, 0.10, statistic=stat, n_boot=2000, block_length=24, seed=1)
        ref = arch_bs.MCS(L, size=0.10, reps=2000, block_size=24, method=stat, bootstrap="stationary", seed=1)
        ref.compute()
        theirs = ref.pvalues["Pvalue"]
        common = theirs.index
        assert np.max(np.abs(ours.pvalues[common].to_numpy() - theirs.to_numpy())) < 0.08
        assert ours.elimination_order[-1] == theirs.index[-1]


def test_mcs_unequal_histories():
    rng = np.random.default_rng(15)
    L = monthly(simulate_losses(600, [0.3, 0.3, 0.6], rng), columns=["a", "b", "c"])
    L.iloc[:300, 2] = np.nan
    r = inf.model_confidence_set(L, n_boot=300, seed=1)
    assert r.n_obs == 300 and any("common sample" in n for n in r.notes)
    r2 = inf.model_confidence_set(L, n_boot=300, seed=1, min_coverage=0.9)
    assert r2.dropped_models == ["c"] and r2.n_obs == 600


def test_mcs_warns_on_short_block():
    rng = np.random.default_rng(16)
    L = monthly(simulate_losses(300, [0.3, 0.3], rng), columns=["a", "b"])
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", inf.InferenceWarning)
        inf.model_confidence_set(L, block_length=6, n_boot=100, seed=1)
    assert any("block length" in str(x.message) for x in w)


# ---------------------------------------------------------------------------
# Mincer-Zarnowitz
# ---------------------------------------------------------------------------


def test_mincer_zarnowitz():
    rng = np.random.default_rng(17)
    T = 600
    x = np.zeros(T)
    for t in range(1, T):
        x[t] = 0.97 * x[t - 1] + 0.25 * rng.standard_normal()
    f = 0.04 * np.exp(x)
    noise = np.convolve(rng.standard_normal(T + 11), np.ones(12) / 12, "valid")
    y = f * np.exp(0.3 * noise - 0.045 / 12 * 0)  # roughly unbiased, overlapping noise
    good = inf.mincer_zarnowitz(monthly(y), monthly(f))
    assert good.params["slope"] == pytest.approx(1.0, abs=0.15)
    assert good.wald_pvalue > 0.01
    assert good.df == inf.default_ewc_terms(T)
    bad = inf.mincer_zarnowitz(monthly(y), monthly(2 * f + 0.02))
    assert bad.params["slope"] == pytest.approx(0.5, abs=0.1)
    assert bad.wald_pvalue < 0.01
    lg = inf.mincer_zarnowitz(monthly(y), monthly(f), log=True)
    assert lg.params["slope"] == pytest.approx(1.0, abs=0.1)
    nw = inf.mincer_zarnowitz(monthly(y), monthly(f), kernel="bartlett")
    assert nw.se["slope"] > 0


# ---------------------------------------------------------------------------
# pairwise, multiplicity, skill
# ---------------------------------------------------------------------------


def test_adjust_pvalues_matches_statsmodels():
    multitest = pytest.importorskip("statsmodels.stats.multitest")
    p = pd.Series([0.01, 0.04, 0.03, 0.2, np.nan, 0.005])
    for ours, theirs in (("holm", "holm"), ("bonferroni", "bonferroni"), ("bh", "fdr_bh")):
        adj = inf.adjust_pvalues(p, ours)
        ref = multitest.multipletests(p.dropna(), method=theirs)[1]
        assert np.allclose(adj.dropna().to_numpy(), ref)
        assert np.isnan(adj.iloc[4])


def test_pairwise_dm():
    rng = np.random.default_rng(18)
    L = monthly(simulate_losses(480, [0.3, 0.32, 0.6, 0.3], rng), columns=["bench", "x", "bad", "y"])
    L.iloc[:100, 3] = np.nan
    r = inf.pairwise_dm(L, benchmark="bench")
    assert r.stat.shape == (4, 4)
    assert np.allclose(r.stat.to_numpy(), -r.stat.T.to_numpy(), equal_nan=True)
    assert (r.pvalue_adj.fillna(1) >= r.pvalue.fillna(1) - 1e-12).all().all()
    assert r.n_obs.loc["y", "bench"] == 380
    vb = r.vs_benchmark
    assert set(vb.index) == {"x", "bad", "y"}
    assert vb.loc["bad", "skill_pct"] < 0 and vb.loc["bad", "pvalue_adj"] < 0.05
    assert (vb["pvalue_adj"] >= vb["pvalue"]).all()


def test_skill_score():
    rng = np.random.default_rng(19)
    b = np.abs(rng.standard_normal(480)) + 0.1
    r = inf.skill_score(b * 0.9, b, n_boot=200, seed=1)
    assert r.skill == pytest.approx(10.0)
    assert r.ci_low == pytest.approx(10.0) and r.ci_high == pytest.approx(10.0)
    X = simulate_losses(480, [0.3, 0.4], rng)
    for method in ("bootstrap", "hac"):
        s = inf.skill_score(monthly(X[:, 0]), monthly(X[:, 1]), method=method, n_boot=500, seed=2)
        assert s.ci_low < s.skill < s.ci_high
        assert s.n_obs == 480


# ---------------------------------------------------------------------------
# slow Monte Carlo checks (reproduce the report's numbers at reduced scale)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_slow_dm_size_ewc_default():
    rng = np.random.default_rng(100)
    p = []
    for _ in range(1000):
        L = simulate_losses(480, [0.3, 0.3], rng)
        p.append(inf.dm_test(L[:, 0], L[:, 1], warn=False).pvalue)
    p = np.array(p)
    assert np.mean(p < 0.05) < 0.08
    assert np.mean(p < 0.10) < 0.14


@pytest.mark.slow
def test_slow_mcs_contains_best_and_shrinks():
    rng = np.random.default_rng(101)
    sig = [0.30, 0.33, 0.33, 0.36, 0.36, 0.38, 0.40, 0.40, 0.45, 0.50]
    best_in, worst_in = [], []
    for i in range(150):
        X = simulate_losses(1080, sig, rng)
        r = inf.model_confidence_set(pd.DataFrame(X, columns=[f"m{j}" for j in range(10)]), 0.10, n_boot=500, seed=i)
        best_in.append("m0" in r.included)
        worst_in.append("m9" in r.included)
    assert np.mean(best_in) >= 0.85
    assert np.mean(worst_in) <= 0.8


@pytest.mark.slow
def test_slow_skill_coverage():
    rng = np.random.default_rng(102)
    big = np.vstack([simulate_losses(5000, [0.3, 0.36], rng) for _ in range(100)])
    truth = 100 * (1 - big[:, 0].mean() / big[:, 1].mean())
    cover = []
    for i in range(300):
        X = simulate_losses(1080, [0.3, 0.36], rng)
        s = inf.skill_score(X[:, 0], X[:, 1], level=0.90, n_boot=500, seed=i, warn=False)
        cover.append(s.ci_low <= truth <= s.ci_high)
    assert 0.78 <= np.mean(cover) <= 0.97
