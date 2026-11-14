"""Size and power of the overlap-robust tests (small Monte Carlos) and the
multiple-testing tools."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from assetforecast.returns.evaluation import inference as inf
from assetforecast.returns.evaluation import metrics as mt


def _simulate_forecasts(rng, T=480, oos=120, h=12, rho=0.95, beta=0.0):
    """Monthly iid returns (plus optional signal), h-month targets, expanding
    purged forecasts: benchmark = historical mean of realized targets; models
    = OLS of the target on one of two persistent predictors. Returns y, the
    benchmark and the two model forecasts at the scored origins."""
    e = rng.standard_normal((T, 2))
    x = np.zeros((T, 2))
    for t in range(1, T):
        x[t] = rho * x[t - 1] + np.sqrt(1 - rho ** 2) * e[t]
    xr = 0.005 + 0.045 * rng.standard_normal(T)
    xr[1:] += 0.045 * beta * x[:-1, 0]
    cs = np.concatenate([[0], np.cumsum(xr)])
    y = np.full(T, np.nan)
    y[:T - h] = cs[h + 1:T + 1] - cs[1:T - h + 1]
    origins = np.arange(oos, T - h)
    yv = np.nan_to_num(y)
    n = origins - h + 1                      # rows 0 .. t-h are realized at t
    Sy = np.cumsum(yv)[origins - h]
    fb = Sy / n
    fm = []
    for j in range(2):
        xj = x[:, j]
        Sx = np.cumsum(xj)[origins - h]
        Sxx = np.cumsum(xj ** 2)[origins - h]
        Sxy = np.cumsum(xj * yv)[origins - h]
        b = (Sxy - Sx * Sy / n) / (Sxx - Sx ** 2 / n)
        fm.append(Sy / n - b * Sx / n + b * xj[origins])
    return y[origins], fb, fm[0], fm[1]


@pytest.mark.parametrize("h", [1, 12])
def test_clark_west_and_dm_size_under_null(h):
    rng = np.random.default_rng(100 + h)
    cw, dm = [], []
    for _ in range(400):
        y, fb, f1, _ = _simulate_forecasts(rng, h=h)
        cw.append(inf.clark_west(y, fb, f1, h)[1])
        # DM's null is equal accuracy of two *non-degenerate* forecasts: two
        # equally noisy perturbations of the true mean; the overlap in y makes the
        # loss differential MA(h-1). (Between two estimated models that converge
        # to the same mean DM is undersized: the nested case Clark-West handles.)
        noise = 0.02 * np.sqrt(h) * rng.standard_normal((len(y), 2))
        m = 0.005 * h
        dm.append(inf.diebold_mariano((y - m - noise[:, 0]) ** 2, (y - m - noise[:, 1]) ** 2, h)[2])
    cw, dm = np.array(cw), np.array(dm)
    # nominal 10%; Monte Carlo s.e. ~1.5pp
    assert 0.04 <= (cw < 0.10).mean() <= 0.17
    assert 0.03 <= (dm < 0.10).mean() <= 0.17
    assert (cw < 0.05).mean() <= 0.10


def test_clark_west_has_power():
    rng = np.random.default_rng(7)
    p = [inf.clark_west(*_simulate_forecasts(rng, h=1, beta=0.3)[:3], 1)[1] for _ in range(100)]
    assert (np.array(p) < 0.05).mean() > 0.5


def _correct_quantiles(rng, n, h):
    xr = rng.standard_normal(n + h + 1)
    y = np.convolve(xr, np.ones(h), "valid")[1:n + 1]
    Q = np.sqrt(h) * stats.norm.ppf(mt.TAUS)[None, :].repeat(n, 0)
    return y, Q


@pytest.mark.parametrize("h", [1, 3, 12])
def test_tail_and_calibration_tests_size(h):
    rng = np.random.default_rng(200 + h)
    kp, cal, ind = [], [], []
    for _ in range(300):
        y, Q = _correct_quantiles(rng, 372, h)
        hits, p = mt.tail_hits(y, Q, "q05")
        kp.append(inf.kupiec(hits, p, h)[1])
        ind.append(inf.christoffersen(hits, h))
        cal.append(inf.calibration_test(y, Q, h)[1])
    for v in (kp, cal, ind):
        v = np.array(v)
        v = v[np.isfinite(v)]
        assert (v < 0.05).mean() <= 0.11


def test_tail_and_calibration_tests_have_power():
    rng = np.random.default_rng(3)
    rej_k, rej_c = [], []
    for _ in range(100):
        y, Q = _correct_quantiles(rng, 372, 1)
        Qn = Q * 0.7                                  # bands 30% too narrow
        hits, p = mt.tail_hits(y, Qn, "q05")
        rej_k.append(inf.kupiec(hits, p, 1)[1] < 0.05)
        rej_c.append(inf.calibration_test(y, Qn, 1)[1] < 0.05)
    assert np.mean(rej_k) > 0.7 and np.mean(rej_c) > 0.7


def test_naive_kupiec_would_overreject_with_overlap():
    # documents why the overlap adjustment exists
    rng = np.random.default_rng(4)
    naive, adj = [], []
    for _ in range(200):
        y, Q = _correct_quantiles(rng, 372, 12)
        hits, p = mt.tail_hits(y, Q, "q05")
        naive.append(inf.kupiec(hits, p, 1)[1])
        adj.append(inf.kupiec(hits, p, 12)[1])
    assert (np.array(naive) < 0.05).mean() > 3 * max((np.array(adj) < 0.05).mean(), 0.02)


def test_hac_lags_cover_overlap():
    for h in (3, 12):
        for n in (60, 400):
            assert inf.hac_lags(h, n) >= h
    assert inf.hac_lags(1, 400) >= 1
    assert inf.block_length(12, 400) >= 12


def test_spa_and_mcs():
    rng = np.random.default_rng(5)
    n = 300
    bench = rng.standard_normal(n) ** 2 + 1
    null = pd.DataFrame({f"m{i}": bench + 0.3 * rng.standard_normal(n) for i in range(5)})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p_null = inf.spa_test(bench, null, h=1, reps=299, seed=1)
        better = null.assign(good=bench - 0.5 + 0.1 * rng.standard_normal(n))
        p_alt = inf.spa_test(bench, better, h=1, reps=299, seed=1)
        same = inf.spa_test(bench, pd.DataFrame({"copy": bench}), h=1, reps=99, seed=1)
        L = better.assign(bench=bench, bench_copy=bench)
        mcs = inf.model_confidence_set(L, h=1, size=0.1, reps=299, seed=1)
    assert p_null > 0.1
    assert p_alt < 0.05
    assert same == 1.0
    assert mcs.loc["good", "in_mcs"] == 1.0
    assert mcs.loc["bench", "mcs_p"] == mcs.loc["bench_copy", "mcs_p"]
    assert mcs.loc["bench", "in_mcs"] == 0.0


def test_fdr():
    p = np.array([0.001, 0.01, 0.02, 0.5, np.nan])
    q_bh, q_by = inf.fdr(p, "bh"), inf.fdr(p, "by")
    assert np.isnan(q_bh[-1])
    assert (q_bh[:4] >= p[:4]).all() and (q_by[:4] >= q_bh[:4]).all()


def test_crps_grid_and_pit():
    rng = np.random.default_rng(6)
    y, Q = _correct_quantiles(rng, 5000, 1)
    crps = mt.crps_grid(y, Q).mean()
    # E[CRPS] of the true N(0,1) forecast is 1/sqrt(pi); the 9-point grid misses
    # the outer tails, so it is close but not equal
    assert abs(crps - 1 / np.sqrt(np.pi)) / (1 / np.sqrt(np.pi)) < 0.15
    pit = mt.pit_values(y[:2000], Q[:2000])
    assert abs(pit.mean() - 0.5) < 0.03 and pit.min() >= 0 and pit.max() <= 1
    assert abs(mt.coverage(y, Q, 90) - 0.9) < 0.02


def _correlated_panel(rng, h, n_dates=240, n_assets=8, rho=0.7, df=None):
    """Correct-model hits for a panel of correlated assets entering at
    staggered dates, overlapping h-month targets, optional t(df) shocks."""
    C = np.full((n_assets, n_assets), rho) + (1 - rho) * np.eye(n_assets)
    L = np.linalg.cholesky(C)
    z = rng.standard_normal((n_dates + h, n_assets)) @ L.T
    if df:
        z = z / np.sqrt(rng.chisquare(df, size=(n_dates + h, 1)) / df) * np.sqrt((df - 2) / df)
    cs = np.vstack([np.zeros((1, n_assets)), np.cumsum(z, axis=0)])
    Y = cs[1 + h:n_dates + 1 + h] - cs[1:n_dates + 1]
    entry = np.linspace(0, n_dates // 2, n_assets).astype(int)
    E = np.arange(n_dates)[:, None] >= entry[None, :]
    rd, ra = np.nonzero(E)
    dates = pd.date_range("1990-01-31", periods=n_dates, freq="ME")[rd]
    return Y[rd, ra], dates, ra


@pytest.mark.parametrize("h", [1, 3, 12])
def test_pooled_kupiec_size_on_correlated_panel(h):
    # the old per-date pooled-rate t-test rejected a correct model 20-87% of the
    # time on panels like this; the combined strand test must hold its size
    rng = np.random.default_rng(300 + h)
    if h == 1:
        qs = {"q01": 0.01, "q05": 0.05}
    else:
        qs = {"q01": 0.01, "q95": 0.95}
    rej = {k: [] for k in qs}
    for i in range(150):
        y, dates, assets = _correlated_panel(rng, h, df=5 if i % 2 else None)
        for k, tau in qs.items():
            # true quantile: Gaussian for even i, simulated t(5) sum for odd i
            q = np.sqrt(h) * stats.norm.ppf(tau) if i % 2 == 0 else _T5Q[(h, tau)]
            hits = ((y < q) if tau < 0.5 else (y > q)).astype(float)
            rej[k].append(inf.kupiec_pooled(hits, min(tau, 1 - tau), h, dates, assets) < 0.05)
    for k, r in rej.items():
        assert np.mean(r) <= 0.08, (h, k, np.mean(r))


def _t5_quantiles():
    rng = np.random.default_rng(0)
    out = {}
    for h in (1, 3, 12):
        g = rng.chisquare(5, size=(200_000, h)) / 5
        s = (rng.standard_normal((200_000, h)) / np.sqrt(g) * np.sqrt(3 / 5)).sum(1)
        for tau in (0.01, 0.05, 0.95, 0.99):
            out[(h, tau)] = np.quantile(s, tau)
    return out


_T5Q = _t5_quantiles()


def test_pooled_kupiec_has_power():
    rng = np.random.default_rng(9)
    rej = []
    for _ in range(60):
        y, dates, assets = _correlated_panel(rng, 1, rho=0.3)
        q = 0.7 * stats.norm.ppf(0.05)              # bands 30% too narrow
        rej.append(inf.kupiec_pooled((y < q).astype(float), 0.05, 1, dates, assets) < 0.05)
    assert np.mean(rej) > 0.7
