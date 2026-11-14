"""Covariance forecasts: recovery, regime tracking, PSD/symmetry, no look-ahead,
combinator consistency, scoring sanity. All on synthetic data except the
dev-mode truncation check."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from assetpred import config, schema
from assetpred.covariance import combine as cmb
from assetpred.covariance import engine, linalg as la, scoring
from assetpred.covariance.base import (CovForecaster, CovLeakageError, CovTrainingSet,
                                       available, get_factory)
from assetpred.covariance.engine import CovSettings, to_matrices, walk_forward_cov
from assetpred.covariance.models import BENCHMARK, FROZEN

ASSETS4 = ["us_eq", "ust_int", "tips", "gold"]


def make_panel(n_months=300, start="1971-01-31", corr_path=None, sd=(0.045, 0.02, 0.015, 0.05),
               base_corr=0.3, late=(("tips", 72), ("gold", 120)), seed=0, days=21, with_daily=True):
    """Daily Gaussian returns aggregated to months. ``corr_path(m)`` gives the
    us_eq/ust_int correlation in month m (others fixed at base_corr)."""
    rng = np.random.default_rng(seed)
    months = pd.date_range(start, periods=n_months, freq="ME")
    sd = np.asarray(sd)
    n = len(sd)
    blocks, day_idx, monthly = [], [], []
    for m, me in enumerate(months):
        R = np.full((n, n), base_corr)
        np.fill_diagonal(R, 1.0)
        if corr_path is not None:
            R[0, 1] = R[1, 0] = corr_path(m)
        R = la.nearest_corr(R)
        bd = pd.bdate_range(me - pd.offsets.MonthBegin(1), me)
        S = R * np.outer(sd, sd) / len(bd)
        Xm = rng.multivariate_normal(np.zeros(n), S, size=len(bd))
        blocks.append(Xm)
        day_idx.extend(bd)
        monthly.append(Xm.sum(axis=0))
    X = np.vstack(blocks)
    daily = pd.DataFrame(np.expm1(X), index=pd.DatetimeIndex(day_idx), columns=ASSETS4[:n])
    xr = pd.DataFrame(np.array(monthly), index=months, columns=ASSETS4[:n])
    for a, k in late:
        if a in xr:
            xr.iloc[:k, xr.columns.get_loc(a)] = np.nan
            daily.loc[daily.index <= months[k - 1], a] = np.nan
    return xr, (daily if with_daily else None)


def make_features(xr):
    rng = np.random.default_rng(5)
    idx = pd.MultiIndex.from_product([xr.index, xr.columns], names=["date", "asset"])
    cpi = np.repeat(np.cumsum(rng.standard_normal(len(xr))) * 0.1 + 3, len(xr.columns))
    rt = np.repeat(rng.standard_normal(len(xr)), len(xr.columns))
    return pd.DataFrame({"cpi_yoy": cpi, "g_real_tbill": rt}, index=idx)


@pytest.fixture(scope="module")
def const_panel():
    return make_panel(n_months=960, start="1921-01-31", seed=1, late=())


# --------------------------------------------------------------------------
# linear algebra
# --------------------------------------------------------------------------
def test_linalg_projections():
    rng = np.random.default_rng(0)
    for _ in range(20):
        A = rng.uniform(-1, 1, (6, 6))
        A = (A + A.T) / 2
        np.fill_diagonal(A, 1)
        R = la.nearest_corr(A)
        assert np.allclose(np.diag(R), 1) and np.array_equal(R, R.T)
        assert np.linalg.eigvalsh(R)[0] > 0
        S = la.make_psd(A)
        assert la.min_eig(S) > 0 and np.array_equal(S, S.T)
    good = la.nearest_corr(np.array([[1, .3], [.3, 1.]]))
    assert np.allclose(good, [[1, .3], [.3, 1]])
    X = rng.standard_normal((200, 5))
    d = la.lw_constant_corr_intensity(X)
    assert 0 <= d <= 1
    # Gram-matrix mean-uncertainty factor keeps PSD and matches 1 + h/T on the diagonal
    counts = np.array([[100, 40], [40, 40.]])
    F = la.mean_uncertainty_factor(counts, 12)
    assert np.allclose(np.diag(F), [1.12, 1.3]) and la.min_eig(F - 1) >= -1e-12


# --------------------------------------------------------------------------
# recovery of a known constant covariance
# --------------------------------------------------------------------------
@pytest.mark.parametrize("model", ["sample", "lw", "vc_ewma_ewma_m_lwb", "vc_daily_ewma_m_lwb",
                                   "vc_daily_daily_m_lw", "ewma_cov"])
@pytest.mark.parametrize("h", [1, 3])
def test_constant_covariance_recovered(const_panel, model, h):
    xr, daily = const_panel
    sd = np.array([0.045, 0.02, 0.015, 0.05])
    R = np.full((4, 4), 0.3)
    np.fill_diagonal(R, 1)
    true = h * R * np.outer(sd, sd)
    st = CovSettings(oos_start=str(xr.index[-200].date()), end=str(xr.index[-h - 1].date()))
    df = walk_forward_cov(model, xr, h, daily, st)
    mats = to_matrices(df)
    avg = np.mean([S for _, S in mats.values()], axis=0)
    rel_sd = np.sqrt(np.diag(avg)) / np.sqrt(np.diag(true)) - 1
    Ravg, _ = la.cov_to_corr(avg)
    # one simulated path: a dynamic estimate averaged over 200 origins still has
    # ~5-8% sd noise (checked across seeds; no systematic bias)
    tol_sd = 0.09 if model in ("sample", "lw") else 0.13
    assert np.abs(rel_sd).max() < tol_sd, rel_sd
    off = np.abs(Ravg - R)[np.triu_indices(4, 1)]
    assert off.max() < (0.2 if model == "ewma_cov" else 0.12), off


def test_sample_diag_equals_hist_mean_sd():
    """The benchmark's variances equal the marginal hist_mean baseline's sd^2."""
    from assetpred.evaluation.synthetic import make_synthetic
    from assetpred.evaluation.walkforward import WalkForwardSettings, walk_forward
    data = make_synthetic(n_assets=3, start="1960-01-31", end="1995-12-31", seed=3)
    for h in (1, 12):
        wf = walk_forward("hist_mean", data.features, data.xr, h,
                          WalkForwardSettings(oos_start="1985-01-31", min_history=60))
        cv = walk_forward_cov("sample", data.xr, h, None, CovSettings(oos_start="1985-01-31"))
        diag = cv[cv.asset_i == cv.asset_j].set_index(["date", "asset_i"])["cov"]
        sd = wf.set_index(["date", "asset"])["sd"]
        common = sd.index.intersection(diag.index)
        assert len(common) > 100
        assert np.allclose(np.sqrt(diag.loc[common].to_numpy()), sd.loc[common].to_numpy(), rtol=1e-8)


# --------------------------------------------------------------------------
# regime shift in correlation
# --------------------------------------------------------------------------
def test_regime_shift_tracked_faster_than_sample():
    """Stock-bond-like correlation flips from +0.5 to -0.5 after 300 months."""
    shift = 300
    xr, daily = make_panel(n_months=480, seed=2, late=(),
                           corr_path=lambda m: 0.5 if m < shift else -0.5)
    xr, daily = xr[["us_eq", "ust_int", "tips"]], daily[["us_eq", "ust_int", "tips"]]
    st = CovSettings(oos_start=str(xr.index[shift].date()), end=str(xr.index[-2].date()))
    fast = ["ewma_cov", "vc_ewma_ewma_m_lwb", "vc_daily_ewma_m_lwb", "vc_daily_daily_m_lw", "vc_ewma_ewma_m"]
    slow = ["vc_ewma_ewma_m_lw", "vc_ewma_dcc_m_lw"]   # full LW / DCC: slower, still beat sample
    cov = pd.concat([walk_forward_cov(m, xr, 1, daily, st) for m in ["sample", *fast, *slow]],
                    ignore_index=True)
    rho = {}
    for m, g in cov.groupby("model"):
        mats = to_matrices(g)
        rho[m] = {k: la.cov_to_corr(mats[xr.index[shift + k]][1])[0][0, 1] for k in (24, 96)}
    assert rho["sample"][24] > 0.2 and rho["sample"][96] > 0.1      # still remembers +0.5
    for m in fast:
        assert rho[m][24] < rho["sample"][24] - 0.25, (m, rho)
        assert rho[m][96] < -0.2, (m, rho)
    for m in slow:
        assert rho[m][96] < rho["sample"][96] - 0.25, (m, rho)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sc, _ = scoring.score_covariances(cov, xr, "sample", subperiods=(), reps=99)
    sb = sc[(sc.asset == "sb") & (sc.metric == "diff_nll") & (sc.period == "full")].set_index("model").value
    for m in fast + slow:
        assert sb[m] > 0, (m, sb.to_dict())


# --------------------------------------------------------------------------
# contract: every registered model, PSD and symmetric, expanding universe
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def small_panel():
    xr, daily = make_panel(n_months=240, seed=4)
    return xr, daily, make_features(xr)


@pytest.mark.parametrize("model", sorted(set(available())))
def test_every_model_psd_symmetric(small_panel, model):
    xr, daily, feat = small_panel
    st = CovSettings(oos_start=str(xr.index[90].date()))
    for h in (1, 12):
        df = walk_forward_cov(model, xr, h, daily, st, features=feat)
        assert list(df.columns) == schema.COV_COLUMNS
        engine.validate_cov_frame(df)
        mats = to_matrices(df)
        sizes = [len(a) for a, _ in mats.values()]
        assert min(sizes) == 2 and max(sizes) == 4    # tips enters at 72+60, gold at 120+60
        for _, S in mats.values():
            assert np.array_equal(S, S.T) and la.min_eig(S) >= 0


def test_frozen_list():
    assert 1 <= len(FROZEN) <= 5
    assert all(m in available() for m in FROZEN) and BENCHMARK in available()


# --------------------------------------------------------------------------
# no look-ahead
# --------------------------------------------------------------------------
class _Probe(CovForecaster):
    seen: list = []

    def predict(self, train):
        _Probe.seen.append((train.origin, train.xr.index.max(),
                            None if train.daily is None else train.daily.index.max()))
        return np.eye(len(train.assets)) * 1e-3


def test_training_set_refuses_future():
    xr, daily = make_panel(n_months=100, seed=6, late=())
    t = xr.index[50]
    with pytest.raises(CovLeakageError):
        CovTrainingSet(1, t, xr, tuple(xr.columns))
    with pytest.raises(CovLeakageError):
        CovTrainingSet(1, t, xr.loc[:t], tuple(xr.columns), daily=daily)
    _Probe.seen = []
    _Probe.uses_daily = True
    walk_forward_cov(_Probe, xr, 3, daily, CovSettings(oos_start=str(xr.index[70].date())))
    assert _Probe.seen
    for origin, mx, dx in _Probe.seen:
        assert mx <= origin and dx <= origin


@pytest.mark.parametrize("model", ["sample", "vc_daily_ewma_m_lw", "vc_daily_daily_m_lw",
                                   "vc_ewma_dcc_m_lw", "vc_daily_macro_m_lw", "ewma_mr"])
def test_future_data_cannot_change_past_forecasts(small_panel, model):
    xr, daily, feat = small_panel
    cut = xr.index[180]
    xr2, daily2, feat2 = xr.copy(), daily.copy(), feat.copy()
    rng = np.random.default_rng(9)
    xr2.loc[xr2.index > cut] = xr2.loc[xr2.index > cut] * 3 + rng.standard_normal(
        xr2.loc[xr2.index > cut].shape) * 0.05
    daily2.loc[daily2.index > cut] = -daily2.loc[daily2.index > cut] * 2
    fd = feat2.index.get_level_values("date") > cut
    feat2.loc[fd] = feat2.loc[fd] + 10
    st = CovSettings(oos_start=str(xr.index[150].date()))
    for h in (1, 12):
        a = walk_forward_cov(model, xr, h, daily, st, features=feat)
        b = walk_forward_cov(model, xr2, h, daily2, st, features=feat2)
        a, b = a[a.date <= cut].reset_index(drop=True), b[b.date <= cut].reset_index(drop=True)
        assert len(a) > 0
        pd.testing.assert_frame_equal(a, b)


def test_dev_mode_truncates_inputs():
    from assetpred.covariance import run
    if not schema.ASSET_RETURNS.exists():
        pytest.skip("processed data not built")
    xr, daily, feat = run.load_inputs(dev=True)
    end = pd.Timestamp(config.DEV_END)
    assert xr.index.max() <= end
    assert daily is None or daily.index.max() <= end
    assert feat is None or feat.index.get_level_values("date").max() <= end
    assert engine.dev_end(12) == "2013-12-31" if config.DEV_END == "2014-12-31" else True
    for h in config.HORIZONS:
        last = pd.Timestamp(engine.dev_end(h))
        assert last + pd.offsets.MonthEnd(h) == end


# --------------------------------------------------------------------------
# combinator
# --------------------------------------------------------------------------
def test_combine_matches_marginal_sd_and_keeps_correlation(small_panel):
    xr, daily, _ = small_panel
    st = CovSettings(oos_start=str(xr.index[200].date()))
    cov = walk_forward_cov("vc_ewma_ewma_m_lw", xr, 3, daily, st)
    rng = np.random.default_rng(1)
    rows = []
    for t, (assets, S) in to_matrices(cov).items():
        for a in assets[:-1]:          # one asset missing from the marginals
            rows.append({"model": "m", "horizon": 3, "date": t, "asset": a, "sd": rng.uniform(.01, .2)})
    marg = pd.DataFrame(rows)
    joint = cmb.combine(cov, marg)
    engine.validate_cov_frame(joint)
    assert joint.model.iloc[0] == "vc_ewma_ewma_m_lw__sd_m"
    jm, cm = to_matrices(joint), to_matrices(cov)
    for t, (assets, S) in jm.items():
        sd = marg[marg.date == t].set_index("asset").loc[list(assets), "sd"].to_numpy()
        assert np.allclose(np.sqrt(np.diag(S)), sd)
        a0, S0 = cm[t]
        ix = [a0.index(a) for a in assets]
        assert np.allclose(la.cov_to_corr(S)[0], la.cov_to_corr(S0[np.ix_(ix, ix)])[0], atol=1e-10)


def test_vc_ewma_diag_equals_vol_ewma_sd():
    """vc_ewma_* variances are exactly the vol_ewma marginal baseline's sd^2."""
    from assetpred.evaluation.synthetic import make_synthetic
    from assetpred.evaluation.walkforward import WalkForwardSettings, walk_forward
    data = make_synthetic(n_assets=3, start="1960-01-31", end="1995-12-31", seed=7, stoch_vol=True)
    for h in (1, 12):
        wf = walk_forward("vol_ewma", data.features, data.xr, h,
                          WalkForwardSettings(oos_start="1985-01-31", min_history=60))
        cv = walk_forward_cov("vc_ewma_ewma_m_lw", data.xr, h, None, CovSettings(oos_start="1985-01-31"))
        diag = cv[cv.asset_i == cv.asset_j].set_index(["date", "asset_i"])["cov"]
        sd = wf.set_index(["date", "asset"])["sd"]
        common = sd.index.intersection(diag.index)
        assert np.allclose(np.sqrt(diag.loc[common].to_numpy()), sd.loc[common].to_numpy(), rtol=1e-8)


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------
class _Oracle(CovForecaster):
    scale = 1.0

    def predict(self, train):
        sd = np.array([0.045, 0.02, 0.015, 0.05])[:len(train.assets)]
        R = np.full((len(sd), len(sd)), 0.3)
        np.fill_diagonal(R, 1)
        return self.scale * train.horizon * R * np.outer(sd, sd)


class _Wrong(_Oracle):
    scale = 4.0


def test_scoring_prefers_truth_and_reports_overlap(const_panel):
    xr, _ = const_panel
    frames = []
    for h in (1, 12):
        st = CovSettings(oos_start=str(xr.index[200].date()))
        for f, name in ((_Oracle, "sample"), (_Wrong, "wrong")):
            frames.append(walk_forward_cov(f, xr, h, None, st, name=name))
    cov = pd.concat(frames, ignore_index=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sc, per_date = scoring.score_covariances(cov, xr, "sample", subperiods=(), reps=99)
    full = sc[sc.period == "full"]

    def get(m, h, tg, met, bm="none"):
        v = full[(full.model == m) & (full.horizon == h) & (full.asset == tg) & (full.metric == met)
                 & (full.benchmark == bm)].value
        assert len(v) == 1, (m, h, tg, met, bm)
        return float(v.iloc[0])
    for h in (1, 12):
        assert get("wrong", h, "joint", "diff_nll", "sample") < 0
        assert get("wrong", h, "joint", "dm_stat_nll", "sample") < 0
        assert abs(get("sample", h, "joint", "mahal") - 1) < 0.25
        nd = get("*", h, "joint", "n_dates")
        assert get("*", h, "joint", "n_indep") == pytest.approx(nd / h)
    assert np.isfinite(get("wrong", 12, "joint", "strand_p_median_nll", "sample"))
    assert np.isnan(get("wrong", 1, "joint", "strand_p_median_nll", "sample"))
