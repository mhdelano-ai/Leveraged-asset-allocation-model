"""Distribution models (``assetpred.models.dist_*``) on synthetic panels:
contract, separation of location from scale/shape, the combinator, the
component refit schedule, and calibration / volatility tracking."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from assetpred import schema
from assetpred.evaluation import scoring
from assetpred.evaluation.synthetic import make_synthetic
from assetpred.evaluation.walkforward import WalkForwardSettings, walk_forward
from assetpred.models import dist_core, registry
from assetpred.models.base import TrainingSet
from assetpred.models.dist_core import (Composite, MeanPlusDist, refit_origin, register_combo,
                                        truncate, weighted_quantile)
from assetpred.models.dist_scale import AggScale, EWMAScale, _fit_affine_qlike, qlike_fit

QC = schema.QUANTILE_COLUMNS
S = WalkForwardSettings(oos_start="1980-01-31", refit_every=12, min_history=60)
NOMINATED = ["dist_ewma_fhs", "dist_lin_agg", "dist_lin_agg_grp", "dist_gbm_agg", "dist_lin_agg_cshape"]


def _wf(model, data, h, settings=S):
    return walk_forward(model, data.features, data.xr, h, settings)


def _pooled(fc, xr, model, metric, bench="none"):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        sc, _ = scoring.score_forecasts(fc, xr, subperiods=(), mt_reps=49, multiple_testing=False)
    s = sc[(sc.model == model) & (sc.asset == "pooled") & (sc.metric == metric)
           & (sc.benchmark == bench) & (sc.period == "full")]
    return s.set_index("horizon")["value"]


@pytest.fixture(scope="module")
def sv_data():
    return make_synthetic(n_assets=4, start="1955-01-31", end="2008-12-31", stoch_vol=True, seed=21)


@pytest.fixture(scope="module")
def null_data():
    return make_synthetic(n_assets=4, start="1955-01-31", end="2008-12-31", stoch_vol=False, seed=22)


# --------------------------------------------------------------------------
def test_registered_and_picklable():
    import pickle
    names = registry.available()
    for n in NOMINATED + ["dist_gbm_vol", "dist_gbm_all", "dist_lin", "dist_ens"]:
        assert n in names
        pickle.dumps(registry.get_factory(n))
        assert isinstance(registry.create(n), Composite)


@pytest.mark.parametrize("h", [1, 12])
def test_dist_ewma_fhs_reproduces_vol_ewma_fhs(sv_data, h):
    a = _wf("vol_ewma_fhs", sv_data, h)
    b = _wf("dist_ewma_fhs", sv_data, h)
    np.testing.assert_allclose(a[["mean", "sd", *QC]].to_numpy(), b[["mean", "sd", *QC]].to_numpy(),
                               rtol=0, atol=1e-12)


@pytest.mark.parametrize("model", NOMINATED)
def test_contract_and_mean_is_historical(sv_data, model):
    """Valid forecasts (walk_forward validates), every eligible row forecast,
    and the location is exactly hist_mean's: all skill is distributional."""
    for h in (1, 3):
        f = _wf(model, sv_data, h)
        ref = _wf("hist_mean", sv_data, h)
        assert len(f) == len(ref)
        np.testing.assert_allclose(f["mean"].to_numpy(), ref["mean"].to_numpy(), atol=1e-12)
        assert (np.diff(f[QC].to_numpy(), axis=1) >= -1e-12).all()


def test_component_cache_does_not_change_results(sv_data):
    dist_core.clear_cache()
    a = _wf("dist_gbm_agg", sv_data, 3)
    b = _wf("dist_gbm_agg", sv_data, 3)          # second run hits the cache
    dist_core.clear_cache()
    np.testing.assert_array_equal(a[QC].to_numpy(), b[QC].to_numpy())


def test_refit_schedule_and_truncation(sv_data):
    xr = sv_data.xr
    h = 3
    t = pd.Timestamp("1996-07-31")
    y = schema.make_targets(xr.loc[:t], h).stack().dropna()
    feats = sv_data.features.sort_index()
    ya = y.reindex(feats.index)
    cut = t - pd.offsets.MonthEnd(h)
    k = int((feats.index.get_level_values("date") <= cut).sum())
    tr = TrainingSet(horizon=h, origin=t, xr=xr.loc[:t], assets=tuple(xr.columns), seed=0,
                     _features=feats.iloc[:k], _targets=ya.iloc[:k])
    t0 = refit_origin(tr, 12)
    assert t0 <= t and t0.month == 1          # anchored on config.OOS_START (a January)
    assert refit_origin(tr, 1) == t
    sub = truncate(tr, t0)
    assert sub.xr.index.max() == t0
    assert sub.y.index.get_level_values("date").max() <= t0 - pd.offsets.MonthEnd(h)
    # targets in the truncated set are realized by t0
    assert np.isfinite(sub.y).all()


def test_derived_state_is_causal(sv_data):
    xr = sv_data.xr
    full = dist_core.derived_long(xr)
    part = dist_core.derived_long(xr.loc[:"1990-12-31"])
    common = part.index
    np.testing.assert_allclose(full.loc[common].to_numpy(), part.to_numpy(), equal_nan=True)


def test_combinator_takes_mean_from_mean_model_and_shape_from_dist(sv_data):
    name = register_combo("ridge", "dist_lin_agg", name="test_ridge__dist_lin_agg")
    combo = _wf(name, sv_data, 1)
    dist = _wf("dist_lin_agg", sv_data, 1)
    # quantile offsets from the location are the dist model's
    np.testing.assert_allclose((combo[QC].to_numpy() - combo[["mean"]].to_numpy()),
                               (dist[QC].to_numpy() - dist[["mean"]].to_numpy()), atol=1e-10)
    assert not np.allclose(combo["mean"], dist["mean"])
    # with hist_mean as the mean model the combo is the dist model itself
    name2 = register_combo("hist_mean", "dist_lin_agg", name="test_hm__dist_lin_agg")
    same = _wf(name2, sv_data, 1)
    np.testing.assert_allclose(same[["mean", *QC]].to_numpy(), dist[["mean", *QC]].to_numpy(), atol=1e-12)
    # cache fingerprint of a combo identifies the mean model
    p = registry.create(name).params()
    assert "ridge" in p["location"] and p["mean_model"] == "ridge"
    registry.unregister(name)
    registry.unregister(name2)


def test_combinator_rejects_non_dist_model():
    with pytest.raises(TypeError):
        MeanPlusDist("ridge", "vol_ewma")


def test_qlike_and_affine_fits_recover_truth():
    rng = np.random.default_rng(0)
    n = 20000
    x = rng.standard_normal(n)
    D = np.column_stack([np.ones(n), x])
    s2 = np.exp(-0.5 + 0.7 * x)
    u = s2 * rng.standard_normal(n) ** 2
    b = qlike_fit(D, u, ridge=0)
    np.testing.assert_allclose(b, [-0.5, 0.7], atol=0.05)
    r = np.exp(rng.standard_normal(n))
    u2 = (0.4 + 0.6 * r) * rng.standard_normal(n) ** 2
    a, bb = _fit_affine_qlike(u2, r)
    assert abs(a - 0.4) < 0.08 and abs(bb - 0.6) < 0.08


def test_weighted_quantile_matches_unweighted():
    x = np.random.default_rng(1).standard_normal(5001)
    q = weighted_quantile(x, np.ones_like(x), np.array([0.05, 0.5, 0.95]))
    np.testing.assert_allclose(q, np.quantile(x, [0.05, 0.5, 0.95]), atol=0.01)


# --------------------------------------------------------------------------
# calibration under the null, and volatility tracking under stochastic vol
@pytest.mark.parametrize("model", ["dist_lin_agg", "dist_gbm_agg"])
def test_calibrated_under_constant_vol(null_data, model):
    fc = pd.concat([_wf(m, null_data, h) for m in ("hist_mean", "vol_ewma", model) for h in (1, 12)],
                   ignore_index=True)
    cov = _pooled(fc, null_data.xr, model, "cover_90")
    calib = _pooled(fc, null_data.xr, model, "calib_p")
    assert ((cov > 0.86) & (cov < 0.94)).all(), cov
    assert (calib > 0.01).all(), calib


@pytest.mark.parametrize("model", ["dist_lin_agg", "dist_gbm_agg", "dist_lin_agg_cshape"])
def test_not_worse_than_vol_ewma_under_stochastic_vol(sv_data, model):
    fc = pd.concat([_wf(m, sv_data, h) for m in ("hist_mean", "vol_ewma", model) for h in (1, 3)],
                   ignore_index=True)
    sk = _pooled(fc, sv_data.xr, model, "crps_skill", "vol_ewma")
    sk_hist = _pooled(fc, sv_data.xr, model, "crps_skill", "hist_mean")
    assert (sk > -0.005).all(), sk
    assert (sk_hist > 0.0).all(), sk_hist
