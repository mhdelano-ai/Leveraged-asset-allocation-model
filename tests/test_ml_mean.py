"""Tests for the standardized-deviation mean models in ``assetpred.models.ml_mean``."""
from __future__ import annotations

import pickle
import warnings

import numpy as np
import pandas as pd
import pytest

from assetpred import schema
from assetpred.evaluation import scoring
from assetpred.evaluation.synthetic import make_synthetic
from assetpred.evaluation.walkforward import WalkForwardSettings, walk_forward
from assetpred.models import ml_mean, registry
from assetpred.models.base import TrainingSet
from assetpred.models.baselines import HistoricalMean

SETTINGS = WalkForwardSettings(oos_start="1980-01-31", refit_every=12, min_history=60)
NAMES = ["ridge_z", "ridge_zg", "combo_z", "gbm_z", "rf_z", "ens_z", "gbm_z_nocal", "gbm_z_cv"]


def _train(data, h, origin="1990-12-31"):
    origin = pd.Timestamp(origin)
    xr = data.xr.loc[:origin]
    y = schema.make_targets(xr, h).stack().dropna()
    y.index = y.index.set_names(["date", "asset"])
    F = data.features.sort_index().loc[: origin - pd.offsets.MonthEnd(h)]
    assets = tuple(a for a in xr.columns if xr[a].notna().sum() >= 60)
    return TrainingSet(horizon=h, origin=origin, xr=xr, assets=assets, seed=7,
                       _features=F, _targets=y.reindex(F.index))


def _r2(fc, xr, model, h):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        sc, _ = scoring.score_forecasts(fc, xr, subperiods=(), mt_reps=99, multiple_testing=False)
    s = sc[(sc.model == model) & (sc.horizon == h) & (sc.asset == "pooled") & (sc.period == "full")
           & (sc.benchmark == "hist_mean")].set_index("metric").value
    return float(s["r2_oos"]), float(s["cw_p"])


@pytest.fixture(scope="module")
def noise():
    return make_synthetic(n_assets=4, beta=0.0, start="1960-01-31", end="2008-12-31", seed=21)


@pytest.fixture(scope="module")
def signal():
    return make_synthetic(n_assets=4, beta=0.35, start="1960-01-31", end="2008-12-31", seed=22)


def test_registered_and_picklable():
    for n in NAMES + ["gbm_z.price", "combo_z.nohind"]:
        f = registry.get_factory(n)
        pickle.loads(pickle.dumps(f))
        assert isinstance(f(), ml_mean.ZScoreMean)


def test_resolve_columns_falls_back_on_foreign_panel(noise):
    cols = list(noise.features.columns)
    assert ml_mean.resolve_columns("price", cols) == cols
    assert ml_mean.resolve_columns(None, cols) == cols


def test_resolve_columns_on_real_panel():
    pytest.importorskip("pyarrow")
    if not schema.FEATURES.exists():
        pytest.skip("no processed feature panel")
    from assetpred.features import select
    cols = list(schema.load_features().columns)
    assert ml_mean.resolve_columns("all", cols) == select.columns()
    price = ml_mean.resolve_columns("price", cols)
    assert set(price) == set(select.columns(families=("price",)))
    nohind = ml_mean.resolve_columns("all-nohind", cols)
    assert set(nohind) == set(select.columns(hindsight=False)) and len(nohind) < len(cols)
    assert ml_mean.resolve_columns("macro+sentiment-nohind", cols) == \
        [c for c in cols if c in set(select.columns(families=("macro", "sentiment"), hindsight=False))]


@pytest.mark.parametrize("h", [1, 12])
def test_anchor_is_hist_mean_and_zero_shrink_reproduces_it(noise, h):
    tr = _train(noise, h)
    m = registry.create("ridge_z").fit(tr)
    hm = HistoricalMean().fit(tr)
    X = noise.features.sort_index().loc[[tr.origin]]
    X = X[X.index.get_level_values("asset").isin(tr.assets)]
    np.testing.assert_allclose(m._anchor_of(X.index.get_level_values("asset"), final=True),
                               hm.predict(X)["mean"].to_numpy(), rtol=1e-12)
    assert 0.0 <= m.b_ <= 1.0
    m._b = 0.0
    np.testing.assert_allclose(m.predict(X)["mean"].to_numpy(), hm.predict(X)["mean"].to_numpy(), rtol=1e-12)


def test_calibration_slope_shrinks_on_noise_and_not_on_signal(noise, signal):
    bn = registry.create("gbm_z").fit(_train(noise, 1)).b_
    bs = registry.create("gbm_z").fit(_train(signal, 1)).b_
    assert bs > 0.5
    assert bn < bs


def test_learners_handle_nan_and_unknown_asset(signal):
    tr = _train(signal, 1)
    m = registry.create("ens_z").fit(tr)
    X = signal.features.sort_index().loc[[tr.origin]].copy()
    X.iloc[0, :] = np.nan
    out = m.predict(X)
    assert np.isfinite(out.to_numpy()).all()


@pytest.mark.parametrize("name", ["ridge_z", "combo_z", "gbm_z"])
def test_no_edge_on_noise(noise, name):
    fc = pd.concat([walk_forward(n, noise.features, noise.xr, 1, SETTINGS) for n in ("hist_mean", "vol_ewma", name)])
    r2, cw = _r2(fc, noise.xr, name, 1)
    assert r2 < 0.005, r2


@pytest.mark.parametrize("name", ["ridge_z", "combo_z", "gbm_z"])
def test_planted_signal_recovered(signal, name):
    fc = pd.concat([walk_forward(n, signal.features, signal.xr, 1, SETTINGS) for n in ("hist_mean", "vol_ewma", name)])
    r2, cw = _r2(fc, signal.xr, name, 1)
    assert r2 > 0.02 and cw < 0.01, (r2, cw)
