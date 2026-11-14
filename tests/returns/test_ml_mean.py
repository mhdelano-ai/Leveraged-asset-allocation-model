"""Tests for the standardized-deviation mean models in
``assetforecast.returns.models.ml_mean`` (ported from assetpred), the pooling
groups and the pinned feature list."""
from __future__ import annotations

import pickle
import warnings

import numpy as np
import pandas as pd
import pytest

from assetforecast.returns import config, schema
from assetforecast.returns.evaluation import scoring
from assetforecast.returns.evaluation.synthetic import make_synthetic
from assetforecast.returns.evaluation.walkforward import WalkForwardSettings, walk_forward
from assetforecast.returns.models import ml_mean, registry
from assetforecast.returns.models.base import TrainingSet
from assetforecast.returns.models.baselines import HistoricalMean
from assetforecast.universe import ASSET_IDS

SETTINGS = WalkForwardSettings(oos_start="1980-01-31", refit_every=12, min_history=60)
NAMES = [*config.FROZEN_MEAN, *config.ROBUSTNESS]


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
    assert len(NAMES) == 8
    for n in NAMES:
        f = registry.get_factory(n)
        pickle.loads(pickle.dumps(f))
        assert isinstance(f(), ml_mean.ZScoreMean)


def test_resolve_columns_falls_back_on_foreign_panel(noise):
    cols = list(noise.features.columns)
    assert ml_mean.resolve_columns("price", cols) == cols
    assert ml_mean.resolve_columns(None, cols) == cols


def test_registered_settings_are_the_frozen_ones():
    want = {
        "ridge_z": (("ridge",), "all"), "combo_z": (("combo",), "all"), "gbm_z": (("gbm",), "all"),
        "rf_z": (("rf",), "all"), "ens_z": (("ridge", "combo", "gbm", "rf"), "all"),
        "gbm_z.price": (("gbm",), "price"),
        "gbm_z.nohind": (("gbm",), "all-nohind"), "combo_z.nohind": (("combo",), "all-nohind"),
    }
    for n, (learners, fs) in want.items():
        m = registry.get_factory(n)()
        assert m.params() == {"calibrate": True, "feature_set": fs, "learners": learners}, n


def test_asset_group_covers_the_universe():
    assert set(config.ASSET_GROUP) == set(ASSET_IDS)
    assert ml_mean.ASSET_GROUP is config.ASSET_GROUP
    assert set(config.ASSET_GROUP.values()) == set(ml_mean.GROUPS)
    # assetpred's grouping carried over to the new ids; REITs stay in equity
    # (unlike universe.Asset.group) and us_small joins equity
    assert {a for a, g in config.ASSET_GROUP.items() if g == "equity"} == \
        {"us_large", "us_small", "intl_dev", "em", "reits"}
    assert {a for a, g in config.ASSET_GROUP.items() if g == "rates"} == {"ust_int", "ust_long", "tips"}
    assert {a for a, g in config.ASSET_GROUP.items() if g == "credit"} == {"ig_credit", "hy_credit"}
    assert {a for a, g in config.ASSET_GROUP.items() if g == "real"} == {"gold", "commodities"}
    with pytest.raises(KeyError, match="pooling group"):
        ml_mean._group_codes(["us_large", "us_eq"])


@pytest.mark.realdata
def test_resolve_columns_on_the_committed_panel():
    """Kept from assetpred's test_ml_mean, against the pinned list."""
    from assetforecast import paths
    if not paths.FEATURE_META.exists():
        pytest.skip("no committed feature panel")
    meta = pd.read_csv(paths.FEATURE_META, keep_default_na=False).set_index("name")
    cols = list(meta.index)
    frozen = list(config.FROZEN_FEATURE_COLUMNS)
    hind = meta["hindsight"].astype(str) == "True"
    assert ml_mean.resolve_columns("all", cols) == frozen
    assert ml_mean.resolve_columns("price", cols) == [c for c in frozen if meta.at[c, "family"] == "price"]
    nohind = ml_mean.resolve_columns("all-nohind", cols)
    assert nohind == [c for c in frozen if not hind[c]] and len(nohind) < len(frozen)
    assert ml_mean.resolve_columns("macro+sentiment-nohind", cols) == \
        [c for c in frozen if meta.at[c, "family"] in ("macro", "sentiment") and not hind[c]]


def test_pinned_feature_list():
    cols = config.FROZEN_FEATURE_COLUMNS
    assert len(cols) == 82 and len(set(cols)) == 82
    assert "bw_sentiment" not in cols and "bw_sentiment_pit" not in cols


@pytest.mark.realdata
def test_pinned_feature_list_matches_assetpred_rule():
    """Derive the list again from the archived panel with assetpred's selection rule
    (drop features whose data ended more than 12 months before the panel's end)."""
    from assetforecast import paths
    d = paths.ARCHIVE_AP / "data"
    if not (d / "features.parquet").exists():
        pytest.skip("no archived assetpred inputs")
    meta = pd.read_csv(d / "feature_meta.csv").set_index("name")
    f = pd.read_parquet(d / "features.parquet")
    dates = f.index.get_level_values("date")
    last = pd.Series({c: dates[f[c].notna().to_numpy()].max() for c in f.columns})
    stale = last < dates.max() - pd.offsets.MonthEnd(12)
    assert sorted(stale[stale].index) == ["bw_sentiment", "bw_sentiment_pit"]
    assert list(config.FROZEN_FEATURE_COLUMNS) == [c for c in meta.index if not stale[c]]


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


@pytest.mark.slow
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


@pytest.mark.slow
@pytest.mark.parametrize("name", ["ridge_z", "combo_z", "gbm_z"])
def test_no_edge_on_noise(noise, name):
    fc = pd.concat([walk_forward(n, noise.features, noise.xr, 1, SETTINGS) for n in ("hist_mean", "vol_ewma", name)])
    r2, cw = _r2(fc, noise.xr, name, 1)
    assert r2 < 0.005, r2


@pytest.mark.slow
@pytest.mark.parametrize("name", ["ridge_z", "combo_z", "gbm_z"])
def test_planted_signal_recovered(signal, name):
    fc = pd.concat([walk_forward(n, signal.features, signal.xr, 1, SETTINGS) for n in ("hist_mean", "vol_ewma", name)])
    r2, cw = _r2(fc, signal.xr, name, 1)
    assert r2 > 0.02 and cw < 0.01, (r2, cw)
