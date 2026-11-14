import numpy as np
import pandas as pd
import pytest
from dataclasses import replace

from volmodel.data.synthetic import make_synthetic_panel
from volmodel.eval.targets import compute_targets
from volmodel.models import get_model, list_models
from volmodel.models.ml import SPECS
from volmodel.models.ml.features import CORE_FEATURES, FeatureState, realized_targets
from volmodel.models.ml.learners import blocked_folds
from volmodel.models.ml.model import MLVol, training_date

NAMES = sorted(SPECS)
EARLY, MIDDLE, LATE = "1979-12-31", "1996-06-30", "2010-12-31"


@pytest.fixture(scope="module")
def panel():
    return make_synthetic_panel(seed=3)


def _model(name, min_years=10):
    m = get_model("vol", name)
    m.min_years = min_years          # the synthetic panel is only 41 years long
    return m


def test_registered():
    got = set(list_models("vol"))
    assert set(NAMES) <= got and len(NAMES) <= 8


@pytest.mark.parametrize("name", NAMES)
def test_valid_cold_warm_and_order(panel, name):
    cold = {}
    for t in [EARLY, MIDDLE, LATE]:
        view = panel.asof(t)
        assets = view.eligible()
        v = _model(name).forecast_vol(view, assets)
        assert list(v.index) == assets and np.isfinite(v).all() and (v > 0).all()
        cold[t] = v
    # early origin has monthly-only assets
    assert len(panel.asof(EARLY).has_daily()) == 0
    # warm, reverse order, with extra origins in between
    m = _model(name)
    for t in [LATE, "2003-06-30", MIDDLE, "1985-12-31", EARLY, LATE]:
        view = panel.asof(t)
        v = m.forecast_vol(view, view.eligible())
        if t in cold:
            pd.testing.assert_series_equal(v, cold[t], rtol=0, atol=0)


@pytest.mark.parametrize("name", ["ml_glm", "ml_gbm_mono", "ml_rf", "ml_gbm_mono_macro"])
def test_no_lookahead(panel, name):
    t = pd.Timestamp(LATE)
    view = panel.asof(t)
    base = _model(name).forecast_vol(view, view.eligible())
    rng = np.random.default_rng(1)
    d, mth = panel.daily.copy(), panel.monthly.copy()
    d.loc[d.index > t] = d.loc[d.index > t] * 3 + rng.normal(0, 0.01, d.loc[d.index > t].shape)
    mth.loc[mth.index > t] = mth.loc[mth.index > t] * 3
    pred = panel.predictors.copy()
    pred.loc[pred["available_date"] > t, "value"] *= 5
    altered = replace(panel, daily=d, monthly=mth, predictors=pred, _shared={})
    v2 = _model(name).forecast_vol(altered.asof(t), view.eligible())
    pd.testing.assert_series_equal(base, v2, rtol=0, atol=0)


def test_targets_match_eval_targets():
    p = make_synthetic_panel(seed=1)
    T = compute_targets(p)
    view = p.asof(p.monthly.index[-1])
    P, A, V = realized_targets(view)
    mine = pd.DataFrame({"origin": view.monthly.index[P], "asset": np.array(view.monthly.columns)[A], "v2": V})
    m = T.vol.merge(mine, on=["origin", "asset"], how="outer")
    assert m["var"].notna().all() and m["v2"].notna().all()
    np.testing.assert_allclose(m["var"], m["v2"], rtol=1e-10)


def test_features_causal(panel):
    """Feature values at month s do not depend on data after s."""
    s = pd.Timestamp("2001-06-30")
    a = FeatureState(panel.asof(s))
    b = FeatureState(panel.asof("2012-12-31"))
    g = a.row_at(s)
    for f in CORE_FEATURES:
        np.testing.assert_allclose(a.F[f][g], b.F[f][b.row_at(s)], rtol=1e-12, equal_nan=True)


def test_daily_scale_ratio_bounded(panel):
    fs = FeatureState(panel.asof(LATE))
    assert np.all(np.abs(fs.lrho) <= 0.5)


def test_blocked_folds_embargo():
    o = pd.date_range("1950-01-31", "1999-12-31", freq="ME")
    origin = np.repeat(o.values, 3)          # 3 assets per origin
    n = 0
    for tr, va in blocked_folds(origin):
        n += 1
        vo, to = pd.DatetimeIndex(origin[va]), pd.DatetimeIndex(origin[tr])
        # every origin is entirely in or out (blocked across assets)
        assert not set(vo) & set(to)
        gap = np.min(np.abs((to.values[:, None] - vo.values[None, :]).astype("timedelta64[D]").astype(int)))
        assert gap > 360
    assert n == 5


def test_fallback_before_min_years(panel):
    m = _model("ml_gbm_mono", min_years=20)
    view = panel.asof("1990-06-30")              # < 20 years of training origins
    v = m.forecast_vol(view, view.eligible())
    ref = get_model("vol", "ref_blend").forecast_vol(view, view.eligible())
    np.testing.assert_allclose(v.values, ref.values, rtol=1e-12)


def test_training_date():
    assert training_date("2001-12-31") == pd.Timestamp("2001-12-31")
    assert training_date("2001-11-30") == pd.Timestamp("2000-12-31")
