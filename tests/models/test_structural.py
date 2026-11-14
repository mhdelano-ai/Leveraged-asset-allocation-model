"""Tests for the structural family (src/assetforecast/models/structural)."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

import assetforecast.models.base as base
import assetforecast.models.structural  # noqa: F401  (registers the family)
from assetforecast.data.panel import PREDICTOR_COLUMNS, Panel
from assetforecast.data.synthetic import make_synthetic_panel
from assetforecast.models.structural import core
from assetforecast.models.structural.factor import StructuralFactorCov
from assetforecast.models.structural.vol import StructuralState, StructuralVol, ref_blend_var

VOL_MODELS = ["structural_normal", "structural_lognormal", "structural_sqrt", "structural_cev"]
COV_MODELS = ["structural_factor_lognormal", "structural_factor_normal", "structural_factor_global"]
ORIGINS = ["1979-12-31", "1992-06-30", "2012-12-31"]   # early one: monthly data only


def _model(kind, name):
    m = base._REGISTRY[kind][name]()
    m.name = name
    return m


def _with_predictors(p: Panel, moodys_end: str | None = None, seed: int = 1) -> Panel:
    """Synthetic panel plus the predictors this family reads (monthly yields,
    one daily yield series, spreads, real yields), so the structural path runs."""
    rng = np.random.default_rng(seed)
    m = p.monthly.index
    n = len(m)
    y = np.clip(0.06 + np.cumsum(rng.normal(0, 0.002, n)), 0.005, 0.15)
    s = np.clip(0.02 + np.cumsum(rng.normal(0, 0.001, n)), 0.004, 0.06)
    ry = y - 0.025
    rows = [
        pd.DataFrame({"obs_date": m, "available_date": m, "series": "macro_long_yield", "value": y}),
        pd.DataFrame({"obs_date": m, "available_date": m + pd.Timedelta(days=1),
                      "series": "credit_moodys_baa_ust10y_spread_m", "value": s}),
        pd.DataFrame({"obs_date": m, "available_date": m + pd.offsets.MonthEnd(1),
                      "series": "credit_gz_spread_m", "value": 0.8 * s}),
        pd.DataFrame({"obs_date": m, "available_date": m + pd.Timedelta(days=14),
                      "series": "tips_real_yield_10y", "value": ry}),
    ]
    d = p.daily.index
    yd = pd.Series(y, index=m).reindex(d, method="bfill").to_numpy() + rng.normal(0, 1e-4, len(d))
    rows.append(pd.DataFrame({"obs_date": d, "available_date": d, "series": "ust_cmt_yield_10y", "value": yd}))
    pred = pd.concat([p.predictors] + rows, ignore_index=True)[PREDICTOR_COLUMNS]
    if moodys_end is not None:
        cut = (pred["series"] == "credit_moodys_baa_ust10y_spread_m") & (pred["obs_date"] > moodys_end)
        pred = pred[~cut]
    pred = pred.sort_values("available_date", kind="stable").reset_index(drop=True)
    return Panel(daily=p.daily, monthly=p.monthly, cash_daily=p.cash_daily, cash_monthly=p.cash_monthly,
                 tiers=p.tiers, predictors=pred)


@pytest.fixture(scope="module")
def plain():
    return make_synthetic_panel(seed=3)


@pytest.fixture(scope="module")
def rich(plain):
    return _with_predictors(plain)


def test_registered():
    assert set(VOL_MODELS) <= set(base._REGISTRY["vol"])
    assert set(COV_MODELS) <= set(base._REGISTRY["cov"])
    assert sum(n.startswith("structural_") for n in base._REGISTRY["vol"]) <= 8
    assert sum(n.startswith("structural_") for n in base._REGISTRY["cov"]) <= 4


@pytest.mark.parametrize("which", ["plain", "rich"])
@pytest.mark.parametrize("name", VOL_MODELS)
def test_vol_valid_at_every_stage(request, which, name):
    panel = request.getfixturevalue(which)
    m = _model("vol", name)
    for t in ORIGINS:
        view = panel.asof(t)
        assets = view.eligible()
        v = m.forecast_vol(view, assets)
        assert list(v.index) == assets
        assert np.isfinite(v).all() and (v > 0).all()


@pytest.mark.parametrize("which", ["plain", "rich"])
@pytest.mark.parametrize("name", COV_MODELS)
def test_cov_valid_and_positive_definite(request, which, name):
    panel = request.getfixturevalue(which)
    m = _model("cov", name)
    for t in ORIGINS:
        view = panel.asof(t)
        assets = view.eligible()
        c = m.forecast_cov(view, assets)
        a = c.to_numpy()
        assert list(c.index) == assets and list(c.columns) == assets
        assert np.isfinite(a).all() and np.allclose(a, a.T)
        assert np.linalg.eigvalsh(a).min() > 0


def test_plain_panel_falls_back_to_ref_blend(plain):
    """Without yields or spreads every asset gets ref_blend exactly."""
    view = plain.asof("2012-12-31")
    assets = view.eligible()
    v = _model("vol", "structural_normal").forecast_vol(view, assets)
    np.testing.assert_allclose(v.values, np.sqrt(ref_blend_var(view, assets).values))


def test_structural_path_is_used(rich):
    view = rich.asof("2012-12-31")
    assets = view.eligible()
    v = _model("vol", "structural_normal").forecast_vol(view, assets)
    ref = np.sqrt(ref_blend_var(view, assets))
    for a in ["ust_int", "ust_long", "tips", "ig_credit", "hy_credit"]:
        assert abs(v[a] / ref[a] - 1) > 1e-6, a
    for a in ["us_large", "gold", "commodities"]:
        assert v[a] == pytest.approx(ref[a])


@pytest.mark.parametrize("kind,name", [("vol", "structural_cev"), ("vol", "structural_lognormal"),
                                       ("cov", "structural_factor_global")])
def test_cold_equals_warm_and_order_free(rich, kind, name):
    t = "2005-06-30"
    def call(m, s):
        view = rich.asof(s)
        out = m.forecast_vol(view, view.eligible()) if kind == "vol" else m.forecast_cov(view, view.eligible())
        return out
    cold = call(_model(kind, name), t)
    m = _model(kind, name)
    for s in ["2012-12-31", "1990-01-31", t, "1999-12-31"]:
        out = call(m, s)
        if s == t:
            warm = out
    again = call(m, t)
    pd.testing.assert_frame_equal(pd.DataFrame(cold), pd.DataFrame(warm))
    pd.testing.assert_frame_equal(pd.DataFrame(cold), pd.DataFrame(again))


@pytest.mark.parametrize("kind,name", [("vol", "structural_cev"), ("vol", "structural_normal"),
                                       ("cov", "structural_factor_lognormal")])
def test_no_lookahead(rich, kind, name):
    t = pd.Timestamp("2004-03-31")
    m = _model(kind, name)
    view = rich.asof(t)
    assets = view.eligible()
    f = (lambda v: m.forecast_vol(v, assets)) if kind == "vol" else (lambda v: m.forecast_cov(v, assets))
    before = f(view)
    # scramble everything after t: returns and predictors
    rng = np.random.default_rng(7)
    daily = rich.daily.copy()
    later = daily.index > t
    daily.loc[later] = daily.loc[later].to_numpy() * 5 + rng.normal(0, 0.01, daily.loc[later].shape)
    monthly = rich.monthly.copy()
    ml = monthly.index > t
    monthly.loc[ml] = monthly.loc[ml].to_numpy() * -3
    pred = rich.predictors.copy()
    pl = pred["obs_date"] > t
    pred.loc[pl, "value"] = pred.loc[pl, "value"] * 10
    # and data published after t about earlier dates
    pa = (pred["available_date"] > t) & ~pl
    pred.loc[pa, "value"] = pred.loc[pa, "value"] * 10
    altered = replace(rich, daily=daily, monthly=monthly, predictors=pred, _shared={})
    after = f(altered.asof(t))
    pd.testing.assert_frame_equal(pd.DataFrame(before), pd.DataFrame(after))


def test_par_duration_matches_closed_form():
    # 10y par bond at 6% (semiannual): (1 - 1.03^-20) / 0.06
    assert core.par_duration(0.06, [10.0]) == pytest.approx((1 - 1.03 ** -20) / 0.06)
    assert core.par_duration(0.0, [10.0]) == pytest.approx(10.0, rel=1e-4)
    # duration falls as yields rise; the long ladder far more than the short
    d_lo, d_hi = core.par_duration([0.02, 0.15], core.MAT_LONG)
    assert d_lo > 2 * d_hi
    assert core.par_duration(0.04, core.MAT_INT) == pytest.approx(7.4, abs=0.3)


def test_normal_model_scales_with_todays_duration(rich):
    """Same yield-vol history, longer duration today -> proportionally higher vol."""
    view = rich.asof("2012-12-31")
    st = StructuralState(view, beta=0.0)
    V, D = st.rate_var("int")
    assert st.treasury("ust_int") == pytest.approx(D ** 2 * V)
    assert D == pytest.approx(core.par_duration(st.y["int"].iloc[-1], core.MAT_INT))


def test_spread_level_uses_gz_when_moodys_stops(plain):
    p = _with_predictors(plain, moodys_end="2008-12-31")
    view = p.asof("2012-12-31")
    pm = core.monthly_predictors(view)
    s = core.spread_level(pm)
    assert np.isfinite(s.iloc[-1])
    # GZ was built as 0.8 x Baa-10y, so the rescaled tail must recover Baa-10y units
    full = core.spread_level(core.monthly_predictors(_with_predictors(plain).asof("2012-12-31")))
    assert s.iloc[-2] == pytest.approx(full.iloc[-2], rel=1e-6)
    v = _model("vol", "structural_normal").forecast_vol(view, view.eligible())
    assert np.isfinite(v).all()


def test_estimate_beta_recovers_elasticity():
    rng = np.random.default_rng(0)
    idx = pd.date_range("1950-01-31", periods=12 * 60, freq="ME")
    y = pd.Series(np.exp(np.log(0.05) + np.cumsum(rng.normal(0, 0.04, len(idx)))), index=idx).clip(0.01, 0.2)
    for beta in [0.0, 1.0]:
        u = pd.Series(0.002 * (y.shift(1) / 0.05) ** beta * rng.standard_normal(len(idx)), index=idx)
        b = core.estimate_beta({"a": u}, {"a": y}, pd.Timestamp("2010-06-30"))
        assert b == pytest.approx(beta, abs=0.3)
    # too little history: the prior
    assert core.estimate_beta({"a": u.iloc[:60]}, {"a": y}, pd.Timestamp("1955-06-30")) == 0.5


def test_tips_prior_before_tips_history_dominates(rich):
    """With few TIPS months, kappa is shrunk toward 1 (TIPS ~ nominal yield vol)."""
    view = rich.asof("2012-12-31")
    st = StructuralState(view, beta=0.0)
    assert np.isfinite(st.tips()) and st.tips() > 0
    m = StructuralVol(beta=0.0)
    assert np.isfinite(m.forecast_vol(view, ["tips", "ust_int"])).all()
