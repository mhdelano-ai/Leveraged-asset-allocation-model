from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from volmodel.data.panel import PREDICTOR_COLUMNS
from volmodel.data.synthetic import make_synthetic_panel
from volmodel.eval.targets import realized_kday_cov
from volmodel.models import get_model, list_models
from volmodel.models.macro.features import ReturnFeatures, asof_monthly, credit_spread, _series_tables
from volmodel.models.macro.model import CACHE_KEY, GROUPS, gamma_glm

MACRO_MODELS = ["macro_base", "macro_state", "macro_state_ip", "macro_implied", "macro_full",
                "macro_base_h", "macro_full_h"]
ORIGINS = ["1979-12-31", "1990-06-30", "2010-12-31"]   # 1979: monthly-only assets


def _fresh(panel, predictors=None, **kw):
    """A copy of ``panel`` with its own (empty) cache."""
    p = panel.predictors if predictors is None else predictors
    return replace(panel, predictors=p.sort_values("available_date", kind="stable").reset_index(drop=True),
                   _shared={}, **kw)


def _with_real_names(panel, seed=0):
    """Synthetic panel whose predictors carry the real series names the family uses."""
    rng = np.random.default_rng(seed)
    m_idx = panel.monthly.index
    iv = panel.predictors[panel.predictors.series == "synthetic_implied_vol"].set_index("obs_date")["value"]
    iv = iv.reindex(m_idx).ffill().bfill()
    rows = []

    def monthly(name, values, lag_months=1):
        rows.append(pd.DataFrame({"obs_date": m_idx, "available_date": m_idx + pd.offsets.MonthEnd(lag_months),
                                  "series": name, "value": values}))

    walk = lambda s: rng.standard_normal(len(m_idx)).cumsum() * s       # noqa: E731
    monthly("credit_moodys_baa_aaa_spread_m", 0.01 * (iv.values / 0.16) ** 1.5, 0)
    monthly("credit_gz_spread_m", 0.015 * (iv.values / 0.16) ** 1.2)
    monthly("macro_term_spread", 0.01 + 0.002 * walk(1))
    monthly("macro_infl_vol_36m", np.exp(np.log(0.02) + 0.05 * walk(1)))
    monthly("macro_infl_yoy", 0.03 + 0.002 * walk(1))
    monthly("macro_long_yield", np.exp(np.log(0.05) + 0.03 * walk(1)), 0)
    monthly("macro_sahm_gap", 0.003 * rng.standard_normal(len(m_idx)))
    monthly("macro_policy_rate_chg_12m", 0.005 * rng.standard_normal(len(m_idx)), 0)
    monthly("macro_real_short_rate", 0.01 + 0.002 * walk(1))
    monthly("macro_usd_broad_yoy", 0.05 * rng.standard_normal(len(m_idx)), 0)
    monthly("macro_ip_yoy", 0.03 * rng.standard_normal(len(m_idx)))
    days = panel.daily.index
    iv_d = iv.reindex(days, method="bfill").ffill().values

    def daily(name, values, start=None):
        d = pd.DataFrame({"obs_date": days, "available_date": days, "series": name, "value": values})
        rows.append(d[d.obs_date >= pd.Timestamp(start)] if start else d)

    daily("iv_vix_vxo_spliced", iv_d * 1.1)
    daily("iv_move", 0.008 * iv_d / 0.16, "1992-01-01")
    daily("ust_ladder_duration_int", np.full(len(days), 6.5))
    daily("ust_ladder_duration_long", np.full(len(days), 15.0))
    daily("iv_gvz", iv_d, "1995-01-01")
    daily("iv_ovx", 2 * iv_d, "1995-01-01")
    pred = pd.concat(rows, ignore_index=True)[PREDICTOR_COLUMNS]
    return _fresh(panel, pred)


@pytest.fixture(scope="module")
def panel():
    return make_synthetic_panel(seed=3)


@pytest.fixture(scope="module")
def rpanel(panel):
    return _with_real_names(panel)


def test_registered():
    assert set(MACRO_MODELS) <= set(list_models("vol"))
    assert len([m for m in list_models("vol") if m.startswith("macro_")]) <= 8


@pytest.mark.parametrize("name", MACRO_MODELS)
@pytest.mark.parametrize("which", ["synthetic", "real_names"])
def test_valid_at_every_stage(panel, rpanel, name, which):
    p = panel if which == "synthetic" else rpanel
    m = get_model("vol", name)
    for t in ORIGINS:
        view = p.asof(t)
        assets = view.eligible()
        v = m.forecast_vol(view, assets)
        assert list(v.index) == assets and np.isfinite(v).all() and (v > 0).all()
        assert (v < 2.0).all()
    # the early origin has assets with monthly data only
    early = p.asof(ORIGINS[0])
    assert set(early.eligible()) - set(early.has_daily())


@pytest.mark.parametrize("name", ["macro_base", "macro_full", "macro_full_h"])
def test_cold_equals_warm_any_order(rpanel, name):
    cold = {}
    for t in ORIGINS:
        p = _fresh(rpanel)
        view = p.asof(t)
        cold[t] = get_model("vol", name).forecast_vol(view, view.eligible())
    p = _fresh(rpanel)
    m = get_model("vol", name)
    for t in ORIGINS[::-1] + ORIGINS:
        view = p.asof(t)
        pd.testing.assert_series_equal(m.forecast_vol(view, view.eligible()), cold[t])
    # a model that has run other origins (e.g. in the shared cache) still agrees
    m.forecast_vol(p.asof("2000-12-31"), p.asof("2000-12-31").eligible())
    view = p.asof(ORIGINS[1])
    pd.testing.assert_series_equal(m.forecast_vol(view, view.eligible()), cold[ORIGINS[1]])


@pytest.mark.parametrize("name", ["macro_base", "macro_full"])
def test_no_lookahead(rpanel, name):
    t = pd.Timestamp("2005-06-30")
    view = rpanel.asof(t)
    ref = get_model("vol", name).forecast_vol(view, view.eligible())
    # alter returns after t and every predictor value published after t
    daily = rpanel.daily.copy()
    daily.loc[daily.index > t] *= 5.0
    monthly = rpanel.monthly.copy()
    monthly.loc[monthly.index > t] = (1 + monthly.loc[monthly.index > t]) ** 2 - 1
    pred = rpanel.predictors.copy()
    late = pred.available_date > t
    pred.loc[late, "value"] = pred.loc[late, "value"] * 3.0 + 1.0
    alt = _fresh(rpanel, pred, daily=daily, monthly=monthly)
    view2 = alt.asof(t)
    out = get_model("vol", name).forecast_vol(view2, view2.eligible())
    pd.testing.assert_series_equal(out, ref)


def test_publication_lag_respected(rpanel):
    """A monthly value observed before t but published after t is not used."""
    t = pd.Timestamp("2005-06-30")
    p = rpanel.predictors
    tab = _series_tables(rpanel.asof(t)).get("macro_term_spread")
    got = asof_monthly(tab, pd.DatetimeIndex([t]))[0]
    pub = p[(p.series == "macro_term_spread") & (p.available_date <= t)].sort_values("obs_date")
    assert got == pub["value"].iloc[-1]
    assert pub["obs_date"].iloc[-1] < t          # the June value (published end-July) is not used


def test_state_variables_enter_and_fall_back_when_missing(rpanel):
    t = "2010-12-31"
    m = get_model("vol", "macro_full")
    view = rpanel.asof(t)
    m.forecast_vol(view, view.eligible())
    assert "cs" in m.last_fits["us_large"]["feats"]
    assert "vix" in m.last_fits["us_large"]["feats"]
    assert "move_d" in m.last_fits["ust_int"]["feats"]
    assert "iv_own" in m.last_fits["gold"]["feats"]
    # VIX stops 2 months and the term spread 8 months before the origin (stale: > 31 days
    # for a daily series, > ~6 months for a monthly one); a 3-month-old print still counts
    pred = rpanel.predictors
    t0 = pd.Timestamp(t)
    drop = ((pred.series.eq("iv_vix_vxo_spliced") & (pred.obs_date > t0 - pd.DateOffset(months=2)))
            | (pred.series.eq("macro_term_spread") & (pred.obs_date > t0 - pd.DateOffset(months=8)))
            | (pred.series.eq("macro_sahm_gap") & (pred.obs_date > t0 - pd.DateOffset(months=3))))
    alt = _fresh(rpanel, pred[~drop])
    view2 = alt.asof(t)
    v = m.forecast_vol(view2, view2.eligible())
    assert np.isfinite(v).all()
    feats = m.last_fits["us_large"]["feats"]
    assert "vix" not in feats and "ts" not in feats and "cs" in feats and "sahm" in feats
    # with no predictors at all the state models reduce exactly to the base model
    none = _fresh(rpanel, rpanel.predictors.iloc[:0])
    view3 = none.asof(t)
    pd.testing.assert_series_equal(m.forecast_vol(view3, view3.eligible()),
                                   get_model("vol", "macro_base").forecast_vol(view3, view3.eligible()))


def test_short_history_predictor_is_dropped(rpanel):
    """A predictor with < 5 years of training origins does not enter (nested fallback)."""
    pred = rpanel.predictors
    t = pd.Timestamp("2010-12-31")
    late = pred.series.eq("iv_vix_vxo_spliced") & (pred.obs_date < pd.Timestamp("2007-01-01"))
    alt = _fresh(rpanel, pred[~late])
    m = get_model("vol", "macro_implied")
    view = alt.asof(t)
    v = m.forecast_vol(view, view.eligible())
    assert np.isfinite(v).all()
    assert m.last_fits["us_large"]["feats"] == ()


def test_credit_spread_falls_back_to_gz_at_the_origin_only(rpanel):
    t = pd.Timestamp("2010-12-31")
    pred = rpanel.predictors
    dead = (pred.series == "credit_moodys_baa_aaa_spread_m") & (pred.obs_date > pd.Timestamp("2009-12-31"))
    alt = _fresh(rpanel, pred[~dead])
    view = alt.asof(t)
    grid = pd.DatetimeIndex(view.monthly.index)
    cs = credit_spread(_series_tables(view), grid)
    assert np.isfinite(cs[-1])                                   # mapped GZ at the origin
    assert np.isnan(cs[-2])                                      # never imputed in history
    true = np.log(rpanel.predictors.query("series == 'credit_moodys_baa_aaa_spread_m'")
                  .set_index("obs_date")["value"].loc["2010-11-30"])   # GZ is published a month late
    assert abs(cs[-1] - true) < 0.02                             # exact log-log relation in this panel


def test_daily_feature_matches_target_estimator(panel):
    view = panel.asof("2008-12-31")
    rf = ReturnFeatures(view)
    i = rf.grid.get_loc(pd.Timestamp("2006-12-31"))
    j = rf.assets.index("us_large")
    X = view.excess_daily.loc["2007-01-01":"2007-12-31"].fillna(0.0)
    sigma, _ = realized_kday_cov(X.to_numpy(), 21)
    assert rf.y[i, j] == pytest.approx(sigma[j, j], rel=1e-9)
    assert rf.v12[rf.grid.get_loc(pd.Timestamp("2007-12-31")), j] == pytest.approx(sigma[j, j], rel=1e-9)


def test_gamma_glm_recovers_coefficients():
    rng = np.random.default_rng(0)
    n = 4000
    X = np.column_stack([np.ones(n), rng.standard_normal((n, 2))])
    b = np.array([-3.0, 0.5, -0.2])
    y = np.exp(X @ b) * rng.gamma(4.0, 0.25, n)
    est = gamma_glm(X, y, np.ones(n), np.zeros(3))
    assert np.allclose(est, b, atol=0.05)
    shrunk = gamma_glm(X, y, np.ones(n) / 100, np.array([0.0, 1e4, 1e4]))
    assert np.all(np.abs(shrunk[1:]) < 0.02)


def test_groups_cover_universe():
    from volmodel.universe import ASSET_IDS
    assert sorted(a for g in GROUPS.values() for a in g) == sorted(ASSET_IDS)


def test_fit_cache_lives_in_panel(rpanel):
    p = _fresh(rpanel)
    view = p.asof("2010-12-31")
    get_model("vol", "macro_state").forecast_vol(view, view.eligible())
    assert CACHE_KEY in p._shared and len(p._shared[CACHE_KEY]) > 0


def test_homogeneous_base_slopes_sum_to_one(rpanel):
    m = get_model("vol", "macro_base_h")
    view = rpanel.asof("2010-12-31")
    m.forecast_vol(view, view.eligible())
    b = m.last_fits["us_large"]["beta"]
    assert b[1:4].sum() == pytest.approx(1.0)
