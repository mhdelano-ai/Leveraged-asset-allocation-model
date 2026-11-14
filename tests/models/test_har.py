import dataclasses

import numpy as np
import pandas as pd
import pytest

from assetforecast.data.synthetic import make_synthetic_panel
from assetforecast.models import get_model, list_models
from assetforecast.models.har import SPECS
from assetforecast.models.har.features import har_features
from assetforecast.models.har.model import HarVol, fit_one

NAMES = sorted(SPECS)
EARLY, MIDDLE, LATE = "1979-12-31", "1992-06-30", "2010-12-31"


@pytest.fixture(scope="module")
def panel():
    return make_synthetic_panel(seed=3)


def test_registered():
    assert set(NAMES) <= set(list_models("vol"))
    assert len(NAMES) <= 8


@pytest.mark.parametrize("name", NAMES)
def test_valid_at_every_stage(panel, name):
    m = get_model("vol", name)
    routes = {}
    for t in [EARLY, MIDDLE, LATE]:
        view = panel.asof(t)
        assets = view.eligible()
        v = m.forecast_vol(view, assets)
        assert list(v.index) == assets
        assert np.isfinite(v.to_numpy()).all() and (v > 0).all()
        assert ((v > 0.01) & (v < 1.0)).all()
        routes[t] = set(m.forecast_var(view, assets)[1])
    # early origin: monthly-only assets, no fit yet -> prior; late: fitted models
    assert routes[EARLY] == {"prior"}
    assert all(r.startswith("M") for r in routes[MIDDLE])
    late_key = "D" if SPECS[name][0] == "D" else "M"
    assert any(r.startswith(late_key) for r in routes[LATE])


@pytest.mark.parametrize("name", NAMES)
def test_cold_equals_warm_in_any_order(panel, name):
    ts = [EARLY, "1999-03-31", MIDDLE, LATE, "2010-01-31"]
    warm = get_model("vol", name)
    got = {t: warm.forecast_vol(panel.asof(t), panel.asof(t).eligible()) for t in ts}
    rev = get_model("vol", name)
    for t in reversed(ts):
        v = rev.forecast_vol(panel.asof(t), panel.asof(t).eligible())
        pd.testing.assert_series_equal(v, got[t])
    for t in ts:
        cold = get_model("vol", name).forecast_vol(panel.asof(t), panel.asof(t).eligible())
        pd.testing.assert_series_equal(cold, got[t])


def _altered(panel, t, seed=11):
    """Same panel up to t; returns after t replaced by noise of a different scale."""
    rng = np.random.default_rng(seed)
    t = pd.Timestamp(t)
    d, m = panel.daily.copy(), panel.monthly.copy()
    dm = d.index > t
    d.loc[dm] = np.where(d.loc[dm].notna(), rng.normal(0, 0.03, d.loc[dm].shape), np.nan)
    mm = m.index > t
    m.loc[mm] = np.where(m.loc[mm].notna(), rng.normal(0, 0.12, m.loc[mm].shape), np.nan)
    return dataclasses.replace(panel, daily=d, monthly=m, _shared={})


@pytest.mark.parametrize("name", ["har_pooled_log_d", "har_asset_qlike_d", "har_pooled_qlike_m"])
def test_no_lookahead(panel, name):
    t = "2004-07-31"
    alt = _altered(panel, t)
    a = get_model("vol", name).forecast_vol(panel.asof(t), panel.asof(t).eligible())
    b = get_model("vol", name).forecast_vol(alt.asof(t), alt.asof(t).eligible())
    pd.testing.assert_series_equal(a, b)
    # and the alteration does matter later
    t2 = "2008-12-31"
    a2 = get_model("vol", name).forecast_vol(panel.asof(t2), panel.asof(t2).eligible())
    b2 = get_model("vol", name).forecast_vol(alt.asof(t2), alt.asof(t2).eligible())
    assert not np.allclose(a2.to_numpy(), b2.to_numpy())


def test_training_targets_fully_observed(panel):
    e = pd.Timestamp("2005-12-31")
    tr = HarVol.training_rows(panel.asof(e))
    grid = tr["F"]["grid"]
    has_y = np.isfinite(tr["Y"]).any(axis=1)
    assert grid[has_y].max() <= e - pd.offsets.MonthEnd(12)
    assert has_y.sum() > 100


def test_estimation_date_is_last_december():
    assert HarVol.estimation_date("2010-12-31") == pd.Timestamp("2010-12-31")
    assert HarVol.estimation_date("2010-11-30") == pd.Timestamp("2009-12-31")
    assert HarVol.estimation_date("2011-01-31") == pd.Timestamp("2010-12-31")


def test_daily_regressors_on_target_scale(panel):
    # synthetic daily returns are serially independent: q ~ 1, and the
    # mapped 12-month daily variance tracks the trailing monthly variance
    F = har_features(panel.asof("2012-12-31"))
    ok = F["daily_ok"]
    assert ok[-1].all()
    q = F["q"][ok]
    assert 0.8 < np.median(q) < 1.2
    ratio = F["D12"][ok] / F["M12"][ok]
    assert 0.85 < np.median(ratio) < 1.15


def test_q_detects_smoothing_and_respects_tier_runs():
    p = make_synthetic_panel(seed=5, stagger=False)
    d = p.daily.copy()
    # smooth one asset (MA over 3 days, as a stale-priced NAV): 21-day / 1-day ratio > 1
    d["hy_credit"] = d["hy_credit"].rolling(3, min_periods=1).mean()
    tiers = p.tiers.copy()
    split = pd.Timestamp("2005-06-30")
    tiers.loc[tiers.index > split, "gold"] = "A"           # a splice on 2005-07
    p2 = dataclasses.replace(p, daily=d, tiers=tiers, _shared={})
    F = har_features(p2.asof("2012-12-31"))
    j = F["assets"].index("hy_credit")
    assert np.nanmedian(F["q"][-60:, j]) > 1.5
    g = pd.DatetimeIndex(F["grid"])
    k = F["assets"].index("gold")
    after = (g > split) & (g < split + pd.offsets.MonthEnd(12))
    assert not F["daily_ok"][after, k].any()               # no pre-splice scale carried over
    assert F["daily_ok"][g == split + pd.offsets.MonthEnd(12), k].all()


def test_qlike_fit_recovers_weights_and_scale():
    rng = np.random.default_rng(0)
    n = 4000
    R = np.exp(rng.normal(0, 0.5, (n, 2)))
    mean = 1.1 * (0.3 + 0.5 * R[:, 0] + 0.2 * R[:, 1])
    y = mean * rng.gamma(6.0, 1 / 6.0, n)                  # unbiased noisy target
    f = fit_one("qlike", ("a", "b"), R, y, np.arange(n))
    assert f.coef == pytest.approx([0.5, 0.2], abs=0.06)
    assert f.w_lr == pytest.approx(0.3, abs=0.06)
    assert np.exp(f.const) == pytest.approx(1.1, rel=0.05)


def test_log_fit_retransforms_to_the_mean():
    rng = np.random.default_rng(1)
    n = 6000
    R = np.exp(rng.normal(0, 0.5, (n, 2)))
    mean = R[:, 0] ** 0.4 * R[:, 1] ** 0.2
    y = mean * rng.gamma(3.0, 1 / 3.0, n)                  # E[y | x] = mean, E[log y | x] < log mean
    for form in ("log", "logcvx"):
        f = fit_one(form, ("a", "b"), R, y, np.arange(n))
        pred = f.predict(R)
        assert np.mean(y) / np.mean(pred) == pytest.approx(1.0, abs=0.03)
        assert f.coef == pytest.approx([0.4, 0.2], abs=0.05)
        assert f.phi > 1.1                                 # smearing is doing real work


def test_convex_constraint_binds():
    rng = np.random.default_rng(2)
    n = 3000
    R = np.exp(rng.normal(0, 0.5, (n, 3)))
    y = R[:, 0] ** 1.3 / R[:, 1] ** 0.3 * rng.gamma(5.0, 0.2, n)
    f = fit_one("logcvx", ("a", "b", "c"), R, y, np.arange(n))
    assert (f.coef >= 0).all() and f.coef.sum() <= 1 + 1e-8
    g = fit_one("log", ("a", "b", "c"), R, y, np.arange(n))
    assert g.coef[1] < 0 and g.coef[0] > 1


def test_too_few_origins_gives_no_fit():
    R = np.ones((50, 2))
    assert fit_one("log", ("a", "b"), R, np.ones(50), np.arange(50)) is None
