from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from volmodel.data.synthetic import make_synthetic_panel
from volmodel.models import get_model, list_models
from volmodel.models.garch import core, prep

NAMES = ["garch_d_garch11", "garch_d_gjr", "garch_d_cgarch", "garch_d_midas", "garch_d_rm2006",
         "garch_w_cgarch", "garch_m_garch11", "garch_m_cgarch"]
EARLY, MIDDLE, LATE = "1979-12-31", "1990-06-30", "2010-12-31"


@pytest.fixture(scope="module")
def panel():
    return make_synthetic_panel(seed=3)


def _fc(model, panel, t):
    view = panel.asof(t)
    return model.forecast_vol(view, view.eligible())


def test_registered():
    got = [m for m in list_models("vol") if m.startswith("garch")]
    assert sorted(got) == sorted(NAMES) and len(got) <= 8


# ------------------------------------------------------------------ numerics
def _sim_garch(n=4000, seed=1, w=0.02, a=0.08, b=0.9):
    rng = np.random.default_rng(seed)
    s2, e = w / (1 - a - b), np.empty(n)
    for i in range(n):
        e[i] = np.sqrt(s2) * rng.standard_t(6) / np.sqrt(1.5)
        s2 = w + a * e[i] ** 2 + b * s2
    return e


def test_garch11_estimate_matches_arch():
    arch = pytest.importorskip("arch")
    e = _sim_garch()
    sp = core.Garch11(252)
    vb = float(np.mean(e * e))
    f = core.estimate(sp, lambda th: sp.nll(th, e, vb) / len(e))
    p = sp.unpack(f.theta)
    r = arch.arch_model(e, mean="Zero", vol="GARCH", dist="t").fit(disp="off").params
    assert f.ok
    assert p["alpha"] == pytest.approx(r["alpha[1]"], abs=0.01)
    assert p["beta"] == pytest.approx(r["beta[1]"], abs=0.01)
    assert p["nu"] == pytest.approx(r["nu"], rel=0.15)


def test_cgarch_filter_matches_loop_and_forecast_matches_iteration():
    e = _sim_garch(2000)
    e2, vb = e * e, float(np.mean(e * e))
    rho, phi, al, be = 0.995, 0.02, 0.05, 0.85
    s2, q = core.cgarch_filter(e2, vb, rho, phi, al, be)
    Q = S = vb
    ref = [S]
    for x in e2:
        Qn = (1 - rho) * vb + rho * Q + phi * (x - S)
        S = Qn + al * (x - Q) + be * (S - Q)
        Q = Qn
        ref.append(S)
    assert np.allclose(s2, ref, rtol=1e-10, atol=1e-12)
    sp = core.CGarch(252)
    prm = {"rho": rho, "phi": phi, "alpha": al, "beta": be, "ab": al + be, "nu": 8.0}
    _, st = sp.filter(e, vb, prm)
    s, qq, tot = st[0], st[1], 0.0
    for _ in range(252):
        tot += s
        qn = (1 - rho) * vb + rho * qq
        s = qn + (al + be) * (s - qq)
        qq = qn
    assert sp.agg_forecast(st, vb, prm, 252) == pytest.approx(tot, rel=1e-10)


def test_estimate_never_raises_on_degenerate_data():
    sp = core.CGarch(252)
    e = np.zeros(800)
    e[::50] = 1e-3
    f = core.estimate(sp, lambda th: sp.nll(th, e, float(np.mean(e * e))) / len(e))
    assert np.all(np.isfinite(f.theta))


def test_rm2006_weights_and_constant_forecast():
    rm = core.RM2006()
    assert len(rm.w) == 15 and rm.w.sum() == pytest.approx(1.0) and np.all(np.diff(rm.w) < 0)
    x1 = np.full(15, 2.0)                 # all components equal: flat forecast
    assert rm.agg_forecast(x1, 252) == pytest.approx(2.0 * 252)


def test_variance_ratio_detects_smoothing_and_resets_at_tier_change():
    rng = np.random.default_rng(0)
    n = 6000
    z = rng.standard_normal(n) * 0.01
    r = z.copy()
    r[:3000] = 0.5 * z[:3000] + 0.5 * np.r_[0.0, z[:2999]]   # smoothed NAV: positive autocorrelation
    seg = np.r_[np.zeros(3000, int), np.ones(3000, int)]
    lv = prep.vr_log(r, seg, 1)
    assert lv[2999] > 0.25                 # true log VR = log 2 ~ 0.69, shrunk toward 0
    assert lv[3000] == 0.0                 # restarts at the splice
    assert abs(lv[-1]) < 0.15              # iid after the splice


def test_blocks_anchor_at_end():
    idx = pd.bdate_range("2000-01-03", periods=23)
    s = pd.Series(np.arange(23, dtype=float) * 1e-3, index=idx)
    tiers = pd.Series("B", index=pd.date_range("1999-12-31", periods=3, freq="ME"))
    b = prep.block_adjusted(s, tiers, idx[-1], 5)
    assert b.index[-1] == idx[-1] and len(b) == 4
    assert b.iloc[-1] == pytest.approx(s.iloc[-5:].sum())


def test_est_date():
    assert prep.est_date("2010-12-31") == pd.Timestamp("2010-12-31")
    assert prep.est_date("2010-11-30") == pd.Timestamp("2009-12-31")


# -------------------------------------------------------- forecaster contract
@pytest.mark.parametrize("name", NAMES)
def test_valid_at_every_stage(panel, name):
    m = get_model("vol", name)
    for t in [EARLY, MIDDLE, LATE]:
        view = panel.asof(t)
        assets = view.eligible()
        v = m.forecast_vol(view, assets)
        assert list(v.index) == assets and np.isfinite(v).all() and (v > 0).all()
        assert (v < 1.5).all() and (v > 0.01).all()


@pytest.mark.parametrize("name", NAMES)
def test_cold_equals_warm_in_any_order(panel, name):
    ts = [MIDDLE, "2009-06-30", LATE]
    warm = get_model("vol", name)
    fw = {t: _fc(warm, panel, t) for t in ts}
    back = get_model("vol", name)
    fb = {t: _fc(back, panel, t) for t in reversed(ts)}
    for t in ts:
        cold = _fc(get_model("vol", name), panel, t)
        pd.testing.assert_series_equal(cold, fw[t])
        pd.testing.assert_series_equal(cold, fb[t])


@pytest.mark.parametrize("name", NAMES)
def test_no_lookahead(panel, name):
    t = pd.Timestamp(LATE)
    f0 = _fc(get_model("vol", name), panel, t)
    d, m = panel.daily.copy(), panel.monthly.copy()
    d.loc[d.index > t] *= 3.0
    m.loc[m.index > t] *= 3.0
    altered = replace(panel, daily=d, monthly=m)
    model = get_model("vol", name)
    _fc(model, altered, "2014-12-31")          # warm the cache on the altered future first
    f1 = _fc(model, altered, t)
    pd.testing.assert_series_equal(f0, f1)


def test_daily_model_uses_daily_data_when_available(panel):
    """Daily-fitted and monthly fallback differ once daily data exist."""
    d = _fc(get_model("vol", "garch_d_cgarch"), panel, LATE)
    m = _fc(get_model("vol", "garch_m_garch11"), panel, LATE)
    assert not np.allclose(d.values, m.values)
    e = _fc(get_model("vol", "garch_d_cgarch"), panel, EARLY)      # monthly-only era: fallback
    me = _fc(get_model("vol", "garch_m_garch11"), panel, EARLY)
    pd.testing.assert_series_equal(e, me)
