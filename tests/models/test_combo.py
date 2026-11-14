"""Tests for the combo family on the synthetic panel.

Cheap synthetic components are run through the harness into a temporary
results directory (never the real one); the combos read them from there.
"""
import numpy as np
import pandas as pd
import pytest
from dataclasses import replace

from volmodel.data.synthetic import make_synthetic_panel
from volmodel.eval.harness import run_model
from volmodel.eval.targets import compute_targets, window_end
from volmodel.models import list_models
from volmodel.models.base import Composite, CorrForecaster, VolForecaster
from volmodel.models.combo import (CORR_MODELS, COV_PAIRS, VOL_MODELS, CorrCombo, MissingComponentError,
                                   VolCombo, block_average, estimation_date, training_data)
from volmodel.models.combo import store as combo_store
from volmodel.models.ref import nearest_corr, pairwise_corr, window_var

ORIGINS = ["1979-12-31", "1996-06-30", "2012-12-31"]   # early one: monthly-only assets
T0 = pd.Timestamp("2004-12-31")                        # the "oracle" turns perfect here


# ------------------------------------------------------------- components
class WindowVol(VolForecaster):
    def __init__(self, months):
        self.months = months

    def forecast_vol(self, view, assets):
        return np.sqrt(window_var(view.excess_monthly[assets], self.months)).reindex(assets)


class OracleVol(VolForecaster):
    """Useless (1/25 of the long-run variance) before T0, the realized target itself from T0.
    A test component: it looks ahead on purpose. (Useless by being far too LOW: a member that
    is far too high can earn a little stacking weight as a level shifter when the others
    under-forecast.)"""

    def __init__(self, targets, scale_after=1.0, scale_before=0.04, t0=T0):
        self.v = targets.vol_wide("var")
        self.scale_after, self.scale_before, self.t0 = scale_after, scale_before, t0

    def forecast_vol(self, view, assets):
        t = view.asof_date
        lr = window_var(view.excess_monthly[assets], None)
        if t >= self.t0 and t in self.v.index:
            r = self.v.loc[t].reindex(assets)
            v = (self.scale_after * r).where(r.notna(), lr)
        else:
            v = self.scale_before * lr
        return np.sqrt(v)


class ScaledTruthVol(VolForecaster):
    """Realized variance times a known constant (look-ahead on purpose)."""

    def __init__(self, targets, k):
        self.v, self.k = targets.vol_wide("var"), k

    def forecast_vol(self, view, assets):
        t = view.asof_date
        lr = window_var(view.excess_monthly[assets], None)
        r = self.v.loc[t].reindex(assets) if t in self.v.index else pd.Series(np.nan, index=assets)
        return np.sqrt((self.k * r).where(r.notna(), self.k * lr))


class WindowCorr(CorrForecaster):
    def __init__(self, months):
        self.months = months

    def forecast_corr(self, view, assets):
        x = view.excess_monthly[assets]
        x = x.iloc[-self.months:] if self.months else x
        return pd.DataFrame(nearest_corr(pairwise_corr(x, min_overlap=12)), index=assets, columns=assets)


@pytest.fixture(scope="module")
def panel():
    return make_synthetic_panel(seed=5)


@pytest.fixture(scope="module")
def store(panel, tmp_path_factory):
    """Temporary results dir holding the synthetic components' stored forecasts."""
    root = tmp_path_factory.mktemp("combo_results")
    T = compute_targets(panel)
    comps = {
        ("vol", "aaa_last12"): WindowVol(12),
        ("vol", "aaa_last60"): WindowVol(60),
        ("vol", "bbb_longrun"): WindowVol(None),
        ("vol", "ccc_oracle"): OracleVol(T),
        ("vol", "ddd_half"): ScaledTruthVol(T, 0.5),
        ("corr", "aaa_corr60"): WindowCorr(60),
        ("corr", "bbb_corrall"): WindowCorr(None),
    }
    for (kind, name), m in comps.items():
        m.name = name
        run_model(m, kind, panel, results_dir=root)
    run_model(_named(WindowVol(12), "eee_partial"), "vol", panel, results_dir=root, end="2005-12-31")
    run_model(_named(WindowVol(12), "fff_late"), "vol", panel, results_dir=root, start="1999-01-31")
    return root


def _named(m, name):
    m.name = name
    return m


VOLS = ["aaa_last12", "aaa_last60", "bbb_longrun"]


def _combos(root):
    return {
        "ew": VolCombo(VOLS, group="none", results_dir=root),
        "hier": VolCombo(VOLS, group="family", results_dir=root),
        "trim": VolCombo(VOLS, group="none", agg="trim", results_dir=root),
        "stack_cal": VolCombo(VOLS, group="family", agg="stack", calibrate=True, results_dir=root),
        "invq": VolCombo(VOLS, group="none", agg="invq", results_dir=root),
        "best": VolCombo(VOLS, group="none", agg="best", results_dir=root),
    }


# ------------------------------------------------------------------ tests
def test_registered_within_budget():
    for kind, models, cap in (("vol", VOL_MODELS, 8), ("corr", CORR_MODELS, 6), ("cov", COV_PAIRS, 4)):
        reg = [n for n in list_models(kind) if n.startswith("combo_")]
        assert set(reg) == set(models) and len(reg) <= cap


def test_default_members_exclude_combos():
    v = VolCombo(group="family")
    assert v.member_list() and not any(m.startswith("combo_") for m in v.member_list())
    c = CorrCombo()
    assert "ref_corr_zero" not in c.member_list()
    assert not any(m.split(":")[-1].startswith("combo_") for m in c.member_list())


def test_estimation_date():
    assert estimation_date("2001-12-31") == pd.Timestamp("2001-12-31")
    assert estimation_date("2001-11-30") == pd.Timestamp("2000-12-31")
    assert estimation_date("2002-01-31") == pd.Timestamp("2001-12-31")


@pytest.mark.parametrize("key", ["ew", "hier", "trim", "stack_cal", "invq", "best"])
def test_valid_output_through_harness(panel, store, tmp_path, key):
    m = _named(_combos(store)[key], f"combo_test_{key}")
    res = run_model(m, "vol", panel, results_dir=tmp_path, step=6)
    assert res.error is None and res.n_new > 50
    early = panel.asof(ORIGINS[0])
    assert not early.has_daily()
    for t in ORIGINS:
        view = panel.asof(t)
        a = view.eligible()
        f = m.forecast_vol(view, a)
        assert list(f.index) == a and np.isfinite(f).all() and (f > 0).all()
        sub = a[::-1][:3]
        assert np.allclose(m.forecast_vol(view, sub).values, f.reindex(sub).values)


def test_equal_weight_is_mean_of_variances(panel, store):
    t = ORIGINS[1]
    view = panel.asof(t)
    a = view.eligible()
    ew = _combos(store)["ew"].forecast_vol(view, a)
    parts = [combo_store.load("vol", m, store).at(pd.Timestamp(t), a) for m in VOLS]
    assert np.allclose(ew.values ** 2, np.mean(parts, axis=0))
    hier = _combos(store)["hier"].forecast_vol(view, a)          # families aaa (2 members), bbb (1)
    fam = (0.5 * (parts[0] + parts[1]) + parts[2]) / 2
    assert np.allclose(hier.values ** 2, fam)


def test_training_cells_are_walk_forward_and_match_project_target(panel):
    E = pd.Timestamp("2003-12-31")
    tr = training_data(panel.asof("2004-05-31"), E)
    assert len(tr.origins) and max(window_end(o) for o in tr.origins) <= E
    T = compute_targets(panel).vol_wide("var")
    from volmodel.universe import ASSET_IDS
    ref = T.reindex(index=tr.origins, columns=ASSET_IDS).to_numpy()
    assert tr.cells.sum() > 1000 and np.isfinite(ref[tr.cells]).all()
    assert np.array_equal(tr.realized[tr.cells], ref[tr.cells])        # the project's own estimator
    # the training set never depends on data after E
    rng = np.random.default_rng(0)
    d, m = panel.daily.copy(), panel.monthly.copy()
    d.loc[d.index > E] *= 3
    m.loc[m.index > E] = rng.normal(0, 0.2, m.loc[m.index > E].shape)
    p2 = replace(panel, daily=d, monthly=m, _shared={})
    tr2 = training_data(p2.asof("2004-05-31"), E)
    assert np.array_equal(np.nan_to_num(tr.realized), np.nan_to_num(tr2.realized))


def test_future_oracle_gets_no_credit_early(panel, store):
    members = VOLS + ["ccc_oracle"]
    m = VolCombo(members, group="none", agg="stack", results_dir=store)
    view = panel.asof("2014-12-31")
    g = len(members)
    k = members.index("ccc_oracle")
    # every training window that ends by E started before T0: the oracle was useless there
    for E in ["1995-12-31", "2004-12-31"]:
        est = m.estimate(view, E)
        assert est.lam > 0
        assert est.weights[k] == pytest.approx((1 - est.lam) / g, abs=1e-6)   # w* = 0: no credit
    # its perfect forecasts at origins >= T0 count only once their windows have ended
    est_late = m.estimate(view, "2014-12-31")
    assert est_late.weights[k] > (1 - est_late.lam) / g + 0.03
    assert est_late.weights[k] - (1 - est_late.lam) / g > \
        m.estimate(view, "2008-12-31").weights[k] - (1 - m.estimate(view, "2008-12-31").lam) / g
    # the same holds for performance (inverse-QLIKE) weights
    q = VolCombo(members, group="none", agg="invq", results_dir=store).estimate(view, "2004-12-31")
    assert q.past_qlike[k] == max(q.past_qlike)


def test_calibration_recovers_known_scale(panel, store):
    m = VolCombo(["ddd_half"], group="none", calibrate=True, results_dir=store)
    view = panel.asof("2013-12-31")
    est = m.estimate(view, "2013-12-31")
    assert est.scale_raw == pytest.approx(2.0, rel=1e-9)                # realized / forecast = 2 exactly
    assert est.scale == pytest.approx(1 + est.lam * 1.0, rel=1e-12)
    f = m.forecast_vol(view, view.eligible())
    raw = combo_store.load("vol", "ddd_half", store).at(pd.Timestamp("2013-12-31"), view.eligible())
    assert np.allclose(f.values ** 2, raw * est.scale)
    # a learned scale shrinks to 1 with no training data
    est0 = m.estimate(panel.asof("1976-12-31"), "1976-12-31")
    assert est0.scale == 1.0


def test_missing_component_raises(panel, store):
    view = panel.asof(ORIGINS[1])
    a = view.eligible()
    with pytest.raises(MissingComponentError, match="no_such_model"):
        VolCombo(["aaa_last12", "no_such_model"], group="none", results_dir=store).forecast_vol(view, a)
    # stored, but not at this origin
    late = panel.asof("2008-06-30")
    with pytest.raises(MissingComponentError, match=r"eee_partial.*2008-06-30"):
        VolCombo(["aaa_last12", "eee_partial"], group="none", results_dir=store).forecast_vol(late, late.eligible())
    # stored at the forecast origin but not at an origin the weights must learn from
    v = panel.asof("2010-06-30")
    with pytest.raises(MissingComponentError, match=r"fff_late.*training origin"):
        VolCombo(["aaa_last12", "fff_late"], group="none", agg="stack",
                 results_dir=store).forecast_vol(v, v.eligible())
    with pytest.raises(MissingComponentError, match="no_such_corr"):
        CorrCombo(["aaa_corr60", "no_such_corr"], group="none", results_dir=store).forecast_corr(view, a)


def test_cold_equals_warm_and_order_free(panel, store):
    t = ORIGINS[1]
    view = panel.asof(t)
    a = view.eligible()
    for key in ["stack_cal", "invq", "trim"]:
        cold = _combos(store)[key].forecast_vol(view, a)
        warm = _combos(store)[key]
        for s in ["2012-12-31", "1979-12-31", "2003-03-31"]:
            v = panel.asof(s)
            warm.forecast_vol(v, v.eligible())
        assert np.array_equal(cold.values, warm.forecast_vol(view, a).values)
    c_cold = CorrCombo(["aaa_corr60", "bbb_corrall"], group="none", block_shrink=0.2,
                       results_dir=store).forecast_corr(view, a)
    cw = CorrCombo(["aaa_corr60", "bbb_corrall"], group="none", block_shrink=0.2, results_dir=store)
    for s in ["2012-12-31", "1979-12-31"]:
        v = panel.asof(s)
        cw.forecast_corr(v, v.eligible())
    assert np.array_equal(c_cold.values, cw.forecast_corr(view, a).values)


def test_no_lookahead(panel, store):
    t = pd.Timestamp("2004-06-30")
    d, m = panel.daily.copy(), panel.monthly.copy()
    d.loc[d.index > t] *= 5
    m.loc[m.index > t] *= -3
    p2 = replace(panel, daily=d, monthly=m, _shared={})
    for key in ["stack_cal", "invq"]:
        a = panel.asof(t).eligible()
        f1 = _combos(store)[key].forecast_vol(panel.asof(t), a)
        f2 = _combos(store)[key].forecast_vol(p2.asof(t), a)
        assert np.array_equal(f1.values, f2.values)


def test_corr_combination_positive_definite(panel, store, tmp_path):
    for shrink in (0.0, 0.2, 1.0):
        m = _named(CorrCombo(["aaa_corr60", "bbb_corrall"], group="family", block_shrink=shrink,
                             results_dir=store), f"combo_test_corr_{int(shrink * 10)}")
        res = run_model(m, "corr", panel, results_dir=tmp_path, step=12)
        assert res.error is None and res.n_new > 10
        for t in ORIGINS:
            view = panel.asof(t)
            a = view.eligible()
            C = m.forecast_corr(view, a).to_numpy()
            assert np.allclose(C, C.T) and np.allclose(np.diag(C), 1) and np.linalg.eigvalsh(C).min() > 0
    w = CorrCombo(["aaa_corr60", "bbb_corrall"], group="none", weights={"aaa_corr60": 3, "bbb_corrall": 1},
                  results_dir=store)
    view = panel.asof(ORIGINS[2])
    a = view.eligible()
    parts = [combo_store.load("corr", x, store).at(pd.Timestamp(ORIGINS[2]), a) for x in ["aaa_corr60", "bbb_corrall"]]
    assert np.allclose(w.forecast_corr(view, a).to_numpy(), 0.75 * parts[0] + 0.25 * parts[1])


def test_block_average_is_pd_and_block_constant():
    rng = np.random.default_rng(2)
    groups = ["equity"] * 4 + ["rates"] * 3 + ["credit"] * 2 + ["real"] * 3
    for _ in range(20):
        A = rng.normal(size=(12, 30))
        C = np.corrcoef(A)
        B = block_average(C, groups)
        assert np.allclose(np.diag(B), 1) and np.allclose(B, B.T) and np.linalg.eigvalsh(B).min() > 0
        assert np.isclose(B[0, 1], (C[:4, :4].sum() - 4) / 12) and np.isclose(B[0, 5], C[:4, 4:7].mean())


def test_composite_cov(panel, store):
    v = VolCombo(VOLS, group="family", results_dir=store)
    c = CorrCombo(["aaa_corr60", "bbb_corrall"], group="family", block_shrink=0.2, results_dir=store)
    view = panel.asof(ORIGINS[2])
    a = view.eligible()
    H = Composite(v, c).forecast_cov(view, a).to_numpy()
    assert np.linalg.eigvalsh(H).min() > 0 and np.allclose(np.sqrt(np.diag(H)), v.forecast_vol(view, a).values)


@pytest.mark.realdata
def test_registered_on_dev_results():
    """Registered combos at three origins against the stored dev forecasts (skipped if absent)."""
    import os
    from volmodel.models import get_model
    from volmodel.paths import ROOT
    rd = ROOT / "results" / "dev" / "forecasts" / "vol" / "statespace_2c_daily.parquet"
    if not rd.exists() or not os.environ.get("VOLMODEL_MAX_DATE"):
        pytest.skip("needs results/dev and VOLMODEL_MAX_DATE")
    from volmodel.data.panel import load_panel
    p = load_panel()
    for kind, names in (("vol", VOL_MODELS), ("corr", CORR_MODELS), ("cov", COV_PAIRS)):
        for n in names:
            m = get_model(kind, n)
            for t in ["1930-06-30", "1990-12-31", "2014-12-31"]:
                v = p.asof(t)
                a = v.eligible()
                out = getattr(m, f"forecast_{kind}")(v, a)
                assert np.isfinite(np.asarray(out)).all()
