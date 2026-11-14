import numpy as np
import pandas as pd
import pytest

from assetforecast.data.panel import PREDICTOR_COLUMNS, Panel
from assetforecast.data.synthetic import make_synthetic_panel
from assetforecast.eval.targets import compute_targets
from assetforecast.models import get_model, list_models
from assetforecast.models.condcorr.data import predictors_at, realized_corr_series
from assetforecast.models.condcorr.embed import embed_anchor_targets
from assetforecast.models.condcorr.markov import fit_ms
from assetforecast.models.condcorr.regression import fit_ridge
from assetforecast.models.ref import nearest_corr

NAMES = ["condcorr_sb_base", "condcorr_sb_state", "condcorr_sb_macro", "condcorr_msw",
         "condcorr_msw_infl", "condcorr_stress"]
ORIGINS = ["1982-06-30", "1996-12-31", "2014-11-30"]   # early one: monthly-only assets


def _with_macro(p: Panel, seed: int = 0) -> Panel:
    """Synthetic panel plus the macro predictors the models ask for (monthly,
    published a month late) so the full information ladder is exercised."""
    rng = np.random.default_rng(seed)
    m = p.monthly.index
    rows = []
    for name, lvl in [("macro_infl_yoy", 0.03), ("macro_infl_vol_36m", 0.01), ("macro_long_yield", 0.06),
                      ("credit_moodys_baa_aaa_spread_m", 0.01)]:
        v = lvl + 0.3 * lvl * rng.standard_normal(len(m)).cumsum() / np.sqrt(len(m))
        rows.append(pd.DataFrame({"obs_date": m, "available_date": m + pd.offsets.MonthEnd(1),
                                  "series": name, "value": v}))
    pred = pd.concat([p.predictors] + rows, ignore_index=True)[PREDICTOR_COLUMNS]
    return Panel(daily=p.daily, monthly=p.monthly, cash_daily=p.cash_daily, cash_monthly=p.cash_monthly,
                 tiers=p.tiers, predictors=pred)


@pytest.fixture(scope="module")
def panel():
    return _with_macro(make_synthetic_panel(seed=3))


def _check_corr(c, assets):
    a = c.to_numpy(float)
    assert list(c.index) == list(assets) and list(c.columns) == list(assets)
    assert np.isfinite(a).all()
    assert np.allclose(a, a.T, atol=1e-12)
    assert np.allclose(np.diag(a), 1.0)
    assert np.abs(a).max() <= 1.0 + 1e-12
    assert np.linalg.eigvalsh(a).min() > 0


def test_registered():
    assert set(NAMES) <= set(list_models("corr"))
    assert len([n for n in list_models("corr") if n.startswith("condcorr_")]) <= 8


@pytest.mark.parametrize("name", NAMES)
def test_valid_at_every_stage(panel, name):
    m = get_model("corr", name)
    for t in ORIGINS:
        view = panel.asof(t)
        assets = view.eligible()
        _check_corr(m.forecast_corr(view, assets), assets)
        # any asset subset / order, including ones without the anchor pairs
        rng = np.random.default_rng(1)
        for _ in range(3):
            sub = list(rng.permutation(assets)[: max(2, len(assets) // 2)])
            _check_corr(m.forecast_corr(view, sub), sub)


@pytest.mark.parametrize("name", NAMES)
def test_conditional_part_is_active_late(panel, name):
    """By the late origin the model must actually move its anchor pairs."""
    m = get_model("corr", name)
    view = panel.asof("2014-11-30")
    m.forecast_corr(view, view.eligible())
    assert m.last_info["pred"] is not None and m.last_info["targets"]


@pytest.mark.parametrize("name", NAMES)
def test_cold_equals_warm_any_order(panel, name):
    ts = ["2003-12-31", "2009-06-30", "2012-01-31"]
    warm = get_model("corr", name)
    first = {t: warm.forecast_corr(panel.asof(t), panel.asof(t).eligible()) for t in ts}
    again = {t: warm.forecast_corr(panel.asof(t), panel.asof(t).eligible()) for t in reversed(ts)}
    for t in ts:
        cold = get_model("corr", name).forecast_corr(panel.asof(t), panel.asof(t).eligible())
        pd.testing.assert_frame_equal(first[t], cold, atol=1e-12, rtol=0)
        pd.testing.assert_frame_equal(again[t], cold, atol=1e-12, rtol=0)


@pytest.mark.parametrize("name", NAMES)
def test_no_lookahead(panel, name):
    t = pd.Timestamp("2008-06-30")
    rng = np.random.default_rng(7)
    d, mth = panel.daily.copy(), panel.monthly.copy()
    later_d, later_m = d.index > t, mth.index > t
    d.loc[later_d] = d.loc[later_d] * rng.uniform(-3, 3, size=d.loc[later_d].shape)
    mth.loc[later_m] = mth.loc[later_m] * rng.uniform(-3, 3, size=mth.loc[later_m].shape)
    pr = panel.predictors.copy()
    late = pr["available_date"] > t
    pr.loc[late, "value"] = pr.loc[late, "value"] * 5 + 1
    alt = Panel(daily=d, monthly=mth, cash_daily=panel.cash_daily, cash_monthly=panel.cash_monthly,
                tiers=panel.tiers, predictors=pr)
    a = get_model("corr", name).forecast_corr(panel.asof(t), panel.asof(t).eligible())
    b = get_model("corr", name).forecast_corr(alt.asof(t), alt.asof(t).eligible())
    pd.testing.assert_frame_equal(a, b, atol=1e-12, rtol=0)


def test_fallback_is_base_matrix_early(panel):
    """Before ten years of training windows the model is the base matrix."""
    view = panel.asof("1982-06-30")
    assets = view.eligible()
    ref = get_model("corr", "ref_corr_last120m").forecast_corr(view, assets)
    for name in NAMES:
        c = get_model("corr", name).forecast_corr(view, assets)
        np.testing.assert_allclose(c.to_numpy(), ref.to_numpy(), atol=1e-12)


def test_realized_series_matches_targets():
    p = make_synthetic_panel(seed=5)
    T = compute_targets(p)
    pairs = [("us_large", "ust_int"), ("ust_long", "gold"), ("intl_dev", "reits")]
    rc, daily = realized_corr_series(p.asof(p.daily.index[-1]), pairs)
    for a, b in pairs:
        t = T.corr[(T.corr.asset_i == a) & (T.corr.asset_j == b)].set_index("origin")["corr"]
        mine = rc[f"{a}~{b}"].copy()
        mine.index = mine.index - pd.offsets.MonthEnd(12)
        j = pd.concat([t, mine.reindex(t.index)], axis=1).dropna()
        assert len(j) > 100
        np.testing.assert_allclose(j.iloc[:, 0], j.iloc[:, 1], atol=1e-10)


def test_predictors_at_respects_publication(panel):
    v = panel.asof("2000-06-30")
    x = predictors_at(v, ["macro_infl_yoy"])
    # value known at 2000-06-30 is the May observation (published a month later)
    full = panel.predictors
    may = full[(full.series == "macro_infl_yoy") & (full.obs_date == pd.Timestamp("2000-05-31"))]["value"].iloc[0]
    assert x["macro_infl_yoy"].iloc[-1] == pytest.approx(may)


@pytest.mark.parametrize("seed", range(20))
def test_embedding_always_positive_definite(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(3, 13))
    # block-structured base with strong within-block correlation
    f = rng.standard_normal((n, 3)) * rng.uniform(0.2, 1.5, (n, 1))
    C0 = f @ f.T + np.diag(rng.uniform(0.01, 0.5, n))
    d = np.sqrt(np.diag(C0))
    C0 = nearest_corr(C0 / np.outer(d, d))
    assets = [f"a{i}" for i in range(n)]
    i, j, k = rng.choice(n, 3, replace=False)
    targets = {(assets[i], assets[j]): float(rng.uniform(-0.99, 0.99)),
               (assets[i], assets[k]): float(rng.uniform(-0.99, 0.99))}
    C, lam = embed_anchor_targets(C0, assets, targets)
    assert 0.0 <= lam <= 1.0
    assert np.allclose(np.diag(C), 1.0) and np.allclose(C, C.T)
    assert np.linalg.eigvalsh(C).min() > 0
    if lam == 1.0:
        for (a, b), v in targets.items():
            assert C[assets.index(a), assets.index(b)] == pytest.approx(v, abs=1e-10)


def test_embedding_moves_only_through_anchor_loadings():
    # two independent blocks: changing a correlation inside block 1 leaves block 2 alone
    C0 = np.eye(5)
    C0[:3, :3] = 0.6
    C0[3:, 3:] = 0.5
    np.fill_diagonal(C0, 1.0)
    assets = list("abcde")
    C, lam = embed_anchor_targets(C0, assets, {("a", "b"): 0.9})
    assert lam == 1.0 and C[0, 1] == pytest.approx(0.9)
    np.testing.assert_allclose(C[3:, 3:], C0[3:, 3:], atol=1e-12)
    np.testing.assert_allclose(C[:3, 3:], 0.0, atol=1e-12)


def test_ridge_signs_and_shrinkage():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((400, 2))
    y = 0.3 * X[:, 0] - 0.3 * X[:, 1] + 0.1 * rng.standard_normal(400)
    f = fit_ridge(X, y, np.full(400, 1 / 12), ["a", "b"], tau=0.2, nonneg=[True, True])
    assert f.coef[0] > 0.15 and f.coef[1] == 0.0
    loose = fit_ridge(X, y, np.full(400, 1.0), ["a", "b"], tau=10.0, nonneg=[False, False])
    assert loose.coef[0] == pytest.approx(0.3, abs=0.03) and loose.coef[1] == pytest.approx(-0.3, abs=0.03)


def test_markov_recovers_two_correlation_regimes():
    rng = np.random.default_rng(1)
    T = 400
    rho = np.where((np.arange(T) // 100) % 2 == 0, 0.5, -0.4)
    S = np.zeros((T, 3))
    n = np.full(T, 4.0)
    for t in range(T):
        L = np.linalg.cholesky([[1, rho[t]], [rho[t], 1]])
        u = rng.standard_normal((4, 2)) @ L.T
        S[t] = [u[:, 0] @ u[:, 0], u[:, 0] @ u[:, 1], u[:, 1] @ u[:, 1]]
    par = fit_ms(S, n)
    assert par.rho[0] == pytest.approx(0.5, abs=0.12) and par.rho[1] == pytest.approx(-0.4, abs=0.12)
    assert np.diag(par.P).min() > 0.95
