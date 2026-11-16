"""The allocation study: weight solvers, the splice, no look-ahead in the replay, and the page."""
import numpy as np
import pandas as pd
import pytest

from assetforecast import allocation as al


def _x(n=150, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("1985-01-31", periods=n, freq="ME")
    vol = np.array([0.045, 0.02, 0.045, 0.025])
    return pd.DataFrame(rng.normal(0.004, 1, (n, 4)) * vol, index=idx, columns=al.ASSETS)


def _fc(x, scale=1.0):
    C = pd.DataFrame(np.diag([0.0225, 0.0045, 0.03]) * scale, index=al.ASSETS[:3], columns=al.ASSETS[:3])
    return {t: C for t in x.index}


def test_min_var_and_max_sharpe_are_long_only_optima():
    rng = np.random.default_rng(3)
    for _ in range(20):
        B = rng.normal(size=(4, 4))
        C = pd.DataFrame(B @ B.T + 0.1 * np.eye(4), index=al.ASSETS, columns=al.ASSETS)
        mu = pd.Series(rng.normal(0.03, 0.05, 4), index=al.ASSETS)
        W = rng.dirichlet(np.ones(4), 4000)
        var = np.einsum("ni,ij,nj->n", W, C.to_numpy(), W)
        w = al.min_var(C)
        assert w.min() >= 0 and w.sum() == pytest.approx(1)
        assert w @ C @ w <= var.min() + 1e-12
        s = al.max_sharpe(mu, C)
        assert s.min() >= 0 and s.sum() == pytest.approx(1)
        if (mu > 0).any():
            assert s @ mu / np.sqrt(s @ C @ s) >= (W @ mu.to_numpy() / np.sqrt(var)).max() - 1e-9
        else:
            assert np.allclose(s, w)


def test_inv_vol_and_scaling():
    C = pd.DataFrame(np.diag([0.04, 0.01, 0.04, 0.01]), index=al.ASSETS, columns=al.ASSETS)
    assert al.inv_vol(C).round(4).tolist() == [0.1667, 0.3333, 0.1667, 0.3333]
    w = al._scaled(al.base_weights(), C)
    assert np.sqrt(w @ C @ w) == pytest.approx(al.TARGET_VOL)
    assert al._scaled(al.base_weights(), C * 1e-4).sum() == pytest.approx(al.frontier.MAX_LEVERAGE)


def test_spliced_mf_matches_the_funds_where_both_exist():
    rng = np.random.default_rng(5)
    idx = pd.date_range("2000-01-31", periods=240, freq="ME")
    t = pd.Series(rng.normal(0.01, 0.04, 240), index=idx)
    f = (0.5 * t - 0.001).iloc[120:]
    mf = al.spliced_mf(t, f, al.diversifiers.measured_drag(t, f))
    assert mf.iloc[120:].equals(f)
    assert mf.iloc[:120].to_numpy() == pytest.approx((0.5 * t - 0.001).iloc[:120].to_numpy())


def test_forecast_row_for_managed_futures_is_the_recent_estimate():
    x = _x()
    S = np.expm1(x)
    est = al.estimates(S, 100, _fc(x)[x.index[100]])
    C, R = est["fcst"][1], est["recent"][1]
    assert C.loc["mf", "mf"] == pytest.approx(R.loc["mf", "mf"])
    assert C.loc["gold", "gold"] == pytest.approx(0.03)
    assert C.loc["mf", "gold"] / np.sqrt(C.loc["mf", "mf"] * 0.03) == pytest.approx(
        R.loc["mf", "gold"] / np.sqrt(R.loc["mf", "mf"] * R.loc["gold", "gold"]))
    assert np.linalg.eigvalsh(C.to_numpy()).min() > 0


def test_replay_has_no_look_ahead_and_charges_borrowing():
    x = _x()
    r1, w1 = al.walk_forward(x, _fc(x), 0.0)
    y = x.copy()
    y.iloc[120:] = y.iloc[120:] * 3 + 0.01
    r2, w2 = al.walk_forward(y, _fc(y), 0.0)
    cut = x.index[120]                                  # weights held in month 120 were set at the end of month 119
    a, b = w1[w1["date"] <= cut], w2[w2["date"] <= cut]
    pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True))
    assert r1.loc[:x.index[119]].equals(r2.loc[:x.index[119]])
    assert r1.index[0] == x.index[al.BURN]
    assert set(r1.columns) == set(al.SLEEVE_RULES + al.PORTFOLIO_RULES) == set(al.RULE_NAMES) == set(al.RULE_INPUTS)
    lev = _fc(x, 0.05)                                  # a tiny forecast risk: the target rule borrows to the cap
    r0, _ = al.walk_forward(x, lev, 0.0)
    r3, w3 = al.walk_forward(x, lev, 0.024)
    borrowed = w3[w3["rule"] == "target_fcst"].set_index("date")[al.ASSETS].sum(axis=1) - 1
    assert borrowed.min() > 0
    assert (r0["target_fcst"] - r3["target_fcst"]).to_numpy() == pytest.approx(0.002 * borrowed.to_numpy())
    assert r0["fixed"].equals(r3["fixed"])


def test_sharpe_diff_and_persistence():
    rng = np.random.default_rng(7)
    idx = pd.date_range("1990-01-31", periods=360, freq="ME")
    a = pd.Series(rng.normal(0.004, 0.02, 360), index=idx)
    d = al.sharpe_diff(a + 0.004, a, n=300)
    assert d["diff_lo"] > 0 and d["diff_lo"] < d["sharpe_diff"] < d["diff_hi"]
    vol = np.repeat([0.01, 0.05, 0.01, 0.05, 0.01, 0.05], 60)
    p = al.persistence({"a": pd.Series(rng.normal(0, 1, 360) * vol, index=idx)})
    assert p.loc[0, "blocks"] == 6 and p.loc[0, "corr_vol"] < -0.9
    assert p.iloc[-1]["asset"] == "pooled" and p.iloc[-1]["pairs"] == 5


@pytest.mark.realdata
def test_committed_tables_and_page_agree():
    from assetforecast import allocation_page as pg
    d = pg.load()
    assert list(d["sleeve_rules"]["rule"].unique()) == al.SLEEVE_RULES
    assert list(d["portfolio_rules"]["rule"].unique()) == al.PORTFOLIO_RULES
    eq = d["sleeve_rules"].query("rule == 'equal'")
    assert (eq["turnover"] == 0).all() and np.allclose(eq[["w_ust_int", "w_gold", "w_mf"]], 1 / 3, atol=1e-4)
    assert len(pg.answers(d)) == 3
    html = pg.fragment(d)
    assert d["meta"]["label"] in html and "<title>" in html[:200]
    assert pg.PAGE.read_text().count(html) == 1
