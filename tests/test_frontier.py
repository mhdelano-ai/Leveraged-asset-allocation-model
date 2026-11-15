"""assetforecast.frontier: the optimisations on a small synthetic case, and the published files."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from assetforecast import frontier as F
from assetforecast.paths import FORECASTS_OUT


@pytest.fixture(scope="module")
def inp():
    rng = np.random.default_rng(7)
    n = 6
    assets = [f"a{i}" for i in range(n)]
    sd = np.array([0.05, 0.08, 0.12, 0.16, 0.20, 0.24])
    B = rng.normal(size=(n, 3))
    R = B @ B.T + 2 * np.eye(n)
    R /= np.sqrt(np.outer(np.diag(R), np.diag(R)))
    cov = pd.DataFrame(R * np.outer(sd, sd), assets, assets)
    mean = np.array([0.012, 0.02, 0.03, 0.05, 0.055, 0.04])
    exp = pd.DataFrame({"mean_12m": mean, "ci90_lo": mean - 0.03, "ci90_hi": mean + 0.03}, index=assets)
    return F.Inputs(cov, exp, {"half": {"a0": 0.5, "a3": 0.5}}, spread=0.01)


def random_long_only(rng, n, k):
    w = rng.dirichlet(np.full(n, 0.5), size=k)
    return w


def test_growth_of_one_asset_is_its_log_mean(inp):
    for i in range(len(inp.assets)):
        s = F.stats(inp, np.eye(len(inp.assets))[i])
        assert s["growth"] == pytest.approx(inp.mean[i], abs=1e-12)
        assert s["vol"] == pytest.approx(np.sqrt(inp.cov[i, i]))
        assert s["growth_hi"] - s["growth_lo"] == pytest.approx(0.06)


def test_borrowing_cost_and_band_scale_with_leverage(inp):
    x = np.full(len(inp.assets), 1 / len(inp.assets))
    one, two = F.stats(inp, x, spread=0.01), F.stats(inp, 2 * x, spread=0.01)
    assert two["drift"] == pytest.approx(2 * one["drift"] - 0.01)
    assert two["vol"] == pytest.approx(2 * one["vol"])
    assert two["growth"] == pytest.approx(two["drift"] - 0.5 * two["vol"] ** 2)
    assert two["growth_hi"] - two["growth_lo"] == pytest.approx(2 * (one["growth_hi"] - one["growth_lo"]))
    assert two["invested"] == pytest.approx(2.0)


def test_named_portfolios(inp):
    rng = np.random.default_rng(0)
    W = random_long_only(rng, len(inp.assets), 4000)
    var = np.einsum("pi,ij,pj->p", W, inp.cov, W)
    a = F.drift(inp)
    mv, ms, rp = F.min_variance(inp), F.max_sharpe(inp), F.risk_parity(inp)
    for x in (mv, ms, rp):
        assert x.sum() == pytest.approx(1) and (x >= 0).all()
    assert mv @ inp.cov @ mv <= var.min() + 1e-9
    assert (ms @ a) / np.sqrt(ms @ inp.cov @ ms) >= ((W @ a) / np.sqrt(var)).max() - 1e-7
    share = rp * (inp.cov @ rp) / (rp @ inp.cov @ rp)
    np.testing.assert_allclose(share, 1 / len(inp.assets), atol=1e-6)


def test_nothing_long_only_beats_the_frontier(inp):
    """No random portfolio within the limits has more growth than the frontier at its volatility."""
    rng = np.random.default_rng(1)
    n = len(inp.assets)
    for levered in (False, True):
        xs = F.curve(inp, levered, n_points=25)
        st = pd.DataFrame([F.stats(inp, x) for x in xs])
        assert (np.diff(st.vol) > 0).all()
        for x in xs:
            assert (x >= 0).all() and x.sum() <= F.MAX_LEVERAGE + 1e-6
            if not levered:
                assert x.sum() == pytest.approx(1, abs=1e-6)
        W = random_long_only(rng, n, 40)
        if levered:
            W = W * rng.uniform(0, F.MAX_LEVERAGE, size=(len(W), 1))
        for w in W:
            s = F.stats(inp, w)
            if st.vol.min() < s["vol"] < st.vol.max():
                best = F.stats(inp, F.at_volatility(inp, s["vol"], levered))
                assert best["vol"] == pytest.approx(s["vol"], abs=1e-6)
                assert s["growth"] <= best["growth"] + 1e-6
    # the levered frontier is the fully invested one or better, and starts in cash
    lev = pd.DataFrame([F.stats(inp, x) for x in F.curve(inp, True, n_points=25)])
    unl = pd.DataFrame([F.stats(inp, x) for x in F.curve(inp, False, n_points=25)])
    for _, u in unl.iloc[::6].iterrows():
        assert F.stats(inp, F.at_volatility(inp, u.vol, True))["growth"] >= u.growth - 1e-6
    assert lev.vol.iloc[0] == 0 and lev.invested.iloc[0] == 0


def test_borrowing_only_past_full_investment_and_spread_hurts(inp):
    cheap = [F.stats(inp, F.at_volatility(inp, 0.15, True, spread=s), s) for s in (0.0, 0.01, 0.03)]
    assert cheap[0]["growth"] > cheap[1]["growth"] > cheap[2]["growth"]
    assert cheap[0]["invested"] >= cheap[1]["invested"] - 1e-9 >= cheap[2]["invested"] - 2e-9
    low = F.at_volatility(inp, 0.02, True)
    assert low.sum() < 1
    ms = F.max_sharpe(inp)
    np.testing.assert_allclose(low / low.sum(), ms, atol=2e-3)      # cash plus the best return-per-risk mix


def test_match_points_and_gap(inp):
    res = F.build(inp, n_points=9)
    p = res["points"].set_index("point")
    assert list(res["frontier"].curve.unique()) == ["unlevered", "levered"]
    assert p.loc["match_half", "vol"] == pytest.approx(p.loc["half", "vol"], abs=1e-5)
    assert p.loc["match_half", "gap"] == pytest.approx(p.loc["match_half", "growth"] - p.loc["half", "growth"])
    assert p.loc["match_half", "gap"] >= -1e-9
    assert p.loc["match_half", "gap_lo"] <= p.loc["match_half", "gap"] <= p.loc["match_half", "gap_hi"]
    assert p.loc[p.kind != "match", "gap"].isna().all()


def test_borrow_terms_are_the_margin_rate_over_tbills():
    cash = pd.Series({pd.Timestamp("2026-09-30"): 0.0030, pd.Timestamp("2026-10-31"): 0.0032,
                      pd.Timestamp("2026-12-31"): 0.0025})
    b = F.borrow_terms("2026-10-31", cash=cash)
    lo, hi, rate = F.MARGIN["tiers"][F.MARGIN_TIER]
    assert (b["margin_rate"], b["tbill_month"], b["tier"]) == (rate, "2026-10-31", "up to $50,000")
    assert b["annual_cost"] == pytest.approx(rate * 365 / 360, abs=1e-6)
    assert b["tbill_rate"] == pytest.approx(12 * 0.0032)
    assert b["spread"] == pytest.approx(b["annual_cost"] - b["tbill_rate"], abs=1e-9)
    # a bigger balance borrows cheaper; the T-bill rate is the origin month's and nothing later
    rates = [r for _, _, r in F.MARGIN["tiers"]]
    assert rates == sorted(rates, reverse=True) and len(b["schedule"]) == len(rates)
    assert F.borrow_terms("2026-10-31", tier=2, cash=cash)["spread"] < b["spread"]
    assert F.borrow_terms("2026-10-31", cash=cash.iloc[:2]) == b
    # an origin before the schedule's date is measured in the schedule's month; a missing month is an error
    assert F.borrow_terms("2020-01-31", cash=cash)["tbill_month"] == "2026-09-30"
    with pytest.raises(ValueError, match="no T-bill return"):
        F.borrow_terms("2026-11-30", cash=cash)


def test_spread_defaults_to_the_inputs(inp):
    x = np.full(len(inp.assets), 2 / len(inp.assets))
    assert F.stats(inp, x) == F.stats(inp, x, spread=inp.spread)
    assert F.stats(inp, x)["drift"] == pytest.approx(F.stats(inp, x, spread=0.0)["drift"] - inp.spread)
    with pytest.raises(ValueError, match="spread"):
        F.Inputs(pd.DataFrame(inp.cov, inp.assets, inp.assets),
                 pd.DataFrame({"mean_12m": inp.mean, "ci90_lo": inp.mean, "ci90_hi": inp.mean}, index=inp.assets),
                 {}, spread=float("nan"))


def test_inputs_must_line_up(inp):
    cov = pd.DataFrame(inp.cov, inp.assets, inp.assets)
    exp = pd.DataFrame({"mean_12m": inp.mean, "ci90_lo": inp.mean, "ci90_hi": inp.mean}, index=inp.assets)
    with pytest.raises(ValueError, match="same assets"):
        F.Inputs(cov, exp.iloc[1:], {}, spread=0.01)


# ------------------------------------------------------------------ the published files
@pytest.fixture(scope="module")
def latest():
    d = FORECASTS_OUT / "latest"
    if not (d / F.FRONTIER_FILE).exists():
        pytest.skip("no published frontier")
    return d


@pytest.mark.realdata
def test_published_frontier_is_what_the_code_writes(latest, tmp_path):
    for name in ("cov_annual.csv", "expected_returns.csv", "forecast.json"):
        (tmp_path / name).write_bytes((latest / name).read_bytes())
    F.publish(tmp_path)
    for name in (F.FRONTIER_FILE, F.POINTS_FILE):
        a, b = pd.read_csv(latest / name), pd.read_csv(tmp_path / name)
        assert list(a.columns) == list(b.columns)
        num = a.select_dtypes("number").columns
        np.testing.assert_allclose(a[num], b[num], atol=2e-5, equal_nan=True)
    assert json.loads((tmp_path / "forecast.json").read_text()) == json.loads((latest / "forecast.json").read_text())


@pytest.mark.realdata
def test_published_frontier_hangs_together(latest):
    f, p = pd.read_csv(latest / F.FRONTIER_FILE), pd.read_csv(latest / F.POINTS_FILE).set_index("point")
    er = pd.read_csv(latest / "expected_returns.csv", index_col=0)
    cov = pd.read_csv(latest / "cov_annual.csv", index_col=0)
    ports = pd.read_csv(latest / "portfolios.csv", index_col=0)
    meta = json.loads((latest / "forecast.json").read_text())["frontier"]
    origin = json.loads((latest / "forecast.json").read_text())["origin"]
    assert meta["borrow"] == F.borrow_terms(origin) and meta["max_leverage"] == F.MAX_LEVERAGE
    spread = meta["borrow_spread"]
    assert spread == meta["borrow"]["spread"] > 0
    assert meta["borrow"]["margin_rate"] == F.MARGIN["tiers"][F.MARGIN_TIER][2]
    assert meta["borrow"]["tbill_month"] == origin
    np.testing.assert_allclose(p.loc[er.index, "growth"], er["mean_12m"], atol=1e-5)
    np.testing.assert_allclose(p.loc[er.index, "growth_lo"], er["ci90_lo"], atol=2e-5)
    np.testing.assert_allclose(p.loc[er.index, "vol"], np.sqrt(np.diag(cov)), atol=1e-5)
    np.testing.assert_allclose(p.loc[ports.index, "vol"], ports["vol_12m_return"], atol=1e-5)
    np.testing.assert_allclose(p.loc[["match_" + k for k in ports.index], "vol"], ports["vol_12m_return"], atol=2e-5)
    w = f[[c for c in f.columns if c.startswith("w_")]]
    assert (w >= 0).all().all()
    np.testing.assert_allclose(w.sum(axis=1), f.invested, atol=1e-4)
    assert f.invested.max() <= F.MAX_LEVERAGE + 1e-4
    assert (f.loc[f.curve == "unlevered", "invested"].round(4) == 1).all()
    drift = er["mean_12m"] + 0.5 * np.diag(cov.loc[er.index, er.index])
    x = w.to_numpy()
    var = np.einsum("pi,ij,pj->p", x, cov.to_numpy(), x)
    g = x @ drift.to_numpy() - spread * np.clip(f.invested - 1, 0, None) - 0.5 * var
    np.testing.assert_allclose(f.growth, g, atol=5e-5)
    np.testing.assert_allclose(f.vol, np.sqrt(var), atol=5e-5)
    # the policy portfolios do not sit above the frontier
    lev = f[f.curve == "levered"]
    for k in ports.index:
        assert p.loc[k, "growth"] <= np.interp(p.loc[k, "vol"], lev.vol, lev.growth) + 1e-4
        assert p.loc["match_" + k, "gap"] >= 0
