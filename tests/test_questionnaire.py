"""The questionnaire: the stock-share rule, the fund arithmetic, and that the committed tables and page agree."""
import numpy as np
import pandas as pd
import pytest

from assetforecast import questionnaire as q

WS = pd.Series({"ust_int": 0.5, "gold": 0.2, "mf": 0.3})


def _ladder(falls):
    shares = np.linspace(0, 1, len(falls))
    return pd.DataFrame({"stocks": shares, "sleeve": 1 - shares, "worst_fall_all": falls})


def test_stock_share_is_the_largest_within_the_limit():
    lad = _ladder([-0.09, -0.07, -0.20, -0.35, -0.54])
    assert q.stock_share(lad, 0.20) == 0.5
    assert q.stock_share(lad, 0.08) == 0.25          # the largest that passes, though 0% does not
    assert q.stock_share(lad, 0.55) == 1.0
    assert q.stock_share(lad, 0.05) is None


def test_decide_backfills_then_overlays_by_the_accounts_financing():
    lad, over = _ladder([-0.09, -0.07, -0.20, -0.35, -0.54]), {"margin": -0.51, "stacked": -0.50}
    assert q.decide("taxable", "near", lad, over)["cash"] == 1.0
    d = q.decide("roth", "medium", lad, over)
    assert (d["stocks"], d["sleeve"], d["borrowed"], d["financing"]) == (0.5, 0.5, 0.0, "none")
    d = q.decide("taxable", "growth", lad, over)
    assert (d["stocks"], d["sleeve"], d["borrowed"], d["financing"]) == (1.0, q.OVERLAY, q.OVERLAY, "margin")
    assert q.decide("traditional", "growth", lad, over)["financing"] == "stacked"
    d = q.decide("taxable", "growth", lad, {"margin": -0.60, "stacked": -0.50})   # the overlay itself breaks the limit
    assert (d["sleeve"], d["borrowed"]) == (0.0, 0.0)


def test_mix_stats_charges_the_spread_on_what_is_borrowed():
    idx = pd.date_range("2000-01-31", periods=120, freq="ME")
    rng = np.random.default_rng(1)
    st, sl = pd.Series(rng.normal(0.005, 0.04, 120), index=idx), pd.Series(rng.normal(0.003, 0.02, 120), index=idx)
    cash = pd.Series(0.002, index=idx)
    free, paid = q.mix_stats(st, sl, cash, 1.0, 0.5, 0.0), q.mix_stats(st, sl, cash, 1.0, 0.5, 0.02)
    assert free["growth_all"] - paid["growth_all"] == pytest.approx(0.01, abs=2e-4)
    assert q.mix_stats(st, sl, cash, 0.6, 0.4, 0.02) == q.mix_stats(st, sl, cash, 0.6, 0.4, 0.0)


def test_fund_plan_delivers_the_exposures():
    veh = q.vehicles()
    d = {"stocks": 1.0, "sleeve": 0.5, "cash": 0.0, "borrowed": 0.5, "financing": "stacked"}
    rows = q.fund_plan(d, WS, veh, stacked=True)
    assert sum(r["weight"] for r in rows) == pytest.approx(1.0)          # no loan: the funds use the capital once
    assert sum(r["exposure"] for r in rows) == pytest.approx(0.5)
    by = {r["role"]: r for r in rows}
    assert by["gold"]["exposure"] == pytest.approx(0.1) and by["gold"]["symbol"] == "GDE"
    assert by["ust_int"]["symbol"] == "RSSB"                             # more Treasuries per dollar of stocks than NTSX
    assert 0.9 < sum(r["stocks"] for r in rows) <= 1.0
    out = q.fund_plan({**d, "financing": "margin"}, WS, veh, stacked=False)
    assert sum(r["weight"] for r in out) == pytest.approx(1.5)
    fees = {r["exposure"]: min(v["fee"] for v in veh["vehicles"] if v["kind"] == "outright" and v["exposure"] == r["exposure"])
            for r in veh["vehicles"]}
    assert all(r["fee"] == fees[q.EXPOSURE[r["role"]]] for r in out if r["role"] != q.STOCKS)
    assert q.fund_plan({"stocks": 0, "sleeve": 0, "cash": 1.0, "borrowed": 0, "financing": "none"}, WS, veh, False)[0]["role"] == "cash"


def test_committed_tables_and_page_agree():
    from assetforecast import allocation as al
    from assetforecast import questionnaire_page as pg
    if not (q.OUT / "answers.csv").exists():
        pytest.skip("results/questionnaire not written")
    d = pg.load()
    ans, lad, meta = d["answers"], d["ladder"], d["meta"]
    assert len(ans) == len(q.ACCOUNTS) * len(q.GOALS)
    for _, r in ans.iterrows():
        limit = q.GOALS[r["goal"]]["loss_limit"]
        if limit is None:
            assert r["cash"] == 1.0 and r["headline"] == "T-bills"
            continue
        assert r["stocks"] == q.stock_share(lad, limit) and r["worst_fall_all"] >= -limit
        assert r["stocks"] + r["sleeve"] == pytest.approx(1 + r["borrowed"])
        assert sum(r[f"w_{a}"] for a in q.SLEEVE) == pytest.approx(r["sleeve"], abs=1e-5)
        assert r["match_vol"] == pytest.approx(r["vol"], abs=1e-4) and r["gap"] >= -1e-6      # the frontier is no worse
        assert r["gap_lo"] <= r["gap"] <= r["gap_hi"]
        if r["borrowed"] > 0:
            assert r["financing"] == q.ACCOUNTS[r["account"]]["financing"]
    # the ladder is the allocation study's replay: its menu is reproduced at the shared mixes
    menu = pd.read_csv(al.OUT / "menu.csv")
    menu = menu[(menu["sleeve_rule"] == q.SLEEVE_RULE) & (menu["borrowed"] == 0)]
    for _, m in menu.iterrows():
        row = lad[np.isclose(lad["stocks"], m["stocks"])].iloc[0]
        for c in ("worst_fall_all", "growth_all", "vol_all", "vol_fcst"):
            assert row[c] == pytest.approx(m[c], rel=1e-4)
    over = pd.read_csv(al.OUT / "menu.csv").query("sleeve_rule == @q.SLEEVE_RULE and sleeve == 0.5 and stocks == 1").iloc[0]
    t = ans[(ans["account"] == "taxable") & (ans["goal"] == "growth")].iloc[0]
    if t["borrowed"] > 0:
        assert t["worst_fall_all"] == pytest.approx(over["worst_fall_all"], rel=1e-4)
        assert t["vol"] == pytest.approx(over["vol_fcst"], rel=1e-4)
    html = pg.fragment(d)
    assert meta["label"] in html and "qn-data" in html
    for k in (*q.ACCOUNTS, *q.GOALS):
        assert f'value="{k}"' in html
    assert (q.REPORT).read_text().count("| Taxable brokerage |") == len(q.GOALS)
