"""The questionnaire: the stock-share rule, the fund arithmetic, and that the committed tables and page agree."""
import numpy as np
import pandas as pd
import pytest

from assetforecast import questionnaire as q

WS = pd.Series({"ust_int": 0.5, "gold": 0.2, "mf": 0.3})


def _curve(falls, margin=-0.51, stacked=-0.50):
    shares = np.linspace(0, 1, len(falls))
    unl = pd.DataFrame({"segment": "unlevered", "stocks": shares, "sleeve": 1 - shares, "worst_fall_all": falls})
    top = pd.DataFrame({"segment": ["margin", "stacked"], "stocks": 1.0, "sleeve": q.OVERLAY, "worst_fall_all": [margin, stacked]})
    return pd.concat([unl, top], ignore_index=True)


def test_stock_share_is_the_largest_within_the_limit():
    unl = _curve([-0.09, -0.07, -0.20, -0.35, -0.54]).query("segment == 'unlevered'")
    assert q.stock_share(unl, 0.20) == 0.5
    assert q.stock_share(unl, 0.08) == 0.25          # the largest that passes, though 0% does not
    assert q.stock_share(unl, 0.55) == 1.0
    assert q.stock_share(unl, 0.05) is None


def test_decide_backfills_then_puts_the_sleeve_on_top_by_the_accounts_financing():
    cv = _curve([-0.09, -0.07, -0.20, -0.35, -0.54])
    assert q.decide("taxable", "near", cv)["cash"] == 1.0
    d = q.decide("roth", "medium", cv)
    assert (d["stocks"], d["sleeve"], d["borrowed"], d["financing"]) == (0.5, 0.5, 0.0, "none")
    d = q.decide("taxable", "growth", cv)
    assert (d["stocks"], d["sleeve"], d["borrowed"], d["financing"]) == (1.0, q.OVERLAY, q.OVERLAY, "margin")
    assert q.on_curve(cv, d)["segment"] == "margin"
    assert q.decide("traditional", "growth", cv)["financing"] == "stacked"
    d = q.decide("taxable", "growth", _curve([-0.09, -0.07, -0.20, -0.35, -0.54], margin=-0.60))   # on top, it breaks the limit
    assert (d["sleeve"], d["borrowed"]) == (0.0, 0.0)
    assert q.on_curve(cv, d)["segment"] == "unlevered"


def test_overlay_caps_stop_where_the_financing_does():
    caps = q.overlay_caps(WS, q.vehicles())
    assert caps["margin"] == pytest.approx(q.frontier.MAX_LEVERAGE - 1)
    assert 0 < caps["stacked"] <= caps["margin"]
    d = {"stocks": 1.0, "sleeve": caps["stacked"], "cash": 0.0, "borrowed": caps["stacked"], "financing": "stacked"}
    assert sum(r["weight"] for r in q.fund_plan(d, WS, q.vehicles(), stacked=True)) <= 1 + 1e-9
    with pytest.raises(ValueError):                   # one step further and the funds need more than the capital
        q.fund_plan({**d, "sleeve": caps["stacked"] + q.STEP}, WS, q.vehicles(), stacked=True)


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
    ans, cv, meta = d["answers"], d["curve"], d["meta"]
    unl = cv[cv["segment"] == "unlevered"]
    assert len(ans) == len(q.ACCOUNTS) * len(q.GOALS)
    assert set(cv["segment"]) == {"unlevered", "margin", "stacked"}
    for seg, cap in meta["settings"]["overlay_caps"].items():
        assert cv[cv["segment"] == seg]["sleeve"].max() == pytest.approx(cap)
    for _, r in ans.iterrows():
        limit = q.GOALS[r["goal"]]["loss_limit"]
        if limit is None:
            assert r["cash"] == 1.0 and r["headline"] == "T-bills"
            continue
        assert r["stocks"] == q.stock_share(unl, limit) and r["worst_fall_all"] >= -limit
        assert r["stocks"] + r["sleeve"] == pytest.approx(1 + r["borrowed"])
        assert sum(r[f"w_{a}"] for a in q.SLEEVE) == pytest.approx(r["sleeve"], abs=1e-5)
        p = q.on_curve(cv, r)                           # every answer is a point of the curve, with its numbers
        assert p["segment"] == (q.ACCOUNTS[r["account"]]["financing"] if r["borrowed"] > 0 else "unlevered")
        for c in ("vol", "growth", "growth_lo", "growth_hi", "worst_fall_all", "growth_all"):
            assert r[c] == pytest.approx(p[c])
    # the curve is the allocation study's replay and forecast risk: its menu is reproduced at the shared mixes
    menu = pd.read_csv(al.OUT / "menu.csv")
    menu = menu[menu["sleeve_rule"] == q.SLEEVE_RULE]
    for _, m in menu.iterrows():
        seg = "margin" if m["borrowed"] > 0 else "unlevered"
        row = cv[(cv["segment"] == seg) & np.isclose(cv["stocks"], m["stocks"]) & np.isclose(cv["sleeve"], m["sleeve"])].iloc[0]
        for a, b in (("worst_fall_all", "worst_fall_all"), ("growth_all", "growth_all"), ("vol_all", "vol_all"), ("vol", "vol_fcst")):
            assert row[a] == pytest.approx(m[b], rel=1e-4)
    assert not {"frontier.csv", "points.csv", "ladder.csv"} & {f.name for f in q.OUT.iterdir()}
    html = pg.fragment(d)
    assert meta["label"] in html and "qn-data" in html
    for k in (*q.ACCOUNTS, *q.GOALS):
        assert f'value="{k}"' in html
    assert (q.REPORT).read_text().count("| Taxable brokerage |") == len(q.GOALS)
