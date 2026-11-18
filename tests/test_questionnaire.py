"""The questionnaire: the two dials, the fund arithmetic, and that the committed tables and page agree."""
import numpy as np
import pandas as pd
import pytest

from assetforecast import questionnaire as q

WS = pd.Series({"ust_int": 0.5, "gold": 0.2, "mf": 0.3})


def _curve(falls, route="stacked"):
    n = len(falls)
    return pd.DataFrame({"route": route, "i": range(n),
                         "stocks": np.linspace(0.1, 1.0, n), "sleeve": 1.0, "cash": [0.5] + [0.0] * (n - 1),
                         "borrowed": [0.0] + list(np.linspace(0.2, 1.0, n - 1)), "worst_fall_all": falls})


def test_decide_takes_the_largest_stock_share_within_the_budget():
    cv = _curve([-0.03, -0.08, -0.15, -0.30, -0.50])
    assert q.decide("stacked", "near", cv)["cash"] == 1.0                      # no budget: T-bills
    d = q.decide("stacked", "medium", cv)                                      # 20%: the third point
    assert (d["i"], d["financing"]) == (2, "stacked") and d["stocks"] == pytest.approx(0.55)
    assert q.decide("stacked", "income", cv)["i"] == 3 and q.decide("stacked", "growth", cv)["i"] == 4
    assert q.decide("stacked", "medium", _curve([-0.03, -0.25, -0.15, -0.30, -0.50]))["i"] == 2   # the largest, not the first break
    d = q.decide("stacked", "medium", _curve([-0.03, -0.25, -0.30, -0.40, -0.50]))
    assert (d["i"], d["financing"], d["cash"]) == (0, "none", 0.5)             # below 100%: T-bills beside it, nothing carried
    assert q.decide("stacked", "medium", _curve([-0.25, -0.30, -0.35, -0.40, -0.50]))["cash"] == 1.0


def test_path_is_every_stock_share_with_the_most_sleeve_the_route_carries():
    veh = q.vehicles()
    for route in q.ROUTES:
        pts = q.path(route, WS, veh)
        assert [s for s, _ in pts] == pytest.approx(list(np.arange(0, 1.0001, q.STEP)))
        for s, d in pts:
            assert q.carries(route, s, d, WS, veh) and not q.carries(route, s, d + q.STEP, WS, veh)
    assert all(s + d == pytest.approx(q.frontier.MAX_LEVERAGE) for s, d in q.path("margin", WS, veh))
    stacked = dict(q.path("stacked", WS, veh))
    assert stacked[0.0] == pytest.approx(1.0)                                  # with no stocks to stack on, the capital itself
    assert all(d <= 1.0 + 1e-9 for d in stacked.values())                      # no loan: never more diversifiers than capital
    assert q.carries("margin", 0.3, 0.6, WS, veh) and q.carries("stacked", 0.3, 0.6, WS, veh)   # nothing above 100%
    assert not q.carries("margin", 1.2, 1.0, WS, veh)


def test_stacked_plan_holds_the_exposures_without_a_loan():
    veh = q.vehicles()
    p = q.stacked_plan(0.7, 1.0, WS, veh)
    funds = {a: (q._fund(veh, a, True), q._fund(veh, a, False)) for a in q.SLEEVE}
    dollars = p["stock_fund"] + sum(p["stacked"].values()) + sum(p["plain"].values())
    assert dollars <= 1 + 1e-9
    for a in q.SLEEVE:                                                         # every diversifier exactly
        assert p["stacked"][a] * funds[a][0]["exposure_per_dollar"] + p["plain"][a] == pytest.approx(WS[a])
    stocks = p["stock_fund"] + sum(p["stacked"][a] * funds[a][0]["stocks_per_dollar"] for a in q.SLEEVE)
    assert stocks + p["short"] == pytest.approx(0.7) and p["short"] <= q.STOCK_TOL
    assert q.stacked_plan(0.1, 1.2, WS, veh) is None                           # more diversifiers than capital, few stocks to stack on
    assert q.stacked_plan(1.0, 1.2, WS, veh) is None
    d = {"stocks": 0.7, "sleeve": 1.0, "cash": 0.0, "borrowed": 0.7, "financing": "stacked"}
    rows = q.fund_plan(d, WS, veh)
    assert sum(r["weight"] for r in rows) <= 1 + 1e-9 and sum(r["exposure"] for r in rows) == pytest.approx(1.0)
    out = q.fund_plan({**d, "financing": "margin"}, WS, veh)                   # on margin: plain funds, more than the capital
    assert sum(r["weight"] for r in out) == pytest.approx(1.7) and all(r["symbol"] not in ("RSSB", "RSST", "GDE") for r in out)
    cash = q.fund_plan({"stocks": 0.1, "sleeve": 0.4, "cash": 0.5, "borrowed": 0.0, "financing": "none"}, WS, veh)
    assert cash[-1]["role"] == "cash" and sum(r["weight"] for r in cash) == pytest.approx(1.0)
    assert [r["role"] for r in q.fund_plan({"stocks": 0, "sleeve": 0, "cash": 1.0, "borrowed": 0, "financing": "none"}, WS, veh)] == ["cash"]


def test_mix_stats_charges_the_spread_on_what_is_borrowed():
    idx = pd.date_range("2000-01-31", periods=120, freq="ME")
    rng = np.random.default_rng(1)
    st, sl = pd.Series(rng.normal(0.005, 0.04, 120), index=idx), pd.Series(rng.normal(0.003, 0.02, 120), index=idx)
    cash = pd.Series(0.002, index=idx)
    free, paid = q.mix_stats(st, sl, cash, 1.0, 0.5, 0.0), q.mix_stats(st, sl, cash, 1.0, 0.5, 0.02)
    assert free["growth_all"] - paid["growth_all"] == pytest.approx(0.01, abs=2e-4)
    assert q.mix_stats(st, sl, cash, 0.6, 0.4, 0.02) == q.mix_stats(st, sl, cash, 0.6, 0.4, 0.0)
    half, whole = q.mix_stats(st, sl, cash, 0.3, 0.2, 0.0), q.mix_stats(st, sl, cash, 0.6, 0.4, 0.0)
    assert half["vol_all"] == pytest.approx(whole["vol_all"] / 2, rel=0.02)   # half the mix, the rest in T-bills


def test_committed_tables_and_page_agree():
    from assetforecast import allocation as al
    from assetforecast import questionnaire_page as pg
    if not (q.OUT / "answers.csv").exists():
        pytest.skip("results/questionnaire not written")
    d = pg.load()
    ans, cv, meta = d["answers"], d["curve"], d["meta"]
    assert len(ans) == sum(len(a["routes"]) for a in q.ACCOUNTS.values()) * len(q.GOALS)
    assert set(cv["route"]) == set(q.ROUTES) and sum(meta["sleeve_weights"].values()) == pytest.approx(1.0)
    for route, p in cv.groupby("route"):
        assert p["stocks"].to_numpy() == pytest.approx(np.arange(0, 1.0001, q.STEP))     # every stock share, in order
        assert p["exposure"].max() <= meta["settings"]["max_leverage"] + 1e-9
        assert p.loc[p["vol"].idxmin(), "stocks"] == pytest.approx(meta["least_risk_stocks"][route])
    assert np.allclose(cv[cv["route"] == "margin"]["exposure"], meta["settings"]["max_leverage"])
    for _, r in ans.iterrows():
        limit = q.GOALS[r["goal"]]["loss_limit"]
        assert r["route"] in q.ACCOUNTS[r["account"]]["routes"]
        if limit is None:
            assert r["cash"] == 1.0 and r["headline"] == "T-bills"
            continue
        p = cv[cv["route"] == r["route"]]
        best = p[p["worst_fall_all"] >= -limit]["i"].max()             # the largest stock share within the budget
        row = p[p["i"] == r["i"]].iloc[0]
        assert r["i"] == best and r["worst_fall_all"] >= -limit
        for c in ("stocks", "sleeve", "vol", "growth", "growth_lo", "growth_hi", "worst_fall_all", "growth_all"):
            assert r[c] == pytest.approx(row[c])
        assert sum(r[f"w_{a}"] for a in q.SLEEVE) == pytest.approx(r["sleeve"], abs=1e-5)
        assert r["cost"] == pytest.approx(r["fund_fees"] + r["loan_cost"], abs=1e-6)       # what each route costs, on one footing
        assert r["growth_all_net"] == pytest.approx(r["growth_all"] - r["fund_fees"], abs=1e-6)
        assert (r["loan_cost"] > 0) == (r["financing"] == "margin")
        f = d["funds"][(d["funds"]["account"] == r["account"]) & (d["funds"]["route"] == r["route"]) & (d["funds"]["goal"] == r["goal"])]
        assert f["exposure"].sum() == pytest.approx(r["sleeve"], abs=1e-4)
        assert r["stocks"] - q.STOCK_TOL - 1e-6 <= f["stocks"].sum() <= r["stocks"] + 1e-6
        assert f["weight"].sum() == pytest.approx(1 + (r["borrowed"] if r["financing"] == "margin" else 0), abs=1e-4)
    # the path is the allocation study's replay and forecast risk: its menu is reproduced where they share a mix
    menu = pd.read_csv(al.OUT / "menu.csv")
    menu = menu[menu["sleeve_rule"] == q.SLEEVE_RULE]
    shared = 0
    for _, m in menu.iterrows():
        row = cv[(cv["route"] == "margin") & np.isclose(cv["stocks"], m["stocks"]) & np.isclose(cv["sleeve"], m["sleeve"])]
        for _, p in row.iterrows():
            shared += 1
            for a, b in (("worst_fall_all", "worst_fall_all"), ("growth_all", "growth_all"), ("vol_all", "vol_all"), ("vol", "vol_fcst")):
                assert p[a] == pytest.approx(m[b], rel=1e-4)
    assert shared >= 1
    assert not {"frontier.csv", "points.csv", "ladder.csv"} & {f.name for f in q.OUT.iterdir()}
    html = pg.fragment(d)
    assert meta["label"] in html and "qn-data" in html
    for k in (*q.ACCOUNTS, *q.GOALS, *q.ROUTES):
        assert f'value="{k}"' in html
    for k in q.ROUTES:                                                 # the two routes are explained side by side
        assert f'id="qn-card-{k}"' in html and q.ROUTES[k]["how"][0] in html
    assert "{spread}" not in html and "{times}" not in html
    assert (q.REPORT).read_text().count("| Taxable brokerage |") == 2 * len(q.GOALS)
