"""The questionnaire: the stock share, the fund arithmetic, and that the committed tables and page agree."""
import numpy as np
import pandas as pd
import pytest

from assetforecast import questionnaire as q

WS = pd.Series({"ust_int": 0.5, "gold": 0.2, "mf": 0.3})


def _curve(falls):
    n = len(falls)
    s = np.linspace(0.0, 1.0, n)
    return pd.DataFrame({"i": range(n), "stocks": s, "sleeve": 1.0 - s, "worst_fall_all": falls})


def test_decide_takes_the_largest_stock_share_within_the_budget():
    cv = _curve([-0.03, -0.08, -0.15, -0.30, -0.50])
    assert q.decide("near", cv)["cash"] == 1.0                                  # no budget: T-bills
    d = q.decide("medium", cv)                                                  # 20%: the third point
    assert (d["i"], d["cash"]) == (2, 0.0) and d["stocks"] == pytest.approx(0.5) and d["sleeve"] == pytest.approx(0.5)
    assert q.decide("income", cv)["i"] == 3 and q.decide("growth", cv)["i"] == 4
    assert q.decide("medium", _curve([-0.03, -0.25, -0.15, -0.30, -0.50]))["i"] == 2   # the largest, not the first break
    assert q.decide("medium", _curve([-0.25, -0.30, -0.35, -0.40, -0.50]))["cash"] == 1.0


def test_path_is_every_stock_share_with_the_sleeve_in_the_rest():
    pts = q.path()
    assert [s for s, _ in pts] == pytest.approx(list(np.arange(0, 1.0001, q.STEP)))
    assert all(s + d == pytest.approx(1.0) for s, d in pts)                    # nothing borrowed, nothing in cash


def test_fund_plan_holds_the_capital_in_plain_funds():
    veh = q.vehicles()
    d = {"stocks": 0.6, "sleeve": 0.4, "cash": 0.0}
    rows = q.fund_plan(d, WS, veh)
    assert sum(r["weight"] for r in rows) == pytest.approx(1.0)
    for a in q.SLEEVE:                                                         # every diversifier, from its cheapest fund
        r = next(r for r in rows if r["role"] == a)
        assert r["weight"] == pytest.approx(0.4 * WS[a]) and r["symbol"] == q._fund(veh, a)["symbol"]
    assert {r["kind"] for r in veh["vehicles"]} == {"outright"}
    assert [r["role"] for r in q.fund_plan({"stocks": 0, "sleeve": 0, "cash": 1.0}, WS, veh)] == ["cash"]
    assert q._fees(rows) == pytest.approx(sum(0.4 * WS[a] * q._fund(veh, a)["fee"] for a in q.SLEEVE))


def test_mix_stats_never_borrows():
    idx = pd.date_range("2000-01-31", periods=120, freq="ME")
    rng = np.random.default_rng(1)
    st, sl = pd.Series(rng.normal(0.005, 0.04, 120), index=idx), pd.Series(rng.normal(0.003, 0.02, 120), index=idx)
    cash = pd.Series(0.002, index=idx)
    half, whole = q.mix_stats(st, sl, cash, 0.3, 0.2), q.mix_stats(st, sl, cash, 0.6, 0.4)
    assert half["vol_all"] == pytest.approx(whole["vol_all"] / 2, rel=0.02)   # half the mix, the rest in T-bills
    with pytest.raises(ValueError, match="borrows"):
        q.mix_stats(st, sl, cash, 1.0, 0.5)


def test_committed_tables_and_page_agree():
    from assetforecast import allocation as al
    from assetforecast import questionnaire_page as pg
    if not (q.OUT / "answers.csv").exists():
        pytest.skip("results/questionnaire not written")
    d = pg.load()
    ans, cv, meta = d["answers"], d["curve"], d["meta"]
    assert list(ans["goal"]) == list(q.GOALS) and sum(meta["sleeve_weights"].values()) == pytest.approx(1.0)
    assert cv["stocks"].to_numpy() == pytest.approx(np.arange(0, 1.0001, q.STEP))     # every stock share, in order
    assert (cv["stocks"] + cv["sleeve"]).to_numpy() == pytest.approx(1.0)
    assert cv.loc[cv["vol"].idxmin(), "stocks"] == pytest.approx(meta["least_risk_stocks"])
    for _, r in ans.iterrows():
        limit = q.GOALS[r["goal"]]["loss_limit"]
        if limit is None:
            assert r["cash"] == 1.0 and r["headline"] == "T-bills"
            continue
        best = cv[cv["worst_fall_all"] >= -limit]["i"].max()                    # the largest stock share within the budget
        row = cv[cv["i"] == r["i"]].iloc[0]
        assert r["i"] == best and r["worst_fall_all"] >= -limit
        for c in ("stocks", "sleeve", "vol", "growth", "growth_lo", "growth_hi", "worst_fall_all", "growth_all"):
            assert r[c] == pytest.approx(row[c])
        assert sum(r[f"w_{a}"] for a in q.SLEEVE) == pytest.approx(r["sleeve"], abs=1e-5)
        assert r["growth_all_net"] == pytest.approx(r["growth_all"] - r["fund_fees"], abs=1e-6)
        f = d["funds"][d["funds"]["goal"] == r["goal"]]
        assert f["weight"].sum() == pytest.approx(1.0, abs=1e-4)
    # the path is the allocation study's replay and forecast risk: its menu is reproduced where they share a mix
    menu = pd.read_csv(al.OUT / "menu.csv")
    menu = menu[menu["sleeve_rule"] == q.SLEEVE_RULE]
    shared = 0
    for _, m in menu.iterrows():
        row = cv[np.isclose(cv["stocks"], m["stocks"]) & np.isclose(cv["sleeve"], m["sleeve"])]
        for _, p in row.iterrows():
            shared += 1
            for a, b in (("worst_fall_all", "worst_fall_all"), ("growth_all", "growth_all"), ("vol_all", "vol_all"), ("vol", "vol_fcst")):
                assert p[a] == pytest.approx(m[b], rel=1e-4)
    assert shared == len(menu)
    assert not {"frontier.csv", "points.csv", "ladder.csv"} & {f.name for f in q.OUT.iterdir()}
    html = pg.fragment(d)
    assert meta["label"] in html and "qn-data" in html
    for k in (*q.ACCOUNTS, *q.GOALS):
        assert f'value="{k}"' in html
    report = q.REPORT.read_text()
    assert all(f"**{a['name']}.**" in report for a in q.ACCOUNTS.values())       # every account's tax notes
    assert not any(w in report.lower() for w in ("margin loan", "stacked", "borrowed on"))
