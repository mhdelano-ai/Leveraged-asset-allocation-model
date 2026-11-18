"""The one-page summary (dashboard/summary.html): its figures are recomputed here from the forecast and
result files, independently of ``assetforecast.dashboard.summary``; the committed page is the current build."""
from __future__ import annotations

import json
import math
import re

import numpy as np
import pandas as pd
import pytest

from assetforecast import paths
from assetforecast.dashboard import summary as sm

LATEST = paths.FORECASTS_OUT / "latest"
F = paths.RESULTS_FINAL
pytestmark = pytest.mark.skipif(not (LATEST / "frontier_points.csv").exists(), reason="forecast not built")


@pytest.fixture(scope="module")
def facts():
    return sm.facts()


@pytest.fixture(scope="module")
def page(facts):
    return sm.render(facts)


def test_units():
    assert sm.pct(0.08018) == "+8.0%" and sm.pct(-0.186) == "−18.6%" and sm.pct(0.0525, 2, sign=False) == "5.25%"
    assert sm.pct(0.0) == "0.0%" and sm.pct(-0.0004) == "0.0%"
    assert sm.simple(0.0) == 0.0 and math.isclose(sm.simple(math.log(1.25)), 0.25)


def test_asset_rows_are_the_published_forecast_in_simple_returns(facts):
    cov = pd.read_csv(LATEST / "cov_annual.csv", index_col=0)
    e = pd.read_csv(LATEST / "expected_returns.csv").set_index("asset")
    d = pd.read_csv(LATEST / "distribution_assets.csv").set_index("asset")
    rows = {r["id"]: r for r in facts["outlook"] if r["kind"] == "asset"}
    assert list(rows) == list(e.index) and len(rows) == 12
    for a, r in rows.items():
        assert r["avg"] == pytest.approx(math.exp(e.mean_12m[a] + 0.5 * cov.loc[a, a]) - 1, abs=2e-5)   # the files hold 5 decimals
        assert r["lo"] == pytest.approx(math.exp(d.q05[a]) - 1) and r["hi"] == pytest.approx(math.exp(d.q95[a]) - 1)
        assert r["mid_lo"] == pytest.approx(math.exp(d.q25[a]) - 1) and r["mid_hi"] == pytest.approx(math.exp(d.q75[a]) - 1)
        assert r["lo"] < r["mid_lo"] < r["mid_hi"] < r["hi"]


def test_portfolio_rows(facts):
    cov = pd.read_csv(LATEST / "cov_annual.csv", index_col=0)
    e = pd.read_csv(LATEST / "expected_returns.csv").set_index("asset")
    drift = e.mean_12m + 0.5 * pd.Series(np.diag(cov), index=cov.index)
    weights = json.loads((LATEST / "forecast.json").read_text())["portfolio_weights"]
    d = pd.read_csv(LATEST / "distribution_portfolios.csv")
    d = d[d["shape"] == "portfolios"].set_index("portfolio")
    rows = {r["id"]: r for r in facts["outlook"] if r["kind"] == "portfolio"}
    assert set(rows) == set(weights)
    for p, r in rows.items():
        w = pd.Series(weights[p])
        assert r["avg"] == pytest.approx(math.exp(float(w @ drift[w.index])) - 1, abs=5e-5)
        assert r["lo"] == pytest.approx(math.exp(d.q05[p]) - 1) and r["hi"] == pytest.approx(math.exp(d.q95[p]) - 1)
    assert rows["p6040"]["detail"] == "60% stocks, 40% bonds" and rows["p_equity"]["detail"] == "100% stocks"


def test_frontier(facts):
    b = facts["frontier"]
    wc = pd.read_csv(LATEST / "frontier.csv").query("curve == 'with_cash'").sort_values("i")
    assert len(b["curve"]) == len(wc)
    assert [c["avg"] for c in b["curve"]] == pytest.approx(list(np.expm1(wc.drift)))
    assert all(c["lo"] <= c["avg"] <= c["hi"] for c in b["curve"])
    assert max(c["invested"] for c in b["curve"]) <= 1 + 1e-6                   # nothing is borrowed
    origin = json.loads((LATEST / "forecast.json").read_text())["origin"]
    cash = pd.read_parquet(paths.PROCESSED / "cash_monthly.parquet")["cash"]
    assert facts["cash_rate"] == pytest.approx(12 * cash.loc[pd.Timestamp(origin)])
    ps = pd.read_csv(F / "portfolio_summary.csv")
    ratio = ps.query("model == 'combo_cov_main' and portfolio == 'gmv' and metric == 'var_ratio' and period == 'dev'")["mean"].item()
    assert b["swing_excess"] == pytest.approx(math.sqrt(ratio) - 1) and b["haircut_swing"] == pytest.approx(math.sqrt(1.5) - 1)
    pts = pd.read_csv(LATEST / "frontier_points.csv").set_index("point")
    for g in b["gaps"]:
        own, best = pts.loc[g["id"]], pts.loc["match_" + g["id"]]
        assert g["gap"] == pytest.approx(math.expm1(best.drift) - math.expm1(own.drift)) and g["gap"] > 0
        assert g["includes_zero"] == (best.gap_lo <= 0 <= best.gap_hi)


def test_recommended_mixes_are_rows_of_the_published_frontier(facts, page):
    mixes = facts["frontier"]["mixes"]
    lev = pd.read_csv(LATEST / "frontier.csv").query("curve == 'with_cash'").sort_values("i").reset_index(drop=True)
    e = pd.read_csv(LATEST / "expected_returns.csv").set_index("asset")
    z05 = pd.read_csv(LATEST / "shape.csv").set_index("tau").z_assets[0.05]
    assert [m["name"] for m in mixes] == list(sm.MIX_NAMES) and len(mixes) == 4
    top = lev.vol.iloc[-1]
    for k, m in enumerate(mixes, 1):
        row = lev.iloc[(lev.vol - k * top / 5).abs().idxmin()]                 # the grid point nearest k fifths of the way
        w = row[[c for c in lev.columns if c.startswith("w_")]].rename(lambda c: c[2:]).astype(float)
        assert m["vol"] == pytest.approx(row.vol) and m["avg"] == pytest.approx(math.expm1(row.drift))
        assert m["weights"] == pytest.approx(w.to_dict()) and sum(m["weights"].values()) == pytest.approx(m["invested"], abs=1e-4)
        assert sum(m["groups"].values()) == pytest.approx(m["invested"], abs=1e-4)
        assert m["cash"] == pytest.approx(max(1 - row.invested, 0)) and m["invested"] <= 1 + 1e-6
        center = float(w @ e.mean_12m[w.index])
        scale = math.sqrt(1.5 * row.vol ** 2 + float(w.abs() @ e.se_12m[w.index]) ** 2)
        assert m["bad"] == pytest.approx(math.expm1(center + scale * z05))
        assert m["bad"] < 0 < m["avg"]
        text = re.sub(r"\s+", " ", page)
        assert (f'<span class="mix-v">{sm.pct(m["avg"])}</span><span class="mix-v">{sm.pct(m["vol"], 1, sign=False)}</span>'
                f'<span class="mix-v">{sm.pct(m["bad"])}</span>') in text
        assert all(f"{round(100 * x)}% {n}" in text for n, x in m["holdings"])
    assert [m["vol"] for m in mixes] == sorted(m["vol"] for m in mixes)
    assert [m["bad"] for m in mixes] == sorted((m["bad"] for m in mixes), reverse=True)   # more swing, worse bad year
    assert sm._funding({"invested": 0.741}) == "74% invested, 26% cash" and sm._funding({"invested": 1.0}) == "fully invested"


def test_trust(facts):
    t = facts["trust"]
    acc = pd.read_csv(F / "accuracy.csv").query("sample == 'last30y' and asset == 'pooled'").iloc[0]
    assert t["below"] == pytest.approx(-acc.two_thirds_lo_pct) and t["above"] == pytest.approx(acc.two_thirds_hi_pct)
    h = t["history"]
    assert h["forecast"][-1][0] > h["actual"][-1][0]              # the forecast runs past the last finished year
    assert len(h["peaks"]) == 2 and h["peaks"][0]["t"] < h["peaks"][1]["t"]
    assert max(v for _, v in h["actual"]) == max(p["actual"] for p in h["peaks"])
    assert sm._window_label("2008-09") == "2008–09" and sm._window_label("2019-12") == "2020"


def test_sentences_follow_the_status():
    assert sm._tested("pass", 7.78, "2014–2025")[:2] == ("chip-on", "Tested")
    cls, chip, text = sm._tested("weak", -2.04, "2014–2025")
    assert (cls, chip) == ("chip-off", "Unproven") and "2.0% behind" in text
    assert "ahead of" in sm._tested("weak", 1.17, "2014–2025")[2]


def test_page_shows_the_facts(facts, page):
    text = re.sub(r"\s+", " ", page)
    for r in facts["outlook"]:
        assert f'{sm.pct(r["avg"])}</span><span class="rng-s">{sm.pct(r["lo"])} to {sm.pct(r["hi"])}' in text
    t, m = facts["trust"], facts["machine"]
    for s in (f'Cash itself yields about {sm.pct(facts["cash_rate"], 1, sign=False)} today', facts["window"],
              "None of them borrows.", f'{sm.num(t["vol_skill"])}% lower than',
              f'For {t["n_zero"]} of the {t["n_assets"]} assets', f'>{m["n_models"]}</b>', f'>{m["since"]}</b>'):
        assert s in text, s


def test_page_is_one_self_contained_printable_sheet(page):
    assert "<script" not in page                                            # nothing runs, nothing is fetched as data
    assert re.findall(r'(?:src|href)="([^"]+)"', page) == [sm.FONTS]         # the only outside resource: the typeface
    assert "@media print" in page and "print-color-adjust: exact" in page
    assert "@media screen and (max-width" in page                           # the phone layout must not apply to paper
    for token in sm.LIGHT:
        assert token in sm.DARK and page.count(f"--{token}:") == 4          # light, dark twice, print


def test_committed_page_is_current(page):
    if not sm.OUT.exists():
        pytest.skip("dashboard/summary.html not built")
    assert sm.OUT.read_text(encoding="utf-8") == page, "run python -m assetforecast.dashboard.summary"
