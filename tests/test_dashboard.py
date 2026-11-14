"""Dashboard generator: builds on the dev results, numbers match the score
tables, missing optional inputs degrade gracefully, dev/holdout labelling."""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from assetpred import config
from assetpred.dashboard import build, narrative
from assetpred.dashboard import extract as ex

DEV = config.RESULTS / "dev"
RUNS, COV_RUN = build.dev_defaults(DEV)
needs_dev = pytest.mark.skipif(
    not all((DEV / "scores" / f"{r}.parquet").exists() for r in [*RUNS, COV_RUN]),
    reason="development results not present")


def merged_scores(d):
    df, _, _ = ex.load_score_runs(DEV, d["score_runs"])
    return ex.Scores(df), df


def page_data(page: str) -> dict:
    m = re.search(r'<script type="application/json" id="dashboard-data">(.*?)</script>', page, re.S)
    assert m, "embedded data block missing"
    return json.loads(m.group(1))


def text_outside_data(page: str) -> str:
    return re.sub(r'<script type="application/json" id="dashboard-data">.*?</script>', "", page, flags=re.S)


@pytest.fixture(scope="module")
def dev_page(tmp_path_factory):
    out = tmp_path_factory.mktemp("dash") / "dashboard.html"
    rc = build.main(["--dev", "--out", str(out)])
    assert rc == 0
    return out.read_text(encoding="utf-8")


@needs_dev
def test_builds_on_dev_under_size_limit_without_placeholders(dev_page):
    n = len(dev_page.encode("utf-8"))
    assert n < 16 * 1024 * 1024
    assert n < build.WARN_BYTES, f"page unexpectedly large: {n / 1e6:.1f} MB"
    body = text_outside_data(dev_page)
    assert "{{" not in body and "}}" not in body
    assert re.search(r"<title>[^<]{3,60}</title>", dev_page)
    assert 2 <= len(re.search(r"<title>([^<]*)</title>", dev_page).group(1).split()) <= 4
    # artifact constraints: only allowed external hosts, dark-mode tokens defined both ways
    for src in re.findall(r'<script[^>]+src="([^"]+)"', dev_page):
        assert src.startswith(("https://cdnjs.cloudflare.com/", "https://cdn.jsdelivr.net/npm/", "https://unpkg.com/"))
    for href in re.findall(r'<link[^>]+rel="stylesheet"[^>]+href="([^"]+)"', dev_page):
        assert href.startswith("https://fonts.googleapis.com/")
    assert ':root:not([data-theme="light"])' in dev_page and ':root[data-theme="dark"]' in dev_page
    d = page_data(dev_page)
    assert d["dev"] is True


@needs_dev
def test_dev_mode_is_labelled(dev_page):
    body = text_outside_data(dev_page)
    assert "Development results only" in body
    assert "has not been scored" in body
    d = page_data(dev_page)
    assert d["periods"]["headline"] == "full" and not d["periods"]["holdout_present"]


@needs_dev
def test_mean_numbers_match_scores(dev_page):
    d = page_data(dev_page)
    assert d["score_runs"] == RUNS
    sc, df = merged_scores(d)
    for h in config.HORIZONS:
        for r in d["mean"]["pooled"][str(h)]:
            v = sc.get(r["model"], h, "pooled", "full", "hist_mean", "r2_oos")
            if np.isfinite(v):
                assert r["r2"] == pytest.approx(v, rel=1e-4, abs=1e-9)
        spa = df[(df.model == "*") & (df.horizon == h) & (df.asset == "pooled") & (df.metric == "spa_p_mse")
                 & (df.run == d["mean"]["info"][str(h)]["spa_run"])].value
        assert d["mean"]["info"][str(h)]["spa_p"] == pytest.approx(float(spa.iloc[0]), rel=1e-4)
        det = sc.get("hist_mean", h, "pooled", "full", "none", "r2_detectable")
        assert d["mean"]["info"][str(h)]["r2_detectable"] == pytest.approx(det, rel=1e-4)
    # the verdict sentence quotes the best pooled R² at each horizon, formatted from the table
    best = {h: max(r["r2"] for r in d["mean"]["pooled"][str(h)] if not r["ref"] and r["r2"] is not None)
            for h in config.HORIZONS}
    text = " ".join(d["verdict"]["mean"])
    for h, v in best.items():
        assert narrative.pct(v) in text


@needs_dev
def test_distribution_covariance_strategy_numbers_match(dev_page):
    d = page_data(dev_page)
    sc, _ = merged_scores(d)
    for r in d["dist"]["pooled"]["1"]:
        v = sc.get(r["model"], 1, "pooled", "full", "vol_ewma", "crps_skill")
        if np.isfinite(v):
            assert r["skill_vol"] == pytest.approx(v, rel=1e-4)
    c = pd.read_parquet(DEV / "scores" / f"{d['cov_run']}.parquet")
    prim = next(r for r in d["cov"]["joint"]["1"] if r["model"] == config.PRIMARY_COV)
    v = c[(c.model == config.PRIMARY_COV) & (c.horizon == 1) & (c.asset == "joint") & (c.period == "full")
          & (c.metric == "diff_nll")].value
    assert prim["diff"] == pytest.approx(float(v.iloc[0]), rel=1e-4)
    if (DEV / "strategy" / "metrics.parquet").exists():
        m = pd.read_parquet(DEV / "strategy" / "metrics.parquet")
        r = next(x for x in d["strategy"]["rows"] if x["family"] == "mean" and x["horizon"] == 1
                 and x["universe"] == "all" and x["rule"] == "mv" and x["mean_model"] == "ens_z")
        v = m[(m.family == "mean") & (m.horizon == 1) & (m.universe == "all") & (m.rule == "mv")
              & (m.mean_model == "ens_z") & (m.period == "full") & (m.basis == "net") & (m.metric == "d_cer")].value
        assert r["d_cer"] == pytest.approx(float(v.iloc[0]), rel=1e-4)


@needs_dev
def test_degrades_without_strategy_and_covariance(tmp_path):
    out = tmp_path / "d.html"
    assert build.main(["--dev", "--no-strategy", "--no-covariance", "--out", str(out)]) == 0
    page = out.read_text(encoding="utf-8")
    d = page_data(page)
    assert d["strategy"] is None and d["cov"] is None
    assert any("not shown" in t for t in d["verdict"]["strategy"])
    assert any("not shown" in t for t in d["verdict"]["cov"])
    assert "{{" not in text_outside_data(page)
    # an empty strategy folder is the same as no strategy test
    empty = tmp_path / "nostrategy"
    empty.mkdir()
    assert build.main(["--dev", "--strategy-dir", str(empty), "--out", str(out)]) == 0
    assert page_data(out.read_text(encoding="utf-8"))["strategy"] is None


@needs_dev
def test_unknown_scores_run_names_the_alternatives(tmp_path):
    with pytest.raises(SystemExit) as e:
        build.main(["--dev", "--scores-run", "no_such_run", "--out", str(tmp_path / "x.html")])
    assert "no_such_run" in str(e.value) and RUNS[0] in str(e.value)


def test_kupiec_switch_defaults_off():
    assert build.SHOW_KUPIEC_P_ABOVE_H1 is False


# --------------------------------------------------------------------------- synthetic holdout root

def _fake_full_root(tmp_path: Path, holdout_r2: float, spa: float) -> Path:
    """A results root shaped like the final run: the dev scores relabelled with
    pre_holdout and holdout periods, with chosen holdout values for ens_z."""
    root = tmp_path / "results"
    (root / "scores").mkdir(parents=True)
    s = pd.read_parquet(DEV / "scores" / f"{RUNS[0]}.parquet")
    full = s[s.period == "full"]
    pre = full.assign(period="pre_holdout")
    hold = full.assign(period="holdout").copy()
    sel = (hold.model == "ens_z") & (hold.asset == "pooled") & (hold.benchmark == "hist_mean")
    hold.loc[sel & (hold.metric == "r2_oos"), "value"] = holdout_r2
    hold.loc[sel & (hold.metric == "dm_mse_p1"), "value"] = 0.001
    out = pd.concat([s, pre, hold], ignore_index=True)
    out.loc[(out.model == "*") & (out.metric == "spa_p_mse"), "value"] = spa
    out.to_parquet(root / "scores" / "fake.parquet", index=False)
    shutil.copy(DEV / "scores" / f"{RUNS[0]}_cumdsse.parquet", root / "scores" / "fake_cumdsse.parquet")
    (root / "scores" / "fake.json").write_text(json.dumps({"models": ["ens_z"], "dev": False}))
    return root


@needs_dev
def test_holdout_is_headline_and_verdict_follows_data(tmp_path):
    root = _fake_full_root(tmp_path, holdout_r2=0.04, spa=0.001)
    out = tmp_path / "full.html"
    assert build.main(["--results-dir", str(root), "--scores-run", "fake", "--no-covariance",
                       "--out", str(out)]) == 0
    page = out.read_text(encoding="utf-8")
    d = page_data(page)
    assert d["periods"]["headline"] == "holdout" and d["periods"]["context"] == "pre_holdout"
    body = text_outside_data(page)
    assert "Holdout results" in body and "Development results only" not in body
    ens = next(r for r in d["mean"]["pooled"]["1"] if r["model"] == "ens_z")
    assert ens["r2"] == pytest.approx(0.04)
    assert d["verdict"]["grid"][0]["cells"]["1"]["status"] == "pass"
    assert "Mean forecasts beat the historical mean" in d["verdict"]["headline"]
    # no forecasts or strategy in this root: those sections are empty, not errors
    assert d["now"]["h"]["1"] is None and d["strategy"] is None

    # same files with a negative holdout R²: the verdict turns
    root2 = _fake_full_root(tmp_path / "b", holdout_r2=-0.02, spa=0.6)
    assert build.main(["--results-dir", str(root2), "--scores-run", "fake", "--no-covariance",
                       "--out", str(out)]) == 0
    d2 = page_data(out.read_text(encoding="utf-8"))
    assert d2["verdict"]["grid"][0]["cells"]["1"]["status"] != "pass"
    assert "No mean forecast beats" in d2["verdict"]["headline"]


def test_full_mode_requires_named_runs(tmp_path):
    with pytest.raises(SystemExit):
        build.main(["--results-dir", str(tmp_path), "--out", str(tmp_path / "x.html")])


@needs_dev
def test_now_section_falls_back_when_composite_missing(tmp_path):
    from assetpred.dashboard import sections
    mean_m, dist_m = config.COMPOSITE.split("__", 1)
    root = tmp_path / "r"
    for m in (dist_m, mean_m, "hist_mean"):
        shutil.copytree(DEV / "forecasts" / m, root / "forecasts" / m)
    now = sections.now_section(root, None)
    x = now["h"][1]
    assert x["source"] == dist_m and config.COMPOSITE in x["note"]
    assert all(r["model_mean"] is not None and r["hist_mean"] is not None for r in x["rows"])
    f = pd.read_parquet(DEV / "forecasts" / dist_m / "h01.parquet")
    last = f[f.date == f.date.max()].set_index("asset")
    r0 = x["rows"][0]
    assert r0["q50"] == pytest.approx(float(last.loc[r0["asset"], "q50"]), rel=1e-4)


def test_render_escapes_and_fills_every_placeholder():
    tpl = build.TEMPLATE.read_text()
    keys = set(re.findall(r"\{\{([A-Z_0-9]+)\}\}", tpl))
    assert keys <= {"TITLE", "H1", "DESCRIPTION", "MODE_CLASS", "MODE_TITLE", "MODE_TEXT", "HEADLINE",
                    "HEADLINE_PERIOD", "FOOTER", "DATA"}
    data = {"dev": True, "periods": {"holdout_present": False, "headline": "full", "labels": {"full": "x"},
                                     "span": {}},
            "holdout_start": "2015-01", "verdict": {"headline": "<b>&</b>"}, "generated": "now", "root": "r",
            "score_runs": ["a"], "cov_run": None, "x": float("nan"), "y": np.float64(1.5)}
    page = build.render(data)
    assert "&lt;b&gt;&amp;&lt;/b&gt;" in page
    assert "{{" not in text_outside_data(page)
    d = page_data(page)
    assert d["x"] is None and d["y"] == 1.5
