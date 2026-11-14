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
        # the verdict reads the pre-registered family SPA ('*mean'), never the all-model row ('*')
        spa = df[(df.model == "*mean") & (df.horizon == h) & (df.asset == "pooled") & (df.metric == "spa_p_mse")
                 & (df.period == "full")].value
        if len(spa):
            assert d["mean"]["info"][str(h)]["spa_p"] == pytest.approx(float(spa.iloc[0]), rel=1e-4)
        det = sc.get("hist_mean", h, "pooled", "full", "none", "r2_detectable")
        assert d["mean"]["info"][str(h)]["r2_detectable"] == pytest.approx(det, rel=1e-4)
    # the verdict sentence quotes the best pooled R² at each horizon, formatted from the table
    best = {h: max(r["r2"] for r in d["mean"]["pooled"][str(h)] if r["role"] == "family" and r["r2"] is not None)
            for h in config.HORIZONS}
    assert all(r["role"] == "family" for r in d["mean"]["pooled"]["1"] if r["model"] in config.FROZEN_MEAN)
    assert all(not m.startswith("*") for m in d["mean"]["models"])
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
    for b in d["cov"]["benches"]:
        v = c[(c.model == config.PRIMARY_COV) & (c.horizon == 1) & (c.asset == "joint") & (c.period == "full")
              & (c.benchmark == b) & (c.metric == "diff_nll")].value
        assert prim["vs"][b]["diff"] == pytest.approx(float(v.iloc[0]), rel=1e-4)
    if "lw" in set(c.benchmark):   # the covariance verdict rests on the comparison with Ledoit-Wolf
        assert d["cov"]["static"] == "lw"
        cell = next(g for g in d["verdict"]["grid"] if g["key"] == "cov")["cells"]["1"]
        assert narrative.num(prim["vs"]["lw"]["diff"], 3, True) in cell["value"]
        assert "difference of score differences" not in " ".join(d["verdict"]["cov"])
    if (DEV / "strategy" / "metrics.parquet").exists():
        m = pd.read_parquet(DEV / "strategy" / "metrics.parquet")
        r = next(x for x in d["strategy"]["rows"] if x["family"] == "mean" and x["horizon"] == 1
                 and x["universe"] == "all" and x["rule"] == "mv" and x["mean_model"] == "ens_z")
        v = m[(m.family == "mean") & (m.horizon == 1) & (m.universe == "all") & (m.rule == "mv")
              & (m.mean_model == "ens_z") & (m.period == "full") & (m.basis == "net") & (m.metric == "d_cer")].value
        assert r["d_cer"] == pytest.approx(float(v.iloc[0]), rel=1e-4)
        # the verdict counts only the BH family; composite and robustness rows are reported individually
        fam = [x for x in d["strategy"]["rows"] if x["family"] == "mean" and x["universe"] == "all"
               and x["horizon"] == 1 and x["role"] == "family"]
        assert {x["mean_model"] for x in fam} <= set(config.FROZEN_MEAN)
        assert f"of {len(fam)} positive" in " ".join(d["verdict"]["strategy"])


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


@needs_dev
def test_kupiec_shown_at_every_horizon(dev_page):
    d = page_data(dev_page)
    assert isinstance(build.SHOW_KUPIEC_P_ABOVE_H1, bool)
    assert d["show_kupiec_long"] is build.SHOW_KUPIEC_P_ABOVE_H1
    if build.SHOW_KUPIEC_P_ABOVE_H1:
        assert any(r["kupiec_q99"] is not None for r in d["dist"]["tails"]["12"])
        assert "Kupiec p" in " ".join(d["verdict"]["dist"])


def test_set_labels_are_not_models():
    df = pd.DataFrame({"model": ["*", "*mean", "*dist", "a", "b"], "horizon": 1, "asset": "pooled",
                       "period": "full", "benchmark": "hist_mean", "metric": "x", "value": 1.0, "run": "r"})
    sc = ex.Scores(df)
    assert sc.models == ["a", "b"]
    assert sc.run_size("r", exclude="a") == 1


@needs_dev
def test_robustness_and_composite_are_not_candidates(dev_page):
    d = page_data(dev_page)
    roles = {r["model"]: r["role"] for r in d["mean"]["pooled"]["1"]}
    for m in getattr(config, "ROBUSTNESS", ()):
        if m in roles:
            assert roles[m] == "robust"
    if config.COMPOSITE in roles:
        assert roles[config.COMPOSITE] == "composite"
        assert "refitted every January" in d["glossary"]["models"][config.COMPOSITE][1]
    assert d["mean"]["family"] == [m for m in config.FROZEN_MEAN if m in roles]


# --------------------------------------------------------------------------- synthetic holdout root

def _relabel(df: pd.DataFrame) -> pd.DataFrame:
    """Dev rows as a full run would carry them: the same values under pre_holdout and holdout too."""
    full = df[df.period == "full"]
    return pd.concat([df, full.assign(period="pre_holdout"), full.assign(period="holdout")], ignore_index=True)


def _fake_full_root(tmp_path: Path, holdout_r2: float, spa_family: float, spa_all: float = 0.5,
                    strategy_q: float | None = None) -> Path:
    """A results root shaped like the final run, built from the dev results: every
    period-``full`` row copied to pre_holdout and holdout (adjusted tests included,
    as MT_PERIODS gives them), with chosen holdout values for ens_z and the SPA rows."""
    root = tmp_path / "results"
    (root / "scores").mkdir(parents=True)
    s = _relabel(pd.read_parquet(DEV / "scores" / f"{RUNS[0]}.parquet"))
    hold = s.period == "holdout"
    sel = hold & (s.model == "ens_z") & (s.asset == "pooled") & (s.benchmark == "hist_mean")
    s.loc[sel & (s.metric == "r2_oos"), "value"] = holdout_r2
    s.loc[sel & (s.metric == "dm_mse_p1"), "value"] = 0.001
    s.loc[hold & (s.model == "*mean") & (s.metric == "spa_p_mse"), "value"] = spa_family
    s.loc[hold & (s.model == "*") & (s.metric == "spa_p_mse"), "value"] = spa_all
    s.to_parquet(root / "scores" / "fake.parquet", index=False)
    shutil.copy(DEV / "scores" / f"{RUNS[0]}_cumdsse.parquet", root / "scores" / "fake_cumdsse.parquet")
    (root / "scores" / "fake.json").write_text(json.dumps({"models": ["ens_z"], "dev": False}))
    c = _relabel(pd.read_parquet(DEV / "scores" / f"{COV_RUN}.parquet"))
    c.to_parquet(root / "scores" / "cov_fake.parquet", index=False)
    if strategy_q is not None:
        (root / "strategy").mkdir()
        m = pd.read_parquet(DEV / "strategy" / "metrics.parquet")
        m = pd.concat([m, m[m.period == "full"].assign(period="holdout")], ignore_index=True)
        sel = ((m.period == "holdout") & (m.family == "mean") & (m.horizon == 1) & (m.universe == "all")
               & (m.rule == "mv") & (m.mean_model == "ens_z") & (m.basis == "net"))
        m.loc[sel & (m.metric == "d_cer"), "value"] = 0.03
        m.loc[sel & (m.metric == "d_cer_q_bh"), "value"] = strategy_q
        m.to_parquet(root / "strategy" / "metrics.parquet", index=False)
        shutil.copy(DEV / "strategy" / "meta.json", root / "strategy" / "meta.json")
    return root


@needs_dev
def test_holdout_is_headline_and_verdict_follows_data(tmp_path):
    root = _fake_full_root(tmp_path, holdout_r2=0.04, spa_family=0.001, strategy_q=0.01)
    out = tmp_path / "full.html"
    assert build.main(["--results-dir", str(root), "--scores-run", "fake", "--cov-scores-run", "cov_fake",
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
    # adjusted tests come from the holdout period itself, so no "whole sample" fallback note
    assert d["mean"]["spa_period"] == "holdout" and d["mean"]["q_period"] == "holdout"
    assert d["cov"]["q_period"] == "holdout" and d["cov"]["headline"] == "holdout"
    assert not any("whole sample only" in p for p in d["facts"]["points"])
    # strategy verdict uses the holdout period
    assert d["strategy"]["period"] == "holdout"
    assert next(g for g in d["verdict"]["grid"] if g["key"] == "strategy")["cells"]["1"]["status"] == "pass"
    # period labels say what the periods contain
    assert "2015" in d["periods"]["labels"]["pre_holdout"] and "straddle" in d["periods"]["labels"]["2010s"]
    assert d["now"]["h"]["1"] is None   # no forecasts in this root: empty section, not an error

    # same files, negative holdout R² and a family SPA that fails: the verdict turns, even when the
    # all-model SPA row says otherwise (it must never drive a verdict)
    root2 = _fake_full_root(tmp_path / "b", holdout_r2=-0.02, spa_family=0.6, spa_all=0.001)
    assert build.main(["--results-dir", str(root2), "--scores-run", "fake", "--no-covariance",
                       "--out", str(out)]) == 0
    d2 = page_data(out.read_text(encoding="utf-8"))
    assert d2["verdict"]["grid"][0]["cells"]["1"]["status"] != "pass"
    assert "No mean forecast beats" in d2["verdict"]["headline"]
    root3 = _fake_full_root(tmp_path / "c", holdout_r2=0.04, spa_family=0.6, spa_all=0.001)
    assert build.main(["--results-dir", str(root3), "--scores-run", "fake", "--no-covariance",
                       "--out", str(out)]) == 0
    d3 = page_data(out.read_text(encoding="utf-8"))
    assert d3["verdict"]["grid"][0]["cells"]["1"]["status"] == "weak"


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
