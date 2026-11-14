"""Build the static results dashboard: one self-contained HTML file.

    python -m assetpred.dashboard.build --dev
    python -m assetpred.dashboard.build --scores-run frozen --cov-scores-run cov_frozen

Reads the results root (``results/``, or ``results/dev/`` with ``--dev``):
marginal scores (``scores/<run>.parquet`` and ``<run>_cumdsse.parquet``),
covariance scores (``scores/<cov run>.parquet`` and ``_per_date``), the
strategy test (``strategy/``), the composite and baseline forecasts at their
last origin (``forecasts/``), the primary covariance forecast
(``covariances/``), the trial logs, and the data-provenance files under
``data/processed``. Writes ``<root>/dashboard.html``.

The page is driven by the files: where a ``holdout`` score period exists it is
the headline and ``pre_holdout`` is context; otherwise (development results)
the page says so prominently. The template is ``template.html`` next to this
file; section data is assembled in ``sections.py`` and the verdict sentences in
``narrative.py``.

Development defaults (``--dev`` without ``--scores-run``): the consolidated
runs ``frozen`` and ``cov_frozen`` written by ``assetpred.pipeline --dev`` when
they exist, else the earlier dev runs that together cover the frozen models
(``mean_frozen dist_final baselines`` and ``cov_dev_frozen``). Without
``--dev`` the runs must be named.
"""
from __future__ import annotations

import argparse
import html
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .. import config
from . import extract as ex
from . import glossary as gl
from . import narrative, sections

TEMPLATE = Path(__file__).with_name("template.html")
TITLE = "Asset Return Predictability"
MAX_BYTES = 16 * 1024 * 1024
WARN_BYTES = 5 * 1024 * 1024

# The pooled tail-hit (Kupiec) p-values at horizons above one month are under
# review and may be unreliable. Flip to True to show them on the page.
SHOW_KUPIEC_P_ABOVE_H1 = False

# With --dev and no run named: the consolidated run written by `python -m assetpred.pipeline --dev` when it
# exists, else the earlier development runs that together cover the frozen models.
DEV_SCORE_RUNS_CONSOLIDATED = ("frozen",)
DEV_COV_RUN_CONSOLIDATED = "cov_frozen"
DEV_SCORE_RUNS = ("mean_frozen", "dist_final", "baselines")
DEV_COV_RUN = "cov_dev_frozen"


def dev_defaults(root: Path) -> tuple[list[str], str]:
    sdir = root / "scores"
    if all((sdir / f"{r}.parquet").exists() for r in DEV_SCORE_RUNS_CONSOLIDATED):
        runs = list(DEV_SCORE_RUNS_CONSOLIDATED)
    else:
        runs = list(DEV_SCORE_RUNS)
    cov = DEV_COV_RUN_CONSOLIDATED if (sdir / f"{DEV_COV_RUN_CONSOLIDATED}.parquet").exists() else DEV_COV_RUN
    return runs, cov


# --------------------------------------------------------------------------- context

def build_context(root: Path, dev: bool, score_runs: list[str], cov_run: str | None,
                  strategy_dir: Path | None) -> dict:
    scores_df, cum, metas = ex.load_score_runs(root, score_runs)
    sc = ex.Scores(scores_df)
    dev = dev or any(m.get("dev") for m in metas.values())
    cs, per_date, cov_meta = ex.load_cov_run(root, cov_run)
    if cov_meta is not None and cov_run:
        cov_meta = dict(cov_meta, run=cov_run)
    st = ex.load_strategy(strategy_dir)

    P = sections.periods(sc, cum, dev)
    ctx = {
        "mean": sections.mean_section(sc, cum, P),
        "dist": sections.dist_section(sc, P),
        "cov": sections.cov_section(cs, per_date, P),
        "strategy": sections.strategy_section(st),
    }
    now = sections.now_section(root, sc)
    method = sections.method_section(root, metas, cov_meta or {}, dev)
    V = narrative.verdict(ctx)

    mean_m, dist_m = gl.composite_parts()
    model_ids = set(sc.models) | set(config.FROZEN_MARGINAL) | set(config.FROZEN_COV) | {"sample"}
    if ctx["cov"]:
        model_ids |= set(ctx["cov"]["models"])
    last_origin = max((v["origin"] for v in now["h"].values() if v), default=None)
    data = {
        "alpha": narrative.ALPHA, "horizons": list(config.HORIZONS), "dev": dev,
        "show_kupiec_long": SHOW_KUPIEC_P_ABOVE_H1,
        "composite": config.COMPOSITE, "mean_model": mean_m, "dist_model": dist_m,
        "primary_cov": config.PRIMARY_COV, "assets": list(config.ASSETS),
        "holdout_start": pd.Timestamp(config.HOLDOUT_START).strftime("%Y-%m"),
        "dev_end": pd.Timestamp(config.DEV_END).strftime("%Y-%m"), "last_origin": last_origin,
        "periods": P, "verdict": V, "now": now, **ctx, "method": method,
        "glossary": {"models": {m: [gl.label(m), gl.describe(m)] for m in sorted(model_ids)},
                     "short_asset": gl.SHORT_ASSET, "assets": dict(config.ASSETS),
                     "rules": gl.STRATEGY_RULES, "terms": gl.TERMS},
        "frozen_models": _frozen_models(sc, ctx),
        "frozen_counts": {"mean": len(config.FROZEN_MEAN), "dist": len(config.FROZEN_DIST),
                          "cov": len(config.FROZEN_COV)},
        "design": _design_text(dev, P),
        "limits": _limits(ctx, method, st),
        "facts": _facts(ctx, method, P, dev),
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "root": _rel(root),
        "score_runs": score_runs, "cov_run": cov_run,
    }
    return data


def _rel(p: Path) -> str:
    try:
        return str(Path(p).resolve().relative_to(config.ROOT))
    except ValueError:
        return str(p)


def _frozen_models(sc, ctx) -> list[dict]:
    rows = []
    for m in config.BASELINES:
        rows.append({"m": m, "role": "Baseline", "d": gl.describe(m)})
    for m in config.FROZEN_MEAN:
        rows.append({"m": m, "role": "Mean model", "d": gl.describe(m)})
    for m in config.FROZEN_DIST:
        rows.append({"m": m, "role": "Distribution model", "d": gl.describe(m)})
    rows.append({"m": config.COMPOSITE, "role": "Composite (the forecast shown)", "d": gl.describe(config.COMPOSITE)})
    for m in config.FROZEN_COV:
        rows.append({"m": m, "role": "Covariance model" + (" (primary)" if m == config.PRIMARY_COV else ""),
                     "d": gl.describe(m)})
    return rows


def _design_text(dev: bool, P: dict) -> list[str]:
    de = pd.Timestamp(config.DEV_END).strftime("%B %Y")
    hs = pd.Timestamp(config.HOLDOUT_START).strftime("%B %Y")
    out = [
        f"Every forecast is made walk-forward: at each month-end from {pd.Timestamp(config.OOS_START):%B %Y} the "
        f"models are fitted on data available at that date only (expanding window, refitted every "
        f"{config.REFIT_EVERY} months, at least {config.MIN_ASSET_HISTORY} months of an asset's history before it "
        f"is forecast) and predict the excess log return over the next 1, 3 and 12 months. Predictors are lagged to "
        f"the date they were published.",
        f"The models were developed and chosen looking only at outcomes realized by {de}. The model lists were then "
        f"frozen in the code. Forecasts made from {hs} on are the holdout: they are scored once, after the freeze, and "
        f"are the evidence that counts. The benchmark throughout is each asset's expanding historical mean.",
        "All significance statements use a 5% threshold. Overlapping multi-month outcomes are handled with "
        "autocorrelation-robust standard errors; the number of independent observations is the number of months "
        "divided by the horizon.",
    ]
    if dev or not P["holdout_present"]:
        out.append("This build contains development results only: the models were selected on these same outcomes, "
                   "so the figures are an optimistic guide to what the holdout will show.")
    return out


def _facts(ctx, method, P, dev) -> dict:
    tiles, points = [], []
    mean = ctx.get("mean")
    if mean:
        ni = [mean["info"][h]["n_indep"] for h in config.HORIZONS]
        tiles.append({"v": " / ".join("n/a" if v is None else f"{v:.0f}" for v in ni),
                      "l": "independent observations at 1, 3 and 12 months (pooled)"})
        dets = [mean["info"][h]["r2_detectable"] for h in config.HORIZONS]
        tiles.append({"v": " / ".join(narrative.pct(d, 1, False) for d in dets),
                      "l": "smallest out-of-sample R² the sample could detect, 1, 3, 12 months"})
        n12 = mean["info"][12]["n_indep"] if 12 in mean["info"] else None
        if n12 is not None:
            points.append(f"The 12-month results rest on about {n12:.0f} independent observations: a model would need an "
                          f"R² near {narrative.pct(mean['info'][12]['r2_detectable'], 0, False)} to stand out from "
                          f"luck, so 12-month verdicts in either direction are weak.")
    tri = method["trials"]
    n_tr = sum(t["n"] for t in tri.values() if t)
    n_frozen = len(config.FROZEN_MEAN) + len(config.FROZEN_DIST) + len(config.FROZEN_COV) + 1
    tiles.append({"v": f"{n_tr}", "l": f"configurations logged during development; {n_frozen} models frozen for the "
                                       f"holdout (plus {len(config.BASELINES)} baselines)"})
    if P["holdout_present"]:
        points.insert(0, "Headline figures are for the holdout, forecasts made after the model lists were frozen. "
                         "Development-period figures are shown alongside as context.")
    else:
        points.insert(0, "These are development results: the models were chosen on these same outcomes, so every "
                         "figure is likely to flatter them. The holdout has not been scored in this build.")
    points.append("Adjusted tests (SPA, model confidence sets, Benjamini-Hochberg q-values) account for the search "
                  "over the frozen models and assets, not for every configuration tried during development.")
    if not SHOW_KUPIEC_P_ABOVE_H1:
        points.append("Tail hit rates are shown at every horizon, but the pooled Kupiec test p-values above one month "
                      "are withheld while their reliability is reviewed.")
    if mean and mean["q_period"] != P["headline"]:
        points.append("The results files carry Benjamini-Hochberg q-values, SPA and model-confidence-set results for the "
                      "whole sample only; where the headline is the holdout, those adjusted figures mix holdout and "
                      "development forecasts.")
    return {"tiles": tiles, "points": points}


def _limits(ctx, method, st) -> list[str]:
    out = []
    assets = {a["asset"]: a for a in method["data"].get("assets", [])}
    diag = (st or {}).get("meta", {}).get("diagnostics", {}) if st else {}
    entry = diag.get("asset_entry", {})
    mean = ctx.get("mean")
    if "gold" in assets:
        s = f"Gold returns start in {assets['gold']['start']} (an ETF price series)"
        if entry.get("gold"):
            s += f"; after the {config.MIN_ASSET_HISTORY}-month minimum history it enters the forecasts only from {entry['gold'][:7]}"
        if mean:
            n1 = mean["grid"][1]["detectable"].get("gold") if 1 in mean["grid"] else None
            if n1 is not None:
                s += f", so gold's own tests have little power (detectable 1-month R² ≈ {narrative.pct(n1, 1, False)})"
        out.append(s + ".")
    out.append("Investment-grade credit is a long-duration corporate bond fund (about 13 years' duration), so it "
               "behaves partly like long Treasuries rather than like a broad credit index.")
    ac = diag.get("ac1_monthly_excess", {})
    if "hy_credit" in ac:
        others = sorted(v for k, v in ac.items() if k != "hy_credit")
        med = others[len(others) // 2] if others else None
        out.append(f"High yield is a mutual-fund NAV whose prices are smoothed: its monthly excess returns have "
                   f"first-order autocorrelation {ac['hy_credit']:.2f} against a median of {med:.2f} for the other "
                   f"assets, so its 1-month predictability is partly stale pricing rather than a tradable signal. The "
                   f"portfolio test is also reported excluding it.")
    else:
        out.append("High yield is a mutual-fund NAV whose prices are smoothed, so its 1-month predictability is "
                   "suspect.")
    spliced = [config.ASSETS[a] for a, v in assets.items() if len(v["segments"]) > 1]
    if spliced:
        out.append("Spliced series (several sources joined over time): " + ", ".join(spliced) + ". The splices are "
                   "listed in the table below; Treasuries and TIPS are synthetic bonds priced off fitted Federal "
                   "Reserve yield curves.")
    if mean and 12 in mean["info"]:
        out.append(f"12-month results rest on about {mean['info'][12]['n_indep']:.0f} independent observations.")
    return out


# --------------------------------------------------------------------------- render

def _sanitize(o):
    if isinstance(o, float):
        return None if not math.isfinite(o) else o
    if isinstance(o, dict):
        return {str(k): _sanitize(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_sanitize(v) for v in o]
    if hasattr(o, "item"):  # numpy scalar
        return _sanitize(o.item())
    return o


def render(data: dict, template: str | None = None) -> str:
    tpl = template if template is not None else TEMPLATE.read_text()
    P = data["periods"]
    if data["dev"] or not P["holdout_present"]:
        mode_class = "dev"
        mode_title = "Development results only"
        spans = P.get("span", {})
        rng = f" (forecast origins {spans[1][0]} to {spans[1][1]} at 1 month)" if 1 in spans else ""
        mode_text = (f"Every number here comes from the period used to build and choose the models{rng}. The "
                     f"holdout, forecasts made from {data['holdout_start']} on, has not been scored in this build. "
                     f"Expect these figures to flatter the models.")
    else:
        mode_class = "final"
        mode_title = "Holdout results"
        mode_text = (f"Headline figures are for forecasts made from {data['holdout_start']} on, scored once after the "
                     f"model lists were frozen. Development-period figures appear as context.")
    blob = json.dumps(_sanitize(data), separators=(",", ":"), allow_nan=False, default=str)
    blob = blob.replace("</", "<\\/").replace("<!--", "<\\!--")
    subs = {
        "TITLE": TITLE,
        "H1": "Can machine learning predict asset-class returns?",
        "DESCRIPTION": "Out-of-sample tests of mean, distribution and covariance forecasts for eleven asset classes.",
        "MODE_CLASS": mode_class, "MODE_TITLE": mode_title, "MODE_TEXT": mode_text,
        "HEADLINE": data["verdict"]["headline"],
        "HEADLINE_PERIOD": P["labels"][P["headline"]],
        "FOOTER": (f"Generated {data['generated']} from {data['root']} · scores: {', '.join(data['score_runs'])}"
                   + (f" · covariance: {data['cov_run']}" if data.get("cov_run") else "")
                   + " · every number on this page is read from those files."),
    }
    out = tpl
    for k, v in subs.items():
        out = out.replace("{{" + k + "}}", html.escape(str(v), quote=True))
    out = out.replace("{{DATA}}", blob)
    return out


def check_output(page: str) -> None:
    if "{{" in page.split('id="dashboard-data">')[0] or "}}" in page.split('id="dashboard-data">')[0]:
        raise RuntimeError("unresolved template placeholder in the page")
    n = len(page.encode("utf-8"))
    if n > MAX_BYTES:
        raise RuntimeError(f"page is {n / 1e6:.1f} MB, over the 16 MB artifact limit")


# --------------------------------------------------------------------------- CLI

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m assetpred.dashboard.build", description=__doc__.split("\n\n")[0])
    ap.add_argument("--dev", action="store_true", help="read results/dev/ (development period only)")
    ap.add_argument("--results-dir", default=None, help="results root (default results/ or results/dev/)")
    ap.add_argument("--out", default=None, help="output HTML (default <results root>/dashboard.html)")
    ap.add_argument("--scores-run", nargs="+", default=None, metavar="NAME",
                    help="marginal scores run(s) under <root>/scores/; several are merged, earlier ones win "
                         f"(default with --dev: {' '.join(DEV_SCORE_RUNS_CONSOLIDATED)} if present, else "
                         f"{' '.join(DEV_SCORE_RUNS)}; required otherwise)")
    ap.add_argument("--cov-scores-run", default=None, metavar="NAME",
                    help=f"covariance scores run (default with --dev: {DEV_COV_RUN_CONSOLIDATED} if present, else "
                         f"{DEV_COV_RUN}; required otherwise "
                         "unless --no-covariance)")
    ap.add_argument("--no-covariance", action="store_true", help="build without the covariance evidence")
    ap.add_argument("--strategy-dir", default=None, help="strategy results folder (default <root>/strategy)")
    ap.add_argument("--no-strategy", action="store_true", help="build without the strategy test")
    args = ap.parse_args(argv)

    root = Path(args.results_dir) if args.results_dir else (config.RESULTS / "dev" if args.dev else config.RESULTS)
    d_runs, d_cov = dev_defaults(root)
    runs = args.scores_run or (d_runs if args.dev else None)
    if not runs:
        avail = ex._listing(root / "scores")
        ap.error("--scores-run is required without --dev; runs in " f"{root / 'scores'}: " + (", ".join(avail) or "(none)"))
    cov_run = None if args.no_covariance else (args.cov_scores_run or (d_cov if args.dev else None))
    if cov_run is None and not args.no_covariance:
        ap.error("--cov-scores-run is required without --dev (or pass --no-covariance)")
    sdir = None if args.no_strategy else Path(args.strategy_dir) if args.strategy_dir else root / "strategy"
    out = Path(args.out) if args.out else root / "dashboard.html"

    data = build_context(root, args.dev, runs, cov_run, sdir)
    page = render(data)
    check_output(page)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    n = len(page.encode("utf-8"))
    print(f"dashboard written to {out} ({n / 1e6:.2f} MB)")
    if n > WARN_BYTES:
        print(f"warning: page is larger than {WARN_BYTES / 1e6:.0f} MB", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
