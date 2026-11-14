"""Command line for the evaluation core.

    python -m volmodel.eval.run targets [--synthetic] [--rebuild]
    python -m volmodel.eval.run forecast --kind vol --model NAME [NAME ...] | --all
           [--start YYYY-MM-DD] [--end YYYY-MM-DD] [--step K] [--jobs J] [--synthetic] [--force]
    python -m volmodel.eval.run score --kind vol [--models A B cov:C ...] [--subset daily]
           [--eras 1999-12-31 2009-12-31] [--synthetic]

``--synthetic`` uses ``make_synthetic_panel(seed=0)`` and writes under
``results/synthetic/`` so test runs never touch real results.
"""
from __future__ import annotations

import argparse
import sys
import time

import pandas as pd

from ..paths import RESULTS
from .harness import KINDS, _panel_from_source, run_models, stored_models
from .losses import PRIMARY, compute_losses, coverage_summary, scoreboard
from .targets import get_targets


def _results_dir(synthetic: bool):
    return RESULTS / "synthetic" if synthetic else RESULTS


def _source(synthetic: bool) -> str:
    return "synthetic" if synthetic else "real"


def cmd_targets(a) -> int:
    panel = _panel_from_source(_source(a.synthetic))
    t0 = time.perf_counter()
    T = get_targets(panel, _results_dir(a.synthetic), rebuild=a.rebuild)
    dt = time.perf_counter() - t0
    v = T.vol
    print(f"targets in {_results_dir(a.synthetic) / 'targets'} ({dt:.2f}s)")
    print(f"observed through {T.meta['observed_through']}; origins {v['origin'].min():%Y-%m}..{v['origin'].max():%Y-%m}"
          f" ({v['origin'].nunique()}); vol rows {len(v)} ({int(v['proxy'].sum())} monthly proxy);"
          f" corr pairs {len(T.corr)}")
    with pd.option_context("display.width", 200):
        print(coverage_summary(T).to_string())
    return 0


def cmd_forecast(a) -> int:
    if a.all:
        from ..models.base import list_models
        names = list_models(a.kind)
    else:
        names = a.model or []
    if not names:
        print("no models given (use --model NAME or --all)", file=sys.stderr)
        return 2
    kw = dict(results_dir=_results_dir(a.synthetic), start=a.start, end=a.end, step=a.step,
              force=a.force, progress=a.jobs <= 1)
    t0 = time.perf_counter()
    res = run_models(a.kind, names, source=_source(a.synthetic), jobs=a.jobs, **kw)
    for r in res:
        print(r)
    print(f"total {time.perf_counter() - t0:.1f}s")
    return 1 if any(r.error for r in res) else 0


def cmd_score(a) -> int:
    rd = _results_dir(a.synthetic)
    models = a.models or stored_models(a.kind, rd)
    if not models:
        print(f"no stored {a.kind} forecasts in {rd / 'forecasts' / a.kind}", file=sys.stderr)
        return 2
    T = get_targets(_panel_from_source(_source(a.synthetic)), rd)
    L = compute_losses(a.kind, models, T, rd)
    sb = scoreboard(L, a.kind, subset=a.subset, eras=a.eras)
    with pd.option_context("display.width", 220, "display.max_columns", 50,
                           "display.float_format", "{:.4f}".format):
        print(f"== {a.kind} scoreboard, subset={a.subset!r}, primary={PRIMARY[a.kind]} "
              f"(common origins only) ==")
        print(sb["overall"].to_string())
        for k in ("by_era", "by_asset"):
            if k in sb:
                print(f"\n-- {k} ({PRIMARY[a.kind]}) --")
                print(sb[k].to_string())
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m volmodel.eval.run")
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("targets", help="compute and cache realized targets")
    t.add_argument("--synthetic", action="store_true")
    t.add_argument("--rebuild", action="store_true")

    f = sub.add_parser("forecast", help="walk-forward forecasts for registered models")
    f.add_argument("--kind", required=True, choices=KINDS)
    f.add_argument("--model", nargs="+")
    f.add_argument("--all", action="store_true")
    f.add_argument("--start")
    f.add_argument("--end")
    f.add_argument("--step", type=int, default=1)
    f.add_argument("--jobs", type=int, default=1)
    f.add_argument("--synthetic", action="store_true")
    f.add_argument("--force", action="store_true")

    s = sub.add_parser("score", help="print the scoreboard for stored forecasts")
    s.add_argument("--kind", required=True, choices=KINDS)
    s.add_argument("--models", nargs="+")
    s.add_argument("--subset", default="daily")
    s.add_argument("--eras", nargs="+")
    s.add_argument("--synthetic", action="store_true")

    a = ap.parse_args(argv)
    return {"targets": cmd_targets, "forecast": cmd_forecast, "score": cmd_score}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
