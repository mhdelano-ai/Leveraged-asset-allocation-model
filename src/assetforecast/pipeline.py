"""Run everything, in order, with one command.

    python -m assetforecast.pipeline [--max-date D] [--jobs N] [--skip STEP ...]
                                     [--rebuild-covariance] [--dry-run] [--list]

Steps (each runs as its own Python process; ``--skip`` takes these names):

1. ``covariance``    every covariance model at every month-end origin, into
                     ``results/full/`` (``scripts/run_all.sh``, about 30 minutes on
                     4 cores). By default an existing ``results/full/`` is REUSED
                     and the step only reports how current it is;
                     ``--rebuild-covariance`` reruns it.
2. ``covhistory``    the selected model's 21-day and 12-month covariance at every
                     origin (``python -m assetforecast.covhistory``)
3. ``features``      the return side's feature panel
                     (``python -m assetforecast.returns.features.build``)
4. ``replication``   the frozen mean models and baselines at h=12
                     (``python -m assetforecast.returns.run``)
5. ``distribution``  expected-return history, predictive distributions and their
                     scores (``python -m assetforecast.returns.distribution --score``)
6. ``strategy``      the pre-declared portfolio rules
                     (``python -m assetforecast.returns.evaluation.strategy``)
7. ``forecast``      the published forecast (``python -m assetforecast.forecast``)
8. ``dashboard``     the dashboard data (``python -m assetforecast.dashboard``), run
                     when that module exists and skipped with a message otherwise

The frozen covariance evaluation (``scripts/score_final.py`` -> ``results/final``)
is NOT a step: it scores the spent 2014-2025 holdout and is rerun by hand only.

The development switch
----------------------
``--max-date D`` sets ``ASSETFORECAST_MAX_DATE=D`` in the environment of every
step process, so ``assetforecast.paths`` is set up for a development run from
the first import in each of them, and passes ``--max-date D`` to the steps that
take it. Then the panel ends at D and the outputs go under ``results/dev/``
(covariance history, features, returns and the forecast, which goes to
``results/dev/forecasts`` rather than ``forecasts/``). The covariance step still
works on ``results/full/``: a stored forecast at origin t uses data through t
only (``scripts/leak_test.py``), and every later step keeps origins <= D. The
dashboard step is skipped under ``--max-date`` (it publishes ``dashboard/``).
Without ``--max-date`` the variable is removed from the steps' environment, so
a value left in the shell cannot turn a full run into a partial one.

Every step also gets ``OMP_NUM_THREADS=1`` (parallelism comes from ``--jobs``)
and runs without ``ASSETFORECAST_DROP_TIERS`` (the return side drops tier D
explicitly; the covariance side must see every tier) or
``ASSETFORECAST_RESULTS`` (each step finds its own store; the forecast step is
pointed at ``results/full``).
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESULTS_FULL = ROOT / "results" / "full"
PRIMARY_COV = RESULTS_FULL / "forecasts" / "cov" / "combo_cov_main.parquet"
RUN_ALL = ROOT / "scripts" / "run_all.sh"
_MAX_ENV = "ASSETFORECAST_MAX_DATE"
_CLEARED_ENV = ("ASSETFORECAST_DROP_TIERS", "ASSETFORECAST_RESULTS")


@dataclass
class Config:
    max_date: str | None = None
    jobs: int = 3
    rebuild_covariance: bool = False
    python: str = sys.executable


@dataclass
class Step:
    name: str
    what: str
    command: Callable[[Config], list[str] | None]   # None: nothing to run (reason in `note`)
    note: Callable[[Config], str | None] = field(default=lambda c: None)


def _md(c: Config) -> list[str]:
    return ["--max-date", c.max_date] if c.max_date else []


def _module(c: Config, mod: str, *args: str) -> list[str]:
    return [c.python, "-m", mod, *args]


# ------------------------------------------------------------------ covariance step
def covariance_status(path: Path = PRIMARY_COV) -> dict:
    """What ``results/full`` holds for the selected model, without loading the panel."""
    if not path.exists():
        return {"exists": False}
    import pyarrow.parquet as pq
    o = pq.read_table(path, columns=["origin"]).column("origin").to_pandas()
    return {"exists": True, "first_origin": pd.Timestamp(o.min()), "last_origin": pd.Timestamp(o.max())}


def _last_complete_month() -> pd.Timestamp | None:
    p = ROOT / "data" / "processed" / "panel_monthly.parquet"
    if not p.exists():
        return None
    import pyarrow.parquet as pq
    d = pq.read_table(p, columns=["date"]).column("date").to_pandas()
    return pd.Timestamp(d.max()) if len(d) else None


def _covariance_command(c: Config) -> list[str] | None:
    if c.rebuild_covariance or not covariance_status()["exists"]:
        return ["bash", str(RUN_ALL), str(c.jobs)]
    return None


def _covariance_note(c: Config) -> str | None:
    st = covariance_status()
    if c.rebuild_covariance:
        return "rebuilding results/full (--rebuild-covariance)"
    if not st["exists"]:
        return f"{PRIMARY_COV.relative_to(ROOT)} is missing: running scripts/run_all.sh (about 30 minutes)"
    msg = (f"reusing results/full ({PRIMARY_COV.name}: origins {st['first_origin'].date()} .. "
           f"{st['last_origin'].date()}); pass --rebuild-covariance to regenerate it")
    try:
        last = _last_complete_month()
    except Exception:   # the status line must never stop the run
        last = None
    if last is not None and not c.max_date and st["last_origin"] < last:
        msg += (f"\n  WARNING: the panel runs to {last.date()} but results/full stops at "
                f"{st['last_origin'].date()}: the forecast will be for the older origin. "
                "Pass --rebuild-covariance after a data update.")
    return msg


def _dashboard_command(c: Config) -> list[str] | None:
    if c.max_date or importlib.util.find_spec("assetforecast.dashboard") is None:
        return None
    return _module(c, "assetforecast.dashboard")


def _dashboard_note(c: Config) -> str | None:
    if c.max_date:
        return "skipped under --max-date (the dashboard publishes the full-sample results)"
    if importlib.util.find_spec("assetforecast.dashboard") is None:
        return "skipped: assetforecast.dashboard is not installed in this tree yet"
    return None


STEPS: tuple[Step, ...] = (
    Step("covariance", "covariance models, every origin -> results/full",
         _covariance_command, _covariance_note),
    Step("covhistory", "per-origin 21-day and 12-month covariance -> results/covariance",
         lambda c: _module(c, "assetforecast.covhistory", *_md(c))),
    Step("features", "return-side feature panel -> data/processed/features.parquet",
         lambda c: _module(c, "assetforecast.returns.features.build", *_md(c))),
    Step("replication", "frozen mean models and baselines at h=12 -> results/returns/{forecasts,scores}",
         lambda c: _module(c, "assetforecast.returns.run", *_md(c), "--jobs", str(c.jobs))),
    Step("distribution", "expected returns, distributions and their scores -> results/returns/{expected,distribution}",
         lambda c: _module(c, "assetforecast.returns.distribution", *_md(c), "--score")),
    Step("strategy", "pre-declared portfolio rules -> results/returns/strategy",
         lambda c: _module(c, "assetforecast.returns.evaluation.strategy", *_md(c), "--jobs", str(c.jobs))),
    Step("forecast", "published forecast -> forecasts/<origin>, forecasts/latest",
         lambda c: _module(c, "assetforecast.forecast", "--results", str(RESULTS_FULL), *_md(c))),
    Step("dashboard", "dashboard data -> dashboard/",
         _dashboard_command, _dashboard_note),
)
STEP_NAMES = tuple(s.name for s in STEPS)


def step_env(c: Config, base: dict | None = None) -> dict:
    """The environment every step runs in (see the module docstring)."""
    env = dict(os.environ if base is None else base)
    for k in _CLEARED_ENV:
        env.pop(k, None)
    env.pop(_MAX_ENV, None)
    if c.max_date:
        env[_MAX_ENV] = c.max_date
    env["OMP_NUM_THREADS"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    return env


def plan(c: Config, skip=()) -> list[tuple[Step, list[str] | None, str | None]]:
    """(step, command or None, note) for every step, in order; skipped steps have no command."""
    out = []
    for s in STEPS:
        if s.name in skip:
            out.append((s, None, "skipped (--skip)"))
        else:
            out.append((s, s.command(c), s.note(c)))
    return out


def run(c: Config, skip=(), runner=subprocess.run, dry_run: bool = False, log=print) -> list[dict]:
    """Run the plan; stop at the first step that fails. Returns one record per step."""
    unknown = set(skip) - set(STEP_NAMES)
    if unknown:
        raise ValueError(f"unknown steps {sorted(unknown)}; steps are {', '.join(STEP_NAMES)}")
    env = step_env(c)
    records = []
    steps = plan(c, skip)
    for i, (s, cmd, note) in enumerate(steps, 1):
        head = f"[{i}/{len(steps)}] {s.name}: {s.what}"
        log(head)
        if note:
            log(f"  {note}")
        if cmd is None:
            records.append({"step": s.name, "ran": False, "returncode": None, "seconds": 0.0, "note": note})
            continue
        log("  $ " + " ".join(cmd))
        if dry_run:
            records.append({"step": s.name, "ran": False, "returncode": None, "seconds": 0.0,
                            "note": "dry run", "command": cmd})
            continue
        t0 = time.time()
        rc = runner(cmd, cwd=str(ROOT), env=env).returncode
        dt = time.time() - t0
        records.append({"step": s.name, "ran": True, "returncode": rc, "seconds": dt, "command": cmd})
        log(f"  {s.name} finished in {dt:.0f}s (exit {rc})")
        if rc != 0:
            raise SystemExit(f"step {s.name!r} failed with exit code {rc}; later steps were not run. "
                             f"Rerun with --skip {' '.join(x.name for x, *_ in steps[:i - 1])} to resume.")
    return records


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m assetforecast.pipeline", description=__doc__.split("\n")[0],
                                 epilog="steps: " + ", ".join(STEP_NAMES))
    ap.add_argument("--max-date", default=None, help="development run: everything ends at D, outputs under results/dev")
    ap.add_argument("--jobs", type=int, default=3, help="parallel jobs for the covariance run, replication and strategy test")
    ap.add_argument("--skip", nargs="+", default=[], metavar="STEP", choices=STEP_NAMES, help="steps not to run")
    ap.add_argument("--rebuild-covariance", action="store_true",
                    help="rerun every covariance model into results/full (about 30 minutes on 4 cores)")
    ap.add_argument("--dry-run", action="store_true", help="print the commands without running them")
    ap.add_argument("--list", action="store_true", help="list the steps and exit")
    a = ap.parse_args(argv)
    if a.list:
        for s in STEPS:
            print(f"{s.name:13s} {s.what}")
        return 0
    md = str(pd.Timestamp(a.max_date).date()) if a.max_date else None
    c = Config(max_date=md, jobs=a.jobs, rebuild_covariance=a.rebuild_covariance)
    t0 = time.time()
    recs = run(c, a.skip, dry_run=a.dry_run)
    ran = [r for r in recs if r["ran"]]
    print(f"pipeline done in {time.time() - t0:.0f}s: ran {', '.join(r['step'] for r in ran) or 'nothing'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
