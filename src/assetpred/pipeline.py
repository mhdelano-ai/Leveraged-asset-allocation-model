"""Run the frozen pipeline end to end: marginal forecasts and scores, covariance
forecasts and scores, the strategy test, then the dashboard.

    python -m assetpred.pipeline --dev     # targets realized by config.DEV_END only
    python -m assetpred.pipeline           # the full run, holdout included

The full run scores the holdout. It is meant to be run once, on the model lists
frozen in ``config``; rerunning it after changing a model turns the holdout into
a tuning sample. It does not rebuild the data: run ``assetpred.data.assets``,
``assetpred.data.predictors`` and ``assetpred.features.build`` first for that.
"""
from __future__ import annotations

import argparse
import sys
import time

from . import config

SCORES_RUN = "frozen"       # results[/dev]/scores/frozen.*
COV_SCORES_RUN = "cov_frozen"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m assetpred.pipeline", description=__doc__.split("\n\n")[0])
    ap.add_argument("--dev", action="store_true", help="development period only; never touches the holdout")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--reps", type=int, default=999, help="bootstrap replications for SPA / MCS")
    ap.add_argument("--skip", nargs="*", default=[], choices=["marginal", "covariance", "strategy", "dashboard"])
    args = ap.parse_args(argv)
    dev = ["--dev"] if args.dev else []

    from .covariance import run as cov_run
    from .evaluation import run as marginal_run
    from .evaluation import strategy

    def dashboard(argv_):
        from .dashboard import build
        return build.main(argv_)

    steps = {
        "marginal": (marginal_run.main, [
            *dev, "--models", *config.FROZEN_MARGINAL, *config.ROBUSTNESS, "--run-name", SCORES_RUN,
            "--jobs", str(args.jobs), "--reps", str(args.reps)]),
        "covariance": (cov_run.main, [
            *dev, "--models", "frozen", "--combine", f"{config.PRIMARY_COV}:{config.COMPOSITE}",
            "--run-name", COV_SCORES_RUN, "--jobs", str(min(args.jobs, 2)), "--reps", str(args.reps),
            "--no-trials"]),
        "strategy": (strategy.main, [*dev, "--jobs", str(min(args.jobs, 3))]),
        "dashboard": (dashboard, [*dev, "--scores-run", SCORES_RUN, "--cov-scores-run", COV_SCORES_RUN]),
    }
    for name, (fn, argv_) in steps.items():
        if name in args.skip:
            continue
        t0 = time.time()
        print(f"\n=== {name}: {' '.join(argv_)}", flush=True)
        rc = fn(argv_)
        if rc:
            print(f"{name} failed with exit code {rc}", file=sys.stderr)
            return int(rc)
        print(f"=== {name} done in {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
