"""Command-line runner: walk-forward forecasts and scores for registry models.

    python -m assetpred.evaluation.run --models ridge my_gbm --horizons 1 3 12

Loads ``data/processed/features.parquet`` and the excess returns via
``assetpred.schema``, audits the features for gross look-ahead, runs the named
models plus the benchmarks (``hist_mean``, ``vol_ewma``; always included) at
each horizon (cached per model and horizon under ``results/forecasts``; reused
when nothing changed unless ``--force``), scores them together and writes
``results/scores/<run-name>.*``.

``--feature-lag k`` runs the models on features shifted k extra months, saved
under ``<model>@lag<k>``: a look-ahead sensitivity check.
"""
from __future__ import annotations

import argparse
import logging
import sys
import warnings

import pandas as pd

from .. import config, schema
from ..models import registry
from ..models.baselines import BENCHMARK, VOL_BENCHMARK
from . import leakage, scoring, walkforward

log = logging.getLogger("assetpred.evaluation.run")


def _need(paths) -> None:
    missing = [p for p in paths if not p.exists()]
    if missing:
        lines = "\n  ".join(str(p) for p in missing)
        raise SystemExit(
            "Processed data is missing; the evaluation needs:\n  " + lines +
            "\nBuild asset_returns/cash with the data stage (assetpred.data) and "
            "features.parquet with the feature stage (assetpred.features) first.")


def load_inputs():
    _need([schema.ASSET_RETURNS, schema.CASH, schema.FEATURES])
    xr = schema.excess_log_returns()
    features = schema.load_features()
    return features, xr


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m assetpred.evaluation.run",
                                 description=__doc__.split("\n\n")[0])
    ap.add_argument("--models", nargs="+", default=[],
                    help=f"registry names (benchmarks {BENCHMARK}, {VOL_BENCHMARK} are always added)")
    ap.add_argument("--horizons", nargs="+", type=int, default=list(config.HORIZONS))
    ap.add_argument("--jobs", type=int, default=4, help="parallel (model, horizon) jobs")
    ap.add_argument("--force", action="store_true", help="recompute cached forecasts")
    ap.add_argument("--run-name", default=None, help="scores file name (default from models)")
    ap.add_argument("--feature-lag", type=int, default=0)
    ap.add_argument("--reps", type=int, default=999, help="bootstrap reps for SPA / MCS")
    ap.add_argument("--skip-audit", action="store_true",
                    help="do not stop on features that look like look-ahead (not recommended)")
    ap.add_argument("--list", action="store_true", help="list registered models and exit")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")

    if args.list:
        print("\n".join(registry.available()))
        return 0
    unknown = [m for m in args.models if m not in registry.available()]
    if unknown:
        raise SystemExit(f"unknown models {unknown}; registered: {', '.join(registry.available())}")
    bad_h = [h for h in args.horizons if h not in config.HORIZONS]
    if bad_h:
        raise SystemExit(f"horizons {bad_h} not in config.HORIZONS {config.HORIZONS}")

    features, xr = load_inputs()
    if not args.skip_audit:
        for h in args.horizons:
            try:
                leakage.audit_feature_leakage(features, xr, h)
            except leakage.SuspectFeatureError as e:
                raise SystemExit(str(e)) from None

    models = list(dict.fromkeys([BENCHMARK, VOL_BENCHMARK, *args.models]))
    lag = args.feature_lag
    if lag:
        lagged = leakage.lag_features(features, lag)
        for m in args.models:
            registry.register(f"{m}@lag{lag}", registry.get_factory(m), overwrite=True)
        test_models = [f"{m}@lag{lag}" for m in args.models]
    settings = walkforward.WalkForwardSettings()
    status = walkforward.run_many(models, args.horizons, features, xr, settings,
                                  n_jobs=args.jobs, force=args.force)
    if lag:
        status = pd.concat([status, walkforward.run_many(
            test_models, args.horizons, lagged, xr, settings, n_jobs=args.jobs, force=args.force)])
        models += test_models
    print(status.to_string(index=False))

    fc = walkforward.load_forecasts(models, args.horizons)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        scores, cum = scoring.score_forecasts(fc, xr, mt_reps=args.reps)
    run_name = args.run_name or "+".join(sorted(args.models) or ["baselines"]) + (
        f"@lag{lag}" if lag else "")
    path = scoring.write_scores(scores, cum, run_name, meta={
        "models": models, "horizons": args.horizons, "settings": vars(settings),
        "data": walkforward.data_fingerprint(features.sort_index(), xr), "reps": args.reps})
    with pd.option_context("display.width", 160, "display.max_columns", 20):
        print(scoring.summary(scores).round(4).to_string())
    print(f"\nscores written to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
