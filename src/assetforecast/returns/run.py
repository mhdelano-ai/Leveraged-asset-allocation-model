"""The replication run: assetpred's frozen mean models and baselines, unchanged,
at h = 12 on the package's 12-asset return panel.

    python -m assetforecast.returns.run [--max-date D] [--models ...] [--jobs N] [--reps R]

Reads the features from ``paths.FEATURES`` and the excess returns from
``assetforecast.returns.inputs`` (the panel without tier D, from 1926-07),
audits the features for gross look-ahead, runs the models (default
``config.REPLICATION_MODELS``; the benchmarks ``hist_mean`` and ``vol_ewma``
are always added) through the walk-forward engine, scores them over the
periods in ``config.PERIODS`` and writes, under ``paths.RESULTS_RETURNS``:

* ``forecasts/<model>/h12.parquet`` + ``.json`` (cached per model; reused when
  data, settings and code are unchanged unless ``--force``),
* ``scores/replication.{parquet,csv,json}`` and ``replication_cumdsse.parquet``.

Every output carries ``config.REPLICATION_LABEL``: both holdouts were spent
before this run, so its scores are a replication, not out-of-sample evidence.

``--max-date D`` sets ``ASSETFORECAST_MAX_DATE`` before ``assetforecast.paths``
is imported: the panel and the features end at D and everything is written
under ``results/dev/returns``. ``--feature-lag k`` runs the models on features
shifted k extra months, saved under ``<model>@lag<k>``: a look-ahead
sensitivity check.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import warnings

import pandas as pd

from . import config
from .evaluation import leakage, scoring, walkforward
from .models import registry
from .models.baselines import BENCHMARK, VOL_BENCHMARK

log = logging.getLogger("assetforecast.returns.run")

HORIZON = 12
RUN_NAME = "replication"
_ENV = "ASSETFORECAST_MAX_DATE"


def _set_max_date(max_date: str | None) -> None:
    """Point ``assetforecast.paths`` at the right results root before it is imported."""
    if max_date:
        os.environ[_ENV] = str(pd.Timestamp(max_date).date())
    mod = sys.modules.get("assetforecast.paths")
    if mod is not None and getattr(mod, "_DEV", None) != bool(os.environ.get(_ENV)):
        raise SystemExit(
            "assetforecast.paths was imported before --max-date could take effect "
            f"({_ENV}={os.environ.get(_ENV)!r} but paths was set up for "
            f"{'a development' if mod._DEV else 'the full-sample'} run); run this module "
            "as a fresh process.")


def load_inputs(max_date: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(features, xr): the feature panel and the monthly excess log returns."""
    from .. import paths
    from . import inputs
    if not paths.FEATURES.exists():
        raise SystemExit(f"Feature panel is missing: {paths.FEATURES}\n"
                         "Build it with the feature stage (assetforecast.returns.features) first.")
    features = pd.read_parquet(paths.FEATURES)
    if list(features.index.names) != ["date", "asset"]:
        raise SystemExit(f"{paths.FEATURES}: index must be (date, asset)")
    panel = inputs.load_return_panel(max_date=max_date)
    xr = inputs.excess_returns(panel)
    cut = xr.index.max()
    if max_date:
        cut = min(cut, pd.Timestamp(max_date))
    features = features[features.index.get_level_values("date") <= cut]
    return features, xr


def check_frozen_inputs(features: pd.DataFrame, xr: pd.DataFrame, asset_group=None) -> None:
    """Refuse to run the frozen models on inputs they were not frozen for:
    every asset needs a pooling group, and the feature set ``"all"`` must resolve
    to exactly ``config.FROZEN_FEATURE_COLUMNS``."""
    from .models import ml_mean
    groups = ml_mean.ASSET_GROUP if asset_group is None else asset_group
    assets = set(xr.columns) | set(features.index.get_level_values("asset"))
    missing = sorted(assets - set(groups))
    if missing:
        raise SystemExit(f"assets without a pooling group in ASSET_GROUP: {missing}")
    got = ml_mean.resolve_columns("all", features.columns)
    if got != list(config.FROZEN_FEATURE_COLUMNS):
        extra = [c for c in got if c not in config.FROZEN_FEATURE_COLUMNS]
        lost = [c for c in config.FROZEN_FEATURE_COLUMNS if c not in got]
        raise SystemExit("feature set 'all' does not resolve to config.FROZEN_FEATURE_COLUMNS "
                         f"(extra {extra}, missing {lost}, order differs: {not extra and not lost})")


def _stamp(models, results_dir) -> None:
    """Add the replication label to each forecast's JSON sidecar (the cache
    fingerprint is untouched, so cached files stay valid)."""
    for m in models:
        p = walkforward.forecast_path(m, HORIZON, results_dir).with_suffix(".json")
        if p.exists():
            meta = json.loads(p.read_text())
            meta["label"] = config.REPLICATION_LABEL
            p.write_text(json.dumps(meta, indent=1, default=str))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m assetforecast.returns.run",
                                 description=__doc__.split("\n\n")[0])
    ap.add_argument("--max-date", default=None,
                    help=f"development run: data end at this date ({_ENV}); "
                         "outputs go under results/dev/returns")
    ap.add_argument("--models", nargs="+", default=list(config.REPLICATION_MODELS),
                    help=f"registry names (default: the replication list; benchmarks {BENCHMARK}, "
                         f"{VOL_BENCHMARK} are always added)")
    ap.add_argument("--jobs", type=int, default=4, help="parallel model jobs")
    ap.add_argument("--reps", type=int, default=999, help="bootstrap reps for SPA / MCS")
    ap.add_argument("--force", action="store_true", help="recompute cached forecasts")
    ap.add_argument("--run-name", default=RUN_NAME, help="scores file name")
    ap.add_argument("--feature-lag", type=int, default=0)
    ap.add_argument("--skip-audit", action="store_true",
                    help="do not stop on features that look like look-ahead (not recommended)")
    ap.add_argument("--list", action="store_true", help="list registered models and exit")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    _set_max_date(args.max_date)
    from .. import paths

    if args.list:
        print("\n".join(registry.available()))
        return 0
    unknown = [m for m in args.models if m not in registry.available()]
    if unknown:
        raise SystemExit(f"unknown models {unknown}; registered: {', '.join(registry.available())}")

    max_date = os.environ.get(_ENV) or None
    features, xr = load_inputs(max_date)
    check_frozen_inputs(features, xr)
    if not args.skip_audit:
        try:
            leakage.audit_feature_leakage(features, xr, HORIZON)
        except leakage.SuspectFeatureError as e:
            raise SystemExit(str(e)) from None

    models = list(dict.fromkeys([BENCHMARK, VOL_BENCHMARK, *args.models]))
    lag = args.feature_lag
    results_dir = paths.RESULTS_RETURNS
    settings = walkforward.WalkForwardSettings()
    parts = [walkforward.run_many(models, [HORIZON], features, xr, settings, results_dir,
                                  n_jobs=args.jobs, force=args.force)]
    if lag:
        lagged = leakage.lag_features(features, lag)
        test_models = [f"{m}@lag{lag}" for m in args.models if m not in (BENCHMARK, VOL_BENCHMARK)]
        for m in test_models:
            registry.register(m, registry.get_factory(m.split("@")[0]), overwrite=True)
        parts.append(walkforward.run_many(test_models, [HORIZON], lagged, xr, settings, results_dir,
                                          n_jobs=args.jobs, force=args.force))
        models += test_models
    _stamp(models, results_dir)
    print(config.REPLICATION_LABEL)
    print(pd.concat(parts).to_string(index=False))

    fc = walkforward.load_forecasts(models, [HORIZON], results_dir)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        scores, cum = scoring.score_forecasts(
            fc, xr, subperiods=config.subperiods(), mt_reps=args.reps,
            mt_periods=config.MT_PERIODS, families=config.SPA_FAMILIES)
    run_name = args.run_name + (f"@lag{lag}" if lag else "")
    path = scoring.write_scores(scores, cum, run_name, results_dir, meta={
        "label": config.REPLICATION_LABEL, "frozen_at": config.FROZEN_AT,
        "oos_origin_start": config.OOS_ORIGIN_START,
        "models": models, "horizons": [HORIZON], "max_date": max_date,
        "settings": vars(settings), "periods": config.PERIODS, "mt_periods": config.MT_PERIODS,
        "feature_columns": len(config.FROZEN_FEATURE_COLUMNS),
        "assets": list(xr.columns), "first_return": str(xr.index.min().date()),
        "last_return": str(xr.index.max().date()),
        "data": walkforward.data_fingerprint(features.sort_index(), xr), "reps": args.reps})
    with pd.option_context("display.width", 160, "display.max_columns", 20):
        print(f"\n{config.REPLICATION_LABEL}.")
        for period in ("full", "holdout", "holdout_ap"):
            if (scores.period == period).any():
                print(f"\nPooled summary, period {period!r}:")
                print(scoring.summary(scores, period).round(4).to_string())
    print(f"\nscores written to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
