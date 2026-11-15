"""The frozen learners re-aimed at each horizon, under new names.

``hz_ridge``, ``hz_gbm`` and ``hz_ens`` are built from the factories registered
as ``ridge_z``, ``gbm_z`` and ``ens_z`` (``returns/models/ml_mean.py``): the
same classes with the same settings, driven by the same walk-forward engine
(purge ``s + h <= t``, refit every 12 months, first origin 1990-01). Only the
horizon handed to the engine differs. They are new models in the protocol's
sense: nothing is written to ``results/returns`` and the frozen registrations
are not changed. At h = 12 a copy must reproduce its original exactly;
``tests/horizons`` checks that on the committed forecasts.

    python -m assetforecast.horizons.mlcopies [--jobs N] [--horizons 1 3 ...]
"""
from __future__ import annotations

import argparse
import sys

import pandas as pd

from . import HORIZONS, STORE

COPIES = {"hz_ridge": "ridge_z", "hz_gbm": "gbm_z", "hz_ens": "ens_z"}


def register_copies() -> None:
    from ..returns.models import registry
    for new, old in COPIES.items():
        registry.register(new, registry.get_factory(old), overwrite=True)


def run(horizons=HORIZONS, jobs: int = 4, force: bool = False) -> pd.DataFrame:
    from ..returns import config, run as rrun
    from ..returns.evaluation import walkforward
    features, xr = rrun.load_inputs(None)
    rrun.check_frozen_inputs(features, xr)
    register_copies()
    short = [h for h in horizons if h + 2 <= config.MIN_ASSET_HISTORY]
    parts = []
    if short:
        parts.append(walkforward.run_many(list(COPIES), short, features, xr,
                                          walkforward.WalkForwardSettings(), STORE, n_jobs=jobs,
                                          force=force))
    for h in [h for h in horizons if h not in short]:
        parts.append(walkforward.run_many(
            list(COPIES), [h], features, xr,
            walkforward.WalkForwardSettings(min_history=h + 2), STORE, n_jobs=jobs, force=force))
    return pd.concat(parts, ignore_index=True)


def load(horizons=HORIZONS) -> pd.DataFrame:
    """Cached forecasts of the copies: columns model, horizon, date, asset, mean."""
    from ..returns.evaluation import walkforward
    fc = walkforward.load_forecasts(list(COPIES), list(horizons), STORE, missing="skip")
    return fc[["model", "horizon", "date", "asset", "mean"]]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m assetforecast.horizons.mlcopies")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--horizons", type=int, nargs="+", default=list(HORIZONS))
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    print(run(a.horizons, a.jobs, a.force).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
