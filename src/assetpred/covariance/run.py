"""Walk-forward covariance forecasts and their scores.

    python -m assetpred.covariance.run --dev --models vc_ewma_ewma ewma_cov --horizons 1 3 12
    python -m assetpred.covariance.run --dev --models frozen --combine vc_ewma_ewma:vol_ewma

``--dev`` truncates the monthly and daily panels at ``config.DEV_END`` *before
anything is computed*, stops each horizon at the last origin whose target is
realized by then, and writes under ``results/dev/`` (covariances, scores, and a
row per configuration in ``results/dev/trials_cov.csv``). Without ``--dev`` the
full sample, including the holdout, is used and results go to ``results/``:
run that once, with ``--models frozen``, after the model list is frozen.

The benchmark ``sample`` is always included. ``--models all`` runs every
registered model, ``--models frozen`` the frozen list. ``--combine C:S`` pairs
the correlation of covariance model C with the per-asset ``sd`` of marginal
model S (read from ``results[/dev]/forecasts/S/``) and scores the result as
``C__sd_S``. At most ``--jobs`` (default 2) worker processes.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path

import pandas as pd

from .. import config, schema
from . import combine as cmb
from . import engine
from .base import available
from .models import BENCHMARK, FROZEN, MODELS
from .scoring import DEFAULT_SUBPERIODS, score_covariances, summary

log = logging.getLogger("assetpred.covariance.run")
DAILY = config.PROCESSED / "asset_returns_daily.parquet"
TRIALS = config.RESULTS / "dev" / "trials_cov.csv"


def load_inputs(dev: bool):
    xr = schema.excess_log_returns()
    daily = pd.read_parquet(DAILY) if DAILY.exists() else None
    if daily is not None:
        daily.index = pd.DatetimeIndex(daily.index)
    features = schema.load_features() if schema.FEATURES.exists() else None
    if dev:
        xr = xr.loc[:config.DEV_END]
        if daily is not None:
            daily = daily.loc[:config.DEV_END]
        if features is not None:
            features = features[features.index.get_level_values("date") <= pd.Timestamp(config.DEV_END)]
    return xr, daily, features


def log_trials(scores: pd.DataFrame, models, run_name: str, data_fp: str, path: Path = TRIALS) -> int:
    """Append one row per model configuration (skipping exact repeats of model,
    parameters, code and data) with its headline dev scores per horizon."""
    from .base import get_factory
    s = scores[(scores.period == "full") & (scores.asset == "joint")]
    code = engine.code_fingerprint()
    rows = []
    for m in models:
        if "__sd_" in m:
            params = {"combined": m}
        else:
            params = get_factory(m)().params()
        row = {"logged_at": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
               "run_name": run_name, "model": m, "params": json.dumps(params, sort_keys=True, default=str),
               "code": code, "data": data_fp}
        for h in sorted(s.horizon.unique()):
            g = s[(s.model == m) & (s.horizon == h)]

            def val(met, bm):
                v = g[(g.metric == met) & (g.benchmark == bm)].value
                return float(v.iloc[0]) if len(v) else float("nan")
            row[f"nll_h{h:02d}"] = val("nll", "none")
            row[f"d_nll_h{h:02d}"] = val("diff_nll", BENCHMARK)
            row[f"p_nll_h{h:02d}"] = val("dm_p_nll", BENCHMARK)
        rows.append(row)
    new = pd.DataFrame(rows)
    if path.exists():
        old = pd.read_csv(path)
        key = ["model"]   # a registered name never changes meaning
        seen = set(map(tuple, old[key].astype(str).to_numpy()))
        new = new[[tuple(map(str, r)) not in seen for r in new[key].to_numpy()]]
        out = pd.concat([old, new], ignore_index=True)
    else:
        out = new
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False, float_format="%.6g")
    return len(new)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m assetpred.covariance.run", description=__doc__.split("\n\n")[0])
    ap.add_argument("--models", nargs="+", default=["frozen"])
    ap.add_argument("--horizons", nargs="+", type=int, default=list(config.HORIZONS))
    ap.add_argument("--combine", nargs="*", default=[], metavar="CORR:SD")
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--reps", type=int, default=999)
    ap.add_argument("--dev", action="store_true",
                    help=f"development run on data through {config.DEV_END} only; writes under results/dev/")
    ap.add_argument("--no-trials", action="store_true", help="do not log to trials_cov.csv")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    if args.list:
        print("\n".join(available()))
        print("frozen:", " ".join(FROZEN) or "(none yet)")
        return 0
    names: list[str] = []
    for m in args.models:
        names += list(MODELS) if m == "all" else list(FROZEN) if m == "frozen" else [m]
    unknown = [m for m in names if m not in available()]
    if unknown:
        raise SystemExit(f"unknown covariance models {unknown}; registered: {', '.join(available())}")
    bad_h = [h for h in args.horizons if h not in config.HORIZONS]
    if bad_h:
        raise SystemExit(f"horizons {bad_h} not in {config.HORIZONS}")
    pairs = []
    for c in args.combine:
        if ":" not in c:
            raise SystemExit(f"--combine expects CORR:SD, got {c!r}")
        cm, sm = c.split(":", 1)
        pairs.append((cm, sm))
        if cm not in available():
            raise SystemExit(f"unknown covariance model {cm!r}")
        names.append(cm)
    models = list(dict.fromkeys([BENCHMARK, *names]))
    if not args.dev:
        print("NOTE: full-sample run; this uses and scores the holdout period.", file=sys.stderr)

    xr, daily, features = load_inputs(args.dev)
    results_dir = config.RESULTS / "dev" if args.dev else config.RESULTS
    settings_for = (lambda h: engine.CovSettings(end=engine.dev_end(h))) if args.dev else (lambda h: engine.CovSettings())
    status = engine.run_many(models, args.horizons, xr, daily, settings_for, results_dir,
                             n_jobs=min(args.jobs, 2), force=args.force, features=features)
    print(status.to_string(index=False))
    cov = engine.load_covariances(models, args.horizons, results_dir)

    scored = list(models)
    for cm, sm in pairs:
        marg = cmb.load_marginals(sm, args.horizons, results_dir)
        if args.dev:
            ends = {h: pd.Timestamp(engine.dev_end(h)) for h in args.horizons}
            marg = marg[marg["date"] <= marg["horizon"].map(ends)]
        joint = cmb.combine(cov[cov.model == cm], marg)
        name = cmb.combined_name(cm, sm)
        for h, g in joint.groupby("horizon"):
            p = engine.cov_path(name, h, results_dir)
            p.parent.mkdir(parents=True, exist_ok=True)
            g.to_parquet(p, index=False)
        cov = pd.concat([cov, joint], ignore_index=True)
        scored.append(name)
    engine.validate_cov_frame(cov)

    subperiods = DEFAULT_SUBPERIODS if args.dev else (
        *DEFAULT_SUBPERIODS, ("pre_holdout", config.OOS_START, config.DEV_END),
        ("holdout", config.HOLDOUT_START, "2099-12-31"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        scores, per_date = score_covariances(cov, xr, BENCHMARK, subperiods, reps=args.reps)
    run_name = args.run_name or ("cov_" + "+".join(sorted(set(scored) - {BENCHMARK}) or ["baselines"]))[:120]
    root = results_dir / "scores"
    root.mkdir(parents=True, exist_ok=True)
    scores.to_parquet(root / f"{run_name}.parquet", index=False)
    scores.to_csv(root / f"{run_name}.csv", index=False, float_format="%.6g")
    per_date.to_parquet(root / f"{run_name}_per_date.parquet", index=False)
    data_fp = engine.data_fingerprint(xr, daily, features)
    (root / f"{run_name}.json").write_text(json.dumps({
        "models": scored, "horizons": args.horizons, "dev": args.dev, "benchmark": BENCHMARK,
        "ends": {h: settings_for(h).end for h in args.horizons}, "data": data_fp,
        "reps": args.reps}, indent=1, default=str))
    if args.dev and not args.no_trials:
        n = log_trials(scores, scored, run_name, data_fp)
        print(f"{n} new configuration(s) logged to {TRIALS}")
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(summary(scores, BENCHMARK).round(4).to_string())
    print(f"\nscores written to {root / (run_name + '.parquet')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
