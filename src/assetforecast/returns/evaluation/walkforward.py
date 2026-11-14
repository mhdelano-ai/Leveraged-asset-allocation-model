"""Expanding-window walk-forward engine with an enforced purge, and the forecast cache.

At each monthly origin t from ``oos_start``:

* asset a is *eligible* if it has at least ``min_history`` months of its own
  excess returns through t;
* the model is refit (a fresh object from its factory) at the first origin,
  every ``refit_every`` months after (the model's own ``refit_every`` if set,
  else the setting), and whenever the eligible set changes;
* the training set holds feature rows dated s with ``s + h <= t`` (month
  arithmetic) and a non-missing target, plus monthly excess returns through t.
  This purge is done here, not in models, and :class:`TrainingSet` re-checks it;
* ``predict`` gets the feature rows dated t for the eligible assets (all-NaN
  rows if the feature panel has none) and its output is validated.

Forecasts are made for every origin up to the last date in both panels, so the
final h origins have forecasts but no realized target yet; scoring drops them.

Cache: one parquet per (model, horizon) at
``<results_dir>/forecasts/<model>/h<hh>.parquet`` (default results dir:
``assetforecast.paths.RESULTS_RETURNS``) with a JSON sidecar recording a
fingerprint of the data, settings, model parameters and the source of the
model's module and of this harness. A cached file is reused only if the
fingerprint matches (or ``force=True`` recomputes).
"""
from __future__ import annotations

import hashlib
import inspect
import json
import logging
import multiprocessing as mp
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from .. import config, schema
from ..models import base as _base
from ..models.base import (Forecaster, InsufficientDataError, TrainingSet,
                           validate_predictions)
from .cv import month_ordinal

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class WalkForwardSettings:
    oos_start: str = config.OOS_START
    refit_every: int = config.REFIT_EVERY
    min_history: int = config.MIN_ASSET_HISTORY
    end: str | None = None   # last origin (default: last date in both panels)


def _training_cutoff(origin_ord: int, horizon: int) -> int:
    """Last row date (month ordinal) usable at origin: target fully realized."""
    return origin_ord - horizon


def check_inputs(features: pd.DataFrame, xr: pd.DataFrame) -> pd.DataFrame:
    """Validate panels against the contract; return features sorted by (date, asset)."""
    if list(features.index.names) != ["date", "asset"]:
        raise ValueError("features: index must be (date, asset)")
    schema._check_month_end(xr.index, "excess returns")
    d = pd.DatetimeIndex(features.index.get_level_values("date"))
    if d.tz is not None or not (d == d + pd.offsets.MonthEnd(0)).all():
        raise ValueError("features: dates must be naive calendar month ends")
    if features.index.has_duplicates:
        raise ValueError("features: duplicate (date, asset) rows")
    unknown = set(features.index.get_level_values("asset")) - set(xr.columns)
    if unknown:
        raise ValueError(f"features: assets with no returns {sorted(unknown)}")
    non_float = [c for c in features.columns if not pd.api.types.is_float_dtype(features[c])]
    if non_float:
        raise ValueError(f"features: non-float columns {non_float}")
    return features.sort_index()


def _resolve(model) -> tuple[str, Callable[[], Forecaster]]:
    if isinstance(model, str):
        from ..models import registry
        return model, registry.get_factory(model)
    if callable(model):
        return getattr(model, "__name__", type(model).__name__), model
    raise TypeError("model must be a registry name or a zero-argument factory")


def walk_forward(model, features: pd.DataFrame, xr: pd.DataFrame, horizon: int,
                 settings: WalkForwardSettings | None = None, name: str | None = None
                 ) -> pd.DataFrame:
    """Run one model at one horizon; return forecasts in ``schema.FORECAST_COLUMNS``.

    ``model`` is a registry name or a zero-argument factory returning a fresh
    :class:`Forecaster`. ``xr`` is the wide monthly excess log return panel
    (``inputs.excess_returns(panel)``); ``features`` the long (date, asset) panel.
    """
    settings = settings or WalkForwardSettings()
    mname, factory = _resolve(model)
    name = name or mname
    features = check_inputs(features, xr)
    h = int(horizon)

    targets = schema.make_targets(xr, h).stack().dropna()
    y_aligned = targets.reindex(features.index)
    feat_ord = month_ordinal(features.index.get_level_values("date"))
    hist = xr.notna().cumsum()

    last = min(xr.index.max(), features.index.get_level_values("date").max())
    if settings.end is not None:
        last = min(last, pd.Timestamp(settings.end))
    origins = xr.index[(xr.index >= pd.Timestamp(settings.oos_start)) & (xr.index <= last)]

    origin_ord = month_ordinal(origins)
    model_obj = None
    fitted_at = None
    fitted_assets = None
    refit_every = None
    out = []
    out_idx = []
    n_fits = 0
    # features are sorted by date, so "rows dated <= d" is a prefix: slice, don't mask
    for i, t in enumerate(origins):
        h_t = hist.loc[t]
        elig = tuple(a for a in xr.columns if h_t[a] >= settings.min_history)
        if not elig:
            continue
        if refit_every is None:
            probe = factory()
            refit_every = int(probe.refit_every or settings.refit_every)
        t_ord = int(origin_ord[i])
        if model_obj is None or i - fitted_at >= refit_every or elig != fitted_assets:
            k = int(np.searchsorted(feat_ord, _training_cutoff(t_ord, h), side="right"))
            train = TrainingSet(
                horizon=h, origin=t, xr=xr.loc[:t], assets=elig,
                seed=int(config.SEED + 100_000 * h + t_ord) % (2**32),
                _features=features.iloc[:k], _targets=y_aligned.iloc[:k])
            m = factory()
            m.name = name
            try:
                m.fit(train)
            except InsufficientDataError as e:
                log.info("%s h=%d %s: skipped (%s)", name, h, t.date(), e)
                model_obj = None
                continue
            model_obj, fitted_at, fitted_assets = m, i, elig
            n_fits += 1
        lo, hi = np.searchsorted(feat_ord, [t_ord, t_ord + 1], side="left")
        idx = pd.MultiIndex.from_arrays([pd.DatetimeIndex([t] * len(elig)), list(elig)],
                                        names=["date", "asset"])
        Xp = features.iloc[lo:hi].reindex(idx)
        pred = model_obj.predict(Xp)
        pred = validate_predictions(pred, idx, where=f"{name} h={h} origin={t.date()}")
        out.append(pred.to_numpy(dtype=float))
        out_idx.append(idx)

    if not out:
        return pd.DataFrame(columns=schema.FORECAST_COLUMNS)
    from ..models.base import PREDICTION_COLUMNS
    full_idx = out_idx[0].append(out_idx[1:]) if len(out_idx) > 1 else out_idx[0]
    df = pd.DataFrame(np.vstack(out), index=full_idx, columns=PREDICTION_COLUMNS).reset_index()
    df.insert(0, "horizon", h)
    df.insert(0, "model", name)
    df["date"] = pd.to_datetime(df["date"])
    df["asset"] = df["asset"].astype(str)
    log.info("%s h=%d: %d forecasts, %d fits", name, h, len(df), n_fits)
    return df[schema.FORECAST_COLUMNS].reset_index(drop=True)


# --------------------------------------------------------------------------
# Cache
# --------------------------------------------------------------------------
def _default_results() -> Path:
    # imported late: the runner sets ASSETFORECAST_MAX_DATE before paths is imported
    from ... import paths
    return paths.RESULTS_RETURNS


def forecast_path(name: str, horizon: int, results_dir: Path | None = None) -> Path:
    root = Path(results_dir) if results_dir is not None else _default_results()
    return root / "forecasts" / name / f"h{int(horizon):02d}.parquet"


def data_fingerprint(features: pd.DataFrame, xr: pd.DataFrame) -> str:
    h = hashlib.sha256()
    for obj in (features, xr):
        h.update(pd.util.hash_pandas_object(obj, index=True).to_numpy().tobytes())
        h.update(",".join(map(str, obj.columns)).encode())
    return h.hexdigest()[:16]


def _source_hash(objs) -> str:
    h = hashlib.sha256()
    for o in objs:
        try:
            h.update(Path(inspect.getfile(o)).read_bytes())
        except (TypeError, OSError):
            h.update(repr(o).encode())
    return h.hexdigest()[:16]


def run_fingerprint(name: str, horizon: int, data_fp: str, settings: WalkForwardSettings,
                    factory: Callable[[], Forecaster]) -> dict:
    m = factory()
    return {
        "model": name, "horizon": int(horizon), "class": type(m).__qualname__,
        "params": json.loads(json.dumps(m.params(), default=str)),
        "data": data_fp, "settings": asdict(settings),
        "code": _source_hash([type(m), _base, sys.modules[__name__]]),
    }


def load_forecasts(names: Iterable[str], horizons: Iterable[int],
                   results_dir: Path | None = None, missing: str = "raise") -> pd.DataFrame:
    """Concatenate cached forecasts; ``missing='skip'`` ignores absent files."""
    frames = []
    for n in names:
        for h in horizons:
            p = forecast_path(n, h, results_dir)
            if not p.exists():
                if missing == "raise":
                    raise FileNotFoundError(f"no cached forecasts for {n} h={h} at {p}")
                continue
            frames.append(pd.read_parquet(p))
    if not frames:
        return pd.DataFrame(columns=schema.FORECAST_COLUMNS)
    df = pd.concat(frames, ignore_index=True)
    return df[schema.FORECAST_COLUMNS]


def run_cached(name: str, features: pd.DataFrame, xr: pd.DataFrame, horizon: int,
               settings: WalkForwardSettings | None = None, results_dir: Path | None = None,
               force: bool = False, data_fp: str | None = None,
               factory: Callable[[], Forecaster] | None = None) -> str:
    """Run (or reuse) the forecasts for one (model, horizon); return 'cached' or 'ran'."""
    settings = settings or WalkForwardSettings()
    if factory is None:
        _, factory = _resolve(name)
    data_fp = data_fp or data_fingerprint(features, xr)
    fp = run_fingerprint(name, horizon, data_fp, settings, factory)
    path = forecast_path(name, horizon, results_dir)
    meta_path = path.with_suffix(".json")
    if not force and path.exists() and meta_path.exists():
        old = json.loads(meta_path.read_text())
        if old.get("fingerprint") == fp:
            return "cached"
    t0 = time.time()
    df = walk_forward(factory, features, xr, horizon, settings, name=name)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.parquet")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)
    meta_path.write_text(json.dumps({
        "fingerprint": fp, "rows": len(df), "seconds": round(time.time() - t0, 2),
        "first_origin": str(df["date"].min().date()) if len(df) else None,
        "last_origin": str(df["date"].max().date()) if len(df) else None,
        "written": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
    }, indent=1, default=str))
    return "ran"


_WORKER: dict = {}


def _init_worker(features, xr):
    _WORKER.update(features=features, xr=xr)
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(1)
    except ImportError:  # pragma: no cover
        pass


def _job(name, factory, horizon, settings, results_dir, force, data_fp):
    status = run_cached(name, _WORKER["features"], _WORKER["xr"], horizon, settings,
                        results_dir, force, data_fp, factory=factory)
    return name, horizon, status


def run_many(names: Iterable[str], horizons: Iterable[int], features: pd.DataFrame,
             xr: pd.DataFrame, settings: WalkForwardSettings | None = None,
             results_dir: Path | None = None, n_jobs: int = 4, force: bool = False
             ) -> pd.DataFrame:
    """Run every (model, horizon) pair, in parallel processes when ``n_jobs > 1``.

    Workers are started with ``forkserver`` (``spawn`` where unavailable), not
    ``fork``: forking a parent that has used OpenMP (LightGBM) can deadlock.
    Each worker gets the data once and is limited to one BLAS/OpenMP thread;
    models with their own thread pools (e.g. LightGBM ``n_jobs``) should
    default to 1 thread. Factories are sent to workers by pickling, so they
    must be module-level classes/functions or ``functools.partial`` objects,
    not lambdas or closures; and a script that calls this must guard its entry
    point with ``if __name__ == "__main__":``. Returns a table of
    (model, horizon, status).
    """
    settings = settings or WalkForwardSettings()
    results_dir = Path(results_dir) if results_dir is not None else _default_results()  # resolve here, not in workers
    features = check_inputs(features, xr)
    data_fp = data_fingerprint(features, xr)
    jobs = [(n, int(h)) for n in names for h in horizons]
    from ..models import registry
    factories = {n: registry.get_factory(n) for n, _ in jobs}  # fail fast on unknown names
    use_pool = n_jobs > 1 and len(jobs) > 1
    if use_pool:
        for n, f in factories.items():
            try:
                pickle.dumps(f)
            except Exception as e:
                raise TypeError(f"factory for model {n!r} cannot be sent to a worker process "
                                f"({e}); register a module-level class/function or a "
                                f"functools.partial, or run with n_jobs=1") from None
    rows = []
    if not use_pool:
        for n, h in jobs:
            rows.append((n, h, run_cached(n, features, xr, h, settings, results_dir, force,
                                          data_fp, factory=factories[n])))
        return pd.DataFrame(rows, columns=["model", "horizon", "status"])
    method = "forkserver" if "forkserver" in mp.get_all_start_methods() else "spawn"
    ctx = mp.get_context(method)
    with ProcessPoolExecutor(max_workers=min(n_jobs, len(jobs)), mp_context=ctx,
                             initializer=_init_worker, initargs=(features, xr)) as ex:
        futs = [ex.submit(_job, n, factories[n], h, settings, results_dir, force, data_fp)
                for n, h in jobs]
        for f in futs:
            rows.append(f.result())
    return pd.DataFrame(rows, columns=["model", "horizon", "status"])
