"""Walk-forward engine for covariance forecasts, mirroring ``evaluation.walkforward``.

At each monthly origin t from ``oos_start`` (to ``end``):

* asset a is eligible if it has at least ``min_history`` months of its own
  excess returns through t (``config.MIN_ASSET_HISTORY``), so the matrix grows
  as assets enter (em_eq 1994, tips 2004, gold 2009);
* the model gets a :class:`CovTrainingSet` holding monthly returns (and daily
  returns / features if it uses them) dated <= t only; it is refit at the first
  origin, every ``refit_every`` origins, and when the eligible set changes;
* the prediction is multiplied by the mean-uncertainty factor
  (``linalg.mean_uncertainty_factor``), symmetrized, eigenvalue-floored and
  checked: every matrix written is symmetric PSD.

Output: long frame in ``schema.COV_COLUMNS`` with the full n x n matrix per
origin (both triangles). Cache: ``<results>/covariances/<model>/hXX.parquet``
with a JSON fingerprint of data, settings, parameters and this package's
source.
"""
from __future__ import annotations

import hashlib
import json
import logging
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from .. import config, schema
from . import linalg as la
from .base import CovContractError, CovForecaster, CovTrainingSet, get_factory

log = logging.getLogger(__name__)
PSD_TOL = 1e-12


@dataclass(frozen=True)
class CovSettings:
    oos_start: str = config.OOS_START
    min_history: int = config.MIN_ASSET_HISTORY
    end: str | None = None          # last origin (default: last month of the panel)
    mean_adjust: bool = True        # add estimation error of the expanding mean


def dev_end(horizon: int) -> str:
    """Last origin whose h-month target is realized by ``config.DEV_END``."""
    return str((pd.Timestamp(config.DEV_END) - pd.offsets.MonthEnd(int(horizon))).date())


def check_matrix(S: np.ndarray, where: str = "") -> None:
    S = np.asarray(S, dtype=float)
    if S.ndim != 2 or S.shape[0] != S.shape[1]:
        raise CovContractError(f"{where}: not a square matrix {S.shape}")
    if not np.isfinite(S).all():
        raise CovContractError(f"{where}: non-finite entries")
    if not np.array_equal(S, S.T):
        raise CovContractError(f"{where}: not symmetric")
    scale = max(float(np.trace(S)) / len(S), 1e-300)
    if la.min_eig(S) < -PSD_TOL * scale:
        raise CovContractError(f"{where}: not PSD (min eig {la.min_eig(S):.3g})")


def _resolve(model) -> tuple[str, Callable[[], CovForecaster]]:
    if isinstance(model, str):
        return model, get_factory(model)
    if callable(model):
        return getattr(model, "__name__", type(model).__name__), model
    raise TypeError("model must be a registry name or a zero-argument factory")


def walk_forward_cov(model, xr: pd.DataFrame, horizon: int, daily: pd.DataFrame | None = None,
                     settings: CovSettings | None = None, name: str | None = None,
                     features: pd.DataFrame | None = None) -> pd.DataFrame:
    settings = settings or CovSettings()
    mname, factory = _resolve(model)
    name = name or mname
    schema._check_month_end(xr.index, "excess returns")
    h = int(horizon)
    hist = xr.notna().cumsum()
    last = xr.index.max()
    if settings.end is not None:
        last = min(last, pd.Timestamp(settings.end))
    origins = xr.index[(xr.index >= pd.Timestamp(settings.oos_start)) & (xr.index <= last)]
    probe = factory()
    uses_daily = bool(getattr(probe, "uses_daily", False)) and daily is not None
    uses_feat = bool(getattr(probe, "uses_features", False)) and features is not None
    if uses_daily:
        daily = daily.sort_index()
        dpos = daily.index.searchsorted(origins + pd.Timedelta(days=1), side="left")
    if uses_feat:
        features = features.sort_index()
        fdates = features.index.get_level_values("date")
    model_obj, fitted_at, fitted_assets = None, None, None
    blocks = []
    n_fits = 0
    for i, t in enumerate(origins):
        h_t = hist.loc[t]
        elig = tuple(a for a in xr.columns if h_t[a] >= settings.min_history)
        if len(elig) < 1:
            continue
        train = CovTrainingSet(
            horizon=h, origin=t, xr=xr.loc[:t], assets=elig,
            daily=daily.iloc[:dpos[i]] if uses_daily else None,
            features=features[fdates <= t] if uses_feat else None)
        refit = int(getattr(probe, "refit_every", 1) or 1)
        if model_obj is None or i - fitted_at >= refit or elig != fitted_assets:
            m = factory()
            m.name = name
            m.fit(train)
            model_obj, fitted_at, fitted_assets = m, i, elig
            n_fits += 1
        S = np.asarray(model_obj.predict(train), dtype=float)
        where = f"{name} h={h} origin={t.date()}"
        if S.shape != (len(elig), len(elig)) or not np.isfinite(S).all():
            raise CovContractError(f"{where}: bad prediction shape {S.shape} or non-finite values")
        if settings.mean_adjust:
            S = S * la.mean_uncertainty_factor(train.counts(), h)
        S = la.make_psd(S)
        check_matrix(S, where)
        n = len(elig)
        blocks.append(pd.DataFrame({
            "date": t, "asset_i": np.repeat(elig, n), "asset_j": np.tile(elig, n),
            "cov": S.ravel()}))
    if not blocks:
        return pd.DataFrame(columns=schema.COV_COLUMNS)
    df = pd.concat(blocks, ignore_index=True)
    df.insert(0, "horizon", h)
    df.insert(0, "model", name)
    df["date"] = pd.to_datetime(df["date"])
    log.info("%s h=%d: %d origins, %d fits", name, h, df["date"].nunique(), n_fits)
    return df[schema.COV_COLUMNS]


# --------------------------------------------------------------------------
# Long frame <-> matrices
# --------------------------------------------------------------------------
def to_matrices(df: pd.DataFrame) -> dict:
    """{date: (assets tuple, n x n array)} for one (model, horizon) frame."""
    out = {}
    for t, g in df.groupby("date", sort=True):
        ai = g["asset_i"].to_numpy().astype(str)
        aj = g["asset_j"].to_numpy().astype(str)
        assets = tuple(dict.fromkeys(ai))
        n = len(assets)
        if len(g) == n * n and (ai == np.repeat(assets, n)).all() and (aj == np.tile(assets, n)).all():
            S = g["cov"].to_numpy(float).reshape(n, n)       # fast path: written by this package
        else:
            piv = g.pivot(index="asset_i", columns="asset_j", values="cov")
            if piv.shape != (n, n) or piv.isna().any().any():
                raise CovContractError(f"date {pd.Timestamp(t).date()}: incomplete matrix")
            S = piv.loc[list(assets), list(assets)].to_numpy(float)
        out[pd.Timestamp(t)] = (assets, S)
    return out


def from_matrices(mats: dict, model: str, horizon: int) -> pd.DataFrame:
    rows = []
    for t, (assets, S) in sorted(mats.items()):
        n = len(assets)
        rows.append(pd.DataFrame({"date": t, "asset_i": np.repeat(assets, n),
                                  "asset_j": np.tile(assets, n), "cov": np.asarray(S).ravel()}))
    df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["date", "asset_i", "asset_j", "cov"])
    df.insert(0, "horizon", int(horizon))
    df.insert(0, "model", model)
    return df[schema.COV_COLUMNS]


def validate_cov_frame(df: pd.DataFrame) -> None:
    """Every (model, horizon, date) block is a full symmetric PSD matrix."""
    if list(df.columns) != schema.COV_COLUMNS:
        raise CovContractError(f"columns {list(df.columns)} != {schema.COV_COLUMNS}")
    for (m, h), g in df.groupby(["model", "horizon"]):
        for t, (assets, S) in to_matrices(g).items():
            check_matrix(S, f"{m} h={h} {t.date()}")


# --------------------------------------------------------------------------
# Cache and parallel runs
# --------------------------------------------------------------------------
def cov_path(name: str, horizon: int, results_dir: Path | None = None) -> Path:
    root = Path(results_dir) if results_dir is not None else config.RESULTS
    return root / "covariances" / name / f"h{int(horizon):02d}.parquet"


def data_fingerprint(*frames) -> str:
    h = hashlib.sha256()
    for obj in frames:
        if obj is None:
            h.update(b"none")
            continue
        h.update(pd.util.hash_pandas_object(obj, index=True).to_numpy().tobytes())
        h.update(",".join(map(str, obj.columns)).encode())
    return h.hexdigest()[:16]


def code_fingerprint() -> str:
    h = hashlib.sha256()
    for p in sorted(Path(__file__).parent.glob("*.py")):
        if p.name in ("run.py", "scoring.py", "combine.py"):
            continue
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def run_cached(name: str, xr, horizon: int, daily=None, settings: CovSettings | None = None,
               results_dir: Path | None = None, force: bool = False, data_fp: str | None = None,
               features=None) -> str:
    settings = settings or CovSettings()
    factory = get_factory(name)
    fp = {"model": name, "horizon": int(horizon), "params": json.loads(json.dumps(factory().params(), default=str)),
          "data": data_fp or data_fingerprint(xr, daily, features), "settings": asdict(settings),
          "code": code_fingerprint()}
    path = cov_path(name, horizon, results_dir)
    meta = path.with_suffix(".json")
    if not force and path.exists() and meta.exists():
        if json.loads(meta.read_text()).get("fingerprint") == fp:
            return "cached"
    t0 = time.time()
    df = walk_forward_cov(name, xr, horizon, daily, settings, features=features)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.parquet")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)
    meta.write_text(json.dumps({
        "fingerprint": fp, "rows": len(df), "seconds": round(time.time() - t0, 2),
        "first_origin": str(df["date"].min().date()) if len(df) else None,
        "last_origin": str(df["date"].max().date()) if len(df) else None,
        "written": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds")}, indent=1, default=str))
    return "ran"


_WORKER: dict = {}


def _init_worker(xr, daily, features):
    _WORKER.update(xr=xr, daily=daily, features=features)
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(1)
    except ImportError:  # pragma: no cover
        pass


def _job(name, horizon, settings, results_dir, force, data_fp):
    return name, horizon, run_cached(name, _WORKER["xr"], horizon, _WORKER["daily"], settings,
                                     results_dir, force, data_fp, _WORKER["features"])


def run_many(names: Iterable[str], horizons: Iterable[int], xr, daily=None,
             settings_for: Callable[[int], CovSettings] | None = None,
             results_dir: Path | None = None, n_jobs: int = 2, force: bool = False,
             features=None) -> pd.DataFrame:
    settings_for = settings_for or (lambda h: CovSettings())
    results_dir = Path(results_dir) if results_dir is not None else config.RESULTS
    jobs = [(n, int(h)) for n in names for h in horizons]
    for n, _ in jobs:
        get_factory(n)
    data_fp = data_fingerprint(xr, daily, features)
    if n_jobs <= 1 or len(jobs) <= 1:
        rows = [(n, h, run_cached(n, xr, h, daily, settings_for(h), results_dir, force, data_fp, features))
                for n, h in jobs]
        return pd.DataFrame(rows, columns=["model", "horizon", "status"])
    method = "forkserver" if "forkserver" in mp.get_all_start_methods() else "spawn"
    with ProcessPoolExecutor(max_workers=min(n_jobs, len(jobs)), mp_context=mp.get_context(method),
                             initializer=_init_worker, initargs=(xr, daily, features)) as ex:
        futs = [ex.submit(_job, n, h, settings_for(h), results_dir, force, data_fp) for n, h in jobs]
        rows = [f.result() for f in futs]
    return pd.DataFrame(rows, columns=["model", "horizon", "status"])


def load_covariances(names, horizons, results_dir: Path | None = None) -> pd.DataFrame:
    frames = []
    for n in names:
        for h in horizons:
            p = cov_path(n, h, results_dir)
            if not p.exists():
                raise FileNotFoundError(f"no cached covariances for {n} h={h} at {p}")
            frames.append(pd.read_parquet(p))
    return pd.concat(frames, ignore_index=True)[schema.COV_COLUMNS]
