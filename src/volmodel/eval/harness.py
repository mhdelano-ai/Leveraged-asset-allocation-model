"""Walk-forward harness: run a model over month-end origins, validate, store.

At each origin ``t`` (chronological): ``view = panel.asof(t)``,
``assets = view.eligible(min_months=36)`` (origins with < 2 eligible assets are
skipped), call the model, VALIDATE the output (see :func:`validate_forecast`)
and append it to the model's file:

    <results>/forecasts/vol/<model>.parquet    origin, asset, vol
    <results>/forecasts/corr/<model>.parquet   origin, asset_i, asset_j, corr   (full N x N)
    <results>/forecasts/cov/<model>.parquet    origin, asset_i, asset_j, cov    (full N x N)

plus a small ``<model>.meta.json`` sidecar (timings, last run). Matrices are
stored as the full square (diagonal included) in the order of ``assets``, so
they round-trip exactly.

Origins are calendar month-ends from the first monthly observation up to
:func:`~volmodel.eval.targets.observed_through` (the latest complete month,
which is the LIVE origin: its forecast is stored, its target does not exist
yet). ``step=k`` keeps month-ends whose ``(12*year + month) % k == 0`` (so
``step=12`` = every December) — anchored to the calendar, not to ``start``, so
quick runs with different ranges hit the same origins and resume cleanly. The
live origin is always included unless ``include_latest=False``.

Resuming: origins already stored are skipped unless ``force=True`` (which
recomputes the selected origins and keeps the rest). Progress is
checkpointed to disk every ``checkpoint_secs`` and when a model fails, so a
crash loses little work.

Public API::

    origins_for(panel, start=None, end=None, step=1, include_latest=True) -> list[Timestamp]
    validate_forecast(kind, out, assets, model, origin) -> Series | DataFrame
    run_model(model, kind, panel, *, results_dir=None, start=None, end=None, step=1,
              force=False, ...) -> RunResult
    run_models(kind, names, *, source="real"|"synthetic", jobs=1, results_dir=None, **kw)
    forecast_path(kind, model, results_dir=None) -> Path
    load_forecasts(kind, model, results_dir=None) -> {origin: Series | DataFrame}
    load_forecasts_long(kind, model, results_dir=None) -> DataFrame
    stored_models(kind, results_dir=None) -> list[str]

``model`` specs for loading: ``"name"`` reads ``forecasts/<kind>/name``;
``"cov:name"`` derives vols (kind="vol") or correlations (kind="corr") from
the stored covariance forecasts of cov model ``name``.
"""
from __future__ import annotations

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd

from ..data.panel import Panel
from ..paths import RESULTS
from .targets import _ts_ns, observed_through

KINDS = ("vol", "corr", "cov")
VALUE_COL = {"vol": "vol", "corr": "corr", "cov": "cov"}
MIN_MONTHS = 36
MIN_ASSETS = 2
SYM_RTOL = 1e-8
DIAG_TOL = 1e-8


class ForecastError(RuntimeError):
    """A model failed or returned an invalid forecast. Message names model, origin, problem."""


# ------------------------------------------------------------------ paths
def results_root(results_dir: Path | None = None) -> Path:
    return Path(results_dir) if results_dir is not None else RESULTS


def forecast_path(kind: str, model: str, results_dir: Path | None = None) -> Path:
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
    if not model or "/" in model or model.startswith("."):
        raise ValueError(f"invalid model name for a file: {model!r}")
    return results_root(results_dir) / "forecasts" / kind / f"{model}.parquet"


def stored_models(kind: str, results_dir: Path | None = None) -> list[str]:
    d = results_root(results_dir) / "forecasts" / kind
    return sorted(p.stem for p in d.glob("*.parquet")) if d.exists() else []


# ---------------------------------------------------------------- origins
def origins_for(panel: Panel, start=None, end=None, step: int = 1,
                include_latest: bool = True) -> list[pd.Timestamp]:
    """Month-end forecast origins, chronological (see module docstring)."""
    if step < 1:
        raise ValueError("step must be >= 1")
    through = observed_through(panel)
    first = pd.Timestamp(panel.monthly.index.min()) + pd.offsets.MonthEnd(0)
    grid = pd.date_range(first, through, freq="ME")
    lo = pd.Timestamp(start) if start is not None else grid[0]
    hi = pd.Timestamp(end) if end is not None else grid[-1]
    g = grid[(grid >= lo) & (grid <= hi)]
    keep = ((g.year * 12 + g.month) % step) == 0
    if include_latest and len(g) and g[-1] == through:
        keep[-1] = True
    return list(g[keep])


# ------------------------------------------------------------- validation
def _fail(model, origin, msg):
    raise ForecastError(f"model {model!r} at origin {pd.Timestamp(origin).date()}: {msg}")


def _check_labels(model, origin, what, got, assets):
    got = list(got)
    if got == list(assets):
        return
    missing = [a for a in assets if a not in got]
    extra = [a for a in got if a not in assets]
    if missing or extra:
        _fail(model, origin, f"{what} has wrong assets (missing {missing}, unexpected {extra})")
    _fail(model, origin, f"{what} is in the wrong order: got {got}, expected {list(assets)}")


def validate_forecast(kind: str, out, assets: list[str], model: str, origin):
    """Check a model output against the rules in models/base.py.

    Returns the cleaned object (float dtype; matrices exactly symmetrized,
    correlation diagonal set to exactly 1). Raises :class:`ForecastError`.
    """
    if kind == "vol":
        if not isinstance(out, pd.Series):
            _fail(model, origin, f"forecast_vol must return a pd.Series, got {type(out).__name__}")
        _check_labels(model, origin, "vol index", out.index, assets)
        try:
            v = out.to_numpy(dtype=float)
        except (TypeError, ValueError):
            _fail(model, origin, "vol values are not numeric")
        bad = [a for a, x in zip(assets, v) if not np.isfinite(x)]
        if bad:
            _fail(model, origin, f"non-finite vol for {bad}")
        bad = [a for a, x in zip(assets, v) if x <= 0]
        if bad:
            _fail(model, origin, f"vol must be > 0, got <= 0 for {bad}")
        return pd.Series(v, index=list(assets), name="vol")

    if kind not in ("corr", "cov"):
        raise ValueError(f"kind must be one of {KINDS}")
    if not isinstance(out, pd.DataFrame):
        _fail(model, origin, f"forecast_{kind} must return a pd.DataFrame, got {type(out).__name__}")
    _check_labels(model, origin, f"{kind} index", out.index, assets)
    _check_labels(model, origin, f"{kind} columns", out.columns, assets)
    try:
        a = out.to_numpy(dtype=float)
    except (TypeError, ValueError):
        _fail(model, origin, f"{kind} values are not numeric")
    if not np.isfinite(a).all():
        i, j = np.argwhere(~np.isfinite(a))[0]
        _fail(model, origin, f"non-finite {kind} entry at ({assets[i]}, {assets[j]})")
    asym = np.abs(a - a.T).max()
    if asym > SYM_RTOL * max(1.0, np.abs(a).max()):
        _fail(model, origin, f"{kind} matrix not symmetric (max |A - A'| = {asym:.3g})")
    a = (a + a.T) / 2
    if kind == "corr":
        dev = np.abs(np.diag(a) - 1).max()
        if dev > DIAG_TOL:
            _fail(model, origin, f"corr diagonal not 1 (max deviation {dev:.3g})")
        np.fill_diagonal(a, 1.0)
    else:
        bad = [x for x, dd in zip(assets, np.diag(a)) if dd <= 0]
        if bad:
            _fail(model, origin, f"cov variance must be > 0, got <= 0 for {bad}")
    ev = np.linalg.eigvalsh(a)
    if ev.min() <= 0:
        _fail(model, origin, f"{kind} matrix not positive definite (min eigenvalue {ev.min():.3g})")
    return pd.DataFrame(a, index=list(assets), columns=list(assets))


# --------------------------------------------------------------- long I/O
def _to_long(kind: str, origin: pd.Timestamp, f) -> pd.DataFrame:
    if kind == "vol":
        return pd.DataFrame({"origin": origin, "asset": f.index.astype("str"), "vol": f.to_numpy()})
    n = len(f)
    names = np.asarray(f.index, dtype=object)
    return pd.DataFrame({
        "origin": origin,
        "asset_i": np.repeat(names, n).astype("str"),
        "asset_j": np.tile(names, n).astype("str"),
        VALUE_COL[kind]: f.to_numpy().ravel(),
    })


def _normalize_long(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["origin"] = _ts_ns(df["origin"]).to_numpy()
    for c in ("asset", "asset_i", "asset_j"):
        if c in df.columns:
            df[c] = df[c].astype("str")
    return df


def _write_atomic(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)


def _read_stored(kind: str, model: str, results_dir) -> pd.DataFrame | None:
    p = forecast_path(kind, model, results_dir)
    return _normalize_long(pd.read_parquet(p)) if p.exists() else None


def load_forecasts_long(kind: str, model: str, results_dir: Path | None = None) -> pd.DataFrame:
    """Stored forecasts in long format. ``model="cov:<name>"`` derives from a cov model."""
    if model.startswith("cov:") and kind in ("vol", "corr"):
        cov = load_forecasts_long("cov", model[4:], results_dir)
        return cov_long_to_vol(cov) if kind == "vol" else cov_long_to_corr(cov)
    df = _read_stored(kind, model, results_dir)
    if df is None:
        raise FileNotFoundError(f"no stored {kind} forecasts for {model!r} at "
                                f"{forecast_path(kind, model, results_dir)}")
    return df


def cov_long_to_vol(cov: pd.DataFrame) -> pd.DataFrame:
    d = cov[cov["asset_i"] == cov["asset_j"]]
    return pd.DataFrame({"origin": d["origin"].to_numpy(), "asset": d["asset_i"].to_numpy(),
                         "vol": np.sqrt(d["cov"].to_numpy())}).reset_index(drop=True)


def cov_long_to_corr(cov: pd.DataFrame) -> pd.DataFrame:
    v = cov[cov["asset_i"] == cov["asset_j"]][["origin", "asset_i", "cov"]]
    v = v.rename(columns={"cov": "_v"})
    out = cov.merge(v, on=["origin", "asset_i"], how="left") \
        .merge(v.rename(columns={"asset_i": "asset_j", "_v": "_w"}), on=["origin", "asset_j"], how="left")
    corr = out["cov"].to_numpy() / np.sqrt(out["_v"].to_numpy() * out["_w"].to_numpy())
    corr = np.where(out["asset_i"].to_numpy() == out["asset_j"].to_numpy(), 1.0, corr)
    return pd.DataFrame({"origin": out["origin"].to_numpy(), "asset_i": out["asset_i"].to_numpy(),
                         "asset_j": out["asset_j"].to_numpy(), "corr": corr})


def long_to_dict(kind: str, df: pd.DataFrame) -> dict:
    """{origin: Series (vol) | DataFrame (corr/cov)} with the stored asset order."""
    out = {}
    if kind == "vol":
        for o, g in df.groupby("origin", sort=True):
            out[pd.Timestamp(o)] = pd.Series(g["vol"].to_numpy(), index=list(g["asset"]), name="vol")
        return out
    col = VALUE_COL[kind]
    for o, g in df.groupby("origin", sort=True):
        names = list(dict.fromkeys(g["asset_i"]))
        n = len(names)
        if len(g) == n * n and list(g["asset_j"].iloc[:n]) == names:
            m = g[col].to_numpy().reshape(n, n)
            out[pd.Timestamp(o)] = pd.DataFrame(m, index=names, columns=names)
        else:  # not in stored square order: pivot
            w = g.pivot(index="asset_i", columns="asset_j", values=col)
            out[pd.Timestamp(o)] = w.reindex(index=names, columns=names)
    return out


def load_forecasts(kind: str, model: str, results_dir: Path | None = None) -> dict:
    """Stored forecasts as {origin: pd.Series (vol) or pd.DataFrame (corr/cov)}."""
    return long_to_dict(kind, load_forecasts_long(kind, model, results_dir))


# ---------------------------------------------------------------- running
@dataclass
class RunResult:
    model: str
    kind: str
    path: Path
    n_new: int = 0
    n_skipped_stored: int = 0
    n_skipped_few_assets: int = 0
    seconds: float = 0.0
    model_seconds: float = 0.0
    error: str | None = None
    timings: list = field(default_factory=list, repr=False)

    def __str__(self):
        s = (f"{self.kind}/{self.model}: {self.n_new} new origins, {self.n_skipped_stored} already stored, "
             f"{self.n_skipped_few_assets} skipped (<{MIN_ASSETS} eligible) in {self.seconds:.1f}s "
             f"(model {self.model_seconds:.1f}s)")
        return s + (f"  FAILED: {self.error}" if self.error else "")


def _call(model, kind, view, assets):
    if kind == "vol":
        return model.forecast_vol(view, assets)
    if kind == "corr":
        return model.forecast_corr(view, assets)
    return model.forecast_cov(view, assets)


def run_model(model, kind: str, panel: Panel, *, results_dir: Path | None = None,
              start=None, end=None, step: int = 1, force: bool = False,
              include_latest: bool = True, min_months: int = MIN_MONTHS,
              checkpoint_secs: float = 60.0, progress: bool = False,
              write: bool = True) -> RunResult:
    """Walk ``model`` forward over the selected origins and store its forecasts.

    ``model`` is a model instance or a registered name (looked up for ``kind``).
    Raises :class:`ForecastError` (after checkpointing completed origins) if the
    model raises or returns an invalid forecast.
    """
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}")
    if isinstance(model, str):
        from ..models.base import get_model
        model = get_model(kind, model)
    name = getattr(model, "name", None) or type(model).__name__
    path = forecast_path(kind, name, results_dir)
    res = RunResult(model=name, kind=kind, path=path)
    t0 = time.perf_counter()

    origins = origins_for(panel, start, end, step, include_latest)
    stored = _read_stored(kind, name, results_dir) if write else None
    if stored is not None and force:
        sel = set(_ts_ns(pd.Series(origins)).tolist())
        stored = stored[~stored["origin"].isin(sel)]
    done = set(stored["origin"].unique()) if stored is not None else set()
    parts = [stored] if stored is not None and len(stored) else []
    new_parts: list[pd.DataFrame] = []
    last_ckpt = time.perf_counter()

    def flush():
        if write and new_parts:
            df = pd.concat(parts + new_parts, ignore_index=True)
            df = _normalize_long(df).sort_values(["origin"], kind="stable").reset_index(drop=True)
            _write_atomic(df, path)

    try:
        for t in origins:
            o = _ts_ns(t)
            if o in done:
                res.n_skipped_stored += 1
                continue
            view = panel.asof(t)
            assets = view.eligible(min_months=min_months)
            if len(assets) < MIN_ASSETS:
                res.n_skipped_few_assets += 1
                continue
            m0 = time.perf_counter()
            try:
                out = _call(model, kind, view, list(assets))
            except Exception as exc:
                raise ForecastError(f"model {name!r} at origin {o.date()}: raised "
                                    f"{type(exc).__name__}: {exc}") from exc
            dt = time.perf_counter() - m0
            res.model_seconds += dt
            res.timings.append((o, dt))
            f = validate_forecast(kind, out, list(assets), name, o)
            new_parts.append(_to_long(kind, o, f))
            res.n_new += 1
            if progress and res.n_new % 50 == 0:
                print(f"  {kind}/{name}: {o.date()} ({res.n_new} done)", flush=True)
            if time.perf_counter() - last_ckpt > checkpoint_secs:
                flush()
                last_ckpt = time.perf_counter()
    except ForecastError as exc:
        res.error = str(exc)
        flush()
        res.seconds = time.perf_counter() - t0
        _write_meta(res, write)
        raise
    flush()
    res.seconds = time.perf_counter() - t0
    _write_meta(res, write)
    return res


def _write_meta(res: RunResult, write: bool) -> None:
    if not write:
        return
    p = res.path.with_suffix(".meta.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({
        "model": res.model, "kind": res.kind, "run_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "n_new": res.n_new, "n_skipped_stored": res.n_skipped_stored,
        "seconds": round(res.seconds, 3), "model_seconds": round(res.model_seconds, 3),
        "mean_model_ms_per_origin": round(1000 * res.model_seconds / max(res.n_new, 1), 3),
        "error": res.error,
    }, indent=2))


# --------------------------------------------------------- many models
def _panel_from_source(source: str, seed: int = 0) -> Panel:
    if source == "synthetic":
        from ..data.synthetic import make_synthetic_panel
        return make_synthetic_panel(seed=seed)
    if source == "real":
        from ..data.panel import load_panel
        return load_panel()
    raise ValueError(f"source must be 'real' or 'synthetic', got {source!r}")


def _worker(kind, name, source, seed, kw):
    panel = _panel_from_source(source, seed)
    try:
        return run_model(name, kind, panel, **kw)
    except ForecastError as exc:
        return RunResult(model=name, kind=kind, path=forecast_path(kind, name, kw.get("results_dir")),
                         error=str(exc))
    except Exception as exc:  # unexpected (e.g. model constructor failed)
        return RunResult(model=name, kind=kind, path=forecast_path(kind, name, kw.get("results_dir")),
                         error=f"{type(exc).__name__}: {exc}")


def run_models(kind: str, names: list[str], *, source: str = "real", seed: int = 0,
               jobs: int = 1, **kw) -> list[RunResult]:
    """Run several REGISTERED models, in parallel across models (processes).

    Each model runs in one process and sees its origins in chronological
    order. Failures are collected (``RunResult.error``), not raised, so one
    bad model does not stop the others. ``kw`` is passed to :func:`run_model`.
    """
    if jobs <= 1 or len(names) <= 1:
        return [_worker(kind, n, source, seed, kw) for n in names]
    out = []
    ctx = get_context("spawn")  # safe with BLAS threads; workers re-import cleanly
    with ProcessPoolExecutor(max_workers=jobs, mp_context=ctx) as ex:
        futs = {ex.submit(_worker, kind, n, source, seed, kw): n for n in names}
        for f in as_completed(futs):
            out.append(f.result())
    order = {n: i for i, n in enumerate(names)}
    return sorted(out, key=lambda r: order.get(r.model, 0))
