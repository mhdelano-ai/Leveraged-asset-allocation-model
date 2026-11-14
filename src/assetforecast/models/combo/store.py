"""Read other models' STORED forecasts, once per process, as dense arrays.

A combo model never re-runs its components: it reads what the harness stored
for them under ``<results>/forecasts/<kind>/<model>.parquet``. ``<results>``
is the results directory in use (``assetforecast.paths.RESULTS``, i.e.
``results/dev`` under ``ASSETFORECAST_MAX_DATE`` and ``results`` otherwise, or
``ASSETFORECAST_RESULTS``) unless the model instance is given an explicit
``results_dir``.

Each component is loaded once per process and per file version (path, mtime,
size) and kept as a dense array on the calendar month-end grid of its stored
origins:

    vol   V[o, a]     forecast VARIANCE (vol squared), NaN where not stored
    corr  C[o, a, b]  forecast correlation, NaN where not stored

Asset axis = ``universe.ASSET_IDS``. A component spec is a model name, or
``"cov:<name>"`` for the vols / correlations implied by a stored covariance
model (as in ``eval.harness.load_forecasts``).

Nothing is ever substituted: asking for an origin or an asset that a
component did not store raises :class:`MissingComponentError` naming the
component, the origin and the assets.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ...eval.harness import forecast_path, load_forecasts_long
from ...universe import ASSET_IDS

POS = {a: i for i, a in enumerate(ASSET_IDS)}
_CACHE: dict[tuple, "Stored"] = {}


class MissingComponentError(RuntimeError):
    """A combo needed a component forecast that is not stored."""


def results_dir_in_use(explicit=None) -> Path:
    if explicit is not None:
        return Path(explicit)
    from ... import paths          # read at call time (tests may point it elsewhere)
    return Path(paths.RESULTS)


def family_of(spec: str) -> str:
    """Model family = name prefix before the first underscore (``cov:`` stripped)."""
    name = spec[4:] if spec.startswith("cov:") else spec
    return name.split("_", 1)[0]


class Stored:
    """Dense forecasts of one component (see module docstring)."""

    def __init__(self, kind: str, spec: str, path: Path, origins: pd.DatetimeIndex, values: np.ndarray):
        self.kind, self.spec, self.path = kind, spec, path
        self.origins = origins
        self.values = values
        self._row = {o: k for k, o in enumerate(origins)}

    def row(self, origin) -> int:
        return self._row.get(pd.Timestamp(origin).as_unit("ns"), -1)

    def on_grid(self, grid: pd.DatetimeIndex) -> np.ndarray:
        """Values reindexed to ``grid`` (NaN where the origin is not stored)."""
        idx = self.origins.get_indexer(grid)
        out = np.full((len(grid),) + self.values.shape[1:], np.nan)
        ok = idx >= 0
        out[ok] = self.values[idx[ok]]
        return out

    def at(self, origin, assets: list[str]) -> np.ndarray:
        """Variances (vol) or the correlation matrix (corr) for ``assets`` at ``origin``."""
        k = self.row(origin)
        if k < 0:
            raise MissingComponentError(
                f"combo component {self.kind} model {self.spec!r} has no stored forecast at origin "
                f"{pd.Timestamp(origin).date()} (file {self.path}); run the component for that origin first")
        ix = [POS[a] for a in assets]
        if self.kind == "vol":
            v = self.values[k, ix]
            bad = [a for a, x in zip(assets, v) if not np.isfinite(x)]
        else:
            v = self.values[k][np.ix_(ix, ix)]
            bad = [a for i, a in enumerate(assets) if not np.isfinite(v[i]).all()]
        if bad:
            raise MissingComponentError(
                f"combo component {self.kind} model {self.spec!r} has no stored forecast for {bad} at "
                f"origin {pd.Timestamp(origin).date()} (file {self.path})")
        return v


def load(kind: str, spec: str, results_dir=None) -> Stored:
    """The stored forecasts of component ``spec`` for ``kind`` ('vol' or 'corr')."""
    if kind not in ("vol", "corr"):
        raise ValueError("combo components are vol or corr forecasts")
    base_kind, name = ("cov", spec[4:]) if spec.startswith("cov:") else (kind, spec)
    root = results_dir_in_use(results_dir)
    path = forecast_path(base_kind, name, root)
    if not path.exists():
        raise MissingComponentError(
            f"combo component {kind} model {spec!r} has no stored forecasts at {path}; "
            f"run it through the harness before the combo")
    st = path.stat()
    key = (str(path.resolve()), kind, spec, st.st_mtime_ns, st.st_size)
    hit = _CACHE.get(key)
    if hit is not None:
        return hit
    df = load_forecasts_long(kind, spec, root)
    origins = pd.DatetimeIndex(np.sort(df["origin"].unique())).as_unit("ns")
    oi = origins.get_indexer(df["origin"])
    N = len(ASSET_IDS)
    if kind == "vol":
        unknown = set(df["asset"]) - set(POS)
        if unknown:
            raise ValueError(f"component {spec!r}: unknown assets {sorted(unknown)}")
        vals = np.full((len(origins), N), np.nan)
        vals[oi, df["asset"].map(POS).to_numpy(int)] = df["vol"].to_numpy(float) ** 2
    else:
        unknown = (set(df["asset_i"]) | set(df["asset_j"])) - set(POS)
        if unknown:
            raise ValueError(f"component {spec!r}: unknown assets {sorted(unknown)}")
        vals = np.full((len(origins), N, N), np.nan)
        vals[oi, df["asset_i"].map(POS).to_numpy(int), df["asset_j"].map(POS).to_numpy(int)] = \
            df["corr"].to_numpy(float)
    out = Stored(kind, spec, path, origins, vals)
    # drop stale versions of the same file
    for k in [k for k in _CACHE if k[:3] == key[:3]]:
        del _CACHE[k]
    _CACHE[key] = out
    return out
