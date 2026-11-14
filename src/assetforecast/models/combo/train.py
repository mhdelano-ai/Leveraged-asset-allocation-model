"""Walk-forward training data for combo weights and level calibration.

Estimation dates. Weights and calibration constants are re-estimated once a
year, at the last December month-end ``E <= t`` (protocol: at most once a
year); a forecast at ``t`` uses the parameters estimated at ``E(t)``.

Training cells at ``E``: (origin ``s``, asset ``a``) with

* ``s + 12 months <= E`` (the target window is fully observed at ``E``),
* ``a`` eligible at ``s`` under the harness rule (>= 36 monthly returns in the
  view at ``s``) and at least 2 eligible assets at ``s`` (otherwise the
  harness stores no forecast there),
* a realized variance for ``(s, a)``.

The realized values are computed FROM THE VIEW, never from the cached target
files (which contain the future): :func:`assetforecast.eval.targets.compute_targets`
- the project's own estimator, daily-based 21-day realized variance with the
monthly proxy where daily coverage fails - applied to ``view.asof(E)`` treated
as a full panel. For ``s + 12 months <= E`` this is identical to the
project target, because the window lies inside the view.

The result is cached in the panel's shared cache dict under a key that holds
``E`` and a digest of the data up to ``E``, so that several combo instances
in one process (and the cold/warm paths) share it and a panel with different
data can never reuse it.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from ...eval.harness import MIN_ASSETS, MIN_MONTHS
from ...eval.targets import compute_targets, window_end
from ...universe import ASSET_IDS

POS = {a: i for i, a in enumerate(ASSET_IDS)}


def estimation_date(t) -> pd.Timestamp:
    """Last December month-end on or before ``t``."""
    t = pd.Timestamp(t)
    y = t.year if t.month == 12 and t == t + pd.offsets.MonthEnd(0) else t.year - 1
    return pd.Timestamp(year=y, month=12, day=31)


@dataclass(frozen=True)
class Training:
    """Realized variances at training origins (rows) x ASSET_IDS (columns)."""
    asof: pd.Timestamp
    origins: pd.DatetimeIndex
    realized: np.ndarray       # O x N realized variance, NaN where none
    cells: np.ndarray          # O x N bool: usable training cell (see module docstring)

    @property
    def n_years(self) -> float:
        """Training length in years: origins with at least one cell / 12."""
        return float(self.cells.any(axis=1).sum()) / 12.0


def _digest(view) -> str:
    h = hashlib.sha1()
    for a in (view.monthly, view.daily, view.cash_monthly, view.cash_daily):
        x = np.ascontiguousarray(a.to_numpy(float))
        h.update(str(x.shape).encode())
        h.update(np.nan_to_num(x, nan=-999.0).tobytes())
        h.update(np.asarray(a.index.asi8).tobytes())
    return h.hexdigest()


def training_data(view, E) -> Training:
    """Training cells and realized variances available at estimation date ``E``."""
    E = pd.Timestamp(E)
    if view.asof_date is not None and E > view.asof_date:
        raise ValueError(f"estimation date {E.date()} is after the view ({view.asof_date.date()})")
    v = view.asof(E)
    key = ("assetforecast.models.combo.training", E, _digest(v))
    shared = view._shared
    hit = shared.get(key)
    if hit is not None:
        return hit
    empty = Training(E, pd.DatetimeIndex([]), np.empty((0, len(ASSET_IDS))),
                     np.zeros((0, len(ASSET_IDS)), bool))
    if len(v.monthly) < 13:
        shared[key] = empty
        return empty
    T = compute_targets(replace(v, asof_date=None))
    tv = T.vol
    if tv.empty:
        shared[key] = empty
        return empty
    tv = tv[tv["origin"].map(window_end) <= E]
    origins = pd.DatetimeIndex(np.sort(tv["origin"].unique())).as_unit("ns")
    R = np.full((len(origins), len(ASSET_IDS)), np.nan)
    known = tv["asset"].isin(POS)
    tv = tv[known]
    R[origins.get_indexer(tv["origin"]), tv["asset"].map(POS).to_numpy(int)] = tv["var"].to_numpy(float)
    cnt = v.monthly.reindex(columns=ASSET_IDS).notna().cumsum()
    cnt = cnt.reindex(origins, method="ffill").fillna(0).to_numpy(float)
    elig = cnt >= MIN_MONTHS
    elig &= (elig.sum(axis=1) >= MIN_ASSETS)[:, None]
    cells = elig & np.isfinite(R) & (R > 0)
    out = Training(E, origins, R, cells)
    shared[key] = out
    return out
