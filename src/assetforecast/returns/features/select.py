"""Named feature sets, so every model and ablation picks columns the same way.

Ported from ``assetpred.features.select``. One change: which features count as
``discontinued`` is pinned (``config.FROZEN_FEATURE_COLUMNS`` are the live
ones) instead of being recomputed from the panel's last date. The old rule
("data stopped more than ``STALE_MONTHS`` before the panel's last date") would
silently change the frozen models' feature set whenever the panel is cut
earlier (``ASSETFORECAST_MAX_DATE``) or grows later. The old rule is still
evaluated at ``FROZEN_REFERENCE_DATE``, the last date of assetpred's archived
panel, whenever the features reach it, and must agree with the pinned set there.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

from .. import config

STALE_MONTHS = 12  # a feature whose data stopped longer ago than this cannot be computed live
# The last date of assetpred's archived feature panel, at which the pinned set
# was what the old rule returned.
FROZEN_REFERENCE_DATE = pd.Timestamp("2026-10-31")


def stale_by_old_rule(features: pd.DataFrame, as_of=None) -> pd.Series:
    """assetpred's rule: True where a feature's last non-missing date (on rows
    dated <= ``as_of``, default the panel's last date) is more than
    ``STALE_MONTHS`` before ``as_of``."""
    dates = features.index.get_level_values("date")
    as_of = dates.max() if as_of is None else pd.Timestamp(as_of)
    keep = (dates <= as_of)
    f, d = features[keep], dates[keep]
    last = pd.Series({c: d[f[c].notna().to_numpy()].max() for c in f.columns})
    return last < as_of - pd.offsets.MonthEnd(STALE_MONTHS)


def check_frozen_columns(features: pd.DataFrame) -> None:
    """The pinned set must be what the old rule gives at ``FROZEN_REFERENCE_DATE``
    (checked only when the features reach that date)."""
    if features.index.get_level_values("date").max() < FROZEN_REFERENCE_DATE:
        return
    old = stale_by_old_rule(features, FROZEN_REFERENCE_DATE)
    live_old = [c for c in features.columns if not old[c]]
    if tuple(live_old) != tuple(config.FROZEN_FEATURE_COLUMNS):
        raise AssertionError(
            "config.FROZEN_FEATURE_COLUMNS disagrees with the old discontinued rule at "
            f"{FROZEN_REFERENCE_DATE.date()}: only pinned {sorted(set(config.FROZEN_FEATURE_COLUMNS) - set(live_old))}, "
            f"only old rule {sorted(set(live_old) - set(config.FROZEN_FEATURE_COLUMNS))}")


class FeatureMetaMissing(RuntimeError):
    """The feature panel or its metadata is not on disk. Deliberately not an
    ``OSError``: a caller must not quietly fall back to a different column list."""


# (meta_path, features_path) override; None means paths.FEATURE_META / paths.FEATURES
_SOURCE: tuple | None = None


def use_source(meta_path=None, features_path=None) -> None:
    """Point the selector at another feature panel (e.g. assetpred's archived
    ``feature_meta.csv`` and ``features.parquet``); no arguments restores the default."""
    global _SOURCE
    _SOURCE = None if meta_path is None and features_path is None else (meta_path, features_path)
    _meta.cache_clear()


def _source():
    from ... import paths
    m, f = _SOURCE if _SOURCE is not None else (None, None)
    return (m or paths.FEATURE_META), (f or paths.FEATURES)


@lru_cache(maxsize=1)
def _meta() -> pd.DataFrame:
    meta_path, features_path = _source()
    for p in (meta_path, features_path):
        if not Path(p).exists():
            raise FeatureMetaMissing(
                f"{p} not found: build it with `python -m assetforecast.returns.features.build`. "
                "The frozen feature sets need its families and hindsight flags.")
    meta = pd.read_csv(meta_path).set_index("name")
    f = pd.read_parquet(features_path)
    missing = set(config.FROZEN_FEATURE_COLUMNS) - set(meta.index)
    if missing:
        raise ValueError(f"pinned features missing from {meta_path}: {sorted(missing)}")
    check_frozen_columns(f)
    dates = f.index.get_level_values("date")
    last = pd.Series({c: dates[f[c].notna().to_numpy()].max() for c in f.columns})
    meta["last_date"] = last
    meta["discontinued"] = ~meta.index.isin(config.FROZEN_FEATURE_COLUMNS)
    return meta


def feature_meta() -> pd.DataFrame:
    """feature_meta.csv plus `last_date` and `discontinued` (= not in the pinned set)."""
    return _meta().copy()


def columns(families: tuple[str, ...] | None = None, *, hindsight: bool = True,
            discontinued: bool = False) -> list[str]:
    """Feature names in panel order.

    families      keep only these (default: all of config.FEATURE_FAMILIES)
    hindsight     False drops features built from series estimated over the full
                  sample (the robustness check)
    discontinued  True keeps features whose source stopped publishing; the
                  default drops them, since a live forecast cannot use them.
                  Which are discontinued is pinned: ``config.FROZEN_FEATURE_COLUMNS``.
    """
    m = _meta()
    fams = config.FEATURE_FAMILIES if families is None else families
    bad = set(fams) - set(config.FEATURE_FAMILIES)
    if bad:
        raise ValueError(f"unknown families {sorted(bad)}")
    keep = m["family"].isin(fams)
    if not hindsight:
        keep &= ~m["hindsight"].astype(bool)
    if not discontinued:
        keep &= ~m["discontinued"]
    return list(m.index[keep])
