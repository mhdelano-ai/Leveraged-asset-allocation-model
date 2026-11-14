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


@lru_cache(maxsize=1)
def _meta() -> pd.DataFrame:
    from ... import paths
    meta = pd.read_csv(paths.FEATURE_META).set_index("name")
    f = pd.read_parquet(paths.FEATURES)
    missing = set(config.FROZEN_FEATURE_COLUMNS) - set(meta.index)
    if missing:
        raise ValueError(f"pinned features missing from {paths.FEATURE_META}: {sorted(missing)}")
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
