"""Named feature sets, so every model and ablation picks columns the same way."""
from __future__ import annotations

from functools import lru_cache

import pandas as pd

from .. import config, schema

STALE_MONTHS = 12  # a feature whose data stopped longer ago than this cannot be computed live


@lru_cache(maxsize=1)
def _meta() -> pd.DataFrame:
    meta = pd.read_csv(schema.FEATURE_META).set_index("name")
    f = schema.load_features()
    dates = f.index.get_level_values("date")
    last = pd.Series({c: dates[f[c].notna().to_numpy()].max() for c in f.columns})
    meta["last_date"] = last
    meta["discontinued"] = last < dates.max() - pd.offsets.MonthEnd(STALE_MONTHS)
    return meta


def feature_meta() -> pd.DataFrame:
    """feature_meta.csv plus `last_date` and `discontinued`."""
    return _meta().copy()


def columns(families: tuple[str, ...] | None = None, *, hindsight: bool = True,
            discontinued: bool = False) -> list[str]:
    """Feature names in panel order.

    families      keep only these (default: all of config.FEATURE_FAMILIES)
    hindsight     False drops features built from series estimated over the full
                  sample (the robustness check)
    discontinued  True keeps features whose source stopped publishing; the
                  default drops them, since a live forecast cannot use them
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
