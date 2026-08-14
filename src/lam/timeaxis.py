"""Unit-safe conversions from a DatetimeIndex to elapsed time.

pandas 3.0 stores datetimes at whatever resolution the source implied --
commonly ``datetime64[us]`` rather than the ``datetime64[ns]`` that older code
assumes. ``DatetimeIndex.asi8`` returns raw integers *in that unit*, so the
familiar ``np.diff(index.asi8) / 86_400e9`` idiom silently produces day counts
1000x too small on a microsecond index.

That failure is quiet and catastrophic here: it zeroes out every time-scaled
accrual in the project at once -- dividend income, bond coupons and the cost of
leverage -- while leaving price returns untouched, so the backtest still runs and
still looks plausible. Convert through an explicit unit instead of trusting the
index's own.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DAYS_PER_YEAR = 365.25
SECONDS_PER_DAY = 86_400.0


def as_days(index: pd.DatetimeIndex) -> np.ndarray:
    """Absolute calendar days since the epoch, resolution-independent."""
    seconds = index.to_numpy().astype("datetime64[s]").astype("int64")
    return seconds / SECONDS_PER_DAY


def day_deltas(index: pd.DatetimeIndex, *, prepend: float | None = None) -> np.ndarray:
    """Calendar days between consecutive entries.

    With ``prepend`` the result matches the index length, so it can be used
    directly alongside a same-length return series.
    """
    diffs = np.diff(as_days(index))
    if prepend is not None:
        diffs = np.concatenate([[float(prepend)], diffs])
    return diffs


def year_fractions(index: pd.DatetimeIndex, *, prepend: float | None = None) -> np.ndarray:
    """Calendar year fractions (ACT/365.25) between consecutive entries."""
    return day_deltas(index, prepend=prepend) / DAYS_PER_YEAR


def median_day_gap(index: pd.DatetimeIndex, first_n: int = 400) -> float:
    """Median spacing in days over the first ``first_n`` entries."""
    if len(index) < 2:
        return float("nan")
    head = index[: min(first_n, len(index))]
    return float(np.median(np.diff(as_days(head))))
