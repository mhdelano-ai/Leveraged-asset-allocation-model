"""Purged time-series cross-validation for tuning *inside* a model's ``fit``.

Rows are (date, asset) observations whose target covers months
``date+1 .. date+h``. Two rows overlap when their target windows share a month,
so a training row dated ``s`` leaks into a validation block of dates
``[a, b]`` unless ``s + h <= a`` (it ends before the block's targets begin) or
``s >= b + h`` (it starts after the block's last target ends). The splitter
drops ("purges") every training row that fails this, plus an optional
``embargo`` of extra months after the block for features with long memory.

Two modes:

* ``"expanding"`` (default, recommended for hyperparameter tuning): the dates
  are cut into ``n_splits + 1`` contiguous blocks; fold k validates on block
  k+1 and trains on everything at least ``h`` months before it. Mirrors the
  outer walk-forward.
* ``"kfold"``: ``n_splits`` contiguous blocks; each block is validated by a
  model trained on all other blocks, purged on both sides. Gives every row an
  out-of-fold prediction (used for residual-based predictive distributions).
  Uses data after the validation block, which is fine inside a training window
  but means it should not be read as a pseudo out-of-sample result.

All indices are positional (``np.ndarray`` of int), usable with ``X.iloc`` and
with scikit-learn's ``cv=`` argument.
"""
from __future__ import annotations

from typing import Iterator

import numpy as np
import pandas as pd


def month_ordinal(dates) -> np.ndarray:
    """Integer month count (year*12 + month) so month arithmetic is exact."""
    d = pd.DatetimeIndex(dates)
    return (d.year * 12 + d.month).to_numpy(dtype=np.int64)


def _row_dates(X) -> pd.DatetimeIndex:
    if isinstance(X, (pd.DataFrame, pd.Series)):
        idx = X.index
        if isinstance(idx, pd.MultiIndex):
            return pd.DatetimeIndex(idx.get_level_values("date"))
        return pd.DatetimeIndex(idx)
    if isinstance(X, pd.MultiIndex):
        return pd.DatetimeIndex(X.get_level_values("date"))
    return pd.DatetimeIndex(X)


def purged_splits(
    X,
    horizon: int,
    n_splits: int = 5,
    mode: str = "expanding",
    embargo: int = 0,
    min_train_dates: int = 24,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Positional (train, validation) index pairs with overlap purging.

    ``X`` is a frame/series indexed by (date, asset) or by date, a MultiIndex,
    or an array of dates. Folds whose purged training set has fewer than
    ``min_train_dates`` distinct dates are skipped, so the result can have
    fewer than ``n_splits`` folds (possibly none: callers must handle that).
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    if mode not in ("expanding", "kfold"):
        raise ValueError("mode must be 'expanding' or 'kfold'")
    m = month_ordinal(_row_dates(X))
    udates = np.unique(m)
    n_blocks = n_splits + 1 if mode == "expanding" else n_splits
    if len(udates) < n_blocks:
        return []
    blocks = np.array_split(udates, n_blocks)
    val_blocks = blocks[1:] if mode == "expanding" else blocks
    out = []
    for blk in val_blocks:
        a, b = blk[0], blk[-1]
        val = (m >= a) & (m <= b)
        before = m + horizon <= a
        if mode == "expanding":
            train = before
        else:
            train = before | (m >= b + horizon + embargo)
        if len(np.unique(m[train])) < min_train_dates:
            continue
        out.append((np.flatnonzero(train), np.flatnonzero(val)))
    return out


class PurgedTimeSeriesSplit:
    """scikit-learn style splitter around :func:`purged_splits`.

    Example inside a model's ``fit``::

        cv = PurgedTimeSeriesSplit(horizon=train.horizon, n_splits=4)
        GridSearchCV(est, grid, cv=cv.split(X)).fit(X, y)
    """

    def __init__(self, horizon: int, n_splits: int = 5, mode: str = "expanding",
                 embargo: int = 0, min_train_dates: int = 24):
        self.horizon = horizon
        self.n_splits = n_splits
        self.mode = mode
        self.embargo = embargo
        self.min_train_dates = min_train_dates

    def split(self, X, y=None, groups=None) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        yield from purged_splits(X, self.horizon, self.n_splits, self.mode,
                                 self.embargo, self.min_train_dates)

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        if X is None:
            return self.n_splits
        return len(purged_splits(X, self.horizon, self.n_splits, self.mode,
                                 self.embargo, self.min_train_dates))
