"""Drawdown mathematics.

Sign convention, fixed once and used everywhere: :func:`max_drawdown` returns a
**negative** number (-0.55 == a 55% peak-to-trough loss), matching the ordinary
meaning of the word. :func:`depth` flips it positive for places where reasoning
about magnitudes is clearer -- chiefly the constraint evaluator. Mixing the two
conventions is the classic source of inverted-comparison bugs, so every function
here states which one it returns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252
WINDOW_3Y = 756


def wealth_index(returns: pd.Series | np.ndarray) -> np.ndarray:
    r = np.asarray(returns, dtype=float)
    return np.cumprod(1.0 + r)


def drawdown_series(returns: pd.Series) -> pd.Series:
    """Running drawdown path (<= 0) from a daily return series."""
    wealth = wealth_index(returns)
    peak = np.maximum.accumulate(wealth)
    return pd.Series(wealth / peak - 1.0, index=returns.index, name="drawdown")


def max_drawdown(returns: pd.Series | np.ndarray) -> float:
    """Worst peak-to-trough loss. Negative (or 0.0 for a monotone series)."""
    r = np.asarray(returns, dtype=float)
    if r.size == 0:
        return 0.0
    wealth = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(wealth)
    return float(np.min(wealth / peak - 1.0))


def depth(returns: pd.Series | np.ndarray) -> float:
    """Max drawdown as a positive magnitude."""
    return -max_drawdown(returns)


def rolling_max_drawdown(
    returns: pd.Series | np.ndarray,
    window: int = WINDOW_3Y,
    *,
    step: int = 1,
) -> np.ndarray:
    """Max drawdown *within* each rolling window, with the peak reset at the
    window start. Returns an array of negative values, one per window.

    ``step`` subsamples window start dates. The optimiser uses ``step=21``
    (monthly starts) because the full daily set is ~15,500 heavily overlapping
    windows costing O(n*window) per evaluation; final validation uses ``step=1``.
    """
    r = np.asarray(returns, dtype=float)
    n = r.size
    if n < window:
        return np.array([max_drawdown(r)]) if n else np.array([0.0])

    wealth = np.cumprod(1.0 + r)
    # Prepend 1.0 so a window can be normalised by the level *entering* it.
    padded = np.concatenate([[1.0], wealth])

    starts = np.arange(0, n - window + 1, step)
    out = np.empty(starts.size, dtype=float)
    for k, s in enumerate(starts):
        seg = padded[s + 1 : s + window + 1] / padded[s]
        peak = np.maximum.accumulate(seg)
        out[k] = np.min(seg / peak - 1.0)
    return out


def underwater_fraction(returns: pd.Series) -> float:
    """Fraction of days spent below the prior high-water mark."""
    return float((drawdown_series(returns) < -1e-12).mean())


def drawdown_episodes(returns: pd.Series, threshold: float = 0.10) -> pd.DataFrame:
    """Table of distinct drawdown episodes deeper than ``threshold``.

    Columns: peak date, trough date, recovery date (NaT if never), depth
    (negative), and durations in calendar days.
    """
    dd = drawdown_series(returns)
    wealth = pd.Series(wealth_index(returns), index=returns.index)
    peak = wealth.cummax()

    in_dd = dd < -1e-12
    episodes = []
    start = None
    for i, flag in enumerate(in_dd.to_numpy()):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            episodes.append((start, i))
            start = None
    if start is not None:
        episodes.append((start, len(dd) - 1))

    rows = []
    for s, e in episodes:
        seg = dd.iloc[s : e + 1]
        if -seg.min() < threshold:
            continue
        trough_pos = int(seg.to_numpy().argmin())
        trough_date = seg.index[trough_pos]
        peak_date = peak.index[max(s - 1, 0)]
        recovered = e < len(dd) - 1 or dd.iloc[e] >= -1e-12
        recovery_date = dd.index[e] if recovered else pd.NaT
        rows.append(
            {
                "peak": peak_date,
                "trough": trough_date,
                "recovery": recovery_date,
                "depth": float(seg.min()),
                "to_trough_days": (trough_date - peak_date).days,
                "to_recover_days": (
                    (recovery_date - peak_date).days if recovery_date is not pd.NaT else np.nan
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("depth").reset_index(drop=True)


def ulcer_index(returns: pd.Series) -> float:
    """RMS drawdown depth in percent -- penalises long shallow pain, not just depth."""
    dd = drawdown_series(returns).to_numpy() * 100.0
    return float(np.sqrt(np.mean(dd**2)))
