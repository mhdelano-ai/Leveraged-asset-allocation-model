"""Loss functions and descriptive accuracy measures (no hypothesis tests; those
are in ``inference``).

Conventions: ``y`` is the realized h-month excess log return, ``mean`` a point
forecast, ``Q`` an (n x K) array of forecast quantiles on ``config.QUANTILES``.
All functions take numpy arrays or Series and return numpy arrays or floats.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config, schema

TAUS = np.asarray(config.QUANTILES, dtype=float)
QCOLS = list(schema.QUANTILE_COLUMNS)
# central intervals: nominal coverage -> (lower, upper) quantile columns
INTERVALS = {50: ("q25", "q75"), 80: ("q10", "q90"), 90: ("q05", "q95"), 98: ("q01", "q99")}
# tail quantiles tested for hit frequency; for tau > 0.5 a "hit" is y above q_tau
TAIL_QUANTILES = ("q01", "q05", "q95", "q99")

# CRPS = 2 * integral_0^1 pinball_tau d tau, approximated on the quantile grid
# with trapezoid weights (tau_0 = 0, tau_{K+1} = 1). The grid has no mass in the
# outer 1% tails, so this is a *grid* CRPS: comparable across models scored on
# the same grid, not equal to the exact CRPS.
_edges = np.concatenate([[0.0], TAUS, [1.0]])
CRPS_WEIGHTS = (_edges[2:] - _edges[:-2]) / 2.0


def _arr(x) -> np.ndarray:
    return np.asarray(x, dtype=float)


def squared_error(y, mean) -> np.ndarray:
    return (_arr(y) - _arr(mean)) ** 2


def pinball(y, Q) -> np.ndarray:
    """Pinball (quantile) loss, n x K."""
    u = _arr(y)[:, None] - _arr(Q)
    return np.maximum(TAUS * u, (TAUS - 1.0) * u)


def crps_grid(y, Q) -> np.ndarray:
    """Per-row CRPS approximated from the quantile grid (see CRPS_WEIGHTS)."""
    return 2.0 * pinball(y, Q) @ CRPS_WEIGHTS


def oos_r2(y, f_model, f_bench) -> float:
    """Campbell-Thompson out-of-sample R^2: 1 - SSE(model) / SSE(benchmark)."""
    sb = squared_error(y, f_bench).sum()
    return float(1.0 - squared_error(y, f_model).sum() / sb) if sb > 0 else np.nan


def coverage(y, Q, level: int) -> float:
    lo, hi = INTERVALS[level]
    y = _arr(y)
    Q = _arr(Q)
    a, b = Q[:, QCOLS.index(lo)], Q[:, QCOLS.index(hi)]
    return float(((y >= a) & (y <= b)).mean())


def bin_probabilities() -> np.ndarray:
    """Probability of each of the K+1 bins cut by the quantile grid."""
    return np.diff(_edges)


def pit_bins(y, Q) -> np.ndarray:
    """Index (0..K) of the grid bin each realization falls in."""
    return (_arr(Q) < _arr(y)[:, None]).sum(axis=1)


def pit_values(y, Q) -> np.ndarray:
    """Probability integral transform from the quantile grid: linear between grid
    quantiles, linear extrapolation (clipped to [0, 1]) beyond q01 and q99 using
    the slope of the outermost interval. For PIT histograms."""
    y, Q = _arr(y), _arr(Q)
    out = np.empty(len(y))
    for i in range(len(y)):
        q = Q[i]
        if y[i] <= q[0]:
            slope = (TAUS[1] - TAUS[0]) / max(q[1] - q[0], 1e-12)
            out[i] = max(0.0, TAUS[0] - slope * (q[0] - y[i]))
        elif y[i] >= q[-1]:
            slope = (TAUS[-1] - TAUS[-2]) / max(q[-1] - q[-2], 1e-12)
            out[i] = min(1.0, TAUS[-1] + slope * (y[i] - q[-1]))
        else:
            # np.interp needs increasing xp; ties are harmless
            out[i] = np.interp(y[i], q, TAUS)
    return out


def tail_hits(y, Q, qcol: str) -> tuple[np.ndarray, float]:
    """Exceedance indicator and its nominal probability for a tail quantile.
    Lower tail (tau < .5): hit = y < q. Upper tail: hit = y > q."""
    tau = TAUS[QCOLS.index(qcol)]
    q = _arr(Q)[:, QCOLS.index(qcol)]
    y = _arr(y)
    if tau < 0.5:
        return (y < q).astype(float), float(tau)
    return (y > q).astype(float), float(1.0 - tau)


def cumulative_sse_difference(df: pd.DataFrame, model_col: str = "mean_model",
                              bench_col: str = "mean_bench", y_col: str = "y") -> pd.Series:
    """Goyal-Welch series: cumulative sum over dates of SE(bench) - SE(model).

    ``df`` indexed by (date, asset) or date; rows on the same date are summed
    (pooled). Rising = model gaining on the benchmark."""
    d = (df[y_col] - df[bench_col]) ** 2 - (df[y_col] - df[model_col]) ** 2
    if isinstance(d.index, pd.MultiIndex):
        d = d.groupby(level="date").sum()
    return d.sort_index().cumsum()
