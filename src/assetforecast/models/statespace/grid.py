"""Exact (non-Gaussian) filter for a univariate AR(1) log-variance state.

The Gaussian Kalman filter on ``log r^2`` is a quasi-likelihood device: the
measurement error of one month is a log chi-square(1), strongly left-skewed.
Here the state is discretized instead (a point-mass filter, Kitagawa 1987):
with the standardized state ``z = (h - mu) / s_x`` on a fixed grid, the AR(1)
transition is one matrix shared by all assets, and each month is updated
with the EXACT likelihood of the return, ``r | h ~ N(0, exp(h))``. The
filtered distribution of ``h`` is then exact up to the grid, and
``E[exp(h)]`` needs no lognormal approximation.
"""
from __future__ import annotations

import numpy as np

G = 241
ZMAX = 5.0
ZGRID = np.linspace(-ZMAX, ZMAX, G)


def transition(phi: float) -> np.ndarray:
    """Row-stochastic matrix K[i, j] = P(z' = z_j | z = z_i)."""
    s = np.sqrt(1.0 - phi ** 2)
    K = np.exp(-0.5 * ((ZGRID[None, :] - phi * ZGRID[:, None]) / s) ** 2)
    return K / K.sum(1, keepdims=True)


def prior() -> np.ndarray:
    p = np.exp(-0.5 * ZGRID ** 2)
    return p / p.sum()


def grid_filter(pi0, K, mu, sx, r2, store=False):
    """Run the point-mass filter.

    pi0 (A, G) filtered distribution before the first month; K (G, G);
    mu, sx (A,); r2 (T, A) squared returns (NaN = missing).
    Returns (pi_T, pred) with pred (T, A) = one-step E[exp(h)] if ``store``.
    """
    pi = pi0.copy()
    T, A = r2.shape
    h = mu[:, None] + sx[:, None] * ZGRID[None, :]          # (A, G)
    eh = np.exp(h)
    pred = np.full((T, A), np.nan) if store else None
    for t in range(T):
        pi = pi @ K
        if store:
            pred[t] = (pi * eh).sum(1)
        x = r2[t]
        ok = np.isfinite(x)
        if ok.any():
            xx = np.where(ok, x, 0.0)
            logl = -0.5 * (h + xx[:, None] / eh)
            logl -= logl.max(1, keepdims=True)
            w = pi * np.exp(logl)
            w /= w.sum(1, keepdims=True)
            pi = np.where(ok[:, None], w, pi)
    return pi, pred


def forecast(pi, K, mu, sx, horizon=12):
    """Sum over the next ``horizon`` months of E[exp(h_{t+k})] (A,)."""
    eh = np.exp(mu[:, None] + sx[:, None] * ZGRID[None, :])
    tot = np.zeros(pi.shape[0])
    p = pi
    for _ in range(horizon):
        p = p @ K
        tot += (p * eh).sum(1)
    return tot
