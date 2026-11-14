"""Sign-constrained ridge regression for overlapping 12-month targets.

    minimize  sum_i w_i (y_i - b0 - x_i' b)^2  +  (s2 / tau^2) * (b0^2 + |b|^2)
    subject to  b_j >= 0  for the sign-constrained columns

``x`` is standardized on the training sample, so ``tau`` is the prior
standard deviation of the effect of a ONE-standard-deviation move of a
predictor on the Fisher-z target (and of the intercept: the target is a
deviation from the baseline, so the prior centre of everything is "the
baseline is right"). ``w_i = 1 / (12 * n_pairs)``: twelve overlapping monthly
windows carry about one year of independent information (an MA(11) error has
long-run variance 12 x its variance), and pooled pairs of nearly identical
assets are not independent evidence. ``s2`` is the weighted mean square of
the target around zero (the variance of the baseline's errors), a deliberately
conservative noise scale: the penalty is the Bayesian ridge for a Gaussian
prior N(0, tau^2) on every coefficient.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import lsq_linear


@dataclass
class RidgeFit:
    names: list[str]
    mean: np.ndarray
    sd: np.ndarray
    coef: np.ndarray          # standardized slopes
    intercept: float
    n_obs: int
    n_years: float            # effective independent years (sum of weights)
    s2: float

    def predict(self, x: np.ndarray) -> float:
        z = (np.asarray(x, float) - self.mean) / self.sd
        return float(self.intercept + z @ self.coef)

    def raw_slopes(self) -> dict[str, float]:
        """Effect of a one-unit (native units) move of each predictor on z."""
        return {n: float(c / s) for n, c, s in zip(self.names, self.coef, self.sd)}


def fit_ridge(X: np.ndarray, y: np.ndarray, w: np.ndarray, names: list[str], *,
              tau: float, nonneg: list[bool]) -> RidgeFit:
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    w = np.asarray(w, float)
    n, k = X.shape
    sw = w.sum()
    mean = (w @ X) / sw
    sd = np.sqrt((w @ (X - mean) ** 2) / sw)
    sd = np.where(sd > 1e-12, sd, 1.0)
    Z = (X - mean) / sd
    s2 = float((w @ y ** 2) / sw)
    s2 = max(s2, 1e-4)
    lam = s2 / tau ** 2
    A = np.column_stack([np.ones(n), Z]) * np.sqrt(w)[:, None]
    b = y * np.sqrt(w)
    A = np.vstack([A, np.sqrt(lam) * np.eye(k + 1)])
    b = np.concatenate([b, np.zeros(k + 1)])
    lb = np.array([-np.inf] + [0.0 if nn else -np.inf for nn in nonneg])
    ub = np.full(k + 1, np.inf)
    sol = lsq_linear(A, b, bounds=(lb, ub), method="bvls")
    beta = sol.x
    return RidgeFit(names=list(names), mean=mean, sd=sd, coef=beta[1:], intercept=float(beta[0]),
                    n_obs=n, n_years=float(sw), s2=s2)
