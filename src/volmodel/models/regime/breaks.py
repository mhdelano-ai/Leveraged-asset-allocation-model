"""Structural breaks in the variance level.

Two detectors, both with sensitivity fixed in advance:

* :func:`bocpd` - Bayesian online change-point detection (Adams & MacKay
  2007). Within a segment, monthly returns are N(0, sigma^2) with
  sigma^2 ~ InvGamma(alpha0, beta0); a new segment starts each month with
  constant hazard H. The run-length posterior is the evidence for a break at
  every past date, and the posterior mean of the current segment's variance
  automatically down-weights pre-break data in proportion to that evidence.

* :func:`icss_kappa2` - binary segmentation with the Sanso, Arago & Carrion
  (2004) kappa-2 CUSUM-of-squares test, whose long-run-variance denominator
  makes it valid under conditional heteroskedasticity (volatility clusters do
  not count as breaks, unlike the original Inclan-Tiao statistic). Critical
  value: sup of a Brownian bridge.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.special import gammaln

# sup|Brownian bridge| quantiles (Kolmogorov distribution)
KS_CRIT = {0.10: 1.224, 0.05: 1.358, 0.01: 1.628}


# ------------------------------------------------------------------ BOCPD
class BOCPDState:
    """Run-length posterior after the last observation processed."""

    __slots__ = ("logp", "alpha", "beta", "n", "loglik")

    def __init__(self, alpha0: float, beta0: float):
        self.logp = np.array([0.0])          # run length 0 before any data
        self.alpha = np.array([alpha0])
        self.beta = np.array([beta0])
        self.n = 0                            # observations processed (incl. NaN skips)
        self.loglik = 0.0                     # log marginal likelihood of the data processed

    def copy(self):
        s = BOCPDState.__new__(BOCPDState)
        s.logp, s.alpha, s.beta, s.n = self.logp.copy(), self.alpha.copy(), self.beta.copy(), self.n
        s.loglik = self.loglik
        return s


def bocpd_update(state: BOCPDState, x: np.ndarray, hazard: float, alpha0: float, beta0: float,
                 prune: float = 1e-10) -> BOCPDState:
    """Process observations ``x`` (NaN = no observation: the state is carried unchanged).

    Run lengths whose posterior falls below ``prune`` relative to the largest
    are dropped (deterministically), which keeps the cost per step bounded.
    """
    lh, l1h = math.log(hazard), math.log1p(-hazard)
    logp, a, b = state.logp, state.alpha, state.beta
    for v in x:
        state.n += 1
        if not np.isfinite(v):
            continue
        x2 = v * v
        # Student-t predictive with 2a dof and scale^2 b/a
        lpred = gammaln(a + 0.5) - gammaln(a) - 0.5 * np.log(2 * math.pi * b) - (a + 0.5) * np.log1p(x2 / (2 * b))
        lj = logp + lpred
        grow = lj + l1h
        m = lj.max()
        ev = m + math.log(np.exp(lj - m).sum())   # log p(x_t | x_1..x_{t-1})
        state.loglik += ev
        cp = ev + lh
        logp = np.concatenate([[cp], grow])
        logp -= logp.max()
        logp -= math.log(np.exp(logp).sum())
        a = np.concatenate([[alpha0], a + 0.5])
        b = np.concatenate([[beta0], b + 0.5 * x2])
        keep = logp > logp.max() + math.log(prune)
        if not keep.all():
            keep[0] = True
            logp, a, b = logp[keep], a[keep], b[keep]
            logp -= math.log(np.exp(logp).sum())
    state.logp, state.alpha, state.beta = logp, a, b
    return state


def bocpd_forecast(state: BOCPDState, hazard: float, prior_mean: float, horizon: int) -> float:
    """Expected mean variance over the next ``horizon`` steps.

    A future step stays in the current segment with probability (1-H)^h; a
    new segment's variance has the prior mean.
    """
    p = np.exp(state.logp)
    seg = float((p * state.beta / (state.alpha - 1.0)).sum())
    h = np.arange(1, horizon + 1)
    stay = (1.0 - hazard) ** h
    return float(np.mean(stay * seg + (1 - stay) * prior_mean))


def run_length_map(state: BOCPDState) -> tuple[int, float]:
    """(most probable run length in observations, its posterior probability)."""
    p = np.exp(state.logp)
    k = int(np.argmax(p))
    return int(round((state.alpha[k] - state.alpha[0]) * 2)) if k else 0, float(p[k])


# ------------------------------------------------------------ kappa-2 ICSS
def bartlett_lrv(e: np.ndarray, bandwidth: int) -> float:
    e = e - e.mean()
    T = len(e)
    v = float(e @ e) / T
    for j in range(1, min(bandwidth, T - 1) + 1):
        w = 1.0 - j / (bandwidth + 1.0)
        v += 2.0 * w * float(e[j:] @ e[:-j]) / T
    return v


def kappa2(e: np.ndarray) -> tuple[float, int]:
    """Sanso-Arago-Carrion kappa-2 on squared returns ``e``: (statistic, break index).

    The break index ``k`` means a change between e[k] and e[k+1]. Bandwidth
    ``max(12, ceil(1.3 sqrt(T)))``: longer than the textbook Newey-West rule
    because monthly squared returns are persistent well beyond a year.
    """
    T = len(e)
    bw = max(12, int(math.ceil(1.3 * math.sqrt(T))))
    om2 = bartlett_lrv(e, bw)
    if not np.isfinite(om2) or om2 <= 0:
        return 0.0, -1
    c = np.cumsum(e - e.mean())
    k = int(np.argmax(np.abs(c[:-1])))
    return float(abs(c[k]) / math.sqrt(T * om2)), k


def icss_kappa2(x: np.ndarray, level: float = 0.01, min_seg: int = 24) -> list[int]:
    """Break positions in the (NaN-free) series ``x`` by binary segmentation.

    A position ``p`` in the result means a new variance level starts at x[p].
    Segments shorter than ``min_seg`` are not tested or created.
    """
    crit = KS_CRIT[level]
    e_all = np.asarray(x, float) ** 2
    breaks: list[int] = []
    stack = [(0, len(e_all))]
    while stack:
        lo, hi = stack.pop()
        if hi - lo < 2 * min_seg:
            continue
        stat, k = kappa2(e_all[lo:hi])
        p = lo + k + 1
        if stat > crit and p - lo >= min_seg and hi - p >= min_seg:
            breaks.append(p)
            stack.append((lo, p))
            stack.append((p, hi))
    return sorted(breaks)
