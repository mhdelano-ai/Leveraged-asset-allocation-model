"""Performance metrics and tests of differences for the strategy test.

Tests of a difference between two monthly excess-return series a and b (same
months) treat the statistic as a smooth function of the means of
``u_t = (a_t, b_t, a_t^2, b_t^2)``:

* HAC delta method (Ledoit & Wolf 2008 style, Bartlett kernel, lags from
  ``inference.hac_lags``), normal CI and Student-t(n-1) p-value;
* paired stationary block bootstrap (Politis & Romano 1994; mean block length
  ``inference.block_length``): percentile CI and the symmetric p-value
  P(|theta* - theta_hat| >= |theta_hat|).

Both resample/estimate months jointly, so serial correlation, fat tails and
the correlation between a and b are respected.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import stats

from . import inference as inf


# ----------------------------------------------------------------- levels
def sharpe(x) -> float:
    x = np.asarray(x, float)
    sd = x.std(ddof=1)
    return float(x.mean() / sd * math.sqrt(12)) if sd > 0 else np.nan


def cer(x, gamma: float) -> float:
    """Annualized certainty equivalent of monthly excess returns, mean-variance
    utility: 12 (mean - gamma/2 var)."""
    x = np.asarray(x, float)
    return float(12 * (x.mean() - 0.5 * gamma * x.var(ddof=0)))


def max_drawdown(total_returns) -> float:
    w = np.cumprod(1 + np.asarray(total_returns, float))
    peak = np.maximum.accumulate(np.concatenate([[1.0], w]))[1:]
    return float((w / peak - 1).min())


def vol_tracking(x, target: float, window: int = 12) -> tuple[float, float]:
    """RMSE of log(rolling-window realized vol / target) and the sd of the
    rolling annualized vol (vol of vol)."""
    x = np.asarray(x, float)
    if len(x) < window + 2:
        return np.nan, np.nan
    sw = np.lib.stride_tricks.sliding_window_view(x, window)
    rv = sw.std(axis=1, ddof=1) * math.sqrt(12)
    rv = rv[rv > 0]
    return float(np.sqrt(np.mean(np.log(rv / target) ** 2))), float(rv.std(ddof=1))


# ----------------------------------------------------------------- differences
def _stat(kind: str, m, gamma: float):
    """kind on moment array(s) m[..., 4] = (Ea, Eb, Ea2, Eb2); annualized."""
    ma, mb, sa2, sb2 = m[..., 0], m[..., 1], m[..., 2], m[..., 3]
    va, vb = sa2 - ma ** 2, sb2 - mb ** 2
    if kind == "sharpe":
        with np.errstate(invalid="ignore", divide="ignore"):
            return math.sqrt(12) * (ma / np.sqrt(va) - mb / np.sqrt(vb))
    if kind == "cer":
        return 12 * ((ma - 0.5 * gamma * va) - (mb - 0.5 * gamma * vb))
    if kind == "mean":
        return 12 * (ma - mb)
    raise ValueError(kind)


def _grad(kind: str, m, gamma: float) -> np.ndarray:
    ma, mb, sa2, sb2 = m
    if kind == "sharpe":
        sa = math.sqrt(sa2 - ma ** 2)
        sb = math.sqrt(sb2 - mb ** 2)
        g = np.array([sa2 / sa ** 3, -sb2 / sb ** 3, -ma / (2 * sa ** 3), mb / (2 * sb ** 3)])
        return math.sqrt(12) * g
    if kind == "cer":
        return 12 * np.array([1 + gamma * ma, -(1 + gamma * mb), -0.5 * gamma, 0.5 * gamma])
    if kind == "mean":
        return 12 * np.array([1.0, -1.0, 0.0, 0.0])
    raise ValueError(kind)


def stationary_bootstrap_indices(n: int, reps: int, block: float, rng) -> np.ndarray:
    """reps x n index matrix of the stationary bootstrap (geometric blocks, wrap-around)."""
    p = 1.0 / block
    start = rng.integers(0, n, size=(reps, n))
    new = rng.random((reps, n)) < p
    new[:, 0] = True
    # position within the current block
    pos = np.arange(n)[None, :].repeat(reps, 0)
    last_new = np.where(new, pos, 0)
    last_new = np.maximum.accumulate(last_new, axis=1)
    s = np.take_along_axis(start, last_new, axis=1)
    return (s + (pos - last_new)) % n


def diff_test(a, b, gamma: float, h: int = 1, idx: np.ndarray | None = None,
              kinds=("sharpe", "cer")) -> dict:
    """Tests of SR(a) - SR(b) and CER(a) - CER(b). ``idx`` is a bootstrap
    index matrix (reps x n) or None to skip the bootstrap."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = len(a)
    U = np.column_stack([a, b, a ** 2, b ** 2])
    m = U.mean(0)
    lags = inf.hac_lags(h, n)
    V = inf.long_run_variance(U, lags)
    out = {"n_months": float(n)}
    d = a - b
    vd = d.var()
    lrv = inf.long_run_variance(d, lags)
    out["n_eff"] = float(n * vd / lrv) if lrv > 0 else np.nan
    Mb = None
    if idx is not None:
        # bootstrap moments: reps x 4
        Mb = np.stack([U[:, j][idx].mean(1) for j in range(4)], axis=-1)
    for k in kinds:
        th = float(_stat(k, m, gamma))
        g = _grad(k, m, gamma)
        se = float(math.sqrt(max(g @ V @ g, 0) / n))
        t = th / se if se > 0 else np.nan
        out[f"d_{k}"] = th
        out[f"d_{k}_se"] = se
        out[f"d_{k}_t"] = t
        out[f"d_{k}_p"] = float(2 * stats.t.sf(abs(t), n - 1)) if np.isfinite(t) else np.nan
        out[f"d_{k}_ci_lo"] = th - 1.96 * se
        out[f"d_{k}_ci_hi"] = th + 1.96 * se
        if Mb is not None:
            tb = _stat(k, Mb, gamma)
            tb = tb[np.isfinite(tb)]
            out[f"d_{k}_boot_lo"] = float(np.quantile(tb, 0.025))
            out[f"d_{k}_boot_hi"] = float(np.quantile(tb, 0.975))
            out[f"d_{k}_boot_p"] = float((np.abs(tb - th) >= abs(th)).mean())
    return out


def breakeven(gross_a, gross_b, to_a, to_b, other_a, other_b, gamma: float) -> float:
    """Uniform one-way proportional cost c (per unit of turnover) at which
    CER(a) = CER(b), where net = gross - c * turnover - other (other =
    financing). 0 if a does not beat b even at zero trading cost; inf if a
    beats b at every cost. Returned in basis points."""
    xa = np.asarray(gross_a, float) - np.asarray(other_a, float)
    xb = np.asarray(gross_b, float) - np.asarray(other_b, float)
    ta, tb = np.asarray(to_a, float), np.asarray(to_b, float)

    def coefs(x, t):
        # CER(x - c t)/12 = mean x - c mean t - g/2 (var x - 2 c cov(x,t) + c^2 var t)
        cxt = np.mean((x - x.mean()) * (t - t.mean()))
        return np.array([x.mean() - 0.5 * gamma * x.var(),
                         -t.mean() + gamma * cxt,
                         -0.5 * gamma * t.var()])

    A, B, C = 12 * (coefs(xa, ta) - coefs(xb, tb))
    if not A > 0:
        return 0.0
    if abs(C) < 1e-18:
        roots = [-A / B] if B < 0 else []
    else:
        disc = B * B - 4 * A * C
        roots = [] if disc < 0 else [(-B + s * math.sqrt(disc)) / (2 * C) for s in (1, -1)]
    roots = [r for r in roots if r > 0]
    return float(min(roots) * 1e4) if roots else float("inf")
