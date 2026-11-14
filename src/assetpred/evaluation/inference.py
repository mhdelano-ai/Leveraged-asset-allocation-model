"""Hypothesis tests that respect overlapping h-month targets, and multiple-testing tools.

Overlap. Forecast errors for h-month targets made at consecutive months share
h-1 months of returns, so loss differentials are MA(h-1) even under the null.
Every test here uses a Bartlett (Newey-West) long-run variance with
``hac_lags(h, n) = max(ceil(1.5 h) for h > 1, floor(4 (n/100)^(2/9)))`` lags
and Student-t(n-1) reference distributions. Lag h-1 alone is too short for the
Bartlett kernel (it down-weights the overlap autocovariances); 1.5h gave close
to nominal size for Clark-West and Diebold-Mariano at h = 12 with ~350
monthly observations in Monte Carlo (``tests/test_inference.py``).

Pooled tests aggregate across assets *per date first* (sum or mean), then apply
the time-series test, so cross-asset correlation is accounted for.

Data snooping. :func:`spa_test` (Hansen 2005) and :func:`model_confidence_set`
(Hansen, Lunde & Nason 2011) use the stationary bootstrap from ``arch`` with
mean block length ``block_length(h, n) = max(2h, ceil(n^(1/3)))``.
:func:`fdr` adjusts p-values across a grid (Benjamini-Hochberg, or
Benjamini-Yekutieli which is valid under arbitrary dependence).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import xlogy


def hac_lags(h: int, n: int) -> int:
    rule = int(math.floor(4 * (max(n, 1) / 100.0) ** (2.0 / 9.0)))
    over = int(math.ceil(1.5 * h)) if h > 1 else 0
    return max(over, rule, 0)


def block_length(h: int, n: int) -> int:
    return int(max(2 * h, math.ceil(max(n, 1) ** (1.0 / 3.0))))


def long_run_variance(x, lags: int) -> np.ndarray | float:
    """Bartlett-kernel long-run variance (scalar for 1-D, matrix for n x k)."""
    x = np.asarray(x, dtype=float)
    one_d = x.ndim == 1
    if one_d:
        x = x[:, None]
    x = x - x.mean(axis=0)
    n = len(x)
    S = x.T @ x / n
    for j in range(1, min(lags, n - 1) + 1):
        G = x[j:].T @ x[:-j] / n
        S += (1.0 - j / (lags + 1.0)) * (G + G.T)
    return float(S[0, 0]) if one_d else S


def mean_test(d, h: int) -> tuple[float, float, float, int]:
    """t-test that E[d] = 0 with HAC variance. Returns (t, p_greater, p_two, n)."""
    d = np.asarray(d, dtype=float)
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 10:
        return np.nan, np.nan, np.nan, n
    v = long_run_variance(d, hac_lags(h, n))
    if not v > 0:
        return np.nan, np.nan, np.nan, n
    t = d.mean() / math.sqrt(v / n)
    return float(t), float(stats.t.sf(t, n - 1)), float(2 * stats.t.sf(abs(t), n - 1)), n


def clark_west(y, f_bench, f_model, h: int, dates=None) -> tuple[float, float]:
    """Clark-West (2007) MSPE-adjusted test; H1: model beats the nested benchmark.

    ``f = (y - f_b)^2 - [(y - f_m)^2 - (f_b - f_m)^2]``. With ``dates`` given
    (panel), f is summed per date before testing. Returns (stat, one-sided p).
    """
    y, fb, fm = (np.asarray(a, dtype=float) for a in (y, f_bench, f_model))
    f = (y - fb) ** 2 - ((y - fm) ** 2 - (fb - fm) ** 2)
    if dates is not None:
        f = pd.Series(f).groupby(np.asarray(dates)).sum().to_numpy()
    t, p, _, _ = mean_test(f, h)
    return t, p


def diebold_mariano(loss_model, loss_bench, h: int, dates=None) -> tuple[float, float, float]:
    """Diebold-Mariano test on loss differentials ``d = loss_bench - loss_model``.

    Positive stat = model has lower loss. Returns (stat, p_one_sided_model_better,
    p_two_sided). For nested point forecasts prefer Clark-West; DM is the right
    test for non-nested comparisons and for distribution losses (CRPS, pinball).
    """
    d = np.asarray(loss_bench, dtype=float) - np.asarray(loss_model, dtype=float)
    if dates is not None:
        d = pd.Series(d).groupby(np.asarray(dates)).sum().to_numpy()
    t, p1, p2, _ = mean_test(d, h)
    return t, p1, p2


def kupiec(hits, p: float, h: int) -> tuple[float, float, float]:
    """Kupiec (1995) unconditional coverage LR test, adjusted for overlap.

    Overlapping targets make hits autocorrelated even with a correct model, so
    the iid LR statistic is divided by the variance inflation factor
    ``VIF = LRV(hits - p) / (p (1 - p))`` (floored at 1) before comparing with
    chi2(1). Returns (hit_rate, LR_adjusted, p_value).
    """
    x = np.asarray(hits, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 10:
        return np.nan, np.nan, np.nan
    k = x.sum()
    pi = k / n
    ll0 = xlogy(n - k, 1 - p) + xlogy(k, p)
    ll1 = xlogy(n - k, 1 - pi) + xlogy(k, pi)
    lr = max(-2.0 * (ll0 - ll1), 0.0)
    vif = max(long_run_variance(x - p, hac_lags(h, n)) / (p * (1 - p)), 1.0) if h > 1 else 1.0
    lr_adj = lr / vif
    return float(pi), float(lr_adj), float(stats.chi2.sf(lr_adj, 1))


def _christoffersen_lr(x: np.ndarray) -> float:
    a, b = x[:-1], x[1:]
    n00 = np.sum((a == 0) & (b == 0)); n01 = np.sum((a == 0) & (b == 1))
    n10 = np.sum((a == 1) & (b == 0)); n11 = np.sum((a == 1) & (b == 1))
    if n01 + n11 == 0 or n00 + n10 == 0 or n00 + n01 == 0 or n10 + n11 == 0:
        return np.nan
    p01 = n01 / (n00 + n01)
    p11 = n11 / (n10 + n11)
    p = (n01 + n11) / (n00 + n01 + n10 + n11)
    ll0 = xlogy(n00 + n10, 1 - p) + xlogy(n01 + n11, p)
    ll1 = xlogy(n00, 1 - p01) + xlogy(n01, p01) + xlogy(n10, 1 - p11) + xlogy(n11, p11)
    return max(-2.0 * (ll0 - ll1), 0.0)


def christoffersen(hits, h: int) -> float:
    """Christoffersen (1998) independence test p-value, adjusted for overlap.

    For h > 1 consecutive hits share h-1 months of returns, so the test is run
    on the h non-overlapping subsequences (every h-th origin) and the smallest
    p-value is Bonferroni-corrected (times the number of usable subsequences).
    NaN when there are too few hits to form the transition table.
    """
    x = np.asarray(hits, dtype=float)
    x = x[np.isfinite(x)]
    ps = []
    for off in range(h):
        lr = _christoffersen_lr(x[off::h])
        if np.isfinite(lr):
            ps.append(stats.chi2.sf(lr, 1))
    if not ps:
        return np.nan
    return float(min(1.0, min(ps) * len(ps)))


def calibration_test(bins, probs, h: int, dates=None) -> tuple[float, float]:
    """HAC Wald test that realizations fall in the quantile-grid bins with the
    nominal probabilities (a PIT uniformity test robust to overlap).

    ``bins`` are bin indices 0..K (``metrics.pit_bins``); ``probs`` the K+1
    nominal probabilities. With ``dates``, deviations are averaged per date.
    Returns (Wald, p) with chi2(K) reference.
    """
    bins = np.asarray(bins)
    K1 = len(probs)
    D = (bins[:, None] == np.arange(K1)[None, :]).astype(float) - np.asarray(probs)[None, :]
    D = D[:, :-1]
    if dates is not None:
        D = pd.DataFrame(D).groupby(np.asarray(dates)).mean().to_numpy()
    n = len(D)
    if n < 20:
        return np.nan, np.nan
    S = long_run_variance(D, hac_lags(h, n))
    m = D.mean(axis=0)
    W = float(n * m @ np.linalg.pinv(S) @ m)
    return W, float(stats.chi2.sf(W, D.shape[1]))


def spa_test(bench_loss, model_losses: pd.DataFrame, h: int, reps: int = 999,
             seed: int | None = None) -> float:
    """Hansen's SPA p-value (consistent) for H0: no model has lower expected loss
    than the benchmark. ``model_losses`` is a date x model frame aligned with
    ``bench_loss``. Accounts for having searched over all the columns."""
    from arch.bootstrap import SPA
    b = np.asarray(bench_loss, dtype=float)
    M = np.asarray(model_losses, dtype=float)
    ok = np.isfinite(b) & np.isfinite(M).all(axis=1)
    b, M = b[ok], M[ok]
    # models whose loss equals the benchmark's (same point forecast) cannot beat it
    distinct = (M - b[:, None]).std(axis=0) > 1e-12 * max(float(np.abs(b).mean()), 1e-300)
    M = M[:, distinct]
    if len(b) < 20:
        return np.nan
    if M.shape[1] == 0:
        return 1.0
    spa = SPA(b, M, block_size=block_length(h, len(b)), reps=reps,
              bootstrap="stationary", seed=seed)
    spa.compute()
    return float(spa.pvalues["consistent"])


def model_confidence_set(losses: pd.DataFrame, h: int, size: float = 0.10, reps: int = 999,
                         seed: int | None = None) -> pd.DataFrame:
    """Hansen-Lunde-Nason MCS over the columns of a date x model loss frame.
    Returns a frame indexed by model with ``mcs_p`` and ``in_mcs`` (at ``size``)."""
    from arch.bootstrap import MCS
    L = losses.dropna()
    if len(L) < 20 or L.shape[1] < 2:
        return pd.DataFrame({"mcs_p": np.nan, "in_mcs": np.nan}, index=losses.columns)
    # models with identical losses (e.g. baselines sharing a point forecast) are
    # tested once and share the result
    rep: dict[str, str] = {}
    keep: list[str] = []
    scale = max(float(L.abs().to_numpy().mean()), 1e-300)
    for c in L.columns:
        match = next((k for k in keep if (L[c] - L[k]).abs().max() <= 1e-12 * scale), None)
        rep[c] = match or c
        if match is None:
            keep.append(c)
    if len(keep) < 2:
        return pd.DataFrame({"mcs_p": 1.0, "in_mcs": 1.0}, index=losses.columns)
    mcs = MCS(L[keep], size=size, reps=reps, block_size=block_length(h, len(L)),
              bootstrap="stationary", method="R", seed=seed)
    mcs.compute()
    pk = mcs.pvalues["Pvalue"]
    p = pd.Series([float(pk[rep[c]]) for c in losses.columns], index=losses.columns)
    return pd.DataFrame({"mcs_p": p, "in_mcs": (p > size).astype(float)})


def fdr(pvalues, method: str = "bh") -> np.ndarray:
    """False-discovery-rate q-values. ``method``: 'bh' (Benjamini-Hochberg,
    valid under positive dependence) or 'by' (Benjamini-Yekutieli, any
    dependence). NaNs are passed through."""
    from statsmodels.stats.multitest import multipletests
    p = np.asarray(pvalues, dtype=float)
    out = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    if ok.any():
        out[ok] = multipletests(p[ok], method={"bh": "fdr_bh", "by": "fdr_by"}[method])[1]
    return out


def detectable_r2(n_indep: float, alpha: float = 0.05) -> float:
    """Rough out-of-sample R^2 a predictor needs for a one-sided t-test at
    ``alpha`` to reject, given ``n_indep`` non-overlapping observations:
    t ~ sqrt(n R^2 / (1 - R^2)). A reality check on how little data there is."""
    if not n_indep or n_indep <= 1:
        return np.nan
    z2 = stats.norm.isf(alpha) ** 2
    return float(z2 / (n_indep + z2))
