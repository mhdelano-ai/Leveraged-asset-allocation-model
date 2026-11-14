"""Building blocks for the ``mcorr`` family: unconditional-structure
estimators of the correlation matrix of monthly (~21-trading-day) log excess
returns under very unequal histories.

Everything works on a T x N array of monthly log excess returns (NaN where
an asset has no data) and a vector of observation weights ``w`` (T,).
All second moments are UNCENTERED, as in the target and the references.

Conventions
-----------
* Monthly returns are the only input. A calendar-month return is one
  non-overlapping ~21-day sum, i.e. exactly the quantity whose correlation
  the target measures, immune to asynchronous closes and stale daily prices
  (reports/data/horizon_evidence.md), and available in every era.
* Effective sample sizes are counted in independent months (Kish:
  ``(sum w)^2 / sum w^2`` over the months used), never in days.
* Every function returns a symmetric positive definite matrix with unit
  diagonal (or says how it repairs one).
"""
from __future__ import annotations

import numpy as np

from ...universe import BY_ID

EW_HALFLIFE = 120          # months; see the report: a round 10 years, the ref_corr_last120m window
MIN_PAIR_OBS = 24          # months of overlap before a pairwise estimate is used at all
EIG_FLOOR = 1e-3           # eigenvalue floor of the positive-definite repair (as the references)


# ---------------------------------------------------------------- weights
def ew_weights(T: int, halflife: float | None = EW_HALFLIFE) -> np.ndarray:
    """Exponential weights on T months, 1 on the latest month; flat if halflife is None."""
    if halflife is None:
        return np.ones(T)
    age = np.arange(T - 1, -1, -1, dtype=float)
    return 0.5 ** (age / halflife)


def kish(w: np.ndarray) -> float:
    s = w.sum()
    return float(s * s / (w * w).sum()) if s > 0 else 0.0


def pair_neff(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    """N x N Kish effective number of months on each pair's common sample."""
    ok = np.isfinite(X).astype(float)
    s1 = (ok * w[:, None]).T @ ok
    s2 = (ok * (w * w)[:, None]).T @ ok
    with np.errstate(invalid="ignore", divide="ignore"):
        n = np.where(s2 > 0, s1 * s1 / s2, 0.0)
    return n


# ------------------------------------------------------------- utilities
def cov_to_corr(S: np.ndarray) -> np.ndarray:
    d = np.sqrt(np.diag(S))
    C = S / np.outer(d, d)
    C = (C + C.T) / 2.0
    np.fill_diagonal(C, 1.0)
    return C


def nearest_corr(c: np.ndarray, floor: float = EIG_FLOOR) -> np.ndarray:
    """Symmetrize, clip eigenvalues at ``floor``, rescale to unit diagonal.
    Leaves a matrix whose smallest eigenvalue is already >= floor unchanged."""
    c = (np.asarray(c, float) + np.asarray(c, float).T) / 2.0
    np.fill_diagonal(c, 1.0)
    w, v = np.linalg.eigh(c)
    if w[0] >= floor:
        return c
    c = (v * np.maximum(w, floor)) @ v.T
    return cov_to_corr(c)


def ensure_pd(c: np.ndarray, floor: float = 1e-8) -> np.ndarray:
    """Minimal repair: only acts if the smallest eigenvalue is below ``floor``."""
    c = (c + c.T) / 2.0
    np.fill_diagonal(c, 1.0)
    if np.linalg.eigvalsh(c)[0] >= floor:
        return c
    return nearest_corr(c)


# ---------------------------------------------------- pairwise estimator
def pairwise_corr(X: np.ndarray, w: np.ndarray, min_obs: int = MIN_PAIR_OBS) -> np.ndarray:
    """Weighted uncentered correlation of each pair on its common sample, then
    the positive-definite repair. Pairs with < ``min_obs`` common months get 0.
    The unequal-history control: every pair uses only its own overlap."""
    ok = np.isfinite(X)
    a0 = np.where(ok, X, 0.0)
    okf = ok.astype(float)
    aw = a0 * w[:, None]
    sxy = aw.T @ a0
    sxx = (aw * a0).T @ okf                     # sum w x_i^2 over months where j is present
    n = okf.T @ okf
    with np.errstate(invalid="ignore", divide="ignore"):
        c = sxy / np.sqrt(sxx * sxx.T)
    c = np.where((n >= min_obs) & np.isfinite(c), c, 0.0)
    np.fill_diagonal(c, 1.0)
    return nearest_corr(np.clip(c, -1.0, 1.0))


# ------------------------------------------- monotone-missing ML (Stambaugh)
def history_order(X: np.ndarray) -> np.ndarray:
    """Column order by first observation (longest history first), stable."""
    T = X.shape[0]
    first = np.array([np.argmax(np.isfinite(X[:, j])) if np.isfinite(X[:, j]).any() else T
                      for j in range(X.shape[1])])
    return np.argsort(first, kind="stable")


def stambaugh_cov(X: np.ndarray, w: np.ndarray, ridge: float = 1e-10) -> np.ndarray:
    """Weighted ML second-moment matrix under a monotone missing pattern
    (Anderson 1957; Stambaugh 1997), uncentered.

    Columns are taken longest-history first. The first column's variance is
    its weighted mean square over its whole sample. Each further column j is
    regressed (weighted least squares, no intercept) on all columns before it,
    over the months where j and all of them are observed (for a monotone
    pattern: j's whole sample); then
        Sigma[j, prev] = beta' Sigma[prev, prev]
        Sigma[j, j]    = s2_j + beta' Sigma[prev, prev] beta
    with s2_j the weighted mean squared residual. So long-history assets keep
    their full-sample moments and each short-history asset is attached
    through its regression on everything already in. Positive definite by
    construction (s2_j > 0). Months with interior gaps are dropped from the
    regressions that would need them (not exact ML then; the real panel has
    none).
    """
    N = X.shape[1]
    order = history_order(X)
    Xo = X[:, order]
    S = np.zeros((N, N))
    for k in range(N):
        y = Xo[:, k]
        if k == 0:
            m = np.isfinite(y)
            S[0, 0] = np.sum(w[m] * y[m] ** 2) / np.sum(w[m])
            continue
        Z = Xo[:, :k]
        m = np.isfinite(y) & np.isfinite(Z).all(axis=1)
        ww, yy, ZZ = w[m], y[m], Z[m]
        sw = ww.sum()
        A = (ZZ * ww[:, None]).T @ ZZ / sw
        b = (ZZ * ww[:, None]).T @ yy / sw
        A = A + ridge * np.trace(A) / k * np.eye(k)
        beta = np.linalg.solve(A, b)
        resid = yy - ZZ @ beta
        s2 = max(np.sum(ww * resid ** 2) / sw, 1e-12 * np.sum(ww * yy ** 2) / sw)
        Sp = S[:k, :k]
        S[k, :k] = beta @ Sp
        S[:k, k] = S[k, :k]
        S[k, k] = s2 + beta @ Sp @ beta
    inv = np.empty(N, dtype=int)
    inv[order] = np.arange(N)
    return S[np.ix_(inv, inv)]


def stambaugh_corr(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    return ensure_pd(cov_to_corr(stambaugh_cov(X, w)))


# ----------------------------------------------------------- block structure
def groups_of(assets: list[str]) -> np.ndarray:
    """Integer block label per asset from ``volmodel.universe`` groups
    (equity, rates, credit, real); unknown ids get a block of their own."""
    names: dict[str, int] = {}
    out = []
    for a in assets:
        g = BY_ID[a].group if a in BY_ID else f"_own_{a}"
        out.append(names.setdefault(g, len(names)))
    return np.array(out)


def block_average(C: np.ndarray, g: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Block-equicorrelation projection: every off-diagonal entry replaced by
    the (weighted) mean of its block-pair cell. With equal weights this is the
    average of C over all within-block relabelings, hence positive definite
    whenever C is."""
    N = len(C)
    T = np.eye(N)
    W = np.ones_like(C) if weights is None else weights
    labels = np.unique(g)
    for a in labels:
        ia = np.flatnonzero(g == a)
        for b in labels:
            if b < a:
                continue
            ib = np.flatnonzero(g == b)
            cell = np.zeros((N, N), bool)
            cell[np.ix_(ia, ib)] = True
            cell[np.ix_(ib, ia)] = True
            np.fill_diagonal(cell, False)
            if not cell.any():
                continue
            v = np.sum(C[cell] * W[cell]) / np.sum(W[cell])
            T[cell] = v
    return T


def shrink_to_target(R: np.ndarray, Tg: np.ndarray, n: np.ndarray) -> tuple[np.ndarray, dict]:
    """Empirical-Bayes linear shrinkage of each correlation toward the target.

    Sampling variance of an estimated correlation (normal theory):
    ``v_ij = (1 - r_ij^2)^2 / n_ij`` with ``n_ij`` the pair's effective number
    of independent months. Dispersion of the true correlations around the
    target: ``tau2 = max(0, mean_ij[(r_ij - T_ij)^2 - v_ij])`` (method of
    moments over pairs). Each pair is then pulled toward the target with its
    own intensity ``delta_ij = v_ij / (v_ij + tau2)`` - the Ledoit-Wolf /
    James-Stein intensity, made pair-specific so that a 1997 asset is shrunk
    harder than a pair of 1926 assets. Repaired to PD only if needed.
    """
    N = len(R)
    iu = np.triu_indices(N, 1)
    if len(iu[0]) == 0:
        return R.copy(), {"tau2": np.nan, "delta_mean": np.nan}
    r, t = R[iu], Tg[iu]
    nn = np.maximum(n[iu], 3.0)
    v = (1.0 - r ** 2) ** 2 / nn
    tau2 = max(0.0, float(np.mean((r - t) ** 2 - v)))
    delta = np.ones_like(v) if tau2 <= 0 else v / (v + tau2)
    out = np.eye(N)
    out[iu] = t + (1.0 - delta) * (r - t)
    out[(iu[1], iu[0])] = out[iu]
    return ensure_pd(out), {"tau2": tau2, "delta_mean": float(delta.mean())}


# -------------------------------------------------- nonlinear shrinkage
def lw_analytical_nonlinear(lam: np.ndarray, n: float) -> np.ndarray:
    """Ledoit & Wolf (2020, Ann. Statist.) analytical nonlinear shrinkage of
    sample eigenvalues ``lam`` (any order) for ``n`` observations, p <= n case
    (the only one that arises: p <= 12 and n >= 36). Returns the shrunk
    eigenvalues in the same order."""
    p = len(lam)
    order = np.argsort(lam)
    l = np.maximum(lam[order], 1e-12)
    h = n ** (-1.0 / 3.0)
    L = np.repeat(l[:, None], p, axis=1)          # L[i, j] = l_i
    H = h * L.T                                   # H[i, j] = h * l_j
    x = (L - L.T) / H                             # (l_i - l_j) / (h l_j)
    ftilde = (3.0 / 4.0 / np.sqrt(5.0)) * np.mean(np.maximum(1.0 - x ** 2 / 5.0, 0.0) / H, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        Hf = (-3.0 / 10.0 / np.pi) * x + (3.0 / 4.0 / np.sqrt(5.0) / np.pi) * (1.0 - x ** 2 / 5.0) * \
            np.log(np.abs((np.sqrt(5.0) - x) / (np.sqrt(5.0) + x)))
    edge = np.isclose(np.abs(x), np.sqrt(5.0))
    Hf[edge] = (-3.0 / 10.0 / np.pi) * x[edge]
    Hftilde = np.mean(Hf / H, axis=1)
    c = p / n
    d = l / ((np.pi * c * l * ftilde) ** 2 + (1.0 - c - np.pi * c * l * Hftilde) ** 2)
    out = np.empty(p)
    out[order] = d
    return out


def nonlinear_shrink(R: np.ndarray, n: float) -> np.ndarray:
    lam, V = np.linalg.eigh(R)
    d = lw_analytical_nonlinear(lam, n)
    d = np.maximum(d, 1e-6 * d.max())
    return ensure_pd(cov_to_corr((V * d) @ V.T))


def rmt_clip(R: np.ndarray, n: float) -> np.ndarray:
    """Eigenvalue clipping (Laloux et al. 2000): eigenvalues below the
    Marchenko-Pastur edge (1 + sqrt(p/n))^2 are replaced by their mean
    (trace preserved), then rescaled to unit diagonal."""
    p = len(R)
    lam, V = np.linalg.eigh(R)
    edge = (1.0 + np.sqrt(p / n)) ** 2
    noise = lam < edge
    if noise.sum() >= 1 and noise.sum() < p:
        lam = lam.copy()
        lam[noise] = lam[noise].mean()
    elif noise.all():
        lam = np.full(p, lam.mean())
    return ensure_pd(cov_to_corr((V * lam) @ V.T))


# ------------------------------------------------------- factor structures
def ledermann_k(p: int, k_max: int) -> int:
    """Largest k <= k_max with non-negative degrees of freedom
    ((p - k)^2 - (p + k)) / 2 of a k-factor model on p variables."""
    k = min(k_max, p - 1)
    while k > 0 and (p - k) ** 2 - (p + k) < 0:
        k -= 1
    return k


def pca_factor_corr(R: np.ndarray, k: int, iters: int = 200, tol: float = 1e-8) -> np.ndarray:
    """Statistical k-factor correlation by iterated principal-axis factoring
    (principal components of R with the diagonal replaced by communalities,
    iterated to a fixed point; close to ML factor analysis). Correlations are
    the off-diagonal of L L'; uniquenesses 1 - communality keep it PD.
    ``k`` is cut to the Ledermann bound; with no identified factor model
    (p <= 3 for k = 1 at p = 3 is exactly identified) R is returned."""
    p = len(R)
    k = ledermann_k(p, k)
    if k == 0:
        return R.copy()
    h = np.clip(1.0 - 1.0 / np.diag(np.linalg.inv(R)), 0.05, 0.995)   # SMC start
    for _ in range(iters):
        Rr = R.copy()
        np.fill_diagonal(Rr, h)
        lam, V = np.linalg.eigh(Rr)
        L = V[:, ::-1][:, :k] * np.sqrt(np.maximum(lam[::-1][:k], 0.0))
        h_new = np.clip(np.sum(L * L, axis=1), 1e-4, 0.995)          # Heywood guard
        if np.max(np.abs(h_new - h)) < tol:
            h = h_new
            break
        h = h_new
    S = L @ L.T
    np.fill_diagonal(S, 1.0)
    S = np.clip(S, -0.999, 0.999)
    np.fill_diagonal(S, 1.0)
    return ensure_pd(S)


def observed_factor_cov(X: np.ndarray, Fx: np.ndarray, w: np.ndarray,
                        is_factor: np.ndarray, factor_col: np.ndarray, min_obs: int = MIN_PAIR_OBS
                        ) -> np.ndarray:
    """Covariance implied by an observed-factor model.

    ``Fx`` (T x K) are factor returns (NaN where missing), ``X`` (T x N) the
    assets. Factor covariance: Stambaugh ML on the factors' own histories.
    Each non-factor asset: weighted LS on all factors over the months where it
    and all factors are observed; residual variance -> diagonal. A factor
    asset (``is_factor[i]``, its factor index ``factor_col[i]``) has loading
    1 on itself and no residual. Positive definite when every non-factor
    asset has a residual and the factors' covariance is PD.
    """
    T, N = X.shape
    K = Fx.shape[1]
    Sf = stambaugh_cov(Fx, w)
    B = np.zeros((N, K))
    D = np.zeros(N)
    for i in range(N):
        if is_factor[i]:
            B[i, factor_col[i]] = 1.0
            continue
        y = X[:, i]
        m = np.isfinite(y) & np.isfinite(Fx).all(axis=1)
        if m.sum() < min_obs:            # too short for a regression: unexplained
            m = np.isfinite(y)
            D[i] = np.sum(w[m] * y[m] ** 2) / np.sum(w[m])
            continue
        ww, yy, ZZ = w[m], y[m], Fx[m]
        sw = ww.sum()
        A = (ZZ * ww[:, None]).T @ ZZ / sw
        b = (ZZ * ww[:, None]).T @ yy / sw
        beta = np.linalg.solve(A + 1e-10 * np.trace(A) / K * np.eye(K), b)
        B[i] = beta
        D[i] = max(np.sum(ww * (yy - ZZ @ beta) ** 2) / sw, 1e-8 * np.sum(ww * yy ** 2) / sw)
    return B @ Sf @ B.T + np.diag(D)
