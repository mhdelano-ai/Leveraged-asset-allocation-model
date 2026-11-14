"""Hidden Markov machinery for regime-switching volatility, in plain numpy.

The forward and backward recursions of an HMM are products of K x K
matrices ``M_t = P diag(b_t)``. Instead of a Python loop over time they are
computed here with a parallel prefix scan (Hillis-Steele): ``log2(T)``
batched matrix products, each normalised to sum one with the log scale
carried separately. All quantities are positive, so the normalisation is
exact up to rounding and nothing underflows. One forward-backward pass on
1,700 monthly observations costs about a millisecond.

Conventions
-----------
* ``logb``: (T, K) log emission densities; a row of zeros = missing observation.
* ``P``: (K, K) row-stochastic transition matrix, ``P[i, j] = Pr(s_t=j | s_{t-1}=i)``.
* ``pi0``: distribution of the state BEFORE the first observation (the
  stationary distribution of ``P`` by default), so ``Pr(s_1) = pi0 P``.
* Filtered probabilities ``Pr(s_t | x_1..x_t)`` use only the past; smoothed
  ones (``gamma``) are used only inside EM, on data up to the estimation date.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.special import gammaln

# ---------------------------------------------------------------- scans


def _norm(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    s = A.sum(axis=(1, 2))
    s = np.where(s > 0, s, 1e-300)
    return A / s[:, None, None], np.log(s)


def prefix_products(M: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Normalised prefix products ``M_1 M_2 ... M_t`` for every t, plus log scales."""
    A, logs = _norm(M)
    T, d = len(A), 1
    while d < T:
        new, ls = _norm(np.matmul(A[:-d], A[d:]))
        logs[d:] = logs[:-d] + logs[d:] + ls
        A[d:] = new
        d *= 2
    return A, logs


def suffix_products(M: np.ndarray) -> np.ndarray:
    """Normalised suffix products ``M_t M_{t+1} ... M_T`` for every t."""
    A, _ = _norm(M[::-1].copy())
    T, d = len(A), 1
    while d < T:
        new, _ = _norm(np.matmul(A[d:], A[:-d]))
        A[d:] = new
        d *= 2
    return A[::-1]


def stationary(P: np.ndarray) -> np.ndarray:
    K = len(P)
    A = np.vstack([P.T - np.eye(K), np.ones(K)])
    b = np.zeros(K + 1)
    b[-1] = 1.0
    pi = np.linalg.lstsq(A, b, rcond=None)[0]
    pi = np.clip(pi, 1e-12, None)
    return pi / pi.sum()


def _emission_scale(logb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mx = logb.max(axis=1)
    return np.exp(logb - mx[:, None]), mx


def filter_probs(logb: np.ndarray, P: np.ndarray, pi0: np.ndarray | None = None) -> tuple[np.ndarray, float]:
    """Filtered state probabilities (T, K) and the log-likelihood."""
    pi0 = stationary(P) if pi0 is None else pi0
    b, mx = _emission_scale(logb)
    M = P[None, :, :] * b[:, None, :]
    A, logs = prefix_products(M)
    a = np.einsum("i,tij->tj", pi0, A)
    s = a.sum(axis=1)
    f = a / s[:, None]
    ll = float(math.log(s[-1]) + logs[-1] + mx.sum())
    return f, ll


def forward_backward(logb: np.ndarray, P: np.ndarray, pi0: np.ndarray | None = None):
    """Filtered, smoothed probabilities, summed two-slice marginals and log-likelihood."""
    pi0 = stationary(P) if pi0 is None else pi0
    T, K = logb.shape
    b, mx = _emission_scale(logb)
    M = P[None, :, :] * b[:, None, :]
    A, logs = prefix_products(M)
    a = np.einsum("i,tij->tj", pi0, A)
    s = a.sum(axis=1)
    f = a / s[:, None]
    ll = float(math.log(s[-1]) + logs[-1] + mx.sum())
    beta = np.ones((T, K))
    if T > 1:
        S = suffix_products(M[1:])          # S[t] = M_{t+2} ... M_T  (0-based t+1 .. T-1)
        beta[:-1] = S.sum(axis=2)
        beta[:-1] /= beta[:-1].sum(axis=1, keepdims=True)
    g = f * beta
    g /= g.sum(axis=1, keepdims=True)
    xi = np.zeros((K, K))
    if T > 1:
        x = f[:-1, :, None] * P[None, :, :] * (b[1:] * beta[1:])[:, None, :]
        x /= x.sum(axis=(1, 2), keepdims=True)
        xi = x.sum(axis=0)
    return f, g, xi, ll


def expected_path(f_last: np.ndarray, P: np.ndarray, levels: np.ndarray, horizon: int) -> float:
    """Mean over h = 1..horizon of E[level_{s_{t+h}}] given Pr(s_t) = f_last."""
    p = np.asarray(f_last, float)
    acc = 0.0
    for _ in range(int(horizon)):
        p = p @ P
        acc += float(p @ levels)
    return acc / max(int(horizon), 1)


# ------------------------------------------------------------ emissions
def t_logpdf(x2: np.ndarray, scale2: np.ndarray, nu: float) -> np.ndarray:
    """log density of a zero-mean Student-t (scale^2, nu) at x with x^2 given; (T,) x (K,) -> (T, K)."""
    c = gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * math.log(nu * math.pi)
    return c - 0.5 * np.log(scale2)[None, :] - (nu + 1) / 2 * np.log1p(x2[:, None] / (nu * scale2[None, :]))


def trans_prior(K: int, stay: float = 9.0) -> np.ndarray:
    """Dirichlet pseudo-counts: prior mean persistence 0.9, worth 10 transitions per row."""
    c = np.full((K, K), 1.0 / max(K - 1, 1))
    np.fill_diagonal(c, stay)
    return c


def init_gamma_by_quantile(score: np.ndarray, K: int) -> np.ndarray:
    """One-hot initial responsibilities from the quantile bucket of ``score`` (NaN -> uniform)."""
    T = len(score)
    g = np.full((T, K), 1.0 / K)
    ok = np.isfinite(score)
    if ok.sum() >= K:
        qs = np.quantile(score[ok], np.linspace(0, 1, K + 1)[1:-1])
        lab = np.searchsorted(qs, score[ok], side="right")
        gg = np.zeros((ok.sum(), K))
        gg[np.arange(ok.sum()), lab] = 1.0
        g[ok] = 0.9 * gg + 0.1 / K
    return g


def rolling_mean_centered(v: np.ndarray, w: int) -> np.ndarray:
    """Centered rolling mean ignoring NaN (window ``w``)."""
    ok = np.isfinite(v)
    a = np.where(ok, v, 0.0)
    c = np.concatenate([[0.0], np.cumsum(a)])
    n = np.concatenate([[0.0], np.cumsum(ok)])
    T = len(v)
    lo = np.clip(np.arange(T) - w // 2, 0, T)
    hi = np.clip(np.arange(T) + (w - w // 2), 0, T)
    cnt = n[hi] - n[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(cnt > 0, (c[hi] - c[lo]) / np.maximum(cnt, 1), np.nan)


# ---------------------------------------------------- univariate MS-t fit
class MSFit:
    """A fitted K-state zero-mean Markov-switching Student-t variance model.

    ``scale2`` are the t scales used for filtering; ``var`` are the regime
    variances used for forecasting: the moment estimate E[x^2 | regime]
    (smoothed-weighted mean of x^2), which is unbiased whatever the true tail
    shape, whereas ``scale2 * nu/(nu-2)`` is not if the regime is not exactly t.
    """

    def __init__(self, P, scale2, var, nu, loglik, n_obs, n_iter, converged):
        self.P, self.scale2, self.var, self.nu = P, scale2, var, nu
        self.loglik, self.n_obs, self.n_iter, self.converged = loglik, n_obs, n_iter, converged

    @property
    def K(self):
        return len(self.var)

    def durations(self):
        return 1.0 / np.maximum(1.0 - np.diag(self.P), 1e-12)

    def logb(self, x: np.ndarray) -> np.ndarray:
        x2 = np.where(np.isfinite(x), x, 0.0) ** 2
        lb = t_logpdf(x2, self.scale2, self.nu)
        lb[~np.isfinite(x)] = 0.0
        return lb

    def forecast(self, x: np.ndarray, horizon: int) -> tuple[float, np.ndarray]:
        f, _ = filter_probs(self.logb(x), self.P)
        return expected_path(f[-1], self.P, self.var, horizon), f[-1]


def _sort_states(P, s2, order):
    return P[np.ix_(order, order)], s2[order]


def fit_ms_t(x: np.ndarray, K: int, nu: float = 5.0, max_iter: int = 300, tol: float = 1e-7,
             var_prior_n: float = 2.0, smooth_window: int = 6) -> MSFit:
    """EM (MAP) fit of a K-state zero-mean MS Student-t model to a 1-d series with NaN gaps.

    Two deterministic starts (quantiles of the centered rolling variance, and
    fixed multiples of the sample variance); the higher penalised likelihood wins.
    States are ordered by variance after every M-step (no label switching).
    """
    x = np.asarray(x, float)
    ok = np.isfinite(x)
    x2 = np.where(ok, x, 0.0) ** 2
    n = int(ok.sum())
    vbar = float(x2[ok].mean())
    s_ref = vbar * (nu - 2) / nu
    prior = trans_prior(K)

    def mstep(g, s2_prev):
        g = g * ok[:, None]
        if s2_prev is None:
            u = np.ones_like(g)
        else:
            u = (nu + 1) / (nu + x2[:, None] / s2_prev[None, :])
        num = (g * u * x2[:, None]).sum(axis=0) + var_prior_n * s_ref
        den = g.sum(axis=0) + var_prior_n
        return np.maximum(num / den, 1e-12 * vbar + 1e-300)

    def run(s2, P):
        ll_old = -np.inf
        it, conv = 0, False
        for it in range(1, max_iter + 1):
            lb = t_logpdf(x2, s2, nu)
            lb[~ok] = 0.0
            f, g, xi, ll = forward_backward(lb, P)
            pen = ll + float(((prior) * np.log(np.maximum(P, 1e-300))).sum())
            s2 = mstep(g, s2)
            Pc = xi + prior
            P = Pc / Pc.sum(axis=1, keepdims=True)
            order = np.argsort(s2)
            P, s2 = _sort_states(P, s2, order)
            if abs(pen - ll_old) < tol * max(n, 1):
                conv = True
                break
            ll_old = pen
        lb = t_logpdf(x2, s2, nu)
        lb[~ok] = 0.0
        f, g, xi, ll = forward_backward(lb, P)
        pen = ll + float((prior * np.log(np.maximum(P, 1e-300))).sum())
        gw = g * ok[:, None]
        var = ((gw * x2[:, None]).sum(axis=0) + var_prior_n * s2 * nu / (nu - 2)) / (gw.sum(axis=0) + var_prior_n)
        return MSFit(P, s2, var, nu, pen, n, it, conv)

    P0 = np.full((K, K), 0.05 / max(K - 1, 1))
    np.fill_diagonal(P0, 0.95)
    fits = []
    # start A: quantile buckets of the centered rolling variance
    g0 = init_gamma_by_quantile(rolling_mean_centered(np.where(ok, x2, np.nan), smooth_window), K)
    s2a = np.sort(mstep(g0, None))
    fits.append(run(s2a, P0.copy()))
    # start B: fixed multiples of the sample variance
    mult = {1: [1.0], 2: [0.6, 2.0], 3: [0.5, 1.2, 3.0]}.get(K, list(np.geomspace(0.5, 3.0, K)))
    fits.append(run(np.array(mult) * s_ref, P0.copy()))
    fits = [f for f in fits if np.isfinite(f.loglik) and np.all(np.isfinite(f.var)) and np.all(np.isfinite(f.P))]
    if not fits:
        raise FloatingPointError("MS fit failed")
    return max(fits, key=lambda f: f.loglik)


# ------------------------------------------- common regime, many assets
class CommonFit:
    """A K-state regime shared by all assets; asset-specific variance per regime.

    Emission: multivariate Student-t (nu) on the assets observed that month,
    scale matrix D_k R D_k with a FIXED correlation matrix R (so that highly
    correlated assets do not count as independent evidence) and asset scales
    D_k = diag(sqrt(scale2[:, k])). Missing assets are marginalised out (the
    marginal of a multivariate t is a multivariate t on the observed block).
    ``var[a, k]`` = E[x_a^2 | regime k] (moment estimate, shrunk towards
    ``level[a] * mult[k]`` with ``prior_n`` pseudo-months) for forecasting.
    """

    def __init__(self, assets, P, scale2, var, R, nu, loglik, n_iter, converged, level, mult):
        self.assets, self.P, self.scale2, self.var, self.R, self.nu = assets, P, scale2, var, R, nu
        self.loglik, self.n_iter, self.converged, self.level, self.mult = loglik, n_iter, converged, level, mult

    @property
    def K(self):
        return self.P.shape[0]

    def durations(self):
        return 1.0 / np.maximum(1.0 - np.diag(self.P), 1e-12)

    def logb(self, X: np.ndarray) -> np.ndarray:
        return _common_logb(X, self.scale2, self.R, self.nu)[0]

    def filter(self, X: np.ndarray) -> np.ndarray:
        return filter_probs(self.logb(X), self.P)[0]


def _patterns(ok: np.ndarray):
    keys = np.packbits(ok, axis=1)
    _, inv = np.unique(keys, axis=0, return_inverse=True)
    inv = np.asarray(inv).ravel()
    out = []
    for g in range(inv.max() + 1):
        rows = np.flatnonzero(inv == g)
        cols = np.flatnonzero(ok[rows[0]])
        out.append((rows, cols))
    return out


def _common_logb(X, scale2, R, nu, pats=None):
    """(logb (T,K), delta (T,K), d (T,)) for the common-regime multivariate t."""
    ok = np.isfinite(X)
    T, A = X.shape
    K = scale2.shape[1]
    pats = _patterns(ok) if pats is None else pats
    logb = np.zeros((T, K))
    delta = np.zeros((T, K))
    dvec = ok.sum(axis=1)
    for rows, cols in pats:
        d = len(cols)
        if d == 0:
            continue
        Ro = R[np.ix_(cols, cols)]
        Ri = np.linalg.inv(Ro)
        ld = float(np.linalg.slogdet(Ro)[1])
        c = gammaln((nu + d) / 2) - gammaln(nu / 2) - 0.5 * d * math.log(nu * math.pi) - 0.5 * ld
        Xo = X[np.ix_(rows, cols)]
        for k in range(K):
            s = np.sqrt(scale2[cols, k])
            Z = Xo / s[None, :]
            dk = np.einsum("ti,ij,tj->t", Z, Ri, Z)
            delta[rows, k] = dk
            logb[rows, k] = c - np.log(s).sum() - 0.5 * (nu + d) * np.log1p(dk / nu)
    return logb, delta, dvec


def _prop_fit(num: np.ndarray, den: np.ndarray, n_iter: int = 50) -> np.ndarray:
    """Weighted ML of var[a, k] = L_a * m_k from sufficient statistics
    (num = sum of weighted x^2, den = sum of weights), by alternating updates."""
    A, K = num.shape
    m = np.ones(K)
    L = num.sum(axis=1) / np.maximum(den.sum(axis=1), 1e-12)
    for _ in range(n_iter):
        L = (num / m[None, :]).sum(axis=1) / np.maximum(den.sum(axis=1), 1e-12)
        m = (num / np.maximum(L, 1e-300)[:, None]).sum(axis=0) / np.maximum(den.sum(axis=0), 1e-12)
        m = m / np.exp(np.mean(np.log(np.maximum(m, 1e-300))))
    return np.maximum(L[:, None] * m[None, :], 1e-300)


def fit_common_ms_t(X: np.ndarray, R: np.ndarray, K: int, nu: float = 5.0, max_iter: int = 300,
                    tol: float = 1e-7, prior_n: float = 12.0, smooth_window: int = 6,
                    assets=None, proportional: bool = False) -> CommonFit:
    """EM fit of the common-regime model.

    ``proportional=False``: free variance per asset and regime (shrunk towards
    level_a * mult_k with ``prior_n`` pseudo-months). ``proportional=True``:
    var[a, k] = level_a * mult_k exactly - one global risk multiplier that
    scales every asset alike.
    """
    X = np.asarray(X, float)
    ok = np.isfinite(X)
    X0 = np.where(ok, X, 0.0)
    X2 = X0 ** 2
    T, A = X.shape
    N = ok.sum(axis=0).astype(float)
    pats = _patterns(ok)
    prior = trans_prior(K)
    raw_level = X2.sum(axis=0) / np.maximum(N, 1)
    tf = (nu - 2) / nu

    def levels_mult(g, s2like):
        """Asset level net of the regime mix, and the common regime multipliers."""
        gw = g[:, None, :] * ok[:, :, None]                         # T, A, K
        occ = gw.sum(axis=0)                                        # A, K
        rel = s2like / np.maximum(s2like.mean(axis=1, keepdims=True), 1e-300)
        m = (occ * rel).sum(axis=0) / np.maximum(occ.sum(axis=0), 1e-12)
        pibar = g[ok.any(axis=1)].mean(axis=0)
        m = m / max(float(pibar @ m), 1e-300)
        mix = (g @ m)                                               # T
        L = X2.sum(axis=0) / np.maximum((ok * mix[:, None]).sum(axis=0), 1e-12)
        return L, m

    def mstep(g, s2_prev, delta, dvec):
        if s2_prev is None:
            u = np.ones((T, K))
        else:
            u = (nu + dvec[:, None]) / (nu + delta)
        w = g * u                                                   # T, K
        num = np.einsum("tk,ta->ak", w, X2)
        den = np.einsum("tk,ta->ak", g, ok.astype(float))
        raw = num / np.maximum(den, 1e-12)
        if proportional:
            return np.maximum(tf * _prop_fit(num, den), 1e-14)
        L, m = levels_mult(g, np.where(den > 0, raw, raw_level[:, None]))
        tgt = tf * L[:, None] * m[None, :]
        s2 = (num + prior_n * tgt) / (den + prior_n)
        return np.maximum(s2, 1e-14)

    def run(s2, P):
        ll_old, it, conv = -np.inf, 0, False
        for it in range(1, max_iter + 1):
            lb, delta, dvec = _common_logb(X, s2, R, nu, pats)
            f, g, xi, ll = forward_backward(lb, P)
            pen = ll + float((prior * np.log(np.maximum(P, 1e-300))).sum())
            s2 = mstep(g, s2, delta, dvec)
            Pc = xi + prior
            P = Pc / Pc.sum(axis=1, keepdims=True)
            order = np.argsort(np.median(s2 / s2.mean(axis=1, keepdims=True), axis=0))
            P, s2 = P[np.ix_(order, order)], s2[:, order]
            if abs(pen - ll_old) < tol * T:
                conv = True
                break
            ll_old = pen
        lb, delta, dvec = _common_logb(X, s2, R, nu, pats)
        f, g, xi, ll = forward_backward(lb, P)
        pen = ll + float((prior * np.log(np.maximum(P, 1e-300))).sum())
        num = np.einsum("tk,ta->ak", g, X2)
        den = np.einsum("tk,ta->ak", g, ok.astype(float))
        raw = np.where(den > 0, num / np.maximum(den, 1e-12), raw_level[:, None])
        L, m = levels_mult(g, raw)
        var = (num + prior_n * L[:, None] * m[None, :]) / (den + prior_n)
        if proportional:
            var = _prop_fit(num, den)
            m = var[0] / np.exp(np.mean(np.log(var[0])))
            L = var[:, 0] / m[0]
        return CommonFit(list(assets) if assets is not None else list(range(A)), P, s2, var, R, nu,
                         pen, it, conv, L, m)

    P0 = np.full((K, K), 0.05 / max(K - 1, 1))
    np.fill_diagonal(P0, 0.95)
    score = np.where(ok, X2 / np.maximum(raw_level, 1e-300)[None, :], np.nan)
    with np.errstate(invalid="ignore"):
        sc = np.nanmean(np.where(ok, score, np.nan), axis=1)
    sc[~ok.any(axis=1)] = np.nan
    g0 = init_gamma_by_quantile(rolling_mean_centered(sc, smooth_window), K)
    fits = [run(mstep(g0, None, None, None), P0.copy())]
    mult = {1: [1.0], 2: [0.6, 2.0], 3: [0.5, 1.2, 3.0]}.get(K, list(np.geomspace(0.5, 3.0, K)))
    fits.append(run(tf * raw_level[:, None] * np.array(mult)[None, :], P0.copy()))
    fits = [f for f in fits if np.isfinite(f.loglik) and np.all(np.isfinite(f.var)) and np.all(np.isfinite(f.P))]
    if not fits:
        raise FloatingPointError("common MS fit failed")
    return max(fits, key=lambda f: f.loglik)
