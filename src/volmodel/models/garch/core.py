"""Numerical core of the GARCH family: linear variance filters, Student-t
likelihoods, bounded estimation with fallbacks, and closed-form aggregated
multi-step forecasts.

Everything here is plain numpy/scipy and frequency-agnostic: a "step" is a
day, a 5-day block or a month depending on the caller. Returns are zero-mean
(uncentered, like the target) and already on the 21-day scale (see
``prep``).

Speed: every variance recursion used here is LINEAR in the squared shocks
once the parameters are fixed (GARCH, GJR, Engle-Lee component GARCH, the
unit-mean GARCH of GARCH-MIDAS), so it is evaluated with
``scipy.signal.lfilter`` (C loop) instead of a Python loop. The 2-D component
model is diagonalized first (its transition matrix always has real
eigenvalues; see :func:`cgarch_filter`).
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize
from scipy.signal import lfilter
from scipy.special import gammaln

NU_BOUNDS = (2.5, 60.0)   # Student-t degrees of freedom: 2.5 keeps the variance finite with margin
NU_START = 8.0


# ------------------------------------------------------------------ helpers
def fingerprint(*arrays) -> str:
    h = hashlib.blake2b(digest_size=16)
    for a in arrays:
        a = np.ascontiguousarray(np.asarray(a, dtype=float))
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def ar1_path(x: np.ndarray, beta: float, y0: float) -> np.ndarray:
    """y[0] = y0, y[n] = beta * y[n-1] + x[n-1]; returns len(x) + 1 values."""
    n = len(x)
    y = np.empty(n + 1)
    y[0] = y0
    if n:
        z = lfilter([1.0], [1.0, -beta], x)
        y[1:] = z
        if y0 != 0.0 and beta != 0.0:
            # beta^k only while it matters (avoids slow denormal arithmetic)
            m = min(n, int(745.0 / max(-math.log(abs(beta)), 1e-300)) + 1)
            k = np.arange(1, m + 1, dtype=float)
            pw = np.exp(k * math.log(abs(beta)))
            if beta < 0:
                pw = pw * np.where(k % 2 == 1, -1.0, 1.0)
            y[1:m + 1] += pw * y0
    return y


def geo_sum(p: float, H: float) -> float:
    """sum_{h=0}^{H-1} p^h for real H >= 0 (p in [0, 1])."""
    if abs(1.0 - p) < 1e-12:
        return float(H)
    return (1.0 - p ** H) / (1.0 - p)


def t_nll(e2: np.ndarray, s2: np.ndarray, nu: float) -> float:
    """Negative log-likelihood of zero-mean standardized Student-t shocks."""
    c = gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * math.log(math.pi * (nu - 2))
    return -(len(e2) * c - 0.5 * np.sum(np.log(s2))
             - (nu + 1) / 2 * np.sum(np.log1p(e2 / (s2 * (nu - 2)))))


def half_life(p: float) -> float:
    """Half-life in steps of a geometric decay with persistence p."""
    if not (0 < p < 1):
        return float("inf") if p >= 1 else 0.0
    return math.log(0.5) / math.log(p)


def p_from_half_life(hl_steps: float) -> float:
    return 0.5 ** (1.0 / max(hl_steps, 1e-9))


# ------------------------------------------------------------------- specs
@dataclass
class Spec:
    """A conditional-variance model with variance targeting.

    ``theta`` is the free parameter vector (bounded box, L-BFGS-B); the
    long-run variance ``vbar`` is NOT a parameter: it is the sample mean
    square of the estimation/filtering window (variance targeting), so the
    level of every forecast is anchored to data rather than to an optimizer.
    """
    name: str
    steps_per_year: float
    bounds: list = field(default_factory=list)
    starts: list = field(default_factory=list)

    # subclasses implement
    def unpack(self, theta) -> dict: ...
    def filter(self, e: np.ndarray, vbar: float, prm: dict, extra=None): ...
    def agg_forecast(self, state, vbar: float, prm: dict, H: float) -> float: ...

    def nll(self, theta, e, vbar, extra=None) -> float:
        prm = self.unpack(theta)
        s2, _ = self.filter(e, vbar, prm, extra)
        s2 = np.maximum(s2[:-1], 1e-8 * vbar)
        v = t_nll(e * e, s2, prm["nu"])
        return v if np.isfinite(v) else 1e300


def _hl_bounds(spy: float, lo_years: float, hi_years: float):
    return p_from_half_life(lo_years * spy), p_from_half_life(hi_years * spy)


class Garch11(Spec):
    """GARCH(1,1): s2[t+1] = w + a e[t]^2 + b s2[t], w = (1 - a - b) vbar.

    theta = (p, share, nu): p = a + b, a = share * p.
    """

    def __init__(self, spy: float):
        lo, hi = _hl_bounds(spy, 1 / spy, 20.0)    # half-life of a shock: one step .. 20 years
        lo = max(lo, 0.3)
        super().__init__("garch11", spy, [(lo, hi), (0.005, 0.6), NU_BOUNDS], [])
        self.starts = [
            [np.clip(p_from_half_life(spy / 4), lo, hi), 0.06, NU_START],   # 3-month half-life
            [np.clip(p_from_half_life(spy / 12), lo, hi), 0.12, NU_START],  # 1 month
            [np.clip(p_from_half_life(spy), lo, hi), 0.04, NU_START],       # 1 year
        ]

    def unpack(self, th):
        p, sh, nu = th
        return {"alpha": sh * p, "beta": (1 - sh) * p, "p": p, "nu": nu}

    def filter(self, e, vbar, prm, extra=None):
        p = prm["p"]
        x = (1 - p) * vbar + prm["alpha"] * e * e
        s2 = ar1_path(x, prm["beta"], vbar)
        return s2, s2[-1]

    def agg_forecast(self, state, vbar, prm, H):
        return H * vbar + (state - vbar) * geo_sum(prm["p"], H)


class GJR(Spec):
    """GJR-GARCH(1,1): s2[t+1] = w + (a + g 1[e<0]) e[t]^2 + b s2[t],
    persistence p = a + g/2 + b (symmetric innovations), w = (1 - p) vbar.

    theta = (p, share, lev, nu): a + g/2 = share * p, g = 2 share p lev.
    """

    def __init__(self, spy: float):
        g = Garch11(spy)
        super().__init__("gjr", spy, [g.bounds[0], (0.005, 0.6), (0.0, 1.0), NU_BOUNDS], [])
        self.starts = [[s[0], s[1], 0.5, s[2]] for s in g.starts]

    def unpack(self, th):
        p, sh, lev, nu = th
        return {"alpha": sh * p * (1 - lev), "gamma": 2 * sh * p * lev, "beta": (1 - sh) * p,
                "p": p, "nu": nu}

    def filter(self, e, vbar, prm, extra=None):
        p = prm["p"]
        e2 = e * e
        x = (1 - p) * vbar + prm["alpha"] * e2 + prm["gamma"] * np.where(e < 0, e2, 0.0)
        s2 = ar1_path(x, prm["beta"], vbar)
        return s2, s2[-1]

    def agg_forecast(self, state, vbar, prm, H):
        return H * vbar + (state - vbar) * geo_sum(prm["p"], H)


def cgarch_filter(e2, vbar, rho, phi, alpha, beta, q0=None):
    """Engle-Lee (1999) component GARCH, written in (q, s = s2 - q):

        q[t+1] = w + rho q[t] + phi (e[t]^2 - s2[t]),       w = (1 - rho) vbar
        s[t+1] = (alpha + beta) s[t] + alpha (e[t]^2 - s2[t])

    i.e. x[t+1] = A x[t] + B e[t]^2 + c with
    A = [[rho - phi, -phi], [-alpha, beta]], B = (phi, alpha), c = (w, 0).
    A has real eigenvalues (discriminant (rho - phi - beta)^2 + 4 phi alpha
    >= 0), so it is diagonalized and each mode is one ``lfilter`` pass.
    Returns (s2, q), both of length len(e2) + 1 (last = one-step forecast).
    """
    w = (1 - rho) * vbar
    A = np.array([[rho - phi, -phi], [-alpha, beta]])
    B = np.array([phi, alpha])
    c = np.array([w, 0.0])
    x0 = np.array([vbar if q0 is None else q0, 0.0])
    lam, V = np.linalg.eig(A)
    lam, V = lam.real, V.real
    if np.linalg.cond(V) > 1e8:          # near-defective: plain loop (rare)
        n = len(e2)
        X = np.empty((n + 1, 2))
        X[0] = x0
        for i in range(n):
            X[i + 1] = A @ X[i] + B * e2[i] + c
    else:
        Vi = np.linalg.inv(V)
        bz, cz, z0 = Vi @ B, Vi @ c, Vi @ x0
        Z = np.column_stack([ar1_path(bz[k] * e2 + cz[k], lam[k], z0[k]) for k in range(2)])
        X = Z @ V.T
    q = X[:, 0]
    return X[:, 0] + X[:, 1], q


class CGarch(Spec):
    """Engle-Lee component GARCH with variance targeting (q-bar = vbar).

    theta = (rho, u1, u2, u3, nu): alpha + beta = rho * u1 (< rho),
    alpha = (alpha + beta) * u2, phi = beta * u3 (< beta): the Engle-Lee
    sufficient conditions for positivity / stationarity hold by construction.
    rho is bounded so that the long-run component's half-life lies between
    3 months and 50 years.
    """

    def __init__(self, spy: float):
        lo, hi = _hl_bounds(spy, 0.25, 50.0)
        super().__init__("cgarch", spy, [(lo, hi), (0.01, 0.995), (0.01, 0.99), (0.001, 0.999), NU_BOUNDS],
                         [])
        r2y = p_from_half_life(2 * spy)        # long-run half-life 2 years
        r6m = p_from_half_life(0.5 * spy)
        tr1m = p_from_half_life(spy / 12)      # transitory half-life 1 month
        tr1w = p_from_half_life(spy / 52)
        self.starts = [
            [np.clip(r2y, lo, hi), min(tr1m / r2y, 0.99), 0.4, 0.2, NU_START],
            [np.clip(r6m, lo, hi), min(tr1w / r6m, 0.99), 0.5, 0.2, NU_START],
            [np.clip(p_from_half_life(8 * spy), lo, hi), min(tr1m / p_from_half_life(8 * spy), 0.99),
             0.3, 0.05, NU_START],
        ]

    def unpack(self, th):
        rho, u1, u2, u3, nu = th
        ab = rho * u1
        a = ab * u2
        b = ab - a
        return {"rho": rho, "alpha": a, "beta": b, "phi": b * u3, "ab": ab, "nu": nu}

    def filter(self, e, vbar, prm, extra=None):
        s2, q = cgarch_filter(e * e, vbar, prm["rho"], prm["phi"], prm["alpha"], prm["beta"])
        return s2, (s2[-1], q[-1])

    def agg_forecast(self, state, vbar, prm, H):
        s2n, qn = state
        return (H * vbar + (qn - vbar) * geo_sum(prm["rho"], H)
                + (s2n - qn) * geo_sum(prm["ab"], H))


# ------------------------------------------------------------- GARCH-MIDAS
MIDAS_K = 36  # months of realized variance feeding the long-run component (Engle, Ghysels & Sohn 2013 use ~3 years)


def beta_weights(w: float, K: int = MIDAS_K) -> np.ndarray:
    k = np.arange(1, K + 1)
    x = (1.0 - (k - 0.5) / K) ** (w - 1.0)
    return x / x.sum()


def midas_logx(rv_m: np.ndarray, w: float, K: int = MIDAS_K, min_months: int = 12) -> np.ndarray:
    """For month index j, log of the beta-weighted average of rv_m[j], rv_m[j-1], ...,
    rv_m[j-K+1] (renormalized over available months; NaN if fewer than ``min_months``)."""
    phi = beta_weights(w, K)
    ok = np.isfinite(rv_m) & (rv_m > 0)
    v = np.where(ok, rv_m, 0.0)
    num = np.convolve(v, phi)[: len(v)]
    den = np.convolve(ok.astype(float), phi)[: len(v)]
    cnt = np.convolve(ok.astype(float), np.ones(K))[: len(v)]
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.log(num / den)
    out[(cnt < min_months) | ~np.isfinite(out)] = np.nan
    return out


class Midas(Spec):
    """GARCH-MIDAS (Engle, Ghysels & Sohn 2013), rolling-window form:

        e[d] = sqrt(tau[m(d)] g[d]) z[d]
        g[d+1] = (1 - p) + a e[d]^2 / tau[m(d)] + b g[d],       p = a + b
        log tau[m] = log(vbar) + m0 + theta (log X[m-1] - mean log X)

    X[m-1] = beta-weighted average (36 months) of the per-step realized
    variance of the months before m. ``extra`` = (month index of each step,
    log X of each month's PREVIOUS months, i.e. aligned so that extra[1][j]
    is the driver of month j). theta = (p, share, m0, theta, w, nu).
    """

    def __init__(self, spy: float):
        g = Garch11(spy)
        super().__init__("midas", spy, [g.bounds[0], (0.005, 0.6), (-2.0, 2.0), (0.0, 1.5), (1.0, 40.0),
                                         NU_BOUNDS], [])
        self.starts = [[s[0], s[1], 0.0, 0.7, 5.0, s[2]] for s in g.starts[:2]]
        self.starts.append([g.starts[0][0], g.starts[0][1], 0.0, 1.0, 1.0, NU_START])

    def unpack(self, th):
        p, sh, m0, th_, w, nu = th
        return {"alpha": sh * p, "beta": (1 - sh) * p, "p": p, "m0": m0, "theta": th_, "w": w, "nu": nu}

    def tau(self, vbar, prm, logx_prev, center):
        return vbar * np.exp(prm["m0"] + prm["theta"] * (logx_prev - center))

    def filter(self, e, vbar, prm, extra=None):
        midx, rv_m, center = extra        # month index per step, per-step RV by month, centering constant
        lx = midas_logx(rv_m, prm["w"])
        lx_prev = np.concatenate([[np.nan], lx[:-1]])           # driver of month j = months < j
        lx_prev = np.where(np.isfinite(lx_prev), lx_prev, center)
        tau_m = self.tau(vbar, prm, lx_prev, center)
        tau_d = tau_m[midx]
        u = e * e / tau_d
        g = ar1_path((1 - prm["p"]) + prm["alpha"] * u, prm["beta"], 1.0)
        tau_next = self.tau(vbar, prm, lx[-1] if np.isfinite(lx[-1]) else center, center)
        s2 = np.empty(len(e) + 1)
        s2[:-1] = tau_d * g[:-1]
        s2[-1] = tau_next * g[-1]           # first step of the next month (origin is a month-end)
        return s2, (tau_next, g[-1])

    def agg_forecast(self, state, vbar, prm, H):
        tau_n, g_n = state
        return tau_n * (H + (g_n - 1.0) * geo_sum(prm["p"], H))


# -------------------------------------------------------------- estimation
@dataclass
class Fit:
    theta: np.ndarray
    nll: float
    ok: bool
    note: str = ""


def estimate(spec: Spec, nll_fn, starts=None, maxiter: int = 200) -> Fit:
    """Minimize ``nll_fn(theta)`` over the spec's box from several starting
    values; the best finite optimum wins. Never raises: if every start fails,
    the first start (a conventional parameterization) is returned, ok=False."""
    starts = starts or spec.starts
    lo = np.array([b[0] for b in spec.bounds])
    hi = np.array([b[1] for b in spec.bounds])
    best = None
    for s0 in starts:
        x0 = np.clip(np.asarray(s0, float), lo + 1e-9, hi - 1e-9)
        try:
            f0 = nll_fn(x0)
            r = minimize(nll_fn, x0, method="L-BFGS-B", bounds=list(zip(lo, hi)),
                         options={"maxiter": maxiter})
            x, f = (r.x, r.fun) if np.isfinite(r.fun) and r.fun <= f0 else (x0, f0)
        except Exception:                               # noqa: BLE001 - never raise
            continue
        if np.isfinite(f) and np.all(np.isfinite(x)) and (best is None or f < best.nll):
            best = Fit(np.clip(x, lo, hi), float(f), True)
    if best is None:
        x0 = np.clip(np.asarray(starts[0], float), lo + 1e-9, hi - 1e-9)
        return Fit(x0, float("nan"), False, "all starts failed; default parameters")
    return best


# ------------------------------------------------------- long memory: RM2006
class RM2006:
    """RiskMetrics 2006 long-memory ARCH (Zumbach 2006), fixed parameters.

    K = 15 EWMAs of squared returns with characteristic times
    tau_k = tau_1 rho^(k-1) (tau_1 = 4, rho = sqrt 2, tau_K = 512 steps),
    s2_k[t+1] = mu_k s2_k[t] + (1 - mu_k) e[t]^2, mu_k = exp(-1/tau_k),
    combined with logarithmically decaying weights
    w_k = (1 - ln tau_k / ln tau_0) / C, tau_0 = 1560 steps (6 years), which
    mimics a hyperbolic (long-memory) autocorrelation of volatility. All
    constants are the published ones (in trading days); nothing is fitted.
    Multi-step forecasts use E[e^2] = sum_k w_k s2_k, i.e. the linear
    system x[t+1] = M x[t], M = diag(mu) + (1 - mu) w'.
    """

    tau0, tau1, taumax, rho = 1560.0, 4.0, 512.0, math.sqrt(2.0)

    def __init__(self, step_days: float = 1.0):
        K = int(round(1 + math.log(self.taumax / self.tau1) / math.log(self.rho)))
        tau = self.tau1 * self.rho ** np.arange(K) / step_days
        self.mu = np.exp(-1.0 / tau)
        w = 1.0 - np.log(tau * step_days) / math.log(self.tau0)
        self.w = w / w.sum()
        self.M = np.diag(self.mu) + np.outer(1 - self.mu, self.w)

    def components(self, e2: np.ndarray, x0: float) -> np.ndarray:
        """One-step-ahead component values after the last observation."""
        return np.array([ar1_path((1 - m) * e2, m, x0)[-1] for m in self.mu])

    def agg_forecast(self, x1: np.ndarray, H: float) -> float:
        n = int(math.floor(H))
        tot, x = 0.0, x1.copy()
        for _ in range(n):
            tot += self.w @ x
            x = self.M @ x
        return tot + (H - n) * float(self.w @ x)
