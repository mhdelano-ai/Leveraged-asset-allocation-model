"""State-space models of latent log variance (see the package docstring).

Shared conventions
------------------
* ``h_{i,t}`` = log of asset i's variance in calendar month t (the monthly
  squared log excess return has conditional mean ``exp(h)``). The 12-month
  target is the sum of the next 12 monthly variances, so the forecast is

      V_{i,t} = s * sum_{k=1..12} E[exp(h_{i,t+k}) | F_t]
              = s * sum_k exp(m_k + v_k / 2)

  with ``(m_k, v_k)`` the k-step predictive mean and variance of ``h`` from
  the filter (the lognormal retransformation), and ``s`` the level
  calibration below.
* Parameters are estimated by maximum (quasi-)likelihood on the view as of
  the last December (``refit`` years apart, anchored to the calendar), using
  every month of every asset up to that date; the state is then updated
  month by month to the origin. A fit depends only on ``view.asof(E)``, so a
  cold call gives the same answer as a warm one (fits are cached by a hash
  of that view).
* ``s`` (level calibration): the Gaussian filter works with
  ``y1 = log(r^2 + .) - KAPPA`` where KAPPA is the mean for NORMAL returns.
  Fat tails given ``h`` shift that mean, and the quasi-likelihood filter's
  variances are approximate, so ``s`` is estimated at each fit as the
  QLIKE-optimal constant for the model's own one-step forecasts,
  ``s = mean(r^2 / E_{t-1} exp(h_t))``, pooled over assets (each asset's first
  36 months excluded). Under a correct model s = 1; it is the estimated
  version of the "log chi-square mean correction".
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from ...universe import BY_ID
from ..base import VolForecaster
from . import grid as G
from .data import SIGMA1_NORMAL, build_obs, fingerprint
from .kalman import kf_multi, kf_univ

HORIZON = 12
MU0 = float(np.log(0.15 ** 2 / 12))   # a 15% vol asset: centre of the vague prior
TAU0 = 1.5                            # vague prior sd of mu (3%..70% vol at 2 sd)
QC = 0.02 ** 2                        # offset random walk: ~0.3 log-var drift in 20 years
TAUC2 = 0.4 ** 2                      # offset prior / splice reset: the 1d/21d ratio range
CALIB_BURN = 36
MIN_POOLED_OBS = 240
MIN_ASSET_OBS = 120
FD_STEP = 1e-4
MAXITER = 100                         # L-BFGS-B iterations (per-asset fit: twice that)

_FIT_CACHE: dict = {}


# --------------------------------------------------------------------------- specs
@dataclass(frozen=True)
class Spec:
    """A univariate-filter model: ncomp AR(1) components, optional daily obs."""
    ncomp: int = 1
    daily: bool = False
    pooled: bool = True
    refit: int = 1

    @property
    def key(self):
        return ("univ", self.ncomp, self.daily, self.pooled, self.refit)

    @property
    def n_state(self):
        return 1 + self.ncomp + int(self.daily)

    @property
    def ic(self):
        return self.ncomp + 1 if self.daily else None

    def param_table(self):
        """(names, start, lower, upper, free) for the full parameter vector."""
        rows = []
        if self.ncomp == 1:
            rows += [("phi1", 0.95, 0.5, 0.998, True), ("lsig1", np.log(0.2), np.log(0.01), np.log(1.5), True)]
        else:
            rows += [("phi1", 0.98, 0.9, 0.999, True), ("lsig1", np.log(0.08), np.log(0.005), np.log(1.0), True),
                     ("phi2", 0.6, 0.0, 0.9, True), ("lsig2", np.log(0.25), np.log(0.01), np.log(1.5), True)]
        rows += [("lH1", np.log(SIGMA1_NORMAL), np.log(1.0), np.log(10.0), True)]
        if self.daily:
            rows += [("lH2", np.log(0.15), np.log(0.01), np.log(2.0), True)]
        rows += [("mubar", MU0, -14.0, 0.0, self.pooled), ("ltau", np.log(TAU0 if not self.pooled else 1.0),
                                                           np.log(0.05), np.log(3.0), self.pooled)]
        names = [r[0] for r in rows]
        return (names, np.array([r[1] for r in rows]), np.array([r[2] for r in rows]),
                np.array([r[3] for r in rows]), np.array([r[4] for r in rows]))

    def arrays(self, Th):
        """State-space matrices for parameter rows Th (B, npar)."""
        names = self.param_table()[0]
        ix = {nm: k for k, nm in enumerate(names)}
        B = Th.shape[0]
        n = self.n_state
        d = np.ones((B, n))
        q = np.zeros((B, n))
        a0 = np.zeros((B, n))
        P0 = np.zeros((B, n, n))
        a0[:, 0] = Th[:, ix["mubar"]]
        P0[:, 0, 0] = np.exp(2 * Th[:, ix["ltau"]])
        for k in range(self.ncomp):
            phi = Th[:, ix[f"phi{k + 1}"]]
            s2 = np.exp(2 * Th[:, ix[f"lsig{k + 1}"]])
            d[:, 1 + k] = phi
            q[:, 1 + k] = s2
            P0[:, 1 + k, 1 + k] = s2 / (1 - phi ** 2)
        if self.daily:
            P0[:, self.ic, self.ic] = TAUC2
        H1 = np.exp(Th[:, ix["lH1"]])
        H2 = np.exp(Th[:, ix["lH2"]]) if self.daily else np.ones(B)
        Z1 = np.zeros(n)
        Z1[: 1 + self.ncomp] = 1.0
        Z2 = Z1.copy()
        if self.daily:
            Z2[self.ic] = 1.0
        return d, q, a0, P0, H1, H2, Z1, Z2


# --------------------------------------------------------------------------- helpers
def estimation_date(asof, months: pd.DatetimeIndex, refit: int) -> pd.Timestamp:
    t = pd.Timestamp(asof if asof is not None else months[-1])
    y = t.year if (t.month == 12 and t.day == 31) else t.year - 1
    y -= y % refit
    return pd.Timestamp(f"{y}-12-31")


def _calibrate(r2, pm, pv, nobs):
    """QLIKE-optimal level constant for one-step forecasts, pooled."""
    vh = np.exp(pm + 0.5 * pv)
    prev = np.vstack([np.zeros((1, nobs.shape[1])), nobs[:-1]])
    ok = np.isfinite(r2) & np.isfinite(vh) & (prev >= CALIB_BURN)
    if ok.sum() < 24:
        return 1.0
    return float(np.mean(r2[ok] / vh[ok]))


def clear_cache():
    _FIT_CACHE.clear()


def _cached(key, fn):
    if key in _FIT_CACHE:
        return _FIT_CACHE[key]
    out = fn()
    if len(_FIT_CACHE) > 400:
        _FIT_CACHE.pop(next(iter(_FIT_CACHE)))
    _FIT_CACHE[key] = out
    return out


# --------------------------------------------------------------------------- univariate fit
def _univ_ll(spec, Th_assets, obs, s0, s1, cols):
    """Loglik per instance; Th_assets (B, npar), instance b -> asset cols[b % len(cols)]."""
    A = len(cols)
    B = Th_assets.shape[0]
    reps = B // A
    d, q, a0, P0, H1, H2, Z1, Z2 = spec.arrays(Th_assets)
    Y1 = np.tile(obs.y1[s0:s1, cols], (1, reps))
    Y2 = np.tile(obs.y2[s0:s1, cols], (1, reps)) if spec.daily else None
    dact = np.tile(obs.dact[s0:s1, cols], (1, reps))
    reset = np.tile(obs.reset[s0:s1, cols], (1, reps))
    return kf_univ(d, q, a0, P0, H1, H2, Y1, Y2, dact, reset, QC, TAUC2, Z1, Z2, spec.ic)[0]


def fit_univ(spec: Spec, view_E) -> dict:
    t0 = time.perf_counter()
    obs = build_obs(view_E)
    A = len(obs.assets)
    names, start, lo, hi, free = spec.param_table()
    npar = len(names)
    any_data = np.isfinite(obs.y1).any(1)
    s0 = int(np.argmax(any_data)) if any_data.any() else 0
    s1 = len(obs.months)
    nobs_asset = np.isfinite(obs.y1).sum(0)
    Th = np.tile(start, (A, 1))
    fidx = np.flatnonzero(free)
    info = {"nit": 0, "nfev": 0, "success": None}
    if spec.pooled and nobs_asset.sum() >= MIN_POOLED_OBS:
        N = float(nobs_asset.sum() + (np.isfinite(obs.y2).sum() if spec.daily else 0))
        k = len(fidx)
        act = np.flatnonzero(nobs_asset > 0)

        def f(x):
            base = start.copy()
            base[fidx] = x
            rows = [base]
            for j in range(k):
                b = base.copy()
                b[fidx[j]] += FD_STEP
                rows.append(b)
            Tb = np.repeat(np.array(rows), len(act), axis=0)   # sets are contiguous
            ll = _univ_ll(spec, Tb, obs, s0, s1, act).reshape(k + 1, len(act)).sum(1)
            info["nfev"] += 1
            return -ll[0] / N, -(ll[1:] - ll[0]) / FD_STEP / N

        res = minimize(f, start[fidx], jac=True, method="L-BFGS-B",
                       bounds=list(zip(lo[fidx], hi[fidx])),
                       options={"maxiter": MAXITER, "ftol": 1e-7, "gtol": 1e-5})
        best = start.copy()
        best[fidx] = res.x
        Th = np.tile(best, (A, 1))
        info.update(nit=int(res.nit), success=bool(res.success))
    elif not spec.pooled:
        est = np.flatnonzero(nobs_asset >= MIN_ASSET_OBS)
        if est.size:
            k = len(fidx)
            N = float(nobs_asset[est].sum())

            ne = len(est)

            def f(x):
                base = np.tile(start, (ne, 1))
                base[:, fidx] = x.reshape(ne, k)
                sets = [base]
                for j in range(k):
                    b = base.copy()
                    b[:, fidx[j]] += FD_STEP
                    sets.append(b)
                ll = _univ_ll(spec, np.vstack(sets), obs, s0, s1, est).reshape(k + 1, ne)
                info["nfev"] += 1
                g = -(ll[1:] - ll[0]) / FD_STEP / N         # (k, n_est)
                return -ll[0].sum() / N, g.T.ravel()

            x0 = np.tile(start[fidx], len(est))
            bnds = list(zip(np.tile(lo[fidx], len(est)), np.tile(hi[fidx], len(est))))
            res = minimize(f, x0, jac=True, method="L-BFGS-B", bounds=bnds,
                           options={"maxiter": 2 * MAXITER, "ftol": 1e-7, "gtol": 1e-5})
            Th[np.ix_(est, fidx)] = res.x.reshape(len(est), k)
            info.update(nit=int(res.nit), success=bool(res.success))
    # final pass: state at E and one-step predictions for the calibration
    d, q, a0, P0, H1, H2, Z1, Z2 = spec.arrays(Th)
    ll, a, P, (pm, pv) = kf_univ(d, q, a0, P0, H1, H2, obs.y1[s0:s1],
                                 obs.y2[s0:s1] if spec.daily else None,
                                 obs.dact[s0:s1], obs.reset[s0:s1], QC, TAUC2, Z1, Z2, spec.ic,
                                 store=True)
    s = _calibrate(obs.r2[s0:s1], pm, pv, obs.nobs[s0:s1])
    return dict(theta=Th, names=names, a=a, P=P, s=s, E=obs.months[-1] if len(obs.months) else None,
                nE=s1, loglik=float(ll.sum()), seconds=time.perf_counter() - t0, **info)


def forecast_univ(spec: Spec, fit: dict, obs, nt: int, assets: list[str]) -> np.ndarray:
    """Annual variance forecast for every asset in obs.assets order."""
    Th = fit["theta"]
    d, q, a0, P0, H1, H2, Z1, Z2 = spec.arrays(Th)
    a, P = fit["a"], fit["P"]
    nE = fit["nE"]
    if nt > nE:
        _, a, P, _ = kf_univ(d, q, a, P, H1, H2, obs.y1[nE:nt],
                             obs.y2[nE:nt] if spec.daily else None,
                             obs.dact[nE:nt], obs.reset[nE:nt], QC, TAUC2, Z1, Z2, spec.ic)
    dd = d[:, :, None] * d[:, None, :]
    n = a.shape[1]
    diag = np.arange(n)
    tot = np.zeros(a.shape[0])
    a = a.copy()
    P = P.copy()
    for _ in range(HORIZON):
        a = a * d
        P = P * dd
        P[:, diag, diag] += q
        m = a @ Z1
        v = (P @ Z1) @ Z1
        tot += np.exp(m + 0.5 * v)
    return fit["s"] * tot


# --------------------------------------------------------------------------- common factor
# corr(log(z1^2 + d), log(z2^2 + d)) for corr(z1, z2) = 0, 0.05, ..., 1 (2e6 draws)
_G1 = np.array([0.0, 0.0014, 0.0053, 0.012, 0.0215, 0.0336, 0.0486, 0.0667, 0.088, 0.1129,
                0.1418, 0.1746, 0.2122, 0.2554, 0.3054, 0.3632, 0.4318, 0.514, 0.6164, 0.754, 1.0])
FACTOR_GROUPS = ["equity", "rates", "credit", "real"]


def noise_corr(rho: np.ndarray):
    """Measurement-noise correlations implied by return correlations rho:
    (log chi-square noise of y1, log RV noise of y2 ~ rho^2)."""
    r = np.clip(np.abs(rho), 0, 1)
    g1 = np.interp(r, np.linspace(0, 1, 21), _G1)
    np.fill_diagonal(g1, 1.0)
    g2 = r ** 2
    np.fill_diagonal(g2, 1.0)
    return g1, g2


@dataclass(frozen=True)
class FactorSpec:
    """h_i = mu_i + lambda_g(i) f + x_i ; f, x_i AR(1); optional daily obs."""
    daily: bool = True
    refit: int = 1

    @property
    def key(self):
        return ("factor", self.daily, self.refit)

    def param_table(self):
        rows = [("phif", 0.97, 0.5, 0.998, True)]
        rows += [(f"lam_{g}", 0.4, 0.0, 2.0, True) for g in FACTOR_GROUPS]
        rows += [("phi1", 0.95, 0.3, 0.998, True), ("lsig1", np.log(0.15), np.log(0.01), np.log(1.5), True),
                 ("lH1", np.log(SIGMA1_NORMAL), np.log(1.0), np.log(10.0), True)]
        if self.daily:
            rows += [("lH2", np.log(0.15), np.log(0.01), np.log(2.0), True)]
        rows += [("mubar", MU0, -14.0, 0.0, True), ("ltau", 0.0, np.log(0.05), np.log(3.0), True)]
        names = [r[0] for r in rows]
        return (names, np.array([r[1] for r in rows]), np.array([r[2] for r in rows]),
                np.array([r[3] for r in rows]), np.array([r[4] for r in rows]))

    def arrays(self, Th, assets, rho):
        names = self.param_table()[0]
        ix = {nm: k for k, nm in enumerate(names)}
        B = Th.shape[0]
        A = len(assets)
        n = 1 + 2 * A + (A if self.daily else 0)
        p = A * (2 if self.daily else 1)
        mu_i, x_i, c_i = 1 + np.arange(A), 1 + A + np.arange(A), 1 + 2 * A + np.arange(A)
        grp = np.array([FACTOR_GROUPS.index(BY_ID[a].group) if a in BY_ID else 0 for a in assets])
        lam = np.stack([Th[:, ix[f"lam_{g}"]] for g in FACTOR_GROUPS], 1)[:, grp]   # (B, A)
        phif, phix = Th[:, ix["phif"]], Th[:, ix["phi1"]]
        s2x = np.exp(2 * Th[:, ix["lsig1"]])
        d = np.ones((B, n))
        q = np.zeros((B, n))
        d[:, 0] = phif
        q[:, 0] = 1 - phif ** 2
        d[:, x_i] = phix[:, None]
        q[:, x_i] = s2x[:, None]
        a0 = np.zeros((B, n))
        a0[:, mu_i] = Th[:, ix["mubar"]][:, None]
        P0 = np.zeros((B, n, n))
        P0[:, 0, 0] = 1.0
        P0[:, mu_i, mu_i] = np.exp(2 * Th[:, ix["ltau"]])[:, None]
        P0[:, x_i, x_i] = (s2x / (1 - phix ** 2))[:, None]
        hZ = np.zeros((B, A, n))
        hZ[:, np.arange(A), 0] = lam
        hZ[:, np.arange(A), mu_i] = 1.0
        hZ[:, np.arange(A), x_i] = 1.0
        g1, g2 = noise_corr(rho)
        H = np.zeros((B, p, p))
        H[:, :A, :A] = np.exp(Th[:, ix["lH1"]])[:, None, None] * g1
        Z = hZ.copy()
        qc_vec = np.zeros(n)
        tau_vec = np.zeros(n)
        if self.daily:
            P0[:, c_i, c_i] = TAUC2
            Z2 = hZ.copy()
            Z2[:, np.arange(A), c_i] = 1.0
            Z = np.concatenate([hZ, Z2], 1)
            H[:, A:, A:] = np.exp(Th[:, ix["lH2"]])[:, None, None] * g2
            qc_vec[c_i] = QC
            tau_vec[c_i] = TAUC2
        return d, q, a0, P0, Z, H, hZ, qc_vec, tau_vec, c_i


def _factor_inputs(spec, obs, s0, s1):
    """Observation matrix, offset masks and each state's first active index
    (relative to s0)."""
    Y = obs.y1[s0:s1]
    A = len(obs.assets)
    n = 1 + 2 * A + (A if spec.daily else 0)
    dact = np.zeros((s1 - s0, n), bool)
    reset = np.zeros((s1 - s0, n), bool)
    if spec.daily:
        Y = np.concatenate([Y, obs.y2[s0:s1]], 1)
        dact[:, 1 + 2 * A:] = obs.dact[s0:s1]
        reset[:, 1 + 2 * A:] = obs.reset[s0:s1]
    has = np.isfinite(obs.y1[:s1]).any(0)
    first = np.where(has, np.argmax(np.isfinite(obs.y1[:s1]), 0), 10 ** 9) - s0
    first = np.maximum(first, 0)
    st = np.zeros(n, int)
    st[1:1 + A] = first
    st[1 + A:1 + 2 * A] = first
    if spec.daily:
        st[1 + 2 * A:] = first
    return Y, dact, reset, st


def return_corr(view_E) -> np.ndarray:
    from ..ref import nearest_corr, pairwise_corr
    return nearest_corr(pairwise_corr(view_E.excess_monthly))


def fit_factor(spec: FactorSpec, view_E) -> dict:
    t0 = time.perf_counter()
    obs = build_obs(view_E)
    A = len(obs.assets)
    rho = return_corr(view_E)
    names, start, lo, hi, free = spec.param_table()
    any_data = np.isfinite(obs.y1).any(1)
    s0 = int(np.argmax(any_data)) if any_data.any() else 0
    s1 = len(obs.months)
    Y, dact, reset, st = _factor_inputs(spec, obs, s0, s1)
    nobs_asset = np.isfinite(obs.y1).sum(0)
    th = start.copy()
    info = {"nit": 0, "nfev": 0, "success": None}
    if nobs_asset.sum() >= MIN_POOLED_OBS and (nobs_asset > 0).sum() >= 2:
        N = float(np.isfinite(Y).sum())
        k = len(start)

        def f(x):
            rows = [x] + [x + FD_STEP * np.eye(k)[j] for j in range(k)]
            Tb = np.array(rows)
            d, q, a0, P0, Z, H, hZ, qcv, tauv, _ = spec.arrays(Tb, obs.assets, rho)
            ll = kf_multi(d, q, a0, P0, Z, H, Y, qcv, tauv, dact, reset, hZ, state_start=st)[0]
            info["nfev"] += 1
            return -ll[0] / N, -(ll[1:] - ll[0]) / FD_STEP / N

        res = minimize(f, start, jac=True, method="L-BFGS-B", bounds=list(zip(lo, hi)),
                       options={"maxiter": MAXITER, "ftol": 1e-7, "gtol": 1e-5})
        th = res.x
        info.update(nit=int(res.nit), success=bool(res.success))
    d, q, a0, P0, Z, H, hZ, qcv, tauv, _ = spec.arrays(th[None, :], obs.assets, rho)
    ll, a, P, (pm, pv) = kf_multi(d, q, a0, P0, Z, H, Y, qcv, tauv, dact, reset, hZ, store=True,
                                  state_start=st)
    s = _calibrate(obs.r2[s0:s1], pm[:, 0], pv[:, 0], obs.nobs[s0:s1])
    return dict(theta=th, names=names, rho=rho, a=a, P=P, s=s, nE=s1, loglik=float(ll[0]),
                seconds=time.perf_counter() - t0, **info)


def forecast_factor(spec: FactorSpec, fit: dict, obs, nt: int) -> np.ndarray:
    th = fit["theta"][None, :]
    d, q, a0, P0, Z, H, hZ, qcv, tauv, _ = spec.arrays(th, obs.assets, fit["rho"])
    a, P = fit["a"], fit["P"]
    nE = fit["nE"]
    if nt > nE:
        Y, dact, reset, st = _factor_inputs(spec, obs, nE, nt)
        _, a, P, _ = kf_multi(d, q, a, P, Z, H, Y, qcv, tauv, dact, reset, hZ, state_start=st)
    dd = d[:, :, None] * d[:, None, :]
    n = a.shape[1]
    Q = np.zeros((1, n, n))
    Q[:, np.arange(n), np.arange(n)] = q
    tot = np.zeros(hZ.shape[1])
    for _ in range(HORIZON):
        a = a * d
        P = P * dd + Q
        m = hZ[0] @ a[0]
        v = np.einsum("an,nm,am->a", hZ[0], P[0], hZ[0])
        tot += np.exp(m + 0.5 * v)
    return fit["s"] * tot


# --------------------------------------------------------------------------- model classes
class StateSpaceVol(VolForecaster):
    """Univariate-filter models (AR(1) or two components; monthly or monthly + daily)."""

    def __init__(self, spec: Spec):
        self.spec = spec
        self.last_fit = None

    def _fit(self, view):
        E = estimation_date(view.asof_date, view.monthly.index, self.spec.refit)
        vE = view.asof(E)
        key = (self.spec.key, E, fingerprint(vE))
        return _cached(key, lambda: fit_univ(self.spec, vE))

    def forecast_vol(self, view, assets):
        fit = self._fit(view)
        self.last_fit = fit
        obs = build_obs(view)
        var = forecast_univ(self.spec, fit, obs, len(obs.months), assets)
        out = pd.Series(np.sqrt(var), index=obs.assets).reindex(assets)
        return _finite(out, view, assets)


class GridVol(VolForecaster):
    """Pooled AR(1), monthly returns only, filtered EXACTLY on a grid.

    Parameters (phi, sigma_eta) and each asset's long-run level mu (the
    Kalman posterior mean at the fit date) come from the pooled AR(1) fit;
    only the filter differs, so the comparison with ``statespace_ar1_pooled``
    isolates what the Gaussian approximation costs.
    """

    def __init__(self):
        self.spec = Spec(ncomp=1, daily=False, pooled=True)

    def _fit(self, view):
        E = estimation_date(view.asof_date, view.monthly.index, self.spec.refit)
        vE = view.asof(E)
        fp = fingerprint(vE)
        kf = _cached((self.spec.key, E, fp), lambda: fit_univ(self.spec, vE))
        return _cached(("grid", E, fp), lambda: _fit_grid(kf, vE))

    def forecast_vol(self, view, assets):
        g = self._fit(view)
        obs = build_obs(view)
        nt = len(obs.months)
        pi = g["pi"]
        if nt > g["nE"]:
            pi, _ = G.grid_filter(pi, g["K"], g["mu"], g["sx"], obs.r2[g["nE"]:nt])
        var = g["s"] * G.forecast(pi, g["K"], g["mu"], g["sx"], HORIZON)
        out = pd.Series(np.sqrt(var), index=obs.assets).reindex(assets)
        return _finite(out, view, assets)


def _fit_grid(kf, vE):
    obs = build_obs(vE)
    names = kf["names"]
    th = kf["theta"][0]
    phi = th[names.index("phi1")]
    sig = np.exp(th[names.index("lsig1")])
    sx = np.full(len(obs.assets), sig / np.sqrt(1 - phi ** 2))
    mu = kf["a"][:, 0].copy()
    K = G.transition(phi)
    any_data = np.isfinite(obs.y1).any(1)
    s0 = int(np.argmax(any_data)) if any_data.any() else 0
    pi0 = np.tile(G.prior(), (len(obs.assets), 1))
    pi, pred = G.grid_filter(pi0, K, mu, sx, obs.r2[s0:], store=True)
    r2 = obs.r2[s0:]
    nobs = obs.nobs[s0:]
    prev = np.vstack([np.zeros((1, nobs.shape[1])), nobs[:-1]])
    ok = np.isfinite(r2) & (prev >= CALIB_BURN)
    s = float(np.mean(r2[ok] / pred[ok])) if ok.sum() >= 24 else 1.0
    return dict(pi=pi, K=K, mu=mu, sx=sx, s=s, nE=len(obs.months))


class FactorVol(VolForecaster):
    """Common log-volatility factor + idiosyncratic AR(1) per asset."""

    def __init__(self, spec: FactorSpec):
        self.spec = spec
        self.last_fit = None

    def _fit(self, view):
        E = estimation_date(view.asof_date, view.monthly.index, self.spec.refit)
        vE = view.asof(E)
        key = (self.spec.key, E, fingerprint(vE))
        return _cached(key, lambda: fit_factor(self.spec, vE))

    def forecast_vol(self, view, assets):
        fit = self._fit(view)
        self.last_fit = fit
        obs = build_obs(view)
        var = forecast_factor(self.spec, fit, obs, len(obs.months))
        out = pd.Series(np.sqrt(var), index=obs.assets).reindex(assets)
        return _finite(out, view, assets)


def _finite(out: pd.Series, view, assets) -> pd.Series:
    """Last-resort fallback (never expected): long-run monthly vol."""
    bad = ~np.isfinite(out) | (out <= 0)
    if bad.any():
        x = view.excess_monthly[assets]
        fb = np.sqrt(12.0 * (x ** 2).mean())
        out = out.where(~bad, fb)
        out = out.where(np.isfinite(out) & (out > 0), 0.15)
    return out
