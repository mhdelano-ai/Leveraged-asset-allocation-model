"""Corrected DCC (Aielli 2013) on monthly returns, with a slowly moving
correlation target and composite (all-pairs) likelihood.

Model
-----
Monthly log excess returns ``r_t`` are standardized by a slow, predictable
volatility (RiskMetrics monthly EWMA, lambda = 0.97):
``h_t = lambda h_{t-1} + (1 - lambda) r_{t-1}^2``, ``e_t = r_t / sqrt(h_t)``.
The cDCC recursion on the standardized returns is

    Q_{t+1} = (1 - a - b) S_{t+1} + a (Q*_t^{1/2} e_t)(Q*_t^{1/2} e_t)' + b Q_t,
    R_t     = Q*_t^{-1/2} Q_t Q*_t^{-1/2},      Q*_t = diag(Q_t)

with correlation targeting: ``S`` is the ``mcorr_shrink_block`` estimate on
the standardized returns (exponentially weighted, half-life 10 years,
monotone-missing ML, shrunk toward block equicorrelation), recomputed each
December from data through that month and held for the next year. So the
long-run level moves slowly (as in DCC-MIDAS, Colacito-Engle-Ghysels 2011)
and only two parameters (a, b) describe the monthly dynamics around it.

Estimation: composite likelihood summed over ALL pairs (Pakel, Shephard,
Sheppard and Engle 2021), each pair on its own sample: a pair enters from the
January after both assets have 36 months of history. Unequal histories
therefore never force truncation to the shortest asset. Parameters are
re-estimated once a year, on all data through the latest December (equal
weights), from a fixed starting point (deterministic: a cold call gives the
same answer). With fewer than 120 pair-months the model has no dynamics.

Forecast: the average expected correlation over the next 12 months. For the
cDCC, ``E_t[Q_{t+h}] = S + (a+b)^{h-1} (Q_{t+1} - S)``, so the 12-month
forecast is ``S_t + kappa (R_{t+1} - S_used)`` with
``kappa = mean_{h=1..12} (a+b)^{h-1}``: the long-run target (computed with
data through the origin) plus the share ``kappa`` of today's deviation that
survives on average over the window.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from .core import (EW_HALFLIFE, block_average, ensure_pd, ew_weights, groups_of, pair_neff,
                   shrink_to_target, stambaugh_corr)

RM_LAMBDA = 0.97          # RiskMetrics (1996) decay for monthly data
ACTIVE_MONTHS = 36        # a pair enters once both assets have 36 months (= eligibility)
MIN_PAIR_MONTHS = 120     # below this many pair-months, no dynamics
HORIZON = 12


def standardize(X: np.ndarray, lam: float = RM_LAMBDA) -> np.ndarray:
    """e_t = r_t / sqrt(h_t), h predictable EWMA; h initialized at the mean
    square of the asset's first 12 months."""
    T, N = X.shape
    E = np.full_like(X, np.nan)
    for j in range(N):
        x = X[:, j]
        idx = np.flatnonzero(np.isfinite(x))
        if idx.size == 0:
            continue
        h = float(np.mean(x[idx[:12]] ** 2))
        h = max(h, 1e-12)
        for t in range(idx[0], idx[-1] + 1):
            if np.isfinite(x[t]):
                E[t, j] = x[t] / np.sqrt(h)
                h = lam * h + (1 - lam) * x[t] ** 2
            # a gap leaves h unchanged
    return E


def target_corr(E: np.ndarray, assets: list[str], halflife=EW_HALFLIFE) -> np.ndarray:
    """``mcorr_shrink_block`` on the given (standardized) returns."""
    w = ew_weights(len(E), halflife)
    R = stambaugh_corr(E, w)
    Tg = block_average(R, groups_of(assets))
    out, _ = shrink_to_target(R, Tg, pair_neff(E, w))
    return out


class _Prepared:
    """Arrays for the recursion, months from the first active pair to the end."""

    def __init__(self, E: np.ndarray, years: np.ndarray, assets: list[str], cache: dict):
        T, N = E.shape
        obs = np.isfinite(E)
        cnt = np.cumsum(obs, axis=0)
        # asset active in year y if it had >= ACTIVE_MONTHS observations by the end of y-1
        uy = np.unique(years)
        last_idx = {y: np.flatnonzero(years == y)[-1] for y in uy}
        act_year = {}
        for y in uy:
            prev = [last_idx[z] for z in uy if z < y]
            act_year[y] = (cnt[prev[-1]] >= ACTIVE_MONTHS) if prev else np.zeros(N, bool)
        act = np.vstack([act_year[y] for y in years])
        iu = np.triu_indices(N, 1)
        pact = act[:, iu[0]] & act[:, iu[1]]
        start = np.flatnonzero(pact.any(axis=1))
        self.ok = start.size > 0
        self.N, self.iu = N, iu
        if not self.ok:
            return
        t0 = start[0]
        # yearly targets: for year y, computed on data through the end of y-1 over active assets
        S = np.zeros((T, len(iu[0])))
        for y in uy:
            rows = np.flatnonzero(years == y)
            if rows[-1] < t0:
                continue
            a = np.flatnonzero(act_year[y])
            full = np.eye(N)
            if a.size >= 2:
                cut = last_idx[max(z for z in uy if z < y)]
                key = (int(y), tuple(assets[k] for k in a), cut)
                if key not in cache:
                    sub = E[: cut + 1][:, a]
                    keep = np.isfinite(sub).any(axis=1)
                    first = np.argmax(keep)
                    cache[key] = target_corr(sub[first:], [assets[k] for k in a])
                full[np.ix_(a, a)] = cache[key]
            S[rows] = full[iu]
        sl = slice(t0, T)
        self.E = np.where(obs, E, 0.0)[sl]
        self.obs = obs[sl]
        self.act = act[sl]
        self.pact = pact[sl]
        self.S = S[sl]
        self.both = self.obs[:, iu[0]] & self.obs[:, iu[1]] & self.pact
        self.n_pair_months = int(self.both.sum())

    def run(self, a: float, b: float, loglik: bool = True):
        """Filter; returns (sum of pairwise corr log-likelihoods, Q diag, Q off, S used) at T+1."""
        i, j = self.iu
        N = self.N
        qd = np.ones(N)
        qo = self.S[0].copy()
        ll = 0.0
        c = 1.0 - a - b
        for t in range(len(self.E)):
            act, pact = self.act[t], self.pact[t]
            qd = np.where(act, qd, 1.0)
            qo = np.where(pact, qo, self.S[t])
            e = self.E[t]
            if loglik:
                m = self.both[t]
                if m.any():
                    rho = qo[m] / np.sqrt(qd[i[m]] * qd[j[m]])
                    rho = np.clip(rho, -0.999, 0.999)
                    ei, ej = e[i[m]], e[j[m]]
                    om = 1.0 - rho * rho
                    ll += -0.5 * np.sum(np.log(om) + (ei * ei + ej * ej - 2 * rho * ei * ej) / om
                                        - (ei * ei + ej * ej))
            x = np.sqrt(qd) * e
            o = self.obs[t]
            s_next = self.S[t + 1] if t + 1 < len(self.S) else self.S[t]
            qd = np.where(o, c + a * x * x + b * qd, c + (a + b) * qd)
            bo = o[i] & o[j]
            qo = np.where(bo, c * s_next + a * x[i] * x[j] + b * qo, c * s_next + (a + b) * qo)
        return ll, qd, qo, (self.S[-1] if len(self.S) else None)


def fit_dcc(prep: _Prepared) -> tuple[float, float]:
    if not prep.ok or prep.n_pair_months < MIN_PAIR_MONTHS:
        return 0.0, 0.0

    def unpack(u):
        p = 0.995 / (1.0 + np.exp(-u[0]))
        a = p / (1.0 + np.exp(-u[1]))
        return a, p - a

    def nll(u):
        a, b = unpack(u)
        return -prep.run(a, b)[0]

    u0 = np.array([np.log(0.97 / 0.025), np.log(0.03 / 0.94)])   # a = 0.03, b = 0.94
    res = minimize(nll, u0, method="Nelder-Mead",
                   options={"xatol": 1e-3, "fatol": 1e-3, "maxiter": 200})
    a, b = unpack(res.x)
    # no gain over the static model (a = b = 0, R = S): keep it static
    if not np.isfinite(res.fun) or -res.fun <= prep.run(0.0, 0.0)[0]:
        return 0.0, 0.0
    return float(a), float(b)
