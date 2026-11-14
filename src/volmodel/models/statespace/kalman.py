"""Kalman filters for the state-space vol models (pure numpy).

``kf_univ`` runs many independent small filters at once (one per asset and,
during estimation, per perturbed parameter vector: the finite-difference
gradient costs one batched pass). The state of one filter is

    [mu, x_1, ..., x_K, (c)]      h = mu + sum_k x_k

``mu`` is the asset's long-run log variance (constant; its prior is the
hierarchical one), ``x_k`` are zero-mean AR(1) components and ``c`` the
offset of the daily-RV observation. Observations are processed one at a
time (univariate treatment, Durbin & Koopman 2012, 6.4), so a missing value
is a skipped update.

``kf_multi`` is the joint filter of the common-factor model: one state vector
for all assets, observation noise correlated across assets.

Both return the Gaussian (quasi-)log-likelihood and, optionally, the
one-step-ahead mean and variance of each asset's ``h``.
"""
from __future__ import annotations

import numpy as np

LOG2PI = np.log(2.0 * np.pi)


def kf_univ(d, q, a0, P0, H1, H2, Y1, Y2, dact, reset, qc, tauc2, Z1, Z2, ic,
            store=False):
    """Batched univariate-treatment Kalman filter.

    d, q    (B, n) diagonal transition and state-noise variance (c excluded)
    a0, P0  (B, n), (B, n, n) state at the start (before the first month)
    H1, H2  (B,) measurement-noise variances of y1 and y2
    Y1, Y2  (T, B) observations (NaN = missing); Y2 may be None
    dact, reset (T, B) bool: offset random walk active / splice this month
    Z1, Z2  (n,) 0/1 loadings of y1 and y2 on the state; ic = index of c (or None)
    Returns (loglik (B,), a, P, pred) where pred = (m, v) (T, B) one-step
    predictions of h = Z1 . state if ``store``.

    A missing observation is processed with an effectively infinite noise
    variance (gain ~1e-30), which keeps the loop free of masks.
    """
    a = a0.copy()
    P = P0.copy()
    T, B = Y1.shape
    n = a.shape[1]
    dd = d[:, :, None] * d[:, None, :]
    Q = np.zeros((B, n, n))
    Q[:, np.arange(n), np.arange(n)] = q
    pm = np.full((T, B), np.nan) if store else None
    pv = np.full((T, B), np.nan) if store else None
    big = 1e30
    m1 = np.isfinite(Y1)
    y1 = np.where(m1, Y1, 0.0)
    h1 = np.where(m1, H1[None, :], big)
    s1 = np.flatnonzero(Z1)
    F1 = np.ones((T, B))
    V1 = np.zeros((T, B))
    use2 = Y2 is not None
    if use2:
        m2 = np.isfinite(Y2)
        y2 = np.where(m2, Y2, 0.0)
        h2 = np.where(m2, H2[None, :], big)
        s2 = np.flatnonzero(Z2)
        F2 = np.ones((T, B))
        V2 = np.zeros((T, B))
        rows2 = m2.any(1)
    if ic is not None:
        cq = qc * dact + tauc2 * reset
    lo1, hi1 = s1[0], s1[-1] + 1        # contiguous index blocks
    for t in range(T):
        a *= d
        P *= dd
        P += Q
        if ic is not None:
            P[:, ic, ic] += cq[t]
        PZ = P[:, :, lo1:hi1].sum(2)
        f = PZ[:, lo1:hi1].sum(1)
        m = a[:, lo1:hi1].sum(1)
        if store:
            pm[t] = m
            pv[t] = f
        f += h1[t]
        v = y1[t] - m
        F1[t] = f
        V1[t] = v
        K = PZ / f[:, None]
        a += K * v[:, None]
        P -= K[:, :, None] * PZ[:, None, :]
        if use2 and rows2[t]:
            PZ = P[:, :, s2].sum(2)
            f = PZ[:, s2].sum(1) + h2[t]
            v = y2[t] - a[:, s2].sum(1)
            F2[t] = f
            V2[t] = v
            K = PZ / f[:, None]
            a += K * v[:, None]
            P -= K[:, :, None] * PZ[:, None, :]
    F1 = np.where(m1, F1, 1.0)
    ll = -0.5 * (m1 * (LOG2PI + np.log(F1) + V1 * V1 / F1)).sum(0)
    if use2:
        F2 = np.where(m2, F2, 1.0)
        ll -= 0.5 * (m2 * (LOG2PI + np.log(F2) + V2 * V2 / F2)).sum(0)
    P = 0.5 * (P + np.swapaxes(P, 1, 2))
    return ll, a, P, (pm, pv)


def kf_multi(d, q, a0, P0, Z, H, Y, qc_vec, tau_vec, dact, reset, hZ, store=False,
             state_start=None):
    """Batched multivariate Kalman filter (common-factor model).

    d, q   (B, n)        diagonal transition / state noise (offsets excluded)
    a0, P0 (B, n), (B, n, n)
    Z      (B, p, n)     loadings of the p observations
    H      (B, p, p)     observation-noise covariance
    Y      (T, p)        observations (NaN = missing), shared by all B
    qc_vec (n,)          random-walk variance of each state while active (offsets)
    tau_vec (n,)         variance added at a splice (offsets)
    dact, reset (T, n)   bool masks on the state index
    hZ     (B, A, n)     loadings of each asset's h (for the stored predictions)
    state_start (n,)     first time index at which a state can matter (its
                         asset's first observation). Before it, a state is
                         independent of everything and stays at its
                         (stationary or constant) prior, so it is left out of
                         the recursion - the same result at a fraction of the
                         cost while few assets exist. Requires P0 to be block
                         diagonal between not-yet-started states and the rest.
    """
    T, p = Y.shape
    B, n = a0.shape
    if state_start is None:
        state_start = np.zeros(n, int)
    a_full = a0.copy()
    P_full = P0.copy()
    ll = np.zeros(B)
    A = hZ.shape[1]
    pm = np.full((T, B, A), np.nan) if store else None
    pv = np.full((T, B, A), np.nan) if store else None
    mask = np.isfinite(Y)
    obs_ok = (np.abs(Z).sum(2) > 0)          # (B, p, n) -> which states each obs loads on
    obs_states = [np.flatnonzero(obs_ok[0, j]) for j in range(p)]
    cache = {}
    act = None
    for t in range(T):
        new_act = np.flatnonzero(state_start <= t)
        if act is None or new_act.size != act.size:
            if act is not None:
                a_full[:, act] = a
                P_full[np.ix_(np.arange(B), act, act)] = P
            act = new_act
            a = a_full[:, act].copy()
            P = P_full[:, act][:, :, act].copy()
            dA, qA = d[:, act], q[:, act]
            ddA = dA[:, :, None] * dA[:, None, :]
            QA = np.zeros_like(P)
            QA[:, np.arange(act.size), np.arange(act.size)] = qA
            qcA, tauA = qc_vec[act], tau_vec[act]
            hZA = hZ[:, :, act]
            cache = {}
        a *= dA
        P *= ddA
        P += QA
        extra = qcA * dact[t, act] + tauA * reset[t, act]
        if extra.any():
            P[:, np.arange(act.size), np.arange(act.size)] += extra
        if store:
            pm[t] = np.einsum("ban,bn->ba", hZA, a)
            pv[t] = np.einsum("ban,bnm,bam->ba", hZA, P, hZA)
        mk = mask[t]
        if not mk.any():
            continue
        key = mk.tobytes()
        if key not in cache:
            idx = np.flatnonzero(mk)
            # an observation can only be used once all states it loads on are active
            idx = np.array([j for j in idx if np.isin(obs_states[j], act).all()], int)
            cache[key] = (idx, Z[:, idx][:, :, act], H[:, idx][:, :, idx])
        idx, Zt, Ht = cache[key]
        if idx.size == 0:
            continue
        v = Y[t, idx][None, :] - np.einsum("bpn,bn->bp", Zt, a)
        ZP = Zt @ P                                       # (B, m, n)
        F = ZP @ np.swapaxes(Zt, 1, 2) + Ht              # (B, m, m)
        L = np.linalg.cholesky(F)
        sol = np.linalg.solve(L, np.concatenate([ZP, v[:, :, None]], 2))
        LZP, Lv = sol[:, :, :-1], sol[:, :, -1]           # L^-1 Z P, L^-1 v
        a += np.einsum("bmn,bm->bn", LZP, Lv)
        P -= np.swapaxes(LZP, 1, 2) @ LZP
        logdet = 2.0 * np.log(np.diagonal(L, axis1=1, axis2=2)).sum(1)
        ll -= 0.5 * (idx.size * LOG2PI + logdet + (Lv ** 2).sum(1))
    if act is not None:
        a_full[:, act] = a
        P_full[np.ix_(np.arange(B), act, act)] = P
    P_full = 0.5 * (P_full + np.swapaxes(P_full, 1, 2))
    return ll, a_full, P_full, (pm, pv)
