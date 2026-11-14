"""Put conditional correlations into a full unconditional matrix.

The conditional models produce new correlations for a few ANCHOR pairs
(e.g. us_large x ust_int). The rest of the matrix must follow coherently:
intl equity, TIPS or IG credit should inherit the stock-bond move in
proportion to how much of them is "equity" and "rates", and the matrix must
stay positive definite for any asset list.

Method (anchor-factor representation, then rescale):

    C0 = B' C_AA B + R,   B = C_AA^{-1} C_A.   (regression of every asset on the
                                                 anchors, in correlation terms)
    R  = C0 - C_A' C_AA^{-1} C_A                (Schur complement: PSD because C0 is PD)
    C1 = B' C_AA* B + R                         (anchor block replaced by its target)
    C  = diag(C1)^{-1/2} C1 diag(C1)^{-1/2}

For the anchors themselves ``B`` is the identity, so their correlations are
exactly the targets; every other pair moves by
``sum_kl b_ik (C_AA* - C_AA)_kl b_jl`` before the rescale, i.e. only through
its loadings on the anchors; the residual structure ``R`` (what makes small
caps closer to large caps than to EM, etc.) is untouched. ``C1`` is PSD
(``C_AA*`` PD, ``R`` PSD) and its diagonal is positive, so ``C`` is a valid
correlation matrix. If the anchor target is not PD with a margin, or the
result's minimum eigenvalue falls below ``floor``, the anchor change is
scaled back by bisection (``lam`` in [0, 1], 0 = the base matrix);
``lam`` is returned so the caller can record how often that binds.
"""
from __future__ import annotations

import numpy as np

from ..ref import nearest_corr

PD_FLOOR = 1e-4


def min_eig(c: np.ndarray) -> float:
    return float(np.linalg.eigvalsh((c + c.T) / 2).min())


def _rebuild(C0, ia, CAA_new):
    CAA = C0[np.ix_(ia, ia)]
    CA = C0[ia, :]
    B = np.linalg.solve(CAA, CA)                  # k x n loadings
    C1 = C0 + B.T @ (CAA_new - CAA) @ B
    d = np.sqrt(np.clip(np.diag(C1), 1e-12, None))
    C = C1 / np.outer(d, d)
    C = (C + C.T) / 2
    np.fill_diagonal(C, 1.0)
    C[np.ix_(ia, ia)] = CAA_new                    # exact (rounding only)
    return C


def embed_anchor_targets(C0: np.ndarray, assets: list[str], targets: dict[tuple[str, str], float],
                         floor: float = PD_FLOOR, iters: int = 40) -> tuple[np.ndarray, float]:
    """Return (C, lam): ``C0`` with the anchor pairs in ``targets`` set to their
    target correlations (as far as positive definiteness allows) and the rest
    of the matrix moved through the anchor-factor representation."""
    C0 = (np.asarray(C0, float) + np.asarray(C0, float).T) / 2
    if min_eig(C0) <= floor:
        C0 = nearest_corr(C0, floor=max(floor, 1e-3))
    pos = {a: i for i, a in enumerate(assets)}
    targets = {p: v for p, v in targets.items() if p[0] in pos and p[1] in pos and p[0] != p[1]}
    if not targets:
        return C0, 1.0
    anchors = sorted({a for p in targets for a in p}, key=pos.get)
    ia = [pos[a] for a in anchors]
    apos = {a: k for k, a in enumerate(anchors)}
    CAA = C0[np.ix_(ia, ia)]
    T = CAA.copy()
    for (a, b), v in targets.items():
        T[apos[a], apos[b]] = T[apos[b], apos[a]] = float(np.clip(v, -0.99, 0.99))
    floor_eff = min(floor, 0.5 * min_eig(C0))

    def build(lam):
        CAA_new = CAA + lam * (T - CAA)
        if min_eig(CAA_new) < floor_eff:
            return None
        C = _rebuild(C0, ia, CAA_new)
        return C if min_eig(C) >= floor_eff else None

    C = build(1.0)
    if C is not None:
        return C, 1.0
    lo, hi, best = 0.0, 1.0, C0
    for _ in range(iters):
        mid = (lo + hi) / 2
        Cm = build(mid)
        if Cm is not None:
            lo, best = mid, Cm
        else:
            hi = mid
    return best, lo
