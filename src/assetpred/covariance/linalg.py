"""Small matrix utilities: PSD repair, correlation <-> covariance, Fisher z, shrinkage.

Everything works on plain numpy arrays (n x n), with NaN marking pairs that
have no joint history.
"""
from __future__ import annotations

import numpy as np

EIG_FLOOR = 1e-10   # relative eigenvalue floor (times the mean eigenvalue)
Z_CLIP = 0.999      # correlations are clipped to +-Z_CLIP before Fisher z


def symmetrize(A: np.ndarray) -> np.ndarray:
    A = np.asarray(A, dtype=float)
    return 0.5 * (A + A.T)


def min_eig(A: np.ndarray) -> float:
    return float(np.linalg.eigvalsh(symmetrize(A))[0])


def make_psd(S: np.ndarray, rel_floor: float = EIG_FLOOR) -> np.ndarray:
    """Symmetrize and clip eigenvalues at ``rel_floor`` times the mean eigenvalue.

    A matrix that is already PSD with eigenvalues above the floor comes back
    unchanged (up to symmetrization)."""
    S = symmetrize(S)
    w, V = np.linalg.eigh(S)
    floor = rel_floor * max(float(np.mean(np.abs(w))), 1e-300)
    if w[0] >= floor:
        return S
    w = np.maximum(w, floor)
    return symmetrize((V * w) @ V.T)


def cov_to_corr(S: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sd = np.sqrt(np.clip(np.diag(S), 0.0, None))
    with np.errstate(invalid="ignore", divide="ignore"):
        R = S / np.outer(sd, sd)
    np.fill_diagonal(R, 1.0)
    return R, sd


def corr_to_cov(R: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return symmetrize(R * np.outer(sd, sd))


def nearest_corr(R: np.ndarray, rel_floor: float = 1e-8, iters: int = 50) -> np.ndarray:
    """Project a symmetric matrix with unit diagonal onto the correlation matrices.

    Higham (2002) alternating projections with Dykstra's correction, finished by
    an eigenvalue floor and rescaling to unit diagonal (which keeps PSD), so the
    result is always a valid, positive-definite correlation matrix. NaN entries
    (no joint history) are set to 0 first."""
    R = symmetrize(np.nan_to_num(np.asarray(R, dtype=float), nan=0.0))
    np.fill_diagonal(R, 1.0)
    if np.linalg.eigvalsh(R)[0] >= rel_floor:
        return R
    Y = R.copy()
    dS = np.zeros_like(R)
    for _ in range(iters):
        Rk = Y - dS
        w, V = np.linalg.eigh(Rk)
        X = (V * np.maximum(w, 0.0)) @ V.T
        dS = X - Rk
        Y = X.copy()
        np.fill_diagonal(Y, 1.0)
        if np.linalg.norm(Y - X) < 1e-10:
            break
    w, V = np.linalg.eigh(symmetrize(Y))
    Y = (V * np.maximum(w, rel_floor)) @ V.T
    d = np.sqrt(np.diag(Y))
    Y = Y / np.outer(d, d)
    np.fill_diagonal(Y, 1.0)
    return symmetrize(Y)


def fisher_z(R: np.ndarray) -> np.ndarray:
    return np.arctanh(np.clip(R, -Z_CLIP, Z_CLIP))


def blend_corr(R_base: np.ndarray, R_dyn: np.ndarray, R_dyn_ref: np.ndarray,
               w: float | np.ndarray, space: str = "z") -> np.ndarray:
    """Add the *anomaly* of a dynamic estimate to a long-run level, in Fisher z.

        z(R) = z(R_base) + w * (z(R_dyn) - z(R_dyn_ref))

    ``R_base`` is the long-run correlation of the target (e.g. of h-month
    sums), ``R_dyn`` a recent estimate measured on another scale (monthly or
    daily returns) and ``R_dyn_ref`` the long-run correlation on that same
    scale, so that level differences between scales (non-synchronous closes,
    stale NAVs, cross-autocorrelation at long horizons) cancel and only the
    timing information is used. Entries where either dynamic matrix is NaN keep
    the base value. The result is projected to the nearest correlation matrix.
    """
    return apply_anomaly(R_base, corr_anomaly(R_dyn, R_dyn_ref, space), w, space)


def corr_anomaly(R_dyn: np.ndarray, R_ref: np.ndarray, space: str = "z") -> np.ndarray:
    """Deviation of a dynamic correlation from its reference level, in Fisher z
    (``space='z'``) or plain correlation units (``'corr'``); NaN where missing."""
    if space == "z":
        return fisher_z(R_dyn) - fisher_z(R_ref)
    if space == "corr":
        return np.asarray(R_dyn, float) - np.asarray(R_ref, float)
    raise ValueError(space)


def apply_anomaly(R_base: np.ndarray, an: np.ndarray, w: float | np.ndarray,
                  space: str = "z") -> np.ndarray:
    """R_base + w * anomaly (NaN anomalies count as 0), projected to a correlation
    matrix. In ``'corr'`` space the result is clipped to +-Z_CLIP first. Fisher z
    stretches differences near +-1, so a small daily move in a 0.95 bond pair
    becomes a large one when added to another 0.95 base: ``'corr'`` is the
    better-behaved choice for highly correlated pairs."""
    an = np.where(np.isfinite(an), an, 0.0)
    base = np.nan_to_num(R_base, nan=0.0)
    if space == "z":
        R = np.tanh(fisher_z(base) + w * an)
    else:
        R = np.clip(base + w * an, -Z_CLIP, Z_CLIP)
    np.fill_diagonal(R, 1.0)
    return nearest_corr(R)


def shrink_corr(R: np.ndarray, delta: float, target: str = "constant") -> np.ndarray:
    """Linear shrinkage of a correlation matrix toward a constant-correlation
    (average off-diagonal) or identity target."""
    n = len(R)
    if n < 2 or delta <= 0:
        return R
    if target == "constant":
        rbar = (R.sum() - n) / (n * (n - 1))
        T = np.full_like(R, rbar)
    elif target == "identity":
        T = np.zeros_like(R)
    else:
        raise ValueError(target)
    np.fill_diagonal(T, 1.0)
    out = (1 - delta) * R + delta * T
    np.fill_diagonal(out, 1.0)
    return symmetrize(out)


def eig_floor_corr(R: np.ndarray, floor: float) -> np.ndarray:
    """Raise the eigenvalues of a correlation matrix to at least ``floor``, then
    rescale to unit diagonal. Touches only the near-singular directions."""
    w, V = np.linalg.eigh(symmetrize(R))
    if w[0] >= floor:
        return R
    A = (V * np.maximum(w, floor)) @ V.T
    d = np.sqrt(np.diag(A))
    A = A / np.outer(d, d)
    np.fill_diagonal(A, 1.0)
    return symmetrize(A)


def lw_constant_corr_intensity(X: np.ndarray) -> float:
    """Ledoit-Wolf (2004, "Honey, I shrunk the sample covariance matrix")
    optimal shrinkage intensity toward the constant-correlation target, for a
    balanced T x n panel. Returns delta in [0, 1]."""
    X = np.asarray(X, dtype=float)
    T, n = X.shape
    if T < 3 or n < 2:
        return 0.0
    Xc = X - X.mean(axis=0)
    S = Xc.T @ Xc / T
    sd = np.sqrt(np.diag(S))
    R = S / np.outer(sd, sd)
    rbar = (R.sum() - n) / (n * (n - 1))
    F = rbar * np.outer(sd, sd)
    np.fill_diagonal(F, np.diag(S))
    Y2 = Xc ** 2
    pi_mat = Y2.T @ Y2 / T - S ** 2
    pi_hat = pi_mat.sum()
    term1 = ((Xc ** 3).T @ Xc) / T
    term2 = np.diag(S)[:, None] * S
    theta_ii = term1 - term2            # theta_ii,ij
    theta_jj = theta_ii.T               # theta_jj,ij
    with np.errstate(invalid="ignore", divide="ignore"):
        rho_off = (rbar / 2) * (np.outer(1 / sd, sd) * theta_ii + np.outer(sd, 1 / sd) * theta_jj)
    np.fill_diagonal(rho_off, 0.0)
    rho_hat = np.trace(pi_mat) + np.nansum(rho_off)
    gamma_hat = np.linalg.norm(S - F, "fro") ** 2
    if not gamma_hat > 0:
        return 1.0
    kappa = (pi_hat - rho_hat) / gamma_hat
    return float(np.clip(kappa / T, 0.0, 1.0))


def mean_uncertainty_factor(counts: np.ndarray, horizon: int) -> np.ndarray:
    """Multiplicative factor for the estimation error of an expanding mean.

    The joint forecasts are centred on ``h * mean(xr)`` per asset (the
    ``hist_mean`` benchmark's mean). With T_i months for asset i and T_ij
    months in common, Cov(h mu_i, h mu_j) = h^2 sigma_ij T_ij / (T_i T_j)
    ~= Sigma_h,ij * h T_ij / (T_i T_j), so the predictive covariance is
    Sigma_h * (1 + h M), M_ij = T_ij / (T_i T_j). M is a Gram matrix, so the
    Schur product keeps PSD; the diagonal factor is (1 + h / T_i), exactly the
    ``sqrt(1 + h/T)`` inflation of the marginal baselines. ``counts`` is the
    n x n matrix of pairwise joint month counts."""
    c = np.asarray(counts, dtype=float)
    T = np.diag(c)
    return 1.0 + horizon * c / np.outer(T, T)
