"""The static ``sample`` covariance: the comparator of the strategy test's risk question.

Expanding, pairwise-complete sample covariance of realized h-month sums of
monthly excess log returns, at each origin built from rows dated <= origin
only. This is assetpred's ``sample`` covariance model (``covariance.models.
SampleCov`` run through ``covariance.engine.walk_forward_cov``); the pieces it
needs are ported here verbatim so that the rest of ``assetpred.covariance``
could be deleted:

* ``h_sample``                 <- ``covariance.estimators.h_sample``
* ``nearest_corr``             <- ``covariance.linalg.nearest_corr``
* ``mean_uncertainty_factor``  <- ``covariance.linalg.mean_uncertainty_factor``
* ``symmetrize``, ``make_psd``, ``min_eig``, ``corr_to_cov``, ``check_matrix``
  <- ``covariance.linalg`` / ``covariance.engine`` (the symmetrize and
  eigenvalue-floor step every matrix went through)
* ``sample_cov_history``       <- ``engine.walk_forward_cov`` restricted to the
  one static model (no fit, no refit schedule, no daily data or features).

At each origin t, asset a is eligible if it has at least ``min_history``
months of its own excess returns through t; the matrix is
``corr_to_cov(nearest_corr(R_h), sd_h)`` from the realized h-month sums
(rows s <= t - h), times the mean-uncertainty factor (the forecasts are
centred on the expanding mean), symmetrized and eigenvalue-floored.

It is used only by the strategy test; the package's covariance forecast is
``covhistory`` (``af_primary``).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import schema

EIG_FLOOR = 1e-10   # relative eigenvalue floor (times the mean eigenvalue)
PSD_TOL = 1e-12
MIN_HISTORY = 60    # assetpred config.MIN_ASSET_HISTORY
NAME = "sample"


class CovContractError(ValueError):
    pass


# --------------------------------------------------------------------------
# linalg (verbatim from assetpred.covariance.linalg)
# --------------------------------------------------------------------------
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


# --------------------------------------------------------------------------
# estimator (verbatim from assetpred.covariance.estimators.h_sample, with the
# CovTrainingSet's ``targets()`` inlined: realized h-month sums of the
# eligible assets' returns dated <= origin)
# --------------------------------------------------------------------------
def h_sample(X: pd.DataFrame, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    """Pairwise-complete sample (variance, correlation) of realized h-month sums.

    ``X``: monthly excess log returns of the eligible assets, rows <= origin."""
    Y = schema.make_targets(X, horizon)
    var = Y.var(ddof=1).to_numpy(float)
    R = Y.corr(min_periods=24).to_numpy(float)
    return var, nearest_corr(R)


def check_matrix(S: np.ndarray, where: str = "") -> None:
    S = np.asarray(S, dtype=float)
    if S.ndim != 2 or S.shape[0] != S.shape[1]:
        raise CovContractError(f"{where}: not a square matrix {S.shape}")
    if not np.isfinite(S).all():
        raise CovContractError(f"{where}: non-finite entries")
    if not np.array_equal(S, S.T):
        raise CovContractError(f"{where}: not symmetric")
    scale = max(float(np.trace(S)) / len(S), 1e-300)
    if min_eig(S) < -PSD_TOL * scale:
        raise CovContractError(f"{where}: not PSD (min eig {min_eig(S):.3g})")


# --------------------------------------------------------------------------
# walk-forward (engine.walk_forward_cov for the one static model)
# --------------------------------------------------------------------------
def sample_cov_at(xr: pd.DataFrame, origin, horizon: int = 12, min_history: int = MIN_HISTORY,
                  mean_adjust: bool = True) -> tuple[tuple, np.ndarray] | None:
    """(eligible assets, covariance) at one origin from ``xr`` rows <= origin;
    None if no asset is eligible."""
    t = pd.Timestamp(origin)
    past = xr.loc[:t]
    n_obs = past.notna().sum()
    elig = tuple(a for a in xr.columns if n_obs[a] >= min_history)
    if not elig:
        return None
    X = past[list(elig)]
    var, R = h_sample(X, horizon)
    S = corr_to_cov(R, np.sqrt(var))
    where = f"{NAME} h={horizon} origin={t.date()}"
    if S.shape != (len(elig), len(elig)) or not np.isfinite(S).all():
        raise CovContractError(f"{where}: bad prediction shape {S.shape} or non-finite values")
    if mean_adjust:
        m = X.notna().to_numpy(float)
        S = S * mean_uncertainty_factor(m.T @ m, horizon)
    S = make_psd(S)
    check_matrix(S, where)
    return elig, S


def sample_cov_history(xr: pd.DataFrame, horizon: int = 12, origins=None, *,
                       min_history: int = MIN_HISTORY, mean_adjust: bool = True,
                       name: str = NAME) -> pd.DataFrame:
    """Static expanding sample covariance at each origin, long in ``schema.COV_COLUMNS``.

    ``xr``: wide monthly excess log returns (month-end index, asset columns;
    the column order is the asset order of every matrix). ``origins``: the
    month-ends to forecast from (default: every month of ``xr``); origins at
    which no asset has ``min_history`` months are skipped. Each block is the
    full n x n matrix of the eligible assets (both triangles)."""
    schema._check_month_end(xr.index, "excess returns")
    h = int(horizon)
    origins = xr.index if origins is None else pd.DatetimeIndex(origins)
    blocks = []
    for t in origins:
        out = sample_cov_at(xr, t, h, min_history, mean_adjust)
        if out is None:
            continue
        elig, S = out
        n = len(elig)
        blocks.append(pd.DataFrame({
            "date": pd.Timestamp(t), "asset_i": np.repeat(elig, n), "asset_j": np.tile(elig, n),
            "cov": S.ravel()}))
    if not blocks:
        return pd.DataFrame(columns=schema.COV_COLUMNS)
    df = pd.concat(blocks, ignore_index=True)
    df.insert(0, "horizon", h)
    df.insert(0, "model", name)
    df["date"] = pd.to_datetime(df["date"])
    return df[schema.COV_COLUMNS]
