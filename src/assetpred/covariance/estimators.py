"""Estimators used by the covariance models. All take a :class:`CovTrainingSet`
(data through the origin only) and return arrays in the order of
``train.assets``. Results are memoized in ``train._cache`` so that a model that
needs the same piece twice computes it once.

Scales. "monthly" quantities describe one-month returns; "h" quantities the
h-month sums that are forecast. The long-run h-month level always comes from
realized overlapping h-month sums (rows s <= t - h), so serial correlation and
cross-serial correlation (momentum, mean reversion, stale prices) are in it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.signal import lfilter

from . import linalg as la
from .base import CovTrainingSet

# Daily returns whose source matches the monthly series closely enough to use
# for timing (REITs daily = VNQ vs Nareit monthly; commodities daily = DJP/BCOM
# vs AQR equal-weight monthly). Fixed a priori, not tuned.
DAILY_MATCHING = ("us_eq", "dev_eq", "em_eq", "ust_int", "ust_long", "tips",
                  "ig_credit", "hy_credit", "gold")
DAYS_PER_MONTH = 21.0


def _memo(train: CovTrainingSet, key, fn):
    c = train._cache
    if key not in c:
        c[key] = fn()
    return c[key]


# --------------------------------------------------------------------------
# Long-run (expanding) levels
# --------------------------------------------------------------------------
def h_sample(train: CovTrainingSet) -> tuple[np.ndarray, np.ndarray]:
    """Pairwise-complete sample (variance, correlation) of realized h-month sums."""
    def f():
        Y = train.targets()
        var = Y.var(ddof=1).to_numpy(float)
        R = Y.corr(min_periods=24).to_numpy(float)
        return var, la.nearest_corr(R)
    return _memo(train, "h_sample", f)


def m_sample(train: CovTrainingSet) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Monthly (mean, variance, correlation), pairwise complete."""
    def f():
        X = train.X
        R = X.corr(min_periods=24).to_numpy(float)
        return X.mean().to_numpy(float), X.var(ddof=1).to_numpy(float), la.nearest_corr(R)
    return _memo(train, "m_sample", f)


def variance_ratio(train: CovTrainingSet, lo: float = 0.25, hi: float = 4.0) -> np.ndarray:
    """var(h-month sum) / (h var(month)) per asset, as in the ``vol_ewma`` baseline."""
    h = train.horizon
    v_h, _ = h_sample(train)
    _, v_1, _ = m_sample(train)
    vr = v_h / (h * v_1)
    return np.where(np.isfinite(vr), np.clip(vr, lo, hi), 1.0)


def term_weight(phi: float, h: int) -> float:
    """sum_{k=1..h} phi^k: weight on the current deviation in an h-month sum
    when a quantity mean-reverts at rate phi per month."""
    k = np.arange(1, h + 1)
    return float((phi ** k).sum())


# --------------------------------------------------------------------------
# Monthly dynamic estimates
# --------------------------------------------------------------------------
def ewma_var(train: CovTrainingSet, lam: float) -> np.ndarray:
    """EWMA of squared demeaned monthly returns (identical to ``vol_ewma``)."""
    def f():
        X = train.X
        dev2 = (X - X.mean()) ** 2
        return dev2.ewm(alpha=1 - lam, adjust=False, ignore_na=True).mean().iloc[-1].to_numpy(float)
    return _memo(train, ("ewma_var", lam), f)


def ewma_cov(train: CovTrainingSet, lam: float) -> np.ndarray:
    """Pairwise EWMA covariance of demeaned monthly returns at the origin.

    Weight lam^(T-1-s) on month s, normalized over the months where both
    assets are observed (pandas ``adjust=True`` convention)."""
    def f():
        X = train.X
        E = (X - X.mean()).to_numpy(float)
        M = np.isfinite(E).astype(float)
        E0 = np.nan_to_num(E)
        T = len(E)
        w = lam ** np.arange(T - 1, -1, -1, dtype=float)
        num = np.einsum("s,si,sj->ij", w, E0, E0)
        den = np.einsum("s,si,sj->ij", w, M, M)
        with np.errstate(invalid="ignore", divide="ignore"):
            return la.symmetrize(np.where(den > 0, num / den, np.nan))
    return _memo(train, ("ewma_cov", lam), f)


def ewma_corr(train: CovTrainingSet, lam: float) -> np.ndarray:
    R, _ = la.cov_to_corr(ewma_cov(train, lam))
    return la.nearest_corr(R)


def rolling_corr(train: CovTrainingSet, window: int) -> np.ndarray:
    X = train.X.iloc[-window:]
    return la.nearest_corr(X.corr(min_periods=max(12, window // 3)).to_numpy(float))


# --------------------------------------------------------------------------
# Daily
# --------------------------------------------------------------------------
def _daily_block(train: CovTrainingSet, assets=DAILY_MATCHING):
    """Zero-filled daily returns of the eligible assets that have a matching
    daily source, with a 'started' mask (on or after the first valid day)."""
    def f():
        D = train.daily
        if D is None or D.empty:
            return None
        cols = [a for a in train.assets if a in D.columns and a in assets]
        if not cols:
            return None
        Dv = D[cols]
        started = Dv.notna().cummax().to_numpy()
        X = np.log1p(Dv.to_numpy(float))
        X = np.where(started & np.isfinite(X), X, 0.0)
        idx = [train.assets.index(a) for a in cols]
        return X, started.astype(float), idx
    return _memo(train, ("daily_block", tuple(assets)), f)


def _leadlag_cov(X: np.ndarray, M: np.ndarray):
    """Lead-lag (lag-1 Newey-West with weight 1) daily covariance, pairwise
    over rows where both assets have started; plus pairwise variances
    restricted to the same rows. Corrects for non-synchronous closes."""
    lag = X[1:].T @ X[:-1]
    C = X.T @ X + lag + lag.T
    a = X ** 2
    a[1:] += 2 * X[1:] * X[:-1]
    V = a.T @ M                          # V[i, j]: var_i over rows where j started
    N = M.T @ M
    with np.errstate(invalid="ignore", divide="ignore"):
        C = np.where(N > 0, C / N, np.nan)
        V = np.where(N > 0, V / N, np.nan)
    return C, V, N


def daily_corr(train: CovTrainingSet, window: int, assets=DAILY_MATCHING, min_lr_days: int = 500):
    """(recent, long-run) daily lead-lag correlations, n x n with NaN where an
    asset has no usable daily history; both on the daily scale."""
    def f():
        n = len(train.assets)
        Rr = np.full((n, n), np.nan)
        Rl = np.full((n, n), np.nan)
        blk = _daily_block(train, assets)
        if blk is None:
            return Rr, Rl
        X, M, idx = blk
        Cl, Vl, Nl = _leadlag_cov(X, M)
        Xr, Mr = X[-window:], M[-window:]
        ok_r = Mr.mean(axis=0) >= 0.9
        Cr, Vr, _ = _leadlag_cov(Xr, Mr)
        with np.errstate(invalid="ignore", divide="ignore"):
            rl = Cl / np.sqrt(np.clip(Vl, 1e-300, None) * np.clip(Vl.T, 1e-300, None))
            rr = Cr / np.sqrt(np.clip(Vr, 1e-300, None) * np.clip(Vr.T, 1e-300, None))
        good = (Nl >= min_lr_days) & np.outer(ok_r, ok_r)
        rl = np.where(good, np.clip(rl, -1, 1), np.nan)
        rr = np.where(good, np.clip(rr, -1, 1), np.nan)
        ix = np.ix_(idx, idx)
        Rr[ix], Rl[ix] = rr, rl
        np.fill_diagonal(Rr, 1.0)
        np.fill_diagonal(Rl, 1.0)
        return Rr, Rl
    return _memo(train, ("daily_corr", window, tuple(assets)), f)


def daily_var_now(train: CovTrainingSet, windows=(21, 63), assets=DAILY_MATCHING,
                  min_lr_days: int = 500) -> np.ndarray:
    """Current monthly variance from daily lead-lag realized variance, on the
    monthly level: var_monthly(months since daily start) * RV_recent / RV_longrun.
    NaN for assets without a usable daily series."""
    def f():
        n = len(train.assets)
        out = np.full(n, np.nan)
        blk = _daily_block(train, assets)
        if blk is None:
            return out
        X, M, idx = blk
        a = X ** 2
        a[1:] += 2 * X[1:] * X[:-1]
        D = train.daily
        first = [D[train.assets[i]].first_valid_index() for i in idx]
        for k, i in enumerate(idx):
            m = M[:, k] > 0
            if m.sum() < min_lr_days or M[-max(windows):, k].mean() < 0.9:
                continue
            rv_lr = a[m, k].mean()
            rv_now = np.mean([max(a[-w:, k].mean(), 0.25 * (X[-w:, k] ** 2).mean()) for w in windows])
            xm = train.X.iloc[:, i]
            xm = xm[xm.index >= first[k] + pd.offsets.MonthEnd(0)]
            if len(xm.dropna()) < 24 or not rv_lr > 0:
                continue
            out[i] = xm.var(ddof=1) * rv_now / rv_lr
        return out
    return _memo(train, ("daily_var_now", tuple(windows), tuple(assets)), f)


# --------------------------------------------------------------------------
# DCC on monthly returns, with missing data
# --------------------------------------------------------------------------
def dcc_residuals(train: CovTrainingSet, lam: float = 0.94, burn: int = 12) -> np.ndarray:
    """Standardized monthly residuals z_s = (x_s - mu) / sigma_{s|s-1}, with
    sigma from the EWMA of squared deviations known at s-1; NaN for the first
    ``burn`` months of each asset."""
    def f():
        X = train.X
        dev = X - X.mean()
        var = (dev ** 2).ewm(alpha=1 - lam, adjust=False, ignore_na=True).mean().shift(1)
        z = dev / np.sqrt(var)
        n_obs = X.notna().cumsum()
        z = z.where(n_obs > burn)
        return z.to_numpy(float)
    return _memo(train, ("dcc_z", lam, burn), f)


def _dcc_path(Z: np.ndarray, Qbar: np.ndarray, a: float, b: float) -> np.ndarray:
    """Deviation path D_s = Q_s - Qbar for s = 0..T (T+1 rows; row s uses data
    through s-1). Missing products are 'no news' (equal to Qbar)."""
    T, n = Z.shape
    P = np.einsum("si,sj->sij", np.nan_to_num(Z), np.nan_to_num(Z))
    M = np.isfinite(Z)
    MM = M[:, :, None] & M[:, None, :]
    P = np.where(MM, P, Qbar[None]) - Qbar[None]
    P = np.concatenate([P, np.zeros((1, n, n))], axis=0)
    return lfilter([0.0, a], [1.0, -b], P, axis=0)


def dcc_qbar(Z: np.ndarray) -> np.ndarray:
    M = np.isfinite(Z).astype(float)
    Z0 = np.nan_to_num(Z)
    N = M.T @ M
    S = Z0.T @ Z0
    V = (Z0 ** 2).T @ M
    with np.errstate(invalid="ignore", divide="ignore"):
        R = S / np.sqrt(V * V.T)
    R = np.where(N >= 24, R, 0.0)
    return la.nearest_corr(R)


def dcc_negll(params, Z, Qbar) -> float:
    a, b = params
    D = _dcc_path(Z, Qbar, a, b)[:-1]
    Q = Qbar[None] + D
    d = np.sqrt(np.clip(np.einsum("sii->si", Q), 1e-8, None))
    rho = np.clip(Q / (d[:, :, None] * d[:, None, :]), -0.999, 0.999)
    M = np.isfinite(Z)
    Z0 = np.nan_to_num(Z)
    iu = np.triu_indices(Z.shape[1], 1)
    r = rho[:, iu[0], iu[1]]
    zi, zj = Z0[:, iu[0]], Z0[:, iu[1]]
    m = M[:, iu[0]] & M[:, iu[1]]
    one = 1 - r ** 2
    ll = -0.5 * (np.log(one) + (zi ** 2 + zj ** 2 - 2 * r * zi * zj) / one) + 0.5 * (zi ** 2 + zj ** 2)
    return float(-(ll * m).sum() / max(m.sum(), 1))


def fit_dcc(Z: np.ndarray) -> tuple[float, float]:
    """Composite (pairwise) likelihood estimate of DCC(1,1) (a, b)."""
    Qbar = dcc_qbar(Z)

    def unpack(u):
        s = 0.999 / (1 + np.exp(-u[0]))       # a + b in (0, 0.999)
        f = 1 / (1 + np.exp(-u[1]))           # share of a
        return s * f, s * (1 - f)

    best = None
    for u0 in ([2.5, -2.5], [4.0, -3.5]):
        res = minimize(lambda u: dcc_negll(unpack(u), Z, Qbar), np.array(u0),
                       method="Nelder-Mead", options={"xatol": 1e-3, "fatol": 1e-7, "maxiter": 200})
        if best is None or res.fun < best.fun:
            best = res
    return unpack(best.x)


def dcc_forecast_corr(Z: np.ndarray, a: float, b: float, h: int) -> tuple[np.ndarray, np.ndarray]:
    """(average forecast correlation over months 1..h, Qbar correlation)."""
    Qbar = dcc_qbar(Z)
    D1 = _dcc_path(Z, Qbar, a, b)[-1]
    acc = np.zeros_like(Qbar)
    for k in range(1, h + 1):
        Q = Qbar + (a + b) ** (k - 1) * D1
        R, _ = la.cov_to_corr(Q)
        acc += R
    return la.nearest_corr(acc / h), Qbar


# --------------------------------------------------------------------------
# Daily correlations aggregated by month (for feature-based correlation models)
# --------------------------------------------------------------------------
def daily_monthly_sums(train: CovTrainingSet, assets=DAILY_MATCHING):
    """Per calendar month, the lead-lag cross-product sums of the daily block:
    C[m] (k x k, x x' + lag + lag'), V[m] (pairwise variance sums), N[m]
    (pairwise day counts), indexed by month-end; plus the positions of the
    daily assets in ``train.assets``. None if no daily data."""
    def f():
        blk = _daily_block(train, assets)
        if blk is None:
            return None
        X, M, idx = blk
        dates = train.daily.index
        Xl = np.vstack([np.zeros((1, X.shape[1])), X[:-1]])
        a = X ** 2 + 2 * X * Xl
        P = np.einsum("di,dj->dij", X, X) + np.einsum("di,dj->dij", X, Xl)
        P = P + P.transpose(0, 2, 1) - np.einsum("di,dj->dij", X, X)
        V = np.einsum("di,dj->dij", a, M)
        N = np.einsum("di,dj->dij", M, M)
        me = (dates + pd.offsets.MonthEnd(0)).normalize()
        starts = np.flatnonzero(np.r_[True, me[1:] != me[:-1]])
        months = pd.DatetimeIndex(me[starts])
        return (months, np.add.reduceat(P, starts, axis=0), np.add.reduceat(V, starts, axis=0),
                np.add.reduceat(N, starts, axis=0), idx)
    return _memo(train, ("daily_msums", tuple(assets)), f)


def _corr_from_sums(C, V, N, min_days):
    with np.errstate(invalid="ignore", divide="ignore"):
        R = C / np.sqrt(np.clip(V, 1e-300, None) * np.clip(np.swapaxes(V, -1, -2), 1e-300, None))
    return np.where(N >= min_days, np.clip(R, -0.999, 0.999), np.nan)


def daily_corr_panel(train: CovTrainingSet, recent_months: int = 6):
    """Monthly panel of daily lead-lag correlations: for each month-end s,
    recent (last ``recent_months`` months), long-run (all days through s), and
    the realized correlation over the next h months (NaN where s + h > t).
    Arrays are (months, k, k)."""
    def f():
        out = daily_monthly_sums(train)
        if out is None:
            return None
        months, C, V, N, idx = out
        h = train.horizon
        cC, cV, cN = (np.cumsum(A, axis=0) for A in (C, V, N))

        def window(A, cA, lo, hi):  # sum over months lo+1..hi (cumulative index)
            z = np.zeros_like(A[:1])
            cz = np.concatenate([z, cA], axis=0)
            return cz[hi + 1] - cz[lo + 1]
        T = len(months)
        pos = np.arange(T)
        lo_r = np.maximum(pos - recent_months, -1)
        rec = _corr_from_sums(window(C, cC, lo_r, pos), window(V, cV, lo_r, pos),
                              window(N, cN, lo_r, pos), int(0.8 * 21 * recent_months))
        lr = _corr_from_sums(cC, cV, cN, 500)
        fut = np.full_like(rec, np.nan)
        ok = pos + h < T
        p = pos[ok]
        fut[ok] = _corr_from_sums(window(C, cC, p, p + h), window(V, cV, p, p + h),
                                  window(N, cN, p, p + h), int(0.8 * 21 * h))
        return months, rec, lr, fut, idx
    return _memo(train, ("daily_corr_panel", recent_months), f)
