"""The bridge: from the 21-day-horizon covariance ``cov_rv`` to the covariance
of the 12-MONTH log excess return ``cov_annual``, plus the distribution around it.

Why a bridge is needed
----------------------
``cov_rv`` (ARCHITECTURE.md) is the annualized covariance of overlapping
21-day returns: 12 x the one-month covariance. The 12-month return is the
sum of 12 monthly returns, so its covariance equals ``cov_rv`` only if
monthly returns are serially uncorrelated, within and across assets. With
standardized monthly returns ``u`` (return / ex-ante monthly vol) whose
lag-``k`` cross-correlation matrix is ``G_k = E[u_t u_{t-k}']``::

    cov_annual = S (R + D) S,      D = sum_{k=1}^{11} (1 - k/12) (G_k + G_k')

where ``S = diag(sqrt(diag cov_rv))`` and ``R`` the correlation of
``cov_rv`` (if the standardized-return correlation equals ``R``). The diagonal of ``R + D`` is the per-asset variance ratio
``VR_i = 1 + 2 sum_k (1 - k/12) rho_i(k)`` (Var of the 12-month return over
12 x the monthly variance); the off-diagonal ``D_ij`` is the lead-lag
contribution to the annual covariance (in correlation units of the monthly
returns). ``D`` is the Bartlett-weighted sum of autocorrelation matrices, i.e.
the Newey-West long-run correlation minus the contemporaneous one.

Estimation (``estimate_long_run``), from the view's MONTHLY log excess
returns only (monthly data exist in every era and are convention-free at
this horizon):

1. **Heteroskedasticity.** Returns are demeaned (in-view mean) and divided by
   an EX-ANTE monthly vol: an EWMA of past squared monthly returns, half-life
   ``HALFLIFE = 12`` months (fixed: one year, the same span as the target
   window), first ``WARMUP = 12`` months used only to initialize it. Crisis
   years then count as much as calm ones, which is what calibration over
   origins needs (each origin weights equally). Without it the estimate is a
   variance-weighted average dominated by the 1930s, 1970s and 2008.
2. **Overlapping estimator** in its autocorrelation form (Bartlett weights,
   all overlapping 12-month windows), with the small-sample bias of each
   demeaned autocorrelation removed (``E[rho_hat_k] = -(n-k)/(n(n-1))`` under
   the null; for a pair, ``-rho_0 (n-k)/(n(n-1))``). The uncorrected
   overlapping estimator is biased down by about ``(q-1)/n`` (0.05 at 20 years).
3. **Standard errors.** Variance ratios: Lo-MacKinlay heteroskedasticity-robust
   form ``Var = sum_k [2(1-k/q)]^2 delta_k / n``, ``delta_k = n sum (u_t
   u_{t-k})^2 / (sum u_t^2)^2`` (iid: ``2(2q-1)(q-1)/(3qn)``, i.e. 0.24 at
   20 years, 0.12 at 80), multiplied by the estimated VR (the estimator's
   relative error is constant; the null form alone under-states the error
   by ~1.5x at VR = 1.7 and over-states it at VR = 0.6). Annual-minus-monthly
   correlation gaps: ``se = (1 - rho_0^2) sqrt(2 sum_k (1-k/q)^2 / n_ij)``
   under no lead-lag (derived from the joint moments of the cross- and
   auto-correlation estimators).
4. **Era / pricing convention** (``era``): ``"market"`` (default) uses only
   months priced the way today's instrument is priced, i.e. it drops
   month-AVERAGE price/yield segments (which induce lag-1 autocorrelation of
   ~0.25 mechanically, Working 1960) and the matrix-priced high-yield fund NAV
   (VWEHX 1980-2007; lag-1 daily AC up to 0.57, documented in
   reports/data/credit.md). ``"strict"`` also drops exchange-traded but thinly
   traded eras (CRSP small caps before IWM, REIT funds before VNQ). ``"all"``
   uses everything; ``"recent"`` the last 360 months of market-priced data.
   The classification is fixed in ``NON_MARKET`` from the documented pricing
   mechanism of each segment (data/processed/segments.csv), not from returns.
5. **Shrinkage** (empirical Bayes toward 1 / toward 0): with prior
   ``VR_i - 1 ~ (0, tau^2)``, ``tau^2 = max(0, mean_i[(VR_i-1)^2 - se_i^2])``
   estimated across the assets in the view, the posterior mean is
   ``1 + w_i (VR_i - 1)``, ``w_i = tau^2 / (tau^2 + se_i^2)``.
6. **Correlations.** The null is that the annual correlation equals the
   21-day one: ``cov_annual = V^1/2 cov_rv V^1/2`` (``V = diag(VR)``); this is
   what the ``vr_*`` bridges do. (The additive form ``S(R + D)S`` with only
   the diagonal of ``D`` would LOWER every annual correlation by
   ``1/sqrt(VR_i VR_j)`` - for us_large-us_small from 0.88 to 0.68 - which the
   data contradict.) ``lr_market`` adds the gap ``g_ij = (rho_0 + D_ij) /
   sqrt(VR_i VR_j) - rho_0`` between the annual and monthly correlation of
   standardized returns, shrunk toward 0 by the same empirical Bayes across
   pairs: a lead-lag adjustment survives only where the cross-section of
   gaps is wider than their noise.
7. **Positive definiteness.** ``R + G`` is floored so that no portfolio's
   annual correlation-part variance falls below 0.5 x its 21-day one
   (``floor_relative``), then scaled by ``V^1/2``: variances are exactly
   ``VR_i x cov_rv_ii``.

No tuning constants besides ``HALFLIFE = 12``, ``q = 12`` and the 0.5 floor.

Parameters are re-estimated at most once a year: at origin ``t`` from the
view truncated at the last December month-end ``<= t`` (cached), so the
answer depends only on the view and is the same cold or warm.

Predictive distribution (``ScaleMixture``)
------------------------------------------
Given ``cov_annual`` the 12-month return is NOT normal: next year's
integrated variance ``V`` is uncertain (vol forecasts have large errors and
volatility clusters within the year). Model: ``R - mu = sqrt(V) Z``,
``V = A eps``, ``log eps ~ N(-s2/2, s2)`` (mean-one lognormal, so ``A``
stays the variance), ``Z ~ N(0,1)`` independent. Then ``Var(R) = A``,
kurtosis ``3 exp(s2)``, and quantiles follow by quadrature
(``ScaleMixture.quantile``). ``s2`` is calibrated walk-forward
(``vol_uncertainty``) from the errors of the reference vol forecast
(``ref_blend``: geometric mean of the all-history and 60-month variances)
against the realized 12-month variance ``sum r_m^2``, pooled over assets:
``s2 = Var(log RV/V_hat) - psi'(6)``, the last term removing the
measurement noise of a 12-observation realized variance under conditional
normality. The MEAN ``mu`` is outside this project: quantiles are for
``R - mu`` and assume ``mu`` is known (planner-supplied); uncertainty about
the mean would widen them further.

Public API::

    Bridge.cov_annual(cov_rv, view) -> DataFrame
    get_bridge(name) / BRIDGES (names: identity, vr_all, vr_market, vr_recent, lr_market)
    estimate_long_run(view, assets=None, era="market") -> LongRunEstimate
    standardized_returns(x) -> DataFrame
    variance_ratio(u, q=12) -> (vr, se_robust, se_iid, n)
    NON_MARKET, convention_labels(index, asset), market_mask(view_or_index, assets, era)
    vol_uncertainty(view) -> s2 ;  ScaleMixture(s2).quantile(p) / .kurtosis / .share_beyond(c)
    predictive_quantiles(cov_annual, weights, s2, probs) -> DataFrame
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib

import numpy as np
import pandas as pd
from scipy.signal import lfilter
from scipy.special import polygamma
from scipy.stats import norm

from ..universe import HORIZON_MONTHS

__all__ = [
    "Q", "HALFLIFE", "WARMUP", "NON_MARKET", "ERAS",
    "convention_labels", "market_mask", "standardized_returns", "variance_ratio",
    "LongRunEstimate", "estimate_long_run", "eb_shrink", "repair_pd",
    "Bridge", "IdentityBridge", "LongRunBridge", "BRIDGES", "get_bridge",
    "ScaleMixture", "vol_uncertainty", "predictive_quantiles",
]

Q = HORIZON_MONTHS          # 12 months summed
HALFLIFE = 12               # months, EWMA ex-ante vol used to standardize
WARMUP = 12                 # months used only to start the EWMA
MIN_OBS = 24                # standardized months below which an estimate is not attempted
RECENT_MONTHS = 360         # "recent" era: last 30 years
VR_BOUNDS = (0.25, 4.0)     # sanity clip on applied ratios
ERAS = ("all", "market", "strict", "recent")

# --------------------------------------------------------------------------
# Pricing convention of each segment (data/processed/segments.csv; reports/data).
# (asset, first month, last month, label). Months not listed are "market":
# priced by trades in exchange-listed securities, ETFs, futures or fixings,
# or by end-of-month quotes - the way today's instruments are priced.
#   averaged : month-AVERAGE prices or yields (induces lag-1 AC ~0.25)
#   smoothed : fund NAV of matrix-priced OTC holdings (stale for weeks)
#   thin     : exchange-traded constituents, but thinly traded in that era
NON_MARKET: list[tuple[str, str, str, str]] = [
    ("us_large", "1871-01-31", "1926-06-30", "averaged"),   # Shiller/Cowles month-average prices
    ("ust_int", "1953-01-31", "1961-12-31", "averaged"),    # H.15 monthly-average yields
    ("ust_long", "1919-01-31", "1961-12-31", "averaged"),   # NBER / H.15 monthly-average yields
    ("ig_credit", "1919-01-31", "1925-12-31", "averaged"),  # Moody's month-average yields
    ("hy_credit", "1980-01-31", "2007-04-30", "smoothed"),  # VWEHX NAV (monthly ex-div 1980-84, daily 1985-2007)
    ("us_small", "1926-01-31", "2000-05-31", "thin"),       # CRSP size quintiles 2-3 before IWM
    ("reits", "1972-01-31", "2004-09-30", "thin"),          # Nareit index (tiny market), FRESX/VGSIX NAVs
]
_EXCLUDE = {"all": set(), "market": {"averaged", "smoothed"}, "recent": {"averaged", "smoothed"},
            "strict": {"averaged", "smoothed", "thin"}}


def convention_labels(index: pd.DatetimeIndex, asset: str) -> np.ndarray:
    """Pricing-convention label ("market", "averaged", "smoothed", "thin") per month."""
    lab = np.full(len(index), "market", dtype=object)
    for a, s, e, lb in NON_MARKET:
        if a == asset:
            m = (index >= pd.Timestamp(s)) & (index <= pd.Timestamp(e) + pd.offsets.MonthEnd(0))
            lab[m] = lb
    return lab


def market_mask(index: pd.DatetimeIndex, assets: list[str], era: str = "market") -> np.ndarray:
    """T x N bool: months usable for estimation under ``era`` (before data availability)."""
    if era not in _EXCLUDE:
        raise ValueError(f"era must be one of {ERAS}")
    ex = _EXCLUDE[era]
    out = np.ones((len(index), len(assets)), bool)
    for j, a in enumerate(assets):
        if ex:
            out[:, j] = ~np.isin(convention_labels(index, a), list(ex))
    return out


# --------------------------------------------------------------- estimation
def standardized_returns(x: pd.DataFrame, halflife: float = HALFLIFE, warmup: int = WARMUP) -> pd.DataFrame:
    """``u_t = (r_t - mean) / sigma_{t-1}``, sigma from an EWMA of past squared
    demeaned monthly returns (ex-ante: month t's own return is not used).

    Each asset's history is taken from its first to last valid month (gaps
    inside are skipped by the EWMA); the first ``warmup`` months only start it.
    """
    lam = 0.5 ** (1.0 / halflife)
    out = pd.DataFrame(np.nan, index=x.index, columns=x.columns)
    for a in x.columns:
        s = x[a].dropna()
        if len(s) <= warmup:
            continue
        r = s.to_numpy(float) - s.mean()
        r2 = r ** 2
        init = r2[:warmup].mean()
        # v[i] = EWMA through i (inclusive); seed with the warm-up mean
        v, _ = lfilter([1 - lam], [1, -lam], r2[warmup:], zi=[lam * init])
        prev = np.concatenate([[init], v[:-1]])          # sigma^2_{t-1} for t = warmup..
        u = r[warmup:] / np.sqrt(np.maximum(prev, 1e-12))
        out.loc[s.index[warmup:], a] = u
    return out


def _lag_products(U: np.ndarray, M: np.ndarray, q: int):
    """Cross-moment sums for lags 0..q-1 with pairwise-valid masking.

    Returns num[k] (N x N): sum_t u_i,t u_j,t-k over t where both assets are valid
    at t and t-k; and for k = 0 the pairwise denominators.
    """
    T, N = U.shape
    U0 = np.where(M, U, 0.0)
    Mf = M.astype(float)
    num = np.zeros((q, N, N))
    cnt = np.zeros((q, N, N))
    for k in range(q):
        if k == 0:
            num[0] = U0.T @ U0
            cnt[0] = Mf.T @ Mf
            continue
        W = Mf[k:] * Mf[:-k]                      # valid at t and t-k (per asset)
        A = U0[k:] * W
        B = U0[:-k] * W
        num[k] = A.T @ B
        cnt[k] = W.T @ W
    return num, cnt


def variance_ratio(u, q: int = Q):
    """Bias-corrected overlapping variance ratio of one standardized series.

    Returns ``(vr, se_robust, se_iid, n)``; NaN-free input of length n.
    """
    u = np.asarray(u, float)
    u = u[np.isfinite(u)]
    n = len(u)
    if n < MIN_OBS:
        return 1.0, np.inf, np.inf, n
    d = u - u.mean()
    s2 = (d * d).sum()
    vr, var_rob = 1.0, 0.0
    for k in range(1, q):
        rho = (d[k:] * d[:-k]).sum() / s2 + (n - k) / (n * (n - 1))
        wk = 2.0 * (1 - k / q)
        vr += wk * rho
        delta = n * ((d[k:] * d[:-k]) ** 2).sum() / s2 ** 2
        var_rob += wk ** 2 * delta / n
    se_iid = np.sqrt(2 * (2 * q - 1) * (q - 1) / (3 * q * n))
    # The null-based forms are exact only at VR = 1. The overlapping estimator is a
    # Bartlett spectral estimate whose RELATIVE error is constant, so the standard
    # error under an alternative is ~ VR x the null one (verified by simulation for
    # AR(1) +-0.3, MA(1) smoothing and stochastic volatility; see tests).
    scale = max(float(vr), VR_BOUNDS[0])
    return float(vr), float(np.sqrt(var_rob)) * scale, float(se_iid) * scale, n


@dataclass
class LongRunEstimate:
    """Everything the bridge estimated from one view (standardized-return units)."""

    assets: list
    era: str
    n: pd.Series                 # standardized months used per asset
    vr_raw: pd.Series            # bias-corrected, unshrunk variance ratios
    vr_se: pd.Series             # robust standard errors
    vr_se_iid: pd.Series
    vr: pd.Series                # shrunk (applied) ratios
    tau2_vr: float
    rho0: pd.DataFrame           # contemporaneous corr of standardized monthly returns (pairwise sample)
    D_raw: pd.DataFrame          # lead-lag sums sum_k (1-k/q)(rho_ij(k) + rho_ji(k)); diag = vr_raw - 1
    rho_annual_raw: pd.DataFrame  # (rho0 + D_raw) / sqrt(vr_raw_i vr_raw_j)
    gap_raw: pd.DataFrame        # rho_annual_raw - rho0 (annual minus monthly correlation)
    gap_se: pd.DataFrame         # (1 - rho0^2) sqrt(2 sum_k (1-k/q)^2 / n_ij) under no lead-lag
    gap: pd.DataFrame            # EB-shrunk gap (zero unless cross=True)
    tau2_gap: float
    n_pair: pd.DataFrame
    anchor: pd.Timestamp | None = None

    @property
    def shrink_weight(self) -> pd.Series:
        return (self.vr - 1) / (self.vr_raw - 1).where((self.vr_raw - 1).abs() > 1e-12)


def eb_shrink(est: np.ndarray, se: np.ndarray, prior: float = 0.0):
    """Empirical-Bayes normal-normal shrinkage of ``est`` toward ``prior``.

    ``tau^2 = max(0, mean[(est-prior)^2 - se^2])`` over finite entries.
    Returns (shrunk, weights, tau2). Infinite se -> weight 0.
    """
    est, se = np.asarray(est, float), np.asarray(se, float)
    ok = np.isfinite(est) & np.isfinite(se)
    if ok.sum() == 0:
        return np.full_like(est, prior), np.zeros_like(est), 0.0
    tau2 = max(0.0, float(np.mean((est[ok] - prior) ** 2 - se[ok] ** 2)))
    w = np.where(ok, tau2 / (tau2 + np.where(ok, se, 1.0) ** 2 + 1e-300), 0.0)
    shrunk = prior + w * np.where(ok, est - prior, 0.0)
    return shrunk, w, tau2


def estimate_long_run(view, assets: list[str] | None = None, era: str = "market", q: int = Q,
                      halflife: float = HALFLIFE, cross: bool = True) -> LongRunEstimate:
    """Estimate variance ratios and annual-vs-monthly correlation gaps from ``view.excess_monthly``."""
    x = view.excess_monthly
    assets = list(x.columns) if assets is None else list(assets)
    x = x[assets]
    U = standardized_returns(x, halflife=halflife)
    idx = U.index
    M = U.notna().to_numpy() & market_mask(idx, assets, "market" if era == "recent" else era)
    if era == "recent" and len(idx):
        M &= np.asarray(idx > idx[-1] - pd.DateOffset(months=RECENT_MONTHS))[:, None]
    Uv = U.to_numpy(float)
    N = len(assets)
    n_i = M.sum(axis=0)
    mu = np.where(n_i > 0, np.nansum(np.where(M, Uv, 0.0), axis=0) / np.maximum(n_i, 1), 0.0)
    Ud = np.where(M, Uv - mu, 0.0)

    # per-asset variance ratios (single-series estimator on each usable sample)
    vr_raw = np.ones(N)
    se = np.full(N, np.inf)
    se_iid = np.full(N, np.inf)
    for j in range(N):
        v, s_r, s_i, n = variance_ratio(Uv[M[:, j], j], q)
        vr_raw[j], se[j], se_iid[j] = v, s_r, s_i
    vr_s, _, tau2 = eb_shrink(vr_raw, se, prior=1.0)
    vr_s = np.clip(vr_s, *VR_BOUNDS)

    # lead-lag sums and the annual-minus-monthly correlation gap
    num, cnt = _lag_products(Ud, M, q)
    ss = (Ud ** 2).T @ M.astype(float)            # sum u_i^2 over months where j is also valid
    wsum = sum((1 - k / q) ** 2 for k in range(1, q))
    with np.errstate(invalid="ignore", divide="ignore"):
        den = np.sqrt(ss * ss.T)
        rho0 = num[0] / den
        n_pair = cnt[0]
        D = np.zeros((N, N))
        for k in range(1, q):
            G = num[k] / den + rho0 * (n_pair - k) / (n_pair * (n_pair - 1))
            D += (1 - k / q) * np.where(np.isfinite(G), G + G.T, 0.0)
        vv = np.sqrt(np.outer(np.maximum(vr_raw, VR_BOUNDS[0]), np.maximum(vr_raw, VR_BOUNDS[0])))
        rho_a = (rho0 + D) / vv
        gap = rho_a - rho0
        gap_se = (1 - rho0 ** 2) * np.sqrt(2 * wsum / n_pair)
    small = ~np.isfinite(gap) | ~np.isfinite(gap_se) | (n_pair < MIN_OBS)
    for k in range(N):
        if not np.isfinite(se[k]):
            small[k, :] = small[:, k] = True
    rho0 = np.where(np.isfinite(rho0), rho0, 0.0)
    D = np.where(small, 0.0, D)
    rho_a = np.where(small, np.nan, rho_a)
    gap = np.where(small, 0.0, gap)
    gap_se = np.where(small, np.inf, gap_se)
    iu = np.triu_indices(N, 1)
    G_s = np.zeros((N, N))
    tau2_g = 0.0
    if cross and N > 1:
        sh, _, tau2_g = eb_shrink(gap[iu], gap_se[iu], prior=0.0)
        G_s[iu] = sh
        G_s = G_s + G_s.T
    np.fill_diagonal(D, vr_raw - 1)
    np.fill_diagonal(rho_a, 1.0)
    np.fill_diagonal(gap, 0.0)
    np.fill_diagonal(gap_se, 0.0)
    S = lambda a: pd.Series(a, index=assets)
    F = lambda a: pd.DataFrame(a, index=assets, columns=assets)
    return LongRunEstimate(
        assets=assets, era=era, n=S(M.sum(axis=0)), vr_raw=S(vr_raw), vr_se=S(se), vr_se_iid=S(se_iid),
        vr=S(vr_s), tau2_vr=tau2, rho0=F(rho0), D_raw=F(D), rho_annual_raw=F(rho_a), gap_raw=F(gap),
        gap_se=F(gap_se), gap=F(G_s), tau2_gap=float(tau2_g), n_pair=F(n_pair), anchor=view.asof_date,
    )


def repair_pd(M: np.ndarray, rel_floor: float = 1e-4) -> np.ndarray:
    """Symmetric PD matrix close to ``M`` with the SAME diagonal.

    Eigenvalues are clipped at ``rel_floor x mean diagonal`` and the result is
    rescaled (congruence by a diagonal) to restore the original diagonal.
    """
    M = (M + M.T) / 2.0
    d0 = np.diag(M).copy()
    w, V = np.linalg.eigh(M)
    floor = rel_floor * max(float(np.mean(d0)), 1e-12)
    if w.min() >= floor:
        return M
    C = (V * np.maximum(w, floor)) @ V.T
    s = np.sqrt(d0 / np.diag(C))
    C = C * np.outer(s, s)
    return (C + C.T) / 2.0


PORTFOLIO_VR_FLOOR = 0.5   # no portfolio's annual correlation-part variance below half the monthly one


def floor_relative(C: np.ndarray, R: np.ndarray, kappa: float = PORTFOLIO_VR_FLOOR) -> np.ndarray:
    """Correlation matrix ``C`` adjusted so that ``w'C w >= kappa w'R w`` for all ``w``.

    Generalized eigenvalues of (C, R) are clipped at ``kappa`` (R assumed PD) and
    the unit diagonal is restored. ``kappa = 0.5`` bounds the implied
    portfolio variance ratio below at a level no estimate comes near
    (monthly autocorrelation of about -0.2 at every lag 1-11), so it only
    acts on directions created by estimation noise in the correlation gaps,
    which optimizers would otherwise exploit.
    """
    R = (R + R.T) / 2.0
    w0, V0 = np.linalg.eigh(R)
    w0 = np.maximum(w0, 1e-10 * max(w0.max(), 1e-12))
    Rh = (V0 * np.sqrt(w0)) @ V0.T
    Rih = (V0 / np.sqrt(w0)) @ V0.T
    K = Rih @ C @ Rih
    K = (K + K.T) / 2.0
    w, V = np.linalg.eigh(K)
    if w.min() >= kappa:
        return C
    K = (V * np.maximum(w, kappa)) @ V.T
    out = Rh @ K @ Rh
    d = np.sqrt(np.diag(out))
    out = out / np.outer(d, d)
    np.fill_diagonal(out, 1.0)
    return (out + out.T) / 2.0


# ------------------------------------------------------------------ bridges
def _anchor(view) -> pd.Timestamp | None:
    """Last December month-end <= asof (parameters re-estimated once a year)."""
    idx = view.monthly.index
    if len(idx) == 0:
        return None
    t = idx[-1]
    dec = idx[idx.month == 12]
    return dec[-1] if len(dec) else t


def _fingerprint(view, anchor) -> str:
    m = view.monthly.loc[:anchor]
    c = view.cash_monthly.loc[:anchor]
    h = hashlib.blake2b(digest_size=16)
    h.update(np.ascontiguousarray(m.to_numpy(float)).tobytes())
    h.update(np.ascontiguousarray(c.to_numpy(float)).tobytes())
    h.update(",".join(m.columns).encode())
    h.update(str(anchor).encode())
    return h.hexdigest()


class Bridge:
    """Maps a ``cov_rv`` forecast to ``cov_annual`` using only ``view``."""

    name = "bridge"

    def cov_annual(self, cov_rv: pd.DataFrame, view) -> pd.DataFrame:  # pragma: no cover
        raise NotImplementedError

    @staticmethod
    def _check(cov_rv: pd.DataFrame) -> np.ndarray:
        C = np.asarray(cov_rv, float)
        if C.shape[0] != C.shape[1] or list(cov_rv.index) != list(cov_rv.columns):
            raise ValueError("cov_rv must be square with matching labels")
        if not np.isfinite(C).all():
            raise ValueError("cov_rv has non-finite entries")
        return (C + C.T) / 2.0


class IdentityBridge(Bridge):
    """``cov_annual = cov_rv`` (serially uncorrelated monthly returns); PD-repaired."""

    name = "identity"

    def cov_annual(self, cov_rv, view=None):
        C = self._check(cov_rv)
        return pd.DataFrame(repair_pd(C, 1e-10), index=cov_rv.index, columns=cov_rv.columns)


class LongRunBridge(Bridge):
    """``cov_annual = V^1/2 S (R + G) S V^1/2``: shrunk variance ratios ``V`` and,
    with ``cross=True``, the shrunk annual-minus-monthly correlation gaps ``G``
    (``cross=False``: annual correlation = 21-day correlation)."""

    def __init__(self, name: str, era: str = "market", cross: bool = False, halflife: float = HALFLIFE):
        self.name, self.era, self.cross, self.halflife = name, era, cross, halflife
        self._cache: dict = {}

    def estimate(self, view, assets: list[str]) -> LongRunEstimate:
        anchor = _anchor(view)
        assets = list(assets)
        key = (_fingerprint(view, anchor), tuple(assets))
        est = self._cache.get(key)
        if est is None:
            sub = view.asof(anchor) if anchor is not None else view
            est = estimate_long_run(sub, assets, era=self.era, halflife=self.halflife, cross=self.cross)
            if len(self._cache) > 512:
                self._cache.clear()
            self._cache[key] = est
        return est

    def cov_annual(self, cov_rv, view):
        C = self._check(cov_rv)
        assets = list(cov_rv.index)
        est = self.estimate(view, assets)
        s = np.sqrt(np.maximum(np.diag(C), 0.0))
        with np.errstate(invalid="ignore", divide="ignore"):
            R = C / np.outer(s, s)
        R = np.where(np.isfinite(R), R, 0.0)
        np.fill_diagonal(R, 1.0)
        R = repair_pd(R, 1e-8)
        if self.cross:
            Ca = R + est.gap.loc[assets, assets].to_numpy()
            np.fill_diagonal(Ca, 1.0)
            Ca = floor_relative(Ca, R)
        else:
            Ca = R
        sa = s * np.sqrt(est.vr.loc[assets].to_numpy())
        A = Ca * np.outer(sa, sa)
        return pd.DataFrame((A + A.T) / 2.0, index=cov_rv.index, columns=cov_rv.columns)


BRIDGES = {
    "identity": IdentityBridge,
    "vr_all": lambda: LongRunBridge("vr_all", era="all"),
    "vr_market": lambda: LongRunBridge("vr_market", era="market"),
    "vr_recent": lambda: LongRunBridge("vr_recent", era="recent"),
    "lr_market": lambda: LongRunBridge("lr_market", era="market", cross=True),
}
DEFAULT_BRIDGE = "vr_market"


def get_bridge(name: str = DEFAULT_BRIDGE) -> Bridge:
    """A fresh bridge instance by name (see ``BRIDGES``)."""
    try:
        return BRIDGES[name]()
    except KeyError:
        raise KeyError(f"unknown bridge {name!r}; known: {sorted(BRIDGES)}") from None


# --------------------------------------------------- predictive distribution
_GH_X, _GH_W = np.polynomial.hermite_e.hermegauss(64)    # nodes for E[f(Y)], Y ~ N(0,1)
_GH_W = _GH_W / _GH_W.sum()
PSI1_6 = float(polygamma(1, HORIZON_MONTHS / 2))           # Var(log chi2_12 / 12) = 0.181


@dataclass(frozen=True)
class ScaleMixture:
    """Standardized 12-month return ``X = sqrt(eps) Z``, ``log eps ~ N(-s2/2, s2)``.

    Var(X) = 1 for every s2; s2 = 0 is the normal.
    """

    s2: float

    @property
    def kurtosis(self) -> float:
        return 3.0 * float(np.exp(self.s2))

    def _scales(self):
        s = np.sqrt(max(self.s2, 0.0))
        return np.exp(0.5 * (-self.s2 / 2 + s * _GH_X))      # sqrt(eps) at the nodes

    def cdf(self, x):
        x = np.asarray(x, float)
        sc = self._scales()
        return (norm.cdf(x[..., None] / sc) * _GH_W).sum(axis=-1)

    def quantile(self, p: float) -> float:
        """p-quantile of X (p < 0.5 gives a negative number)."""
        if self.s2 <= 0:
            return float(norm.ppf(p))
        from scipy.optimize import brentq
        return float(brentq(lambda x: self.cdf(x) - p, -50.0, 50.0, xtol=1e-10))

    def share_beyond(self, c: float) -> float:
        """P(|X| > c)."""
        return float(2 * (1 - self.cdf(c)))


S2_FALLBACK = 0.30      # used only when a view has < min_obs forecast errors (round number)


def _month_var_shares(view, assets) -> pd.DataFrame:
    """Per asset-month sum of squared DAILY log excess returns (NaN if < 10 days).

    Used only through scale-free ratios within a 12-month window, so the
    pricing convention of the daily source (a constant factor per segment)
    cancels.
    """
    xd = view.excess_daily[assets]
    if xd.empty:
        return pd.DataFrame(np.nan, index=view.monthly.index, columns=assets)
    g = xd.groupby(pd.Grouper(freq="ME"))
    s = (xd ** 2).groupby(pd.Grouper(freq="ME")).sum(min_count=1)
    n = g.count()
    s = s.where(n >= 10)
    return s.reindex(view.monthly.index)


def vol_uncertainty(view, min_obs: int = 120, era: str = "market") -> dict:
    """Walk-forward dispersion of next-year variance around the reference forecast.

    For every past origin ``s`` (12-month window inside the view) and every
    asset with >= 60 months before ``s`` and a window of ``era``-priced months:
    ``V_hat`` = ref_blend variance (geometric mean of 12 x mean r^2 over all
    history and over the last 60 months), ``RV`` = sum of the 12 squared
    monthly returns in the window, ``e = log(RV / V_hat)``.

    ``RV`` measures the integrated variance ``V`` with noise: for conditionally
    normal months ``RV/V ~ Gamma(a, 1/a)`` with ``a = n_eff / 2``,
    ``n_eff = (sum sigma_m^2)^2 / sum sigma_m^4`` (12 if the months are equally
    volatile, fewer when one month carries much of the year's variance).
    ``n_eff`` is measured from daily data where every month of the window has
    it (scale-free), else 12. Then, with ``m = psi(a) - log a`` and
    ``v = psi'(a)``:

      s2      = Var(e - m) - mean(v)                       (default)
      s2_gap  = 2 [log mean(RV/V_hat) - mean(log RV/V_hat) + mean(m)]
                (Jensen gap; does not need V_hat unbiased, cross-check)

    Returns a dict with s2, s2_gap, var_e, mean_e, mean_neff, n.
    """
    x = view.excess_monthly
    assets = list(x.columns)
    mv = _month_var_shares(view, assets)
    ok_era = market_mask(x.index, assets, era)
    E, Mn, Vn, R, NE = [], [], [], [], []
    for j, a in enumerate(assets):
        s = x[a]
        first, last = s.first_valid_index(), s.last_valid_index()
        if first is None:
            continue
        sl = slice(x.index.get_loc(first), x.index.get_loc(last) + 1)
        r = s.iloc[sl].to_numpy(float)
        if np.isnan(r).any() or len(r) < 60 + Q:
            r = np.nan_to_num(r)  # interior gaps are not expected; treat as zero-return months
        n = len(r)
        if n < 60 + Q:
            continue
        c2 = np.concatenate([[0.0], np.cumsum(r ** 2)])
        t = np.arange(60, n - Q + 1)
        vhat = np.sqrt(12 * c2[t] / t * 12 * (c2[t] - c2[t - 60]) / 60)
        rv = c2[t + Q] - c2[t]
        okm = ok_era[sl, j].astype(float)
        co = np.concatenate([[0.0], np.cumsum(okm)])
        win_ok = (co[t + Q] - co[t]) == Q
        d = mv[a].iloc[sl].to_numpy(float)
        dok = np.isfinite(d)
        d0 = np.where(dok, d, 0.0)
        c1, cq, cn = (np.concatenate([[0.0], np.cumsum(z)]) for z in (d0, d0 ** 2, dok.astype(float)))
        s1, sq, nn = c1[t + Q] - c1[t], cq[t + Q] - cq[t], cn[t + Q] - cn[t]
        with np.errstate(invalid="ignore", divide="ignore"):
            neff = np.where((nn == Q) & (sq > 0), s1 ** 2 / sq, float(Q))
        neff = np.clip(neff, 2.0, float(Q))
        ok = win_ok & (vhat > 0) & (rv > 0)
        aa = neff[ok] / 2
        E.append(np.log(rv[ok] / vhat[ok]))
        R.append(rv[ok] / vhat[ok])
        Mn.append(polygamma(0, aa) - np.log(aa))
        Vn.append(polygamma(1, aa))
        NE.append(neff[ok])
    nobs = sum(len(e) for e in E)
    if nobs < min_obs:
        return {"s2": S2_FALLBACK, "s2_gap": np.nan, "var_e": np.nan, "mean_e": np.nan,
                "mean_neff": np.nan, "n": nobs}
    e, m, v, ratio, ne = (np.concatenate(z) for z in (E, Mn, Vn, R, NE))
    s2 = max(0.0, float(np.var(e - m, ddof=1) - v.mean()))
    s2_gap = max(0.0, float(2 * (np.log(ratio.mean()) - e.mean() + m.mean())))
    return {"s2": s2, "s2_gap": s2_gap, "var_e": float(e.var(ddof=1)), "mean_e": float(e.mean()),
            "mean_neff": float(ne.mean()), "n": int(nobs)}


def predictive_quantiles(cov_annual: pd.DataFrame, weights: pd.Series | pd.DataFrame | None = None,
                         s2: float = 0.0, probs=(0.05, 0.01)) -> pd.DataFrame:
    """Lower-tail quantiles of the 12-month log excess return minus its mean.

    ``weights``: None (each asset), a Series (one portfolio) or a DataFrame
    (portfolios x assets). Returns sigma, and for each p the normal quantile,
    the mixture quantile and their ratio. Assumes the mean is known and is
    added by the caller (``R_p = mu_p + quantile``).
    """
    A = cov_annual
    if weights is None:
        W = pd.DataFrame(np.eye(len(A)), index=A.index, columns=A.columns)
    elif isinstance(weights, pd.Series):
        W = weights.to_frame().T.reindex(columns=A.columns, fill_value=0.0)
    else:
        W = weights.reindex(columns=A.columns, fill_value=0.0)
    sig = np.sqrt(np.einsum("pi,ij,pj->p", W.to_numpy(), A.to_numpy(), W.to_numpy()))
    mix = ScaleMixture(s2)
    out = {"sigma": sig}
    for p in probs:
        qn, qm = norm.ppf(p), mix.quantile(p)
        out[f"q{p:g}_normal"] = sig * qn
        out[f"q{p:g}_mixture"] = sig * qm
        out[f"ratio{p:g}"] = np.full(len(sig), qm / qn)
    return pd.DataFrame(out, index=W.index)
