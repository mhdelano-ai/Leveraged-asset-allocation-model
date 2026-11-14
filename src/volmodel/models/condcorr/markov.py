"""Two-state Markov-switching model of the stock-bond relationship.

Observations (month ``m``), for the pair (us_large, rates anchor):
  * when both assets have daily data in the month (>= 10 returns each):
    non-overlapping ~5-trading-day sums of daily log excess returns inside
    the month (4-5 vectors; a 5-day sum is long enough for US-close pairs,
    see reports/data/horizon_evidence.md);
  * otherwise the month's log excess return (1 vector).
Each vector is standardized by the asset's trailing volatility (root mean
square of the previous 12 monthly log excess returns, scaled to the block
length) and winsorized at +-4, so that the states describe CO-MOVEMENT and
not volatility. Sufficient statistics per month: ``S_m = sum u u'``, ``n_m``.

Model: state ``k`` in {pos, neg}; ``u ~ N(0, Sigma_k)`` with variances
COMMON to both states (pooled) and a state-specific correlation ``rho_k``
(left free, the variances would make the states volatility regimes: the
likelihood gain from vol clustering swamps the correlation signal); first-order Markov chain with transition
matrix ``P`` (fixed) or, in the TVTP variant, staying probabilities
``logit p_kk(t) = a_k + c_k * infl_yoy(t)`` with ``c_pos >= 0`` and
``c_neg <= 0`` (high inflation makes the positive-correlation regime more
likely to persist / to be entered).

Estimation: EM (Hamilton filter + Kim smoother) on the months from 1952 to
the last December (refit yearly; deterministic start values). Priors, fixed
in advance: transition pseudo-counts 59 stay / 1 switch per state (prior
mean duration 60 months - the regimes of the literature last decades - with
the weight of five years of data); 10 pseudo-observations of an
uncorrelated unit-variance pair in each Sigma_k (keeps them well defined);
TVTP slopes N(0, 1) per unit of standardized inflation.

Forecast at origin ``t``: FILTERED probabilities ``pi_t`` (no smoothing,
only data <= t), propagated ``pi_{t+h} = pi_t P^h`` for h = 1..12 (with the
TVTP variant the transition matrix at the origin's inflation is held fixed),
the 12-month mixture ``Sigma = sum_k wbar_k Sigma_k`` with ``wbar`` the mean
of the propagated probabilities, and ``rho = Sigma_12 / sqrt(Sigma_11 Sigma_22)``.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .data import month_grid, predictors_at

TRAIN_START = pd.Timestamp("1952-01-31")
MIN_MONTHS = 120
BLOCK = 5
WINSOR = 4.0
PRIOR_STAY, PRIOR_SWITCH = 59.0, 1.0
PRIOR_SIGMA_N = 10.0
MAX_ITER, TOL = 300, 1e-7
INFL = "macro_infl_yoy"


@dataclass
class MSParams:
    Sig: np.ndarray            # 2 x 2 x 2  (state, i, j); state 0 = higher correlation
    P: np.ndarray | None       # 2 x 2 fixed transition matrix (None for TVTP)
    a: np.ndarray | None       # TVTP intercepts (logit stay), per state
    c: np.ndarray | None       # TVTP slopes on standardized inflation
    infl_mean: float = 0.0
    infl_sd: float = 1.0
    loglik: float = np.nan
    n_months: int = 0

    @property
    def rho(self) -> np.ndarray:
        return self.Sig[:, 0, 1] / np.sqrt(self.Sig[:, 0, 0] * self.Sig[:, 1, 1])

    def trans(self, infl_std: float | np.ndarray | None = None) -> np.ndarray:
        if self.P is not None:
            return self.P
        x = 0.0 if infl_std is None or not np.isfinite(infl_std) else float(infl_std)
        stay = 1.0 / (1.0 + np.exp(-(self.a + self.c * x)))
        return np.array([[stay[0], 1 - stay[0]], [1 - stay[1], stay[1]]])


# ------------------------------------------------------------- observations
def monthly_obs(view, a: str, b: str) -> pd.DataFrame:
    """Per month-end: S11, S12, S22 (sums of standardized cross-products), n."""
    grid = month_grid(view)
    xm = view.excess_monthly.reindex(index=grid)[[a, b]]
    # trailing vol from the previous 12 months (known at the start of the month)
    sig = np.sqrt((xm ** 2).rolling(12, min_periods=6).mean().shift(1))
    out = pd.DataFrame(np.nan, index=grid, columns=["S11", "S12", "S22", "n"])
    ok_m = xm.notna().all(axis=1) & sig.notna().all(axis=1)
    u = (xm / sig).clip(-WINSOR, WINSOR)
    out.loc[ok_m, "S11"] = (u[a] ** 2)[ok_m]
    out.loc[ok_m, "S12"] = (u[a] * u[b])[ok_m]
    out.loc[ok_m, "S22"] = (u[b] ** 2)[ok_m]
    out.loc[ok_m, "n"] = 1.0
    d = view.excess_daily[[a, b]]
    if len(d):
        mo = d.index + pd.offsets.MonthEnd(0)
        cnt = d.notna().groupby(mo).sum()
        both = cnt.index[(cnt[a] >= 10) & (cnt[b] >= 10)]
        dd = d[mo.isin(both)]
        if len(dd):
            mo2 = (dd.index + pd.offsets.MonthEnd(0)).as_unit("ns")
            mcode, muniq = pd.factorize(mo2, sort=True)
            pos = pd.Series(mcode).groupby(mcode).cumcount().to_numpy()
            size = np.bincount(mcode)[mcode]
            nblk = np.maximum(size // BLOCK, 1)
            blk = np.minimum(pos // BLOCK, nblk - 1)           # remainder joins the last block
            bid = mcode * 64 + blk                             # unique block id (blocks < 64)
            bcode, buniq = pd.factorize(bid, sort=True)
            x = dd.fillna(0.0).to_numpy(float)
            r = np.column_stack([np.bincount(bcode, weights=x[:, 0]), np.bincount(bcode, weights=x[:, 1])])
            ln = np.bincount(bcode).astype(float)
            bmonth = buniq // 64
            msize = np.bincount(mcode)[bmonth].astype(float)
            months = pd.DatetimeIndex(muniq)[bmonth]
            s = sig.reindex(months).to_numpy(float)
            scale = np.sqrt(ln / msize)[:, None]
            ub = (r / (s * scale)).clip(-WINSOR, WINSOR)
            okb = np.isfinite(ub).all(axis=1)
            st = pd.DataFrame({"S11": ub[:, 0] ** 2, "S12": ub[:, 0] * ub[:, 1], "S22": ub[:, 1] ** 2,
                               "n": 1.0}, index=months)[okb]
            agg = st.groupby(level=0).sum()
            agg = agg.reindex(grid).dropna()
            out.loc[agg.index] = agg.to_numpy()
    return out


# ------------------------------------------------------------------ filter
def _emission_loglik(S: np.ndarray, n: np.ndarray, Sig: np.ndarray) -> np.ndarray:
    """T x 2 log densities (constants dropped) of the month's vectors per state."""
    out = np.empty((len(n), 2))
    for k in range(2):
        inv = np.linalg.inv(Sig[k])
        ld = np.log(np.linalg.det(Sig[k]))
        tr = inv[0, 0] * S[:, 0] + 2 * inv[0, 1] * S[:, 1] + inv[1, 1] * S[:, 2]
        out[:, k] = -0.5 * (n * ld + tr)
    return out


def _trans_seq(par: MSParams, xs: np.ndarray | None, T: int) -> np.ndarray:
    if par.P is not None:
        return np.broadcast_to(par.P, (T, 2, 2))
    return np.stack([par.trans(x) for x in xs])


def hamilton_filter(L: np.ndarray, Pt: np.ndarray, pi0: np.ndarray):
    """Filtered probabilities (T x 2), predicted (T x 2), log likelihood.
    ``Pt[t]`` is the transition from month t-1 to t."""
    T = len(L)
    filt = np.empty((T, 2))
    pred = np.empty((T, 2))
    ll = 0.0
    p = pi0
    m = L.max(axis=1, keepdims=True)
    e = np.exp(L - m)
    for t in range(T):
        pr = p @ Pt[t] if t > 0 else p
        pred[t] = pr
        f = pr * e[t]
        s = f.sum()
        ll += np.log(s) + m[t, 0]
        p = f / s
        filt[t] = p
    return filt, pred, ll


def _smooth(filt, pred, Pt):
    T = len(filt)
    sm = np.empty_like(filt)
    sm[-1] = filt[-1]
    xi = np.zeros((T, 2, 2))      # joint P(s_{t-1}=i, s_t=j | all)
    for t in range(T - 2, -1, -1):
        ratio = sm[t + 1] / np.maximum(pred[t + 1], 1e-300)
        J = filt[t][:, None] * Pt[t + 1] * ratio[None, :]
        xi[t + 1] = J
        sm[t] = J.sum(axis=1)
    return sm, xi


def fit_ms(S: np.ndarray, n: np.ndarray, xs: np.ndarray | None = None, tvtp: bool = False) -> MSParams:
    T = len(n)
    Sig = np.array([[[1.0, 0.4], [0.4, 1.0]], [[1.0, -0.3], [-0.3, 1.0]]])
    P = np.array([[0.98, 0.02], [0.02, 0.98]])
    par = MSParams(Sig=Sig, P=None if tvtp else P, a=None, c=None)
    if tvtp:
        mu, sd = float(np.nanmean(xs)), float(np.nanstd(xs)) or 1.0
        par.infl_mean, par.infl_sd = mu, sd
        xs = np.where(np.isfinite(xs), (xs - mu) / sd, 0.0)
        par.a = np.log(np.array([0.98, 0.98]) / 0.02)
        par.c = np.zeros(2)
    pi0 = np.array([0.5, 0.5])
    prev = -np.inf
    prior_S = PRIOR_SIGMA_N * np.eye(2)
    v1, v2 = S[:, 0].sum() / n.sum(), S[:, 2].sum() / n.sum()   # common variances
    for k in range(2):
        rk = par.Sig[k, 0, 1]
        par.Sig[k] = np.array([[v1, rk * np.sqrt(v1 * v2)], [rk * np.sqrt(v1 * v2), v2]])
    for _ in range(MAX_ITER):
        L = _emission_loglik(S, n, par.Sig)
        Pt = _trans_seq(par, xs, T)
        filt, pred, ll = hamilton_filter(L, Pt, pi0)
        sm, xi = _smooth(filt, pred, Pt)
        # M-step: states share the variances and differ only in correlation
        # (moment form of the constrained M-step; pseudo-observations of an
        # uncorrelated unit-variance pair keep each state well defined)
        for k in range(2):
            w = sm[:, k]
            M = np.array([[w @ S[:, 0], w @ S[:, 1]], [w @ S[:, 1], w @ S[:, 2]]]) + prior_S
            r = M[0, 1] / np.sqrt(M[0, 0] * M[1, 1])
            par.Sig[k] = np.array([[v1, r * np.sqrt(v1 * v2)], [r * np.sqrt(v1 * v2), v2]])
        pi0 = sm[0]
        if not tvtp:
            cnt = xi[1:].sum(axis=0)
            cnt += np.array([[PRIOR_STAY, PRIOR_SWITCH], [PRIOR_SWITCH, PRIOR_STAY]])
            par.P = cnt / cnt.sum(axis=1, keepdims=True)
        else:
            for k in range(2):
                stay = xi[1:, k, k]
                tot = xi[1:, k, :].sum(axis=1)
                x = xs[1:]

                def nll(th, stay=stay, tot=tot, x=x):
                    eta = th[0] + th[1] * x
                    lp = -np.logaddexp(0, -eta)
                    lq = -np.logaddexp(0, eta)
                    f = -(stay @ lp + (tot - stay) @ lq)
                    # Beta(59, 1)-equivalent prior on the intercept, N(0, 1) on the slope
                    f -= PRIOR_STAY * (-np.logaddexp(0, -th[0])) + PRIOR_SWITCH * (-np.logaddexp(0, th[0]))
                    return f + 0.5 * th[1] ** 2
                bnd = [(None, None), (0.0, None) if k == 0 else (None, 0.0)]
                r = minimize(nll, np.array([par.a[k], par.c[k]]), method="L-BFGS-B", bounds=bnd)
                par.a[k], par.c[k] = r.x
        if abs(ll - prev) < TOL * max(1.0, abs(ll)):
            break
        prev = ll
    if par.rho[0] < par.rho[1]:                    # label: state 0 = higher correlation
        par.Sig = par.Sig[::-1].copy()
        if par.P is not None:
            par.P = par.P[::-1, ::-1].copy()
        else:
            par.a, par.c = par.a[::-1].copy(), par.c[::-1].copy()
    par.loglik, par.n_months = float(ll), T
    return par


def forecast_rho(par: MSParams, pi: np.ndarray, x_now: float | None = None, h: int = 12):
    P = par.trans(x_now)
    w = np.zeros(2)
    p = pi.copy()
    for _ in range(h):
        p = p @ P
        w += p
    w /= h
    Sig = w[0] * par.Sig[0] + w[1] * par.Sig[1]
    return float(Sig[0, 1] / np.sqrt(Sig[0, 0] * Sig[1, 1])), w


# ------------------------------------------------------------------ driver
_CACHE: dict = {}


def _key(*arrays) -> str:
    h = hashlib.sha1()
    for a in arrays:
        h.update(np.ascontiguousarray(a, dtype=float).tobytes())
    return h.hexdigest()


@dataclass
class MSForecast:
    rho: float
    prob_pos: float           # filtered probability of the high-correlation state at t
    w: np.ndarray
    par: MSParams


def predict_ms(view, a: str, b: str, tvtp: bool = False) -> MSForecast | None:
    obs = monthly_obs(view, a, b)
    obs = obs[obs.index >= TRAIN_START].dropna()
    if len(obs) == 0:
        return None
    origin = month_grid(view)[-1]
    y = origin.year if origin.month == 12 else origin.year - 1
    F = pd.Timestamp(year=y, month=12, day=31)
    xs_all = None
    if tvtp:
        # value known at the end of the PREVIOUS month drives the transition into this month
        xs_all = predictors_at(view, [INFL], month_grid(view))[INFL].shift(1).reindex(obs.index).to_numpy(float)
    fit_obs = obs[obs.index <= F]
    if len(fit_obs) < MIN_MONTHS:
        return None
    S = fit_obs[["S11", "S12", "S22"]].to_numpy(float)
    n = fit_obs["n"].to_numpy(float)
    xs = xs_all[: len(fit_obs)] if tvtp else None
    key = (a, b, tvtp, F, _key(S, n, xs if xs is not None else np.zeros(1)))
    par = _CACHE.get(key)
    if par is None:
        par = fit_ms(S, n, xs, tvtp)
        if len(_CACHE) > 512:
            _CACHE.clear()
        _CACHE[key] = par
    # filter through the origin with the fitted parameters
    S_all = obs[["S11", "S12", "S22"]].to_numpy(float)
    n_all = obs["n"].to_numpy(float)
    xstd = None
    if tvtp:
        xstd = np.where(np.isfinite(xs_all), (xs_all - par.infl_mean) / par.infl_sd, 0.0)
    L = _emission_loglik(S_all, n_all, par.Sig)
    Pt = _trans_seq(par, xstd, len(n_all))
    filt, _, _ = hamilton_filter(L, Pt, np.array([0.5, 0.5]))
    pi = filt[-1]
    if obs.index[-1] < origin:                     # no observation this month: predict forward
        gap = (origin.year - obs.index[-1].year) * 12 + origin.month - obs.index[-1].month
        pi = pi @ np.linalg.matrix_power(par.trans(xstd[-1] if tvtp else None), gap)
    x_now = None
    if tvtp:
        xn = predictors_at(view, [INFL], month_grid(view))[INFL].iloc[-1]
        x_now = (xn - par.infl_mean) / par.infl_sd if np.isfinite(xn) else 0.0
    rho, w = forecast_rho(par, pi, x_now)
    return MSForecast(rho=rho, prob_pos=float(pi[0]), w=w, par=par)
