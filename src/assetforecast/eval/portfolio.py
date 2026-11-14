"""Portfolio tests of covariance forecasts: does a forecast matrix WORK when
used the way planners use it, to build portfolios and predict their risk?

Statistical losses on individual vols and correlations live elsewhere in
``assetforecast.eval``. This module asks four questions of each forecast matrix
``H_t`` (annualized covariance of overlapping 21-trading-day log excess
returns over the window ``(t, t + 12 months]``, see ARCHITECTURE.md):

1. **Risk prediction.** For a fixed set of portfolios built from ``H_t``,
   is the predicted vol ``sqrt(w'H w)`` close to the vol realized in the
   window? (bias statistic, QLIKE, squared log error)
2. **Economic value.** Do minimum-variance portfolios built from ``H_t``
   realize low variance (Engle and Colacito 2006: comparing realized
   variance of GMV portfolios ranks covariance forecasts without needing
   return forecasts)? How accurately does a 10%-vol target land? How much do
   optimized weights churn from month to month?
3. **Annual calibration** (for the bridge's ``cov_annual``): are 12-month
   portfolio log excess returns, standardized by their predicted annual
   vol, unit-variance?
4. **Summaries** by model, portfolio and era over common origins.

Conventions (all binding for interpretation of the numbers)
-----------------------------------------------------------
* **No realized data in construction.** Weights at origin ``t`` depend only on
  ``H_t`` (and, for random portfolios, on the origin and the asset set).
  The test universe is ``assets(H_t)`` minus assets with insufficient daily
  data in the window; this uses data *availability*, never return values,
  and is identical for every model at that origin. Dropped assets are listed
  in the ``dropped`` column of every output row.
* **Holding rule.** Weights are held FIXED with DAILY rebalancing over the
  whole window. Portfolio daily log excess return is approximated by the
  weighted sum of asset daily log excess returns, ``r_p = w'r``, so its
  21-day log excess return is ``w's`` (``s`` = the assets' 21-day sums).
  Realized portfolio variance is then the quadratic form ``w' S w``.
* **Realized measure.** PRIMARY: ``S`` is exactly the ``cov_rv`` target
  matrix (``targets.realized_kday_cov``): ``(n_rows/21) * mean s s'`` over the
  overlapping 21-day sums lying inside the window, uncentered. So
  ``w'Sw`` is the uncentered annualized variance of the portfolio's
  overlapping 21-day returns. It is insensitive to asynchronous closes and
  to smoothed / overreacting pricing conventions, and as noisy as the target
  (~17% SE on a realized vol). DIRECT (secondary, ``*_direct`` metrics,
  DIAGNOSTIC ONLY): ``S = X'X`` (summed over the year) from same-day 1-day returns
  with missing days set to 0. It is CONVENTION-DEPENDENT: it depends on how
  each source is priced (smoothed NAVs read low, ETFs of foreign/illiquid
  assets read high), is biased low for portfolios mixing US and non-US
  assets, and is NOT what forecasts target; never rank models on it.
* **Missing data.** An asset is in the window's daily test universe if it
  has a daily observation on at least ``min_coverage`` (default 90%) of the
  window's panel days and at least ``min_days_per_month`` (10) in every
  calendar month of the window (the same rules as the targets). Missing
  days of kept assets mean "did not trade; the next return spans the gap",
  so they contribute 0 to the 21-day sums and to the direct measure.
* **Losses** (``PV`` predicted variance, ``RV`` realized variance):
  ``qlike = RV/PV - log(RV/PV) - 1`` (Patton's robust QLIKE, >= 0, 0 when
  exact) and ``var_ratio = RV/PV`` (bias statistic whose mean is 1 for an
  unbiased forecast) are consistent under the noisy realized measure.
  ``log_ratio = log(sqrt(RV)/sqrt(PV))`` and ``sq_log_err = log(RV/PV)^2``
  are NOT: with ~17% noise on a realized vol a perfect forecast has mean
  ``log_ratio`` of about -0.03 (Jensen) and ``sq_log_err`` favours
  forecasts biased low. Read them as descriptive only.
* **Turnover** of optimized weights is one-way, ``0.5 * sum|w_t - w_{t-1}|``
  between consecutive month-end origins of the same model (target weights,
  ignoring drift; assets entering or leaving the universe count as trades).

Speed: closed forms (GMV, inverse-vol), a primal active-set QP for long-only
GMV and a damped Newton method for risk parity; realized matrices are
computed once per origin and shared across models. See ``score_portfolios``.
"""
from __future__ import annotations

import zlib
from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from ..data.panel import Panel
from ..paths import FORECASTS
from ..universe import ASSET_IDS, HORIZON_MONTHS
from .targets import HORIZON_DAYS, MIN_DAILY_FRAC, MIN_DAYS_PER_MONTH, realized_kday_cov

__all__ = [
    "POLICIES", "PORTFOLIOS", "METRICS",
    "window_end", "regularize", "gmv_weights", "gmv_long_only_weights", "erc_weights",
    "inverse_vol_weights", "equal_weights", "policy_weights", "random_weights",
    "portfolio_weights", "risk_contributions",
    "RealizedWindow", "RealizedCache", "realized_window",
    "score_portfolios", "annual_z_scores", "calibration_summary", "summarize",
    "cov_forecasts_from_long", "load_cov_forecasts",
]

# --------------------------------------------------------------------------
# Fixed policy mixes. ``weights`` are renormalized over the listed assets that
# are present at the origin; the policy is skipped unless every ``required``
# asset is present.
POLICIES: dict[str, dict] = {
    # classic 60/40: US large caps vs intermediate Treasuries
    "p6040": {"weights": {"us_large": 0.60, "ust_int": 0.40},
              "required": ("us_large", "ust_int")},
    # diversified planner mix: 55% equity, 33% bonds/credit, 12% real assets
    "p_balanced": {"weights": {"us_large": 0.30, "us_small": 0.05, "intl_dev": 0.15, "em": 0.05,
                               "ust_int": 0.15, "ust_long": 0.05, "tips": 0.05, "ig_credit": 0.05,
                               "hy_credit": 0.03, "reits": 0.04, "gold": 0.04, "commodities": 0.04},
                   "required": ("us_large", "ust_int")},
    # all-equity global mix (tests within-equity correlations, incl. async closes)
    "p_equity": {"weights": {"us_large": 0.55, "us_small": 0.10, "intl_dev": 0.25, "em": 0.10},
                 "required": ("us_large", "intl_dev")},
}

DETERMINISTIC = ("gmv", "gmv_long", "erc", "invvol", "ew")
OPTIMIZED = ("gmv", "gmv_long", "erc", "invvol", "erc_vt")      # turnover reported for these
VT_NAME = "erc_vt"
PORTFOLIOS = (*DETERMINISTIC, *POLICIES, "random", VT_NAME)       # names when random_detail=False

METRICS = (
    "pred_vol",          # sqrt(w'Hw)
    "real_vol",          # sqrt(w'Sw), S = realized 21-day target matrix (primary)
    "real_var",          # w'Sw (Engle-Colacito: lower is better for gmv/gmv_long)
    "log_ratio",         # log(real_vol / pred_vol), bias statistic (NOT noise-robust: < 0 on average when exact)
    "qlike",             # RV/PV - log(RV/PV) - 1 on variances (robust)
    "sq_log_err",        # log(RV/PV)^2 on variances (NOT noise-robust)
    "real_vol_direct",   # DIAGNOSTIC, convention-dependent: sqrt of uncentered 1-day variance of same-day w'r
    "log_ratio_direct",
    "qlike_direct",
    "gross",             # sum |w| (leverage for erc_vt, short exposure for gmv)
    "turnover",          # 0.5 * sum |w_t - w_{t-1}| (optimized portfolios only)
    "vt_abs_err",        # |real_vol - target| (erc_vt only)
    "var_ratio",         # RV/PV (robust bias statistic: mean 1 when unbiased)
)
_M = {m: i for i, m in enumerate(METRICS)}


# --------------------------------------------------------------- helpers
def window_end(origin) -> pd.Timestamp:
    """End of the 12-month window ``(t, end]``; month-end origins map to month-ends."""
    t = pd.Timestamp(origin)
    if t.is_month_end:
        return t + pd.offsets.MonthEnd(HORIZON_MONTHS)
    return t + pd.DateOffset(months=HORIZON_MONTHS)


def regularize(H, rel_floor: float = 1e-10) -> np.ndarray:
    """Symmetrize and, only if needed, lift eigenvalues to ``rel_floor * max_eig``.

    Leaves any matrix with condition number below ``1/rel_floor`` unchanged
    (up to symmetrization), so GMV still amplifies a forecast's genuine
    estimation error; it only prevents numerical breakdown (non-PD input,
    singular solves, negative predicted variances).
    """
    H = np.asarray(H, dtype=float)
    H = 0.5 * (H + H.T)
    if not np.all(np.isfinite(H)):
        raise ValueError("covariance forecast contains non-finite values")
    ev, V = np.linalg.eigh(H)
    floor = rel_floor * max(ev[-1], 1e-300)
    if ev[0] >= floor:
        return H
    ev = np.maximum(ev, floor)
    return (V * ev) @ V.T


def _solve_ones(H: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.solve(H, np.ones(len(H)))
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(H, np.ones(len(H)), rcond=None)[0]


def gmv_weights(H) -> np.ndarray:
    """Unconstrained global minimum variance: ``H^{-1}1 / 1'H^{-1}1`` (shorts allowed)."""
    H = np.asarray(H, dtype=float)
    x = _solve_ones(H)
    return x / x.sum()


def gmv_long_only_weights(H, tol: float = 1e-12, max_iter: int = 200) -> np.ndarray:
    """Long-only GMV: min w'Hw s.t. sum(w)=1, w>=0. Primal active-set method.

    Warm start: drop assets with negative weight from the unconstrained GMV
    (repeatedly, at most a few rounds) and start from the GMV of the
    remaining set if it is long-only; otherwise start from equal weights.
    Each iteration solves the equality-constrained GMV on the free set
    (closed form), steps toward it until a weight hits zero (then fixes it),
    and at a stationary point releases the fixed asset with the most negative
    KKT multiplier ``mu_i = (Hw)_i - w'Hw``. Finite convergence for PD ``H``;
    falls back to scipy SLSQP if ``max_iter`` is hit (never observed in tests).
    """
    H = np.asarray(H, dtype=float)
    n = len(H)
    w = np.full(n, 1.0 / n)
    free = np.ones(n, dtype=bool)
    cand = np.ones(n, dtype=bool)
    for _ in range(4):                                   # warm start
        F = np.flatnonzero(cand)
        x = _solve_ones(H[np.ix_(F, F)])
        x = x / x.sum()
        if (x >= 0).all():
            w = np.zeros(n)
            w[F] = x
            free = cand
            break
        cand = cand.copy()
        cand[F[x < 0]] = False
        if not cand.any():
            break
    for _ in range(max_iter):
        F = np.flatnonzero(free)
        x = _solve_ones(H[np.ix_(F, F)])
        p = x / x.sum() - w[F]
        if np.max(np.abs(p)) <= 1e-13:
            g = H @ w
            lam = w @ g
            fixed = np.flatnonzero(~free)
            if fixed.size == 0:
                return w
            mu = g[fixed] - lam
            j = np.argmin(mu)
            if mu[j] >= -tol * max(abs(lam), 1e-300):
                return w
            free[fixed[j]] = True
            continue
        neg = p < 0
        alpha, block = 1.0, -1
        if neg.any():
            ratios = -w[F][neg] / p[neg]
            k = np.argmin(ratios)
            if ratios[k] < 1.0:
                alpha, block = max(ratios[k], 0.0), F[neg][k]
        w[F] += alpha * p
        if block >= 0:
            w[block] = 0.0
            free[block] = False
        w[w < 0] = 0.0
        w /= w.sum()
    return _gmv_long_only_scipy(H, w)


def _gmv_long_only_scipy(H: np.ndarray, w0: np.ndarray) -> np.ndarray:  # pragma: no cover
    from scipy.optimize import minimize
    n = len(H)
    res = minimize(lambda w: w @ H @ w, w0, jac=lambda w: 2 * H @ w, method="SLSQP",
                   bounds=[(0, None)] * n,
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1, "jac": lambda w: np.ones(n)}],
                   options={"ftol": 1e-15, "maxiter": 500})
    w = np.clip(res.x, 0, None)
    return w / w.sum()


def erc_weights(H, tol: float = 1e-18, max_iter: int = 100) -> np.ndarray:
    """Equal risk contribution (risk parity), long-only, summing to 1.

    Solves ``min_y 0.5 y'Hy - (1/n) sum(log y)`` (strictly convex; its
    minimizer has ``y_i (Hy)_i = 1/n``) by damped Newton with a positivity-
    preserving backtracking line search, then ``w = y / sum(y)``.
    """
    H = np.asarray(H, dtype=float)
    n = len(H)
    b = np.full(n, 1.0 / n)
    y = 1.0 / np.sqrt(np.diag(H))
    y /= np.sqrt(y @ H @ y)

    def f(v):
        return 0.5 * v @ H @ v - b @ np.log(v)

    fy = f(y)
    for _ in range(max_iter):
        g = H @ y - b / y
        d = -np.linalg.solve(H + np.diag(b / y ** 2), g)
        dec = -g @ d
        if dec < tol:
            break
        s = 1.0
        neg = d < 0
        if neg.any():
            s = min(1.0, 0.99 * np.min(-y[neg] / d[neg]))
        while True:
            yn = y + s * d
            fn = f(yn)
            if fn <= fy - 0.25 * s * dec or s < 1e-12:
                break
            s *= 0.5
        y, fy = yn, fn
    return y / y.sum()


def risk_contributions(w, H) -> np.ndarray:
    """Fractional risk contributions ``w_i (Hw)_i / w'Hw`` (sum to 1)."""
    w = np.asarray(w, dtype=float)
    H = np.asarray(H, dtype=float)
    rc = w * (H @ w)
    return rc / rc.sum()


def inverse_vol_weights(H) -> np.ndarray:
    v = 1.0 / np.sqrt(np.diag(np.asarray(H, dtype=float)))
    return v / v.sum()


def equal_weights(n: int) -> np.ndarray:
    return np.full(n, 1.0 / n)


def policy_weights(assets: list[str], spec: Mapping) -> np.ndarray | None:
    """Policy mix over ``assets`` (renormalized over listed assets present), or None."""
    present = set(assets)
    if not all(a in present for a in spec["required"]):
        return None
    w = np.array([spec["weights"].get(a, 0.0) for a in assets], dtype=float)
    return w / w.sum()


def _random_seed(origin, assets: Iterable[str], seed: int) -> list[int]:
    t = pd.Timestamp(origin)
    key = zlib.crc32(",".join(sorted(assets)).encode())
    return [int(seed), t.year, t.month, t.day, key]


def random_weights(assets: list[str], origin, n_random: int = 20, seed: int = 0) -> np.ndarray:
    """``n_random`` x ``len(assets)`` flat-Dirichlet long-only weights.

    Seeded by (seed, origin, sorted asset set) and drawn in sorted-asset order,
    so every model receives the same portfolios at the same origin regardless
    of the column order of its matrix.
    """
    if n_random <= 0:
        return np.empty((0, len(assets)))
    order = sorted(assets)
    rng = np.random.default_rng(_random_seed(origin, order, seed))
    draws = rng.dirichlet(np.ones(len(order)), size=n_random)
    pos = {a: i for i, a in enumerate(order)}
    return draws[:, [pos[a] for a in assets]]


def _build(H: np.ndarray, assets: list[str], origin, n_random: int, seed: int,
           policies: Mapping) -> tuple[list[str], np.ndarray]:
    """Names and P x N weight matrix. ``H`` must already be regularized."""
    n = len(assets)
    names = list(DETERMINISTIC)
    rows = [gmv_weights(H), gmv_long_only_weights(H), erc_weights(H),
            inverse_vol_weights(H), equal_weights(n)]
    for pname, spec in policies.items():
        w = policy_weights(assets, spec)
        if w is not None:
            names.append(pname)
            rows.append(w)
    R = random_weights(assets, origin, n_random, seed)
    names += [f"random_{k:02d}" for k in range(len(R))]
    W = np.vstack([np.vstack(rows), R]) if len(R) else np.vstack(rows)
    return names, W


def portfolio_weights(cov: pd.DataFrame, origin=None, *, n_random: int = 20, seed: int = 0,
                      policies: Mapping | None = None) -> pd.DataFrame:
    """All test portfolios built from one forecast matrix (rows: portfolio, cols: assets).

    Uses only ``cov`` (and origin/asset set for the random draws). The
    vol-targeted portfolio is ``erc`` scaled by ``target / sqrt(w'Hw)``; it is
    added in ``score_portfolios``.
    """
    assets = list(cov.index)
    H = regularize(cov.reindex(index=assets, columns=assets).values)
    names, W = _build(H, assets, origin if origin is not None else pd.Timestamp(0), n_random, seed,
                      POLICIES if policies is None else policies)
    return pd.DataFrame(W, index=names, columns=assets)


# ------------------------------------------------------------ realized side
@dataclass(frozen=True)
class RealizedWindow:
    """Realized quantities for the window ``(origin, end]``.

    assets        daily test universe (panel column order)
    dropped       panel assets with insufficient daily data in the window
    cov           primary realized covariance: the 21-day target matrix, aligned to ``assets``
    cov_direct    DIAGNOSTIC: X'X summed over the year, from same-day 1-day returns (missing = 0);
                  depends on the sources' pricing conventions
    n_days        panel daily rows in the window
    complete      the daily window is fully observed in the panel
    annual_assets assets with all 12 monthly observations in the window
    annual_ret    their 12-month log excess returns (sum of monthly log excess)
    """

    origin: pd.Timestamp
    end: pd.Timestamp
    assets: tuple
    dropped: tuple
    cov: np.ndarray
    cov_direct: np.ndarray
    n_days: int
    complete: bool
    annual_assets: tuple
    annual_ret: np.ndarray

    @property
    def pos(self) -> dict:
        return {a: i for i, a in enumerate(self.assets)}


class RealizedCache:
    """Computes and memoizes ``RealizedWindow`` per origin (shared across models).

    Reads only ``panel.excess_daily`` / ``panel.excess_monthly``.
    ``horizon_days`` is the target's summing horizon (21); coverage defaults
    match ``assetforecast.eval.targets``.
    """

    def __init__(self, panel: Panel, *, min_coverage: float = MIN_DAILY_FRAC,
                 horizon_days: int = HORIZON_DAYS, min_days_per_month: int = MIN_DAYS_PER_MONTH,
                 min_days: int = 100):
        xd = panel.excess_daily
        xm = panel.excess_monthly
        self.columns = list(xd.columns)
        self.d_index = xd.index
        self.d_values = xd.to_numpy(dtype=float)
        self.m_columns = list(xm.columns)
        self.m_index = xm.index
        self.m_values = xm.to_numpy(dtype=float)
        self.min_coverage, self.horizon_days = min_coverage, int(horizon_days)
        self.min_days_per_month, self.min_days = min_days_per_month, min_days
        self._month_code = (self.d_index.year * 12 + self.d_index.month).to_numpy()
        self._memo: dict[pd.Timestamp, RealizedWindow] = {}

    def get(self, origin) -> RealizedWindow:
        t = pd.Timestamp(origin)
        rw = self._memo.get(t)
        if rw is None:
            rw = self._compute(t)
            self._memo[t] = rw
        return rw

    def _compute(self, t: pd.Timestamp) -> RealizedWindow:
        end = window_end(t)
        idx = self.d_index
        lo, hi = idx.searchsorted(t, side="right"), idx.searchsorted(end, side="right")
        complete = len(idx) > 0 and idx[-1] >= pd.offsets.BDay().rollback(end)
        X = self.d_values[lo:hi]
        n_days = X.shape[0]
        okX = np.isfinite(X)
        nobs = okX.sum(axis=0)
        if complete and n_days >= self.horizon_days:
            _, inv = np.unique(self._month_code[lo:hi], return_inverse=True)
            per_month = np.zeros((inv.max() + 1, X.shape[1]))
            np.add.at(per_month, inv, okX.astype(float))
            keep = (nobs >= max(self.min_coverage * n_days, self.min_days)) \
                & (per_month.min(axis=0) >= self.min_days_per_month)
        else:
            keep = np.zeros(len(self.columns), bool)
        assets = tuple(a for a, k in zip(self.columns, keep) if k)
        dropped = tuple(a for a, k in zip(self.columns, keep) if not k)
        if assets:
            Xk = X[:, keep]
            cov, _ = realized_kday_cov(Xk, self.horizon_days)
            X0 = np.where(np.isfinite(Xk), Xk, 0.0)
            cov_direct = X0.T @ X0  # the window is one year: the sum is already annual
        else:
            cov = cov_direct = np.empty((0, 0))
        # monthly / annual
        mlo = self.m_index.searchsorted(t, side="right")
        mhi = self.m_index.searchsorted(end, side="right")
        M = self.m_values[mlo:mhi]
        if M.shape[0] == HORIZON_MONTHS and (self.m_index[mhi - 1] == end or not t.is_month_end):
            ok = np.isfinite(M).all(axis=0)
            annual_assets = tuple(a for a, o in zip(self.m_columns, ok) if o)
            annual_ret = M[:, ok].sum(axis=0)
        else:
            annual_assets, annual_ret = (), np.empty(0)
        return RealizedWindow(t, end, assets, dropped, cov, cov_direct, n_days, bool(complete),
                              annual_assets, annual_ret)


def realized_window(panel: Panel, origin, **kw) -> RealizedWindow:
    """One-off ``RealizedWindow`` (use ``RealizedCache`` for many origins)."""
    return RealizedCache(panel, **kw).get(origin)


# ----------------------------------------------------------------- scoring
def _qf(W: np.ndarray, S: np.ndarray) -> np.ndarray:
    """Row-wise quadratic forms w'Sw."""
    return np.einsum("pi,ij,pj->p", W, S, W)


def _losses(pv: np.ndarray, rv: np.ndarray):
    with np.errstate(divide="ignore", invalid="ignore"):
        lr = np.log(rv / pv)
        return 0.5 * lr, rv / pv - lr - 1.0, lr ** 2


def _empty_block(P: int) -> np.ndarray:
    return np.full((P, len(METRICS)), np.nan)


def _fill_metrics(out: np.ndarray, W: np.ndarray, H: np.ndarray, S: np.ndarray, Sd: np.ndarray) -> None:
    pv, rv, rvd = _qf(W, H), _qf(W, S), _qf(W, Sd)
    lr, ql, sle = _losses(pv, rv)
    lrd, qld, _ = _losses(pv, rvd)
    out[:, _M["pred_vol"]] = np.sqrt(pv)
    out[:, _M["real_vol"]] = np.sqrt(rv)
    out[:, _M["real_var"]] = rv
    out[:, _M["log_ratio"]] = lr
    out[:, _M["qlike"]] = ql
    out[:, _M["sq_log_err"]] = sle
    out[:, _M["real_vol_direct"]] = np.sqrt(rvd)
    out[:, _M["log_ratio_direct"]] = lrd
    out[:, _M["qlike_direct"]] = qld
    out[:, _M["gross"]] = np.abs(W).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        out[:, _M["var_ratio"]] = rv / pv


def _subset(cov: pd.DataFrame, universe: tuple) -> tuple[list, list, np.ndarray | None]:
    """Test universe (in ``universe`` order), dropped forecast assets, regularized sub-matrix."""
    labels = list(cov.index)
    lpos = {a: i for i, a in enumerate(labels)}
    avail = set(universe)
    U = [a for a in universe if a in lpos]
    dropped = [a for a in labels if a not in avail]
    if len(U) < 2:
        return U, dropped, None
    cv = cov.to_numpy(dtype=float) if list(cov.columns) == labels else \
        cov.reindex(columns=labels).to_numpy(dtype=float)
    fi = [lpos[a] for a in U]
    return U, dropped, regularize(cv[np.ix_(fi, fi)])


def _turnover(prev: tuple | None, assets: list[str], w: np.ndarray, origin: pd.Timestamp) -> float:
    if prev is None:
        return np.nan
    p_origin, p_assets, p_w = prev
    if p_origin + pd.offsets.MonthEnd(1) != origin:
        return np.nan
    a = dict(zip(p_assets, p_w))
    b = dict(zip(assets, w))
    return 0.5 * sum(abs(b.get(k, 0.0) - a.get(k, 0.0)) for k in set(a) | set(b))


def score_portfolios(panel: Panel, forecasts: Mapping[str, Mapping], *,
                     cache: RealizedCache | None = None, n_random: int = 20, seed: int = 0,
                     vol_target: float = 0.10, random_detail: bool = False,
                     policies: Mapping | None = None) -> pd.DataFrame:
    """Score covariance forecasts by the portfolios they build.

    Parameters
    ----------
    panel        full Panel (realized returns are read from it, never weights)
    forecasts    ``{model: {origin: N x N DataFrame}}`` of annualized covariance
                 of 21-day log excess returns (the ``cov_rv`` target) (use ``cov_forecasts_from_long`` /
                 ``load_cov_forecasts`` for the stored parquet format)
    cache        a shared ``RealizedCache`` (pass one when scoring models in
                 several calls; it also sets ``min_coverage`` etc.)
    n_random     number of Dirichlet random long-only portfolios per origin
    vol_target   predicted-vol target for the ``erc_vt`` portfolio
    random_detail  if False (default), random portfolios are reported as one
                 portfolio ``random`` holding the MEAN of each metric over the
                 draws; if True, each ``random_kk`` is reported separately

    Returns
    -------
    Long DataFrame ``origin, model, portfolio, metric, value, n_assets, dropped``
    (only finite values; ``dropped`` lists forecast assets excluded at that
    origin for lack of daily data in the window, comma-joined, "" if none).
    Origins whose window is not fully observed are skipped. Pivot with e.g.
    ``df.query("portfolio=='gmv' and metric=='real_var'").pivot(index='origin', columns='model', values='value')``.
    """
    cache = cache or RealizedCache(panel)
    policies = POLICIES if policies is None else policies
    pnames = list(DETERMINISTIC) + list(policies)
    rnames = [f"random_{k:02d}" for k in range(n_random)] if random_detail else (["random"] if n_random else [])
    all_names = pnames + rnames + [VT_NAME]
    P = len(all_names)
    pidx = {p: i for i, p in enumerate(all_names)}

    blocks, origins_out, models_out, nassets_out, dropped_out = [], [], [], [], []
    for model, by_origin in forecasts.items():
        prev: dict[str, tuple] = {}
        for origin in sorted(by_origin, key=pd.Timestamp):
            t = pd.Timestamp(origin)
            rw = cache.get(t)
            if not rw.complete:
                continue
            U, dropped, H = _subset(by_origin[origin], rw.assets)
            if len(U) < 2:
                continue
            pos = rw.pos
            ix = [pos[a] for a in U]
            S = rw.cov[np.ix_(ix, ix)]
            Sd = rw.cov_direct[np.ix_(ix, ix)]
            names, W = _build(H, U, t, n_random, seed, policies)
            # vol-targeted ERC
            w_erc = W[names.index("erc")]
            w_vt = w_erc * vol_target / np.sqrt(w_erc @ H @ w_erc)
            names = names + [VT_NAME]
            W = np.vstack([W, w_vt])
            vals = np.full((len(names), len(METRICS)), np.nan)
            _fill_metrics(vals, W, H, S, Sd)
            vals[-1, _M["vt_abs_err"]] = abs(vals[-1, _M["real_vol"]] - vol_target)
            for name in OPTIMIZED:
                i = names.index(name)
                vals[i, _M["turnover"]] = _turnover(prev.get(name), U, W[i], t)
                prev[name] = (t, U, W[i])
            out = _empty_block(P)
            for i, name in enumerate(names):
                if name.startswith("random_"):
                    if not random_detail:
                        continue
                out[pidx[name]] = vals[i]
            if not random_detail and n_random:
                rmask = np.array([nm.startswith("random_") for nm in names])
                out[pidx["random"]] = vals[rmask].mean(axis=0) if rmask.any() else np.nan
                out[pidx["random"], _M["turnover"]] = np.nan
            blocks.append(out)
            origins_out.append(t)
            models_out.append(model)
            nassets_out.append(len(U))
            dropped_out.append(",".join(dropped))

    return _to_long(blocks, origins_out, models_out, nassets_out, dropped_out, all_names, METRICS)


def _to_long(blocks, origins, models, nassets, dropped, pnames, metrics) -> pd.DataFrame:
    cols = ["origin", "model", "portfolio", "metric", "value", "n_assets", "dropped"]
    if not blocks:
        return pd.DataFrame(columns=cols)
    V = np.stack(blocks)                                   # R x P x M
    R, P, M = V.shape
    flat = V.reshape(-1)
    ok = np.isfinite(flat)
    r = np.repeat(np.arange(R), P * M)[ok]
    p = np.tile(np.repeat(np.arange(P), M), R)[ok]
    m = np.tile(np.arange(M), R * P)[ok]
    origins = pd.DatetimeIndex(origins)
    return pd.DataFrame({
        "origin": origins[r],
        "model": pd.Categorical.from_codes(pd.Index(models).factorize()[0][r],
                                           categories=pd.Index(models).factorize()[1]),
        "portfolio": pd.Categorical.from_codes(p, categories=list(pnames)),
        "metric": pd.Categorical.from_codes(m, categories=list(metrics)),
        "value": flat[ok],
        "n_assets": np.asarray(nassets, dtype=np.int16)[r],
        "dropped": pd.Categorical(np.asarray(dropped, dtype=object)[r]),
    })[cols]


# ------------------------------------------------------ annual calibration
def annual_z_scores(panel: Panel, cov_annual: Mapping[str, Mapping], *,
                    cache: RealizedCache | None = None, n_random: int = 20, seed: int = 0,
                    policies: Mapping | None = None) -> pd.DataFrame:
    """Standardized 12-month portfolio returns ``z = w'R / sqrt(w' A w)``.

    ``cov_annual`` is ``{model: {origin: DataFrame}}``: forecast covariance of
    the 12-MONTH log excess return (from the bridge). Weights are built from
    ``A`` itself (same portfolio set as ``score_portfolios`` minus the
    vol-targeted one, which has the same z as ``erc``); ``R`` is the vector of
    realized 12-month log excess returns (sum of monthly log excess returns,
    so monthly-only assets participate). Universe: forecast assets with all
    12 monthly observations in the window. Each random portfolio is kept
    separately (``random_kk``) because z is not additive across portfolios.

    Returns long ``origin, model, portfolio, metric, value, n_assets, dropped``
    with metrics ``z``, ``ret`` (w'R) and ``pred_vol`` (sqrt(w'Aw)). z is NOT
    demeaned here; see ``calibration_summary``.
    """
    cache = cache or RealizedCache(panel)
    policies = POLICIES if policies is None else policies
    names_all = list(DETERMINISTIC) + list(policies) + [f"random_{k:02d}" for k in range(n_random)]
    pidx = {p: i for i, p in enumerate(names_all)}
    metrics = ("z", "ret", "pred_vol")
    blocks, origins_out, models_out, nassets_out, dropped_out = [], [], [], [], []
    for model, by_origin in cov_annual.items():
        for origin in sorted(by_origin, key=pd.Timestamp):
            t = pd.Timestamp(origin)
            rw = cache.get(t)
            if not rw.annual_assets:
                continue
            avail = {a: i for i, a in enumerate(rw.annual_assets)}
            U, dropped, A = _subset(by_origin[origin], rw.annual_assets)
            if len(U) < 2:
                continue
            R = rw.annual_ret[[avail[a] for a in U]]
            names, W = _build(A, U, t, n_random, seed, policies)
            ret = W @ R
            pv = np.sqrt(_qf(W, A))
            out = np.full((len(names_all), 3), np.nan)
            for i, nm in enumerate(names):
                out[pidx[nm]] = (ret[i] / pv[i], ret[i], pv[i])
            blocks.append(out)
            origins_out.append(t)
            models_out.append(model)
            nassets_out.append(len(U))
            dropped_out.append(",".join(dropped))
    return _to_long(blocks, origins_out, models_out, nassets_out, dropped_out, names_all, metrics)


def calibration_summary(z: pd.DataFrame, *, demean: str = "sample", horizon: int = HORIZON_MONTHS,
                        block: int = 24, n_boot: int = 2000, level: float = 0.90,
                        seed: int = 0, pool_random: bool = True) -> pd.DataFrame:
    """Is Var(z) = 1? Per (model, portfolio), from ``annual_z_scores`` output.

    Demeaning (``demean``):
      ``"sample"`` (default) subtract each series' full-sample mean of z (a
          covariance forecast says nothing about the expected return; the
          equity premium would otherwise inflate E[z^2] by Sharpe^2, ~0.15 for
          60/40). The lost degree of freedom is corrected: on the overlapping
          monthly series the mean has variance ~ ``horizon/n`` (overlap of
          12-month sums), so ``var = mean((z-zbar)^2) / (1 - horizon/n)``; on
          non-overlapping annual series the usual ``n-1`` divisor.
      ``"none"`` use raw ``mean(z^2)`` (tests mean and variance jointly).

    Columns:
      n              origins (overlapping, monthly)
      mean_z         mean of z
      var_z          variance of z from all overlapping origins (point estimate)
      ci_lo, ci_hi   ``level`` moving-block bootstrap percentile interval for
                     var_z (block = ``block`` months >= 12 to respect overlap;
                     the demeaning mean is held fixed in the resamples)
      var_z_nonoverlap  average over the 12 calendar-month offsets of the
                     variance computed from that offset's non-overlapping
                     annual series
      n_nonoverlap   average number of non-overlapping years per offset
      chi2_lo, chi2_hi  normal-theory chi-square interval for var_z_nonoverlap
                     using the df of ONE offset (conservative: averaging over
                     offsets is at least as precise as a single offset)
      share_gt1, share_gt2  share of |z - mean| (or |z|) beyond 1 and 2
                     (normal: 0.317 and 0.046)
    ``pool_random`` adds portfolio ``random_pooled`` per model: all random
    portfolios' z pooled (per-origin average of squared deviations).
    """
    from scipy.stats import chi2

    d = z[z["metric"] == "z"]
    if d.empty:
        return pd.DataFrame()
    d = d.assign(model=d["model"].astype(str), portfolio=d["portfolio"].astype(str))
    groups: list[tuple[str, str, pd.DataFrame]] = []
    for (model, port), g in d.groupby(["model", "portfolio"], sort=True):
        groups.append((model, port, g.pivot(index="origin", columns="portfolio", values="value")))
    if pool_random:
        for model, g in d[d["portfolio"].str.startswith("random_")].groupby("model", sort=True):
            groups.append((model, "random_pooled", g.pivot(index="origin", columns="portfolio", values="value")))

    rng = np.random.default_rng(seed)
    alpha = 1 - level
    rows = []
    for model, port, Z in groups:
        Z = Z.sort_index()
        n = len(Z)
        if n < 2:
            continue
        zbar = Z.mean(axis=0) if demean == "sample" else pd.Series(0.0, index=Z.columns)
        if demean not in ("sample", "none"):
            raise ValueError("demean must be 'sample' or 'none'")
        dev = Z - zbar
        q = (dev ** 2).mean(axis=1, skipna=True).to_numpy()     # per-origin, averaged over pooled portfolios
        corr = (1 - horizon / n) if (demean == "sample" and n > horizon) else 1.0
        var_z = q.mean() / corr
        # moving-block bootstrap
        b = min(block, n)
        nb = int(np.ceil(n / b))
        starts = rng.integers(0, n - b + 1, size=(n_boot, nb))
        idx = (starts[:, :, None] + np.arange(b)).reshape(n_boot, -1)[:, :n]
        boot = q[idx].mean(axis=1) / corr
        ci_lo, ci_hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
        # non-overlapping: by calendar-month offset
        months = Z.index.month
        v_off, n_off = [], []
        for mth in np.unique(months):
            Zk = Z[months == mth]
            nk = len(Zk)
            if nk < 2:
                continue
            if demean == "sample":
                vk = ((Zk - Zk.mean(axis=0)) ** 2).sum(axis=0) / (nk - 1)
            else:
                vk = (Zk ** 2).sum(axis=0) / nk
            v_off.append(float(vk.mean()))
            n_off.append(nk)
        if v_off:
            v_no = float(np.mean(v_off))
            n_no = float(np.mean(n_off))
            df = max(int(min(n_off)) - (1 if demean == "sample" else 0), 1)
            chi_lo = df * v_no / chi2.ppf(1 - alpha / 2, df)
            chi_hi = df * v_no / chi2.ppf(alpha / 2, df)
        else:
            v_no = n_no = chi_lo = chi_hi = np.nan
        a = np.abs(dev.to_numpy()).ravel()
        a = a[np.isfinite(a)]
        rows.append({
            "model": model, "portfolio": port, "n": n,
            "mean_z": float(np.nanmean(Z.to_numpy())), "var_z": var_z,
            "ci_lo": ci_lo, "ci_hi": ci_hi,
            "var_z_nonoverlap": v_no, "n_nonoverlap": n_no, "chi2_lo": chi_lo, "chi2_hi": chi_hi,
            "share_gt1": float((a > 1).mean()), "share_gt2": float((a > 2).mean()),
            "demean": demean, "level": level,
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------- summary
def _eras(origins: pd.DatetimeIndex, eras) -> dict[str, np.ndarray]:
    out = {"all": np.ones(len(origins), bool)}
    if eras is None:
        return out
    if eras == "decade":
        dec = (origins.year // 10) * 10
        for d in sorted(set(dec)):
            out[f"{d}s"] = dec == d
        return out
    for name, (start, end) in eras.items():
        lo = pd.Timestamp(start) if start is not None else pd.Timestamp.min
        hi = pd.Timestamp(end) if end is not None else pd.Timestamp.max
        out[name] = (origins >= lo) & (origins <= hi)
    return out


def summarize(results: pd.DataFrame, *, eras=None, common_origins: bool = True) -> pd.DataFrame:
    """Mean of each metric by model and portfolio (and era) over common origins.

    ``results`` is ``score_portfolios`` output. For each (portfolio, metric),
    origins are restricted to those where EVERY model in ``results`` has a
    value (``common_origins=True``) so models are compared on the same
    windows. ``eras``: None (full sample only), ``"decade"``, or
    ``{name: (start, end)}`` on origin dates (inclusive; None = open).
    Returns ``model, portfolio, metric, era, mean, n_origins``.
    """
    rows = []
    r = results.assign(model=results["model"].astype(str),
                       portfolio=results["portfolio"].astype(str),
                       metric=results["metric"].astype(str))
    for (port, met), g in r.groupby(["portfolio", "metric"], sort=False):
        wide = g.pivot_table(index="origin", columns="model", values="value", aggfunc="first")
        if common_origins:
            wide = wide.dropna(axis=0, how="any")
        if wide.empty:
            continue
        for era, mask in _eras(pd.DatetimeIndex(wide.index), eras).items():
            sub = wide[mask]
            if sub.empty:
                continue
            means = sub.mean(axis=0)
            counts = sub.notna().sum(axis=0)
            for model in wide.columns:
                rows.append((model, port, met, era, float(means[model]), int(counts[model])))
    return pd.DataFrame(rows, columns=["model", "portfolio", "metric", "era", "mean", "n_origins"])


# ------------------------------------------------------------------- I/O
def cov_forecasts_from_long(df: pd.DataFrame) -> dict[pd.Timestamp, pd.DataFrame]:
    """``origin, asset_i, asset_j, cov`` (long) -> ``{origin: N x N DataFrame}``.

    Accepts either both triangles or one; missing mirror entries are filled
    by symmetry. Asset order follows ``ASSET_IDS``.
    """
    out = {}
    rank = {a: i for i, a in enumerate(ASSET_IDS)}
    for origin, g in df.groupby("origin", sort=True):
        m = g.pivot_table(index="asset_i", columns="asset_j", values="cov", aggfunc="first")
        assets = sorted(set(m.index) | set(m.columns), key=lambda a: (rank.get(a, 1e9), a))
        m = m.reindex(index=assets, columns=assets)
        m = m.fillna(m.T)
        out[pd.Timestamp(origin)] = m
    return out


def load_cov_forecasts(models: Iterable[str], root=None) -> dict[str, dict]:
    """Read ``results/forecasts/cov/<model>.parquet`` for each model."""
    root = FORECASTS / "cov" if root is None else root
    return {m: cov_forecasts_from_long(pd.read_parquet(root / f"{m}.parquet")) for m in models}
