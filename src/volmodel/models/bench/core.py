"""Monthly building blocks shared by the bench models.

Everything the bench family does is a weighted average, over calendar
months, of a per-month estimate of the month's variance (or co-moment) on
the TARGET's scale: the variance of 21-trading-day log excess returns,
accumulated over one calendar month. Three per-month measurements exist:

``r2``  squared calendar-month log excess return ``r_m^2``. Always available
        (monthly history is the longest), calendar-free, one draw per month.

``xd``  from DAILY data, the same estimator as the target
        (``volmodel.eval.targets.realized_kday_cov``) restricted to the sums
        ending in month ``m``::

            xd_m = D_m * mean_{d in m} s_d^2 / 21

        ``s_d`` = sum of the 21 daily log excess returns ending on day ``d``
        (a missing day inside the asset's daily span counts 0, as in the
        target), ``D_m`` = panel trading days in month ``m``. Summed over 12
        months this is ``n_rows * (per-day variance)`` - the target's own
        annualization, which is correct under 252- and 295-day calendars
        alike. Used only if at least ``MIN_SUMS`` (10) valid sums end in the
        month; a sum is valid only if all its 21 days are on or after the
        asset's first daily return. Because it uses 21-day sums it inherits
        the target's robustness to 1-day pricing conventions (smoothed NAVs,
        ETF overshoot) and needs no splice handling.

``x1``  from 1-day returns: ``x1_m = sum_{d in m} r_d^2`` (rescaled by
        ``D_m / span days`` in a partial first month). More precise than
        ``xd`` but on the 1-DAY scale, which differs from the 21-day scale by
        0.69-1.20 depending on the source; only usable after multiplying by a
        trailing 21-day/1-day variance ratio (see ``vol.DailyRatioVol``).

``x`` = ``xd`` where available, else ``r2``: the "daily where it exists,
monthly otherwise" series used by the ``_d`` models.

All three are in units of one month's variance, so 12 x a weighted mean is
an annual variance comparable with the target.

Exponential weights are by MONTH age: month ``m`` gets ``0.5 ** (age/hl)``
with ``age = 0`` for the origin month; missing months are dropped and the
weights renormalized (so a short history gets a near-equal-weight mean, which
is the natural fallback). ``hl=None`` means equal weights (expanding mean).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

K_DAYS = 21          # the target's horizon in trading days
MIN_SUMS = 10        # valid 21-day sums (or daily returns) needed in a month
MIN_PAIR_MONTHS = 24 # months a correlation needs before it is trusted (else 0)


class Moments:
    """Per-month variance measurements for ``assets`` from one view.

    Attributes (M months x N assets unless noted): ``months``, ``r`` (monthly
    log excess return), ``r2``, ``xd``, ``x1``, ``x``, ``D`` (M trading days),
    and, if ``pairs=True``, ``C`` / ``V`` / ``ok`` (M x N x N): per-month
    co-moment, pair-specific variance of asset i, and validity, measured from
    21-day sums when both assets have them in that month and from monthly
    returns otherwise (or monthly only when ``daily=False``).
    """

    def __init__(self, view, assets: list[str], *, pairs: bool = False, daily: bool = True,
                 one_day: bool = False):
        self.assets = list(assets)
        mon = view.excess_monthly.reindex(columns=self.assets)
        self.months = mon.index
        self.r = mon.to_numpy(float)
        self.r2 = self.r ** 2
        M, N = self.r.shape
        self.M, self.N = M, N
        self.D = np.zeros(M)
        self.xd = np.full((M, N), np.nan)
        self.x1 = np.full((M, N), np.nan)
        self._S = None
        if daily and len(view.daily):
            self._daily(view, one_day)
        self.x = np.where(np.isfinite(self.xd), self.xd, self.r2)
        if pairs:
            self._pairs(daily)

    # ------------------------------------------------------------- daily
    def _daily(self, view, one_day: bool):
        E = view.excess_daily.reindex(columns=self.assets).to_numpy(float)
        T, N = E.shape
        dm = view.daily.index.values.astype("datetime64[M]")
        mm = self.months.values.astype("datetime64[M]")
        pos = np.searchsorted(mm, dm)
        if not (pos < len(mm)).all() or not (mm[np.minimum(pos, len(mm) - 1)] == dm).all():
            raise ValueError("daily dates fall outside the monthly index")
        M = self.M
        self.D = np.bincount(pos, minlength=M).astype(float)
        fin = np.isfinite(E)
        first = np.where(fin.any(0), fin.argmax(0), T)            # first daily return per asset
        X0 = np.where(fin, E, 0.0)
        cs = np.vstack([np.zeros((1, N)), np.cumsum(X0, axis=0)])
        S = np.zeros((T, N))
        if T >= K_DAYS:
            S[K_DAYS - 1:] = cs[K_DAYS:] - cs[:-K_DAYS]
        d = np.arange(T)[:, None]
        V = (d - (K_DAYS - 1)) >= first[None, :]                  # all 21 days inside the daily span
        S = np.where(V, S, 0.0)
        nS = np.stack([np.bincount(pos, weights=V[:, j], minlength=M) for j in range(N)], 1)
        sS = np.stack([np.bincount(pos, weights=S[:, j] ** 2, minlength=M) for j in range(N)], 1)
        with np.errstate(invalid="ignore", divide="ignore"):
            self.xd = np.where(nS >= MIN_SUMS, self.D[:, None] * sS / np.maximum(nS, 1) / K_DAYS, np.nan)
        self._S, self._V, self._pos = S, V.astype(float), pos
        if one_day:
            span = d >= first[None, :]
            nobs = np.stack([np.bincount(pos, weights=fin[:, j], minlength=M) for j in range(N)], 1)
            nspan = np.stack([np.bincount(pos, weights=span[:, j], minlength=M) for j in range(N)], 1)
            s1 = np.stack([np.bincount(pos, weights=X0[:, j] ** 2, minlength=M) for j in range(N)], 1)
            with np.errstate(invalid="ignore", divide="ignore"):
                self.x1 = np.where(nobs >= MIN_SUMS, s1 * self.D[:, None] / np.maximum(nspan, 1), np.nan)

    # ------------------------------------------------------------- pairs
    def _pairs(self, daily: bool):
        M, N = self.M, self.N
        r = self.r
        okm = np.isfinite(r)
        r0 = np.where(okm, r, 0.0)
        C = r0[:, :, None] * r0[:, None, :]
        V = np.broadcast_to((r0 ** 2)[:, :, None], (M, N, N)).copy()
        ok = okm[:, :, None] & okm[:, None, :]
        if daily and self._S is not None:
            S, Vd, pos = self._S, self._V, self._pos
            starts = np.searchsorted(pos, np.arange(M + 1))
            for m in np.nonzero(np.diff(starts))[0]:
                a, b = starts[m], starts[m + 1]
                Sm, Vm = S[a:b], Vd[a:b]
                K = Vm.T @ Vm
                dmode = K >= MIN_SUMS
                if not dmode.any():
                    continue
                f = self.D[m] / K_DAYS / np.maximum(K, 1)
                C[m] = np.where(dmode, (Sm.T @ Sm) * f, C[m])
                V[m] = np.where(dmode, ((Sm ** 2).T @ Vm) * f, V[m])
                ok[m] = ok[m] | dmode
        self.C, self.V, self.ok = C, V, ok


# ------------------------------------------------------------------ weights
def month_weights(M: int, hl: float | None) -> np.ndarray:
    """Weight of each of M months (oldest first) at the last month."""
    if hl is None:
        return np.ones(M)
    age = np.arange(M - 1, -1, -1, dtype=float)
    return 0.5 ** (age / hl)


def ew_mean(x: np.ndarray, hl: float | None, mask: np.ndarray | None = None) -> np.ndarray:
    """Weighted mean over months (axis 0) of the finite entries of ``x``."""
    ok = np.isfinite(x) if mask is None else (mask & np.isfinite(x))
    w = month_weights(x.shape[0], hl).reshape((-1,) + (1,) * (x.ndim - 1))
    num = (w * np.where(ok, x, 0.0)).sum(0)
    den = (w * ok).sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return num / den


def ew_series(x: np.ndarray, hl: float | None) -> np.ndarray:
    """``ew_mean`` evaluated at EVERY month (row s uses months <= s only)."""
    ok = np.isfinite(x)
    x0 = np.where(ok, x, 0.0)
    if hl is None:
        num, den = np.cumsum(x0, 0), np.cumsum(ok, 0).astype(float)
    else:
        from scipy.signal import lfilter
        phi = 0.5 ** (1.0 / hl)
        num = lfilter([1.0], [1.0, -phi], x0, axis=0)
        den = lfilter([1.0], [1.0, -phi], ok.astype(float), axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def ew_corr(mo: Moments, hl: float | None) -> tuple[np.ndarray, np.ndarray]:
    """Pairwise uncentered EW correlation and the effective number of months
    behind each pair (``(sum w)^2 / sum w^2`` over the pair's valid months).
    Pairs with fewer than ``MIN_PAIR_MONTHS`` valid months get 0."""
    w = month_weights(mo.M, hl)[:, None, None]
    okw = w * mo.ok
    num = (okw * mo.C).sum(0)
    v = (okw * mo.V).sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        c = num / np.sqrt(v * v.T)
        neff = okw.sum(0) ** 2 / (okw * w).sum(0)
    n = mo.ok.sum(0)
    c = np.where((n >= MIN_PAIR_MONTHS) & np.isfinite(c), np.clip(c, -1.0, 1.0), 0.0)
    np.fill_diagonal(c, 1.0)
    return c, np.where(np.isfinite(neff), neff, 0.0)


def nearest_corr(c: np.ndarray, floor: float = 1e-3) -> np.ndarray:
    """Symmetrize, clip eigenvalues at ``floor``, rescale to unit diagonal.
    (Same repair as the reference models, so differences are in estimation.)"""
    c = (c + c.T) / 2.0
    w, v = np.linalg.eigh(c)
    c = (v * np.maximum(w, floor)) @ v.T
    d = np.sqrt(np.diag(c))
    c = c / np.outer(d, d)
    np.fill_diagonal(c, 1.0)
    return (c + c.T) / 2.0


def segment_start(view, assets: list[str]) -> pd.Series:
    """First month of each asset's current data-tier segment (the months since
    its tier letter last changed). The tier is the only splice information a
    view carries; it misses same-tier splices (e.g. VEIEX -> EEM)."""
    t = view.tiers.reindex(columns=assets)
    out = {}
    for a in assets:
        s = t[a].dropna()
        if s.empty:
            out[a] = pd.NaT
            continue
        ch = s.ne(s.shift())
        out[a] = s.index[ch.to_numpy().nonzero()[0][-1]]
    return pd.Series(out)
