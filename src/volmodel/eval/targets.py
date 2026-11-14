"""Realized 12-month targets, computed once from the FULL panel and cached.

The target (ARCHITECTURE.md, "What is being forecast") is the annualized
covariance of OVERLAPPING ``k``-TRADING-DAY LOG EXCESS returns realized in the
window ``(t, t + 12 calendar months]``, ``k = horizon_days`` (default 21):

    Sigma = (252 / k) * mean over the window of  s s'

where ``s`` is the vector of ``k``-day log excess return sums ending on each
trading day of the window whose ``k`` days all lie inside the window
(``M = n_rows - k + 1`` sums, ``n_rows`` = panel daily rows in the window),
UNCENTERED. Variances and correlations both come from this one matrix, which
is positive semi-definite by construction (a Gram matrix). The trading-day
clock is the panel's daily calendar (the US equity trading calendar).

Why k days and not one: measured 1-day volatility depends on how a source is
priced (smoothed NAVs understate it, ETFs of foreign/illiquid assets overstate
it, and the conventions meet at splice dates). At 21 days the sources agree,
and the monthly-only eras measure the same quantity.

For every month-end origin ``t`` whose window is fully observed in the panel
(see :func:`observed_through`):

vol      per-asset ``sqrt(Sigma_ii)``. If the asset fails the daily coverage
         rule, the monthly proxy ``var = 12 * mean(r_m^2)`` over the 12
         calendar-month log excess returns is used instead (the same estimand
         measured less precisely: a calendar month is ~21 trading days) and
         the row is flagged ``proxy=True``. Neither available: no row.
corr     ``Sigma_ij / sqrt(Sigma_ii Sigma_jj)`` for every pair where BOTH
         assets are daily-based in the window; no row otherwise (= NaN). No
         correlation is ever derived from monthly proxies.
cov      ``Sigma`` on the daily-based assets (``Targets.cov_matrix``); equal
         to ``D C D`` with the vol and corr above.
fwd_ret  forward 12-month log excess return per asset = sum of the 12 monthly
         log excess returns in the window (no row unless all 12 are present).

Missing daily values (deliberate choice, documented)
----------------------------------------------------
In the panel a NaN on a calendar day means the asset did not trade that day
and its next return spans the gap (off-calendar returns are compounded into
the next calendar day). So in the ``k``-day sums a missing day contributes 0:
the sum ending on any day still contains everything that accrued over those
``k`` days, and a holiday does not invalidate the ``k`` sums that cover it.
Only at the two ends of a gap that straddles the window boundary can a return
be attributed to a neighbouring window. No rescaling is applied for missing
days (rescaling would double-count gap-spanning returns).

An asset still needs adequate coverage to get a daily-based target (rules
kept from the 1-day definition; module constants recorded in the metadata):

* ``trading days`` (``n_days``) = rows of the daily index inside the window on
  which at least one asset has a return.
* An asset's target is DAILY-based iff it has returns on at least
  ``MIN_DAILY_FRAC`` (90%) of the window's trading days AND at least
  ``MIN_DAYS_PER_MONTH`` (10) daily returns in each of the 12 calendar months,
  AND the window has at least ``k`` rows. The per-month condition stops an
  asset whose daily history starts or stops mid-window from passing (its
  zeros would then be real missing data, not holidays); such windows use the
  proxy. Up to ~10% scattered missing days are therefore treated as
  non-trading days; if they were in fact LOST returns, the variance would be
  biased low by at most that share.
* Pairwise terms use exactly the same ``M`` sums for both assets, and a pair
  exists only if both assets are daily-based, i.e. both are in their daily
  era throughout the window. (The old ``MIN_PAIR_FRAC`` joint-coverage rule
  is no longer needed: a holiday of one asset no longer removes sums.)
* The monthly proxy requires all 12 monthly log excess returns.

Statistical properties (verified by simulation): for i.i.d. normal returns
the standard error of a realized 12-month vol is ~17% (about 17 effective
observations; vs ~4.4% for the old 1-day definition and ~20% for the monthly
proxy), ~18-19% with persistent volatility and fat tails. The variance is
unbiased; the vol is ~1.5% low by Jensen. Uncentered sums include the squared
mean: a relative upward bias of ``k * SR^2 / 252`` in variance (0.02 for a
Sharpe ratio of 0.5, i.e. 1% in vol), identical to the monthly proxy's
``SR^2 / 12``, so the two eras stay consistent. Centering on the window mean
would instead bias the variance down by ~9% (the overlapping sums share the
window's drift), which is why the target is uncentered.

Everything is vectorized with prefix sums over the daily and monthly arrays;
there is no per-origin pandas slicing.

Public API::

    observed_through(panel) -> Timestamp
    compute_targets(panel, horizon_days=21) -> Targets
    get_targets(panel, results_dir=None, rebuild=False, horizon_days=21) -> Targets   # cached
    load_targets(results_dir=None, horizon_days=21) -> Targets
    realized_kday_cov(X, k) -> (Sigma, n_sums)      # one window, for other modules
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..data.panel import Panel
from ..paths import RESULTS
from ..universe import HORIZON_MONTHS, TRADING_DAYS

TARGETS_VERSION = 2          # 1 = 1-day variances + 3-day correlations; 2 = k-day sums
HORIZON_DAYS = 21            # default k: trading days per summed return
MIN_DAILY_FRAC = 0.90
MIN_DAYS_PER_MONTH = 10
DEFINITION = ("annualized uncentered covariance of overlapping {k}-trading-day log excess return sums "
              "in (t, t+12 months]; missing days contribute 0; monthly proxy 12*mean(r_m^2) for assets "
              "failing daily coverage (no correlations)")
_TIER_RANK = {"A": 0, "B": 1, "C": 2, "D": 3}
_TIER_LETTER = np.array(["A", "B", "C", "D"], dtype=object)


def _ts_ns(x) -> pd.Series | pd.Timestamp:
    """Normalize timestamps (scalar or array-like) to datetime64[ns]."""
    if isinstance(x, (pd.Timestamp, str, np.datetime64)):
        return pd.Timestamp(x).as_unit("ns")
    return pd.to_datetime(pd.Series(x)).astype("datetime64[ns]")


def window_end(t) -> pd.Timestamp:
    """End of the target window for month-end origin ``t``: t + 12 month-ends."""
    return pd.Timestamp(t) + pd.offsets.MonthEnd(HORIZON_MONTHS)


def observed_through(panel: Panel) -> pd.Timestamp:
    """Last month-end whose month is completely observed in ``panel``.

    The last monthly row with any data, stepped back one month if the daily
    data stop more than 4 calendar days before that month's last business day
    (i.e. the final month is partial). Windows ending on or before this date
    are fully observed; this is also the latest (live) forecast origin.
    """
    m = panel.monthly
    has = m.notna().any(axis=1)
    if not has.any():
        raise ValueError("panel has no monthly data")
    last_m = pd.Timestamp(m.index[has.to_numpy()].max())
    d = panel.daily
    if len(d):
        dh = d.notna().any(axis=1)
        if dh.any():
            last_d = pd.Timestamp(d.index[dh.to_numpy()].max())
            month_start = last_m - pd.offsets.MonthBegin(1)
            last_bday = last_m if last_m.dayofweek < 5 else last_m - pd.offsets.BDay(1)
            if month_start <= last_d < last_bday - pd.Timedelta(days=4):
                last_m = last_m - pd.offsets.MonthEnd(1)
    return last_m.normalize()


def kday_sums(X0: np.ndarray, k: int) -> np.ndarray:
    """Overlapping ``k``-row sums of a T x N array (no NaN): row m = rows m..m+k-1."""
    if X0.shape[0] < k:
        return np.empty((0,) + X0.shape[1:])
    return np.lib.stride_tricks.sliding_window_view(X0, k, axis=0).sum(axis=-1)


def realized_kday_cov(X: np.ndarray, k: int = HORIZON_DAYS) -> tuple[np.ndarray, int]:
    """Realized target matrix for ONE window of daily log excess returns.

    ``X`` is the n_rows x N array of the window's daily rows (NaN = did not
    trade, contributes 0). Returns ``((252/k) * mean_m s_m s_m', M)`` with
    ``M = n_rows - k + 1`` overlapping sums; an empty-window matrix is NaN.
    Coverage rules are the caller's business. Shared by ``portfolio``.
    """
    X0 = np.where(np.isfinite(X), X, 0.0)
    Y = kday_sums(X0, k)
    M = Y.shape[0]
    if M == 0:
        return np.full((X.shape[1], X.shape[1]), np.nan), 0
    return Y.T @ Y * (TRADING_DAYS / k) / M, M


# ---------------------------------------------------------------- container
@dataclass
class Targets:
    """Realized targets in long format (one row per origin x asset / pair).

    vol      origin, asset, vol, var, proxy (bool), n_obs (daily obs, or 12 for
             the proxy), n_days (window trading days), tier (worst data tier
             in the window, or NaN)
    corr     origin, asset_i, asset_j, corr, n_obs (number of overlapping
             k-day sums); i < j in panel column order; only valid pairs
    fwd_ret  origin, asset, ret
    meta     version, horizon_days, rules, observed_through, fingerprint, asset order
    """

    vol: pd.DataFrame
    corr: pd.DataFrame
    fwd_ret: pd.DataFrame
    meta: dict = field(default_factory=dict)

    # ----------------------------------------------------------- accessors
    @property
    def assets(self) -> list[str]:
        return list(self.meta.get("assets", sorted(self.vol["asset"].unique())))

    @property
    def horizon_days(self) -> int | None:
        return self.meta.get("horizon_days")

    @property
    def origins(self) -> pd.DatetimeIndex:
        """Origins with at least one vol target."""
        return pd.DatetimeIndex(sorted(self.vol["origin"].unique()))

    def vol_wide(self, column: str = "vol") -> pd.DataFrame:
        """origin x asset frame of ``column`` (vol, var, proxy, n_obs, tier)."""
        w = self.vol.pivot(index="origin", columns="asset", values=column)
        return w.reindex(columns=[a for a in self.assets if a in w.columns])

    def vol_scale(self) -> pd.Series:
        """Per-asset FIXED evaluation scale: mean realized vol over all origins.

        Used to put squared vol errors of assets with very different vol
        levels on a common footing. A fixed per-asset constant keeps the
        scaled MSE a consistent (Patton-robust) loss under noisy targets.
        """
        return self.vol.groupby("asset")["vol"].mean().reindex(self.assets).dropna()

    def _dense(self):
        """Cached dense arrays: origins, V (O x N vol), PX (O x N proxy), C (O x N x N corr)."""
        cache = self.__dict__.get("_dense_cache")
        if cache is None:
            assets = self.assets
            pos = {a: k for k, a in enumerate(assets)}
            origins = self.origins
            O, N = len(origins), len(assets)
            V = np.full((O, N), np.nan)
            PX = np.zeros((O, N), dtype=bool)
            oi = origins.get_indexer(self.vol["origin"])
            ai = self.vol["asset"].map(pos).to_numpy(int)
            V[oi, ai] = self.vol["vol"].to_numpy(float)
            PX[oi, ai] = self.vol["proxy"].to_numpy(bool)
            C = np.full((O, N, N), np.nan)
            co = origins.get_indexer(self.corr["origin"])
            ok = co >= 0
            ci = self.corr["asset_i"].map(pos).to_numpy()[ok].astype(int)
            cj = self.corr["asset_j"].map(pos).to_numpy()[ok].astype(int)
            r = self.corr["corr"].to_numpy(float)[ok]
            C[co[ok], ci, cj] = r
            C[co[ok], cj, ci] = r
            d = np.arange(N)
            C[:, d, d] = 1.0
            cache = (origins, pos, V, PX, C)
            self.__dict__["_dense_cache"] = cache
        return cache

    def _row(self, origin) -> int:
        origins = self._dense()[0]
        k = origins.get_indexer([_ts_ns(origin)])[0]
        return int(k)

    def vol_at(self, origin) -> pd.DataFrame:
        """Vol targets at one origin, indexed by asset: columns vol, proxy (empty if none)."""
        origins, pos, V, PX, _ = self._dense()
        k = self._row(origin)
        if k < 0:
            return pd.DataFrame({"vol": pd.Series([], dtype=float), "proxy": pd.Series([], dtype=bool)})
        ok = np.isfinite(V[k])
        names = [a for a, m in zip(self.assets, ok) if m]
        return pd.DataFrame({"vol": V[k][ok], "proxy": PX[k][ok]}, index=names)

    def daily_assets(self, origin) -> list[str]:
        """Assets with a DAILY-based vol target at ``origin`` (panel order)."""
        _, _, V, PX, _ = self._dense()
        k = self._row(origin)
        if k < 0:
            return []
        return [a for a, v, p in zip(self.assets, V[k], PX[k]) if np.isfinite(v) and not p]

    def _idx(self, assets):
        pos = self._dense()[1]
        missing = [a for a in assets if a not in pos]
        if missing:
            raise KeyError(f"unknown assets {missing}")
        return np.array([pos[a] for a in assets], dtype=int)

    def corr_matrix(self, origin, assets: list[str] | None = None) -> pd.DataFrame:
        """Realized correlation matrix (unit diagonal, NaN where no valid pair)."""
        assets = self.daily_assets(origin) if assets is None else list(assets)
        return pd.DataFrame(self.corr_array(origin, assets), index=assets, columns=assets)

    def corr_array(self, origin, assets: list[str]) -> np.ndarray:
        C = self._dense()[4]
        k, ix = self._row(origin), self._idx(assets)
        if k < 0:
            out = np.full((len(ix), len(ix)), np.nan)
            np.fill_diagonal(out, 1.0)
            return out
        return C[k][np.ix_(ix, ix)].copy()

    def cov_matrix(self, origin, assets: list[str] | None = None) -> pd.DataFrame:
        """Realized covariance (the k-day target matrix ``Sigma``, = ``D C D``).

        With ``assets=None`` uses the daily-based assets (all their pairs are
        valid under the k-day definition). With explicit ``assets``, proxy
        vols are allowed on the diagonal but any pair lacking a realized
        correlation is NaN.
        """
        if assets is None:
            assets = self.complete_daily_set(origin)
        return pd.DataFrame(self.cov_array(origin, assets), index=assets, columns=assets)

    def cov_array(self, origin, assets: list[str]) -> np.ndarray:
        V = self._dense()[2]
        k, ix = self._row(origin), self._idx(assets)
        v = V[k][ix] if k >= 0 else np.full(len(ix), np.nan)
        return self.corr_array(origin, assets) * np.outer(v, v)

    def complete_daily_set(self, origin) -> list[str]:
        """Daily-based assets at ``origin`` with every pairwise corr valid.

        Under the k-day definition every pair of daily-based assets is valid,
        so this equals :meth:`daily_assets` except for degenerate windows (an
        asset with zero realized variance has no correlation); kept as a
        guard.
        """
        assets = self.daily_assets(origin)
        if len(assets) < 2:
            return assets
        miss = np.isnan(self.corr_array(origin, assets))
        keep = list(range(len(assets)))
        while miss[np.ix_(keep, keep)].any():
            sub = miss[np.ix_(keep, keep)].sum(axis=0)
            keep.pop(int(sub.argmax()))
        return [assets[i] for i in keep]

    # ----------------------------------------------------------------- I/O
    def save(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        for name, df in (("vol", self.vol), ("corr", self.corr), ("fwd_ret", self.fwd_ret)):
            tmp = directory / f".{name}.parquet.tmp"
            df.to_parquet(tmp, index=False)
            tmp.replace(directory / f"{name}.parquet")
        (directory / "meta.json").write_text(json.dumps(self.meta, indent=2, default=str))

    @classmethod
    def load(cls, directory: Path) -> "Targets":
        directory = Path(directory)
        meta = json.loads((directory / "meta.json").read_text())
        out = {}
        for name in ("vol", "corr", "fwd_ret"):
            df = pd.read_parquet(directory / f"{name}.parquet")
            df["origin"] = _ts_ns(df["origin"]).to_numpy()
            for c in ("asset", "asset_i", "asset_j"):
                if c in df.columns:
                    df[c] = df[c].astype("str")
            out[name] = df
        return cls(meta=meta, **out)


# ------------------------------------------------------------- computation
def _prefix(a: np.ndarray) -> np.ndarray:
    """Prefix sums with a leading zero row: sum of rows [i, j) = P[j] - P[i]."""
    out = np.zeros((a.shape[0] + 1,) + a.shape[1:], dtype=float)
    np.cumsum(a, axis=0, out=out[1:])
    return out


def compute_targets(panel: Panel, min_daily_frac: float = MIN_DAILY_FRAC,
                    min_days_per_month: int = MIN_DAYS_PER_MONTH,
                    horizon_days: int = HORIZON_DAYS) -> Targets:
    """Compute all realized targets from the FULL panel (vectorized)."""
    if panel.asof_date is not None:
        raise ValueError("targets must be computed from the full panel, not an asof view")
    k = int(horizon_days)
    if k < 1:
        raise ValueError("horizon_days must be >= 1")
    assets = list(panel.monthly.columns)
    N, H = len(assets), HORIZON_MONTHS
    through = observed_through(panel)
    meta = _meta(assets, through, min_daily_frac, min_days_per_month, k)

    # month-end grid; origin at grid position p has window months p+1..p+H
    first = panel.monthly.index.min()
    grid = pd.date_range(pd.Timestamp(first) + pd.offsets.MonthEnd(0), through, freq="ME")
    G = len(grid)
    n_orig = G - H
    if n_orig <= 0:
        return _empty_targets(meta)
    origins = grid[:n_orig]

    # --------------------------------------------------------- monthly proxy
    xm = panel.excess_monthly.reindex(grid).to_numpy(float)          # G x N
    okm = np.isfinite(xm)
    xm0 = np.where(okm, xm, 0.0)
    Pm_cnt, Pm_sq, Pm_sum = _prefix(okm.astype(float)), _prefix(xm0 ** 2), _prefix(xm0)
    a, b = np.arange(1, n_orig + 1), np.arange(1 + H, n_orig + 1 + H)  # rows [a, b)
    m_cnt = Pm_cnt[b] - Pm_cnt[a]
    m_full = m_cnt >= H - 0.5
    proxy_var = np.where(m_full, (Pm_sq[b] - Pm_sq[a]) * H / np.maximum(m_cnt, 1), np.nan)
    fwd = np.where(m_full, Pm_sum[b] - Pm_sum[a], np.nan)

    # worst tier in the window
    tiers = panel.tiers.reindex(index=grid, columns=assets)
    tr = tiers.apply(lambda s: s.map(_TIER_RANK)).to_numpy(float)    # NaN if no tier
    tr = np.where(np.isfinite(tr), tr, -1.0)
    win = np.lib.stride_tricks.sliding_window_view(tr, H, axis=0)[1:n_orig + 1]  # n_orig x N x H
    worst = win.max(axis=2)

    # ------------------------------------------------------- daily coverage
    d = panel.excess_daily.reindex(columns=assets)
    d = d.loc[d.index <= through]
    didx = d.index
    xd = d.to_numpy(float)
    okd = np.isfinite(xd)
    xd0 = np.where(okd, xd, 0.0)
    trade = okd.any(axis=1)
    ends = origins + pd.offsets.MonthEnd(H)
    s = didx.searchsorted(origins, side="right")                      # window rows [s, e)
    e = didx.searchsorted(ends, side="right")
    Pd_cnt, Pd_tr = _prefix(okd.astype(float)), _prefix(trade.astype(float))
    n_obs = Pd_cnt[e] - Pd_cnt[s]                                     # n_orig x N
    n_days = Pd_tr[e] - Pd_tr[s]                                      # n_orig
    month_cnt = pd.DataFrame(okd.astype(float), index=didx).groupby(
        didx + pd.offsets.MonthEnd(0)).sum().reindex(grid, fill_value=0.0).to_numpy()
    mc_win = np.lib.stride_tricks.sliding_window_view(month_cnt, H, axis=0)[1:n_orig + 1]
    min_month = mc_win.min(axis=2)

    # ----------------------------------------------- k-day realized matrix
    # Y row m = sum of daily rows m..m+k-1 (missing = 0); window rows [s, e)
    # hold the sums with rows m in [s, e - k], i.e. Y rows [s, e - k + 1).
    Y = kday_sums(xd0, k)
    hi = np.maximum(e - k + 1, s)
    n_sums = (hi - s).astype(float)                                   # n_orig
    P_yy = _prefix(np.einsum("ti,tj->tij", Y, Y))                     # len(Y) + 1 rows
    hi_c = np.minimum(hi, len(P_yy) - 1)                              # no-op unless < k rows in all
    s_c = np.minimum(s, hi_c)
    G_win = P_yy[hi_c] - P_yy[s_c]                                    # n_orig x N x N
    with np.errstate(invalid="ignore", divide="ignore"):
        Sig = G_win * (TRADING_DAYS / k) / n_sums[:, None, None]
    diag = np.diagonal(Sig, axis1=1, axis2=2)

    daily_ok = (n_days[:, None] > 0) & (n_sums[:, None] > 0) \
        & (n_obs >= min_daily_frac * n_days[:, None]) & (min_month >= min_days_per_month) \
        & np.isfinite(diag) & (diag > 0)
    var = np.where(daily_ok, diag, proxy_var)
    proxy = ~daily_ok & np.isfinite(proxy_var)
    has = np.isfinite(var)
    oi, ai = np.nonzero(has)
    an = np.asarray(assets, dtype=object)
    vol_long = pd.DataFrame({
        "origin": origins[oi].astype("datetime64[ns]"),
        "asset": an[ai],
        "vol": np.sqrt(var[oi, ai]),
        "var": var[oi, ai],
        "proxy": proxy[oi, ai],
        "n_obs": np.where(proxy[oi, ai], H, n_obs[oi, ai]).astype(int),
        "n_days": np.where(proxy[oi, ai], 0, n_days[oi]).astype(int),
        "tier": np.where(worst[oi, ai] >= 0, _TIER_LETTER[np.clip(worst[oi, ai], 0, 3).astype(int)], None),
    })
    vol_long["asset"] = vol_long["asset"].astype("str")

    # ------------------------------------------------------- correlations
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = Sig / np.sqrt(diag[:, :, None] * diag[:, None, :])
    valid = daily_ok[:, :, None] & daily_ok[:, None, :] & np.isfinite(corr)
    iu, ju = np.triu_indices(N, k=1)
    vmask = valid[:, iu, ju]
    oo, pp = np.nonzero(vmask)
    corr_long = pd.DataFrame({
        "origin": origins[oo].astype("datetime64[ns]"),
        "asset_i": an[iu[pp]], "asset_j": an[ju[pp]],
        "corr": np.clip(corr[oo, iu[pp], ju[pp]], -1.0, 1.0),
        "n_obs": n_sums[oo].astype(int),
    })
    corr_long["asset_i"] = corr_long["asset_i"].astype("str")
    corr_long["asset_j"] = corr_long["asset_j"].astype("str")

    # ------------------------------------------------------- forward return
    fo, fa = np.nonzero(np.isfinite(fwd))
    fwd_long = pd.DataFrame({"origin": origins[fo].astype("datetime64[ns]"),
                             "asset": an[fa], "ret": fwd[fo, fa]})
    fwd_long["asset"] = fwd_long["asset"].astype("str")

    return Targets(vol=vol_long, corr=corr_long, fwd_ret=fwd_long, meta=meta)


def _rules(f, m, k) -> dict:
    return {"horizon_days": k, "min_daily_frac": f, "min_days_per_month": m,
            "missing_day": "contributes 0 to k-day sums", "centered": False,
            "trading_days": TRADING_DAYS, "horizon_months": HORIZON_MONTHS}


def _meta(assets, through, f, m, k) -> dict:
    return {
        "version": TARGETS_VERSION, "horizon_days": k, "definition": DEFINITION.format(k=k),
        "assets": assets, "observed_through": str(pd.Timestamp(through).date()),
        "rules": _rules(f, m, k),
    }


def _empty_targets(meta) -> Targets:
    vol = pd.DataFrame({"origin": pd.Series([], dtype="datetime64[ns]"), "asset": pd.Series([], dtype="str"),
                        "vol": [], "var": [], "proxy": pd.Series([], dtype=bool),
                        "n_obs": pd.Series([], dtype=int), "n_days": pd.Series([], dtype=int),
                        "tier": pd.Series([], dtype=object)})
    corr = pd.DataFrame({"origin": pd.Series([], dtype="datetime64[ns]"), "asset_i": pd.Series([], dtype="str"),
                         "asset_j": pd.Series([], dtype="str"), "corr": [], "n_obs": pd.Series([], dtype=int)})
    fwd = pd.DataFrame({"origin": pd.Series([], dtype="datetime64[ns]"), "asset": pd.Series([], dtype="str"),
                        "ret": []})
    return Targets(vol, corr, fwd, meta)


# ------------------------------------------------------------------ caching
def panel_fingerprint(panel: Panel, horizon_days: int = HORIZON_DAYS) -> str:
    """Hash of the return data and the target definition; changes invalidate the cache."""
    h = hashlib.sha256()
    for obj in (panel.daily, panel.monthly, panel.cash_daily, panel.cash_monthly):
        h.update(pd.util.hash_pandas_object(obj, index=True).to_numpy().tobytes())
        labels = list(obj.columns) if isinstance(obj, pd.DataFrame) else [obj.name]
        h.update(str(labels).encode())
    h.update(json.dumps([TARGETS_VERSION, _rules(MIN_DAILY_FRAC, MIN_DAYS_PER_MONTH, int(horizon_days))],
                        sort_keys=True).encode())
    return h.hexdigest()[:16]


def targets_dir(results_dir: Path | None = None, horizon_days: int = HORIZON_DAYS) -> Path:
    """``<results>/targets`` for the default horizon, ``targets_h<k>`` otherwise."""
    name = "targets" if int(horizon_days) == HORIZON_DAYS else f"targets_h{int(horizon_days)}"
    return Path(results_dir or RESULTS) / name


def _definition_ok(meta: dict, horizon_days: int) -> bool:
    return meta.get("version") == TARGETS_VERSION and meta.get("horizon_days") == int(horizon_days)


def load_targets(results_dir: Path | None = None, horizon_days: int = HORIZON_DAYS) -> Targets:
    """Cached targets; raises if missing or built under another target definition."""
    d = targets_dir(results_dir, horizon_days)
    if not (d / "meta.json").exists():
        raise FileNotFoundError(f"no cached targets in {d}; run `python -m volmodel.eval.run targets`")
    t = Targets.load(d)
    if not _definition_ok(t.meta, horizon_days):
        raise ValueError(f"cached targets in {d} were built under another definition "
                         f"(version {t.meta.get('version')}, horizon_days {t.meta.get('horizon_days')}; "
                         f"expected version {TARGETS_VERSION}, horizon_days {horizon_days}); "
                         f"run `python -m volmodel.eval.run targets --rebuild`")
    return t


def get_targets(panel: Panel, results_dir: Path | None = None, rebuild: bool = False,
                horizon_days: int = HORIZON_DAYS) -> Targets:
    """Cached targets for ``panel``; recomputed if missing, stale, of another
    definition (version / horizon), or ``rebuild``."""
    d = targets_dir(results_dir, horizon_days)
    fp = panel_fingerprint(panel, horizon_days)
    if not rebuild and (d / "meta.json").exists():
        try:
            t = Targets.load(d)
            if t.meta.get("fingerprint") == fp and _definition_ok(t.meta, horizon_days):
                return t
        except Exception:  # corrupt cache: rebuild
            pass
    t = compute_targets(panel, horizon_days=horizon_days)
    t.meta["fingerprint"] = fp
    t.save(d)
    return t
