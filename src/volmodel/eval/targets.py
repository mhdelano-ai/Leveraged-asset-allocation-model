"""Realized 12-month targets, computed once from the FULL panel and cached.

For every month-end origin ``t`` whose window ``(t, t + 12 calendar months]``
is fully observed in the panel (see :func:`observed_through`):

vol      per-asset annualized realized vol of daily LOG EXCESS returns,
         uncentered: ``var = sum(r_d^2) * 252 / n`` over the asset's own daily
         observations in the window. If the asset fails the daily coverage
         rule, the monthly proxy ``var = 12 * mean(r_m^2)`` over the 12 monthly
         log excess returns is used instead and the row is flagged
         ``proxy=True``. If neither is available there is no row.
corr     per-pair realized correlation from OVERLAPPING 3-day log excess
         returns (sums of three consecutive daily rows, all three inside the
         window), uncentered (consistent with the uncentered variances):
         ``sum(x_i x_j) / sqrt(sum(x_i^2) sum(x_j^2))`` over 3-day sums jointly
         observed for both assets. Only pairs where BOTH assets have a
         daily-based vol target get a value; others have no row (= NaN).
cov      ``D C D`` on the daily-based assets (see :meth:`Targets.cov_matrix`).
fwd_ret  forward 12-month log excess return per asset = sum of the 12 monthly
         log excess returns in the window (no row unless all 12 are present).

Coverage rules (module constants, recorded in the cache metadata):

* ``trading days`` in a window = rows of the daily index inside the window on
  which at least one asset has a return.
* An asset's target is DAILY-based iff it has returns on at least
  ``MIN_DAILY_FRAC`` (90%) of the window's trading days AND at least
  ``MIN_DAYS_PER_MONTH`` (10) daily returns in each of the 12 calendar months.
  The second condition stops an asset whose daily history starts or stops
  mid-window from passing on a technicality; such windows use the proxy.
* The monthly proxy requires all 12 monthly log excess returns.
* A pair's correlation requires both assets daily-based and at least
  ``MIN_PAIR_FRAC`` (50%) of the window's possible 3-day sums jointly
  observed. (A 3-day sum is missing if any of its three days is missing, so
  scattered single-asset holidays thin the joint sample roughly threefold;
  50% still leaves >= ~125 overlapping observations.)

Everything is vectorized with prefix sums over the daily and monthly arrays;
there is no per-origin pandas slicing.

Public API::

    observed_through(panel) -> Timestamp
    compute_targets(panel) -> Targets
    get_targets(panel, results_dir=None, rebuild=False) -> Targets   # cached
    load_targets(results_dir=None) -> Targets
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

TARGETS_VERSION = 1
MIN_DAILY_FRAC = 0.90
MIN_DAYS_PER_MONTH = 10
MIN_PAIR_FRAC = 0.50
CORR_DAYS = 3
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


# ---------------------------------------------------------------- container
@dataclass
class Targets:
    """Realized targets in long format (one row per origin x asset / pair).

    vol      origin, asset, vol, var, proxy (bool), n_obs (daily obs, or 12 for
             the proxy), n_days (window trading days), tier (worst data tier
             in the window, or NaN)
    corr     origin, asset_i, asset_j, corr, n_obs (joint 3-day sums);
             i < j in panel column order; only valid pairs are present
    fwd_ret  origin, asset, ret
    meta     rules, observed_through, fingerprint, asset order
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
        """Realized covariance ``D C D``.

        With ``assets=None`` uses the largest set of daily-based assets whose
        pairwise correlations are all valid (assets with the most missing pairs
        are dropped first, which only happens if joint coverage fails).
        With explicit ``assets``, proxy vols are allowed on the diagonal but
        any pair lacking a realized correlation is NaN.
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
        """Daily-based assets at ``origin`` with every pairwise corr valid."""
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
                    min_pair_frac: float = MIN_PAIR_FRAC) -> Targets:
    """Compute all realized targets from the FULL panel (vectorized)."""
    if panel.asof_date is not None:
        raise ValueError("targets must be computed from the full panel, not an asof view")
    assets = list(panel.monthly.columns)
    N, H = len(assets), HORIZON_MONTHS
    through = observed_through(panel)

    # month-end grid; origin at grid position p has window months p+1..p+H
    first = panel.monthly.index.min()
    grid = pd.date_range(pd.Timestamp(first) + pd.offsets.MonthEnd(0), through, freq="ME")
    G = len(grid)
    n_orig = G - H
    if n_orig <= 0:
        return _empty_targets(assets, through)
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

    # ------------------------------------------------------------- daily vol
    d = panel.excess_daily.reindex(columns=assets)
    d = d.loc[d.index <= through]
    didx = d.index
    xd = d.to_numpy(float)
    okd = np.isfinite(xd)
    xd0 = np.where(okd, xd, 0.0)
    trade = okd.any(axis=1)
    ends = origins + pd.offsets.MonthEnd(H)
    s = didx.searchsorted(origins, side="right")
    e = didx.searchsorted(ends, side="right")
    Pd_cnt, Pd_sq, Pd_tr = _prefix(okd.astype(float)), _prefix(xd0 ** 2), _prefix(trade.astype(float))
    n_obs = Pd_cnt[e] - Pd_cnt[s]                                     # n_orig x N
    n_days = Pd_tr[e] - Pd_tr[s]                                      # n_orig
    ss = Pd_sq[e] - Pd_sq[s]

    # per-month daily counts on the grid, min over each window's 12 months
    month_cnt = pd.DataFrame(okd.astype(float), index=didx).groupby(
        didx + pd.offsets.MonthEnd(0)).sum().reindex(grid, fill_value=0.0).to_numpy()
    mc_win = np.lib.stride_tricks.sliding_window_view(month_cnt, H, axis=0)[1:n_orig + 1]
    min_month = mc_win.min(axis=2)
    daily_ok = (n_days[:, None] > 0) & (n_obs >= min_daily_frac * n_days[:, None]) \
        & (min_month >= min_days_per_month)
    daily_var = np.where(daily_ok, ss * TRADING_DAYS / np.maximum(n_obs, 1), np.nan)

    var = np.where(daily_ok, daily_var, proxy_var)
    proxy = ~daily_ok & np.isfinite(proxy_var)
    has = np.isfinite(var)
    oi, ai = np.nonzero(has)
    vol_long = pd.DataFrame({
        "origin": origins[oi].astype("datetime64[ns]"),
        "asset": np.asarray(assets, dtype=object)[ai],
        "vol": np.sqrt(var[oi, ai]),
        "var": var[oi, ai],
        "proxy": proxy[oi, ai],
        "n_obs": np.where(proxy[oi, ai], H, n_obs[oi, ai]).astype(int),
        "n_days": np.where(proxy[oi, ai], 0, n_days[oi]).astype(int),
        "tier": np.where(worst[oi, ai] >= 0, _TIER_LETTER[np.clip(worst[oi, ai], 0, 3).astype(int)], None),
    })
    vol_long["asset"] = vol_long["asset"].astype("str")

    # ------------------------------------------------------- 3-day corr
    y = pd.DataFrame(xd, index=didx).rolling(CORR_DAYS, min_periods=CORR_DAYS).sum().to_numpy()
    oky = np.isfinite(y)
    y0 = np.where(oky, y, 0.0)
    ok_f = oky.astype(float)
    # T x N x N arrays: cross products, own squares on the joint sample, joint counts
    P_xy = _prefix(np.einsum("ti,tj->tij", y0, y0))
    P_xx = _prefix(np.einsum("ti,tj->tij", y0 ** 2, ok_f))           # x_i^2 where j observed
    P_n = _prefix(np.einsum("ti,tj->tij", ok_f, ok_f))
    s3 = np.minimum(s + CORR_DAYS - 1, e)
    sxy, sxx, nj = P_xy[e] - P_xy[s3], P_xx[e] - P_xx[s3], P_n[e] - P_n[s3]
    syy = np.swapaxes(sxx, 1, 2)
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = sxy / np.sqrt(sxx * syy)
    possible = np.maximum(n_days - (CORR_DAYS - 1), 0)[:, None, None]
    valid = daily_ok[:, :, None] & daily_ok[:, None, :] & (nj >= min_pair_frac * possible) \
        & (possible > 0) & np.isfinite(corr)
    iu, ju = np.triu_indices(N, k=1)
    vmask = valid[:, iu, ju]
    oo, pp = np.nonzero(vmask)
    an = np.asarray(assets, dtype=object)
    corr_long = pd.DataFrame({
        "origin": origins[oo].astype("datetime64[ns]"),
        "asset_i": an[iu[pp]], "asset_j": an[ju[pp]],
        "corr": np.clip(corr[oo, iu[pp], ju[pp]], -1.0, 1.0),
        "n_obs": nj[oo, iu[pp], ju[pp]].astype(int),
    })
    corr_long["asset_i"] = corr_long["asset_i"].astype("str")
    corr_long["asset_j"] = corr_long["asset_j"].astype("str")

    # ------------------------------------------------------- forward return
    fo, fa = np.nonzero(np.isfinite(fwd))
    fwd_long = pd.DataFrame({"origin": origins[fo].astype("datetime64[ns]"),
                             "asset": an[fa], "ret": fwd[fo, fa]})
    fwd_long["asset"] = fwd_long["asset"].astype("str")

    meta = _meta(assets, through, min_daily_frac, min_days_per_month, min_pair_frac)
    return Targets(vol=vol_long, corr=corr_long, fwd_ret=fwd_long, meta=meta)


def _meta(assets, through, f, m, p) -> dict:
    return {
        "version": TARGETS_VERSION, "assets": assets, "observed_through": str(through.date()),
        "rules": {"min_daily_frac": f, "min_days_per_month": m, "min_pair_frac": p,
                  "corr_days": CORR_DAYS, "trading_days": TRADING_DAYS, "horizon_months": HORIZON_MONTHS},
    }


def _empty_targets(assets, through) -> Targets:
    vol = pd.DataFrame({"origin": pd.Series([], dtype="datetime64[ns]"), "asset": pd.Series([], dtype="str"),
                        "vol": [], "var": [], "proxy": pd.Series([], dtype=bool),
                        "n_obs": pd.Series([], dtype=int), "n_days": pd.Series([], dtype=int),
                        "tier": pd.Series([], dtype=object)})
    corr = pd.DataFrame({"origin": pd.Series([], dtype="datetime64[ns]"), "asset_i": pd.Series([], dtype="str"),
                         "asset_j": pd.Series([], dtype="str"), "corr": [], "n_obs": pd.Series([], dtype=int)})
    fwd = pd.DataFrame({"origin": pd.Series([], dtype="datetime64[ns]"), "asset": pd.Series([], dtype="str"),
                        "ret": []})
    return Targets(vol, corr, fwd, _meta(assets, through, MIN_DAILY_FRAC, MIN_DAYS_PER_MONTH, MIN_PAIR_FRAC))


# ------------------------------------------------------------------ caching
def panel_fingerprint(panel: Panel) -> str:
    """Hash of the return data and target rules; changes invalidate the cache."""
    h = hashlib.sha256()
    for obj in (panel.daily, panel.monthly, panel.cash_daily, panel.cash_monthly):
        h.update(pd.util.hash_pandas_object(obj, index=True).to_numpy().tobytes())
        labels = list(obj.columns) if isinstance(obj, pd.DataFrame) else [obj.name]
        h.update(str(labels).encode())
    h.update(json.dumps([TARGETS_VERSION, MIN_DAILY_FRAC, MIN_DAYS_PER_MONTH, MIN_PAIR_FRAC,
                         CORR_DAYS]).encode())
    return h.hexdigest()[:16]


def targets_dir(results_dir: Path | None = None) -> Path:
    return Path(results_dir or RESULTS) / "targets"


def load_targets(results_dir: Path | None = None) -> Targets:
    d = targets_dir(results_dir)
    if not (d / "meta.json").exists():
        raise FileNotFoundError(f"no cached targets in {d}; run `python -m volmodel.eval.run targets`")
    return Targets.load(d)


def get_targets(panel: Panel, results_dir: Path | None = None, rebuild: bool = False) -> Targets:
    """Cached targets for ``panel``; recomputed if missing, stale or ``rebuild``."""
    d = targets_dir(results_dir)
    fp = panel_fingerprint(panel)
    if not rebuild and (d / "meta.json").exists():
        try:
            t = Targets.load(d)
            if t.meta.get("fingerprint") == fp:
                return t
        except Exception:  # corrupt cache: rebuild
            pass
    t = compute_targets(panel)
    t.meta["fingerprint"] = fp
    t.save(d)
    return t
