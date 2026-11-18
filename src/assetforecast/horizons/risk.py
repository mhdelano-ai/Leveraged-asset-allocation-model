"""Realized volatility and correlation at every horizon, and simple forecasts of both.

Realized measures (window ``(t, t + h months]``, daily log excess returns, a
day on which an asset did not trade counts 0, as in ``eval/targets.py``)
-----------------------------------------------------------------------
For ``k`` in ``K = (21, 1)``: with ``s`` the overlapping k-day sums whose k days
all lie inside the window, variance ``= (252/k) mean(s_a^2)`` and correlation
``= sum(s_a s_b) / sqrt(sum(s_a^2) sum(s_b^2))``, both uncentered.

* ``k = 21`` is the package's target definition (``ARCHITECTURE.md``): sources
  priced by different conventions agree at 21 days. A 1-month window holds at
  most a couple of 21-day sums, so at h = 1 the variance is 12 times the
  squared calendar-month return (the architecture's monthly fallback) and the
  correlation is undefined. Short windows make this measure very noisy: an
  unpredictable measurement error raises every forecaster's loss and pulls
  skill toward zero.
* ``k = 1`` is ordinary daily realized variance: far less noisy at short
  horizons, but it depends on how each source is priced
  (``reports/data/horizon_evidence.md``), with level shifts at splices.

Forecasters (everything from sums that end at or before the origin)
-------------------------------------------------------------------
``naive``      the same statistic over the asset's (pair's) whole daily history:
               the forecast if risk were unpredictable. The bar.
``trail_L``    the statistic over the last L months (3, 12, 60 for volatility;
               12, 36, 60 for correlation)
``half``       the mean of ``trail_12`` (``trail_36`` for correlation) and ``naive``
``har``        volatility only: per asset, a regression of the log realized
               variance on the logs of ``trail_3``, ``trail_12`` and ``trail_60``,
               fitted on windows already closed at the origin (``s + h <= t``),
               expanding, refit every 12 months, at least ``HAR_MIN`` rows;
               forecast ``exp(fit + half the residual variance)``
``published``  h = 12, k = 21 only: the selected model ``combo_cov_main``
               (``cov_21d`` of ``results/covariance/cov_history.parquet``), as a
               reference point for what the package already does

An asset is forecast once it has ``MIN_DAYS`` of daily history (a pair: both).

Scores
------
Volatility: QLIKE on the variance, ``v/f - log(v/f) - 1``; correlation: squared
error. ``skill = 1 - mean loss / mean loss of naive`` on the origins where both
exist; ``win``: the share of origins on which the model's loss is the lower one
(a single crisis cannot move it, unlike the mean); ``dm_t``, ``dm_p``: Diebold-Mariano against ``naive`` (one-sided, model
better, Bartlett long-run variance with 1.5 h lags). Pooled rows average the
loss over assets (pairs) per origin first; the per-asset correlation rows
average over the pairs that contain the asset.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..returns.evaluation import inference
from . import EARLY_START, HORIZONS, MIN_INDEP, PERIODS, period_mask

K = (21, 1)
MIN_DAYS = 1260          # daily history before an asset (pair) is forecast
HAR_MIN = 60             # realized rows before the regression is used
HAR_REFIT = 12
VOL_TRAILS = (3, 12, 60)
CORR_TRAILS = (12, 36, 60)
VOL_MODELS = ("naive", "trail_3", "trail_12", "trail_60", "half", "har", "published")
CORR_MODELS = ("naive", "trail_12", "trail_36", "trail_60", "half", "published")


class Daily:
    """Prefix sums of the products of k-day sums: every window statistic is a difference."""

    def __init__(self, x: pd.DataFrame, months: pd.DatetimeIndex, k: int):
        self.k, self.assets, self.months = k, list(x.columns), months
        started = x.notna().cummax().to_numpy()
        x0 = x.fillna(0.0).to_numpy()
        T, N = x0.shape
        c = np.vstack([np.zeros((1, N)), np.cumsum(x0, axis=0)])
        s = np.zeros((T, N))
        s[k - 1:] = c[k:] - c[:-k]
        valid = np.zeros((T, N), dtype=bool)
        valid[k - 1:] = started[: T - k + 1]            # all k days at or after the first quote
        s[~valid] = 0.0
        self.cum = c                                    # prefix of daily returns
        self.pp = np.zeros((T + 1, N, N))               # prefix of s_a s_b
        np.cumsum(s[:, :, None] * s[:, None, :], axis=0, out=self.pp[1:])
        self.nn = np.zeros((T + 1, N), dtype=np.int64)  # prefix count of valid sums
        np.cumsum(valid, axis=0, out=self.nn[1:])
        # prefix index just after the last day of each month end
        self.pos = np.searchsorted(x.index.values, months.values, side="right")
        first = np.array([valid[:, a].argmax() if valid[:, a].any() else T for a in range(N)])
        self.first_pair = np.maximum(first[:, None], first[None, :])   # first day both valid

    def _stat(self, lo, hi):
        """(variance T x N, correlation T x N x N, count T x N) over sums ending at
        prefix positions [lo, hi); NaN where an asset is not valid throughout."""
        lo, hi = np.asarray(lo), np.asarray(hi)
        good = hi > lo
        lo_c = np.where(good, lo, hi)
        d = self.pp[hi] - self.pp[lo_c]
        n = self.nn[hi] - self.nn[lo_c]
        full = good[:, None] & (n == (hi - lo_c)[:, None]) & (n > 0)
        diag = np.einsum("tii->ti", d)
        var = np.where(full, diag / np.maximum(n, 1) * (252.0 / self.k), np.nan)
        with np.errstate(invalid="ignore", divide="ignore"):
            corr = d / np.sqrt(diag[:, :, None] * diag[:, None, :])
        corr[~(full[:, :, None] & full[:, None, :])] = np.nan
        return var, corr, n

    def _blank(self):
        M, N = len(self.months), len(self.assets)
        return np.full((M, N), np.nan), np.full((M, N, N), np.nan)

    def realized(self, h: int):
        """Realized variance and correlation for the window after each month end."""
        i = np.arange(len(self.months) - h)
        var, corr = self._blank()
        lo, hi = self.pos[i] + self.k - 1, self.pos[i + h]
        v, c, n = self._stat(lo, hi)
        if self.k == 21 and h == 1:      # the monthly fallback of the target definition
            r = self.cum[self.pos[i + 1]] - self.cum[self.pos[i]]
            ok = self.nn[self.pos[i]] > 0          # the asset was trading before the month began
            v, c = np.where(ok, 12.0 * r * r, np.nan), np.full_like(c, np.nan)
        var[i], corr[i] = v, c
        return var, corr

    def trailing(self, L: int):
        i = np.arange(L, len(self.months))
        var, corr = self._blank()
        v, c, _ = self._stat(self.pos[i - L] + self.k - 1, self.pos[i])
        var[i], corr[i] = v, c
        return var, corr

    def expanding(self):
        """Whole-history statistic through each month end (pairs: from the first day both exist)."""
        hi = self.pos
        d = self.pp[hi]
        n = self.nn[hi]
        diag = np.einsum("tii->ti", d)
        var = np.where(n >= MIN_DAYS, diag / np.maximum(n, 1) * (252.0 / self.k), np.nan)
        ppd = np.einsum("tii->ti", self.pp)                 # prefix of s_a^2
        N = len(self.assets)
        ia, ib = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
        # sums of s_a^2 and s_b^2 over the days on which the pair (a, b) both exist
        da = diag[:, :, None] - ppd[self.first_pair, ia][None]
        db = diag[:, None, :] - ppd[self.first_pair, ib][None]
        npair = np.minimum(n[:, :, None], n[:, None, :])
        with np.errstate(invalid="ignore", divide="ignore"):
            corr = d / np.sqrt(da * db)
        corr[npair < MIN_DAYS] = np.nan
        return var, corr


def _har(logy: np.ndarray, X: np.ndarray, h: int) -> np.ndarray:
    """Expanding purged log regression for one asset; returns forecast variances."""
    M = len(logy)
    out = np.full(M, np.nan)
    okx = np.isfinite(X).all(axis=1)
    oky = okx & np.isfinite(logy)
    beta = s2 = None
    last_fit = -10**9
    for t in range(M):
        if not okx[t]:
            continue
        if t - last_fit >= HAR_REFIT:
            rows = np.flatnonzero(oky[: max(t - h + 1, 0)])
            if len(rows) >= HAR_MIN:
                A = np.column_stack([np.ones(len(rows)), X[rows]])
                beta, *_ = np.linalg.lstsq(A, logy[rows], rcond=None)
                res = logy[rows] - A @ beta
                s2 = float(res @ res / max(len(rows) - A.shape[1], 1))
                last_fit = t
        if beta is not None:
            out[t] = np.exp(beta[0] + X[t] @ beta[1:] + 0.5 * s2)
    return out


def _published(months, assets):
    from .. import paths
    if not paths.COV_HISTORY.exists():
        return None
    c = pd.read_parquet(paths.COV_HISTORY, columns=["origin", "asset_i", "asset_j", "cov_21d"])
    M, N = len(months), len(assets)
    cov = np.full((M, N, N), np.nan)
    ai = pd.Index(assets)
    t, i, j = months.get_indexer(pd.DatetimeIndex(c["origin"])), ai.get_indexer(c["asset_i"]), ai.get_indexer(c["asset_j"])
    ok = (t >= 0) & (i >= 0) & (j >= 0)
    cov[t[ok], i[ok], j[ok]] = c["cov_21d"].to_numpy()[ok]
    cov[t[ok], j[ok], i[ok]] = c["cov_21d"].to_numpy()[ok]
    var = np.einsum("tii->ti", cov).copy()
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = cov / np.sqrt(var[:, :, None] * var[:, None, :])
    return var, corr


def _frame(cols: dict, index: pd.MultiIndex, k: int, h: int) -> pd.DataFrame:
    f = pd.DataFrame(cols, index=index).dropna(subset=["naive"]).reset_index()
    f.insert(0, "horizon", h)
    f.insert(0, "k", k)
    return f


def run(panel, horizons=HORIZONS, ks=K, start: str = EARLY_START):
    """(vol, corr): long frames with the realized value (``target``) and one
    column per forecaster. ``vol`` holds variances (annualized)."""
    from ..returns import inputs
    through = inputs.observed_through(panel)
    x = panel.excess_daily
    months = pd.date_range(x.index[0] + pd.offsets.MonthEnd(0), through, freq="ME")
    x = x.loc[:through]
    assets = list(x.columns)
    N = len(assets)
    iu = np.triu_indices(N, 1)
    pairs = [f"{assets[i]}|{assets[j]}" for i, j in zip(*iu)]
    keep = months >= pd.Timestamp(start)
    vols, corrs = [], []
    for k in ks:
        d = Daily(x, months, k)
        nv, nc = d.expanding()
        tv, tc = {}, {}
        for L in sorted(set(VOL_TRAILS) | set(CORR_TRAILS)):
            tv[L], tc[L] = d.trailing(L)
        ready = (d.nn[d.pos] >= MIN_DAYS)                     # asset has its history at the origin
        fv = {"naive": nv, **{f"trail_{L}": tv[L] for L in VOL_TRAILS}, "half": 0.5 * (tv[12] + nv)}
        fc = {"naive": nc, **{f"trail_{L}": tc[L] for L in CORR_TRAILS}, "half": 0.5 * (tc[36] + nc)}
        pub = _published(months, assets) if k == 21 else None
        with np.errstate(invalid="ignore", divide="ignore"):
            Xh = np.stack([np.log(tv[L]) for L in VOL_TRAILS], axis=2)
        for h in horizons:
            rv, rc = d.realized(h)
            with np.errstate(invalid="ignore", divide="ignore"):
                ly = np.log(np.where(rv > 0, rv, np.nan))
            har = np.column_stack([_har(ly[:, a], Xh[:, a, :], h) for a in range(N)])
            cols = {"target": rv, **fv, "har": har}
            if pub is not None and h == 12:
                cols["published"] = pub[0]
            vols.append(_frame({c: np.where(ready, v, np.nan)[keep].ravel() for c, v in cols.items()},
                               pd.MultiIndex.from_product([months[keep], assets], names=["date", "asset"]), k, h))
            if k == 21 and h == 1:
                continue
            rp = ready[:, :, None] & ready[:, None, :]
            cols = {"target": rc, **fc}
            if pub is not None and h == 12:
                cols["published"] = pub[1]
            corrs.append(_frame({c: np.where(rp, v, np.nan)[keep][:, iu[0], iu[1]].ravel() for c, v in cols.items()},
                                pd.MultiIndex.from_product([months[keep], pairs], names=["date", "pair"]), k, h))
    corr = pd.concat(corrs, ignore_index=True) if corrs else pd.DataFrame(
        columns=["k", "horizon", "date", "pair", "target", "naive"])
    return pd.concat(vols, ignore_index=True), corr


def qlike(v, f):
    r = np.asarray(v, dtype=float) / np.asarray(f, dtype=float)
    return r - np.log(r) - 1.0


def _skill(lm: pd.Series, lb: pd.Series, dates, h: int) -> dict:
    """Skill and DM from per-row losses, averaged per origin first."""
    g = pd.DataFrame({"m": np.asarray(lm), "b": np.asarray(lb)}).groupby(np.asarray(dates)).mean()
    n = len(g)
    t, p1, _ = inference.diebold_mariano(g["m"].to_numpy(), g["b"].to_numpy(), h)
    mb = g["b"].mean()
    return {"skill": 1 - g["m"].mean() / mb if mb > 0 else np.nan, "dm_t": t, "dm_p": p1,
            "win": float((g["m"] < g["b"]).mean()),
            "loss": float(g["m"].mean()), "loss_naive": float(mb),
            "n": n, "n_indep": n / h, "thin": n / h < MIN_INDEP}


def score_vol(vol: pd.DataFrame, periods=PERIODS) -> pd.DataFrame:
    rows = []
    for (k, h), g in vol.groupby(["k", "horizon"]):
        g = g[(g["target"] > 0) & g["target"].notna()]
        for m in [c for c in VOL_MODELS if c in g.columns and c != "naive"]:
            w = g[g[m].notna() & (g[m] > 0)]
            lm, lb = qlike(w["target"], w[m]), qlike(w["target"], w["naive"])
            for per in periods:
                sel = period_mask(w["date"], h, per)
                for a in [*pd.unique(w["asset"]), "pooled"]:
                    s = sel if a == "pooled" else sel & (w["asset"] == a).to_numpy()
                    if s.sum() < 12:
                        continue
                    rows.append({"k": k, "horizon": h, "model": m, "asset": a, "period": per,
                                 **_skill(lm[s], lb[s], w["date"].to_numpy()[s], h),
                                 "vol_realized": float(np.sqrt(w["target"].to_numpy()[s].mean()))})
    return pd.DataFrame(rows)


def score_corr(corr: pd.DataFrame, periods=PERIODS) -> pd.DataFrame:
    rows = []
    pa = corr["pair"].str.split("|", expand=True)
    corr = corr.assign(a=pa[0], b=pa[1])
    assets = pd.unique(pd.concat([corr["a"], corr["b"]]))
    for (k, h), g in corr.groupby(["k", "horizon"]):
        g = g[g["target"].notna()]
        for m in [c for c in CORR_MODELS if c in g.columns and c != "naive"]:
            w = g[g[m].notna()]
            lm = ((w["target"] - w[m]) ** 2).to_numpy()
            lb = ((w["target"] - w["naive"]) ** 2).to_numpy()
            for per in periods:
                sel = period_mask(w["date"], h, per)
                for a in [*assets, "pooled"]:
                    s = sel if a == "pooled" else sel & ((w["a"] == a) | (w["b"] == a)).to_numpy()
                    if s.sum() < 12:
                        continue
                    rows.append({"k": k, "horizon": h, "model": m, "asset": a, "period": per,
                                 **_skill(lm[s], lb[s], w["date"].to_numpy()[s], h)})
    return pd.DataFrame(rows)
