"""Predictive distributions of 12-month log excess returns: historical mean,
covariance-forecast scale, empirical shape (model ``dist_hm_af``).

For origin t and asset a::

    q_k(t, a) = m(t, a) + s(t, a) * Z_k(t)
    m(t, a)   = mean_12m: 12 x the expanding mean monthly log excess return (``expected``)
    s(t, a)   = sqrt(Sigma_t[a, a] + se(t, a)^2)      the predictive scale
    Sigma_t   = the package's 12-month covariance forecast made at t (``covhistory``)
    se(t, a)  = the standard error of m(t, a) (``expected.mean_se``)
    Z_k(t)    = the shape: quantiles of past standardized outcomes
                z(s, a) = (y(s, a) - m(s, a)) / s(s, a), pooled across assets,
                over origins s with s + 12 months <= t, recentred (see below),
                tails floored at the Gaussian with the same variance.

A policy portfolio p (``assetforecast.eval.portfolio.POLICIES``) with weights
w over the eligible assets present in the forecast gets the same treatment
with m_p = w'm, Sigma_p = w' Sigma w, se_p = sum |w_i| se_i and y_p = w'y, and a
shape pooled over the policy portfolios' own past z. The asset-pooled shape
applied to portfolios is written as the sensitivity model
``dist_hm_af.assetshape``.

Every quantity at t uses data through t only: m and se from returns through
t, Sigma_t from the covariance history (the selected model's forecast at t, the
bridge fitted on data through t), and z(s, .) only for windows that closed by
t. (The covariance MODEL was selected in 2026 on the whole history; that choice
is not point-in-time and nothing here can make it so.)

Design choices and their reasons
--------------------------------
These were fixed from first principles and simulated data
(``tests/returns/test_distribution.py`` and the WP5 scratch Monte Carlo), not
by scoring on the real panel.

1. **Mean uncertainty enters the scale, s^2 = Sigma_ii + se^2.** The old
   baselines widened by sqrt(1 + h/T), the iid version of the same thing. Here
   se is the serial-correlation-robust standard error published with the
   expected return, so the interval a reader sees around the mean and the
   width of the predictive distribution are built from the same number.
   Because z is standardized by the same s at each past origin (when the mean
   was known less well), the pooled shape does not count the mean's error
   twice: z measures outcomes in units of "what was then known to be
   uncertain", and today's s puts today's uncertainty back.
2. **The pooled z is recentred (its sample mean removed, spread kept).**
   The owner fixed the location at the historical mean; a non-zero mean of z
   would move the distribution away from it by a "correction" learned from
   the same in-sample errors the mean-model study showed to be unpredictable.
   Only the location is removed: the spread is NOT rescaled to 1, so the
   shape absorbs any level bias of the covariance forecast (its target is
   uncentred and its realized/forecast variance ran 1.05-1.11) and of the
   mean's standard error. Asset-specific mean errors stay in the spread,
   because only one pooled scalar is removed.
3. **Tails floored at the Gaussian.** Overlapping 12-month windows and the
   strong cross-asset correlation of bad years leave few independent
   observations behind the 1% and 5% quantiles (``shape_info`` reports them:
   an effective sample size for the tail indicator, and the number of
   separate episodes behind each tail). An empirical quantile from a handful
   of episodes can be thinner than any reasonable distribution, and an
   interval that is too narrow is the costly error here. So the 1%, 5%, 95%
   and 99% shape quantiles are each pushed out to at least the quantile of a
   normal with the pooled z's standard deviation (lower tails min, upper
   tails max); the empirical value stands where it is wider. Both versions
   are written (``z`` and ``z_raw`` in the shape history).
4. **No shape before ``MIN_RESID_DATES`` (120) residual origins.** Ten
   years of closed windows is about ten independent years; with fewer the
   shape is mostly noise. Before that no forecast is produced (the period is
   long before anything is scored: asset shapes exist from 1942, portfolio
   shapes from the late 1970s).
5. Equal weights over the whole residual history (no recency weighting, no
   per-group shapes): no free parameter to tune.

Portfolio log returns. The base package's portfolio tests
(``eval.portfolio``) treat a policy portfolio as held at fixed weights with
daily rebalancing and approximate its log excess return by the weighted sum of
the assets' log excess returns (``z = w'R / sqrt(w'Aw)`` in
``annual_z_scores``). The same approximation is used here, for the mean, the
realized outcomes and the residuals, so the portfolio rows are consistent
with the covariance side's calibration tests. It omits the diversification
return 0.5 (sum w_i Sigma_ii - w'Sigma w), which is +0.3 to +0.6% a year for
the policy mixes (the true rebalanced log return is higher).

Functions
---------
``standardized_residuals(xr, cov_hist, origins)``           z per (origin, asset)
``portfolio_residuals(xr, cov_hist, origins, policies)``    z per (origin, portfolio)
``pooled_shape(z, origin)`` / ``shape_info(z, origin)``     the shape at an origin
``distribution_at(origin, panel, cov_annual, shapes)``      the published tables at one origin
``build(panel, cov_hist)`` / ``history(panel, cov_hist)``   every origin (``FORECAST_COLUMNS``)
``score(history, baseline_forecasts, ...)``                 scoring (descriptive: see ``REPLICATION_LABEL``)
``centered_quantiles``, ``weighted_quantile``, ``GRID_W``, ``grid_mean``: shape
helpers ported from ``assetpred.models.dist_core``.

CLI: ``python -m assetforecast.returns.distribution [--max-date D] [--score]``.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from . import config, schema
from . import expected as _exp

log = logging.getLogger(__name__)

__all__ = [
    "TAUS", "GRID_W", "grid_mean", "weighted_quantile", "centered_quantiles",
    "MODEL", "MODEL_ASSETSHAPE", "HORIZON", "MIN_RESID_DATES", "TAIL_FLOOR",
    "standardized_residuals", "portfolio_residuals", "residual_frame", "portfolio_frame",
    "pooled_shape", "shape_info", "distribution_at", "build", "history", "write", "score",
    "portfolio_hist_mean_baseline",
]

# --------------------------------------------------------------------------
# Shape helpers (ported verbatim from assetpred.models.dist_core)
# --------------------------------------------------------------------------
TAUS = np.asarray(config.QUANTILES, dtype=float)
K = len(TAUS)
_edges = np.concatenate([[0.0], TAUS, [1.0]])
GRID_W = (_edges[2:] - _edges[:-2]) / 2.0
GRID_W = GRID_W / GRID_W.sum()          # quantile-grid weights for an approximate mean


def grid_mean(Q: np.ndarray) -> np.ndarray:
    """Approximate mean of distributions given by quantiles on the grid (rows)."""
    return np.asarray(Q, float) @ GRID_W


def weighted_quantile(x, w, taus) -> np.ndarray:
    """Quantiles of x with sampling weights w (Hazen-type interpolation)."""
    o = np.argsort(x)
    x, w = np.asarray(x, float)[o], np.asarray(w, float)[o]
    c = (np.cumsum(w) - 0.5 * w) / w.sum()
    return np.interp(taus, c, x)


def centered_quantiles(z, w=None) -> np.ndarray:
    """Empirical quantiles of z minus its sample mean (scale kept, unlike
    ``models.base.empirical_z_quantiles`` which also rescales to unit sd);
    optional observation weights."""
    z = np.asarray(z, float)
    ok = np.isfinite(z)
    if w is None:
        z = z[ok]
        return np.quantile(z - z.mean(), TAUS)
    z, w = z[ok], np.asarray(w, float)[ok]
    return weighted_quantile(z - np.average(z, weights=w), w, TAUS)


# --------------------------------------------------------------------------
# Constants of the design (fixed a priori; see the module docstring)
# --------------------------------------------------------------------------
MODEL = "dist_hm_af"
MODEL_ASSETSHAPE = "dist_hm_af.assetshape"     # portfolios with the asset-pooled shape
HORIZON = 12
MIN_RESID_DATES = 120
TAIL_FLOOR = ("q01", "q05", "q95", "q99")
_QCOLS = list(schema.QUANTILE_COLUMNS)
_FLOOR_IDX = np.array([_QCOLS.index(c) for c in TAIL_FLOOR])
_NORM = stats.norm.ppf(TAUS)
KINDS = ("assets", "portfolios")


def _policies(policies=None) -> dict:
    if policies is None:
        from ..eval.portfolio import POLICIES
        return POLICIES
    return policies


def _policy_weights(assets: list[str], spec) -> np.ndarray | None:
    from ..eval.portfolio import policy_weights
    return policy_weights(assets, spec)


def _month_end(t) -> pd.Timestamp:
    return pd.Timestamp(t) + pd.offsets.MonthEnd(0)


def _cutoff(origin, horizon: int = HORIZON) -> pd.Timestamp:
    """Last residual origin s whose window has closed by ``origin``: s + horizon months <= origin."""
    return _month_end(origin) - pd.offsets.MonthEnd(horizon)


def _mean_tables(xr: pd.DataFrame, expected: pd.DataFrame | None):
    """Wide (origin x asset) tables of mean_12m and se_12m from an expected-return history."""
    if expected is None:
        expected = _exp.history(xr)
    m = expected.pivot(index="origin", columns="asset", values="mean_12m")
    se = expected.pivot(index="origin", columns="asset", values="se_12m")
    cols = [a for a in xr.columns if a in m.columns]
    return m.reindex(columns=cols), se.reindex(columns=cols)


def _origins(cov_hist: dict, origins) -> list[pd.Timestamp]:
    keys = sorted(pd.Timestamp(k) for k in cov_hist)
    if origins is None:
        return keys
    want = {pd.Timestamp(o) for o in origins}
    return [k for k in keys if k in want]


# --------------------------------------------------------------------------
# Residuals
# --------------------------------------------------------------------------
def residual_frame(xr: pd.DataFrame, cov_hist: dict, origins=None, expected=None,
                   horizon: int = HORIZON) -> pd.DataFrame:
    """Per (origin, asset): ``y`` (realized 12-month log excess return, NaN if
    not yet closed), ``mean``, ``se``, ``scale`` = sqrt(Sigma_ii), ``pscale`` =
    sqrt(Sigma_ii + se^2) and ``z`` = (y - mean) / pscale.

    Rows: origins in ``cov_hist`` (optionally restricted to ``origins``) and
    assets with both a covariance forecast and an eligible historical mean."""
    m, se = _mean_tables(xr, expected)
    y = schema.make_targets(xr, horizon)
    recs = []
    for t in _origins(cov_hist, origins):
        if t not in m.index:
            continue
        C = cov_hist[t]
        mt, st = m.loc[t], se.loc[t]
        yt = y.loc[t] if t in y.index else None
        diag = pd.Series(np.diag(C.to_numpy(float)), index=C.index)
        for a in C.index:
            if a not in mt.index or not np.isfinite(mt[a]):
                continue
            v = float(diag[a])
            ps = float(np.sqrt(v + st[a] ** 2))
            ya = float(yt[a]) if yt is not None and a in yt.index else np.nan
            recs.append((t, a, ya, float(mt[a]), float(st[a]), float(np.sqrt(v)), ps,
                         (ya - mt[a]) / ps))
    df = pd.DataFrame(recs, columns=["origin", "asset", "y", "mean", "se", "scale", "pscale", "z"])
    return df.set_index(["origin", "asset"])


def standardized_residuals(xr: pd.DataFrame, cov_hist: dict, origins=None, expected=None,
                           horizon: int = HORIZON) -> pd.Series:
    """z = (y - mean_s) / sqrt(Sigma_s,ii + se_s^2) per (origin, asset), realized rows only.

    ``xr``: wide monthly log excess returns of the return panel
    (``inputs.excess_returns``); ``cov_hist``: {origin: asset x asset 12-month
    covariance} (``covhistory.load(kind="annual")``); ``expected``: an
    ``expected.history`` frame (computed if omitted)."""
    f = residual_frame(xr, cov_hist, origins, expected, horizon)
    return f["z"].dropna().rename("z")


def portfolio_frame(xr: pd.DataFrame, cov_hist: dict, origins=None, policies=None,
                    expected=None, horizon: int = HORIZON) -> pd.DataFrame:
    """Per (origin, portfolio): ``y`` = w'y, ``mean`` = w'm, ``se`` = sum|w| se
    (an upper bound: mean errors treated as perfectly correlated), ``scale`` =
    sqrt(w' Sigma w), ``pscale``, ``z`` and ``weights`` (JSON).

    Universe at each origin: assets in the covariance forecast with an eligible
    historical mean, in the forecast's order; weights are the policy mix
    renormalized over it (``eval.portfolio.policy_weights``; a policy is
    skipped when one of its required assets is missing). y is NaN unless every
    asset in the universe has a realized outcome."""
    policies = _policies(policies)
    m, se = _mean_tables(xr, expected)
    y = schema.make_targets(xr, horizon)
    recs = []
    for t in _origins(cov_hist, origins):
        if t not in m.index:
            continue
        C = cov_hist[t]
        mt, st = m.loc[t], se.loc[t]
        U = [a for a in C.index if a in mt.index and np.isfinite(mt[a])]
        if not U:
            continue
        S = C.loc[U, U].to_numpy(float)
        yt = y.loc[t, U].to_numpy(float) if t in y.index else np.full(len(U), np.nan)
        mu, sev = mt[U].to_numpy(float), st[U].to_numpy(float)
        for name, spec in policies.items():
            w = _policy_weights(U, spec)
            if w is None:
                continue
            v = float(w @ S @ w)
            sep = float(np.abs(w) @ sev)
            ps = float(np.sqrt(v + sep ** 2))
            yp = float(w @ yt) if np.isfinite(yt).all() else np.nan
            mp = float(w @ mu)
            wj = json.dumps({a: round(float(x), 6) for a, x in zip(U, w) if x != 0})
            recs.append((t, name, yp, mp, sep, float(np.sqrt(v)), ps, (yp - mp) / ps, wj))
    df = pd.DataFrame(recs, columns=["origin", "portfolio", "y", "mean", "se", "scale", "pscale",
                                     "z", "weights"])
    return df.set_index(["origin", "portfolio"])


def portfolio_residuals(xr: pd.DataFrame, cov_hist: dict, origins=None, policies=None,
                        expected=None, horizon: int = HORIZON) -> pd.Series:
    """z = (w'y - w'm) / sqrt(w'Sigma w + se_p^2) per (origin, portfolio), realized rows only
    (see ``portfolio_frame``)."""
    f = portfolio_frame(xr, cov_hist, origins, policies, expected, horizon)
    return f["z"].dropna().rename("z")


# --------------------------------------------------------------------------
# Shape
# --------------------------------------------------------------------------
def _neff(values: np.ndarray, dates: np.ndarray, horizon: int) -> float:
    """iid-equivalent count of rows for the mean of ``values`` (pooled rows, overlapping windows).

    Var(overall mean) is estimated from the long-run variance (EWC, as for the
    expected returns) of the per-date sums; n_eff = var(row) N^2 / (n_dates LRV)."""
    from ..eval.inference import long_run_variance
    N = len(values)
    if N < 3:
        return float(N)
    s = pd.Series(values).groupby(dates).sum()
    s.index = pd.DatetimeIndex(s.index)
    full = s.reindex(pd.date_range(s.index[0], s.index[-1], freq="ME"), fill_value=0.0)
    if len(full) < 8:
        return float(N)
    r = long_run_variance(full, kernel="ewc", h=horizon)
    v = float(np.var(values, ddof=1))
    if not (r.lrv > 0 and v > 0):
        return float(N)
    return float(min(N, v * N ** 2 / (len(full) * r.lrv)))


def _episodes(dates: np.ndarray, horizon: int) -> int:
    """Number of separate episodes among hit dates: a gap of more than ``horizon``
    months starts a new one (windows less than a year apart overlap)."""
    if len(dates) == 0:
        return 0
    d = np.unique(pd.DatetimeIndex(dates).to_period("M").astype("int64"))
    return int(1 + np.sum(np.diff(d) > horizon))


def shape_info(z: pd.Series, origin, horizon: int = HORIZON,
               min_dates: int = MIN_RESID_DATES, diagnostics: bool = True) -> dict:
    """The shape at ``origin`` and what stands behind it.

    ``z``: Series indexed (origin, name) of standardized residuals. Uses rows
    with ``s + horizon months <= origin``. Keys: ``q`` (the shape used, 9
    quantiles), ``q_raw`` (centred empirical quantiles), ``q_gauss`` (the
    floor), ``mean`` (removed), ``sd`` (of the centred z), ``n_rows``,
    ``n_dates``, ``n_series``, ``first_origin``, ``last_origin`` and, with
    ``diagnostics``: ``n_eff`` (for the mean of z), ``n_eff_q01``,
    ``n_eff_q05`` (for the lower-tail indicators) and ``episodes_q01``,
    ``episodes_q05`` (separate episodes with a row below that quantile). ``q``
    is all NaN when fewer than ``min_dates`` residual origins exist.
    """
    cut = _cutoff(origin, horizon)
    d = z.index.get_level_values(0)
    sub = z[(d <= cut) & np.isfinite(z.to_numpy(float))]
    dates = np.asarray(sub.index.get_level_values(0))
    nd = len(np.unique(dates))
    info = {"origin": _month_end(origin), "n_rows": int(len(sub)), "n_dates": int(nd),
            "n_series": int(sub.index.get_level_values(1).nunique()),
            "first_origin": pd.Timestamp(dates.min()) if nd else pd.NaT,
            "last_origin": pd.Timestamp(dates.max()) if nd else pd.NaT}
    nan9 = np.full(K, np.nan)
    if nd < min_dates:
        info.update(q=nan9, q_raw=nan9, q_gauss=nan9, mean=np.nan, sd=np.nan, n_eff=np.nan,
                    n_eff_q01=np.nan, n_eff_q05=np.nan, episodes_q01=0, episodes_q05=0)
        return info
    v = sub.to_numpy(float)
    mu = float(v.mean())
    zc = v - mu
    q_raw = centered_quantiles(v)
    sd = float(zc.std(ddof=1))
    q_gauss = sd * _NORM
    q = q_raw.copy()
    lower = _FLOOR_IDX[TAUS[_FLOOR_IDX] < 0.5]
    upper = _FLOOR_IDX[TAUS[_FLOOR_IDX] > 0.5]
    q[lower] = np.minimum(q[lower], q_gauss[lower])
    q[upper] = np.maximum(q[upper], q_gauss[upper])
    q = np.maximum.accumulate(q)
    info.update(q=q, q_raw=q_raw, q_gauss=q_gauss, mean=mu, sd=sd)
    if diagnostics:
        info["n_eff"] = _neff(zc, dates, horizon)
        for c, tau in (("q01", 0.01), ("q05", 0.05)):
            hit = zc <= q_raw[_QCOLS.index(c)]
            info[f"n_eff_{c}"] = _neff(hit.astype(float), dates, horizon)
            info[f"episodes_{c}"] = _episodes(dates[hit], horizon)
    return info


def pooled_shape(z: pd.Series, origin, horizon: int = HORIZON) -> np.ndarray:
    """The 9 shape quantiles at ``origin`` from residuals with ``s + horizon months <= origin``
    (recentred, Gaussian tail floor; all NaN before ``MIN_RESID_DATES``). See ``shape_info``."""
    return shape_info(z, origin, horizon, diagnostics=False)["q"]


def _shape_q_sd(shape) -> tuple[np.ndarray, float]:
    """(quantiles, sd) from a ``shape_info`` dict or a bare quantile array."""
    if isinstance(shape, dict):
        return np.asarray(shape["q"], float), float(shape["sd"])
    q = np.asarray(shape, float)
    return q, float(np.sqrt(GRID_W @ (q - grid_mean(q)) ** 2))   # rough, grid-based


# --------------------------------------------------------------------------
# Forecasts
# --------------------------------------------------------------------------
def _rows(frame: pd.DataFrame, shape, name_col: str) -> pd.DataFrame:
    """Forecast rows from a residual/portfolio frame at one origin and a shape."""
    q, sdz = _shape_q_sd(shape)
    if not np.isfinite(q).all() or frame.empty:
        return pd.DataFrame(columns=[name_col, "mean", "scale", "sd", *_QCOLS])
    ps = frame["pscale"].to_numpy(float)
    mean = frame["mean"].to_numpy(float)
    Q = mean[:, None] + ps[:, None] * q[None, :]
    out = pd.DataFrame(Q, columns=_QCOLS)
    out.insert(0, "sd", ps * sdz)
    out.insert(0, "scale", frame["scale"].to_numpy(float))
    out.insert(0, "mean", mean)
    out.insert(0, name_col, list(frame.index.get_level_values(-1)))
    return out


def distribution_at(origin, panel, cov_annual: pd.DataFrame, shapes: dict, policies=None,
                    expected: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The published distribution tables at one origin.

    ``panel``: the return-side panel (or a wide xr frame); ``cov_annual``: the
    12-month covariance forecast at ``origin`` (asset x asset); ``shapes``:
    {"assets": shape, "portfolios": shape} where a shape is a ``shape_info``
    dict (preferred: it carries the sd) or a 9-quantile array; a missing
    "portfolios" entry falls back to the asset shape (that is how the
    sensitivity check is produced). ``expected``: optional
    ``expected.history`` frame covering ``origin``.

    Returns ``(assets_df, portfolios_df)``: columns ``asset, mean, scale, sd,
    q01..q99`` and ``portfolio, mean, scale, sd, q01..q99, weights``. ``mean``
    is the historical mean (mean_12m), ``scale`` sqrt of the covariance
    forecast's variance, ``sd`` the predictive sd (sqrt(scale^2 + se^2) times
    the shape's sd), ``weights`` a JSON object.
    """
    from .inputs import excess_returns
    t = _month_end(origin)
    xr = panel if isinstance(panel, pd.DataFrame) else excess_returns(panel)
    xr = xr.loc[:t]
    if expected is None:
        e = _exp.expected_from_xr(xr, t).reset_index()
        e.insert(0, "origin", t)
        expected = e
    else:
        expected = expected[expected["origin"] == t]
    cov_hist = {t: cov_annual}
    fa = residual_frame(xr, cov_hist, None, expected)
    fp = portfolio_frame(xr, cov_hist, None, policies, expected)
    assets_df = _rows(fa, shapes["assets"], "asset")
    port_df = _rows(fp, shapes.get("portfolios", shapes["assets"]), "portfolio")
    if not port_df.empty:
        port_df["weights"] = fp["weights"].to_numpy()
    return assets_df.reset_index(drop=True), port_df.reset_index(drop=True)


def _forecast_frame(rows: pd.DataFrame, name_col: str, model: str) -> pd.DataFrame:
    f = rows.rename(columns={name_col: "asset"})
    f.insert(0, "horizon", HORIZON)
    f.insert(0, "model", model)
    return f[schema.FORECAST_COLUMNS]


def build(panel, cov_hist: dict, policies=None, expected: pd.DataFrame | None = None) -> dict:
    """Everything at every origin of ``cov_hist``.

    Returns a dict with ``assets`` (``FORECAST_COLUMNS``, model ``dist_hm_af``),
    ``portfolios`` (``FORECAST_COLUMNS`` with the portfolio name in ``asset``;
    models ``dist_hm_af`` and ``dist_hm_af.assetshape``), ``residuals``
    (asset and portfolio residual frames, long), ``shapes`` (shape history,
    long: one row per origin, kind and quantile) and ``expected`` (the
    expected-return history used).
    """
    from .inputs import excess_returns
    xr = panel if isinstance(panel, pd.DataFrame) else excess_returns(panel)
    last = xr.index.max()
    cov_hist = {pd.Timestamp(k): v for k, v in cov_hist.items() if pd.Timestamp(k) <= last}
    if expected is None:
        expected = _exp.history(xr)
    fa = residual_frame(xr, cov_hist, None, expected)
    fp = portfolio_frame(xr, cov_hist, None, policies, expected)
    za, zp = fa["z"].dropna(), fp["z"].dropna()
    out_a, out_p, out_ps, shape_rows = [], [], [], []
    origins = sorted(set(fa.index.get_level_values(0)) | set(fp.index.get_level_values(0)))
    fa_by = dict(tuple(fa.groupby(level=0)))
    fp_by = dict(tuple(fp.groupby(level=0)))

    def add(out, frame, shape, name_col, t):
        r = _rows(frame, shape, name_col)
        if not r.empty:
            r.insert(0, "date", t)
            out.append(r)

    for t in origins:
        sa = shape_info(za, t)
        sp = shape_info(zp, t)
        for kind, s in (("assets", sa), ("portfolios", sp)):
            for k, c in enumerate(_QCOLS):
                shape_rows.append({
                    "origin": t, "kind": kind, "tau": TAUS[k], "q": c, "z": s["q"][k],
                    "z_raw": s["q_raw"][k], "z_gauss": s["q_gauss"][k], "mean_removed": s["mean"],
                    "sd": s["sd"], "n_rows": s["n_rows"], "n_dates": s["n_dates"],
                    "n_series": s["n_series"], "n_eff": s["n_eff"], "n_eff_q01": s["n_eff_q01"],
                    "n_eff_q05": s["n_eff_q05"], "episodes_q01": s["episodes_q01"],
                    "episodes_q05": s["episodes_q05"], "first_origin": s["first_origin"],
                    "last_origin": s["last_origin"]})
        if t in fa_by:
            add(out_a, fa_by[t], sa, "asset", t)
        if t in fp_by:
            add(out_p, fp_by[t], sp, "portfolio", t)
            add(out_ps, fp_by[t], sa, "portfolio", t)

    def cat(frames, name_col, model):
        if not frames:
            return pd.DataFrame(columns=schema.FORECAST_COLUMNS)
        return _forecast_frame(pd.concat(frames, ignore_index=True), name_col, model)

    assets = cat(out_a, "asset", MODEL)
    ports = pd.concat([cat(out_p, "portfolio", MODEL), cat(out_ps, "portfolio", MODEL_ASSETSHAPE)],
                      ignore_index=True)
    resid = pd.concat([
        fa.reset_index().rename(columns={"asset": "name"}).assign(kind="assets"),
        fp.drop(columns="weights").reset_index().rename(columns={"portfolio": "name"})
          .assign(kind="portfolios")], ignore_index=True)
    resid = resid[["origin", "kind", "name", "y", "mean", "se", "scale", "pscale", "z"]]
    for f in (assets, ports):
        f["date"] = pd.to_datetime(f["date"])
        f["horizon"] = f["horizon"].astype(int)
    return {"assets": assets.reset_index(drop=True), "portfolios": ports.reset_index(drop=True),
            "residuals": resid, "shapes": pd.DataFrame(shape_rows), "expected": expected,
            "weights": fp["weights"]}


def history(panel, cov_hist: dict, policies=None, expected=None) -> pd.DataFrame:
    """The asset forecasts at every origin (``FORECAST_COLUMNS``, model ``dist_hm_af``).
    ``build`` returns the portfolio forecasts and diagnostics as well."""
    return build(panel, cov_hist, policies, expected)["assets"]


# --------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------
def _meta(result: dict, cov_source: str) -> dict:
    sh = result["shapes"]
    return {
        "label": config.REPLICATION_LABEL,
        "note": ("Designed after both holdouts had been seen: every score of this model is "
                 "descriptive, not out-of-sample evidence. Only origins from "
                 f"{config.OOS_ORIGIN_START} on (targets wholly after FROZEN_AT "
                 f"{config.FROZEN_AT}) can be evidence."),
        "model": MODEL, "horizon": HORIZON, "covariance": cov_source,
        "location": "expanding historical mean (= hist_mean), returns.expected",
        "scale": "sqrt(Sigma_ii + se^2); portfolios sqrt(w'Sigma w + (sum|w| se)^2)",
        "shape": ("pooled z = (y - m_s)/scale_s over windows closed by t; recentred, spread kept; "
                  f"tails {list(TAIL_FLOOR)} floored at N(0, sd_z^2); "
                  f"no forecast before {MIN_RESID_DATES} residual origins"),
        "portfolio_log_return": "w'r (daily-rebalanced approximation, as eval.portfolio)",
        "first_origin_assets": str(result["assets"]["date"].min().date()) if len(result["assets"]) else None,
        "last_origin": str(result["assets"]["date"].max().date()) if len(result["assets"]) else None,
        "n_shape_origins": int(sh["origin"].nunique()) if len(sh) else 0,
    }


def write(result: dict, root=None, expected_root=None, cov_source: str = "covhistory annual") -> Path:
    """Write ``build`` output under ``root`` (default ``RESULTS_RETURNS/distribution``)
    and the expected-return history under ``expected_root``."""
    if root is None or expected_root is None:
        from .. import paths
        root = root or paths.RESULTS_RETURNS / "distribution"
        expected_root = expected_root or paths.RESULTS_RETURNS / "expected"
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    _exp.write_history(result["expected"], expected_root)
    result["assets"].to_parquet(root / "history.parquet", index=False)
    result["portfolios"].to_parquet(root / "history_portfolios.parquet", index=False)
    result["shapes"].to_parquet(root / "shape_history.parquet", index=False)
    result["residuals"].to_parquet(root / "residuals.parquet", index=False)
    (root / "meta.json").write_text(json.dumps(_meta(result, cov_source), indent=1, default=str))
    return root / "history.parquet"


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------
def portfolio_hist_mean_baseline(port_hist: pd.DataFrame, residuals: pd.DataFrame,
                                 min_rows: int = config.MIN_ASSET_HISTORY) -> pd.DataFrame:
    """The ``hist_mean`` analogue for policy portfolios, for scoring only.

    Same mean as ``dist_hm_af`` (w'm); Gaussian with sd = sqrt(var of the
    portfolio's own realized 12-month outcomes at origins s with s + 12 <= t +
    se_p^2), i.e. a static scale instead of the covariance forecast. Needs
    ``min_rows`` closed outcomes."""
    f = port_hist[port_hist["model"] == MODEL]
    r = residuals[residuals["kind"] == "portfolios"]
    out = []
    for p, g in f.groupby("asset"):
        rr = r[r["name"] == p].set_index("origin").sort_index()
        yv = rr["y"]
        for _, row in g.iterrows():
            t = row["date"]
            past = yv[(yv.index <= _cutoff(t)) & yv.notna()]
            if len(past) < min_rows:
                continue
            se = float(rr.at[t, "se"]) if t in rr.index else 0.0
            sd = float(np.sqrt(past.var(ddof=1) + se ** 2))
            out.append({"model": "hist_mean", "horizon": HORIZON, "date": t, "asset": p,
                        "mean": row["mean"], "sd": sd,
                        **{c: row["mean"] + sd * zq for c, zq in zip(_QCOLS, _NORM)}})
    return pd.DataFrame(out, columns=schema.FORECAST_COLUMNS)


def score(history: pd.DataFrame, baseline_forecasts: pd.DataFrame, xr: pd.DataFrame | None = None,
          portfolios: pd.DataFrame | None = None, residuals: pd.DataFrame | None = None,
          mt_reps: int = 999) -> pd.DataFrame:
    """Score ``dist_hm_af`` (descriptively; see ``REPLICATION_LABEL``).

    Assets: ``history`` together with ``baseline_forecasts`` (the replication's
    ``hist_mean`` and ``vol_ewma`` at h=12, ``FORECAST_COLUMNS``) through
    ``scoring.score_forecasts`` on the common (origin, asset) rows, by
    ``config.PERIODS``, with the multiple-testing rows for ``config.MT_PERIODS``.
    Portfolios (when ``portfolios`` and ``residuals`` from ``build`` are
    given): ``dist_hm_af`` and ``dist_hm_af.assetshape`` against the portfolio
    ``hist_mean`` analogue (``portfolio_hist_mean_baseline``), with
    ``targets=`` the realized portfolio outcomes; asset ids in the output are
    the portfolio names. ``xr`` defaults to the return panel's excess returns.

    Returns ``SCORE_COLUMNS`` plus ``target`` ("assets" / "portfolios").
    """
    from .evaluation import scoring
    if xr is None:
        from .inputs import excess_returns, load_return_panel
        xr = excess_returns(load_return_panel())
    base = baseline_forecasts[baseline_forecasts["horizon"] == HORIZON]
    fc = pd.concat([history[schema.FORECAST_COLUMNS], base[schema.FORECAST_COLUMNS]],
                   ignore_index=True)
    sa, _ = scoring.score_forecasts(fc, xr, subperiods=config.subperiods(),
                                    mt_periods=config.MT_PERIODS, mt_reps=mt_reps)
    frames = [sa.assign(target="assets")]
    if portfolios is not None and residuals is not None:
        bl = portfolio_hist_mean_baseline(portfolios, residuals)
        r = residuals[residuals["kind"] == "portfolios"]
        tg = r.set_index(["origin", "name"])["y"].dropna()
        tg.index = tg.index.set_names(["date", "asset"])
        fcp = pd.concat([portfolios[schema.FORECAST_COLUMNS], bl], ignore_index=True)
        sp, _ = scoring.score_forecasts(fcp, None, benchmarks=("hist_mean",),
                                        subperiods=config.subperiods(),
                                        mt_periods=config.MT_PERIODS, mt_reps=mt_reps, targets=tg)
        frames.append(sp.assign(target="portfolios"))
    out = pd.concat(frames, ignore_index=True)
    out.attrs["label"] = config.REPLICATION_LABEL
    return out


def write_scores(scores: pd.DataFrame, root=None) -> Path:
    if root is None:
        from .. import paths
        root = paths.RESULTS_RETURNS / "distribution"
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    scores.to_parquet(root / "scores.parquet", index=False)
    meta = {"label": config.REPLICATION_LABEL,
            "note": ("dist_hm_af was designed after both holdouts had been seen; these scores "
                     "describe it, they are not out-of-sample evidence. Portfolio rows compare "
                     "against a portfolio hist_mean analogue (static Gaussian scale)."),
            "periods": config.PERIODS, "mt_periods": list(config.MT_PERIODS)}
    (root / "scores.json").write_text(json.dumps(meta, indent=1))
    return root / "scores.parquet"


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def _roots(max_date: str | None) -> tuple[Path, Path]:
    """(RESULTS_RETURNS, COV_HISTORY) for a full run or a --max-date run, without
    relying on the environment variable having been set before import."""
    root = Path(__file__).resolve().parents[3] / "results"
    if max_date:
        return root / "dev" / "returns", root / "dev" / "covariance" / "cov_history.parquet"
    return root / "returns", root / "covariance" / "cov_history.parquet"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--max-date", default=None, help="truncate the panel (development run)")
    ap.add_argument("--score", action="store_true",
                    help="also score against results/returns/forecasts/{hist_mean,vol_ewma}/h12.parquet")
    ap.add_argument("--reps", type=int, default=999, help="bootstrap reps for SPA/MCS")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    from .. import covhistory
    from .inputs import excess_returns, load_return_panel
    ret_root, cov_path = _roots(args.max_date)
    panel = load_return_panel(max_date=args.max_date)
    xr = excess_returns(panel)
    cov_hist = covhistory.load(kind="annual", path=cov_path)
    res = build(xr, cov_hist)
    p = write(res, ret_root / "distribution", ret_root / "expected", cov_source=str(cov_path))
    log.info("wrote %s (%d asset rows, %d portfolio rows)", p, len(res["assets"]), len(res["portfolios"]))
    if args.score:
        base = pd.concat([pd.read_parquet(ret_root / "forecasts" / m / "h12.parquet")
                          for m in ("hist_mean", "vol_ewma")], ignore_index=True)
        sc = score(res["assets"], base, xr, res["portfolios"], res["residuals"], mt_reps=args.reps)
        log.info("wrote %s", write_scores(sc, ret_root / "distribution"))


if __name__ == "__main__":
    main()
