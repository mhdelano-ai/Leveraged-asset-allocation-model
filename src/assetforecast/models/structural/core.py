"""Shared building blocks of the structural family.

Everything here works on MONTHLY data aligned to ``view.monthly.index``:

* observable exposures - par-bond durations computed from today's yields
  (:func:`par_duration`), ETF-adjusted where the panel's asset is an ETF;
* "normalized" return histories - a bond's monthly log excess return divided
  by its duration at the start of the month is (minus) the yield change that
  moved it, in decimal yield units; dividing further by ``y ** beta`` gives
  the shock of a CEV yield process (beta = 0 normal, 1 lognormal);
* window variance estimators identical in form to the reference models
  (``12 * mean(x ** 2)``, uncentered), so a structural forecast differs from
  its reference counterpart only through the structure.

Monthly returns are used throughout (never 1-day returns): a squared calendar-
month return is one draw of the 21-day-horizon variance the target measures,
which sidesteps the strong autocorrelation of pre-1990 daily yield changes
(reports/data/treasuries.md) without any correction.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ------------------------------------------------------------------ constants
# Ladders exactly as the panel builds the Treasury assets (reports/data/treasuries.md)
MAT_INT = np.arange(7.0, 10.01, 0.5)          # ust_int: 7, 7.5, ..., 10y par bonds
MAT_LONG = np.arange(20.0, 30.01, 1.0)        # ust_long: 20, 21, ..., 30y par bonds
# TIPS: outstanding issues span ~1-30y with a market-weighted average maturity of
# ~8-9y; a 2-15y par ladder has average maturity 8.5y and duration 7.1-8.1 for real
# yields of 0-3.5%, in line with the ETF's stated effective duration (~7.5).
MAT_TIPS = np.arange(2.0, 15.01, 0.5)
# Credit: LQD's average maturity is ~12y (the panel's ig_credit is vol-matched to LQD
# in every era); a high-yield index's ~6y. Only the yield-level dependence matters:
# the rate loading is estimated (see credit_components), absorbing the level.
MAT_IG = np.array([12.0])
MAT_HY = np.array([6.0])

YIELD_FLOOR = 0.01       # CEV/lognormal scaling never uses a yield below 1%
SPREAD_FLOOR = 0.0025    # spread levels floored at 25bp for DTS scaling
RECENT = 60              # recent window of the blend, as in ref_blend
MIN_OBS = 12             # as in ref: a window needs 12 months to be trusted
STALE_MONTHS = 3         # a monthly predictor older than this is treated as dead


def par_duration(y, maturities) -> np.ndarray:
    """Modified duration of an equal-weight ladder of semiannual par bonds.

    For a par bond of maturity M at yield y: D = (1 - (1 + y/2)^(-2M)) / y
    (limit M as y -> 0). Reproduces the panel's ``ust_ladder_duration_*``
    predictors to within 0.5% (median 0.0%).
    """
    y = np.asarray(y, float)
    m = np.asarray(maturities, float)
    yy = y[..., None]
    small = np.abs(yy) < 1e-6
    yy = np.where(small, 1e-6, yy)
    d = (1.0 - (1.0 + yy / 2.0) ** (-2.0 * m)) / yy
    return d.mean(-1)


def blend_var(x: pd.Series, recent: int = RECENT, how: str = "blend") -> float:
    """Annualized uncentered variance of a monthly series, as the reference models.

    ``how``: "full" (all history), "recent" (last ``recent`` months, falling back
    to all history when thin) or "blend" (geometric mean of the two, = ref_blend).
    """
    x = x.dropna() if isinstance(x, pd.Series) else pd.Series(x).dropna()
    if len(x) == 0:
        return np.nan
    full = 12.0 * float(np.mean(x.to_numpy() ** 2))
    if how == "full":
        return full
    w = x.iloc[-recent:]
    rec = 12.0 * float(np.mean(w.to_numpy() ** 2)) if len(w) >= min(MIN_OBS, recent) else full
    if how == "recent":
        return rec
    if full <= 0 or rec <= 0:
        return max(full, rec)
    return float(np.sqrt(full * rec))


# ---------------------------------------------------------------- predictors
_SERIES = [
    "macro_long_yield",
    "ust_cmt_yield_10y", "ust_cmt_yield_20y", "ust_cmt_yield_30y",
    "ust_ladder_duration_int", "ust_ladder_duration_long",
    "ust_etf_duration_est_int", "ust_etf_duration_est_long",
    "tips_real_yield_10y",
    "credit_moodys_baa_ust10y_spread_m", "credit_moodys_baa_yield_m", "credit_gz_spread_m",
]


def monthly_predictors(view) -> pd.DataFrame:
    """Last published value in each calendar month of the series this family uses,
    on ``view.monthly.index`` (all-NaN columns for series the view lacks)."""
    idx = view.monthly.index
    w = view.predictors_wide()
    cols = [c for c in _SERIES if c in w.columns]
    if not cols or len(w) == 0:
        return pd.DataFrame(np.nan, index=idx, columns=_SERIES)
    w = w[cols].dropna(how="all")
    m = w.groupby(w.index.to_period("M")).last()
    m.index = m.index.to_timestamp(how="end").normalize()
    m = m.reindex(idx)
    return m.reindex(columns=_SERIES)


def _fill_tail(s: pd.Series, limit: int = STALE_MONTHS) -> pd.Series:
    return s.ffill(limit=limit)


def yield_levels(pm: pd.DataFrame) -> pd.DataFrame:
    """Month-end yield levels used for durations and CEV scaling.

    y_int:  10y CMT (daily from 1962), else the monthly long yield (10y CMT
            monthly average after 1953, long Treasury bonds before).
    y_long: 20y CMT, else 30y, else 10y, else the monthly long yield.
    """
    ml = _fill_tail(pm["macro_long_yield"])
    c10 = _fill_tail(pm["ust_cmt_yield_10y"])
    c20 = _fill_tail(pm["ust_cmt_yield_20y"])
    c30 = _fill_tail(pm["ust_cmt_yield_30y"])
    y_int = c10.combine_first(ml)
    y_long = c20.combine_first(c30).combine_first(c10).combine_first(ml)
    return pd.DataFrame({"int": y_int, "long": y_long})


def etf_ratio(pm: pd.DataFrame, which: str) -> pd.Series:
    """Effective-duration correction for the ETF era: trailing 12-month mean of
    (regression-estimated ETF duration / par-ladder duration), clipped to
    [0.75, 1.25]; 1 where no estimate exists (before 2003-08)."""
    est = pm[f"ust_etf_duration_est_{which}"]
    lad = pm[f"ust_ladder_duration_{which}"]
    r = (est / lad).where(lambda s: np.isfinite(s) & (s > 0))
    r = r.rolling(12, min_periods=1).mean()
    r = _fill_tail(r)
    return r.clip(0.75, 1.25).fillna(1.0)


def treasury_durations(pm: pd.DataFrame, etf_adjust: bool = True) -> pd.DataFrame:
    y = yield_levels(pm)
    d = pd.DataFrame({"int": par_duration(y["int"].to_numpy(), MAT_INT),
                      "long": par_duration(y["long"].to_numpy(), MAT_LONG)}, index=pm.index)
    d = d.where(y.notna())
    if etf_adjust:
        d["int"] *= etf_ratio(pm, "int")
        d["long"] *= etf_ratio(pm, "long")
    return d


def cev_scale(y: pd.Series, beta: float) -> pd.Series:
    """y ** beta with the yield floored at YIELD_FLOOR (beta = 0 gives 1)."""
    if beta == 0:
        return pd.Series(1.0, index=y.index).where(y.notna())
    return np.maximum(y, YIELD_FLOOR) ** beta


def spread_level(pm: pd.DataFrame) -> pd.Series:
    """Investment-grade spread level (decimal), monthly.

    Moody's Baa minus the 10y CMT (monthly average, 1953-), before 1953 Moody's
    Baa yield minus the long Treasury yield. Where those are no longer published
    (Moody's ends 2025-12) the Gilchrist-Zakrajsek spread (live) is rescaled to
    Baa-10y units by the median ratio of the two over their last 60 common
    months. Floored at SPREAD_FLOOR.
    """
    s1 = pm["credit_moodys_baa_ust10y_spread_m"]
    s0 = pm["credit_moodys_baa_yield_m"] - pm["macro_long_yield"]
    s = s1.combine_first(s0)
    gz = pm["credit_gz_spread_m"]
    last = s.last_valid_index()
    if gz.notna().any() and (last is None or last < s.index[-1]):
        common = (s / gz).where(np.isfinite(s / gz) & (gz > 0)).dropna()
        ratio = float(common.iloc[-60:].median()) if len(common) >= 12 else np.nan
        if np.isfinite(ratio):
            tail = (gz * ratio)
            s = s.combine_first(tail if last is None else tail.where(tail.index > last))
    s = _fill_tail(s)
    return s.clip(lower=SPREAD_FLOOR)


# ------------------------------------------------------------- CEV exponent
def estimate_beta(u: dict[str, pd.Series], y: dict[str, pd.Series], asof: pd.Timestamp,
                  prior: float = 0.5, min_years: int = 15) -> float:
    """Walk-forward elasticity of basis-point yield volatility to the yield level.

    Uses calendar years that ENDED before the origin's year (re-estimated once a
    year): for each asset and complete year, log mean(u^2) (u = monthly yield
    change implied by returns) is regressed on log y at the previous year-end,
    with an asset fixed effect. beta = slope / 2, clipped to [0, 1.5]. Fewer
    than ``min_years`` asset-years: ``prior``.
    """
    end = pd.Timestamp(year=pd.Timestamp(asof).year - 1, month=12, day=31)
    rows = []
    for k, uk in u.items():
        uk = uk.loc[:end].dropna()
        if uk.empty:
            continue
        yk = y[k]
        g = uk.groupby(uk.index.year)
        for yr, x in g:
            if len(x) < 12:
                continue
            y0 = yk.get(pd.Timestamp(year=yr - 1, month=12, day=31), np.nan)
            v = float(np.mean(x.to_numpy() ** 2))
            if np.isfinite(y0) and v > 0:
                rows.append((k, np.log(max(y0, YIELD_FLOOR)), np.log(v)))
    if len(rows) < min_years:
        return prior
    d = pd.DataFrame(rows, columns=["k", "x", "v"])
    # within-asset demeaning = asset fixed effects
    xd = d["x"] - d.groupby("k")["x"].transform("mean")
    vd = d["v"] - d.groupby("k")["v"].transform("mean")
    den = float((xd ** 2).sum())
    if den <= 0:
        return prior
    return float(np.clip(0.5 * (xd * vd).sum() / den, 0.0, 1.5))
