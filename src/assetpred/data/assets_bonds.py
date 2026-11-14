"""Constant-maturity Treasury and TIPS total returns from the Fed's fitted curves.

Method (Gurkaynak-Sack-Wright nominal curve, feds200628; Gurkaynak-Sack-Wright
TIPS curve, feds200805):

* At each rebalance date t-1 (the last curve observation of a month for the
  monthly panel, every curve observation for the daily file) buy a newly issued
  par bond of maturity N years with semiannual coupons. Its coupon is the par
  rate implied by the fitted discount function at t-1:
  ``c = 2 (1 - d(N)) / sum_k d(k/2)``, k = 1..2N.
* At t, price the same cash flows (now due at k/2 - dt, dt = days/365.25) off the
  fitted curve of date t. No coupon falls inside one month, so the simple total
  return is ``P_t / 100 - 1`` (dirty price; accrued interest is part of the return).
* The discount function is the published Svensson (or, before 1980 for
  nominals and 2004 for TIPS, Nelson-Siegel) parametrisation with
  continuously-compounded zero yields; parameters are those published per day.
* TIPS: the same computation on the real curve gives the real return; the
  nominal return multiplies by the change in the reference CPI (CPI-U NSA
  lagged 3 months, interpolated by day as in the Treasury's index-ratio
  formula). The deflation floor is ignored.

Only maturities the Fed reports for that date are used (the Fed blanks fitted
yields beyond the longest outstanding security): this is why the long bond is
15 years before 1981-07 and 20 years after.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .assets_sources import fred_series, gsw_nominal, gsw_tips


def _zero_yield(m: np.ndarray, p: pd.Series) -> np.ndarray:
    """Continuously-compounded zero yield (percent) at maturities m (years)."""
    m = np.maximum(np.asarray(m, float), 1e-6)
    b0, b1, b2, b3, t1, t2 = (p["BETA0"], p["BETA1"], p["BETA2"], p.get("BETA3", 0.0), p["TAU1"], p.get("TAU2", np.nan))
    x1 = m / t1
    f1 = (1 - np.exp(-x1)) / x1
    y = b0 + b1 * f1 + b2 * (f1 - np.exp(-x1))
    if np.isfinite(b3) and b3 != 0 and np.isfinite(t2) and t2 > 0:
        x2 = m / t2
        f2 = (1 - np.exp(-x2)) / x2
        y = y + b3 * (f2 - np.exp(-x2))
    return y


def _discount(m: np.ndarray, p: pd.Series) -> np.ndarray:
    m = np.asarray(m, float)
    return np.exp(-_zero_yield(m, p) / 100 * m)


def par_coupon(n_years: float, p: pd.Series) -> float:
    t = np.arange(1, int(round(2 * n_years)) + 1) / 2
    d = _discount(t, p)
    return 2 * (1 - d[-1]) / d.sum()


def hold_return(n_years: float, p0: pd.Series, p1: pd.Series, dt_years: float) -> float:
    """Return from buying the N-year par bond on curve p0 and repricing on p1 after dt."""
    c = par_coupon(n_years, p0)
    t = np.arange(1, int(round(2 * n_years)) + 1) / 2 - dt_years
    if t[0] <= 0:
        raise ValueError("holding period spans a coupon date")
    cf = np.full(t.shape, 100 * c / 2)
    cf[-1] += 100
    return float((cf * _discount(t, p1)).sum() / 100 - 1)


def _max_reported_maturity(curve: pd.DataFrame, prefix: str) -> pd.Series:
    cols = [c for c in curve.columns if c.startswith(prefix) and c[len(prefix):].isdigit()]
    mats = np.array([int(c[len(prefix):]) for c in cols])
    have = curve[cols].notna().to_numpy()
    return pd.Series(np.where(have.any(axis=1), (have * mats).max(axis=1), 0), index=curve.index)


def _month_end_rows(curve: pd.DataFrame) -> pd.DataFrame:
    return curve.groupby(curve.index.to_period("M")).tail(1)


def _returns(curve: pd.DataFrame, maturity, rows: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """maturity: float, or callable(max_reported_maturity) -> float|None."""
    maxmat = _max_reported_maturity(rows, prefix)
    out = []
    for i in range(1, len(rows)):
        d0, d1 = rows.index[i - 1], rows.index[i]
        n = maturity(maxmat.iloc[i - 1]) if callable(maturity) else maturity
        if n is None or maxmat.iloc[i - 1] < n:
            continue
        dt = (d1 - d0).days / 365.25
        out.append((d0, d1, n, hold_return(n, rows.iloc[i - 1], rows.iloc[i], dt)))
    return pd.DataFrame(out, columns=["start", "end", "maturity", "ret"]).set_index("end")


def treasury(maturity, daily: bool = False) -> pd.DataFrame:
    """Constant-maturity nominal Treasury par-bond returns. Index = holding-period end date."""
    curve = gsw_nominal()
    rows = curve if daily else _month_end_rows(curve)
    return _returns(curve, maturity, rows, "SVENY")


def ref_cpi(dates: pd.DatetimeIndex) -> pd.Series:
    """Treasury reference CPI: CPI-U NSA of month m-3 plus (day-1)/days-in-month
    of the step to month m-2."""
    cpi = fred_series("CPIAUCNS")
    cpi.index = cpi.index.to_period("M")
    # A month BLS never published (October 2025, government shutdown) is replaced
    # as Treasury does for TIPS (31 CFR 356, App. B): last CPI times its trailing
    # 12-month growth to the 1/12 power.
    full = pd.period_range(cpi.index.min(), cpi.index.max(), freq="M")
    cpi = cpi.reindex(full)
    for m in cpi.index[cpi.isna()]:
        base = cpi[m - 1]
        cpi[m] = base * (base / cpi[m - 13]) ** (1 / 12)
    out = []
    for d in dates:
        m = pd.Period(d, "M")
        a, b = cpi.get(m - 3, np.nan), cpi.get(m - 2, np.nan)
        out.append(a + (d.day - 1) / m.days_in_month * (b - a))
    return pd.Series(out, index=dates)


def tips(maturity: float, daily: bool = False) -> pd.DataFrame:
    curve = gsw_tips()
    rows = curve if daily else _month_end_rows(curve)
    real = _returns(curve, maturity, rows, "TIPSY")
    rc_end = ref_cpi(pd.DatetimeIndex(real.index))
    rc_start = ref_cpi(pd.DatetimeIndex(real["start"]))
    real["real_ret"] = real["ret"]
    real["ret"] = (1 + real["real_ret"]) * (rc_end.to_numpy() / rc_start.to_numpy()) - 1
    return real.dropna(subset=["ret"])


def to_month_index(df: pd.DataFrame) -> pd.Series:
    s = df["ret"].copy()
    s.index = pd.DatetimeIndex(s.index).to_period("M").to_timestamp(how="end").normalize()
    return s
