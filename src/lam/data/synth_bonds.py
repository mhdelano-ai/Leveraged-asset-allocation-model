"""Synthetic Treasury total returns from FRED constant-maturity yields.

Treasury ETFs only start in 2002 (TLT/IEF), but a leveraged strategy has to be
tested through the 1970s inflation and the 1980-82 Volcker rate shock. FRED's
constant-maturity series reach back to 1962, so total returns are reconstructed
from yields.

Method: hold a par bond of constant maturity ``M`` and roll it every day. Each
day the bond is repriced at the new yield, aged by one day, and credited coupon
accrual::

    a(i, n) = (1 - (1 + i)^-n) / i
    P(y, M, c) = 100 * [ (c/2) * a(y/2, 2M) + (1 + y/2)^(-2M) ]

    r_t = [ P(y_t, M - dt, c=y_{t-1}) + 100 * y_{t-1} * dt ] / 100 - 1

This is the **exact** par-bond reprice, deliberately not the usual
``-D_mod * dy`` duration approximation. Duration is a first-order expansion and
breaks down on large yield moves -- which is exactly what 1980-82 consists of, the
single most important regime in this study for pricing leverage.

``M`` is *calibrated*, not assumed -- ETF baskets are not their nominal tenor.
Measured against the modern ETFs, the best fits are M=23.5y on the 30y yield for
TLT (+1bp/yr CAGR, corr 0.940) and M=8.0y on the 10y yield for IEF (-48bp/yr,
corr 0.947).

An optional roll-down term is available but **off by default**. Repricing an aged
bond at the un-aged tenor's yield does ignore the pull down the curve, and adding
the term narrows IEF's gap (-48bp -> +35bp) -- but it *widens* TLT's (+1bp ->
+51bp), because the long end is nearly flat and a two-tenor slope proxy is crude
there. Since calibrating ``M`` already absorbs most of the effect and the
residual without it is conservative (it understates bond returns rather than
flattering them), the default omits it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..timeaxis import DAYS_PER_YEAR, year_fractions


def par_bond_price(yld: np.ndarray, maturity: np.ndarray | float, coupon: np.ndarray) -> np.ndarray:
    """Price (per 100 face) of a semiannual bond given annual-decimal inputs."""
    y = np.asarray(yld, dtype=float)
    c = np.asarray(coupon, dtype=float)
    n = 2.0 * np.asarray(maturity, dtype=float)
    i = y / 2.0

    # Guard the i -> 0 limit, where the annuity factor degenerates to n.
    tiny = np.abs(i) < 1e-12
    i_safe = np.where(tiny, 1e-12, i)
    discount = (1.0 + i_safe) ** (-n)
    annuity = np.where(tiny, n, (1.0 - discount) / i_safe)
    return 100.0 * ((c / 2.0) * annuity + discount)


def modified_duration(yld: np.ndarray, maturity: float, coupon: np.ndarray) -> np.ndarray:
    """Modified duration by central difference on the pricing function."""
    h = 1e-5
    p_up = par_bond_price(np.asarray(yld) + h, maturity, coupon)
    p_dn = par_bond_price(np.asarray(yld) - h, maturity, coupon)
    p_0 = par_bond_price(yld, maturity, coupon)
    return -(p_up - p_dn) / (2.0 * h) / p_0


def total_return(
    cmt_yield_pct: pd.Series,
    maturity: float,
    *,
    slope_pct_per_year: pd.Series | None = None,
) -> pd.Series:
    """Daily total return of a constant-maturity par-bond roll.

    Parameters
    ----------
    cmt_yield_pct:
        FRED constant-maturity yield in percent (e.g. ``DGS10``).
    maturity:
        Effective maturity in years. Calibrate against a modern ETF rather than
        using the nominal tenor.
    slope_pct_per_year:
        Optional local curve slope ``dy/dM`` in percent per year of maturity,
        aligned to ``cmt_yield_pct``. Supplies the roll-down term.
    """
    y = (cmt_yield_pct.astype(float) / 100.0).dropna().sort_index()
    if len(y) < 2:
        raise ValueError("need at least two yield observations")

    dt = year_fractions(y.index)
    y_prev = y.to_numpy()[:-1]
    y_now = y.to_numpy()[1:]

    price = par_bond_price(y_now, maturity - dt, y_prev)
    accrual = 100.0 * y_prev * dt
    ret = (price + accrual) / 100.0 - 1.0

    if slope_pct_per_year is not None:
        slope = (slope_pct_per_year.reindex(y.index).ffill() / 100.0).to_numpy()[1:]
        d_mod = modified_duration(y_now, maturity, y_prev)
        ret = ret + d_mod * np.nan_to_num(slope) * dt

    return pd.Series(ret, index=y.index[1:], name=f"ust{maturity:g}y")


def curve_slope(short_pct: pd.Series, long_pct: pd.Series, tenor_gap_years: float) -> pd.Series:
    """Local ``dy/dM`` in percent per year from two adjacent CMT tenors."""
    aligned = pd.concat([short_pct, long_pct], axis=1).ffill().dropna()
    return (aligned.iloc[:, 1] - aligned.iloc[:, 0]) / tenor_gap_years


def calibrate_maturity(
    cmt_yield_pct: pd.Series,
    benchmark_returns: pd.Series,
    *,
    grid: np.ndarray | None = None,
    slope_pct_per_year: pd.Series | None = None,
) -> tuple[float, pd.DataFrame]:
    """Pick the effective maturity minimising tracking error vs a real ETF.

    Returns the best maturity and the full diagnostic grid.
    """
    if grid is None:
        grid = np.arange(2.0, 30.5, 0.5)

    rows = []
    for m in grid:
        synth = total_return(cmt_yield_pct, float(m), slope_pct_per_year=slope_pct_per_year)
        joined = pd.concat([synth, benchmark_returns], axis=1, join="inner").dropna()
        if len(joined) < 250:
            continue
        a, b = joined.iloc[:, 0], joined.iloc[:, 1]
        years = (joined.index[-1] - joined.index[0]).days / DAYS_PER_YEAR
        rows.append(
            {
                "maturity": float(m),
                "corr": float(a.corr(b)),
                "te_ann": float((a - b).std() * np.sqrt(252)),
                "cagr_synth": float((1 + a).prod() ** (1 / years) - 1),
                "cagr_bench": float((1 + b).prod() ** (1 / years) - 1),
                "vol_synth": float(a.std() * np.sqrt(252)),
                "vol_bench": float(b.std() * np.sqrt(252)),
            }
        )

    diagnostics = pd.DataFrame(rows).set_index("maturity")
    best = float(diagnostics["te_ann"].idxmin())
    return best, diagnostics
