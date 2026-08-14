"""Nasdaq-100 daily total return, reconstructed back to 1985.

``^NDX`` is a price index and, like every index on Yahoo, its ``adjclose`` column
is a copy of ``close`` -- it carries no dividends. The Nasdaq-100 total-return
index (``^XNDX``) is not served by the chart endpoint at all, so total return has
to be built.

The obvious construction -- splice QQQ's adjusted close in from its 1999-03-10
inception -- is a trap. QQQ's early history on Yahoo contains bad prints: on
2000-01-07 it moves +12.37% against the index's +5.65%, and on 2000-09-22 +5.91%
against −0.46%. Over 1999-2002 the two series correlate at only 0.962, which at
that era's ~50% volatility is a 13%/yr tracking error -- far too large to be the
0.20% expense ratio and the dividend timing. The index is the clean series; the
fund is not.

So the price path is always ``^NDX``, and dividends are accrued onto it at a
**measured** yield::

    TR_t = TR_{t-1} * (P_t / P_{t-1}) * (1 + y)^(dt / 365.25)

``y`` is not assumed. It is estimated from the QQQ-versus-index gap over
2010-2026, where QQQ's data is clean (correlation 0.9987 from 2016), and the
fund's expense ratio is added back to recover the gross index yield.

The one genuine assumption is that this yield also applies before 1999. The
Nasdaq-100 of the late 1980s and 1990s paid less than today's -- Apple, Microsoft
and Nvidia all initiated dividends later -- so the pre-1999 reconstruction is if
anything slightly generous. :func:`yield_sensitivity` quantifies exactly how much
that assumption is worth, and it is small next to the effects this project
studies.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..timeaxis import year_fractions
from . import yahoo

# Expense ratios of the calibration funds, added back to recover the gross yield
# of the underlying index.
FUND_FEES = {"QQQ": 0.0020, "ONEQ": 0.0021}

# Window over which the fund tracks its index cleanly enough to calibrate on.
# Before 2010 QQQ's Yahoo history carries bad prints (see module docstring).
CALIBRATION_START = "2010-01-01"

INDEX_FUNDS = {"^NDX": "QQQ", "^IXIC": "ONEQ"}


def implied_dividend_yield(
    index_symbol: str = "^NDX",
    *,
    start: str = CALIBRATION_START,
    refresh: bool = False,
) -> float:
    """Gross dividend yield of ``index_symbol``, measured from its tracking fund.

    The fund's adjusted close is a genuine total return net of fees, so

        (1 + y) = (1 + fund CAGR) / (1 + index price CAGR) * (1 + expense ratio)

    The ratio form matters. :func:`total_return` accrues the yield
    *multiplicatively* onto the price path, so estimating it as a difference of
    CAGRs leaves a cross term ``CAGR_index * y`` behind -- 17bp/yr on the
    Nasdaq-100, which is most of the expense ratio it is trying to recover.
    """
    fund_symbol = INDEX_FUNDS[index_symbol]
    price = yahoo.prices(index_symbol, field="close", refresh=refresh)
    fund = yahoo.total_return_prices(fund_symbol, refresh=refresh)

    joined = pd.concat(
        [price.pct_change().rename("index"), fund.pct_change().rename("fund")],
        axis=1,
        join="inner",
    ).dropna()
    joined = joined.loc[start:]
    if len(joined) < 500:
        raise ValueError(f"{index_symbol}: only {len(joined)} calibration days from {start}")

    years = (joined.index[-1] - joined.index[0]).days / 365.25
    growth_index = float(np.prod(1.0 + joined["index"].to_numpy()))
    growth_fund = float(np.prod(1.0 + joined["fund"].to_numpy()))
    net_yield = (growth_fund / growth_index) ** (1.0 / years) - 1.0
    return float((1.0 + net_yield) * (1.0 + FUND_FEES[fund_symbol]) - 1.0)


def total_return(
    index_symbol: str = "^NDX",
    *,
    dividend_yield: float | None = None,
    refresh: bool = False,
) -> pd.Series:
    """Daily total returns for a Nasdaq price index.

    ``dividend_yield`` defaults to the fund-implied estimate. Pass a number to
    override it -- :func:`yield_sensitivity` uses that to bound the assumption.
    """
    if dividend_yield is None:
        dividend_yield = implied_dividend_yield(index_symbol, refresh=refresh)

    price = yahoo.prices(index_symbol, field="close", refresh=refresh)
    price_ret = price.pct_change().dropna()

    # Accrue on calendar days, so a three-day weekend earns three days of yield.
    dt = year_fractions(price.index)
    accrual = (1.0 + dividend_yield) ** dt - 1.0

    total = (1.0 + price_ret.to_numpy()) * (1.0 + np.nan_to_num(accrual)) - 1.0
    name = "ndx_tr" if index_symbol == "^NDX" else "nasdaq_tr"
    return pd.Series(total, index=price_ret.index, name=name)


def ndx_total_return(*, dividend_yield: float | None = None, refresh: bool = False) -> pd.Series:
    """Nasdaq-100 daily total return, 1985-10-01 onward."""
    return total_return("^NDX", dividend_yield=dividend_yield, refresh=refresh)


def composite_total_return(
    *, dividend_yield: float | None = None, refresh: bool = False
) -> pd.Series:
    """Nasdaq Composite daily total return, 1971-02-05 onward.

    Fourteen years longer than the Nasdaq-100 and it reaches the 1973-74 bear,
    where the Composite fell ~60% -- the only pre-1987 stress event available for
    a growth-equity book. Its yield is calibrated off ONEQ (2003-), a shorter and
    later window than QQQ's, so treat it as the weaker of the two series.
    """
    return total_return("^IXIC", dividend_yield=dividend_yield, refresh=refresh)


def validate_reconstruction(
    index_symbol: str = "^NDX", *, start: str = CALIBRATION_START, refresh: bool = False
) -> dict:
    """Compare the reconstruction against its tracking fund over the clean window.

    The reconstruction is calibrated to match the fund's *level* of return by
    construction, so the CAGR gap is not an independent test. The informative
    numbers are the correlation and the tracking error: those say whether the
    daily path is right, which is what a drawdown and a leverage rule respond to.
    """
    from ..metrics.core import cagr
    from ..metrics.drawdown import max_drawdown

    fund_symbol = INDEX_FUNDS[index_symbol]
    recon = total_return(index_symbol, refresh=refresh)
    fund = yahoo.total_return_prices(fund_symbol, refresh=refresh).pct_change().dropna()

    joined = pd.concat([recon.rename("recon"), fund.rename("fund")], axis=1, join="inner").dropna()
    joined = joined.loc[start:]
    a, b = joined["recon"], joined["fund"]
    return {
        "index": index_symbol,
        "fund": fund_symbol,
        "n_days": int(len(joined)),
        "start": str(joined.index[0].date()),
        "end": str(joined.index[-1].date()),
        "corr": float(a.corr(b)),
        "cagr_recon": cagr(a),
        "cagr_fund": cagr(b),
        "cagr_diff_bps": (cagr(a) - cagr(b)) * 1e4,
        "maxdd_recon": max_drawdown(a),
        "maxdd_fund": max_drawdown(b),
        "maxdd_diff_pp": (max_drawdown(a) - max_drawdown(b)) * 100,
        "te_ann": float((a - b).std() * np.sqrt(252)),
        "implied_yield": implied_dividend_yield(index_symbol, start=start, refresh=refresh),
    }


def yield_sensitivity(
    index_symbol: str = "^NDX", yields: tuple[float, ...] = (0.0, 0.005, 0.007, 0.010, 0.015)
) -> pd.DataFrame:
    """CAGR and drawdown of the reconstruction across dividend-yield assumptions.

    The point of the table is to show that the assumption is not load-bearing:
    the yield shifts the level of return but leaves the drawdown path -- which is
    what the leverage rules react to -- essentially untouched.
    """
    from ..metrics.core import ann_vol, cagr
    from ..metrics.drawdown import max_drawdown

    rows = []
    for y in yields:
        r = total_return(index_symbol, dividend_yield=y)
        rows.append(
            {
                "dividend_yield": y,
                "cagr": cagr(r),
                "vol": ann_vol(r),
                "max_dd": max_drawdown(r),
            }
        )
    return pd.DataFrame(rows).set_index("dividend_yield")
