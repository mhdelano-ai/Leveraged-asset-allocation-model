"""S&P 500 daily total return, reconstructed back to 1927.

``^GSPC`` is a price index. Yahoo reports an ``adjclose`` column for it, but for
indices that column is identical to ``close`` -- it carries no dividends at all.
Using it as a total-return series silently discards 2-4%/yr, which over a
multi-decade backtest is the difference between a strategy beating the benchmark
and losing to it.

Reconstruction accrues Shiller's trailing-12m dividend yield onto the daily price
path::

    TR_t = TR_{t-1} * (P_t / P_{t-1}) * (1 + y_m)^(dt / 365.25)

The price path is daily ``^GSPC``; only the yield comes from Shiller (whose own
price column is a monthly *average* and would erase intramonth crashes -- see
``shiller`` module docstring).

Validated against actual ``^SP500TR`` over 1988-2026: 4bp/yr CAGR difference and
0.02pp max-drawdown difference. From 1988 the actual index is spliced in, so the
reconstruction is only load-bearing before then.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..timeaxis import year_fractions
from . import shiller, yahoo

SPLICE_DATE = "1988-01-04"  # ^SP500TR inception


def reconstructed_tr_returns(*, refresh: bool = False) -> pd.Series:
    """Daily S&P 500 total returns built from price + Shiller dividend yield."""
    price = yahoo.prices("^GSPC", field="close", refresh=refresh)
    dy_monthly = shiller.dividend_yield(refresh=refresh)

    price_ret = price.pct_change().dropna()

    # Map each day onto its month's trailing yield; forward-fill past the end of
    # Shiller's series (it lags the market by a month or two).
    month_key = price_ret.index.to_period("M").to_timestamp()
    dy = dy_monthly.reindex(pd.DatetimeIndex(month_key.unique()).sort_values()).ffill()
    daily_dy = pd.Series(dy.reindex(month_key).to_numpy(), index=price_ret.index).ffill()

    dt = year_fractions(price.index)
    div_accrual = (1.0 + daily_dy.to_numpy()) ** dt - 1.0

    total = (1.0 + price_ret.to_numpy()) * (1.0 + np.nan_to_num(div_accrual)) - 1.0
    return pd.Series(total, index=price_ret.index, name="spx_tr")


def spx_total_return(*, refresh: bool = False) -> pd.Series:
    """Daily S&P 500 total returns: reconstruction pre-1988, ``^SP500TR`` after."""
    recon = reconstructed_tr_returns(refresh=refresh)
    actual = yahoo.prices("^SP500TR", field="close", refresh=refresh).pct_change().dropna()

    spliced = pd.concat([recon.loc[: pd.Timestamp(SPLICE_DATE)].iloc[:-1], actual])
    spliced = spliced[~spliced.index.duplicated(keep="last")].sort_index()
    spliced.name = "spx_tr"
    return spliced


def validate_reconstruction(*, refresh: bool = False) -> dict:
    """Compare the reconstruction against actual ``^SP500TR`` over the overlap."""
    from ..metrics.core import cagr
    from ..metrics.drawdown import max_drawdown

    recon = reconstructed_tr_returns(refresh=refresh)
    actual = yahoo.prices("^SP500TR", field="close", refresh=refresh).pct_change().dropna()
    joined = pd.concat([recon, actual.rename("actual")], axis=1, join="inner").dropna()

    a, b = joined["spx_tr"], joined["actual"]
    return {
        "n_days": int(len(joined)),
        "start": str(joined.index[0].date()),
        "end": str(joined.index[-1].date()),
        "corr": float(a.corr(b)),
        "cagr_recon": cagr(a),
        "cagr_actual": cagr(b),
        "cagr_diff_bps": (cagr(a) - cagr(b)) * 1e4,
        "maxdd_recon": max_drawdown(a),
        "maxdd_actual": max_drawdown(b),
        "maxdd_diff_pp": (max_drawdown(a) - max_drawdown(b)) * 100,
        "te_ann": float((a - b).std() * np.sqrt(252)),
    }
