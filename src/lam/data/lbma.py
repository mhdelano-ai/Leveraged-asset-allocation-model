"""LBMA gold price fixings.

The LBMA publishes its full daily fixing history as plain JSON, back to
1968-04-01, with records shaped ``{"d": "YYYY-MM-DD", "v": [USD, GBP, EUR]}``.
This is the cleanest long gold series available without a paid feed -- the FRED
LBMA mirrors (``GOLDAMGBD228NLBM``, ``GOLDPMGBD228NLBM``) are discontinued and
return 404.

Note ``DEFAULT_START``: before the Nixon shock of August 1971 the dollar gold
price was administratively pegged, so the pre-1971 series shows near-zero
volatility. Feeding that to a risk-parity sizer would hand it a fake riskless
real asset and produce enormous phantom leverage.
"""

from __future__ import annotations

import json

import pandas as pd

from . import cache
from .http import get

URL_PM = "https://prices.lbma.org.uk/json/gold_pm.json"
URL_AM = "https://prices.lbma.org.uk/json/gold_am.json"

# Gold floated on 1971-08-15. Use a clean month boundary just after.
DEFAULT_START = "1971-09-01"


def gold_usd(*, use_pm: bool = True, refresh: bool = False) -> pd.Series:
    """Daily USD gold fixing. Full history from 1968-04-01."""
    url = URL_PM if use_pm else URL_AM
    key = "lbma_gold_pm" if use_pm else "lbma_gold_am"
    if cache.has(key) and not refresh:
        return cache.load(key)

    raw = get(url)
    records = json.loads(raw)

    dates, values = [], []
    for rec in records:
        vals = rec.get("v") or []
        if not vals or vals[0] is None:
            continue
        usd = float(vals[0])
        if usd <= 0:
            continue
        dates.append(rec["d"])
        values.append(usd)

    out = pd.Series(values, index=pd.DatetimeIndex(pd.to_datetime(dates)), name="gold")
    out = out[~out.index.duplicated(keep="last")].sort_index()

    cache.store(key, out, url=url, raw=raw)
    return out


def gold_floating(*, refresh: bool = False) -> pd.Series:
    """Gold from the start of the floating-rate era (post Bretton Woods)."""
    return gold_usd(refresh=refresh).loc[DEFAULT_START:]
