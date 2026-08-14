"""Interest rates, with two interchangeable upstreams.

FRED is the natural home for constant-maturity Treasury yields, but it throttles
aggressively and can stay unreachable for long stretches -- during this build it
blocked both ``fredgraph.csv`` and ``/data/*.txt`` for an extended period. Since
every levered return in the project depends on a financing rate, a single point
of failure there is unacceptable.

Yahoo publishes the same constant-maturity series as indices and reaches back
just as far:

===========  ==========  ==============  ===========
Tenor        Yahoo       FRED            Yahoo start
===========  ==========  ==============  ===========
13-week      ``^IRX``    ``DTB3``        1960-01-04
5-year       ``^FVX``    ``DGS5``        1962-01-02
10-year      ``^TNX``    ``DGS10``       1962-01-02
30-year      ``^TYX``    ``DGS30``       1977-02-15
===========  ==========  ==============  ===========

These are the same underlying data: ``^TNX`` opens at 4.06 on 1962-01-02, which
is exactly ``DGS10``'s first observation. Yahoo is tried first because it is
reliably reachable here; FRED is the fallback and the cross-check.

``^IRX`` and ``DTB3`` are both **discount** rates and must be converted to
bond-equivalent yield before use as a financing or cash rate.
"""

from __future__ import annotations

import pandas as pd

from . import fred, yahoo

TENORS = {
    "3m": {"yahoo": "^IRX", "fred": "DTB3", "years": 0.25, "discount_basis": True},
    "5y": {"yahoo": "^FVX", "fred": "DGS5", "years": 5.0, "discount_basis": False},
    "10y": {"yahoo": "^TNX", "fred": "DGS10", "years": 10.0, "discount_basis": False},
    "30y": {"yahoo": "^TYX", "fred": "DGS30", "years": 30.0, "discount_basis": False},
}


def cmt_yield_pct(tenor: str, *, prefer: str = "yahoo", refresh: bool = False) -> pd.Series:
    """Constant-maturity yield in percent for ``tenor`` (e.g. ``"10y"``).

    Returned as quoted -- the 3m series is still on a discount basis. Use
    :func:`bill_rate_bey` when a usable rate is needed.
    """
    if tenor not in TENORS:
        raise KeyError(f"unknown tenor {tenor!r}; expected one of {sorted(TENORS)}")
    spec = TENORS[tenor]
    order = ("yahoo", "fred") if prefer == "yahoo" else ("fred", "yahoo")

    errors = []
    for source in order:
        try:
            if source == "yahoo":
                out = yahoo.prices(spec["yahoo"], field="close", refresh=refresh)
            else:
                out = fred.series(spec["fred"], refresh=refresh)
            out = out.dropna()
            out = out[out > 0]
            out.name = f"cmt_{tenor}"
            return out
        except Exception as exc:  # noqa: BLE001 - try the other upstream
            errors.append(f"{source}: {type(exc).__name__}: {exc}")
    raise RuntimeError(f"could not load {tenor} yield from any source -- " + " | ".join(errors))


def bill_rate_bey(*, prefer: str = "yahoo", refresh: bool = False) -> pd.Series:
    """3-month T-bill rate as a bond-equivalent **decimal** yield.

    This is the cash return and the base financing rate. The discount-to-BEY
    conversion is not cosmetic: at 1980s double-digit levels it is worth tens of
    basis points, and that is exactly the regime where the cost of leverage
    decides whether the strategy works.
    """
    quoted = cmt_yield_pct("3m", prefer=prefer, refresh=refresh)
    out = fred.discount_to_bey(quoted)
    out.name = "rf"
    return out


def financing_base(*, prefer: str = "yahoo", refresh: bool = False) -> pd.Series:
    """Annual-decimal base rate that borrowing is priced off."""
    return bill_rate_bey(prefer=prefer, refresh=refresh)


def curve_slope_pct(
    short_tenor: str, long_tenor: str, *, prefer: str = "yahoo", refresh: bool = False
) -> pd.Series:
    """Local ``dy/dM`` in percent per year of maturity, for the roll-down term."""
    short = cmt_yield_pct(short_tenor, prefer=prefer, refresh=refresh)
    long = cmt_yield_pct(long_tenor, prefer=prefer, refresh=refresh)
    gap = TENORS[long_tenor]["years"] - TENORS[short_tenor]["years"]
    joined = pd.concat([short.rename("s"), long.rename("l")], axis=1).ffill().dropna()
    return ((joined["l"] - joined["s"]) / gap).rename("slope")
