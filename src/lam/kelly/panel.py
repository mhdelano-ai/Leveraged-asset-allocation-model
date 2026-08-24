"""Monthly total-return panels for the four-asset Kelly question.

The universe asked for -- US equity, international ex-US equity, US bonds,
international ex-US bonds, plus cash -- has a data problem that dominates the
answer, so it is handled explicitly rather than papered over.

**International bonds are the binding constraint.** The asset class as it is
actually bought today (currency-hedged global aggregate ex-USD) has a tradable
history starting 2013-06, when BNDX launched. Thirteen years is not enough to
estimate a Kelly weight: the standard error on a mean return over 13 years is
roughly vol/sqrt(13), which for a 4% vol asset is ~1.1pp/yr -- comparable to the
entire risk premium being estimated.

So there are two panels, and they are reported together because they disagree:

============  ==============  ====================================================
Panel         Span            Composition
============  ==============  ====================================================
``modern``    2013-06 ->      The real funds: VTI, VXUS, BND, BNDX. Nothing
                              synthetic, nothing survivorship-selected -- and far
                              too short, over a sample containing one bond bear
                              market and no equity decade-long drawdown.
``long``      1993-01 ->      Index funds where they exist (VFINX, VBMFX) and the
                              longest-running hedged foreign bond fund (PFORX)
                              plus an active international equity fund (VTRIX)
                              where they do not. 33 years, at the cost of two
                              actively managed, survivorship-selected sleeves.
============  ==============  ====================================================

``long``'s two active sleeves survived to 2026 and were picked *because* they
did, so treat their returns as optimistic -- roughly 50-100bp/yr on the
international equity sleeve, which flatters its Kelly weight.

Cash is the 3-month T-bill in bond-equivalent yield, the same series the rest of
the project uses for the risk-free rate and as the base financing rate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..data import rates, yahoo

# Label -> (modern fund, long-history proxy). The pair is kept together so the
# two panels cannot drift apart in ordering.
UNIVERSE: dict[str, tuple[str, str]] = {
    "us_equity": ("VTI", "VFINX"),
    "intl_equity": ("VXUS", "VTRIX"),
    "us_bonds": ("BND", "VBMFX"),
    "intl_bonds": ("BNDX", "PFORX"),
}

ASSETS = list(UNIVERSE)

PRETTY = {
    "us_equity": "US equities",
    "intl_equity": "Intl ex-US equities",
    "us_bonds": "US bonds",
    "intl_bonds": "Intl ex-US bonds",
}

# What each proxy actually is, so the caveats travel with the numbers.
PROXY_NOTES = {
    "VTI": "Vanguard Total Stock Market ETF (index)",
    "VXUS": "Vanguard Total International Stock ETF (index, ex-US)",
    "BND": "Vanguard Total Bond Market ETF (index, US aggregate)",
    "BNDX": "Vanguard Total International Bond ETF (index, ex-US, USD-hedged)",
    "VFINX": "Vanguard 500 Index (index; large-cap only, not total market)",
    "VTRIX": "Vanguard International Value (ACTIVE, survivorship-selected)",
    "VBMFX": "Vanguard Total Bond Market Index (index, US aggregate)",
    "PFORX": "PIMCO Foreign Bond USD-Hedged (ACTIVE, survivorship-selected)",
}

PANELS = {
    "modern": {"which": 0, "start": "2013-07-01"},
    "long": {"which": 1, "start": "1993-01-01"},
}


@dataclass(frozen=True)
class KellyPanel:
    """Monthly simple total returns plus the matching cash return.

    ``daily``/``daily_cash`` carry the same history at daily frequency. Monthly
    data is the right resolution for estimating a Kelly weight; it is the wrong
    resolution for asking whether a margin call fires, because a maintenance
    breach is a *path* event inside the month. Both are kept so each question is
    answered at the frequency it needs.
    """

    name: str
    returns: pd.DataFrame  # columns = ASSETS, monthly simple returns
    cash: pd.Series  # monthly simple return on 3m T-bills
    tickers: dict[str, str]
    daily: pd.DataFrame | None = None
    daily_cash: pd.Series | None = None

    @property
    def excess(self) -> pd.DataFrame:
        return self.returns.sub(self.cash, axis=0)

    @property
    def months(self) -> int:
        return len(self.returns)

    def __str__(self) -> str:
        span = f"{self.returns.index[0]:%Y-%m} -> {self.returns.index[-1]:%Y-%m}"
        legs = ", ".join(f"{PRETTY[a]}={self.tickers[a]}" for a in ASSETS)
        return f"panel {self.name}: {span} ({self.months} months); {legs}"


def _monthly_returns(symbol: str) -> pd.Series:
    """Month-end to month-end total return, from daily adjusted closes."""
    daily = yahoo.total_return_prices(symbol)
    monthly = daily.resample("ME").last()
    return monthly.pct_change().dropna().rename(symbol)


def _daily_returns(symbol: str) -> pd.Series:
    return yahoo.total_return_prices(symbol).pct_change().dropna().rename(symbol)


def _monthly_cash() -> pd.Series:
    """Monthly simple return on 3-month bills.

    The bill rate is an annual bond-equivalent yield; the month's return is the
    average quoted rate over the month de-annualised geometrically. Averaging
    within the month rather than taking the month-end quote matters in 1994 and
    2022, when the level moved several hundred basis points inside a year.
    """
    daily_bey = rates.bill_rate_bey(prefer="fred")
    monthly_bey = daily_bey.resample("ME").mean()
    return ((1.0 + monthly_bey) ** (1.0 / 12.0) - 1.0).rename("cash")


def build_panel(name: str, *, end: str | None = None) -> KellyPanel:
    if name not in PANELS:
        raise ValueError(f"unknown panel {name!r}; choose from {sorted(PANELS)}")
    spec = PANELS[name]
    which = spec["which"]
    tickers = {asset: UNIVERSE[asset][which] for asset in ASSETS}

    cols = {asset: _monthly_returns(ticker) for asset, ticker in tickers.items()}
    frame = pd.DataFrame(cols)[ASSETS].dropna()
    cash = _monthly_cash()

    joined = frame.join(cash.rename("cash"), how="inner").dropna()
    joined = joined.loc[spec["start"] :]
    if end is not None:
        joined = joined.loc[:end]
    if joined.empty:
        raise RuntimeError(f"panel {name}: no overlapping months")

    returns = joined[ASSETS]
    if returns.isna().to_numpy().any():
        raise RuntimeError(f"panel {name}: NaNs survived the join")
    if not np.isfinite(returns.to_numpy()).all():
        raise RuntimeError(f"panel {name}: non-finite returns")

    daily_cols = {asset: _daily_returns(ticker) for asset, ticker in tickers.items()}
    daily = pd.DataFrame(daily_cols)[ASSETS].dropna()
    # Bills are quoted on business days only; forward-fill covers holidays, and
    # the annual BEY becomes a calendar-day rate so a three-day weekend accrues
    # three days of interest rather than one.
    bey = rates.bill_rate_bey(prefer="fred").reindex(daily.index).ffill().bfill()
    day_gap = pd.Series(daily.index, index=daily.index).diff().dt.days.fillna(1.0).clip(1, 5)
    daily_cash = ((1.0 + bey) ** (day_gap / 365.0) - 1.0).rename("cash")
    window = slice(returns.index[0] - pd.offsets.MonthBegin(1), returns.index[-1])
    daily = daily.loc[window]
    daily_cash = daily_cash.loc[window]

    return KellyPanel(
        name=name,
        returns=returns,
        cash=joined["cash"],
        tickers=tickers,
        daily=daily,
        daily_cash=daily_cash,
    )
