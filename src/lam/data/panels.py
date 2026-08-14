"""Assemble the asset-return panels the strategy trades.

A diversified universe and a long history pull in opposite directions. The
tradable ETFs for most sleeves begin between 2001 and 2007, but a *leveraged*
strategy is only credible if it has been through the 1970s inflation, 1987,
2000-02 and 2008. One panel cannot do both jobs, so there are three:

============  ==============  =========================================
Panel         Span            Role
============  ==============  =========================================
``L`` long    1971-09 ->      **Binds the drawdown constraint.** Reduced
                              universe, but spans every major crisis and
                              both stock/bond correlation regimes.
``X`` extend  1986-11 ->      Near-full universe via long-history mutual
                              funds. Structural cross-check.
``M`` modern  2006-02 ->      Fully tradable ETFs. Implementation
                              realism and the vehicle head-to-head only.
============  ==============  =========================================

The constraint verdict is reported on **L** and must also hold on **X** with the
same parameters. ``M`` contains a single crisis and is never used to validate the
constraint -- passing a drawdown test on a sample with one drawdown means nothing.

Mutual-fund sleeves in ``X`` are actively managed and were selected because they
still exist in 2026, so they carry survivorship bias. Treat their returns as
optimistic by roughly 100bp/yr.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import lbma, rates, synth_bonds, synth_equity, yahoo

# Effective maturities calibrated against TLT and IEF (see synth_bonds).
LONG_UST_MATURITY = 23.5
INTERM_UST_MATURITY = 8.0

# The instrument a real implementation would hold for each sleeve.
#
# Panels L and X trade synthetic series -- par-bond rolls off constant-maturity
# yields, the LBMA gold fix -- because the ETFs did not exist for most of the
# history. These tickers are the modern proxy you would actually buy, not the
# series that was backtested; ``Panel.tickers_are_proxies`` says which is which.
SLEEVE_TICKERS = {
    "us_equity": "VTI",
    "intl_equity": "EFA",
    "em_equity": "EEM",
    "long_ust": "TLT",
    "interm_ust": "IEF",
    "ig_credit": "LQD",
    "hy_credit": "HYG",
    "gold": "GLD",
    "commodities": "DBC",
    "reit": "VNQ",
}

# One-way transaction cost per unit turnover, by sleeve.
SLEEVE_COSTS = {
    "us_equity": 0.0003,
    "intl_equity": 0.0006,
    "em_equity": 0.0012,
    "long_ust": 0.0002,
    "interm_ust": 0.0002,
    "ig_credit": 0.0005,
    "hy_credit": 0.0015,
    "gold": 0.0004,
    "commodities": 0.0010,
    "reit": 0.0006,
}


@dataclass
class Panel:
    name: str
    returns: pd.DataFrame
    benchmark: pd.Series
    rf: pd.Series
    eligible: pd.DataFrame
    description: str

    @property
    def costs(self) -> np.ndarray:
        return np.array([SLEEVE_COSTS.get(c, 0.0005) for c in self.returns.columns])

    @property
    def tickers(self) -> list[str]:
        """The instrument to hold for each sleeve, in column order."""
        return [SLEEVE_TICKERS.get(c, "?") for c in self.returns.columns]

    @property
    def tickers_are_proxies(self) -> bool:
        """True when the panel trades synthetic series rather than the ETFs.

        Panels L and X reach back before the ETFs existed, so their tickers are
        what you would buy today, not what was backtested.
        """
        return self.name != "M"

    def coverage(self) -> pd.DataFrame:
        rows = []
        for col in self.returns.columns:
            s = self.returns[col].dropna()
            rows.append(
                {
                    "sleeve": col,
                    "start": s.index.min().date() if len(s) else None,
                    "end": s.index.max().date() if len(s) else None,
                    "n": len(s),
                    "cost_bps": SLEEVE_COSTS.get(col, 0.0005) * 1e4,
                }
            )
        return pd.DataFrame(rows).set_index("sleeve")

    def __str__(self) -> str:
        return (
            f"Panel {self.name}: {len(self.returns)} days, "
            f"{self.returns.index.min().date()} -> {self.returns.index.max().date()}, "
            f"{self.returns.shape[1]} sleeves"
        )


def _fund_returns(symbol: str) -> pd.Series:
    return yahoo.total_return_prices(symbol).pct_change().dropna()


def _collateralised_commodity(rf: pd.Series) -> pd.Series:
    """``^SPGSCI`` is an excess-return index; a real position also earns
    collateral interest, so the T-bill yield is added back."""
    idx = yahoo.prices("^SPGSCI", field="close").pct_change().dropna()
    carry = rf.reindex(idx.index).ffill().fillna(0.0) / 252.0
    return (idx + carry).rename("commodities")


def _synthetic_treasuries(rf: pd.Series) -> tuple[pd.Series, pd.Series]:
    long_ust = synth_bonds.total_return(rates.cmt_yield_pct("30y"), LONG_UST_MATURITY)
    interm_ust = synth_bonds.total_return(rates.cmt_yield_pct("10y"), INTERM_UST_MATURITY)
    return long_ust.rename("long_ust"), interm_ust.rename("interm_ust")


def _assemble(
    series_map: dict[str, pd.Series],
    *,
    name: str,
    start: str,
    description: str,
    benchmark: pd.Series,
    rf: pd.Series,
    min_sleeves: int = 2,
) -> Panel:
    frame = pd.DataFrame(series_map).sort_index()
    frame = frame.loc[start:]

    # A sleeve is eligible only from its own first observation -- never
    # back-filled. Trading a sleeve before it existed is the purest form of
    # lookahead.
    eligible = frame.notna()
    for col in frame.columns:
        first = frame[col].first_valid_index()
        eligible[col] = frame.index >= first if first is not None else False

    frame = frame[eligible.sum(axis=1) >= min_sleeves]
    eligible = eligible.loc[frame.index]

    bench = benchmark.reindex(frame.index).dropna()
    frame = frame.loc[bench.index]
    eligible = eligible.loc[bench.index]

    return Panel(
        name=name,
        returns=frame,
        benchmark=bench,
        rf=rf.reindex(frame.index).ffill(),
        eligible=eligible,
        description=description,
    )


def build_long(start: str = "1971-09-01") -> Panel:
    """Panel L -- the binding validation panel."""
    rf = rates.bill_rate_bey()
    spx = synth_equity.spx_total_return()
    long_ust, interm_ust = _synthetic_treasuries(rf)
    gold = lbma.gold_floating().pct_change().dropna().rename("gold")

    return _assemble(
        {
            "us_equity": spx,
            "long_ust": long_ust,
            "interm_ust": interm_ust,
            "gold": gold,
            "commodities": _collateralised_commodity(rf),
        },
        name="L",
        start=start,
        description=(
            "Long backbone, 1971-09+. Synthetic S&P total return, synthetic "
            "Treasury par-bond rolls, LBMA gold, collateralised GSCI. Spans "
            "1973-74, 1980-82, 1987, 2000-02, 2008, 2020 and 2022."
        ),
        benchmark=spx,
        rf=rf,
    )


def build_extended(start: str = "1986-11-14") -> Panel:
    """Panel X -- near-full universe via long-history mutual funds."""
    rf = rates.bill_rate_bey()
    spx = synth_equity.spx_total_return()
    long_ust, interm_ust = _synthetic_treasuries(rf)
    gold = lbma.gold_floating().pct_change().dropna().rename("gold")

    return _assemble(
        {
            "us_equity": spx,
            "intl_equity": _fund_returns("VTRIX"),
            "long_ust": long_ust,
            "interm_ust": interm_ust,
            "ig_credit": _fund_returns("VWESX"),
            "hy_credit": _fund_returns("VWEHX"),
            "gold": gold,
            "commodities": _collateralised_commodity(rf),
            "reit": _fund_returns("FRESX"),
        },
        name="X",
        start=start,
        description=(
            "Extended universe, 1986-11+. Adds intl equity, IG and HY credit and "
            "REITs via long-history mutual funds. Actively managed and "
            "survivorship-selected -- treat as ~100bp/yr optimistic."
        ),
        benchmark=spx,
        rf=rf,
    )


def build_modern(start: str = "2006-02-06") -> Panel:
    """Panel M -- fully tradable ETFs. Implementation realism only."""
    rf = rates.bill_rate_bey()
    spx = synth_equity.spx_total_return()

    return _assemble(
        {
            "us_equity": _fund_returns("VTI"),
            "intl_equity": _fund_returns("EFA"),
            "em_equity": _fund_returns("EEM"),
            "long_ust": _fund_returns("TLT"),
            "interm_ust": _fund_returns("IEF"),
            "ig_credit": _fund_returns("LQD"),
            "hy_credit": _fund_returns("HYG"),
            "gold": _fund_returns("GLD"),
            "commodities": _fund_returns("DBC"),
            "reit": _fund_returns("VNQ"),
        },
        name="M",
        start=start,
        description=(
            "Modern tradable ETFs, 2006-02+. Contains a single crisis, so it is "
            "used for cost realism and the vehicle comparison, never to validate "
            "the drawdown constraint."
        ),
        benchmark=spx,
        rf=rf,
    )


BUILDERS = {"L": build_long, "X": build_extended, "M": build_modern}


def build(name: str) -> Panel:
    if name not in BUILDERS:
        raise KeyError(f"unknown panel {name!r}; expected one of {sorted(BUILDERS)}")
    return BUILDERS[name]()
