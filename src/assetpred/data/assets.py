"""Build the asset-return panel, cash, segment metadata, validation and daily returns.

    python -m assetpred.data.assets            # rebuild from cached raw files (downloads if missing)
    python -m assetpred.data.assets --refresh-yahoo   # also re-pull Yahoo histories

Outputs (data/processed/):
  asset_returns.parquet        monthly simple total returns, one column per config.ASSETS id
  cash.parquet                 column `cash`: 1-month T-bill return over the month
  asset_meta.csv               one row per splice segment (monthly panel and cash)
  asset_validation.csv         splice-overlap, ETF/fund and daily-vs-monthly checks
  asset_returns_daily.parquet  daily simple total returns on trading days (union of the sources'
                               observation dates, incl. pre-1953 NYSE Saturdays; NaN = that
                               source has no observation that day, its move is in the next return)
  asset_meta_daily.csv         one row per daily segment

Every return is computed from end-of-period levels (month-end index levels,
month-end fund NAVs plus distributions, or month-end fitted curves). No source
here is a within-month average. Segments are spliced by *returns*: a later
segment takes over from its first month, or (mode="extend") only after the
earlier segment's data ends, which is how sources with a publication lag are
topped up with the most recent months.
"""
from __future__ import annotations

import argparse
import dataclasses
import logging
import warnings

import numpy as np
import pandas as pd

from .. import config, schema
from . import assets_bonds as B
from . import assets_sources as S
from .assets_validate import validate

log = logging.getLogger("assetpred.assets")

DAILY = config.PROCESSED / "asset_returns_daily.parquet"
DAILY_META = config.PROCESSED / "asset_meta_daily.csv"
VALIDATION = config.PROCESSED / "asset_validation.csv"

UST_INT_YEARS = 5.0
UST_LONG_YEARS = 20.0
UST_LONG_EARLY_YEARS = 15.0   # longest maturity the Fed reports 1971-11..1981-06
TIPS_YEARS = 7.0
HY_FIRST_MONTH = "1985-02-28"  # Yahoo's VWEHX NAV is stale (unchanged ~95% of days) before 1985

# Representative ETFs / funds each final series is compared with (validation only).
VALIDATORS = {
    "us_eq": ["VTI", "SPY"],
    "dev_eq": ["EFA", "VEA"],
    "em_eq": ["EEM", "VWO"],
    "ust_int": ["IEI", "VFITX"],
    "ust_long": ["TLT", "VUSTX"],
    "tips": ["TIP", "VIPSX"],
    "ig_credit": ["LQD", "VFICX"],
    "hy_credit": ["HYG", "SPHIX"],
    "reits": ["VNQ"],
    "gold": ["IAU", "GLD"],
    "commodities": ["DJP", "DBC", "GSG"],
}


@dataclasses.dataclass
class Segment:
    key: str                 # short id used in validation output
    series: pd.Series        # full available history of this source (monthly, month-end index)
    source: str
    description: str
    start: str | None = None  # first month this segment is used (default: its first month)
    mode: str = "takeover"    # "takeover" or "extend"


def last_complete_month(today: pd.Timestamp | None = None) -> pd.Timestamp:
    today = pd.Timestamp.today() if today is None else today
    return (today.to_period("M") - 1).to_timestamp(how="end").normalize()


# ------------------------------------------------------------------ segment definitions
def monthly_segments() -> tuple[dict[str, list[Segment]], list[Segment]]:
    ff = S.french_us()
    rf = ff["rf"]
    vti = S.yahoo_monthly_tr("VTI")

    segs: dict[str, list[Segment]] = {}
    segs["us_eq"] = [
        Segment("ff_us_mkt", ff["mkt"], "Kenneth French data library, F-F_Research_Data_Factors (Mkt-RF + RF)",
                "CRSP value-weighted US market (NYSE/AMEX/NASDAQ), total return"),
        Segment("VTI", vti, "Yahoo Finance via yfinance: VTI close + distributions",
                "Vanguard Total Stock Market ETF total return; tops up months not yet in the French file", mode="extend"),
    ]
    segs["dev_eq"] = [
        Segment("ff_intl_eafe", S.french_eafe_1975(), "Kenneth French data library, F-F_International_Indices.zip, Ind_all.Dat (value-weight dollar returns, Mkt)",
                "Fama/French international index: countries weighted by EAFE weights, USD total return (MSCI-based data)"),
        Segment("ff_dev_ex_us", S.french_region("dev_ex_us"), "Kenneth French data library, Developed_ex_US_3_Factors (Mkt-RF + RF)",
                "Fama/French developed ex-US market (22 countries incl. Canada), USD total return"),
        Segment("msci_world_ex_usa", S.msci("world_ex_usa"), "MSCI end-of-month index levels (app2.msci.com), index 991000, GRTR, USD",
                "MSCI World ex USA gross total return"),
    ]
    segs["em_eq"] = [
        Segment("ff_emerging", S.french_region("emerging"), "Kenneth French data library, Emerging_5_Factors (Mkt-RF + RF)",
                "Fama/French emerging-markets market portfolio, USD total return"),
        Segment("msci_em", S.msci("em"), "MSCI end-of-month index levels (app2.msci.com), index 891800, GRTR, USD",
                "MSCI Emerging Markets gross total return"),
    ]
    ust5 = B.to_month_index(B.treasury(UST_INT_YEARS))
    segs["ust_int"] = [
        Segment("gsw_5y", ust5, "Fed GSW nominal curve (feds200628.csv), month-end fitted Svensson/Nelson-Siegel parameters",
                "Synthetic 5-year constant-maturity par Treasury: buy at month-end t-1 at par, reprice off curve at t (see assets_bonds)"),
    ]
    ust15 = B.to_month_index(B.treasury(UST_LONG_EARLY_YEARS))
    ust20 = B.to_month_index(B.treasury(UST_LONG_YEARS))
    segs["ust_long"] = [
        Segment("gsw_15y", ust15, "Fed GSW nominal curve (feds200628.csv)",
                "Synthetic 15-year par Treasury; the Fed reports fitted yields only to 15y before 1981-07, so 15y is the longest defensible maturity"),
        Segment("gsw_20y", ust20, "Fed GSW nominal curve (feds200628.csv)",
                "Synthetic 20-year constant-maturity par Treasury"),
    ]
    segs["tips"] = [
        Segment("gsw_tips_7y", B.to_month_index(B.tips(TIPS_YEARS)),
                "Fed GSW TIPS curve (feds200805.csv) + FRED CPIAUCNS for the index ratio",
                "Synthetic 7-year real par TIPS, real price return x reference-CPI accrual (3-month lag, daily interpolation); deflation floor ignored"),
    ]
    segs["ig_credit"] = [
        Segment("VWESX", S.yahoo_monthly_tr("VWESX"), "Yahoo Finance via yfinance: VWESX NAV + dividends + capital gains",
                "Vanguard Long-Term Investment-Grade Fund (Investor) total return, net of ~0.2% fees. LONG duration (~13y) IG, mostly corporates; no free broad-IG index history exists (ICE BofA on FRED truncated to 3 years)"),
    ]
    segs["hy_credit"] = [
        Segment("VWEHX", S.yahoo_monthly_tr("VWEHX"), "Yahoo Finance via yfinance: VWEHX NAV + dividends + capital gains",
                "Vanguard High-Yield Corporate Fund (Investor) total return, net of ~0.25% fees; BB/B-tilted HY. Starts 1985-02 because Yahoo NAVs are stale before 1985",
                start=HY_FIRST_MONTH),
    ]
    segs["reits"] = [
        Segment("nareit_all_equity", S.nareit("All Equity REITs"), "Nareit MonthlyHistoricalReturns.xls (reit.com), All Equity REITs total-return index level",
                "FTSE Nareit All Equity REITs total return"),
    ]
    lbma = S.lbma_gold_pm()
    gold = []
    if lbma is not None:
        gold.append(Segment("lbma_pm", S._monthly_from_levels(lbma), "LBMA Gold Price PM (prices.lbma.org.uk)",
                            "Spot gold, last PM fixing of each month, USD"))
    gold.append(Segment("GLD", S.yahoo_monthly_tr("GLD"), "Yahoo Finance via yfinance: GLD close",
                        "SPDR Gold Shares (physically backed; spot gold less 0.40%/yr fee), NY close",
                        mode="extend" if gold else "takeover"))
    segs["gold"] = gold
    aqr_tr = (S.aqr_commodities() + rf).dropna()
    segs["commodities"] = [
        Segment("aqr_ew_plus_rf", aqr_tr, "AQR 'Commodities for the Long Run' (excess return of equal-weight futures portfolio) + French RF",
                "Equal-weight, fully collateralised commodity futures (month-end futures prices): AQR excess return plus 1-month T-bill = total return"),
        Segment("DJP", S.yahoo_monthly_tr("DJP"), "Yahoo Finance via yfinance: DJP close",
                "iPath Bloomberg Commodity Index Total Return ETN (BCOM TR less 0.75%/yr); tops up months after the AQR file ends",
                mode="extend"),
    ]
    cash = [
        Segment("ff_rf", rf, "Kenneth French data library, F-F_Research_Data_Factors (RF)",
                "1-month T-bill return (Ibbotson to 2024-05, ICE BofA 1M T-bill index after)"),
        Segment("dtb4wk", S.tbill_4wk_monthly(), "FRED DTB4WK (4-week bill discount rate, last observation of previous month)",
                "Return of a 4-week bill bought at the previous month-end, rolled for the days in the month; tops up months not yet in the French file",
                mode="extend"),
    ]
    return segs, cash


def splice(segments: list[Segment], end: pd.Timestamp) -> tuple[pd.Series, list[dict]]:
    pieces, meta = [], []
    for i, seg in enumerate(segments):
        s = seg.series.dropna()
        s = s[s.index <= end]
        if s.empty:
            continue
        if seg.mode == "extend" and pieces:
            start = pieces[-1].index.max() + pd.offsets.MonthEnd(1)
        else:
            start = pd.Timestamp(seg.start) if seg.start else s.index.min()
        s = s[s.index >= start]
        if s.empty:
            continue
        if pieces:
            pieces[-1] = pieces[-1][pieces[-1].index < s.index.min()]
            meta[-1]["segment_end"] = pieces[-1].index.max()
            if pieces[-1].empty:
                pieces.pop(), meta.pop()
        pieces.append(s)
        meta.append({"segment_start": s.index.min(), "segment_end": s.index.max(),
                     "source": seg.source, "description": seg.description, "key": seg.key})
    out = pd.concat(pieces).sort_index()
    full = pd.date_range(out.index.min(), out.index.max(), freq="ME")
    if len(full) != len(out) or out.isna().any():
        missing = full.difference(out.index)
        raise ValueError(f"gap in spliced series: {list(missing[:5])}")
    return out, meta


# ------------------------------------------------------------------ daily
def daily_returns(end: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Daily simple total returns where free daily history exists."""
    cols, meta = {}, []

    def add(asset, pieces):
        out = []
        for s, src, desc in pieces:
            s = s.dropna()
            if out:
                s = s[s.index > out[-1].index.max()]
            s = s[s.index <= end]
            if s.empty:
                continue
            out.append(s)
            meta.append({"asset": asset, "segment_start": s.index.min().date(), "segment_end": s.index.max().date(),
                         "source": src, "description": desc})
        cols[asset] = pd.concat(out)

    ffd = S.french_us(daily=True)
    add("us_eq", [(ffd["mkt"], "French F-F_Research_Data_Factors_daily (Mkt-RF + RF)", "CRSP VW US market"),
                  (S.yahoo_daily_tr("VTI"), "yfinance VTI", "Vanguard Total Stock Market ETF, after the French file ends")])
    msci_w = S.msci("world_ex_usa", daily=True)
    ffdev = S.french_region("dev_ex_us", daily=True)
    add("dev_eq", [(ffdev[ffdev.index < msci_w.index.min()], "French Developed_ex_US_3_Factors_Daily (Mkt-RF + RF)", "FF developed ex-US market"),
                   (msci_w, "MSCI daily GRTR 991000", "MSCI World ex USA gross (local close; non-synchronous with US)")])
    add("em_eq", [(S.msci("em", daily=True), "MSCI daily GRTR 891800", "MSCI EM gross (local close; non-synchronous with US)")])
    for asset, mat in (("ust_int", UST_INT_YEARS),):
        add(asset, [(B.treasury(mat, daily=True)["ret"], "Fed GSW nominal curve", f"{mat:g}y par bond repriced daily")])
    ul = B.treasury(lambda mm: UST_LONG_YEARS if mm >= UST_LONG_YEARS else (UST_LONG_EARLY_YEARS if mm >= UST_LONG_EARLY_YEARS else None), daily=True)
    add("ust_long", [(ul.loc[ul.maturity == UST_LONG_EARLY_YEARS, "ret"], "Fed GSW nominal curve", "15y par bond repriced daily (before 1981-07)"),
                     (ul.loc[ul.maturity == UST_LONG_YEARS, "ret"], "Fed GSW nominal curve", "20y par bond repriced daily")])
    add("tips", [(B.tips(TIPS_YEARS, daily=True)["ret"], "Fed GSW TIPS curve + CPIAUCNS", "7y real par TIPS repriced daily, with index-ratio accrual")])
    add("ig_credit", [(S.yahoo_daily_tr("VWESX"), "yfinance VWESX", "Vanguard Long-Term IG fund NAV total return")])
    hy = S.yahoo_daily_tr("VWEHX")
    add("hy_credit", [(hy[hy.index > "1985-01-31"], "yfinance VWEHX", "Vanguard HY Corporate fund NAV total return")])
    add("reits", [(S.yahoo_daily_tr("VNQ"), "yfinance VNQ", "Vanguard Real Estate ETF (MSCI US IMI Real Estate 25/50; not the Nareit index)")])
    lbma = S.lbma_gold_pm()
    gp = []
    if lbma is not None:
        gp.append((lbma.pct_change().iloc[1:], "LBMA Gold Price PM", "spot gold fixing"))
    gp.append((S.yahoo_daily_tr("GLD"), "yfinance GLD", "SPDR Gold Shares"))
    add("gold", gp)
    add("commodities", [(S.yahoo_daily_tr("DJP"), "yfinance DJP", "iPath BCOM TR ETN (not the AQR equal-weight index used monthly)")])

    df = pd.DataFrame(cols)
    # Keep every date some source observed: the NYSE traded on Saturdays until 1952,
    # and dropping those days would lose ~1/6 of early US returns.
    df = df.dropna(how="all").sort_index()
    df.index.name = "date"
    df = df[[a for a in config.ASSETS if a in df.columns]]
    return df, pd.DataFrame(meta)


# ------------------------------------------------------------------ build
def build(end: pd.Timestamp | None = None, with_daily: bool = True) -> pd.DataFrame:
    end = last_complete_month() if end is None else end
    segs, cash_segs = monthly_segments()

    cash, cash_meta = splice(cash_segs, end)
    cols, meta = {}, []
    for asset in config.ASSETS:
        s, m = splice(segs[asset], end)
        cols[asset] = s
        meta += [{"asset": asset, **row} for row in m]
    meta += [{"asset": "cash", **row} for row in cash_meta]

    last = min(s.index.max() for s in cols.values())
    last = min(last, cash.index.max())
    if last < end:
        warnings.warn(f"panel ends {last.date()} (requested {end.date()}): "
                      + ", ".join(f"{a}={s.index.max().date()}" for a, s in cols.items() if s.index.max() < end))
    panel = pd.DataFrame(cols).loc[:last]
    panel.index.name = "date"
    panel.index.freq = None
    cash = cash.loc[:last]
    if cash.index.min() > panel.index.min():
        raise ValueError("cash must cover the whole asset panel")
    meta_df = pd.DataFrame(meta)
    meta_df["segment_end"] = meta_df["segment_end"].where(meta_df["segment_end"] < last, last)
    for c in ("segment_start", "segment_end"):
        meta_df[c] = pd.to_datetime(meta_df[c]).dt.date

    config.PROCESSED.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(schema.ASSET_RETURNS)
    cash_df = cash.rename("cash").to_frame()
    cash_df.index.name = "date"
    cash_df.to_parquet(schema.CASH)
    meta_df[["asset", "segment_start", "segment_end", "source", "description", "key"]].to_csv(schema.ASSET_META, index=False)

    daily = dmeta = None
    if with_daily:
        daily, dmeta = daily_returns(last)
        daily.to_parquet(DAILY)
        dmeta.to_csv(DAILY_META, index=False)

    validate(panel, cash, segs, cash_segs, VALIDATORS, daily=daily, daily_meta=dmeta, path=VALIDATION)
    return panel


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--refresh-yahoo", action="store_true", help="re-download Yahoo histories")
    ap.add_argument("--no-daily", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.refresh_yahoo:
        for t in {"VTI", "VWESX", "VWEHX", "GLD", "DJP", "VNQ", *sum(VALIDATORS.values(), [])}:
            S.yahoo(t, refresh=True)
    panel = build(with_daily=not args.no_daily)
    xr = schema.excess_log_returns(panel, schema.load_cash())
    summary = pd.DataFrame({
        "first": panel.apply(lambda s: s.first_valid_index().date()),
        "last": panel.apply(lambda s: s.last_valid_index().date()),
        "xr_mean_ann": xr.mean() * 12,
        "xr_vol_ann": xr.std() * np.sqrt(12),
        "ac1": panel.apply(lambda s: s.dropna().autocorr()),
    })
    print(summary.round(4).to_string())


if __name__ == "__main__":
    main()
