"""Raw predictor panel: valuation/carry, macro and sentiment series, lagged to
the month end at which an investor could have known them.

Rebuild with ``python -m assetpred.data.predictors``. Writes
``data/processed/predictors.parquet`` (wide, month-end index) and
``data/processed/predictor_meta.csv`` (one row per column).

Point-in-time conventions
-------------------------
* ``realtime == "market"``: prices and yields that are never revised. Month-end
  observation from daily data where available; a few are monthly averages of
  daily data (known at month end, but staler) and say so in the description.
* ``realtime == "vintage"``: rebuilt from ALFRED vintages. The value at t is the
  latest observation in the vintage in force at t; growth rates are computed
  inside that vintage. These are already as-of, so ``lag_months`` is 0.
* ``realtime == "lagged_revised"``: today's revised data, shifted forward by
  ``lag_months`` (the month m value is placed at m + lag) so that it is not
  used before its first release. Where ALFRED has vintages for part of the
  history, values are instead placed at their actual first-release month (never
  earlier than the fixed lag), which catches delayed releases such as the
  2013, 2018-19 and 2025 shutdowns. Revisions remain a (smaller) leak.
* A month with no new release keeps the latest published value for up to
  ``FFILL_LIMIT`` months (that value was known); nothing is filled past a
  series' last observation.
* ``hindsight == True``: the series is estimated over the full sample (factor
  models, full-sample regressions or principal components) or re-estimated
  every vintage, so even when lagged its history embeds later information.
  Exclude these in a robustness check.

Price-based features (momentum, trend, reversal, realised volatility of the
asset returns) are deliberately not built here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from .. import config, schema
from . import predictors_sources as S

PANEL_START = "1926-01-31"
REALTIME = ("vintage", "lagged_revised", "market")
META_COLUMNS = ["name", "family", "scope", "asset", "source", "lag_months", "realtime",
                "hindsight", "first_date", "last_date", "description"]


@dataclass(frozen=True)
class Spec:
    name: str
    family: str
    asset: str | None          # None -> scope "global"
    source: str
    lag: int                   # months the series is shifted forward
    realtime: str
    loader: Callable[[], pd.Series]
    description: str
    hindsight: bool = False
    lagged_in_loader: bool = False  # loader already applied `lag` to its slow inputs

    @property
    def scope(self) -> str:
        return "global" if self.asset is None else "asset"


# ------------------------------------------------------------------ helpers
_ALFRED: dict[tuple, dict[str, pd.Series]] = {}


def _alfred(series_id: str, start: str, end: str | None = None) -> dict[str, pd.Series]:
    key = (series_id, start, end)
    if key not in _ALFRED:
        _ALFRED[key] = S.alfred_asof(series_id, start, end,
                                     {"yoy": ("logdiff", 12), "d3": ("logdiff", 3)})
    return _ALFRED[key]


def _timed(values: Callable[[], pd.Series], alfred_id: str, start: str, lag: int) -> Callable[[], pd.Series]:
    """Revised values placed at their first-release month (ALFRED vintage dates
    from ``start``), never earlier than ``lag`` months after the reference month."""
    return lambda: S.release_timed(values(), _alfred(alfred_id, start)["refdate"], lag)


def _gsw(col: str) -> Callable[[], pd.Series]:
    return lambda: S.month_end_last(S.fed_curve("nominal")[col])


def _tips(col: str) -> Callable[[], pd.Series]:
    return lambda: S.month_end_last(S.fed_curve("tips")[col])


def _cpi_lagged() -> pd.Series:
    """CPI-U NSA placed at the month it is first known (release month m+1)."""
    cpi = S.fred_monthly("CPIAUCNS")
    cpi.index = cpi.index + pd.offsets.MonthEnd(1)
    return cpi


def _us_dy() -> pd.Series:
    d = S.shiller()["D"].dropna()
    d.index = d.index + pd.offsets.MonthEnd(4)
    p = S.sp500_month_end()
    return (d / p).dropna()


def _us_ey() -> pd.Series:
    e = S.shiller()["E"].dropna()
    e.index = e.index + pd.offsets.MonthEnd(6)
    p = S.sp500_month_end()
    return (e / p).dropna()


def _us_cape() -> pd.Series:
    sh = S.shiller()
    cpi = S.fred_monthly("CPIAUCNS")
    real_e = (sh["E"] / cpi).dropna()
    avg10 = real_e.rolling(120, min_periods=120).mean().dropna()
    avg10.index = avg10.index + pd.offsets.MonthEnd(6)   # earnings known ~6 months later
    real_p = S.sp500_month_end() / _cpi_lagged()          # deflate by CPI known at t
    return (real_p / avg10).dropna()


def _devx_dy() -> pd.Series:
    div = S.french_intl_market(True) - S.french_intl_market(False)
    return div.rolling(12, min_periods=12).sum().dropna()


def _devx_bm() -> pd.Series:
    bm = S.french_intl_bm()                         # row y: B(FY y-1) / ME(Dec y-1)
    px = (1 + S.french_intl_market(False)).cumprod()  # price-only index
    out = {}
    for t in px.index:
        y = t.year if t.month >= 6 else t.year - 1  # ratio row y usable from June y
        dec = pd.Timestamp(f"{y - 1}-12-31")
        if y in bm.index and dec in px.index:
            out[t] = bm[y] * px[dec] / px[t]
    return pd.Series(out, dtype=float)


def _gold_real() -> pd.Series:
    return (S.gold_monthly() / _cpi_lagged()).dropna() * 100


def _wti_roll() -> pd.Series:
    f1, f2 = S.eia_wti_future(1), S.eia_wti_future(2)
    both = pd.concat([f1, f2], axis=1, keys=["f1", "f2"]).dropna()
    r = 12 * np.log(both["f1"] / both["f2"])
    return S.month_end_last(r)


def _usd() -> pd.Series:
    major = S.fred_daily_me("DTWEXM")
    afe = S.fred_daily_me("DTWEXAFEGS")
    afe = afe[afe.index >= "2006-01-31"]
    link = afe.index[0]
    chained = major[link] * afe / afe[link]
    return pd.concat([major[major.index < link], chained]).sort_index()


def _retail_rt() -> pd.Series:
    old = _alfred("RETAIL", "1966-02", "2001-05")["yoy"]
    new = _alfred("RSAFS", "2001-06")["yoy"]
    return pd.concat([old[old.index < new.index[0]], new])


def _vrp() -> pd.Series:
    vix = S.cboe_vix()
    svar = S.goyal_welch()["svar"]
    return ((vix / 100) ** 2 / 12 - svar).dropna() * 1e4


def _umcsent() -> pd.Series:
    s = S.fred_monthly("UMCSENT")
    full = s.reindex(pd.date_range(s.index[0], s.index[-1], freq="ME"))
    return full.ffill(limit=2)   # pre-1978 survey is quarterly: carry the latest reading


def _weekly(series_id: str, days: int, window: int = 1) -> Callable[[], pd.Series]:
    return lambda: S.weekly_asof(S.fred_raw(series_id), days, window)


def _bw(col: str) -> Callable[[], pd.Series]:
    return lambda: S.baker_wurgler()[col].dropna()


F = config.FEATURE_FAMILIES
VC, MA, SE = "valuation_carry", "macro", "sentiment"
assert {VC, MA, SE} <= set(F)

FRED = "FRED"
GSW = "Fed GSW nominal curve (feds200628)"
GSWR = "Fed GSW TIPS curve (feds200805)"
ALFRED = "ALFRED vintages"

SPECS: list[Spec] = [
    # ============================================================ valuation_carry
    Spec("tbill_3m", VC, None, f"{FRED} DTB3", 0, "market", lambda: S.fred_daily_me("DTB3"),
         "3-month T-bill secondary-market rate, discount basis, % p.a.; last trading day of month."),
    *[Spec(f"ust_zero_{m:02d}y", VC, a, f"{GSW} SVENY{m:02d}", 0, "market", _gsw(f"SVENY{m:02d}"),
           f"{m}-year zero-coupon Treasury yield (Gurkaynak-Sack-Wright fitted curve, continuously "
           f"compounded, %); last business day of month. Each day's curve is fitted to that day's prices only.")
      for m, a in [(1, None), (2, None), (3, None), (5, "ust_int"), (7, "ust_int"),
                   (10, "ust_long"), (20, "ust_long"), (30, "ust_long")]],
    Spec("term_spread_10y_3m", VC, None, f"{FRED} DGS10, DTB3", 0, "market",
         lambda: (S.fred_daily_me("DGS10") - S.fred_daily_me("DTB3")).dropna(),
         "10-year constant-maturity Treasury yield minus 3-month T-bill rate, percentage points, month end."),
    *[Spec(f"acm_tp_{m:02d}y", VC, a, f"NY Fed ACM model ACMTP{m:02d}", 0, "lagged_revised",
           (lambda m=m: S.acm()[f"ACMTP{m:02d}"].dropna()),
           f"Adrian-Crump-Moench {m}-year Treasury term premium, %, month end. [HINDSIGHT] Affine model "
           f"whose parameters are estimated over the full sample and re-estimated as data arrive, so past "
           f"values are revised and embed later information.", hindsight=True)
      for m, a in [(2, None), (5, "ust_int"), (10, "ust_long")]],
    Spec("kw_tp_10y", VC, "ust_long", f"{FRED} THREEFYTP10", 0, "lagged_revised",
         lambda: S.fred_daily_me("THREEFYTP10"),
         "Kim-Wright 10-year Treasury term premium, %, month end. [HINDSIGHT] Three-factor affine model "
         "estimated over the full sample with survey data; history is revised.", hindsight=True),
    *[Spec(f"tips_real_{m:02d}y", VC, "tips", f"{GSWR} TIPSY{m:02d}", 0, "market", _tips(f"TIPSY{m:02d}"),
           f"{m}-year zero-coupon TIPS real yield (Gurkaynak-Sack-Wright, %), month end. Starts 1999.")
      for m in (5, 10)],
    *[Spec(f"breakeven_{m:02d}y", VC, "tips", f"{GSWR} BKEVEN{m:02d}", 0, "market", _tips(f"BKEVEN{m:02d}"),
           f"{m}-year zero-coupon breakeven inflation (GSW nominal minus TIPS curve, %), month end.")
      for m in (5, 10)],
    Spec("breakeven_5y5y", VC, "tips", f"{GSWR} BKEVEN5F05", 0, "market", _tips("BKEVEN5F05"),
         "5-year breakeven inflation 5 years forward (GSW, %), month end."),
    Spec("aaa_yield", VC, "ig_credit", f"{FRED} AAA", 0, "market", lambda: S.fred_monthly("AAA"),
         "Moody's seasoned Aaa corporate bond yield, %; MONTHLY AVERAGE of daily yields (known at month end, "
         "staler than a month-end print)."),
    Spec("baa_yield", VC, "ig_credit", f"{FRED} BAA", 0, "market", lambda: S.fred_monthly("BAA"),
         "Moody's seasoned Baa corporate bond yield, %; monthly average of daily yields."),
    Spec("baa_aaa_spread", VC, None, f"{FRED} BAA, AAA", 0, "market",
         lambda: (S.fred_monthly("BAA") - S.fred_monthly("AAA")).dropna(),
         "Default spread: Moody's Baa minus Aaa yield, pp; monthly averages."),
    Spec("baa_ust10_spread", VC, "ig_credit", f"{FRED} BAA, GS10", 0, "market",
         lambda: (S.fred_monthly("BAA") - S.fred_monthly("GS10")).dropna(),
         "Moody's Baa yield minus 10-year constant-maturity Treasury, pp; both monthly averages."),
    Spec("gz_spread", VC, None, "Federal Reserve, Gilchrist-Zakrajsek (ebp_csv)", 1, "lagged_revised",
         lambda: S.ebp()["gz_spread"].dropna(),
         "Gilchrist-Zakrajsek corporate credit spread (average spread of senior unsecured bonds of US "
         "non-financial firms over matched synthetic Treasuries, pp), all rating grades. Lagged 1 month "
         "for publication. Not model-estimated."),
    Spec("us_div_yield", VC, "us_eq", "Shiller ie_data D; S&P 500 month-end close (Goyal-Welch, FRED SP500)", 4,
         "lagged_revised", _us_dy,
         "S&P 500 trailing 12-month dividends / month-end index level. Dividends (Shiller, interpolated from "
         "quarterly totals) are lagged 4 months; price is not lagged. lag_months refers to the dividend input.", lagged_in_loader=True),
    Spec("us_earn_yield", VC, "us_eq", "Shiller ie_data E; S&P 500 month-end close", 6, "lagged_revised", _us_ey,
         "S&P 500 trailing 12-month reported earnings / month-end index level. Earnings are lagged 6 months "
         "(quarterly reports arrive with a lag and Shiller interpolates within the quarter); price is not lagged.", lagged_in_loader=True),
    Spec("us_cape", VC, "us_eq", "Shiller ie_data E; FRED CPIAUCNS; S&P 500 month-end close", 6, "lagged_revised",
         _us_cape,
         "Cyclically adjusted P/E: real month-end S&P 500 (deflated by CPI known at t) / 10-year average real "
         "earnings ending 6 months before t. Recomputed here rather than taken from Shiller's column, which "
         "uses unlagged earnings and an average price.", lagged_in_loader=True),
    Spec("us_bm_djia", VC, "us_eq", "Goyal-Welch PredictorData (b/m)", 1, "lagged_revised",
         lambda: S.goyal_welch()["b/m"].dropna(),
         "Dow Jones Industrials book-to-market (Goyal-Welch): prior-year-end book (two years back in Jan-Feb) "
         "over month-end price; shifted a further month. Ends with the Goyal-Welch update (2025-12)."),
    Spec("devx_div_yield", VC, "dev_eq", "Ken French international indices (Ind_all, with and without dividends)",
         0, "market", _devx_dy,
         "Developed ex-US (EAFE + Canada) trailing 12-month dividend yield: sum over 12 months of USD total "
         "return minus price return of French's value-weighted international index. French updates the file "
         "annually, so the series ends at the last update (2025-12)."),
    Spec("devx_bm", VC, "dev_eq", "Ken French international indices (Ind_all ratios)", 6, "lagged_revised",
         _devx_bm,
         "Developed ex-US market book-to-market: French's December B/M (book for fiscal year y-1), used from "
         "June of year y (6-month accounting lag), rolled forward monthly with the price-only index. "
         "Ends 2025-12.", lagged_in_loader=True),
    Spec("gold_real", VC, "gold", "datahub gold-prices monthly; FRED CPIAUCNS", 0, "market", _gold_real,
         "Real gold price: monthly-average USD gold price / CPI-U NSA known at t (prior month's index) x 100. "
         "Gold is datahub's monthly series (an average over the month, not a month-end fix)."),
    Spec("wti_roll_yield", VC, "commodities", "EIA Cushing WTI futures contracts 1 and 2 (RCLC1, RCLC2)", 0,
         "market", _wti_roll,
         "WTI crude futures roll yield, annualised: 12 x ln(F1/F2) on the last common trading day of the month. "
         "Positive = backwardation. EIA stopped publishing the contract series in April 2024."),

    # ===================================================================== macro
    Spec("cpi_yoy", MA, None, f"{FRED} CPIAUCNS", 1, "lagged_revised", _timed(lambda: S.yoy(S.fred_monthly("CPIAUCNS")), "CPIAUCNS", "1949-04", 1),
         "CPI-U all items, not seasonally adjusted, 12-month log change, placed at the month it was first "
         "published (ALFRED release dates from 1949; at least 1 month after the reference month). The NSA "
         "index is not revised after release, so this is effectively real-time.", lagged_in_loader=True),
    Spec("core_cpi_yoy", MA, None, f"{FRED} CPILFENS", 1, "lagged_revised", _timed(lambda: S.yoy(S.fred_monthly("CPILFENS")), "CPIAUCNS", "1949-04", 1),
         "CPI-U less food and energy, NSA, 12-month log change; placed at the headline CPI release month "
         "(same release), at least 1 month after the reference month. NSA index is not revised.",
         lagged_in_loader=True),
    Spec("infl_exp_1y_mich", MA, None, f"{FRED} MICH", 1, "lagged_revised", lambda: S.fred_monthly("MICH"),
         "University of Michigan median expected inflation over the next year, %. Final reading is released "
         "late in the survey month; lagged 1 month to be safe."),
    Spec("ip_yoy_rt", MA, None, f"{ALFRED} INDPRO", 0, "vintage", lambda: _alfred("INDPRO", "1950-01")["yoy"],
         "Industrial production, 12-month log change of the latest month in the vintage in force at t "
         "(release ~mid m+1, so normally reference month t-1)."),
    Spec("payrolls_yoy_rt", MA, None, f"{ALFRED} PAYEMS", 0, "vintage", lambda: _alfred("PAYEMS", "1955-06")["yoy"],
         "Nonfarm payrolls, 12-month log change of the latest month in the vintage in force at t (first Friday "
         "of m+1 release)."),
    Spec("payrolls_3m_rt", MA, None, f"{ALFRED} PAYEMS", 0, "vintage", lambda: _alfred("PAYEMS", "1955-06")["d3"],
         "Nonfarm payrolls, 3-month log change (not annualised) of the latest month, within the vintage in force at t."),
    Spec("unemployment_rt", MA, None, f"{ALFRED} UNRATE", 0, "vintage", lambda: _alfred("UNRATE", "1960-04")["level"],
         "Civilian unemployment rate, %, latest month in the vintage in force at t."),
    Spec("avg_hours_mfg_rt", MA, None, f"{ALFRED} AWHMAN", 0, "vintage", lambda: _alfred("AWHMAN", "1961-12")["level"],
         "Average weekly hours of production workers, manufacturing; latest value in the vintage in force at t."),
    Spec("housing_starts_rt", MA, None, f"{ALFRED} HOUST", 0, "vintage", lambda: _alfred("HOUST", "1960-08")["level"],
         "Housing starts, thousands SAAR; latest value in the vintage in force at t."),
    Spec("building_permits", MA, None, f"{FRED} PERMIT", 1, "lagged_revised", _timed(lambda: S.fred_monthly("PERMIT"), "PERMIT", "1999-09", 1),
         "New private housing permits, thousands SAAR; revised data lagged 1 month, and from Sep 1999 placed "
         "at the actual first-release month from ALFRED (catches shutdown delays).", lagged_in_loader=True),
    Spec("retail_sales_yoy_rt", MA, None, f"{ALFRED} RETAIL (to 2001-05), RSAFS (from 2001-06)", 0, "vintage",
         _retail_rt,
         "Retail sales, 12-month log change within the vintage in force at t; the discontinued SIC-based RETAIL "
         "series until its last vintage, the NAICS-based RSAFS after."),
    Spec("capacity_util", MA, None, f"{FRED} TCU", 1, "lagged_revised", _timed(lambda: S.fred_monthly("TCU"), "TCU", "1996-12", 1),
         "Total industry capacity utilisation, %; revised data lagged 1 month, and from Dec 1996 placed at the "
         "first-release month from ALFRED.", lagged_in_loader=True),
    Spec("cfnai", MA, None, f"{FRED} CFNAI", 1, "lagged_revised", _timed(lambda: S.fred_monthly("CFNAI"), "CFNAI", "2011-06", 1),
         "Chicago Fed National Activity Index; released ~4 weeks after the month, lagged 1 month (first-release "
         "month from ALFRED from mid-2011). [HINDSIGHT] First principal component of 85 indicators with weights "
         "estimated on the full revised sample.", hindsight=True, lagged_in_loader=True),
    Spec("jobless_claims_4wk", MA, None, f"{FRED} ICSA", 0, "lagged_revised", _weekly("ICSA", 5, 4),
         "Initial jobless claims, 4-week average of the last four weeks published by month end (week ending "
         "Saturday, published the following Thursday). Seasonal factors are revised; lag applied in days, so "
         "lag_months is 0."),
    Spec("m2_yoy_rt", MA, None, f"{ALFRED} M2SL", 0, "vintage", lambda: _alfred("M2SL", "1980-03")["yoy"],
         "M2 money stock, 12-month log change within the vintage in force at t (vintages from 1980)."),
    Spec("bank_ci_loans_yoy", MA, None, f"{FRED} BUSLOANS", 1, "lagged_revised", _timed(lambda: S.yoy(S.fred_monthly("BUSLOANS")), "BUSLOANS", "1996-12", 1),
         "Commercial and industrial loans at commercial banks, 12-month log change; revised data lagged 1 month "
         "(first-release month from ALFRED from Dec 1996). Contains reclassification breaks (e.g. 2010).",
         lagged_in_loader=True),
    Spec("consumer_credit_yoy", MA, None, f"{FRED} TOTALSL", 2, "lagged_revised", _timed(lambda: S.yoy(S.fred_monthly("TOTALSL")), "TOTALSL", "1996-12", 2),
         "Total consumer credit outstanding, 12-month log change; G.19 is released ~5 weeks after the month, "
         "so lagged 2 months (first-release month from ALFRED from Dec 1996 when later).", lagged_in_loader=True),
    Spec("fed_funds", MA, None, f"{FRED} FEDFUNDS", 0, "market", lambda: S.fred_monthly("FEDFUNDS"),
         "Effective federal funds rate, monthly average of daily rates, %. (The final day's rate is published "
         "the next morning; immaterial for a monthly average.)"),
    Spec("nfci", MA, None, f"{FRED} NFCI", 0, "lagged_revised", _weekly("NFCI", 5),
         "Chicago Fed National Financial Conditions Index, latest weekly reading published by month end "
         "(Friday reading, published the next Wednesday). [HINDSIGHT] Dynamic factor model re-estimated "
         "each week over the full sample; the whole history is revised.", hindsight=True),
    Spec("anfci", MA, None, f"{FRED} ANFCI", 0, "lagged_revised", _weekly("ANFCI", 5),
         "Adjusted NFCI (purged of the business cycle and inflation), latest weekly reading published by month "
         "end. [HINDSIGHT] Full-sample factor model plus full-sample regression on macro conditions.",
         hindsight=True),
    Spec("stl_fsi", MA, None, f"{FRED} STLFSI4", 0, "lagged_revised", _weekly("STLFSI4", 6),
         "St. Louis Fed Financial Stress Index (v4), latest weekly reading published by month end. "
         "[HINDSIGHT] Principal component of 18 standardised series, re-estimated on the full sample.",
         hindsight=True),
    Spec("kc_fsi", MA, None, f"{FRED} KCFSI", 1, "lagged_revised", lambda: S.fred_monthly("KCFSI"),
         "Kansas City Fed Financial Stress Index, monthly; lagged 1 month. [HINDSIGHT] Principal component "
         "with full-sample loadings.", hindsight=True),
    Spec("usd_major", MA, None, f"{FRED} DTWEXM (to 2005-12), DTWEXAFEGS (from 2006-01)", 0, "market", _usd,
         "Nominal trade-weighted US dollar against major currencies, month end. The Fed's discontinued "
         "major-currency index until Jan 2006, then chained to its successor, the advanced foreign economies "
         "index, at that month (the link uses only data known then)."),
    Spec("wti_spot", MA, None, f"{FRED} WTISPLC", 0, "market", lambda: S.fred_monthly("WTISPLC"),
         "WTI crude spot price, USD/bbl, monthly average (known at month end; staler than a month-end print)."),

    # ================================================================= sentiment
    Spec("vix", SE, None, "CBOE VIX_History.csv", 0, "market", S.cboe_vix,
         "CBOE VIX, close on the last trading day of the month. Starts 1990."),
    Spec("vxo", SE, None, f"{FRED} VXOCLS", 0, "market", lambda: S.fred_daily_me("VXOCLS"),
         "CBOE S&P 100 volatility index (old VIX methodology), month-end close, 1986 to its 2021 discontinuation."),
    Spec("vrp", SE, "us_eq", "CBOE VIX; Goyal-Welch svar", 0, "market", _vrp,
         "Variance risk premium proxy: VIX^2/12 (implied monthly variance) minus realised variance of daily "
         "S&P 500 returns in the same month (Goyal-Welch svar), x 1e4 (monthly %^2). Both known at month end. "
         "Ends with the Goyal-Welch update (2025-12)."),
    Spec("equity_put_call", SE, "us_eq", "CBOE equitypcarchive.csv + equitypc.csv", 0, "market",
         lambda: S.month_mean(S.cboe_equity_pc_daily()),
         "CBOE equity-option put/call volume ratio, monthly average of daily ratios. Free history covers only "
         "Oct 2003 - Oct 2019; definition changed in 2012 (ETF options removed)."),
    Spec("umich_sentiment", SE, None, f"{FRED} UMCSENT", 1, "lagged_revised", _umcsent,
         "University of Michigan consumer sentiment index; lagged 1 month. Quarterly before 1978 (latest "
         "reading carried for up to three months)."),
    Spec("bw_sentiment", SE, None, "Wurgler, Investor_Sentiment_Data (SENT^)", 1, "lagged_revised", _bw("SENT^ "),
         "Baker-Wurgler investor sentiment index, orthogonalised to macro conditions; lagged 1 month; ends "
         "2018-12. [HINDSIGHT] First principal component of the components, estimated over the full sample, "
         "after full-sample regressions on macro variables.", hindsight=True),
    Spec("bw_closed_end_discount", SE, None, "Wurgler, Investor_Sentiment_Data (cefd)", 1, "lagged_revised", _bw("cefd"),
         "Value-weighted closed-end fund discount, % (Baker-Wurgler component, raw); lagged 1 month; ends 2018-12."),
    Spec("bw_ipo_count", SE, None, "Wurgler, Investor_Sentiment_Data (nipo)", 1, "lagged_revised", _bw("nipo"),
         "Number of IPOs in the month (Baker-Wurgler component, raw); lagged 1 month; ends 2018-12."),
    Spec("bw_ipo_first_day_ret", SE, None, "Wurgler, Investor_Sentiment_Data (ripo)", 1, "lagged_revised", _bw("ripo"),
         "Average first-day IPO return, % (Baker-Wurgler component, raw); lagged 1 month; ends 2018-12."),
    Spec("bw_equity_share", SE, None, "Wurgler, Investor_Sentiment_Data (s)", 1, "lagged_revised", _bw("s"),
         "Equity share of new equity and debt issues (Baker-Wurgler component, raw); lagged 1 month; ends 2018-12."),
    Spec("bw_div_premium", SE, None, "Wurgler, Investor_Sentiment_Data (pdnd)", 1, "lagged_revised", _bw("pdnd"),
         "Dividend premium: log difference of average market-to-book of payers and non-payers (Baker-Wurgler "
         "component, raw, annual accounting data); lagged 1 month; ends 2018-12."),
    Spec("net_equity_issuance", SE, "us_eq", "Goyal-Welch PredictorData (ntis)", 1, "lagged_revised",
         lambda: S.goyal_welch()["ntis"].dropna(),
         "Net equity expansion: 12-month moving sum of net issues by NYSE stocks / total end-of-year market "
         "cap (Goyal-Welch, from CRSP); lagged 1 month; ends 2025-12."),
    Spec("margin_debt", SE, "us_eq", "FINRA margin statistics", 1, "lagged_revised",
         lambda: S.finra_margin().iloc[:, 0].dropna().astype(float),
         "Debit balances in customers' securities margin accounts, USD millions (FINRA, from 1997); released "
         "the following month, lagged 1 month."),
    Spec("excess_bond_premium", SE, None, "Federal Reserve, Gilchrist-Zakrajsek (ebp_csv)", 1, "lagged_revised",
         lambda: S.ebp()["ebp"].dropna(),
         "Gilchrist-Zakrajsek excess bond premium, pp; lagged 1 month. [HINDSIGHT] Residual of a full-sample "
         "regression of bond spreads on firm default risk; the history is re-estimated with each update.",
         hindsight=True),
    Spec("epu_news", SE, None, "policyuncertainty.com US news-based EPU", 1, "lagged_revised", S.epu_news,
         "Baker-Bloom-Davis news-based economic policy uncertainty index (historical newspaper index before "
         "1985); lagged 1 month. Scaled to mean 100 over a fixed base period (a constant rescaling)."),
    Spec("gpr", SE, None, "Caldara-Iacoviello GPR (data_gpr_export.xls)", 1, "lagged_revised",
         lambda: S.gpr()["GPR"].dropna(),
         "Geopolitical risk index (10 newspapers, from 1985); lagged 1 month. Scaled to mean 100 over "
         "1985-2019 (a constant rescaling)."),
    Spec("gpr_hist", SE, None, "Caldara-Iacoviello GPR (data_gpr_export.xls)", 1, "lagged_revised",
         lambda: S.gpr()["GPRH"].dropna(),
         "Historical geopolitical risk index (3 newspapers, from 1900); lagged 1 month."),
]


# ------------------------------------------------------------------- build
FFILL_LIMIT = 3  # months a value may be carried while it is still the latest one published


def _lagged(s: pd.Series, lag: int) -> pd.Series:
    s = s.dropna().astype(float)
    s.index = pd.DatetimeIndex(s.index) + pd.offsets.MonthEnd(0)
    s = s[~s.index.duplicated(keep="last")].sort_index()
    if lag:
        s.index = s.index + pd.offsets.MonthEnd(lag)
    # A month with no new release (skipped publication, quarterly survey, no IPOs)
    # still has a latest-known value; carry it a little, never past the last value.
    full = pd.date_range(s.index[0], s.index[-1], freq="ME")
    return s.reindex(full).ffill(limit=FFILL_LIMIT)


def build(specs: list[Spec] = SPECS, end: pd.Timestamp | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    end = S.last_complete_month() if end is None else pd.Timestamp(end)
    index = pd.date_range(PANEL_START, end, freq="ME", name="date")
    cols, meta = {}, []
    for sp in specs:
        assert sp.family in config.FEATURE_FAMILIES and sp.realtime in REALTIME
        assert sp.asset is None or sp.asset in config.ASSETS
        s = _lagged(sp.loader(), 0 if sp.lagged_in_loader else sp.lag).reindex(index)
        if s.notna().sum() == 0:
            raise RuntimeError(f"{sp.name}: no data in panel range")
        cols[sp.name] = s
        valid = s.dropna().index
        meta.append(dict(name=sp.name, family=sp.family, scope=sp.scope, asset=sp.asset or "",
                         source=sp.source, lag_months=sp.lag, realtime=sp.realtime, hindsight=sp.hindsight,
                         first_date=valid[0].date().isoformat(), last_date=valid[-1].date().isoformat(),
                         description=sp.description))
        print(f"{sp.name:26s} {valid[0]:%Y-%m} .. {valid[-1]:%Y-%m}  n={len(valid)}", flush=True)
    panel = pd.DataFrame(cols, index=index)
    panel = panel.loc[panel.notna().any(axis=1).idxmax():]  # drop leading all-empty rows
    return panel, pd.DataFrame(meta, columns=META_COLUMNS)


def main() -> None:
    panel, meta = build()
    schema._check_month_end(panel.index, "predictors")
    schema.PREDICTORS.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(schema.PREDICTORS)
    meta.to_csv(schema.PREDICTOR_META, index=False)
    print(f"wrote {panel.shape[1]} series, {panel.index[0]:%Y-%m} .. {panel.index[-1]:%Y-%m} -> {schema.PREDICTORS}")


if __name__ == "__main__":
    main()
