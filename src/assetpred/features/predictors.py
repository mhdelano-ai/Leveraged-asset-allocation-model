"""Valuation/carry, macro and sentiment features from the raw predictor panel.

The raw predictors are already lagged to availability (see the schema
docstring); nothing here lags them again. Raw levels are turned into
stationary, comparable signals with changes, spreads, and standardisations
against each series' own past (expanding or trailing windows only).

Two generic per-asset slots let a pooled model read an asset's own carry and
own valuation from the same column on every row (``carry``, ``value``). Their
asset-by-asset definitions are in ``CARRY_DEF`` and ``VALUE_DEF``; assets with
no credible measure are NaN.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config
from .common import Feat, Inputs, chg, ez, logchg

VC, MACRO, SENT = "valuation_carry", "macro", "sentiment"

# Gold traded at a fixed official price until the early 1970s and US residents
# could not hold it until 1975-01; the real gold price is standardised only
# against its free-market history.
GOLD_FREE_MARKET_START = pd.Timestamp("1975-01-31")


def _real7(p: pd.DataFrame) -> pd.Series:
    """7-year zero TIPS real yield, linear interpolation of the 5y and 10y."""
    return 0.6 * p["tips_real_05y"] + 0.4 * p["tips_real_10y"]


# asset -> (description, sources); the formulas are in _slots
CARRY_DEF = {
    "us_eq": ("S&P 500 dividend yield (%) minus 3m T-bill", ("us_div_yield", "tbill_3m")),
    "dev_eq": ("Developed ex-US trailing dividend yield (%) minus 3m T-bill; ends 2025-12 with the French file", ("devx_div_yield", "tbill_3m")),
    "ust_int": ("5y GSW zero yield minus 3m T-bill", ("ust_zero_05y", "tbill_3m")),
    "ust_long": ("10y GSW zero yield minus 3m T-bill (10y used throughout for a consistent series; the GSW 20y starts only 1981-07, the asset is 15y/20y)", ("ust_zero_10y", "tbill_3m")),
    "tips": ("7y real zero yield (interp. 5y/10y GSW TIPS) + trailing 12m CPI inflation (expected index accrual) minus 3m T-bill", ("tips_real_05y", "tips_real_10y", "cpi_yoy", "tbill_3m")),
    "ig_credit": ("Average of Moody's Aaa and Baa seasoned long corporate yields (monthly averages) minus 3m T-bill", ("aaa_yield", "baa_yield", "tbill_3m")),
    "gold": ("Minus the 3m T-bill: gold pays no income, so its carry over cash (as in the gold futures basis, ignoring lease rates) is -r_f", ("tbill_3m",)),
    "commodities": ("trailing 12-month mean of the WTI crude futures roll yield (12 x ln F1/F2, %), a single-commodity proxy for the basket's basis; already an excess-of-cash carry; ends 2024-04", ("wti_roll_yield",)),
}
VALUE_DEF = {
    "us_eq": ("expanding z of log(1/CAPE)", ("us_cape",)),
    "dev_eq": ("expanding z of log developed ex-US book-to-market; ends 2025-12", ("devx_bm",)),
    "ust_int": ("expanding z of the 5y real yield (5y zero minus trailing 12m CPI inflation)", ("ust_zero_05y", "cpi_yoy")),
    "ust_long": ("expanding z of the 10y real yield (10y zero minus trailing 12m CPI inflation)", ("ust_zero_10y", "cpi_yoy")),
    "tips": ("expanding z of the 7y TIPS real yield (min 60 months, so from 2004)", ("tips_real_05y", "tips_real_10y")),
    "ig_credit": ("expanding z of the Baa minus 10y Treasury spread", ("baa_ust10_spread",)),
    "gold": ("minus the expanding z of the log real gold price (monthly average price / CPI), using free-market history from 1975 only", ("gold_real",)),
}


def _slots(p: pd.DataFrame, assets: list[str]) -> list[Feat]:
    tb = p["tbill_3m"]
    infl = 100 * p["cpi_yoy"]
    carry = pd.DataFrame(np.nan, index=p.index, columns=assets)
    carry["us_eq"] = 100 * p["us_div_yield"] - tb
    carry["dev_eq"] = 100 * p["devx_div_yield"] - tb
    carry["ust_int"] = p["ust_zero_05y"] - tb
    carry["ust_long"] = p["ust_zero_10y"] - tb
    carry["tips"] = _real7(p) + infl - tb
    carry["ig_credit"] = (p["aaa_yield"] + p["baa_yield"]) / 2 - tb
    carry["gold"] = -tb
    # trailing 12-month mean: removes the seasonal pattern in the crude curve and
    # damps one-off spikes (e.g. -830% annualised in 2020-04); NaN once the series ends
    roll = p["wti_roll_yield"]
    carry["commodities"] = (100 * roll.rolling(12, min_periods=12).mean()).where(roll.notna())

    gold = np.log(p["gold_real"]).where(p.index >= GOLD_FREE_MARKET_START)
    value = pd.DataFrame(np.nan, index=p.index, columns=assets)
    value["us_eq"] = ez(-np.log(p["us_cape"]))
    value["dev_eq"] = ez(np.log(p["devx_bm"]))
    value["ust_int"] = ez(p["ust_zero_05y"] - infl)
    value["ust_long"] = ez(p["ust_zero_10y"] - infl)
    value["tips"] = ez(_real7(p))
    value["ig_credit"] = ez(p["baa_ust10_spread"])
    value["gold"] = -ez(gold)

    def doc(defs, empty_note):
        parts = [f"{a}: {d}" for a, (d, _) in defs.items()]
        empty = [a for a in assets if a not in defs]
        return "; ".join(parts) + f". NaN for {', '.join(empty)} ({empty_note})."

    src = lambda defs: tuple(sorted({s for _, ss in defs.values() for s in ss}))  # noqa: E731
    return [
        Feat("carry", VC, carry,
             "Own carry slot, % p.a. over cash (what the asset earns if prices do not move). "
             + doc(CARRY_DEF, "no yield or basis series in the panel"), src(CARRY_DEF)),
        Feat("value", VC, value,
             "Own valuation slot, oriented so high = cheap, standardised against the series' own past (expanding, min 60 months). "
             + doc(VALUE_DEF, "no credible valuation series; commodity value is left to lt_rev_60_12"), src(VALUE_DEF)),
    ]


def _valuation_global(p: pd.DataFrame) -> list[Feat]:
    tb, infl = p["tbill_3m"], 100 * p["cpi_yoy"]
    real10 = p["ust_zero_10y"] - infl
    F = lambda n, v, d, s: Feat(n, VC, v, d, tuple(s))  # noqa: E731
    return [
        F("g_tbill", tb, "3-month T-bill rate, % (level of the cash rate).", ["tbill_3m"]),
        F("g_tbill_chg_12m", chg(tb, 12), "12-month change in the 3m T-bill rate, pp (policy cycle).", ["tbill_3m"]),
        F("g_real_tbill", tb - infl, "3m T-bill minus trailing 12m CPI inflation, % (real short rate / monetary stance).", ["tbill_3m", "cpi_yoy"]),
        F("g_term_spread", p["term_spread_10y_3m"], "10y CMT minus 3m T-bill, pp (yield-curve slope).", ["term_spread_10y_3m"]),
        F("g_term_spread_chg_12m", chg(p["term_spread_10y_3m"], 12), "12-month change in the 10y-3m term spread, pp.", ["term_spread_10y_3m"]),
        F("g_default_spread", p["baa_aaa_spread"], "Moody's Baa minus Aaa yield, pp (monthly averages).", ["baa_aaa_spread"]),
        F("g_default_spread_z", ez(p["baa_aaa_spread"]), "Expanding z of the Baa-Aaa default spread.", ["baa_aaa_spread"]),
        F("g_gz_spread", p["gz_spread"], "Gilchrist-Zakrajsek corporate bond credit spread, pp (all grades).", ["gz_spread"]),
        F("g_gz_spread_chg_3m", chg(p["gz_spread"], 3), "3-month change in the GZ credit spread, pp (credit momentum).", ["gz_spread"]),
        F("g_us_cape_ey_z", ez(-np.log(p["us_cape"])), "Expanding z of log(1/CAPE): US equity valuation as a global risk-appetite state.", ["us_cape"]),
        F("g_us_dy_z", ez(np.log(p["us_div_yield"])), "Expanding z of the log S&P 500 dividend yield.", ["us_div_yield"]),
        F("g_eq_bond_gap", 100 / p["us_cape"] - real10, "Excess CAPE yield: 100/CAPE minus the 10y real yield (10y zero minus trailing CPI inflation), pp.", ["us_cape", "ust_zero_10y", "cpi_yoy"]),
        F("g_acm_tp_10y", p["acm_tp_10y"], "ACM 10-year Treasury term premium, %. Hindsight: full-sample affine model.", ["acm_tp_10y"]),
        F("g_breakeven_10y", p["breakeven_10y"], "10y zero breakeven inflation (GSW), % (from 1999).", ["breakeven_10y"]),
        F("g_breakeven_10y_chg_12m", chg(p["breakeven_10y"], 12), "12-month change in 10y breakeven inflation, pp.", ["breakeven_10y"]),
    ]


def _macro(p: pd.DataFrame) -> list[Feat]:
    F = lambda n, v, d, s: Feat(n, MACRO, v, d, tuple(s))  # noqa: E731
    u = p["unemployment_rt"]
    u3 = u.rolling(3, min_periods=3).mean()
    sahm = u3 - u3.rolling(13, min_periods=13).min()
    return [
        F("cpi_yoy", 100 * p["cpi_yoy"], "Headline CPI 12-month log change, % (as first published).", ["cpi_yoy"]),
        F("cpi_yoy_chg_12m", 100 * chg(p["cpi_yoy"], 12), "12-month change in headline CPI inflation, pp (inflation momentum).", ["cpi_yoy"]),
        F("core_cpi_yoy", 100 * p["core_cpi_yoy"], "Core CPI 12-month log change, %.", ["core_cpi_yoy"]),
        F("infl_exp_1y", p["infl_exp_1y_mich"], "Michigan median 1-year inflation expectation, % (from 1978).", ["infl_exp_1y_mich"]),
        F("ip_yoy", 100 * p["ip_yoy_rt"], "Industrial production 12-month log change, % (real-time vintage).", ["ip_yoy_rt"]),
        F("payrolls_yoy", 100 * p["payrolls_yoy_rt"], "Nonfarm payrolls 12-month log change, % (real-time vintage).", ["payrolls_yoy_rt"]),
        F("payrolls_3m", 400 * p["payrolls_3m_rt"], "Nonfarm payrolls 3-month log change, annualised % (real-time vintage).", ["payrolls_3m_rt"]),
        F("unemp_chg_12m", chg(u, 12), "12-month change in the unemployment rate, pp (each end from the vintage in force at its date).", ["unemployment_rt"]),
        F("unemp_sahm", sahm, "Sahm-type gap: 3-month average unemployment rate minus its minimum over the prior 12 months, pp.", ["unemployment_rt"]),
        F("hours_mfg_chg_12m", chg(p["avg_hours_mfg_rt"], 12), "12-month change in average weekly manufacturing hours (leading indicator).", ["avg_hours_mfg_rt"]),
        F("housing_starts_yoy", 100 * logchg(p["housing_starts_rt"], 12), "Housing starts 12-month log change, %.", ["housing_starts_rt"]),
        F("permits_yoy", 100 * logchg(p["building_permits"], 12), "Building permits 12-month log change, %.", ["building_permits"]),
        F("retail_sales_yoy", 100 * p["retail_sales_yoy_rt"], "Nominal retail sales 12-month log change, % (real-time vintage).", ["retail_sales_yoy_rt"]),
        F("capacity_util_chg_12m", chg(p["capacity_util"], 12), "12-month change in capacity utilisation, pp.", ["capacity_util"]),
        F("cfnai_ma3", p["cfnai"].rolling(3, min_periods=3).mean(), "3-month average of the Chicago Fed National Activity Index. Hindsight: full-sample PCA weights.", ["cfnai"]),
        F("claims_yoy", 100 * logchg(p["jobless_claims_4wk"], 12), "Initial jobless claims (4-week average) 12-month log change, %.", ["jobless_claims_4wk"]),
        F("real_m2_yoy", 100 * (p["m2_yoy_rt"] - p["cpi_yoy"]), "M2 growth minus CPI inflation (12-month log changes), % (from 1980).", ["m2_yoy_rt", "cpi_yoy"]),
        F("ci_loans_yoy", 100 * p["bank_ci_loans_yoy"], "C&I bank loans 12-month log change, % (contains reclassification breaks).", ["bank_ci_loans_yoy"]),
        F("consumer_credit_yoy", 100 * p["consumer_credit_yoy"], "Consumer credit 12-month log change, %.", ["consumer_credit_yoy"]),
        F("fed_funds_chg_12m", chg(p["fed_funds"], 12), "12-month change in the effective fed funds rate, pp.", ["fed_funds"]),
        F("fed_funds_chg_3m", chg(p["fed_funds"], 3), "3-month change in the effective fed funds rate, pp.", ["fed_funds"]),
        F("nfci", p["nfci"], "Chicago Fed NFCI level (positive = tighter than average). Hindsight: full-sample factor model.", ["nfci"]),
        F("nfci_chg_3m", chg(p["nfci"], 3), "3-month change in NFCI. Hindsight: full-sample factor model.", ["nfci"]),
        F("usd_chg_3m", 100 * logchg(p["usd_major"], 3), "Trade-weighted US dollar 3-month log change, %.", ["usd_major"]),
        F("usd_chg_12m", 100 * logchg(p["usd_major"], 12), "Trade-weighted US dollar 12-month log change, %.", ["usd_major"]),
        F("oil_yoy", 100 * logchg(p["wti_spot"], 12), "WTI spot (monthly average) 12-month log change, %.", ["wti_spot"]),
    ]


# Baker-Wurgler components and the sign that makes high = optimistic sentiment
BW_COMPONENTS = {
    "bw_closed_end_discount": -1.0,
    "bw_ipo_count": 1.0,
    "bw_ipo_first_day_ret": 1.0,
    "bw_equity_share": 1.0,
    "bw_div_premium": -1.0,
}


def _sentiment(p: pd.DataFrame, rv_us_eq: pd.Series) -> list[Feat]:
    F = lambda n, v, d, s: Feat(n, SENT, v, d, tuple(s))  # noqa: E731
    lvix = np.log(p["vix"])
    vrp = p["vix"] ** 2 / 12 - rv_us_eq.reindex(p.index)
    comp = []
    for c, sign in BW_COMPONENTS.items():
        x = p[c]
        if c == "bw_ipo_count":
            x = np.log1p(x)
        comp.append(sign * ez(x))
    comp = pd.concat(comp, axis=1)
    bw_pit = comp.mean(axis=1).where(comp.notna().sum(axis=1) >= 4)
    lgpr = np.log(p["gpr_hist"])
    return [
        F("log_vix", lvix, "Log VIX at month end (from 1990).", ["vix"]),
        F("vix_rel_12m", lvix - lvix.rolling(12, min_periods=12).mean(), "Log VIX minus its trailing 12-month mean (fear shock).", ["vix"]),
        F("vrp", vrp, "Variance risk premium: VIX^2/12 minus the realised variance of daily US equity (CRSP VW / VTI) log returns in month t, monthly %^2. "
          "Computed here rather than taken from the predictor 'vrp', which ends 2025-12.", ["vix"]),
        F("umich_sentiment_z", ez(p["umich_sentiment"]), "Expanding z of Michigan consumer sentiment.", ["umich_sentiment"]),
        F("bw_sentiment", p["bw_sentiment"], "Baker-Wurgler orthogonalised investor sentiment; ends 2018-12. Hindsight: full-sample PCA and regressions.", ["bw_sentiment"]),
        F("bw_sentiment_pit", bw_pit, "Point-in-time Baker-Wurgler-style sentiment: mean of expanding z-scores of the raw components "
          "(minus closed-end discount, log1p IPO count, IPO first-day return, equity share, minus dividend premium); needs 4 of 5; ends 2018-12.",
          list(BW_COMPONENTS)),
        F("net_equity_issuance", p["net_equity_issuance"], "Goyal-Welch net equity expansion (12m net issues / market cap); ends 2025-12.", ["net_equity_issuance"]),
        F("margin_debt_yoy", 100 * logchg(p["margin_debt"], 12), "FINRA margin debt 12-month log change, % (leverage-driven sentiment; from 1998).", ["margin_debt"]),
        F("excess_bond_premium", p["excess_bond_premium"], "Gilchrist-Zakrajsek excess bond premium, pp. Hindsight: full-sample regression residual.", ["excess_bond_premium"]),
        F("epu_z", ez(np.log(p["epu_news"])), "Expanding z of the log news-based economic policy uncertainty index.", ["epu_news"]),
        F("gpr_shock", lgpr - lgpr.rolling(12, min_periods=12).mean(), "Log historical geopolitical risk index minus its trailing 12-month mean.", ["gpr_hist"]),
    ]


def predictor_features(inp: Inputs, rv_us_eq: pd.Series) -> list[Feat]:
    # put predictors on the return calendar first; nothing here shifts them
    p = inp.predictors.reindex(inp.predictors.index.union(inp.returns.index))
    assets = list(config.ASSETS)
    return _slots(p, assets) + _valuation_global(p) + _macro(p) + _sentiment(p, rv_us_eq)
