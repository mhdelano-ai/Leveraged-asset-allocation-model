"""Constants for the return side (expected returns, the mean-model replication,
distributions and the strategy test).

Ported from ``assetpred.config``. What changed and why:

* ``ASSETS`` is the package universe (``assetforecast.universe``), 12 ids;
  ``HORIZONS`` is 12 months only.
* The distribution, composite and covariance model lists are gone (those
  models were not carried over); ``SPA_FAMILIES`` holds the mean family only.
* The single dev/holdout split is replaced by the period table ``PERIODS``.
* New: ``ASSET_GROUP`` (the mean models' pooling groups, keyed by the new ids),
  ``FROZEN_FEATURE_COLUMNS``, ``RETURN_DROP_TIERS``, ``RETURN_START``,
  ``REPLICATION_LABEL``, ``FROZEN_AT``.

Everything that sets what a frozen model computes (``QUANTILES``,
``OOS_START``, ``REFIT_EVERY``, ``MIN_ASSET_HISTORY``, ``SEED``, the model
lists) has the value it had when the models were frozen.

This module deliberately does not import ``assetforecast.paths``: the runner
sets ``ASSETFORECAST_MAX_DATE`` before that module is first imported.
"""
from __future__ import annotations

import pandas as pd

from ..universe import ASSETS as _UNIVERSE

# Asset id -> name, in universe order. Ids are the column names of every
# returns panel and the `asset` level of every (date, asset) index.
ASSETS = {a.id: a.name for a in _UNIVERSE}

HORIZONS = (12,)  # months ahead
QUANTILES = (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)
FEATURE_FAMILIES = ("price", "valuation_carry", "macro", "sentiment")

# Return panel: the package panel without tier D (month-average eras), from
# 1926-07-31. Passed explicitly as `drop_tiers`; never set globally.
RETURN_DROP_TIERS = "D"
RETURN_START = "1926-07-31"

# Walk-forward evaluation (unchanged from assetpred)
OOS_START = "1990-01-31"     # first forecast origin that is scored
REFIT_EVERY = 12             # months between refits; expanding window
MIN_ASSET_HISTORY = 60       # months of an asset's own returns before it is forecast
SEED = 20261114

# Pooling groups of the mean models (``models.ml_mean``). REITs are "equity"
# here, as in assetpred; this is NOT ``universe.Asset.group`` (where REITs are
# "real"). Must cover exactly ``universe.ASSET_IDS``; unknown ids raise.
ASSET_GROUP = {
    "us_large": "equity", "us_small": "equity", "intl_dev": "equity", "em": "equity",
    "reits": "equity",
    "ust_int": "rates", "ust_long": "rates", "tips": "rates",
    "ig_credit": "credit", "hy_credit": "credit",
    "gold": "real", "commodities": "real",
}

# The model lists frozen in assetpred on development evidence, before either
# holdout was scored. Re-run here unchanged as a replication.
BASELINES = ("hist_mean", "vol_ewma", "hist_quantile", "vol_ewma_fhs", "zero")
FROZEN_MEAN = ("ridge_z", "combo_z", "gbm_z", "rf_z", "ens_z", "gbm_z.price")
# Robustness rows: the same learners without the features estimated over the
# full sample. Scored and reported, but in no test family.
ROBUSTNESS = ("gbm_z.nohind", "combo_z.nohind")
REPLICATION_MODELS = (*BASELINES, *FROZEN_MEAN, *ROBUSTNESS)

# Families for the "does any model beat the benchmark" (SPA) tests.
SPA_FAMILIES = {"mean": FROZEN_MEAN}

# The features `select.columns()` returned on assetpred's full archived panel
# (`results/archive/assetpred/data`): every feature except those whose data
# ended more than 12 months before the panel's last date, i.e. all 84 but
# `bw_sentiment` and `bw_sentiment_pit` (Baker-Wurgler, ends 2019-01). Pinned so
# the frozen models see the same feature set whatever `ASSETFORECAST_MAX_DATE` is.
FROZEN_FEATURE_COLUMNS = (
    "mom_1m", "mom_3m", "mom_6m", "mom_12m", "mom_12_1", "lt_rev_60_12", "mom_1m_vs", "mom_3m_vs",
    "mom_12_1_vs", "trend_ma10", "trend_ma3_12", "dd_12m", "dd_max", "vol_m_12m", "vol_m_36m",
    "vol_d_1m", "vol_d_3m", "vol_d_ratio", "beta_eq_36m", "xs_rank_mom_12_1_vs",
    "xs_rank_mom_3m_vs", "xs_rank_lt_rev", "g_eq_mom_12_1", "g_bond_mom_12_1", "g_cmdty_mom_12_1",
    "g_eq_vol_d_1m", "g_eq_dd_max", "g_sb_corr_d_3m", "g_sb_corr_m_36m", "g_trend_breadth",
    "carry", "value", "g_tbill", "g_tbill_chg_12m", "g_real_tbill", "g_term_spread",
    "g_term_spread_chg_12m", "g_default_spread", "g_default_spread_z", "g_gz_spread",
    "g_gz_spread_chg_3m", "g_us_cape_ey_z", "g_us_dy_z", "g_eq_bond_gap", "g_acm_tp_10y",
    "g_breakeven_10y", "g_breakeven_10y_chg_12m", "cpi_yoy", "cpi_yoy_chg_12m", "core_cpi_yoy",
    "infl_exp_1y", "ip_yoy", "payrolls_yoy", "payrolls_3m", "unemp_chg_12m", "unemp_sahm",
    "hours_mfg_chg_12m", "housing_starts_yoy", "permits_yoy", "retail_sales_yoy",
    "capacity_util_chg_12m", "cfnai_ma3", "claims_yoy", "real_m2_yoy", "ci_loans_yoy",
    "consumer_credit_yoy", "fed_funds_chg_12m", "fed_funds_chg_3m", "nfci", "nfci_chg_3m",
    "usd_chg_3m", "usd_chg_12m", "oil_yoy", "log_vix", "vix_rel_12m", "vrp", "umich_sentiment_z",
    "net_equity_issuance", "margin_debt_yoy", "excess_bond_premium", "epu_z", "gpr_shock",
)

REPLICATION_LABEL = ("Replication on the 12-asset panel, frozen models unchanged; both holdouts "
                     "already spent, so not out-of-sample evidence")

# The date the assetpred models were merged into this package. Only forecasts
# whose whole 12-month target lies after it are out-of-sample evidence for the
# return side: origins from OOS_ORIGIN_START (the month-end of FROZEN_AT's month,
# whose first target month starts after FROZEN_AT).
FROZEN_AT = "2026-11-14"
OOS_ORIGIN_START = str((pd.Timestamp(FROZEN_AT) + pd.offsets.MonthEnd(0)).date())

# Periods by forecast origin (label -> first origin, last origin), inclusive.
#   dev         the years the models were chosen on (targets realized by 2014-12)
#   holdout     the covariance side's 2014-2025 holdout (origins 2014-01..2025-10)
#   holdout_ap  assetpred's own holdout, for row-by-row comparison with it
#   post        origins after the holdouts; their targets were NOT unseen at FROZEN_AT
#   oos         origins whose whole target lies after FROZEN_AT: the only evidence
_OPEN_START, _OPEN_END = "1900-01-31", "2099-12-31"
DECADES = {
    "1990s": ("1990-01-01", "1999-12-31"),
    "2000s": ("2000-01-01", "2009-12-31"),
    "2010s": ("2010-01-01", "2019-12-31"),
    "2020s": ("2020-01-01", "2029-12-31"),
}
PERIODS = {
    "dev": (_OPEN_START, "2013-12-31"),
    "holdout": ("2014-01-31", "2025-10-31"),
    "holdout_ap": ("2015-01-31", "2025-10-31"),
    "post": ("2025-11-30", _OPEN_END),
    "oos": (OOS_ORIGIN_START, _OPEN_END),
    **DECADES,
}
# Periods for which the multiple-testing adjustments are computed.
MT_PERIODS = ("full", "dev", "holdout", "holdout_ap")


def subperiods() -> tuple[tuple[str, str, str], ...]:
    """``PERIODS`` in the ``(label, start, end)`` form ``scoring`` takes."""
    return tuple((k, s, e) for k, (s, e) in PERIODS.items())
