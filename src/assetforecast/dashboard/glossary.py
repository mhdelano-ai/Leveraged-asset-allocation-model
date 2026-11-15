"""Plain-language names for the models, rules, assets and portfolios the dashboard and report name.

This is static text about *what a thing is*, never about how it scored: every
result comes from a results file. Merged from ``assetpred.dashboard.glossary``
(the mean, distribution and strategy entries) with the covariance side's names.
A model without an entry is shown under its registry id.
"""
from __future__ import annotations

# model id -> (short label, one-sentence description)
MODELS: dict[str, tuple[str, str]] = {
    # baselines of the return side
    "hist_mean": ("Historical mean",
                  "The bar to beat, and the published expected return: each asset's expanding average excess "
                  "return since its data begins."),
    "zero": ("Zero forecast", "An excess return of exactly zero, as a sanity reference."),
    "vol_ewma": ("EWMA volatility",
                 "Historical-mean centre with a normal distribution whose width tracks an exponentially "
                 "weighted average of recent squared returns."),
    "vol_ewma_fhs": ("EWMA volatility, empirical shape",
                     "The EWMA volatility baseline with the shape taken from past standardized returns."),
    "hist_quantile": ("Historical quantiles", "Quantiles of each asset's past 12-month returns."),
    # frozen mean models (assetpred; re-run unchanged in the replication)
    "ridge_z": ("Ridge regression", "Linear regression on all predictors with a ridge penalty, slopes common to all assets."),
    "combo_z": ("Forecast combination", "Average of many one-predictor regressions, per asset group."),
    "gbm_z": ("Gradient boosting", "Shallow gradient-boosted trees on all predictors."),
    "rf_z": ("Random forest", "A shallow random forest on all predictors."),
    "ens_z": ("Ensemble", "Equal-weight average of the ridge, combination, boosting and forest forecasts."),
    "gbm_z.price": ("Boosting, price predictors only", "Gradient boosting restricted to momentum, reversal and volatility predictors."),
    "gbm_z.nohind": ("Boosting without hindsight features",
                     "Robustness row: boosting without the features estimated over the full sample. In no test family."),
    "combo_z.nohind": ("Combination without hindsight features",
                       "Robustness row: the combination without the features estimated over the full sample. In no test family."),
    # assetpred's frozen distribution models (archived results only; not ported)
    "dist_ewma_fhs": ("EWMA scale, empirical shape", "Non-ML reference: the EWMA volatility baseline with an empirical shape."),
    "dist_lin_agg": ("Linear volatility model", "A small linear model of next month's variance, aggregated to the horizon, with an empirical shape."),
    "dist_lin_agg_grp": ("Linear volatility, shape by asset class", "The linear volatility model with a shape per asset class."),
    "dist_lin_agg_cshape": ("Linear volatility, conditional shape", "The linear volatility model with a shape that varies with conditions."),
    "dist_gbm_agg": ("Boosted volatility model", "The linear volatility model with boosted trees in place of the regression."),
    # the package's predictive distribution
    "dist_hm_af": ("Published distribution",
                   "Historical-mean centre, scale from the package's 12-month covariance forecast plus the mean's "
                   "own uncertainty, shape from past standardized outcomes."),
    "dist_hm_af.assetshape": ("Published distribution, asset shape",
                              "Policy portfolios with the shape pooled over single assets instead of over portfolios: "
                              "a sensitivity check."),
}

STRATEGY_RULES: dict[str, tuple[str, str]] = {
    "mv": ("Mean-variance", "Constrained mean-variance (risk aversion 5, long only, at most 100% per asset and 200% "
                            "gross). Exposure rises and falls with the forecasts, so it rewards timing."),
    "mv_vt": ("Mean-variance at 10% vol", "The same objective at a fixed 10% ex-ante volatility: rewards only the "
                                          "choice between assets, not how much risk to take."),
    "timing": ("Per-asset timing", "1/N of capital per asset, each sleeve scaled by its forecast Sharpe ratio. No optimizer."),
    "vt_rp": ("Inverse-vol at 10% vol", "Risk-model test only: inverse-volatility weights scaled to 10% ex-ante volatility."),
}

PORTFOLIOS: dict[str, str] = {
    "p6040": "60/40 (US large cap / intermediate Treasuries)",
    "p_balanced": "Balanced 12-asset mix",
    "p_equity": "Global equity mix",
}

PORTFOLIO_SHORT: dict[str, str] = {"p6040": "60/40", "p_balanced": "the balanced mix", "p_equity": "the equity mix"}

SHAPES: dict[str, str] = {
    "portfolios": "Published: shape from past portfolio outcomes",
    "assets": "Sensitivity: shape from past single-asset outcomes",
}

# Short names for chart axes and compact tables.
SHORT_ASSET: dict[str, str] = {
    "us_large": "US large", "us_small": "US small", "intl_dev": "Intl dev", "em": "EM", "ust_int": "UST 7y",
    "ust_long": "UST 20y", "tips": "TIPS", "ig_credit": "IG credit", "hy_credit": "High yield", "reits": "REITs",
    "gold": "Gold", "commodities": "Commod.",
}

# assetpred's eleven asset ids (archived results) -> the package's ids.
ARCHIVE_ASSET_IDS: dict[str, str] = {"us_eq": "us_large", "dev_eq": "intl_dev", "em_eq": "em"}


def label(model: str) -> str:
    return MODELS.get(model, (model, ""))[0]


def describe(model: str) -> str:
    return MODELS.get(model, (model, "Model as registered in the repository."))[1]


def rule_label(rule: str) -> str:
    return STRATEGY_RULES.get(rule, (rule, ""))[0]
