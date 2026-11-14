"""Plain-language descriptions of the models and terms the dashboard names.

This is static explanatory text about *what a model is*, never about how it
scored: every result on the page comes from the results files. Keep the keys in
step with ``config`` (``BASELINES``, ``FROZEN_MEAN``, ``FROZEN_DIST``,
``COMPOSITE``, ``FROZEN_COV``); a model without an entry is shown with its
registry name and a generic gloss.
"""
from __future__ import annotations

from .. import config

# model id -> (short label, one-sentence description)
MODELS: dict[str, tuple[str, str]] = {
    # baselines
    "hist_mean": ("Historical mean",
                  "The bar to beat: each asset's expanding average excess return since its data begins, "
                  "with a normal distribution as wide as the standard deviation of its past h-month returns."),
    "zero": ("Zero forecast", "Forecasts an excess return of exactly zero (cash-like), as a sanity reference."),
    "vol_ewma": ("EWMA volatility",
                 "The volatility baseline: historical-mean centre with a normal distribution whose width tracks "
                 "an exponentially weighted average of recent squared returns (decay 0.94), mean-reverting over the horizon."),
    "vol_ewma_fhs": ("EWMA volatility, empirical shape",
                     "As the EWMA volatility baseline, but the shape of the distribution comes from past "
                     "standardized returns (filtered historical simulation) instead of the normal curve."),
    "hist_quantile": ("Historical quantiles",
                      "Quantiles of each asset's past h-month returns, no conditioning on current conditions."),
    # mean models (all predict a standardized target and are shrunk by an out-of-fold calibration slope)
    "ridge_z": ("Ridge regression",
                "Linear regression on all predictors with a ridge penalty and slopes common to all assets."),
    "combo_z": ("Forecast combination",
                "Average of many one-predictor regressions (Rapach-Strauss-Zhou style), per asset group."),
    "gbm_z": ("Gradient boosting",
              "Shallow gradient-boosted trees (LightGBM) on all predictors."),
    "rf_z": ("Random forest", "A shallow random forest on all predictors."),
    "ens_z": ("Ensemble mean",
              "Equal-weight average of the ridge, combination, boosting and random-forest forecasts, then "
              "calibrated. This is the mean half of the composite forecast."),
    "gbm_z.price": ("Boosting, price predictors only",
                    "The gradient-boosting model restricted to price-based predictors (momentum, reversal, "
                    "volatility)."),
    # distribution (scale and shape) models; all are centred on the historical mean
    "dist_ewma_fhs": ("EWMA scale, empirical shape",
                      "Non-ML distribution reference: identical forecasts to the EWMA-volatility baseline with "
                      "empirical shape."),
    "dist_lin_agg": ("Linear volatility model",
                     "A small linear model of next month's variance from recent volatility measures (EWMA, "
                     "12-month, daily realized volatility, VIX, down-moves), aggregated to h months, with an "
                     "empirical shape. This is the scale half of the composite forecast."),
    "dist_lin_agg_grp": ("Linear volatility, shape by asset class",
                         "The linear volatility model with a distribution shape estimated per asset class."),
    "dist_lin_agg_cshape": ("Linear volatility, conditional shape",
                            "The linear volatility model with a shape that may vary with conditions (quantile "
                            "boosting)."),
    "dist_gbm_agg": ("Boosted volatility model",
                     "As the linear volatility model, but the variance model is gradient-boosted trees: the "
                     "machine-learning version of the scale model."),
    "dist_lin_agg_ret": ("Linear volatility, monthly inputs only",
                         "Ablation of the linear volatility model without daily realized volatility or VIX."),
    # covariance models
    "sample": ("Static sample covariance",
               "The covariance benchmark: the expanding sample covariance of past h-month returns."),
    "lw": ("Static shrinkage covariance",
           "The sample covariance with its correlations shrunk toward their average (Ledoit-Wolf). Static, "
           "but better conditioned."),
    "ewma_cov": ("RiskMetrics EWMA covariance",
                 "Exponentially weighted covariance of monthly returns (decay 0.97), scaled by h."),
    "vc_daily_ewma_m_lwb": ("Daily-vol, EWMA-correlation model",
                            "Volatilities from daily realized variance, mean-reverting toward their long-run "
                            "level; correlations are a shrunk long-run level plus the recent EWMA deviation from "
                            "it. The primary covariance model."),
    "vc_ewma_ewma_m_lwb": ("Monthly-only vol and correlation model",
                           "The primary model's structure using monthly data only; its volatilities equal the "
                           "EWMA-volatility baseline."),
    "vc_daily_daily_m_lw": ("Daily-correlation model",
                            "Daily-realized volatilities and a correlation deviation estimated from 126 days of "
                            "daily returns, shrunk toward constant correlation."),
}

# Tags used by the narrative to pick "the ML model" vs "the linear model" without hard-coding names.
ML_DIST = ("dist_gbm_agg",)
LINEAR_DIST = ("dist_lin_agg", "dist_lin_agg_grp", "dist_lin_agg_cshape")
ML_MEAN = ("gbm_z", "rf_z", "gbm_z.price")
LINEAR_MEAN = ("ridge_z", "combo_z")

STRATEGY_RULES = {
    "mv": ("Mean-variance", "Constrained mean-variance (risk aversion 5, long only, at most 100% per asset and "
                            "200% gross). Exposure rises and falls with the forecasts, so it rewards timing."),
    "mv_vt": ("Mean-variance at 10% vol", "The same objective at a fixed 10% ex-ante volatility: rewards only "
                                          "the choice between assets, not how much risk to take."),
    "timing": ("Per-asset timing", "1/N of capital per asset, each sleeve scaled by its forecast Sharpe ratio "
                                   "(Campbell-Thompson weight, clipped to 0-150%). No optimizer."),
    "vt_rp": ("Inverse-vol at 10% vol", "Risk-model test only: inverse-volatility weights scaled to 10% "
                                         "ex-ante volatility."),
    "ew": ("1/N", "Equal weight across the eligible assets, rebalanced monthly."),
    "sixty_forty": ("60/40", "60% US equities, 40% intermediate Treasuries, rebalanced monthly."),
}

TERMS = [
    ("Excess return", "Log return of the asset minus the log return of one-month T-bills, summed over the "
                      "h months after the forecast date."),
    ("Out-of-sample R²", "1 − (squared forecast errors of the model) / (squared errors of the historical mean). "
                         "Positive means the model's point forecasts were closer than the historical mean's. "
                         "For monthly returns, values of a fraction of a percent are economically meaningful."),
    ("Detectable R²", "Roughly the out-of-sample R² a model would need for 5% significance given the number of "
                      "non-overlapping observations. Below it, a true edge of that size could not be told "
                      "apart from luck in this sample."),
    ("CRPS", "Continuous ranked probability score: the error of a whole predictive distribution, averaged "
             "over all quantiles. CRPS skill = 1 − CRPS(model) / CRPS(benchmark); positive is better."),
    ("Pinball loss", "The error of one quantile forecast. Skill by quantile shows which part of the "
                     "distribution (centre or tails) a model improves."),
    ("Coverage", "How often the realized return fell inside a central predictive interval. A calibrated 90% "
                 "interval contains about 90% of outcomes."),
    ("PIT test", "Tests whether realized returns fall at the predicted quantiles as often as they should "
                 "(probability integral transform at the 10th, 50th and 90th percentiles). A small p-value "
                 "means miscalibration."),
    ("Tail hit rate", "Share of outcomes below the 1st/5th or above the 95th/99th predicted percentile; "
                      "the nominal rates are 1% and 5%. The Kupiec test asks whether the rate differs from "
                      "nominal."),
    ("Clark-West test", "Tests whether a model's forecasts contain information beyond the historical mean. "
                        "Read it only together with R²: very noisy forecasts can pass it while losing on R²."),
    ("Diebold-Mariano test", "Tests whether one model's average loss is lower than another's, with "
                             "standard errors that allow for overlapping horizons."),
    ("SPA test", "Hansen's superior predictive ability test: is the best of a whole set of models better than "
                 "the benchmark, allowing for having searched over the set?"),
    ("Model confidence set", "The set of models that cannot be statistically distinguished from the best "
                             "(at 10%)."),
    ("Benjamini-Hochberg q", "A p-value adjusted for testing many cells at once (false discovery rate)."),
    ("Log score (NLL)", "Negative Gaussian log likelihood of the realized return vector under the forecast "
                        "covariance, per asset. Lower is better; the difference against a benchmark is in "
                        "nats per asset."),
    ("CER", "Certainty-equivalent return: the risk-free return an investor with risk aversion 5 would accept "
            "instead of the strategy. Gains are annualized."),
    ("Independent observations", "The number of months divided by the horizon: overlapping 12-month returns "
                                 "share 11 of their 12 months, so a 25-year sample holds only about two "
                                 "dozen independent 12-month outcomes."),
]


def label(model: str) -> str:
    if model == config.COMPOSITE:
        return "Composite forecast"
    if "__sd_" in model:
        c, sd = model.split("__sd_", 1)
        return f"{label(c)}, {label(sd).lower()} vols"
    return MODELS.get(model, (model, ""))[0]


def describe(model: str) -> str:
    if model == config.COMPOSITE:
        mean, dist = composite_parts()
        return (f"The forecast this project would show: the mean of {label(mean)} ({mean}) with the "
                f"distribution width and shape of the {label(dist)} ({dist}).")
    if "__sd_" in model:
        c, s = model.split("__sd_", 1)
        return f"Correlations of {label(c)} ({c}) combined with the volatilities of {label(s)} ({s})."
    return MODELS.get(model, (model, "Model as registered in the repo (no plain-language entry)."))[1]


def composite_parts() -> tuple[str, str]:
    mean, dist = config.COMPOSITE.split("__", 1)
    return mean, dist

# Short asset names for chart axes (full names are config.ASSETS).
SHORT_ASSET = {
    "us_eq": "US equities", "dev_eq": "Dev. ex-US eq.", "em_eq": "EM equities", "ust_int": "Treasuries 5y",
    "ust_long": "Treasuries 20y", "tips": "TIPS 7y", "ig_credit": "IG credit (long)", "hy_credit": "High yield",
    "reits": "REITs", "gold": "Gold", "commodities": "Commodities",
}
