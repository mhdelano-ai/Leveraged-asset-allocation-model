# Model selection, recorded before the holdout was scored

Written on 2026-11-14, after development on data through 2014 and before any
forecast for an origin after December 2013 had been scored against what
happened. The git history shows this file committed before the holdout
scoring script existed.

## What was chosen

| Role | Covariance model | Volatility | Correlation |
|---|---|---|---|
| **Primary** | `combo_cov_main` | `combo_vol_hier_trim`: mean within each model family, then the mean across families after dropping the highest and lowest family for each asset | `combo_corr_ew`: plain average of 25 correlation forecasts |
| Simple runner-up | `combo_cov_simple` | `combo_vol_cal_m_hl12`: 12-month exponential average of squared monthly returns times one constant re-estimated each December | `bench_corr_d_hl36`: 3-year exponentially weighted correlation of 21-day return sums |
| Best single models | `pair_cov_single` | `statespace_2c_daily` | `mcorr_nls` |
| Reference | `pair_cov_ref` | `ref_blend` | `ref_corr_last120m` |

Bridge to the covariance of 12-month returns: `vr_market`. Tails: the
lognormal scale mixture in `volmodel.bridge`, with its dispersion estimated
walk-forward.

## Why

On data through 2014 the leading volatility models (state-space, the pooled
linear model, the pooled multi-horizon regressions, the family averages)
could not be told apart: a 90% Model Confidence Set kept all twenty
candidates tested. With no detectable winner, the choice went to the
candidate that depends least on any one model being right. The trimmed
family average has no estimated weights, is not dominated by families with
many near-duplicate members, and tied the best single model on the widest
sample with the tightest confidence interval. The averaged correlation
matrix was the only correlation forecast that beat the ten-year reference
significantly.

The cost is operational: the primary needs every model family to run. The
runner-up exists for anyone who wants something that can be rebuilt in an
afternoon; on development data it gave up 3–11% in volatility loss (not
significant) and was clearly worse in 2000–13.

Things tried and rejected before this point, all on development data:
learned combination weights (no better than equal weights), walk-forward
level calibration of the leaders (made them worse), macroeconomic and
option-implied predictors (no detectable value), nonlinear learners (worse
than the linear model on the same features), and dropping the month-average
price eras from the history (neutral to slightly worse).

## What the holdout can and cannot show

The holdout is forecast origins from January 2014 to October 2025: about 142
overlapping 12-month windows, roughly twelve independent years, containing
2020 and 2022. Simulations done for this project show that a true 20%
difference in mean loss would be detected well under half the time in a
sample this short. The holdout is therefore a check that nothing collapses
out of sample, not a re-ranking. Expectations written down in advance:

1. The primary's volatility QLIKE is lower than the best reference's
   (`ref_last12m` and `ref_blend`) on the holdout. A reversal of more than
   the development-period gap would be evidence of overfitting.
2. The primary's covariance loss (`qlike_rel`) is lower than `pair_cov_ref`'s.
3. For fixed-weight portfolios (60/40, balanced, risk parity, random
   long-only) realized variance divided by predicted variance averages
   between 0.75 and 1.35. On development data it was 0.97–1.11.
4. Minimum-variance portfolios built from the forecast realize more variance
   than predicted, by a factor above 1.2. This is expected, not a failure.
5. With the bridge, standardized 12-month returns of the policy portfolios
   have variance nearer 1 than without it — though twelve independent years
   cannot test this with any precision.

Whatever the holdout shows is reported as it comes out. The selection above
is not revised in the light of it.
