# The bridge: from `cov_rv` to `cov_annual`

Code: `src/assetforecast/bridge/__init__.py` (estimators, bridges, predictive distribution),
`src/assetforecast/bridge/validate.py` (every table below; `python -m assetforecast.bridge.validate`),
tests `tests/test_bridge.py`. All numbers: `ASSETFORECAST_MAX_DATE=2014-12-31`, full monthly run
(1,116 origins 1922-2014), estimation walk-forward from the view only.

## 1. What the bridge does

`cov_rv` is 12 x the covariance of one-month returns. The 12-month return is a sum of 12
monthly returns, so with standardized monthly returns `u` (return / ex-ante vol) and
lag-k cross-correlations `G_k`:

    cov_annual = S (R + D) S,   D = sum_{k=1..11} (1 - k/12) (G_k + G_k')

`VR_i = 1 + D_ii` is the variance ratio (12-month variance / 12 x monthly variance).

```python
from assetforecast.bridge import get_bridge, vol_uncertainty, predictive_quantiles
A  = get_bridge("vr_market").cov_annual(cov_rv, view)     # DataFrame, same assets, PD
s2 = vol_uncertainty(view)["s2"]                          # tail parameter (section 5)
q  = predictive_quantiles(A, weights, s2)                  # 5% / 1% quantiles of R - mu
```

### Variants (5)

| name | variance ratios estimated from | annual correlation |
|---|---|---|
| `identity` | none (VR = 1) | = 21-day correlation |
| `vr_all` | all monthly history | = 21-day correlation |
| **`vr_market`** (default) | months priced like today's instrument (below) | = 21-day correlation |
| `vr_recent` | last 360 months of market-priced data | = 21-day correlation |
| `lr_market` | market-priced months | 21-day corr + shrunk annual-minus-monthly gap |

`vr_*`: `cov_annual = V^1/2 cov_rv V^1/2`, `V = diag(shrunk VR)`. An earlier draft used the
additive form `S(R + diag(VR-1))S`. That form lowers every annual correlation by
`1/sqrt(VR_i VR_j)` (us_large-us_small would go from 0.88 to 0.68), and the data
contradict it: the raw annual correlation is 0.87. It was replaced before any calibration
results were looked at in the final form.

### Estimator (no tuned constants)

1. **Data:** monthly log excess returns from the view only. They exist in every era and
   do not depend on the pricing convention at this horizon.
2. **Heteroskedasticity:** demean, then divide by an ex-ante EWMA vol of past squared
   monthly returns. The half-life is 12 months, fixed in advance to match the target
   window; the first 12 months only initialize it. Every month then carries equal weight,
   as every origin does in the calibration. The raw (unstandardized) estimator is a
   variance-weighted average dominated by the 1930s, 1970s and 2008.
3. **Overlapping estimator:** Bartlett-weighted autocorrelations over all lags 1-11. Each
   demeaned autocorrelation gets the small-sample bias correction `(n-k)/(n(n-1))`, or
   `rho_0 (n-k)/(n(n-1))` for a cross term. Without it the estimator is biased down by
   about (q-1)/n.
4. **Standard error:** the Lo-MacKinlay heteroskedasticity-robust form, multiplied by
   VR-hat. The overlapping estimator is a Bartlett spectral estimate, whose relative error
   is constant. With the null form alone, a 95% interval covered only 77-86% when VR was
   about 1.5-1.7, and was far too wide when VR was about 0.6. With the scaling the stated
   error matches the simulated dispersion within 10% in every case tested (table below).
   The iid formula gives `se = VR x sqrt(14.1/n)`: 0.24 at 20 years and 0.12 at 80 years
   when VR = 1.
5. **Shrinkage:** empirical Bayes toward 1 across the assets in the view,
   `tau^2 = max(0, mean[(VR-1)^2 - se^2])` and `w_i = tau^2/(tau^2 + se_i^2)`. Group
   pooling was not implemented: within-group estimation errors are highly correlated
   because equities share the same years, so a group mean adds little information.
6. **Re-estimation:** at most once a year, from the view cut at the last December. It is
   cached on a hash of the data, so cold and warm calls give the same answer.

Simulation check (80 series of 60 years each, `tests/test_bridge.py`; 300 series in the
development run):

| process | true VR | mean estimate (n=960) | actual sd | stated se | 95% coverage |
|---|---|---|---|---|---|
| AR(1) +0.3 | 1.755 | 1.69 | 0.18 | 0.20 | 0.93 |
| AR(1) -0.3 | 0.568 | 0.59 | 0.07 | 0.07 | 0.96 |
| MA(1) 0.5 (smoothed price) | 1.733 | 1.69 | 0.20 | 0.21 | 0.93 |
| AR(1) 0.2 + stochastic vol | 1.448 | 1.41 | 0.21 | 0.21 | 0.92 |
| iid | 1.000 | 1.01 | 0.12 | 0.12 | 0.94 |

A residual downward bias of about 4% remains when VR is about 1.7. The EWMA standardizer
damps persistent runs. The estimate is therefore slightly conservative.

## 2. Variance ratios per asset (view at 2014-12)

Columns: `n` = standardized market-priced months; `raw` = bias-corrected estimate with its
robust `se`; `shrunk` = the value applied. The `strict` column also drops the thin-trading
eras. Under `recent` the EB prior variance is 0, so all assets shrink to 1.00 at this view.

| asset | n | raw (market) | se | shrunk (market) | raw (all) | shrunk (all) | raw (strict) | shrunk (strict) | raw (last 30y) | se |
|---|---|---|---|---|---|---|---|---|---|---|
| us_large | 1062 | 1.39 | 0.17 | **1.26** | 1.63 | 1.53 | 1.39 | 1.19 | 1.30 | 0.29 |
| us_small | 1050 | 1.27 | 0.15 | **1.20** | 1.27 | 1.24 | 1.19 | 1.03 | 0.91 | 0.18 |
| intl_dev | 468 | 1.56 | 0.30 | **1.22** | 1.56 | 1.35 | 1.56 | 1.13 | 1.73 | 0.39 |
| em | 294 | 1.39 | 0.33 | **1.14** | 1.39 | 1.22 | 1.39 | 1.08 | 1.39 | 0.33 |
| ust_int | 636 | 1.41 | 0.24 | **1.21** | 1.45 | 1.33 | 1.41 | 1.13 | 0.97 | 0.19 |
| ust_long | 636 | 1.36 | 0.23 | **1.19** | 1.52 | 1.42 | 1.36 | 1.12 | 0.89 | 0.19 |
| tips | 203 | 1.10 | 0.32 | **1.04** | 1.10 | 1.06 | 1.10 | 1.02 | 1.10 | 0.32 |
| ig_credit | 1068 | 1.56 | 0.20 | **1.33** | 1.67 | 1.51 | 1.56 | 1.23 | 1.06 | 0.25 |
| hy_credit | 92 | 1.39 | 0.77 | **1.04** | 1.38 | 1.22 | 1.39 | 1.02 | 1.39 | 0.77 |
| reits | 504 | 1.22 | 0.22 | **1.12** | 1.22 | 1.17 | 1.32 | 1.03 | 1.23 | 0.26 |
| gold | 508 | 1.34 | 0.23 | **1.17** | 1.34 | 1.25 | 1.34 | 1.11 | 1.24 | 0.25 |
| commodities | 1050 | 1.55 | 0.20 | **1.33** | 1.55 | 1.43 | 1.55 | 1.23 | 1.41 | 0.30 |

`tau^2` by era: all 0.145, market 0.058, strict 0.027, recent 0.

Walk-forward shrunk VR (`vr_market`) is fairly stable over time. us_large ranges
1.34 / 1.28 / 1.28 / 1.23 / 1.24 / 1.26 at the end of 1950 / 1970 / 1990 / 2000 / 2007 /
2014, and ig_credit 1.39-1.44. The Treasury ratios rise from 1.0-1.05 (1950-70) to
1.2-1.27 (2000-14). hy_credit stays at 1.00 until it has market-priced data (HYG, 2007).

Every raw ratio is above 1, at 1.1-1.6. Most are individually 1.5-2.8 se above 1. Equal
ratios across assets are not rejected: the cross-sectional dispersion is barely larger
than the noise, which is why the shrinkage is substantial.

## 3. The era / pricing-convention question

Raw VR by convention and period:

| asset | segment | years | VR | se |
|---|---|---|---|---|
| us_large | month-average prices | 1872-1926 | 1.97 | 0.32 |
| us_large | market | 1926-1984 / 1985-2014 | 1.45 / 1.30 | 0.22 / 0.29 |
| ust_long | month-average yields | 1920-1961 | 1.75 | 0.32 |
| ust_long | market | 1962-1984 / 1985-2014 | 1.33 / 0.89 | 0.35 / 0.19 |
| ust_int | market | 1962-1984 / 1985-2014 | 1.28 / 0.97 | 0.35 / 0.19 |
| ig_credit | month-average yields | 1920-1925 | 2.83 | 1.36 |
| ig_credit | market | 1926-1984 / 1985-2014 | 1.69 / 1.06 | 0.26 / 0.25 |
| hy_credit | VWEHX NAV (smoothed) | 1981-2007 | 1.47 | 0.34 |
| hy_credit | HYG | 2007-2014 | 1.39 | 0.77 |
| us_small | CRSP, thin trading | 1927-2000 | 1.29 | 0.16 |
| us_small | IWM | 2000-2014 | 1.19 | 0.38 |
| reits | Nareit / FRESX / VGSIX (thin) | 1973-2004 | 1.21 | 0.23 |
| reits | VNQ | 2004-2014 | 1.32 | 0.56 |
| commodities | market | 1927-1984 / 1985-2014 | 1.62 / 1.41 | 0.26 / 0.30 |
| gold | market | 1972-1984 / 1985-2014 | 1.55 / 1.24 | 0.51 / 0.25 |
| intl_dev | market | 1976-1984 / 1985-2014 | 1.04 / 1.73 | 0.38 / 0.39 |

Local-mean diagnostic (`vr_local`): VR around a centered 120-month moving mean instead of
the full-sample mean. Values: us_large 1.28, us_small 1.22, intl_dev 1.67, em 1.34,
ust_int 1.11, ust_long 1.06, ig_credit 1.32, reits 1.21, gold 1.06, commodities 1.33. This
estimate is biased down by about 0.1 by the filter itself.

**Answer.**

- **Month-average segments are excluded.** Averaging mechanically adds lag-1
  autocorrelation of about 0.25 (us_large 1.97, ust_long 1.75, ig 2.83).
- **VWEHX is excluded.** It is a matrix-priced high-yield fund NAV, documented as smoothed
  for weeks.
- **These exclusions define `market`, the default.** The rule is fixed from the documented
  pricing mechanism in `NON_MARKET` (segments.csv), not from returns.
- **Thin-trading eras are kept.** These are exchange-traded constituents with stale
  prices lasting days, not months: CRSP small caps before 2000 and REIT funds before VNQ.
  Their VRs (1.29, 1.21) are no larger than the market-priced ones of the same asset
  class, so excluding them (`strict`) only removes data.
- **"Only recent decades" is not adopted.** It changes the answer for bonds: Treasury and
  IG VRs fall to about 0.9-1.06 after 1985. The local-mean diagnostic shows why: for
  bonds and gold most of the full-sample excess over 1 comes from decade-long drifts in
  the mean, such as the 1950-81 rise in yields and the fall after. For equities and
  commodities it is short-run persistence (vr_local about 1.2-1.3).
- **The planner's mean decides which is right.** If the planner uses a constant long-run
  expected return, those drifts are genuine dispersion around that mean, and `vr_market`
  is the right choice. If the planner updates expected returns with the rate regime, the
  bond ratios should be about 1.05.
- **The bridge assumes a constant mean.** That is also what `calibration_summary`
  evaluates (z demeaned by its full-sample mean).

## 4. Correlations at the annual horizon

The gap is `g_ij = annual correlation - monthly correlation` of standardized returns, with
`se = (1 - rho^2) sqrt(2 sum_k (1-k/12)^2 / n)`. I derived this from the joint moments of
the cross- and auto-correlation estimators under no lead-lag. It is small for highly
correlated pairs because their autocorrelations move together. The last two columns are a
model-free check: the raw correlation of monthly returns and of overlapping 12-month sums.

| pair | n | monthly | annual | gap | se | t | applied | raw 1m | raw 12m |
|---|---|---|---|---|---|---|---|---|---|
| us_large-ust_long | 636 | 0.14 | 0.19 | +0.05 | 0.10 | 0.5 | 0 | 0.12 | 0.19 |
| us_large-ust_int | 636 | 0.13 | 0.16 | +0.03 | 0.10 | 0.3 | 0 | 0.14 | 0.17 |
| us_large-intl_dev | 468 | 0.63 | 0.61 | -0.02 | 0.07 | -0.3 | 0 | 0.64 | 0.69 |
| us_large-em | 294 | 0.69 | 0.40 | -0.28 | 0.08 | -3.5 | 0 | 0.72 | 0.51 |
| us_large-us_small | 1050 | 0.88 | 0.87 | -0.01 | 0.02 | -0.6 | 0 | 0.91 | 0.91 |
| us_large-hy_credit | 92 | 0.74 | 1.09 | +0.35 | 0.13 | 2.8 | 0 | 0.76 | 0.87 |
| us_large-reits | 504 | 0.58 | 0.62 | +0.04 | 0.08 | 0.5 | 0 | 0.58 | 0.65 |
| ig_credit-hy_credit | 92 | 0.70 | 0.67 | -0.02 | 0.14 | -0.2 | 0 | 0.69 | 0.82 |
| ust_int-ig_credit | 636 | 0.77 | 0.86 | +0.09 | 0.04 | 2.1 | 0 | 0.80 | 0.86 |
| gold-commodities | 508 | 0.40 | 0.46 | +0.06 | 0.10 | 0.6 | 0 | 0.39 | 0.54 |
| ust_long-tips | 203 | 0.59 | 0.33 | -0.26 | 0.12 | -2.2 | 0 | 0.58 | 0.46 |

Across all 66 pairs, 9.1% have |t| > 2 (4.6% expected with no lead-lag effect). The mean
gap is -0.035 and the mean |gap| is 0.10, compared with a typical se of about 0.10. The
empirical-Bayes variance of true gaps is 0 at the 2014 view and at most anchors (0.018 in
1980 and 0.006 in 2005, where the largest applied gaps are 0.12 and 0.15). So `lr_market`
equals `vr_market` almost everywhere.

**Conclusion.** Annual correlations are not distinguishable from 21-day correlations in
this data, with three candidates:

- **us_large-em, -0.28.** This rests on 1990-2014 only and is driven by the 1990s
  divergence.
- **ust_long-tips, -0.26.** This is 17 years of data.
- **us_large-hy, +0.35.** This is 2007-2014, essentially one crisis. The annual estimate
  exceeds 1, which is noise.

Given 66 tests, two or three |t| > 2 values are expected by chance. None is applied.

The design choice that does matter is keeping annual correlation equal to 21-day
correlation. The additive VR-only form would understate the annual correlation of related
assets by 15-25% and make diversification look better than it is.

## 5. Predictive distribution (tails)

Model: `R - mu = sqrt(V) Z`, where `V = A eps`, `log eps ~ N(-s2/2, s2)` and Z is
standard normal. Because `E[eps] = 1`, `A` (cov_annual) stays the variance. Implied
kurtosis is `3 exp(s2)`.

`s2` is calibrated walk-forward (`vol_uncertainty`) from the errors of the reference
forecast `ref_blend`, measured against the realized 12-month variance `sum r_m^2`. The
calculation is `s2 = Var(e - m) - mean(v)`, where m and v are the mean and variance of
`log Gamma(a, 1/a)`. Here `a = n_eff/2` and `n_eff` is the effective number of months in
the window, measured scale-free from daily data. Without this correction one crash month
carrying most of a year's variance makes the 12-observation chi-square correction too
small: 0.67 versus 0.61.

A Jensen-gap estimator gives a cross-check that does not require `ref_blend` to be
unbiased. In simulation both recover the dispersion of the true integrated variance to
within 0.03 (tests).

Real data:

- **2014 view:** `s2` = 0.61, Jensen-gap cross-check 0.72, from 7,103 asset-origins.
- **Walk-forward range:** 0.30-1.39, median 0.80.

At `s2 = 0.61`, relative to a normal with the same variance:

| s2 | kurtosis | q05 / normal | q01 / normal | q0.5% / normal | P(abs > 1 sd) | P(abs > 2 sd) |
|---|---|---|---|---|---|---|
| 0 (normal) | 3.0 | 1 | 1 | 1 | 0.317 | 0.046 |
| 0.45 | 4.7 | 0.98 | 1.11 | 1.16 | 0.271 | 0.053 |
| **0.61** | 5.5 | **0.97** | **1.14** | 1.21 | 0.258 | 0.053 |
| 0.80 | 6.7 | 0.96 | 1.17 | 1.26 | 0.244 | 0.054 |

At the same variance the 5% loss is about 3% smaller than the normal's and the 1% loss
about 14% larger.

**Assumptions:**

- **Mean.** The mean is outside this project. The quantiles are of `R - mu` with `mu`
  supplied by the planner and treated as known. Mean uncertainty would widen them, and
  the planner should add the quantile to their own mean.
- **Symmetry.** The mixture is symmetric, so negative skew is not modeled.
- **Reference model.** `s2` is calibrated to `ref_blend` errors. A better vol model
  deserves a smaller `s2`, so with such a model these tails are conservative.

Validation: reference input with `vr_market`, market-priced windows, all assets and
policy/optimized portfolios pooled (13,152 overlapping z values). In the "scaled" rows z
is divided by its own sample sd, which tests shape only. In the "raw" rows z is in units
of the predicted sigma.

| | P(abs > 1) | P(abs > 2) | P(z < normal 5% quantile) | P(z < mixture 5% quantile) | P(z < normal 1% quantile) | P(z < mixture 1% quantile) | kurtosis |
|---|---|---|---|---|---|---|---|
| normal prediction | 0.317 | 0.046 | 0.05 | - | 0.01 | - | 3 |
| mixture prediction (median s2) | 0.244 | 0.054 | - | 0.05 | - | 0.01 | 6.7 |
| vr_market, scaled (shape) | 0.252 | 0.053 | 0.059 | 0.063 | 0.026 | 0.018 | 5.4 |
| vr_market, raw | 0.308 | 0.078 | 0.076 | 0.082 | 0.040 | 0.029 | 5.5 |
| identity, raw | 0.351 | 0.094 | 0.089 | 0.097 | 0.046 | 0.034 | 5.2 |

**Shape.** The realized standardized 12-month returns are clearly leptokurtic (kurtosis
about 5.4), and the mixture matches the 1-sd and 2-sd shares and the 1% tail much better
than the normal: 1.8% versus 2.6% realized below the quantile, against 1% nominal. Both
models overshoot at 5% (5.9-6.3%) because the realized distribution is negatively skewed.
The mixture's `s2` (median 0.80) is somewhat too large for the realized kurtosis, which
corresponds to about 0.45.

**Raw.** In predicted-sigma units, every share is too high because the reference vol
forecast under-predicts (section 6). With `vr_market` and the mixture the realized share
below the 1% quantile is 2.9%, against 4.6% for identity and the normal.

**Limits.** The 1% tail rests on about a dozen independent bad years per asset, so the
1% numbers cannot separate s2 = 0.45 from 0.8.

## 6. Calibration of standardized 12-month returns

The statistic is `calibration_summary` var(z), sample-demeaned, with a 90% moving-block
(24-month) bootstrap interval. Ideal = 1. Origins run 1922-2013, and only windows whose
assets are market-priced are used.

**Oracle input** (realized 21-day covariance of the same window, daily-based assets,
1926+). This removes the vol-forecast error and isolates the bridge. The values are
divided by the simulated null E[z^2] = 0.96, the value z = R/sqrt(RV) would have with a
correct bridge. Unconstrained GMV is omitted: a 12 x 12 realized matrix from about 11
independent 21-day sums is near-singular by construction.

| portfolio | n | identity | vr_all | **vr_market** | vr_recent | lr_market |
|---|---|---|---|---|---|---|
| us_large | 1051 | 1.32 [1.10, 1.50] | 0.79 [0.67, 0.90] | **1.12 [0.92, 1.29]** | 1.20 [1.00, 1.37] | 1.12 |
| us_small | 1051 | 1.29 [1.09, 1.50] | 0.97 [0.80, 1.11] | **1.10 [0.92, 1.28]** | 1.17 | 1.10 |
| intl_dev | 337 | 1.23 [0.92, 1.54] | 0.98 | **1.02 [0.75, 1.27]** | 1.12 | 1.02 |
| em | 236 | 1.12 [0.93, 1.45] | 0.94 | **0.99 [0.81, 1.27]** | 1.07 | 0.99 |
| ust_int | 625 | 1.21 [1.03, 1.42] | 0.99 | **1.09 [0.92, 1.30]** | 1.15 | 1.09 |
| ust_long | 625 | 1.15 [0.95, 1.33] | 0.76 [0.63, 0.88] | **1.02 [0.85, 1.20]** | 1.08 | 1.02 |
| tips | 204 | 1.13 [0.77, 1.33] | 1.13 | **1.13** | 1.12 | 1.13 |
| ig_credit | 409 | 1.20 [0.94, 1.38] | 0.75 [0.59, 0.85] | **0.85 [0.67, 0.95]** | 1.07 | 0.85 |
| hy_credit (HYG only) | 81 | 1.12 [0.50, 1.59] | 0.91 | **1.10** | 1.12 | 1.10 |
| reits | 326 | 1.18 [0.91, 1.49] | 1.01 | **1.04 [0.81, 1.30]** | 1.10 | 1.04 |
| gold | 509 | 1.42 [1.18, 1.63] | 1.24 [1.01, 1.43] | **1.28 [1.03, 1.47]** | 1.36 | 1.28 |
| commodities | 276 | 1.30 [0.99, 1.73] | 0.90 | **0.95 [0.72, 1.25]** | 1.15 | 0.95 |
| p6040 | 625 | 1.20 [0.99, 1.39] | 0.78 [0.65, 0.89] | **1.01 [0.82, 1.18]** | 1.14 | 1.01 |
| p_balanced | 625 | 1.11 [0.92, 1.30] | 0.76 [0.63, 0.89] | **0.94 [0.79, 1.11]** | 1.04 | 0.95 |
| p_equity | 337 | 1.06 [0.76, 1.38] | 0.76 [0.55, 0.99] | **0.86 [0.63, 1.12]** | 1.01 | 0.88 |
| ew | 1051 | 1.30 [1.09, 1.52] | 0.90 [0.76, 1.04] | **1.11 [0.93, 1.29]** | 1.17 | 1.13 |
| invvol | 1051 | 1.31 [1.11, 1.52] | 0.91 [0.77, 1.04] | **1.14 [0.94, 1.32]** | 1.19 | 1.15 |
| erc | 1051 | 1.30 [1.09, 1.50] | 0.91 [0.77, 1.04] | **1.13 [0.94, 1.31]** | 1.17 | 1.15 |
| gmv_long | 1051 | 1.32 [1.12, 1.51] | 0.98 [0.83, 1.10] | **1.18 [0.99, 1.34]** | 1.21 | 1.18 |
| random (pooled) | 1051 | 1.34 [1.16, 1.53] | 0.94 [0.82, 1.06] | **1.16 [0.99, 1.33]** | 1.21 [1.04, 1.37] | 1.17 |

The oracle results by origin era are below (median over assets, then the random
portfolios). They confirm the effect is not one episode:

| era | identity | vr_market |
|---|---|---|
| 1926-45 | 1.82 / 1.84 | 1.70 / 1.72 |
| 1946-79 | 1.32 / 1.29 | 1.10 / 1.07 |
| 1980-99 | 1.14 / 1.34 | 1.02 / 1.11 |
| 2000-13 | 1.06 / 1.01 | 0.85 / 0.84 |

The 1930s are under-predicted by every bridge. In 2000-13 there was no excess annual
variance, and `vr_market` over-predicts there.

**Reference input** (`ref_blend` vols x `ref_corr_last120m`), market-priced windows:

| portfolio | identity | vr_all | **vr_market** | vr_recent | lr_market |
|---|---|---|---|---|---|
| p6040 | 1.26 [0.92, 1.72] | 0.82 [0.57, 1.11] | **1.02 [0.73, 1.40]** | 1.20 | 1.02 |
| p_balanced | 1.30 [0.83, 1.91] | 0.92 | **1.07 [0.71, 1.60]** | 1.24 | 1.10 |
| p_equity | 1.34 [0.79, 2.13] | 0.96 | **1.09 [0.65, 1.79]** | 1.30 | 1.13 |
| ew | 1.46 [0.96, 2.06] | 1.05 | **1.28 [0.82, 1.80]** | 1.36 | 1.29 |
| erc | 1.59 [1.11, 2.20] | 1.15 | **1.40 [0.96, 1.95]** | 1.48 | 1.41 |
| gmv_long | 2.08 [1.57, 2.68] | 1.51 | **1.82 [1.36, 2.36]** | 1.90 | 1.82 |
| gmv | 4.80 [2.27, 8.23] | 2.69 | **4.53 [2.07, 7.89]** | 4.64 | 5.30 |
| random (pooled) | 1.51 [1.08, 2.05] | 1.09 | **1.31 [0.88, 1.79]** | 1.40 | 1.32 |
| us_large / ust_long / ig_credit | 1.66 / 1.98 / 2.00 | 0.98 / 1.32 / 1.21 | **1.42 / 1.79 / 1.61** | | |

With reference inputs every bridge stays above 1 for most single assets and optimized
portfolios. Comparing with the oracle rows, the remaining excess comes from the reference
vol and correlation forecasts. For example, GMV at 4.5 is a correlation-forecast problem,
and 2000-13 gives 1.6-2.7 because 2008 was under-predicted. The bridge removes about the
same share of the excess under both inputs.

## 7. Recommendation and confidence

**Default: `vr_market`.** It was chosen before validation. The validation supports it
and selected nothing.

**Confidence that VR > 1 on average: high.**

- With the vol-forecast error removed (oracle), identity gives var(z) of about 1.3 for
  every diversified portfolio, and the 90% intervals exclude 1 (random pooled
  [1.16, 1.53]).
- Var(z) is above 1 in 1926-45, 1946-79 and 1980-99.
- Every raw asset VR is above 1.

**Confidence in the level: moderate.**

- `vr_market` leaves the oracle portfolios at about 1.1-1.18, with intervals touching 1.
- `vr_all` over-corrects to about 0.9. Its us_large, ust_long and ig ratios include the
  month-average eras, and those single-asset ratios fall to 0.75-0.79 with intervals
  excluding 1.
- The truth for a diversified portfolio is probably slightly above `vr_market`.

**Confidence in per-asset differences: low.** No asset's ratio is distinguishable from the
common level. The shrinkage reflects that.

**Choose `vr_recent` for bonds only if** the planner's expected returns follow rate
regimes (section 3).

## 8. What the data cannot distinguish

- **Per-asset VR differences.** With se of 0.15-0.35, the market-era ratios (1.1-1.6) are
  consistent with a common value of about 1.3. hy_credit (92 HYG months) and tips (17
  years) are essentially unidentified, and both get about 1.04.
- **Momentum versus regime shifts in the mean.** For bonds and gold, the excess over 1 is
  mostly decade-scale drift in the mean. Whether that is "variance" depends on the mean
  the planner uses; the data cannot settle it.
- **Annual versus monthly correlations.** No pair is reliably different. The three
  nominally significant gaps rest on 8-25 years of data.
- **The portfolio-level residual** in the oracle tests (about 1.15) could come from
  positive lead-lag effects across assets that are too weak to see pair by pair,
  estimation noise, or the 1930s. It cannot be attributed.
- **Era stability.** 2000-13 has no excess annual variance, and the 1930s have far more
  than any estimate. With about 90 independent years overall, a ratio that changes by
  era cannot be ruled out.
- **Tails.** The 1% quantile rests on about a dozen independent events per series. s2
  between 0.45 and 0.8 fits the data equally well.

## 9. Tried and dropped

| idea | what happened |
|---|---|
| Additive VR-only form `S(R + diag(VR-1))S` | Lowers annual correlations by `1/sqrt(VR_i VR_j)`, contradicted by the data. Replaced by `V^1/2 cov_rv V^1/2` before the final validation. |
| Shrinking lead-lag sums `D_ij` toward 0 | Equivalent to the additive form for pairs. Replaced by shrinking the correlation gap toward 0, so the null is "annual corr = monthly corr". |
| PD repair by plain eigenvalue clipping | Unconstrained GMV built from cross-adjusted matrices exploited near-null directions (var(z) up to 59 at step 6). Replaced by `floor_relative`: no portfolio's annual correlation-part variance goes below 0.5x its 21-day one. 0.5 is a round bound far from any estimate. |
| Null-only standard errors | Under-covered at VR about 1.7 (77%). Now scaled by VR. |
| chi2(12) noise correction for s2 | Gave 0.67. It ignores within-year concentration of variance; the n_eff correction gives 0.61. |
| Group pooling of VRs | Not implemented. Within-group estimation errors are almost perfectly correlated (equities share the same years). |
| Excluding thin-trading eras (`strict`) | Reported as sensitivity only. Their VRs are not larger than market-priced ones. |

## 10. Tests, run time, caveats

- **Tests:** `tests/test_bridge.py` has 24 tests, all passing in about 10 s (one is
  marked `realdata`). `tests/test_contract.py` passes. Coverage:
  - VR recovery within stated error for AR(1) +-0.3, an MA(1) smoothed price and iid.
  - Robustness to stochastic volatility, and the ex-ante standardizer.
  - z-calibration: on AR(1) data the bridge gives var(z) of about 1 and identity about
    1.57.
  - iid data gives a bridge equal to identity within 12%.
  - Common autocorrelation does not lower correlations.
  - PD and asset order on well- and ill-conditioned inputs.
  - A 36-month asset is heavily shrunk; fewer than 24 months gives VR = 1.
  - No look-ahead, cold equals warm, yearly re-estimation, the convention mask, the
    portfolio floor.
  - Mixture moments and quantiles against Monte Carlo, and `s2` recovery.
- **Run time:** about 11 ms per bridge call, or 12 s for all 1,116 origins on one core.
  The full validation (5 bridges, 2 inputs, 4 z-score sets) takes 11.5 min wall time on
  shared cores.
- **Caveats:**
  - `cov_rv` targets are uncentered, so they include `12 mu^2`, about 1-2% of variance.
    The bridge treats them as centered.
  - The hy_credit oracle row covers HYG windows only. In the VWEHX era even the 21-day
    realized variance is smoothed.
  - The calibration tables use the same history the walk-forward estimates learn from.
    Estimates use only data before each origin, but the eras overlap heavily, so these
    are not independent out-of-sample confirmations.
  - Nothing after 2014 was used.
