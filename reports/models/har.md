# HAR family: direct multi-horizon regressions for the 12-month variance

Code: `src/assetforecast/models/har/` (`features.py` regressors, `model.py` estimation,
`__init__.py` registry). Tests: `tests/models/test_har.py`. All numbers below are
from the development panel (`ASSETFORECAST_MAX_DATE=2014-12-31`, results in `results/dev/`),
every month-end origin, scored on targets observed by 2014-12.

## Summary

* Eight pre-specified models, all of kind `vol`. On every subset and in every era
  after 1945 all pooled variants beat every reference by a wide margin. They lose
  to `ref_last12m` in 1926-1945, and that era (four Depression years in particular)
  is why the gain on the `daily` subset is not significant.
* `daily` subset, 1,051 common origins (1926-06..2013-12): best HAR `har_pooled_qlike_d`
  QLIKE 0.391 vs best reference `ref_last12m` 0.505. Skill 22.6%, 90% CI
  [-11.2, 43.9] (Fieller-EWC), DM t = -1.27, p = 0.22.
* `all` subset, 1,104 common origins (1922-01..2013-12): best HAR `har_pooled_logcvx_d`
  QLIKE 0.395 vs best reference `ref_blend` 0.537. Skill 26.4%, 90% CI [13.2, 34.8],
  DM p = 0.011. Every pooled variant is significant at 5% against `ref_blend`.
* Horizon weights (pooled, daily-mapped, 2014 fit): the last year (1-, 3- and 12-month
  windows) gets 0.64 (log) to 0.78 (QLIKE). The long run (5 years plus full history)
  gets 0.36 to 0.22. Since about 1950 the 1- and 3-month weights have been stable. Weight
  has moved from the 12-month window to the 5-year window over time (see below).
* What the comparisons within the family show: pooling clearly beats per-asset
  regressions. Daily-mapped regressors beat monthly ones, but not significantly
  overall; the gain is concentrated after 2000. The convex constraint helps a little,
  and the help comes only from the early fits, where it binds. Fitting in logs and
  fitting by QLIKE in levels are indistinguishable. Asset intercepts add nothing.

## Definitions

### Regressors (`features.py`)

All regressors are variances on the target's scale: the annualized variance of 21-day
log excess returns. Each one is measured over a trailing window that ends at the
origin `t`, from the view only. One vectorized function computes them for every
month-end of a view. The training row at `s` is therefore exactly what a forecast at
`s` would have seen.

| name | definition | used in |
|---|---|---|
| `LR` | 12 · mean(r_m²) over the asset's whole monthly history to `t` | normalization, "long run" |
| `M60`, `M12`, `M3` | 12 · mean(r_m²) over the last 60 / 12 / 3 months (M60 uses what exists if the history is shorter) | monthly set |
| `D1`, `D3`, `D12` | D_yr · v1(h) · q over the last 1 / 3 / 12 months | daily set |

* `v1(h)` = Σ x_d² / n_rows over the window. Daily log excess returns are used, a missing
  day counts as 0, and every panel row counts, exactly as in the target.
* `D_yr` = the number of panel daily rows in the last 12 months (about 252, or about 295
  before 1952). This is the target's own annualization.
* `q` maps 1-day variance to the 21-day scale. It is the asset's trailing ratio
  q_raw = [mean of squared overlapping 21-row sums / 21] / v1, measured over
  (t − 36 months, t] and **cut at the start of the asset's current tier run** (`view.tiers`),
  so a pre-splice scale is never carried across a tier change. q_raw is then shrunk
  toward 1 in logs: q = q_raw^(n/(n+18)), with n the window length in months.
* **Daily set available** only if all of the following hold: the tier at `t` is A or B; the
  current tier run is at least 12 months long, so all three daily windows and the q
  window lie inside one run; and daily returns cover at least 90% of the rows of the
  last 12 months and of the q window (the target's coverage rule).
* Monthly 1-month squared returns are deliberately excluded. log χ²₁ has variance 4.9,
  far above the signal, so the regressor carries almost no information. A check
  confirmed this (see "Tried and dropped").
* Measurement choices: short windows (1 and 3 months) are measured from daily data or
  not at all, since one or three monthly squares are mostly noise. The 12-month window
  uses daily data where it can, which gives about 9% relative error in variance against
  about 41% from 12 squared monthly returns. The 60-month window and the long run use
  monthly data in every era, which keeps them immune to splices and consistent with the
  target scale.

The q values reproduce the horizon evidence. Examples: about 1.3-2.1 for pre-2000
`us_small` (smoothed), about 1.4-1.8 for VWEHX `hy_credit`, about 1.75 for FRESX `reits`,
about 0.8 for SPY-era `us_large`, and about 0.67-0.8 for ETFs in 2008-2011.

### Model forms (`model.py`)

Everything is normalized by the asset's own long-run variance, R_h = RV_h / LR, so that
coefficients are comparable across assets and pooling is meaningful.

* **log**: log(y/LR) = c + Σ_h b_h · max(log R_h, log 0.05) + e, estimated by OLS.
  The forecast is ŷ = LR · exp(c + Σ b_h log R_h) · φ, where φ = mean(exp(ê)) over the
  training rows (Duan's smearing estimator).
  - Retransformation: QLIKE is minimized by E[y | x], and E[y | x] = exp(x'b) · E[exp(e)].
    Smearing estimates that factor nonparametrically, with no normality assumption.
    Fitted φ ≈ 1.35-1.45.
  - Dropping φ, i.e. forecasting exp(E[log y]) (roughly the median), costs +0.077 QLIKE.
  - The implied weight on the long run is 1 − Σb. The forecast is a geometric
    combination LR^(1−Σb) · Π RV_h^b_h, times exp(c)·φ.
* **logcvx**: the same model with b_h ≥ 0 and Σ b_h ≤ 1 (a convex combination of
  horizons plus a pull to the long run). It is solved as a small QP (SLSQP) and the
  intercept stays free.
* **qlike**: ŷ = κ · (w_0 · LR + Σ_h w_h · RV_h), with w = softmax(θ) (so w ≥ 0 and
  Σw = 1) and κ > 0. It is fitted by minimizing the mean QLIKE of y against ŷ over the
  training rows (L-BFGS, analytic gradient). The form is linear in variances, so no
  retransformation is needed. κ is the QLIKE-optimal common scale (fitted 1.23-1.45; see
  "Calibration").

### Pooling

* **pooled**: one coefficient vector for all assets.
* **fe**: pooled slopes plus asset intercepts, shrunk toward the common one:
  δ_a = mean residual_a · n_a / (n_a + 120). φ is recomputed after the shift.
* **asset**: the asset alone when it has at least 120 training origins, otherwise the
  pooled fit.

### Training (walk-forward, inside the model)

* At origin `t` the model uses the coefficients estimated at e = the last December 31
  ≤ `t`, from `view.asof(e)`. That is one re-estimation per year; the regressors are
  updated monthly.
* Training rows are every (month-end s, asset) with:
  - at least 36 months of monthly history at s (the harness's eligibility rule), and
  - a realized target that is fully observed in that view (s + 12 months ≤ e).
* The targets are computed by `assetforecast.eval.targets.compute_targets` on the truncated
  view. This is the scored estimator itself (daily-based or monthly proxy), derived
  from the view and never from cached files.
* The daily set is fitted on rows where it is available. The monthly set is fitted on
  all rows.
* Rows overlap (monthly origins, 12-month targets). That does not bias the point
  estimates, and no standard errors are used.

### Forecast routing (handles unequal histories)

Each asset is forecast by the first route in this chain that works:

1. The daily-set fit, if the asset has daily regressors at `t` (D models only).
2. The monthly-set fit.
3. The prior: √(LR · M60), the `ref_blend` variance.

A fit exists once it has at least 120 distinct training origins. With the per-asset
option, "fit" means the asset's own fit when it exists, else the pooled one. An asset
that has just reached 36 months of monthly history gets the pooled coefficients
applied to its own regressors (LR and M60 then both cover its 36 months). In the
development backtest, the daily fits start at the end of 1938 and the monthly fits at
the end of 1884. Before that the prior is used.

### Registered models

| model | regressors | form | pooling | contrast it isolates |
|---|---|---|---|---|
| `har_pooled_log_m` | monthly: M3, M12, M60 (+LR) | log | pooled | baseline, monthly information only |
| `har_pooled_log_d` | daily-mapped: D1, D3, D12, M60 (+LR) | log | pooled | value of daily data (vs `_log_m`) |
| `har_pooled_logcvx_d` | daily | log, b ≥ 0, Σb ≤ 1 | pooled | constraints (vs `_log_d`) |
| `har_pooled_logfe_d` | daily | log | pooled slopes, shrunk asset intercepts | what to pool (vs `_log_d`) |
| `har_asset_log_d` | daily | log | per asset | pooling (vs `_log_d`) |
| `har_pooled_qlike_m` | monthly | QLIKE, convex | pooled | QLIKE + monthly |
| `har_pooled_qlike_d` | daily | QLIKE, convex | pooled | log vs level/QLIKE (vs `_logcvx_d`) |
| `har_asset_qlike_d` | daily | QLIKE, convex | per asset | pooling under constraints |

### Fixed numbers and why

| constant | value | reason |
|---|---|---|
| windows | 1, 3, 12, 60 months, full history | HAR convention adapted to a 12-month horizon: month, quarter, year, five years, all |
| `Q_WINDOW` | 36 months | trades the precision of q against speed of adaptation to within-tier source changes that the view cannot see |
| `Q_PRIOR_MONTHS` | 18 | se(log q_raw)² ≈ 0.09·12/n against a cross-segment prior sd of log q ≈ 0.25 (vol ratios 0.69-1.20 in horizon_evidence.md): n0 = 0.09·12/0.0625 ≈ 18 |
| `MIN_RUN` | 12 months | all daily windows must lie inside one tier run |
| `MIN_COVER` | 0.90 | the target's own coverage rule |
| `LOG_FLOOR` | log 0.05 | bounds the leverage of a near-zero short-window measurement (a vol below 22% of the long-run vol); binds only for M3 |
| `MIN_ORIGINS` | 120 | ten years of origins, about ten independent annual targets, before a regression is trusted |
| `FE_PRIOR` | 120 origins | the same "ten independent years" prior on an asset intercept |
| re-estimation | yearly (December) | the protocol's maximum frequency |

None of these was changed after seeing a backtest score.

## Results

QLIKE (lower is better), means over the origins common to all models.
`ref_*` are the reference benchmarks.

### `daily` subset (assets with a daily-based target), 1,051 origins 1926-06..2013-12

| model | QLIKE | 1926-45 | 1946-79 | 1980-99 | 2000-13 | skill vs ref_last12m (90% CI) | DM t, p |
|---|---|---|---|---|---|---|---|
| har_pooled_qlike_d | **0.391** | 0.615 | 0.364 | 0.243 | 0.354 | 22.6 [-11.2, 43.9] | -1.27, 0.22 |
| har_pooled_logcvx_d | 0.393 | 0.651 | 0.352 | 0.239 | 0.348 | 22.3 [-14.3, 45.7] | -1.18, 0.25 |
| har_pooled_logfe_d | 0.399 | 0.653 | 0.362 | 0.244 | 0.357 | 20.9 [-14.0, 43.0] | -1.16, 0.26 |
| har_pooled_log_d | 0.400 | 0.671 | 0.359 | 0.239 | 0.348 | 20.8 [-16.2, 44.3] | -1.10, 0.28 |
| har_asset_qlike_d | 0.410 | 0.675 | 0.351 | 0.256 | 0.406 | 18.7 [-17.1, 42.2] | -1.03, 0.31 |
| har_pooled_qlike_m | 0.412 | 0.623 | 0.363 | 0.231 | 0.492 | 18.5 [-15.1, 39.8] | -1.08, 0.29 |
| har_pooled_log_m | 0.415 | 0.660 | 0.358 | 0.236 | 0.464 | 17.9 [-17.0, 40.4] | -1.02, 0.32 |
| har_asset_log_d | 0.470 | 0.910 | 0.354 | 0.274 | 0.415 | 7.0 [-44.4, 41.5] | -0.30, 0.77 |
| ref_last12m | 0.505 | **0.418** | 0.599 | 0.313 | 0.673 | | |
| ref_last60m | 0.520 | 0.792 | 0.410 | 0.280 | 0.749 | | |
| ref_blend | 0.567 | 0.958 | 0.471 | 0.313 | 0.618 | | |
| ref_longrun | 0.837 | 1.475 | 0.720 | 0.576 | 0.600 | | |

Era origin counts: 235 / 408 / 240 / 168.

By asset (daily subset):

| model | us_large | us_small | intl_dev | em | ust_int | ust_long | tips | ig_credit | hy_credit | reits | gold | commod. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| har_pooled_qlike_d | 0.336 | 0.310 | 0.278 | 0.275 | 0.401 | 0.378 | 0.325 | 0.280 | 0.586 | 0.315 | 0.362 | 0.253 |
| har_pooled_logcvx_d | 0.353 | 0.310 | 0.270 | 0.271 | 0.387 | 0.353 | 0.310 | 0.269 | 0.598 | 0.303 | 0.353 | 0.257 |
| har_pooled_logfe_d | 0.351 | 0.318 | 0.273 | 0.283 | 0.409 | 0.368 | 0.337 | 0.268 | 0.628 | 0.313 | 0.341 | 0.253 |
| har_pooled_log_d | 0.357 | 0.321 | 0.270 | 0.271 | 0.387 | 0.353 | 0.310 | 0.269 | 0.598 | 0.303 | 0.353 | 0.257 |
| har_asset_qlike_d | 0.362 | 0.309 | 0.292 | 0.286 | 0.418 | 0.367 | 0.329 | 0.303 | 0.675 | 0.450 | 0.384 | 0.285 |
| har_pooled_qlike_m | 0.344 | 0.344 | 0.347 | 0.331 | 0.374 | 0.414 | 0.376 | 0.437 | 0.728 | 0.483 | 0.311 | 0.263 |
| har_pooled_log_m | 0.366 | 0.334 | 0.338 | 0.312 | 0.384 | 0.391 | 0.351 | 0.401 | 0.732 | 0.454 | 0.309 | 0.264 |
| har_asset_log_d | 0.454 | 0.312 | 0.238 | 0.323 | 0.421 | 0.363 | 0.306 | 0.296 | 0.790 | 0.432 | 0.444 | 0.301 |
| ref_last12m | 0.390 | 0.402 | 0.517 | 0.473 | 0.712 | 0.678 | 0.508 | 0.616 | 1.062 | 0.592 | 0.420 | 0.406 |
| ref_last60m | 0.496 | 0.403 | 0.513 | 0.410 | 0.531 | 0.513 | 0.565 | 0.446 | 1.362 | 0.603 | 0.401 | 0.415 |
| ref_blend | 0.470 | 0.414 | 0.380 | 0.349 | 0.517 | 0.645 | 0.562 | 0.466 | 1.070 | 0.660 | 0.374 | 0.308 |
| ref_longrun | 0.662 | 0.588 | 0.298 | 0.305 | 0.646 | 1.171 | 0.584 | 0.712 | 0.927 | 0.837 | 0.499 | 0.362 |

### `all` subset (adds monthly-proxy targets), 1,104 origins 1922-01..2013-12

| model | QLIKE | ..1925 | 1926-45 | 1946-79 | 1980-99 | 2000-13 | skill vs ref_blend (90% CI) | DM t, p |
|---|---|---|---|---|---|---|---|---|
| har_pooled_logcvx_d | **0.395** | 0.498 | 0.545 | 0.402 | 0.247 | 0.348 | 26.4 [13.2, 34.8] | -2.75, 0.011 |
| har_pooled_log_d | 0.400 | 0.531 | 0.554 | 0.405 | 0.247 | 0.348 | 25.5 [11.8, 34.4] | -2.63, 0.015 |
| har_pooled_qlike_d | 0.402 | 0.512 | 0.563 | 0.406 | 0.248 | 0.354 | 25.1 [11.0, 34.1] | -2.56, 0.017 |
| har_pooled_logfe_d | 0.407 | 0.476 | 0.575 | 0.414 | 0.250 | 0.357 | 24.1 [11.7, 32.6] | -2.73, 0.012 |
| har_pooled_log_m | 0.416 | 0.531 | 0.550 | 0.404 | 0.245 | 0.464 | 22.6 [11.2, 30.1] | -2.76, 0.011 |
| har_pooled_qlike_m | 0.420 | 0.512 | 0.566 | 0.401 | 0.240 | 0.492 | 21.7 [9.7, 30.0] | -2.63, 0.015 |
| har_asset_qlike_d | 0.425 | 0.513 | 0.569 | 0.436 | 0.259 | 0.406 | 20.8 [7.1, 29.7] | -2.32, 0.030 |
| har_asset_log_d | 0.459 | 0.531 | 0.639 | 0.467 | 0.283 | 0.415 | 14.5 [-1.2, 25.4] | -1.61, 0.12 |
| ref_blend | 0.537 | 0.502 | 0.774 | 0.515 | 0.288 | 0.618 | | |
| ref_last60m | 0.557 | 0.519 | 0.768 | 0.528 | 0.270 | 0.749 | | |
| ref_last12m | 0.580 | 0.543 | 0.557 | 0.701 | 0.336 | 0.673 | | |
| ref_longrun | 0.699 | **0.491** | 0.977 | 0.725 | 0.488 | 0.600 | | |

Era origin counts: 48 / 240 / 408 / 240 / 168.

By asset (all subset):

| model | us_large | us_small | intl_dev | em | ust_int | ust_long | tips | ig_credit | hy_credit | reits | gold | commod. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| har_pooled_logcvx_d | 0.339 | 0.310 | 0.271 | 0.280 | 0.378 | 0.429 | 0.310 | 0.464 | 0.582 | 0.295 | 0.353 | 0.315 |
| har_pooled_log_d | 0.343 | 0.321 | 0.271 | 0.280 | 0.378 | 0.433 | 0.310 | 0.465 | 0.582 | 0.295 | 0.353 | 0.315 |
| har_pooled_qlike_d | 0.323 | 0.310 | 0.282 | 0.288 | 0.392 | 0.463 | 0.325 | 0.488 | 0.568 | 0.300 | 0.362 | 0.301 |
| har_pooled_logfe_d | 0.337 | 0.318 | 0.273 | 0.292 | 0.400 | 0.456 | 0.337 | 0.481 | 0.610 | 0.301 | 0.341 | 0.319 |
| har_pooled_log_m | 0.352 | 0.334 | 0.324 | 0.317 | 0.375 | 0.454 | 0.351 | 0.514 | 0.708 | 0.400 | 0.309 | 0.317 |
| har_pooled_qlike_m | 0.330 | 0.344 | 0.335 | 0.338 | 0.367 | 0.483 | 0.376 | 0.546 | 0.701 | 0.417 | 0.311 | 0.303 |
| har_asset_qlike_d | 0.347 | 0.309 | 0.292 | 0.297 | 0.407 | 0.472 | 0.329 | 0.498 | 0.652 | 0.394 | 0.384 | 0.346 |
| har_asset_log_d | 0.435 | 0.312 | 0.247 | 0.328 | 0.409 | 0.465 | 0.306 | 0.504 | 0.762 | 0.415 | 0.444 | 0.398 |
| ref_blend | 0.450 | 0.414 | 0.330 | 0.341 | 0.503 | 0.618 | 0.562 | 0.689 | 1.030 | 0.551 | 0.374 | 0.376 |
| ref_last60m | 0.477 | 0.403 | 0.437 | 0.396 | 0.518 | 0.580 | 0.565 | 0.722 | 1.304 | 0.509 | 0.401 | 0.444 |
| ref_last12m | 0.374 | 0.402 | 0.540 | 0.534 | 0.672 | 0.791 | 0.508 | 0.721 | 1.010 | 0.509 | 0.420 | 0.457 |
| ref_longrun | 0.632 | 0.588 | 0.263 | 0.301 | 0.618 | 0.931 | 0.584 | 0.909 | 0.895 | 0.680 | 0.499 | 0.480 |

On these common origins, the pooled daily HAR models beat every reference on every
asset except `intl_dev` and `em` in the `all` subset. There `ref_longrun` (0.263 /
0.301) is close to or below HAR (0.27-0.29); those two assets have short histories
whose long-run mean happens to be a good forecast.

### Comparisons within the family (DM, EWC)

| a vs b | `daily` Δ QLIKE (t, p) | `all` Δ QLIKE (t, p) | reading |
|---|---|---|---|
| pooled_log_d vs pooled_log_m | -0.015 (-0.69, 0.50) | -0.016 (-0.87, 0.40) | daily information helps; the gain is all in 2000-13 (0.348 vs 0.464) and not significant overall |
| pooled_qlike_d vs pooled_qlike_m | -0.021 (-0.79, 0.44) | -0.018 (-0.81, 0.43) | same |
| pooled_logcvx_d vs pooled_log_d | -0.007 (-1.61, 0.12) | -0.0045 (-2.23, 0.036) | constraints help only where they bind (early fits, 1938-1945, when the free fit puts a negative weight on M60); afterwards the two are identical |
| pooled_qlike_d vs pooled_logcvx_d | -0.002 (-0.12, 0.91) | +0.007 (0.73, 0.47) | log + smearing and level QLIKE are indistinguishable |
| pooled_logfe_d vs pooled_log_d | -0.000 (-0.06, 0.95) | +0.008 (0.69, 0.50) | asset intercepts buy nothing |
| asset_log_d vs pooled_log_d | +0.070 (1.80, 0.085) | +0.059 (2.52, 0.019) | free per-asset regressions are clearly worse |
| asset_qlike_d vs pooled_qlike_d | +0.020 (1.25, 0.23) | +0.023 (1.70, 0.10) | per-asset is worse even when constrained |

### Horizon weights

Fits of the pooled models at selected Decembers. The "scale" column is exp(c)·φ for
the log forms and κ for the QLIKE form, i.e. the forecast when every R_h = 1.

`har_pooled_log_d` (= `logcvx_d` from 1950 on; the constraint binds only before then):

| year | D1 | D3 | D12 | M60 | LR (1−Σb) | scale | origins |
|---|---|---|---|---|---|---|---|
| 1940 | 0.17 | 0.27 | 0.40 | -0.29 | 0.45 | 2.37 | 151 |
| 1950 | 0.16 | 0.19 | 0.52 | 0.02 | 0.12 | 1.54 | 271 |
| 1970 | 0.19 | 0.24 | 0.36 | 0.08 | 0.13 | 1.65 | 511 |
| 1990 | 0.18 | 0.29 | 0.12 | 0.27 | 0.13 | 1.50 | 751 |
| 2014 | 0.18 | 0.26 | 0.20 | 0.14 | 0.22 | 1.39 | 1039 |

`har_pooled_qlike_d`:

| year | D1 | D3 | D12 | M60 | LR | κ |
|---|---|---|---|---|---|---|
| 1950 | 0.00 | 0.47 | 0.48 | 0.05 | 0.00 | 1.45 |
| 1970 | 0.18 | 0.37 | 0.34 | 0.04 | 0.09 | 1.45 |
| 1990 | 0.17 | 0.40 | 0.16 | 0.18 | 0.08 | 1.37 |
| 2014 | 0.15 | 0.43 | 0.21 | 0.08 | 0.13 | 1.29 |

`har_pooled_log_m` (2014): M3 0.17, M12 0.40, M60 0.19, LR 0.25, scale 1.37.
Before 1926 the monthly fit is trained only on the smoothed-average-price `us_large`,
`ust_long` and `ig_credit` histories. There it put 0.9-1.0 on the long run and a
negative weight on M3.

Two readings:

* **Last year vs long run.** In the daily-mapped models the last year (D1 + D3 + D12)
  gets 0.64 (log) to 0.79 (QLIKE). The 5-year window plus the full history get 0.21-0.36.
  - The 1-month window alone gets 0.15-0.19 in every log fit since 1950 and 0.10-0.18
    in every QLIKE fit since 1955.
    Daily-measured recent variance is precise enough to carry real weight even for a
    12-month horizon.
  - The 1- and 3-month weights are stable from 1950 on. Weight has migrated from the
    12-month window (0.5 in 1950) toward the 5-year window and the long run (about 0.3
    combined since 1990). Plausibly this reflects more post-1970 assets whose volatility
    reverts to a level that differs from their early-history mean.
* **Across assets** (convex per-asset log fits, 2014 December; see
  `har_asset_qlike_d` for the QLIKE versions):
  - The last-year weight ranges from 0.35-0.45 (`em`, `gold`, `tips`, `intl_dev`) to
    0.62-0.68 (`ust_long`, `us_large`, `hy_credit`, `commodities`). The long-run weight ranges from 0.23
    (`us_small`) to 0.64 (`em`).
  - The free per-asset log fits are unstable: for `tips`, b_M60 = -2.7 and an implied
    LR weight of 3.6; for `em`, b_M60 = -1.0. This is the fragility that makes
    `har_asset_log_d` the worst model in the family.
  - The cross-asset differences are therefore mostly estimation noise. With 150-460
    overlapping training origins per asset outside US equities, they cannot be told
    apart from the pooled vector.

### Calibration

Forecasts are calibrated in ratio terms: mean(v_r/v_f) = 1.01-1.03 on both subsets for
the pooled daily models, which is the condition QLIKE asks for. The median of
v_r/v_f is about 0.67, and mean(v_r)/mean(v_f) is about 0.80. Aggregated in levels,
the forecast variance is therefore about 25% above the realized variance. The
references are calibrated in levels (about 1.0) but not in ratio (1.3-1.6).

The mechanism: realized variance is right-skewed relative to any forecast. The scale
that makes E[y/ŷ] = 1 (κ ≈ 1.3, or exp(c)·φ ≈ 1.4) comes from calm states, where y/ŷ
has a long right tail. The same scale over-forecasts in turbulent states.

For planning this matters. A user who wants a level-unbiased vol should not read these
as median or level forecasts. Of the 12-month realized vols, about two-thirds come in
below the forecast.

## Where it loses

* **1926-1945 on the `daily` subset** (`us_large` and `us_small` only): 0.62-0.67 vs
  0.42 for `ref_last12m`.
  - The cause: until 1938 the forecast comes from the monthly fit trained on 1871-1925
    data. In that data variance barely moved and the fit learned to put about 0.9 on
    the long run.
  - During 1929-1932 the HAR forecasts therefore stayed near 0.19-0.33 for `us_large`
    while realized vol was 0.30-0.57, and `ref_last12m` reacted faster. The single
    1929 origin year costs 3.9 (HAR) vs 1.9 (ref_last12m) in mean QLIKE.
  - This era alone is why the daily-subset skill CI includes zero. In every later era
    the pooled daily HAR models beat the best reference of that era by 11-42%.
* **Before 1926 (`all` subset)**: a tie with `ref_longrun` and `ref_blend` (0.48-0.53
  vs 0.49-0.50). There is little to learn before the fits exist.
* `ref_longrun` is roughly level with HAR on `intl_dev` and `em` (`all` subset).

## Tried and dropped (not registered)

These checks ran at `--step 6` (184 / 176 origins). Each variant was compared with
its registered base by DM. The p-values use the monthly default h on a semiannual
grid, so they are approximate. These are design checks, not a search: no registered
model changed as a result.

| variant | base | Δ QLIKE `daily` | Δ QLIKE `all` | verdict |
|---|---|---|---|---|
| no 21/1-day mapping (q = 1) | har_pooled_log_d | +0.013 (t 3.8, p 0.02) | +0.009 (p 0.06) | the mapping matters: the 1-day scale misleads, as the protocol warns |
| same, QLIKE form | har_pooled_qlike_d | +0.012 (p 0.03) | +0.008 (p 0.07) | same |
| q not shrunk toward 1 | har_pooled_log_d | +0.002 (p 0.66) | +0.003 (p 0.53) | shrinkage harmless, slightly better |
| q window not cut at tier changes | har_pooled_log_d | +0.001 (p 0.71) | +0.001 (p 0.78) | the cut costs nothing and is required by the protocol |
| monthly set plus a 1-month squared return | har_pooled_log_m | +0.002 (p 0.62) | +0.000 (p 0.89) | no information, as expected |
| log model without smearing (exp of mean log) | har_pooled_log_d | +0.077 (p 0.33) | +0.057 (p 0.13) | retransformation is essential; a median forecast loses under QLIKE |

## Run time

Full monthly run, one core per model (1,116 stored origins, about 610 early origins
skipped by the harness for having fewer than 2 eligible assets):

| models | model time |
|---|---|
| `log_m`, `log_d`, `logcvx_d`, `logfe_d` | about 70-80 s each |
| `asset_log_d`, `qlike_m` | about 100-110 s |
| `qlike_d`, `asset_qlike_d` | about 135-150 s |

The cost per origin is about 70-135 ms: the features are recomputed for the full
history, plus a yearly fit (about 0.3-1 s). Everything is far below the 10-minute
limit.

## Tests (`tests/models/test_har.py`, 28 tests, about 20 s)

* Every registered model gives a valid forecast at an early, a middle and a late
  synthetic origin. The routing is checked: prior at the early origin with monthly-only
  assets, monthly fits in the middle, daily fits late.
* The answer is the same cold, warm, and in reverse order.
* No look-ahead: altering all data after `t` leaves the forecast at `t` unchanged and
  changes later ones.
* Training targets are fully observed by the estimation date.
* q ≈ 1 on serially independent data and q > 1.5 on a 3-day-smoothed series. The
  daily set is unavailable for 12 months after a tier change.
* The QLIKE fit recovers known weights and scale.
* Log fits with smearing reproduce the conditional mean. The convex constraint binds
  where the free fit violates it. Thin samples give no fit.

`tests/test_contract.py` passes.

## Caveats

* **Within-tier source changes are invisible to the view.** Examples: `hy_credit`
  VWEHX→HYG in 2007, `em` VEIEX→EEM, `reits` VGSIX→VNQ, the Treasury curve sources.
  q only adapts to these over its 36-month window, so for up to three years the daily
  regressors carry a blend of the old and new scales. A `source` column in the view
  (or a per-asset-month splice flag) would let q be cut there too.
* **The pooled scale.** The intercept / κ ≈ 1.3-1.4 partly reflects that expanding
  long-run means lagged the post-1965 rise in bond and credit volatility
  (mean y/LR ≈ 1.8-2.0 for the Treasury and credit series). It is a feature of 1920-2013
  that may not persist. The FE variant, which lets each asset have its own level, did
  not help.
* **What the training rows are made of.** Before 1926 the training rows come from tier-D
  monthly-average-price series (`us_large` Shiller, NBER yields), which understate
  variance and its variability. The monthly fits learned from them carried into the
  Depression.
* **Significance.** Overlapping rows make the effective sample small: n_eff ≈ 85-140
  in the skill tests. Differences inside the family below about 0.02 QLIKE are not
  resolvable.
* **Calibration.** Forecasts are deliberately above the median outcome (see
  "Calibration").

Shared-code changes needed: none. Nice to have: `Panel` exposing the per-asset-month
`source` (or a splice flag) alongside `tiers`, so daily-to-21-day mappings can be cut
at every splice, not only at tier changes.
