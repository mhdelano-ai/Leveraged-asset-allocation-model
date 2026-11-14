# Regime-switching and structural-break volatility models (`regime`)

Family package `src/assetforecast/models/regime/` (kind `vol`, 8 models), tests
`tests/models/test_regime.py`. All results below are development results
(`ASSETFORECAST_MAX_DATE=2014-12-31`, every month-end origin, stored in
`results/dev/forecasts/vol/regime_*.parquet`); nothing after 2014 was looked at.

## Summary

* **Structural-break (BOCPD) models are the best of the family and beat every
  reference model** on both subsets: QLIKE 0.415 (`regime_bocpd`) vs 0.505 for the
  best reference (`ref_last12m`) on the daily subset (skill +17.7%, 90% CI
  [-10, +37], DM p = 0.24: not significant), and 0.433-0.440 vs 0.537 for the
  best reference (`ref_blend`) on `all` (skill +18 to +19%, 90% CI about
  [+7, +28], DM p = 0.01-0.02: significant one at a time; after a Holm
  adjustment over the family's eight models only `regime_bocpd_eb` stays
  significant, at 10%). They do **not** beat a plain 12-month
  half-life exponential average of squared monthly returns from the `bench`
  family (`bench_vol_m_hl12`, 0.409 daily; differences within +-5%, p > 0.5).
* **Markov switching helps only with three states.** `regime_ms3` (0.484 daily,
  0.467 all) beats every reference on `all` (skill +13%, CI [+3, +20], p = 0.04)
  and is level with `ref_last12m` on daily; `regime_ms2` and the weekly 2-state
  model are no better than `ref_blend`. MS models win clearly in 2000-2013 and
  lose badly in 1926-1945, when volatility reached levels the regime variances,
  fixed by earlier history, could not represent.
* **A common regime loses to per-asset regimes** (`regime_common3` 0.774 vs
  `regime_ms3` 0.484 daily, p = 0.04; 0.646 vs 0.467 on all, p = 0.01). With
  asset-specific variances per state the joint likelihood is dominated by
  shifts in the *relative* volatility of bonds and equities, so the "common
  regimes" it finds are eras (1926-51 low bond / high equity vol; the calm
  1950s-60s; the post-1966 inflation-rates era), not a risk-on/risk-off switch.
  Forcing one global multiplier (proportional variances) was worse still.
* **Breaks in the long-run level are rare once volatility clustering is
  allowed for.** The HAC kappa-2 test at 1% finds exactly one break in 12
  assets (long Treasuries, 1979-10), so `regime_break_blend` equals
  `ref_blend` to three decimals. BOCPD does find breaks, but mostly at crises
  (1929, 1987, 2008); what it delivers is an adaptive window, not a level-shift
  detector.
* **At a 12-month horizon, regime structure adds nothing measurable beyond a
  slow-ish weighted average**: the best MS model is level with a 36-month EWMA
  and 18% worse than a 12-month one; the BOCPD models are level with the
  12-month EWMA.

## Models

Notation: `x_t` monthly log excess return of one asset; all variances are of
`x` (the 21-day scale; a calendar month is about 21 trading days) and are
annualized by 12, exactly as the monthly proxy of the target. Forecasts are
expected **mean** variance over the next 12 months, i.e. `E[sigma^2]`, not a
vol-of-expected-vol, because QLIKE is minimized by the expected variance.

### Shared machinery

* **Estimation schedule.** Parameters are estimated on the data up to the
  latest 31 December on or before the origin (the *estimation date*), i.e. at
  most once a year, and cached under a hash of exactly the data used. The
  state (filtered probabilities, run-length posterior) is recomputed from the
  view at every origin. A cold call equals a warm one by construction (tested).
* **Filtered, never smoothed.** Forecasts use `Pr(s_T | x_1..x_T)` only.
  Smoothed probabilities appear only inside EM, on data up to the estimation date.
* **Hamilton filter by prefix scan.** The forward and backward recursions are
  products of `K x K` matrices `P diag(b_t)`; `hmm.py` computes all prefix and
  suffix products with a Hillis-Steele scan (log2 T batched matrix products,
  each normalized, scale carried in logs). One forward-backward pass on 1,700
  months costs ~3 ms in plain numpy; matches a textbook loop to 1e-15 (tested).
  No numba (its installation was not permitted here; nothing needs it).
* **Never raises, never NaN.** Any failure, or an asset without a usable fit,
  falls back to the asset's full-history mean squared monthly return (=
  `ref_longrun`); `last_error` records the reason. In the dev run no origin
  needed the exception path.

### Markov switching: `regime_ms2`, `regime_ms3`

`x_t | s_t = k ~ t_5(0, c_k^2)`, `s_t` a K-state Markov chain with transition
matrix `P`. EM (MAP) with:

* t(5) emissions (fixed; a round, standard value for monthly returns): the EM
  weights `u = (nu+1)/(nu + x^2/c^2)` down-weight single outliers so that one
  month (1987-10) cannot become a regime (tested: a 10-sigma month in a
  constant-variance series does not create a short-lived state).
* **Forecast levels are moment estimates**: `v_k = sum_t gamma_tk x_t^2 / sum_t gamma_tk`
  (smoothed weights in-sample), not `c_k^2 nu/(nu-2)`. The t ML scale is
  biased for `E[x^2]` whenever the regime is not exactly t(5) (by about +30%
  for normal data); the moment estimate is unbiased whatever the tail shape.
  The t-scale is used only to classify months.
* Dirichlet prior on each row of `P`: 9 pseudo-transitions on the diagonal,
  1 spread over the off-diagonal (prior mean persistence 0.9 worth 10 months;
  negligible against 100s of months, prevents absorbing/empty states).
  Inverse-gamma-like prior: 2 pseudo-months at the sample variance per state
  (prevents a collapsing state).
* States ordered by variance after every M-step (no label switching). Two
  deterministic starts (quantile buckets of the centered 6-month rolling mean
  of `x^2`; fixed multiples of the sample variance); the higher penalized
  likelihood wins.
* Forecast: `sigma^2 = 12 * (1/12) sum_{h=1..12} f_T' P^h v` with `f_T` the
  filtered probabilities at the origin: the full 12-step-ahead regime
  distribution, not "the current regime persists".
* Minimum history for an own fit: 120 months (K=2), 240 (K=3), i.e. every
  regime visited repeatedly. Shorter assets **borrow** the pooled structure of
  the fitted assets (mean `P`, geometric-mean regime multipliers
  `v_k / (pi' v)`), scale it by their own mean `x^2`, and run the filter on
  their own data. Before any asset can be fitted (us_large 1874-1881) the
  fallback is `ref_longrun`.

### Weekly Markov switching: `regime_ms2_weekly`

The same 2-state model on non-overlapping **5-day sums** of daily log excess
returns (blocks anchored to the panel's daily calendar; a NaN day counts 0 as
in the target). Mapping to the 21-day scale: within each run of constant
quality tier (`view.tiers`; a tier change marks a splice), `kappa =
[sum r_m^2 / sum days_m] / [sum s_w^2 / (5 n_w)]` over the complete daily months
of that run up to the estimation date, shrunk to 1 in logs with weight
`n/(n+60)` (a prior worth 5 years: roughly where the ratio's sampling error,
~25% in variance, equals the true departures, 0.7-1.16 at k = 5). Each block is
multiplied by `sqrt(kappa)` of its own segment, so no pre-splice scale crosses
a splice. Forecast: `N * mean_{h=1..N/5} E[y^2_{T+h}] / 5`, `N` = trading days
in the last 12 months (252, or ~295 before 1952). At least 5 years of daily
data; otherwise the monthly 2-state model.

### Common regime: `regime_common3`

One 3-state chain shared by all assets with at least 24 months of history at
the estimation date. Emission for the assets observed in month t: multivariate
t(5) with scale matrix `D_k R D_k`, `D_k = diag(c_{a,k})`, and a **fixed
correlation matrix** `R` (pairwise uncentered monthly correlation up to the
estimation date, repaired to PD) so that correlated assets are not counted as
independent evidence; missing assets are marginalized (the marginal of a
multivariate t is a multivariate t on the observed block). Asset-specific
variance per regime, shrunk towards `level_a * mult_k` with 12 pseudo-months (a
year of data: decisive for a 36-month asset, negligible for a 90-year one),
where `mult_k` is the common regime multiplier and `level_a` the asset's mean
`x^2` net of its regime mix. This is how a short-history asset gets a regime
signal it could not estimate itself. M-step for the scales uses the diagonal
(t-weighted) update, an approximation given `R`; forecast levels are again
moment estimates. At least 120 months of data overall.

### Bayesian online change-point detection: `regime_bocpd`, `regime_bocpd_eb`, `regime_bocpd_weekly`

Adams & MacKay (2007). Within a segment `x_t ~ N(0, sigma^2)`,
`sigma^2 ~ InvGamma(alpha0 = 3, (alpha0-1) v0)`, `v0` = the asset's mean `x^2` up
to the estimation date (prior mean = long-run level; alpha0 = 3 is the smallest
round value with a finite prior variance, worth 6 months). Constant hazard
`H`. The run-length posterior is updated every month (pruned at 1e-10 of the
mode). Forecast:

    sigma^2 = 12 * mean_{h=1..12} [ (1-H)^h * E_post(segment variance) + (1 - (1-H)^h) * v0 ]

with `E_post(segment variance) = sum_r p(r) beta_r / (alpha_r - 1)`: the current
level estimated from post-break data only, with pre-break data down-weighted
exactly in proportion to the posterior evidence for a break.

* `regime_bocpd`: `H = 1/240` (expected segment 20 years: the order of the
  documented level shifts - bonds before/after 1979, the splices), fixed a priori.
* `regime_bocpd_eb`: `H` chosen each year by **empirical Bayes**: the value on
  the grid of round expected segment lengths {1, 3, 5, 10, 20, 40 years} that
  maximizes the summed BOCPD marginal likelihood of all assets with 60+ months
  up to the estimation date. (Added after a step-12 look at a 5-year hazard
  showed the result is sensitive to `H`; rather than pick `H` from the
  scoreboard, it is estimated inside the model. It chose 3 years at every
  estimation date checked: 1935, 1975, 2013.)
* `regime_bocpd_weekly`: the same on the kappa-mapped 5-day sums of
  `regime_ms2_weekly`, hazard per block `H * 12 * 5 / 252` (same expected
  segment length in calendar time), monthly version for assets with under 5
  years of daily data.

### ICSS level + blend: `regime_break_blend`

Breaks found by binary segmentation with the Sanso-Arago-Carrion (2004)
kappa-2 CUSUM-of-squares test on monthly squared returns up to the estimation
date. Its denominator is a Bartlett long-run variance (bandwidth
`max(12, ceil(1.3 sqrt(T)))`: monthly squared returns are persistent beyond a
year), which makes it valid under volatility clustering, unlike Inclan-Tiao.
Critical value: sup of a Brownian bridge at **1%** (1.628; several assets
re-tested every year), segments at least 24 months. Forecast:
`sqrt(level_since_last_break * var_last_60m)` - exactly `ref_blend` with the
full-history level replaced by the post-break level, so the comparison with
`ref_blend` isolates the value of break detection.

## Results

QLIKE (headline), on origins common to all models shown, 1,051 origins
(daily subset) and 1,104 (all). Lower is better. `log_err` is a bias
statistic (+0.03 for a perfect forecast).

### Daily subset (assets whose target comes from daily data)

| model | qlike | se_var_scaled | log_err |
|---|---|---|---|
| regime_bocpd | **0.415** | 5.346 | 0.092 |
| regime_bocpd_weekly | 0.416 | 7.441 | 0.065 |
| regime_bocpd_eb | 0.431 | 5.050 | 0.111 |
| regime_ms3 | 0.484 | 7.031 | 0.123 |
| ref_last12m | 0.505 | 5.745 | -0.019 |
| ref_last60m | 0.520 | 6.433 | 0.070 |
| regime_break_blend | 0.566 | 5.866 | 0.106 |
| ref_blend | 0.567 | 5.858 | 0.101 |
| regime_ms2 | 0.571 | 5.767 | 0.105 |
| regime_ms2_weekly | 0.580 | 6.017 | 0.207 |
| regime_common3 | 0.774 | 6.713 | 0.088 |
| ref_longrun | 0.837 | 6.819 | 0.131 |

By era:

| model | 1926-1945 | 1946-1979 | 1980-1999 | 2000-2013 |
|---|---|---|---|---|
| regime_bocpd | 0.561 | 0.364 | 0.238 | 0.591 |
| regime_bocpd_weekly | 0.536 | 0.424 | 0.244 | 0.471 |
| regime_bocpd_eb | 0.639 | 0.398 | 0.219 | 0.527 |
| regime_ms3 | 0.818 | 0.384 | 0.309 | 0.508 |
| ref_last12m | 0.418 | 0.599 | 0.313 | 0.673 |
| ref_last60m | 0.792 | 0.410 | 0.280 | 0.749 |
| regime_break_blend | 0.958 | 0.471 | 0.316 | 0.608 |
| ref_blend | 0.958 | 0.471 | 0.313 | 0.618 |
| regime_ms2 | 1.014 | 0.460 | 0.378 | 0.496 |
| regime_ms2_weekly | 0.941 | 0.519 | 0.344 | 0.560 |
| regime_common3 | 1.452 | 0.564 | 0.547 | 0.657 |
| ref_longrun | 1.475 | 0.720 | 0.576 | 0.600 |
| n origins | 235 | 408 | 240 | 168 |

By asset:

| model | us_large | us_small | intl_dev | em | ust_int | ust_long | tips | ig_credit | hy_credit | reits | gold | commod. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| regime_bocpd | 0.391 | 0.335 | 0.437 | 0.371 | 0.397 | 0.445 | 0.556 | 0.501 | 0.699 | 0.611 | 0.340 | 0.333 |
| regime_bocpd_weekly | 0.345 | 0.357 | 0.350 | 0.333 | 0.445 | 0.480 | 0.524 | 0.377 | 0.607 | 0.440 | 0.411 | 0.331 |
| regime_bocpd_eb | 0.344 | 0.370 | 0.340 | 0.326 | 0.366 | 0.482 | 0.491 | 0.504 | 0.662 | 0.585 | 0.324 | 0.262 |
| regime_ms3 | 0.431 | 0.364 | 0.320 | 0.273 | 0.401 | 0.478 | 0.362 | 0.484 | 0.793 | 0.680 | 0.364 | 0.278 |
| ref_last12m | 0.390 | 0.402 | 0.517 | 0.473 | 0.712 | 0.678 | 0.508 | 0.616 | 1.062 | 0.592 | 0.420 | 0.406 |
| ref_last60m | 0.496 | 0.403 | 0.513 | 0.410 | 0.531 | 0.513 | 0.565 | 0.446 | 1.362 | 0.603 | 0.401 | 0.415 |
| regime_break_blend | 0.470 | 0.414 | 0.380 | 0.349 | 0.517 | 0.606 | 0.562 | 0.493 | 1.070 | 0.660 | 0.374 | 0.308 |
| ref_blend | 0.470 | 0.414 | 0.380 | 0.349 | 0.517 | 0.645 | 0.562 | 0.466 | 1.070 | 0.660 | 0.374 | 0.308 |
| regime_ms2 | 0.513 | 0.390 | 0.293 | 0.294 | 0.511 | 0.633 | 0.388 | 0.412 | 0.807 | 0.712 | 0.413 | 0.268 |
| regime_ms2_weekly | 0.480 | 0.471 | 0.292 | 0.297 | 0.479 | 0.402 | 0.490 | 0.422 | 1.020 | 0.843 | 0.444 | 0.413 |
| regime_common3 | 0.640 | 0.502 | 0.326 | 0.324 | 0.675 | 0.964 | 0.777 | 0.662 | 1.042 | 0.925 | 0.510 | 0.351 |
| ref_longrun | 0.662 | 0.588 | 0.298 | 0.305 | 0.646 | 1.171 | 0.584 | 0.712 | 0.927 | 0.837 | 0.499 | 0.362 |

Skill against the best reference on this subset (`ref_last12m`); Fieller-EWC
90% interval and two-sided DM (EWC) p-value from `assetforecast.eval.inference`:

| model | skill % | 90% CI | DM p |
|---|---|---|---|
| regime_ms2 | -13.1 | [-80.0, 32.5] | 0.65 |
| regime_ms3 | +4.2 | [-44.8, 36.8] | 0.85 |
| regime_ms2_weekly | -14.9 | [-78.0, 24.8] | 0.56 |
| regime_common3 | -53.3 | [-155.3, 19.0] | 0.23 |
| regime_bocpd | +17.7 | [-10.1, 37.0] | 0.24 |
| regime_bocpd_eb | +14.5 | [-22.3, 38.7] | 0.43 |
| regime_bocpd_weekly | +17.7 | [-12.5, 38.0] | 0.27 |
| regime_break_blend | -12.2 | [-71.4, 29.4] | 0.64 |

### All (adds the monthly-proxy eras)

| model | qlike | se_var_scaled | log_err |
|---|---|---|---|
| regime_bocpd_eb | **0.433** | 3.159 | 0.159 |
| regime_bocpd_weekly | 0.433 | 5.087 | 0.112 |
| regime_bocpd | 0.440 | 3.484 | 0.127 |
| regime_ms3 | 0.467 | 4.131 | 0.186 |
| regime_ms2 | 0.512 | 3.487 | 0.177 |
| regime_ms2_weekly | 0.517 | 3.566 | 0.224 |
| regime_break_blend | 0.536 | 3.487 | 0.153 |
| ref_blend | 0.537 | 3.480 | 0.148 |
| ref_last60m | 0.557 | 3.765 | 0.105 |
| ref_last12m | 0.579 | 3.706 | -0.009 |
| regime_common3 | 0.646 | 3.876 | 0.141 |
| ref_longrun | 0.699 | 3.979 | 0.191 |

By era (the first era here starts in 1922):

| model | 1922-1945 | 1946-1979 | 1980-1999 | 2000-2013 |
|---|---|---|---|---|
| regime_bocpd_eb | 0.549 | 0.435 | 0.224 | 0.527 |
| regime_bocpd_weekly | 0.505 | 0.478 | 0.243 | 0.471 |
| regime_bocpd | 0.513 | 0.444 | 0.237 | 0.591 |
| regime_ms3 | 0.625 | 0.439 | 0.298 | 0.508 |
| regime_ms2 | 0.692 | 0.488 | 0.349 | 0.496 |
| regime_ms2_weekly | 0.668 | 0.504 | 0.330 | 0.560 |
| regime_break_blend | 0.729 | 0.515 | 0.291 | 0.608 |
| ref_blend | 0.729 | 0.515 | 0.288 | 0.618 |
| ref_last60m | 0.726 | 0.528 | 0.270 | 0.749 |
| ref_last12m | 0.555 | 0.701 | 0.336 | 0.673 |
| regime_common3 | 0.854 | 0.602 | 0.463 | 0.657 |
| ref_longrun | 0.896 | 0.725 | 0.487 | 0.600 |
| n origins | 288 | 408 | 240 | 168 |

By asset:

| model | us_large | us_small | intl_dev | em | ust_int | ust_long | tips | ig_credit | hy_credit | reits | gold | commod. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| regime_bocpd_eb | 0.330 | 0.370 | 0.314 | 0.331 | 0.362 | 0.508 | 0.491 | 0.611 | 0.639 | 0.489 | 0.324 | 0.315 |
| regime_bocpd_weekly | 0.332 | 0.357 | 0.316 | 0.332 | 0.440 | 0.520 | 0.524 | 0.581 | 0.591 | 0.399 | 0.411 | 0.325 |
| regime_bocpd | 0.375 | 0.335 | 0.383 | 0.366 | 0.397 | 0.500 | 0.556 | 0.627 | 0.678 | 0.518 | 0.340 | 0.326 |
| regime_ms3 | 0.412 | 0.364 | 0.312 | 0.282 | 0.385 | 0.536 | 0.362 | 0.572 | 0.761 | 0.557 | 0.364 | 0.372 |
| regime_ms2 | 0.491 | 0.390 | 0.288 | 0.300 | 0.486 | 0.605 | 0.388 | 0.581 | 0.784 | 0.592 | 0.413 | 0.365 |
| regime_ms2_weekly | 0.460 | 0.471 | 0.288 | 0.303 | 0.457 | 0.475 | 0.490 | 0.584 | 0.984 | 0.683 | 0.444 | 0.404 |
| regime_break_blend | 0.450 | 0.414 | 0.330 | 0.341 | 0.503 | 0.596 | 0.562 | 0.699 | 1.030 | 0.551 | 0.374 | 0.376 |
| ref_blend | 0.450 | 0.414 | 0.330 | 0.341 | 0.503 | 0.618 | 0.562 | 0.689 | 1.030 | 0.551 | 0.374 | 0.376 |
| ref_last60m | 0.477 | 0.403 | 0.437 | 0.396 | 0.518 | 0.580 | 0.565 | 0.722 | 1.304 | 0.509 | 0.401 | 0.444 |
| ref_last12m | 0.374 | 0.402 | 0.540 | 0.534 | 0.672 | 0.791 | 0.508 | 0.721 | 1.010 | 0.509 | 0.420 | 0.457 |
| regime_common3 | 0.611 | 0.502 | 0.282 | 0.319 | 0.656 | 0.790 | 0.777 | 0.807 | 1.005 | 0.732 | 0.510 | 0.411 |
| ref_longrun | 0.632 | 0.588 | 0.263 | 0.301 | 0.618 | 0.931 | 0.584 | 0.909 | 0.895 | 0.680 | 0.499 | 0.480 |

Skill against the best reference on `all` (`ref_blend`):

| model | skill % | 90% CI | DM p |
|---|---|---|---|
| regime_ms2 | +4.6 | [-3.7, 11.3] | 0.32 |
| regime_ms3 | +13.0 | [3.3, 20.0] | 0.04 |
| regime_ms2_weekly | +3.7 | [-5.4, 10.2] | 0.44 |
| regime_common3 | -20.3 | [-32.0, -7.5] | 0.02 |
| regime_bocpd | +18.1 | [6.7, 27.4] | 0.02 |
| regime_bocpd_eb | +19.4 | [10.2, 26.5] | 0.01 |
| regime_bocpd_weekly | +19.3 | [7.6, 28.0] | 0.02 |
| regime_break_blend | +0.2 | [-0.5, 0.8] | 0.65 |

Eight models were compared against the reference: with a Holm adjustment
over the eight, only `regime_bocpd_eb` on `all` stays significant at 10%
(0.01 x 8 = 0.08); none at 5%.

### Pairwise questions

Skill of A relative to B (positive = A better), same inference:

| A vs B | subset | QLIKE A | QLIKE B | skill A % | 90% CI | DM p |
|---|---|---|---|---|---|---|
| common3 vs ms3 | daily | 0.774 | 0.484 | -59.9 | [-91.2, -18.3] | 0.04 |
| common3 vs ms3 | all | 0.646 | 0.467 | -38.2 | [-57.4, -16.6] | 0.01 |
| common3 vs ms2 | all | 0.646 | 0.512 | -26.1 | [-37.4, -12.3] | 0.01 |
| ms3 vs ms2 | daily | 0.484 | 0.571 | +15.2 | [2.4, 22.8] | 0.07 |
| ms3 vs ms2 | all | 0.467 | 0.512 | +8.8 | [1.8, 14.4] | 0.05 |
| ms2_weekly vs ms2 | daily | 0.580 | 0.571 | -1.6 | [-20.7, 8.7] | 0.82 |
| bocpd_weekly vs bocpd | daily | 0.416 | 0.415 | -0.0 | [-14.4, 12.2] | 1.00 |
| break_blend vs ref_blend | daily | 0.566 | 0.567 | +0.2 | [-0.5, 0.8] | 0.65 |
| ms3 vs bench_vol_m_hl36 | daily | 0.484 | 0.464 | -4.3 | [-18.5, 10.8] | 0.61 |
| ms3 vs bench_vol_m_hl12 | daily | 0.484 | 0.409 | -18.2 | [-55.3, 11.3] | 0.30 |
| bocpd vs bench_vol_m_hl12 | daily | 0.415 | 0.409 | -1.5 | [-15.6, 9.6] | 0.83 |
| bocpd vs bench_vol_m_hl12 | all | 0.440 | 0.451 | +2.4 | [-5.1, 8.3] | 0.54 |
| bocpd_eb vs bench_vol_m_hl12 | all | 0.433 | 0.451 | +3.9 | [-8.0, 13.1] | 0.53 |
| bocpd vs bench_vol_m_hl36 | daily | 0.415 | 0.464 | +10.5 | [-1.3, 20.8] | 0.13 |
| bocpd vs bench_vol_m_hl36 | all | 0.440 | 0.484 | +9.2 | [0.8, 17.1] | 0.08 |

(`bench_vol_m_hl12/36/120` are another family's exponentially weighted means of
squared monthly returns with half-lives of 1, 3 and 10 years, read from
`results/dev`; `bench_vol_m_hl120` scores 0.619 daily.)

## What the models find

### Regimes (per asset, monthly, fitted on data to 2014-12)

Annualized vol per regime and expected duration `1/(1-p_kk)` in months; the
episodes are the months where the filtered probability of the top state
exceeds 0.5 (3+ months).

| asset | 2-state vols / durations | 3-state vols | 3-state durations | top state of the 3-state model |
|---|---|---|---|---|
| us_large | 12.8%, 36.1% / 207, 19 | 9.1, 14.5, 38.7% | 32, 56, 19 | 1929-10..1934-03, 1938, 1939, 1974-10..1975-05, 2009-01..09 |
| us_small | 18.5, 48.1 / 239, 40 | 15.1, 23.2, 56.3 | 46, 35, 26 | 1931-34, 1938-40, 2009-02..05 |
| intl_dev | 14.4, 23.5 / 37, 15 | 13.2, 18.1, 27.4 | 26, 18, 10 | 1990-09..12, 2008-10..2009-09 |
| em | 16.6, 28.3 / 16, 20 | 15.8, 22.5, 30.9 | 15, 12, 15 | 1998-99, 2000-02, 2008-10..2010-03, 2011-12 |
| ust_int | 2.7, 7.6 / 37, 117 | 2.6, 5.8, 10.2 | 49, 60, 30 | 1970-71, 1980-83, 1984-88, 2002-04, 2008-10 |
| ust_long | 3.0, 10.8 / 50, 91 | 2.5, 6.5, 12.2 | 30, 24, 142 | 1932, 1969-72, 1975-76, 1980-1999 (most), 2001-2014 |
| ig_credit | 1.8, 6.7 / 48, 88 | 1.8, 4.8, 9.3 | 48, 35, 25 | 1933, 1970-72, 1974-76, 1980-85, 2003-04, 2008-10 |
| hy_credit | 5.4, 15.8 / 34, 9 | 4.9, 9.2, 25.4 | 28, 13, 7 | 2008-10..2009-09 |
| reits | 13.1, 30.4 / 71, 17 | 12.1, 18.4, 63.7 | 69, 32, 9 | 2008-10..2009-09 |
| gold | 13.8, 29.1 / 61, 32 | 13.1, 18.9, 34.0 | 83, 23, 17 | 1973-75, 1978-81, 1982-83, 2008-09, 2011-12 |
| commodities | 9.5, 20.6 / 72, 78 | 7.7, 14.0, 25.5 | 68, 61, 32 | 1929-42 (most), 1947-49, 1972-75, 1999, 2008-09 |
| tips | 3.4, 7.4 / 19, 28 | (below 240 months: borrows pooled structure) | | |

The 3-state equity models have a clear crisis state (38-64% vol) that lines
up with 1929-34, 1938-39, 1974, 2008-09; the middle state is ordinary
turbulence. For bonds the "regimes" are mostly eras (the low-vol state is the
1940s peg and the 1950s-60s; the high state the 1980s and the post-2001 TLT/IEF
era with its higher duration), i.e. level shifts that a Markov chain models as
very persistent states (ust_long top-state duration 142 months). The 1987
crash does not create a state anywhere (one month; the t(5) emission absorbs
it): us_large's 2-state turbulent episode after it is 1987-11..1988-02.

**How many regimes are identifiable** (BIC of the K-state t(5) model, data to
2014, difference from the best):

| asset | n | K=1 | K=2 | K=3 | K=4 | best |
|---|---|---|---|---|---|---|
| us_large | 1727 | 109.7 | 0 | 2.8 | 48.1 | 2 |
| us_small | 1062 | 96.2 | 0 | 13.7 | 58.3 | 2 |
| intl_dev | 480 | 0 | 4.5 | 32.9 | 75.6 | 1 |
| em | 306 | 0 | 8.5 | 35.9 | 75.5 | 1 |
| ust_int | 740 | 102.1 | 0 | 7.6 | 35.3 | 2 |
| ust_long | 1151 | 400.8 | 21.8 | 0 | 26.0 | 3 |
| tips | 215 | 2.7 | 0 | 22.1 | 59.5 | 2 |
| ig_credit | 1151 | 333.1 | 0 | 1.8 | 34.1 | 2 |
| hy_credit | 419 | 31.2 | 0 | 18.9 | 58.5 | 2 |
| reits | 516 | 14.8 | 0 | 14.7 | 57.9 | 2 |
| gold | 520 | 31.2 | 0 | 23.6 | 66.9 | 2 |
| commodities | 1062 | 120.8 | 0 | 1.9 | 45.4 | 2 |

In-sample, two regimes are identifiable for most assets (three for long
Treasuries, where the third is an era), and with t(5) emissions none for
intl_dev and em (25-40 years of data). Out of sample the 3-state model
nevertheless forecasts better than the 2-state one (+9 to +15%, p = 0.05-0.07):
the crisis state lets the calm state be genuinely calm.

### Common regime

At 2014-12 the three common states have expected durations of 46, 19 and 8
months and line up with **eras**, not with risk-off episodes:

* state 0 (31% of months): 1871-1926 (us_large only, much of it), 1949-1966,
  1977-78 - everything calm; equity 10-13%, bonds 3-5%.
* state 1 (48%): most of 1966-2014 - bonds turbulent (ust_long 11.5%, ust_int
  7.7%), equities moderate (15%).
* state 2 (21%): 1926-1951 plus 1991 - equities very volatile (23-35%),
  bonds calm (ust_long 3.5%: the peg).

Because each state has its own variance for each asset, the joint likelihood
is maximized by splitting history where the *cross-section* of volatilities
changes (the bond-equity mix), which is mostly a slow era effect. The filter
therefore stays in the modern-era state 1 throughout 2008-09 (filtered
probability 0.86-1.00 every month; 1987-10 likewise): the common model's
us_large forecast was 14-16% through the crisis, against 39% for its own
crisis level in the per-asset 3-state model. A 36-month asset inherits
whatever era-mix its history happened to cover. That is why it loses to every
per-asset model.

### Breaks

kappa-2 ICSS, 1%, data to 2014-12: **one break in twelve assets**:
`ust_long` 1979-10 (vol 5.4% before, 12.4% after - the Volcker change of
operating procedure, and the start of high-duration bond volatility). Nothing
for gold (its data start in 1971-09, after the float), REITs, credit or the
splices. With a HAC denominator long enough to absorb volatility clustering
the test has little power on monthly data; at 5% it adds ust_long 1966-02,
ig_credit 1967-01 (the start of the inflation era) and gold 1983-03 (end of
the post-float turbulence).

BOCPD (`H = 1/240`, data to 2014-12; segment starts where the MAP run length
resets with posterior > 0.3):

| asset | segment starts |
|---|---|
| us_large | 1873-09, 1893-05, 1940-07 |
| us_small | 1929-10, 1940-06, 1987-10 |
| em | 1997-08 |
| ust_int | 1957-11, 1966-02, 1969-09, 1971-09, 1979-10 |
| ust_long | 1931-10, 1939-09, 1979-10, 2008-11 |
| tips | 2008-09 |
| ig_credit | 1931-10, 1939-08, 1951-03, 1979-09, 2008-09, 2009-08 |
| hy_credit | 2007-06, 2008-09 |
| reits | 1987-10, 2008-10 |
| gold | 1983-03 |
| commodities | 1972-08 |
| intl_dev | none |

The bond breaks (1966-71 inflation, 1979-10 Volcker, the 1951 Accord for
credit) are genuine level shifts; most of the rest are crises. BOCPD's
advantage over the references is not that it finds level shifts but that it
behaves as an adaptive window: after a break it estimates the level from the
post-break months only, and with no break it averages decades.

## Does regime structure add anything at 12 months beyond a weighted average?

Not measurably. The ranking `bench_vol_m_hl12` (0.409) ~ `regime_bocpd`
(0.415) ~ `regime_bocpd_weekly` (0.416) < `regime_bocpd_eb` (0.431) <
`bench_vol_m_hl36` (0.464) ~ `regime_ms3` (0.484) on the daily subset says that
what matters at this horizon is how fast the level estimate adapts, and a
1-year half-life is about right; the Markov-switching structure (discrete
levels fixed by history, 12-step regime propagation) does not beat a
continuous average with similar memory. The MS models do have a distinct
profile - best of all in 2000-2013 (ms2 0.496, ms3 0.508 vs 0.649 for the
12-month EWMA and 0.673 for `ref_last12m`; presumably because the
12-step regime propagation pulls a crisis forecast back towards the calm
level instead of extrapolating the crisis - not verified origin by origin) and worst in
1926-1945 (they cannot represent volatility above the highest regime level
seen before). That pattern suggests value in combination, not on its own.

## Where they lose

* **Unprecedented volatility (1926-1945)**: every model whose levels are fitted
  on history (MS, common, `ref_blend`-type) loses to `ref_last12m`; regime
  variances re-estimated yearly catch up too slowly. BOCPD loses less.
* **hy_credit and reits** (all models): smoothed NAV eras and the 2008 jump;
  BOCPD weekly is the best of the family there.
* **Common regime**: everywhere (see above).
* **Bias**: all members over-forecast on average (`log_err` 0.07-0.22 against a
  Jensen offset of ~0.03), as do the references with long memories; the
  2-state weekly model most (0.21): its 52-week horizon is several regime
  durations, so it reverts fully to the unconditional mean, which is
  dominated by crisis variance.

## Variants tried and dropped

All on the same code, scored on the stated origins only (QLIKE; refs on the
same origins in brackets). None was registered.

| variant | origins | daily | all | why dropped |
|---|---|---|---|---|
| `common3` with proportional variances (one global multiplier `L_a m_k`) | Decembers (88/92) | 0.905 (common3 0.704) | 0.743 (0.585) | worse than the free common model, which is already the worst |
| BOCPD with hazard 1/60 (5 years) | Decembers | 0.361 (bocpd 0.395) | 0.379 (0.410) | a sensitivity check; choosing it would be tuning on the scoreboard. Replaced by `regime_bocpd_eb`, which estimates the hazard in the model |
| MS2 on 1-day returns (k = 1, kappa-mapped; `MSWeekly(2, k=1)`) | every 5th December, 17/18 (cost: 22,000 observations per asset per yearly fit) | 0.644 (weekly 0.512, monthly 0.571, `ref_last12m` 0.325) | 0.524 (0.448, 0.496) | worst of the three frequencies on the origins it was run on; 1-day returns are the most exposed to the scale problems and add no forecasting power here |
| kappa-2 ICSS at 5% (breaks only, not forecast) | data to 2014 | | | finds ust_long 1966-02 and 1979-10, ig_credit 1967-01, gold 1983-03; registering a looser threshold after seeing the 1% result would be tuning |

Not attempted: 4+ regimes (rejected by BIC for every asset), fitting the
t degrees of freedom (one more fragile parameter per state; the moment-based
levels make the forecast insensitive to it), regime-dependent correlations in
the common model.

## Run time

Full monthly dev run (1,116 forecast origins, 1874-2014), single process,
measured with load average 10-15 on the shared machine (`--jobs 2`):

| model | seconds | ms per origin |
|---|---|---|
| regime_ms2 | 137 | 120 |
| regime_ms3 | 322 | 283 |
| regime_ms2_weekly | 573 | 507 |
| regime_common3 | 237 | 207 |
| regime_bocpd | 43 | 35 |
| regime_bocpd_eb | 191 | 166 |
| regime_bocpd_weekly | 176 | 155 |
| regime_break_blend | 24 | 18 |

All under the 10-minute limit; `regime_ms2_weekly` is the closest (it fits
4,600 weekly observations per asset per year), and would be well under on an
idle core. Cost is dominated by the yearly EM fits.

## Tests

`tests/models/test_regime.py` (37 tests with the contract tests, ~100 s):
every registered model returns valid forecasts at 1979 (monthly-only assets),
1990 and 2010 on the synthetic panel and never hits the exception path;
cold = warm = any call order (exact equality); altering all data after `t`
(on a warmed-up model) does not change the forecast at `t`; the scan-based
forward-backward equals a textbook loop; the 12-month forecast uses `P^h` and
tends to the stationary mean; EM recovers simulated regimes; a single 10-sigma
month does not create a regime; the common model recovers a shared regime and
transfers it to a short-history asset; BOCPD finds a simulated level shift;
the kappa-2 test finds a real break and stays quiet (at most 6 of 20) on
persistent stochastic-volatility series without one.

## Caveats

* Eight models compared with the references: only `regime_bocpd_eb` on `all`
  survives a Holm adjustment, and only at 10%.
* `regime_bocpd_eb` was added after seeing that the hazard matters (a step-12
  look at H = 1/60); its hazard is estimated walk-forward from the likelihood,
  the grid is round numbers, and it chose 3 years, not the 5 years that was
  looked at.
* `regime_bocpd`'s prior mean `v0` (the asset's full-history mean squared
  return up to the estimation date) also acts as the long-run level that the
  forecast mixes in for possible future breaks; with H = 1/240 that weight is
  under 3% at 12 months.
* The common model's scale M-step is a diagonal approximation given `R`
  (generalized EM; convergence is checked on the penalized likelihood).
* The weekly mapping uses tier changes as splice markers; splices within a
  tier (e.g. em VEIEX -> EEM, both A) are not detected.
* The weekly BOCPD hazard per block assumes 252 trading days a year; before
  1952 (about 295) the expected segment is about 15% shorter in calendar time.
* Forecasts before 1926 (us_large, ust_long, ig_credit from tier-D month-average
  data) are produced but are scored only in `all`, and those eras' measured
  volatility is understated by about 15%.
