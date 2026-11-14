# combo — forecast combination and level calibration

Package `src/volmodel/models/combo/` (`store.py` reads stored component
forecasts, `train.py` builds walk-forward training data from the view, `vol.py`,
`corr.py`, `analysis.py` regenerates every table below), tests
`tests/models/test_combo.py`. Development panel (`VOLMODEL_MAX_DATE=2014-12-31`),
every month-end origin, targets version 3 (21-day realized covariance). Nothing
after 2014 was looked at.

Reproduce everything (once the components are stored), about 5 minutes on one core:

```
export VOLMODEL_MAX_DATE=2014-12-31 OMP_NUM_THREADS=1
python -m volmodel.models.combo.analysis --forecast --out combo_tables.md
```

`--forecast` re-runs all 16 registered combos from scratch (`force=True`;
vol, then corr, then cov). The unregistered variants are recomputed in memory
by the same classes and never stored. Nothing is hard-coded from current
numbers: member lists come from the registry, and weights and calibration
constants are re-estimated from the view. After the data correction the
numbers will move a little.

## Answers first

Losses: QLIKE for vol (lower is better). Skill is the % reduction in mean
loss, with a 90% Fieller-EWC CI; p is the two-sided DM-EWC p-value. `daily` has
1,051 common origins (1926-06..2013-12) and `all` has 1,104 (1922-01..2013-12).

1. **Does combining beat the best single model?** No. It ties the group of
   leaders and does not beat the ex-post best. On `daily`, `statespace_2c_daily`
   scores 0.358. The best registered combo, `combo_vol_ew`, scores 0.383:
   −6.9% [−20.8, +6.6], p = 0.37. `combo_vol_hier_trim` scores 0.388:
   −8.4% [−22.6, +6.3], p = 0.32. On `all` they tie: 0.393 vs 0.393 for
   `combo_vol_hier_trim` (−0.1% [−11.8, +12.3], p = 0.99).
   - Against the references the combos gain as much as the leaders do, with
     tighter intervals. On `all`, `combo_vol_hier_trim` is +26.8% vs
     `ref_blend` [+18.4, +32.5], p = 0.002. `statespace_2c_daily` is +26.8%
     with a wider interval [+10.7, +37.1], p = 0.02. On `daily` the combos are
     +23–24% vs `ref_last12m`, not significant (p ≈ 0.22–0.24, because of
     1926–45), the same as every leader.
   - The comparison that matters ex ante is against the model you would have
     picked. Picking, each December, the single model with the best past mean
     QLIKE (`x_best_model`) scores 0.520 on `daily`, worse than
     `ref_last12m`. Every combo beats it by about 25% (p ≈ 0.03). The
     backtest cannot tell the ex-post best model from the others: all 20
     models tested, including both references, stay in the 90% Model
     Confidence Set.
   - The eras explain it. In 1946–79 and 1980–99 the combos are the best
     forecasts in the study (0.333 / 0.213 on `daily` vs 0.352 / 0.236 for
     `statespace_2c_daily`). In 2000–13 they lose clearly to the leaders that
     use daily data (0.40 vs 0.29).
2. **Does level calibration help?** Only for the models that under-forecast,
   and only after 1946. Overall it hurts more models than it helps.
   - Calibrating each of the 62 single models walk-forward with one pooled
     constant improves 11 of 67 calibrated-vs-raw pairs on `daily` (19 on
     `all`). One improves significantly: `ref_last12m`, +17.5% [+6.6, +23.2],
     p = 0.03. 22–24 get significantly worse, all of them regression-type
     leaders and the combos (−3% to −9%).
   - The damage comes from 1926–45. The constant estimated on 1922–28 says
     forecasts were too high (c_hat = 0.69, applied c = 0.92 in Dec 1929), so
     forecasts were cut just before the Depression. In Dec 1935 the constant
     overshoots the other way (1.15).
   - After 1946 calibration is close to neutral (±0.01) for models already near
     1 and helps the under-forecasters. `bench_vol_m_hl12`: −0.031 in 1946–79,
     −0.064 in 2000–13. `garch_d_rm2006`: −0.046 / −0.015. `ref_last12m`:
     −0.14 / −0.16.
   - The two pre-specified "cheap and robust" candidates gain a little overall,
     not significantly: `combo_vol_cal_rm2006` 0.390 vs 0.400 (+2.4%, p = 0.50),
     `combo_vol_cal_m_hl12` 0.401 vs 0.409 (+2.0%, p = 0.67).
   - One constant per asset group is no better than the pooled one.
3. **Are learned weights better than equal weights?** No: the usual finding.
   - Shrunk stacking over family means (`combo_vol_stack`) is −2.2% vs equal
     family weights [−5.3, +1.1], p = 0.25 on `daily`; −1.5% on `all`.
   - Inverse-QLIKE weights are identical to equal weights to three decimals.
   - Choosing the past-best family is −14.7% (p = 0.02). Choosing the past-best
     model is −36% (p = 0.03).
   - The one learned scheme that scores well is unregistered stacking over all
     62 individual models (`x_stack_models`): +5.3% vs equal weight on `daily`
     [−4.0, +13.1], p = 0.30. All of that comes from 1926–45 (+16%). It is
     −6.5% in 2000–13 and +0.6% on `all`.
4. **How much does the 1926–45 weakness shrink?** On `daily`, hardly at all.
   - The equal-weight combos score 0.63–0.64 in 1926–45, 1.51–1.53 × `ref_last12m`
     (0.418). That is the middle of the leaders: `ml_glm` 1.26,
     `statespace_2c_daily` 1.30, HAR 1.47–1.56, `macro_base_h` 1.75. A
     combination inherits the average slow mean reversion.
   - On `all` the combos tie `ref_last12m` in 1926–45 (0.549–0.558 vs 0.557).
   - What does fix the era is fast, simple models: `garch_d_rm2006` (1.07) and
     `bench_vol_m_hl12` (1.14). Calibration makes the combos worse there (1.75).
5. **How stable are the learned weights?** Moderately stable after about 1975,
   not before.
   - Shrunk weights move by 3.4% a year on average (half the sum of absolute
     changes; maximum 35%, in the 1930s). The unshrunk optimum w* moves by 5.2%
     a year (maximum 51%).
   - GARCH gets 0.47–0.66 of the weight from 1945 to 1965, then 0.24–0.29.
     `statespace` rises from 0.03 to 0.28 after 1980.
   - `ml`, `ref` and `regime` get w* = 0 at 2014. The weights chase the last
     regime, which is why they do not beat equal weights.

**Correlation.** `combo_corr_ew` (equal weight over 25 stored correlation
matrices) is the only correlation forecast in the study that beats
`ref_corr_last120m` significantly on pairwise MSE: +4.7% [+2.1, +6.8], p = 0.013.
`combo_corr_hier` is +3.7% [+1.3, +5.7], p = 0.026. The single
`bench_corr_d_hl36` has a lower point MSE (+5.9%) with a wider interval
[−0.5, +11.3], p = 0.12. Under the covariance loss with fixed `ref_blend` vols,
the combinations rank first:

| correlation | `qlike_rel` | vs `ref_corr_last120m` (−0.213) |
|---|---|---|
| `combo_corr_hier_shrink` | −0.314 | p = 0.10 |
| `combo_corr_ew` | −0.312 | p = 0.04 |
| `mcorr_nls` | −0.301 | |

**Covariance and portfolios.** `combo_cov_main` (`combo_vol_hier_trim` ×
`combo_corr_ew`) scores `qlike_rel` −0.524 against −0.213 for the reference
composite (`ref_blend` × `ref_corr_last120m`): difference −0.311, p = 0.002.

| covariance | `qlike_rel` |
|---|---|
| `combo_cov_main` | −0.524 |
| `statespace_2c_daily` × `mcorr_nls` | −0.494 |
| `structural_factor_global` | −0.386 |
| reference composite | −0.213 |

It predicts fixed-weight portfolio risk with realized/predicted variance
0.97–1.11 (reference 1.11–1.38).

**Minimum-variance under-prediction.** The factor-of-two under-prediction of
minimum-variance risk is real for the reference: realized/predicted variance
2.40 (4.79 in 2000–13). The combos cut it to 1.36–1.41, and to 1.25 for
long-only minimum variance. That is still a 40% under-prediction, and 2.3 in
2000–13. The combos do not lower realized minimum-variance risk significantly
(+2.3%, p = 0.32). They churn more: 6.5% a month one-way vs 3.4%.

**Recommendation.**
- Volatility: `combo_vol_hier_trim`.
- Correlation: `combo_corr_ew`.
- Covariance: `combo_cov_main`.
- Sceptic's runner-up: `combo_cov_simple` = 12-month EWMA of squared monthly
  returns × one walk-forward scale constant, × the 3-year EW correlation of
  21-day sums (`bench_corr_d_hl36`). It gives up about 3–10% on vol QLIKE
  (not significant), much more in 2000–13, and the conditioning benefit in the
  optimizer (GMV under-prediction 1.69 vs 1.41).
- Details are in "Recommendation" below.

## Models

### Inputs and walk-forward discipline (shared)

- **Members are stored forecasts.** `store.load(kind, model)` reads
  `<results>/forecasts/<kind>/<model>.parquet` once per process and file
  version (path, mtime, size). It keeps the forecasts as dense arrays on the
  origin grid: variance for vol, the full matrix for corr; `"cov:<name>"` gives
  the correlations of a stored covariance model.
- `<results>` is `volmodel.paths.RESULTS`, read at call time, unless the
  instance has `results_dir` set.
- Nothing is substituted. A missing file, origin or asset raises
  `MissingComponentError` naming the component and the origin, at forecast
  time and at every training origin the weights or the calibration need.
- **Coverage check.** All 88 component files (62 vol, 23 corr, 3 cov) hold
  every month-end origin 1922-01..2014-12 (1,116 origins, no gaps, the same
  assets as the harness eligibility).
- **Estimation dates.** Weights and calibration constants are re-estimated
  each December E. A forecast at t uses E(t), the last December on or before
  t (protocol: at most once a year).
- **Training cells at E** are (origin s, asset a) such that:
  - s + 12 months ≤ E;
  - a is eligible at s by the harness rule (at least 36 monthly returns in the
    view at s, and at least 2 eligible assets);
  - a realized value exists.
- **Realized values** come from `targets.compute_targets` applied to
  `view.asof(E)` treated as a full panel. That is the project estimator: 21-day
  realized variance, with the monthly proxy where daily coverage fails. It is
  never read from the cached target files.
- A test checks that the result equals the project target exactly on every
  training cell and does not change when data after E are altered. The result
  is cached in the panel's shared cache under (E, digest of the data up to E).
- **Weighting of cells.** Every origin counts once, and the assets of an
  origin equally, as in the pooled scoreboard.
- **Shrinkage.** Learned quantities are shrunk toward their uninformative
  values with λ = Y / (Y + 20), where Y is the number of training years.
  - Why 20 years: half weight on the data after two to three volatility
    cycles; a round number fixed before any scoring.
  - Values: λ = 0.26 in 1929, 0.53 in 1945, 0.82 in 2014.

### Volatility (8 registered)

All averaging is arithmetic in **variance**, decided a priori:

- QLIKE is minimized by the conditional mean of the variance.
- A mean of unbiased variance forecasts is unbiased.
- Means of vols or log variances are biased low (Jensen) whenever members
  disagree, and QLIKE punishes under-forecasts most.
- A variant that averages in log variance (`x_logspace_hier`) is reported
  below: −2.4% on `daily` (n.s.), +0.1% on `all`.

| model | definition |
|---|---|
| `combo_vol_ew` | mean of the 62 registered vol models of other families |
| `combo_vol_hier` | mean within each of the 9 families, then equal weight across families |
| `combo_vol_hier_trim` | family means; per asset, drop the highest and lowest family, average the other 7 (applies with at least 5 groups) |
| `combo_vol_stack` | family means; simplex weights w* minimizing past QLIKE of G w (SLSQP from equal weights, pooled over assets); w = λ w* + (1−λ)/9 |
| `combo_vol_hier_cal` | `combo_vol_hier` × c |
| `combo_vol_stack_cal` | `combo_vol_stack` × c |
| `combo_vol_cal_m_hl12` | `bench_vol_m_hl12` × c |
| `combo_vol_cal_rm2006` | `garch_d_rm2006` × c |

- **Calibration constant.** c = 1 + λ (c_hat − 1), where c_hat is the mean
  over past training cells of realized / forecast variance. c_hat is exactly
  the QLIKE-minimizing multiplicative constant, since
  argmin_c mean(r/(c f) + log(c f)) = mean(r/f).
- **Families** are the model-name prefixes: bench (8 models), garch (8),
  har (8), macro (7), ml (7, including the within-family average `ml_avg`),
  ref (4), regime (8), statespace (8), structural (4).
- `ref_longrun` and other weak models are members by design. Equal weight over
  everything stored was the brief, and no member was chosen by its score.

### Correlation (4 registered)

A convex combination of correlation matrices is a correlation matrix: unit
diagonal, positive definite if any member is. No repair is ever applied.

| model | definition |
|---|---|
| `combo_corr_ew` | equal weight over all registered corr models of other families plus the correlations of the 3 `structural_factor_*` cov models: 25 matrices, excluding only `ref_corr_zero` (the identity is a diagnostic, not a forecast) |
| `combo_corr_hier` | family means (bench, condcorr, mcorr, ref, structural), equal weight across families |
| `combo_corr_hier_shrink` | `combo_corr_hier` shrunk 20% toward B(C) |
| `combo_corr_fastslow` | 0.5 `condcorr_msw` (fast: Markov stock–bond switching) + 0.5 `mcorr_shrink_block` (slow, well conditioned) |

- **Block target.** B(C) replaces every entry of C by its asset-group block
  mean: within-group off-diagonal mean, between-group mean, unit diagonal.
  B(C) is the average of P C P′ over the within-group permutations P, so it is
  positive definite, and so is any convex mix with C.
- **Why 20%.** Above the intensities `mcorr` estimated (0.08–0.19), because the
  aim is conditioning for optimizers, not correlation MSE. A round number fixed
  in advance.
- **Fast/slow members.** `condcorr_msw` was chosen over `msw_infl` a priori,
  because it does not depend on the inflation series.

### Covariance (4 registered; `Composite` of combos computed from the stored components)

| model | vol | corr | status |
|---|---|---|---|
| `combo_cov_hier` | `combo_vol_hier` | `combo_corr_hier` | pre-specified |
| `combo_cov_cal` | `combo_vol_hier_cal` | `combo_corr_hier_shrink` | pre-specified |
| `combo_cov_main` | `combo_vol_hier_trim` | `combo_corr_ew` | **the recommendation**; chosen after the vol and corr evaluation (a choice among 3 vol × 4 corr registered pieces, disclosed) |
| `combo_cov_simple` | `combo_vol_cal_m_hl12` | `bench_corr_d_hl36` | the runner-up |

`combo_cov_simple` was registered first with `combo_corr_fastslow`. That
version is reported below as `x_cov_simple_fastslow`. The pairing was changed
after the evaluation, because a sceptic's version should not need a Markov
model.

## Results: volatility

### Overall (registered combos, selected variants, the leaders, the references)

| model | `daily` QLIKE | R/F | skill vs `ref_last12m` [90% CI], p | skill vs `statespace_2c_daily` [CI], p | `all` QLIKE | R/F | skill vs `ref_blend` [CI], p |
|---|---|---|---|---|---|---|---|
| statespace_2c_daily | **0.358** | 1.10 | +29.0 [−3.1, 48.9] 0.12 | — | **0.393** | 1.03 | +26.8 [10.7, 37.1] 0.022 |
| x_stack_models (dropped) | 0.363 | 1.17 | +28.1 [1.9, 45.4] 0.085 | −1.3 [−16.8, 12.7] 0.88 | 0.394 | 1.11 | +26.6 [16.1, 34.0] 0.004 |
| ml_glm | 0.367 | 1.03 | +27.3 [−0.5, 43.2] 0.10 | −2.4 [−20.5, 10.0] 0.77 | 0.450 | 1.13 | +16.2 [−4.4, 33.1] 0.18 |
| **combo_vol_ew** | 0.383 | 1.09 | +24.1 [−11.8, 46.9] 0.22 | −6.9 [−20.8, 6.6] 0.37 | 0.396 | 1.03 | +26.2 [16.0, 33.2] 0.004 |
| **combo_vol_hier** | 0.387 | 1.10 | +23.3 [−13.2, 46.6] 0.24 | −8.0 [−22.2, 6.0] 0.32 | 0.399 | 1.03 | +25.8 [15.8, 32.7] 0.004 |
| **combo_vol_hier_trim** | 0.388 | 1.11 | +23.1 [−13.8, 46.9] 0.24 | −8.4 [−22.6, 6.3] 0.32 | **0.393** | 1.05 | +26.8 [18.4, 32.5] 0.002 |
| **combo_vol_cal_rm2006** | 0.390 | 1.29 | +22.7 [−3.0, 39.2] 0.13 | −9.0 [−24.7, 4.6] 0.27 | 0.402 | 1.13 | +25.2 [10.0, 35.5] 0.022 |
| har_pooled_qlike_d | 0.391 | 1.13 | +22.6 [−11.2, 43.9] 0.22 | −9.1 [−20.6, 1.7] 0.16 | 0.402 | 1.01 | +25.1 [11.0, 34.1] 0.017 |
| har_pooled_logcvx_d | 0.393 | 1.11 | +22.3 [−14.3, 45.7] 0.25 | −9.6 [−16.9, −1.0] 0.074 | 0.395 | 1.01 | +26.4 [13.2, 34.8] 0.011 |
| **combo_vol_stack** | 0.395 | 1.14 | +21.7 [−14.4, 45.0] 0.26 | −10.4 [−23.0, 2.8] 0.18 | 0.404 | 1.07 | +24.7 [15.1, 31.3] 0.004 |
| garch_d_rm2006 | 0.400 | 1.41 | +20.8 [−2.2, 35.7] 0.12 | −11.7 [−34.2, 6.9] 0.30 | 0.413 | 1.24 | +23.2 [7.1, 34.2] 0.035 |
| **combo_vol_cal_m_hl12** | 0.401 | 1.16 | +20.5 [−2.3, 35.6] 0.13 | −12.0 [−40.8, 11.1] 0.39 | 0.435 | 1.15 | +19.0 [6.9, 28.2] 0.023 |
| bench_vol_mr_est | 0.403 | 1.11 | +20.2 [−15.0, 42.3] 0.27 | −12.4 [−22.2, −3.1] 0.037 | 0.401 | 1.03 | +25.3 [13.5, 32.8] 0.009 |
| bench_vol_m_hl12 | 0.409 | 1.30 | +18.9 [−0.8, 34.1] 0.11 | −14.3 [−49.9, 15.8] 0.43 | 0.451 | 1.28 | +16.1 [2.5, 27.9] 0.060 |
| **combo_vol_hier_cal** | 0.411 | 1.11 | +18.7 [−22.4, 45.3] 0.37 | −14.6 [−29.9, 2.0] 0.14 | 0.411 | 1.04 | +23.4 [13.6, 30.2] 0.005 |
| regime_bocpd | 0.415 | 1.23 | +17.7 [−10.1, 37.0] 0.24 | −16.0 [−43.4, 8.0] 0.26 | 0.440 | 1.19 | +18.1 [6.7, 27.4] 0.022 |
| **combo_vol_stack_cal** | 0.417 | 1.14 | +17.4 [−23.3, 44.0] 0.39 | −16.4 [−29.9, −1.0] 0.084 | 0.416 | 1.06 | +22.6 [13.0, 29.3] 0.005 |
| macro_base_h | 0.418 | 1.08 | +17.2 [−24.6, 43.1] 0.41 | −16.7 [−32.0, −1.9] 0.068 | 0.474 | 1.15 | +11.8 [−4.2, 27.9] 0.21 |
| x_best_model (past-best single, dropped) | 0.520 | 1.48 | −3.0 [−43.1, 28.4] 0.88 | −45.1 [−68.1, −15.8] 0.027 | 0.508 | 1.31 | +5.4 [−7.1, 17.9] 0.45 |
| ref_last12m | 0.505 | 1.54 | — | −40.9 [−95.8, 3.0] 0.12 | 0.579 | 1.60 | −7.9 [−35.2, 13.1] 0.55 |
| ref_blend | 0.567 | 1.37 | −12.4 [−71.4, 29.2] 0.64 | −58.4 [−84.2, −25.2] 0.017 | 0.537 | 1.24 | — |

Notes on the table:
- R/F = mean over origins of the cross-asset mean of realized/forecast
  variance; 1 = QLIKE-calibrated. It is above 1 for almost everything here
  because 1926–45 and 2000–13 under-forecast.
- Against the ex-ante past-best model, every registered combo has skill
  +19% to +26% with p = 0.03–0.10.
- The 90% Model Confidence Set (R statistic, Götze-Künsch bootstrap) keeps
  all 20 of these models (registered combos, leaders, references and
  `x_best_model`) on both subsets.

### By era (QLIKE)

| model | `daily` 1926–45 | 1946–79 | 1980–99 | 2000–13 | 26–45 ÷ `ref_last12m` | `all` 1926–45 | 1946–79 | 1980–99 | 2000–13 |
|---|---|---|---|---|---|---|---|---|---|
| combo_vol_ew | 0.631 | **0.333** | 0.214 | 0.399 | 1.51 | 0.549 | **0.380** | 0.222 | 0.399 |
| combo_vol_hier | 0.640 | 0.336 | 0.213 | 0.405 | 1.53 | 0.555 | 0.382 | 0.221 | 0.405 |
| combo_vol_hier_trim | 0.640 | 0.338 | 0.213 | 0.410 | 1.53 | 0.554 | 0.384 | 0.221 | 0.410 |
| combo_vol_stack | 0.640 | 0.360 | 0.213 | 0.399 | 1.53 | 0.558 | 0.398 | 0.221 | 0.399 |
| combo_vol_hier_cal | 0.732 | 0.340 | 0.216 | 0.412 | 1.75 | 0.601 | 0.386 | 0.224 | 0.412 |
| combo_vol_stack_cal | 0.729 | 0.362 | 0.218 | 0.399 | 1.75 | 0.604 | 0.400 | 0.226 | 0.399 |
| combo_vol_cal_m_hl12 | 0.531 | 0.347 | 0.239 | 0.585 | 1.27 | 0.543 | 0.431 | 0.242 | 0.585 |
| combo_vol_cal_rm2006 | 0.497 | 0.442 | 0.246 | 0.324 | 1.19 | 0.501 | 0.456 | 0.254 | 0.324 |
| x_stack_models | 0.530 | 0.332 | **0.208** | 0.425 | 1.27 | 0.503 | 0.395 | **0.217** | 0.425 |
| x_best_model | 0.867 | 0.462 | 0.238 | 0.576 | 2.08 | 0.757 | 0.476 | 0.249 | 0.576 |
| statespace_2c_daily | 0.545 | 0.352 | 0.236 | **0.288** | 1.30 | 0.527 | 0.426 | 0.249 | **0.288** |
| ml_glm | 0.525 | 0.353 | 0.226 | 0.382 | 1.26 | 0.572 | 0.444 | 0.238 | 0.382 |
| har_pooled_logcvx_d | 0.651 | 0.352 | 0.239 | 0.348 | 1.56 | 0.545 | 0.402 | 0.247 | 0.348 |
| garch_d_rm2006 | 0.448 | 0.488 | 0.247 | 0.339 | 1.07 | **0.476** | 0.491 | 0.258 | 0.339 |
| bench_vol_mr_est | 0.617 | 0.383 | 0.237 | 0.388 | 1.48 | 0.518 | 0.415 | 0.250 | 0.388 |
| bench_vol_m_hl12 | 0.477 | 0.378 | 0.229 | 0.649 | 1.14 | 0.514 | 0.468 | 0.233 | 0.649 |
| regime_bocpd | 0.561 | 0.364 | 0.238 | 0.591 | 1.34 | 0.541 | 0.444 | 0.237 | 0.591 |
| macro_base_h | 0.731 | 0.345 | 0.245 | 0.407 | 1.75 | 0.841 | 0.426 | 0.252 | 0.407 |
| ref_last12m | **0.418** | 0.599 | 0.313 | 0.673 | 1.00 | 0.557 | 0.701 | 0.336 | 0.673 |
| ref_blend | 0.958 | 0.471 | 0.313 | 0.618 | 2.29 | 0.774 | 0.515 | 0.288 | 0.618 |
| origins | 235 | 408 | 240 | 168 | | 240 | 408 | 240 | 168 |

**Mean realized/forecast variance by era** (`daily`):

| model | 1926–45 | 1946–79 | 1980–99 | 2000–13 |
|---|---|---|---|---|
| combo_vol_hier | 1.47 | 0.96 | 0.86 | 1.23 |
| combo_vol_hier_trim | 1.51 | 0.98 | 0.86 | 1.24 |
| combo_vol_cal_m_hl12 | 1.29 | 1.18 | 0.87 | 1.32 |
| combo_vol_cal_rm2006 | 1.54 | 1.48 | 0.92 | 1.01 |
| statespace_2c_daily | 1.51 | 1.07 | 0.83 | 1.00 |
| bench_vol_m_hl12 (raw) | 1.31 | 1.33 | 1.04 | 1.57 |
| garch_d_rm2006 (raw) | 1.52 | 1.60 | 1.10 | 1.20 |

- Calibration takes `rm2006` from 1.20 to 1.01 in 2000–13, and `m_hl12` from
  1.57 to 1.32 there and from 1.33 to 1.18 in 1946–79.
- The combos are about 1 after 1946 but 1.23 in 2000–13. They inherit the
  members that do not use daily data, which reacted late to 2008.

### By asset (QLIKE, `daily`, cells common to all models)

| model | us_large | us_small | intl_dev | em | ust_int | ust_long | tips | ig_credit | hy_credit | reits | gold | commod. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| combo_vol_ew | 0.329 | 0.326 | 0.282 | 0.284 | 0.332 | 0.334 | 0.312 | 0.321 | 0.661 | 0.391 | 0.328 | 0.249 |
| combo_vol_hier_trim | 0.331 | 0.330 | 0.287 | 0.289 | 0.334 | 0.342 | 0.317 | 0.323 | 0.672 | 0.413 | 0.328 | 0.252 |
| combo_vol_stack | 0.331 | 0.329 | 0.280 | 0.282 | 0.370 | 0.363 | 0.320 | 0.322 | 0.674 | 0.377 | 0.334 | 0.246 |
| combo_vol_cal_m_hl12 | 0.370 | 0.329 | 0.429 | 0.368 | 0.427 | 0.440 | 0.428 | 0.489 | 0.866 | 0.512 | 0.319 | 0.309 |
| combo_vol_cal_rm2006 | 0.304 | 0.346 | 0.257 | 0.303 | 0.471 | 0.422 | 0.348 | 0.265 | 0.487 | 0.369 | 0.358 | 0.242 |
| statespace_2c_daily | 0.287 | 0.327 | 0.256 | 0.338 | 0.395 | 0.339 | 0.320 | 0.242 | **0.421** | 0.264 | 0.310 | 0.222 |
| ml_glm | 0.322 | 0.299 | 0.259 | 0.296 | 0.362 | 0.337 | 0.278 | 0.309 | 0.634 | 0.410 | 0.331 | 0.244 |
| har_pooled_logcvx_d | 0.353 | 0.310 | 0.270 | 0.271 | 0.387 | 0.353 | 0.310 | 0.269 | 0.598 | 0.303 | 0.353 | 0.257 |
| ref_last12m | 0.390 | 0.402 | 0.517 | 0.473 | 0.712 | 0.678 | 0.508 | 0.616 | 1.062 | 0.592 | 0.420 | 0.406 |
| ref_blend | 0.470 | 0.414 | 0.380 | 0.349 | 0.517 | 0.645 | 0.562 | 0.466 | 1.070 | 0.660 | 0.374 | 0.308 |

- The combos are best or close to best on the rates assets (ust_int 0.33 vs
  0.36–0.40 for the leaders).
- They are worst among the leaders on hy_credit (0.66 vs 0.42 for
  `statespace_2c_daily`) and reits.
- On `all` the pattern is similar. ig_credit is 0.49 for the combos vs 0.46–0.65
  for the leaders.

### Level calibration (each single model calibrated walk-forward, pooled constant)

Selected rows. The full 67-row table is in the generated tables. The d
columns are the change in era mean QLIKE (negative = calibration helped).

| model, `daily` | raw | calibrated | skill [90% CI] | p | d 26–45 | d 46–79 | d 80–99 | d 00–13 |
|---|---|---|---|---|---|---|---|---|
| ref_last12m | 0.505 | 0.416 | +17.5 [6.6, 23.2] | 0.033 | −0.020 | −0.141 | −0.015 | −0.162 |
| garch_d_rm2006 | 0.400 | 0.390 | +2.4 [−4.1, 8.5] | 0.50 | +0.049 | −0.046 | −0.001 | −0.015 |
| bench_vol_m_hl12 | 0.409 | 0.401 | +2.0 [−8.1, 8.2] | 0.67 | +0.054 | −0.031 | +0.010 | −0.064 |
| bench_vol_m_hl12, per-group c | 0.409 | 0.395 | +3.4 [−7.7, 9.7] | 0.51 | +0.031 | −0.035 | +0.019 | −0.070 |
| statespace_2c_daily | 0.358 | 0.378 | −5.5 [−9.2, −0.1] | 0.097 | +0.073 | +0.002 | +0.009 | +0.003 |
| ml_glm | 0.367 | 0.394 | −7.5 [−13.8, −2.5] | 0.014 | +0.078 | +0.011 | +0.027 | −0.003 |
| har_pooled_logcvx_d | 0.393 | 0.415 | −5.6 [−8.9, −0.5] | 0.081 | +0.084 | +0.003 | +0.007 | +0.004 |
| combo_vol_hier → `_cal` | 0.387 | 0.411 | −6.1 [−9.6, −1.1] | 0.060 | +0.092 | +0.003 | +0.003 | +0.007 |
| combo_vol_hier, per-group c | 0.387 | 0.411 | −6.2 [−15.7, −0.5] | 0.071 | +0.053 | +0.026 | +0.006 | +0.004 |
| combo_vol_stack → `_cal` | 0.395 | 0.417 | −5.5 [−9.2, −0.5] | 0.078 | +0.089 | +0.002 | +0.005 | +0.000 |

- **Count.** Calibration improves 11 of 67 calibrated-vs-raw pairs on `daily`
  and 19 of 67 on `all`. It is significantly better (p < 0.10) in 1 and
  significantly worse in 24 (`daily`) and 22 (`all`).
- **On `all`.** `ref_last12m` +17.3% [+9.9, +22.3], p = 0.006; `rm2006` +2.7%
  (p = 0.26); `m_hl12` +3.5% (p = 0.34). The regression leaders are −2.6% to −3%
  (p 0.02–0.05).

**Calibration constants through time** (c applied, with c_hat in brackets):

| December | hier_cal | stack_cal | cal_m_hl12 | cal_rm2006 |
|---|---|---|---|---|
| 1929 | 0.92 (0.69) | 0.93 (0.72) | 0.97 (0.87) | 0.92 (0.68) |
| 1935 | 1.15 (1.37) | 1.14 (1.36) | 1.15 (1.38) | 1.09 (1.23) |
| 1945 | 1.03 (1.05) | 1.03 (1.05) | 1.06 (1.12) | 1.05 (1.09) |
| 1965 | 0.94 (0.91) | 0.96 (0.94) | 1.08 (1.12) | 1.04 (1.06) |
| 1985 | 1.02 (1.03) | 1.03 (1.05) | 1.20 (1.26) | 1.19 (1.26) |
| 2005 | 1.00 (1.00) | 1.02 (1.02) | 1.19 (1.23) | 1.19 (1.24) |
| 2014 | 1.03 (1.03) | 1.03 (1.04) | 1.23 (1.28) | 1.20 (1.24) |

- The combos are already calibrated (c within 3% of 1 after 1945), so there is
  little to gain and the early noise costs.
- The EWMA and RiskMetrics need about +20% in variance. Per asset group in
  2014: credit 1.39, rates 1.28, real 1.20, equity 1.19.

### Learned versus equal weights, and aggregation choices

| comparison | `daily` skill [CI], p | 26–45 | 46–79 | 80–99 | 00–13 | `all` skill, p |
|---|---|---|---|---|---|---|
| combo_vol_stack vs combo_vol_hier | −2.2 [−5.3, 1.1] 0.25 | −0.0 | −7.1 | −0.0 | +1.5 | −1.5, 0.28 |
| x_invq vs combo_vol_hier | +0.0 [−0.2, 0.3] 0.96 | +0.2 | −0.3 | −0.1 | +0.1 | −0.1, 0.45 |
| x_best_family vs combo_vol_hier | −14.7 [−23.1, −5.8] 0.018 | −7.6 | −25.0 | −2.4 | −18.6 | −14.4, 0.04 |
| x_stack_models vs combo_vol_ew | +5.3 [−4.0, 13.1] 0.30 | +16.0 | +0.4 | +2.7 | −6.5 | +0.6, 0.84 |
| x_best_model vs combo_vol_ew | −35.7 [−53.4, −12.5] 0.028 | −37.5 | −38.7 | −11.5 | −44.3 | −28.1, 0.03 |
| combo_vol_stack_cal vs combo_vol_hier_cal | −1.6 [−4.5, 1.3] 0.34 | +0.4 | −6.5 | −0.9 | +3.0 | −1.0, 0.40 |
| combo_vol_hier vs combo_vol_ew | −1.0 [−1.6, −0.3] 0.041 | −1.5 | −0.8 | +0.2 | −1.4 | −0.6, 0.11 |
| combo_vol_hier_trim vs combo_vol_hier | −0.4 [−1.5, 1.5] 0.68 | −0.1 | −0.5 | +0.1 | −1.1 | +1.4, 0.52 |
| x_median vs combo_vol_hier | −0.7 [−2.7, 2.2] 0.63 | +0.5 | −1.1 | −0.2 | −2.8 | +0.5, 0.83 |
| x_logspace_hier vs combo_vol_hier | −2.4 [−6.4, 4.7] 0.48 | −5.1 | −0.4 | +2.2 | −3.8 | +0.1, 0.97 |

- Equal weight over all 62 models is 1% better than the family-balanced mean
  (p = 0.04 on `daily`, 0.11 on `all`). The families with more near-duplicate
  members (HAR, statespace, GARCH, regime) are the better ones on average.
- That is a property of this zoo, not a reason to prefer a scheme that lets
  any family which registers 8 variants dominate.

**`combo_vol_stack` family weights (after shrinkage; equal = 0.111):**

| December | bench | garch | har | macro | ml | ref | regime | statespace | structural | λ |
|---|---|---|---|---|---|---|---|---|---|---|
| 1929 | 0.24 | 0.08 | 0.08 | 0.18 | 0.08 | 0.08 | 0.08 | 0.08 | 0.08 | 0.26 |
| 1945 | 0.16 | 0.47 | 0.06 | 0.05 | 0.05 | 0.05 | 0.05 | 0.05 | 0.05 | 0.53 |
| 1965 | 0.04 | 0.66 | 0.09 | 0.04 | 0.04 | 0.04 | 0.04 | 0.04 | 0.04 | 0.68 |
| 1985 | 0.31 | 0.24 | 0.11 | 0.08 | 0.03 | 0.03 | 0.03 | 0.15 | 0.03 | 0.76 |
| 2005 | 0.32 | 0.26 | 0.07 | 0.09 | 0.02 | 0.02 | 0.02 | 0.18 | 0.03 | 0.81 |
| 2014 | 0.09 | 0.29 | 0.13 | 0.06 | 0.02 | 0.02 | 0.02 | 0.28 | 0.10 | 0.82 |

- Yearly turnover of the weights, 1930–2014: shrunk 0.034 on average (maximum
  0.35); unshrunk w* 0.052 (maximum 0.51).
- Unshrunk w* at 2014: garch 0.33, statespace 0.31, har 0.13, bench 0.09,
  structural 0.09, macro 0.05, and 0 for ml, ref and regime.
- The weight on GARCH in 1945–65 is what costs stacking 7% in 1946–79. GARCH
  had been the best family through the Depression and was then the worst
  leader in 1946–79 (0.49).

## Results: correlation

Pairwise MSE, `daily` subset: 1,015 common origins (1929-06..2013-12); skill
is vs `ref_corr_last120m`. The `stock_bond` column covers 625 origins
1961–2013; `full12` covers 168 origins 2000–13.

| model | MSE | bias | skill [90% CI], p | 1926–45 | 1946–79 | 1980–99 | 2000–13 | `stock_bond` MSE | `full12` MSE |
|---|---|---|---|---|---|---|---|---|---|
| bench_corr_d_hl36 | **0.0406** | +0.001 | +5.9 [−0.5, 11.3] 0.12 | 0.0017 | 0.0253 | 0.0720 | 0.0791 | 0.0991 | 0.0791 |
| bench_corr_shrink_block | 0.0409 | +0.001 | +5.3 [−1.4, 11.3] 0.17 | 0.0017 | 0.0253 | 0.0730 | 0.0793 | 0.0978 | 0.0793 |
| **combo_corr_ew** | 0.0412 | −0.007 | +4.7 [**2.1, 6.8**] **0.013** | 0.0029 | 0.0256 | **0.0706** | 0.0823 | 0.1062 | 0.0823 |
| **combo_corr_hier** | 0.0416 | −0.009 | +3.7 [**1.3, 5.7**] **0.026** | 0.0029 | 0.0257 | 0.0713 | 0.0834 | 0.1082 | 0.0834 |
| **combo_corr_hier_shrink** | 0.0422 | −0.009 | +2.2 [−1.3, 5.6] 0.27 | 0.0029 | 0.0255 | 0.0732 | 0.0852 | 0.1071 | 0.0852 |
| **combo_corr_fastslow** | 0.0427 | −0.009 | +1.1 [−4.6, 5.9] 0.70 | 0.0029 | 0.0298 | 0.0712 | 0.0804 | 0.1064 | 0.0804 |
| condcorr_msw_infl | 0.0429 | −0.014 | +0.7 [−7.5, 7.7] 0.87 | 0.0019 | 0.0307 | 0.0733 | **0.0773** | **0.0925** | **0.0773** |
| ref_corr_last120m | 0.0432 | −0.003 | — | 0.0019 | 0.0269 | 0.0764 | 0.0842 | 0.1143 | 0.0842 |
| mcorr_shrink_block | 0.0432 | −0.004 | −0.0 | 0.0046 | 0.0275 | 0.0712 | 0.0870 | 0.1204 | 0.0870 |
| mcorr_nls | 0.0440 | −0.018 | −2.0 | 0.0074 | 0.0292 | 0.0703 | 0.0860 | 0.1218 | 0.0860 |
| condcorr_msw | 0.0444 | −0.015 | −2.8 | 0.0019 | 0.0341 | 0.0735 | 0.0782 | 0.0990 | 0.0782 |
| mcorr_dcc | 0.0466 | −0.016 | −8.0 | 0.0231 | 0.0291 | 0.0703 | 0.0830 | 0.1115 | 0.0830 |

- **us_large–ust_int squared error** (1946–79 / 1980–99 / 2000–13):
  - `combo_corr_ew` 0.054 / 0.097 / 0.186;
  - `combo_corr_fastslow` 0.072 / 0.093 / 0.169;
  - `condcorr_msw_infl` 0.071 / 0.091 / 0.122;
  - `ref_corr_last120m` 0.057 / 0.104 / 0.204.
- The equal-weight combinations adapt to the stock–bond sign change only
  partly. The Markov models adapt fast but raise false alarms in 1946–79.
  Mixing the two (`fastslow`) lands in between.

### Covariance loss with fixed vols (`qlike_rel`, `daily`, 1,015 origins)

| corr model | with `ref_blend` vols | diff vs ref corr, p | 2000–13 | with `combo_vol_hier` vols | diff, p |
|---|---|---|---|---|---|
| x_corr_ew_shrink (dropped) | **−0.316** | −0.103, 0.084 | −0.189 | −0.521 | −0.050, 0.22 |
| combo_corr_hier_shrink | −0.314 | −0.101, 0.099 | −0.194 | −0.518 | −0.047, 0.26 |
| combo_corr_ew | −0.312 | −0.099, 0.043 | −0.133 | −0.524 | −0.052, 0.11 |
| combo_corr_hier | −0.309 | −0.096, 0.057 | −0.135 | −0.521 | −0.049, 0.15 |
| mcorr_nls | −0.301 | −0.089, 0.029 | +0.030 | −0.488 | −0.017, 0.38 |
| mcorr_dcc | −0.298 | −0.085, 0.015 | −0.063 | −0.514 | −0.042, 0.14 |
| combo_corr_fastslow | −0.287 | −0.074, 0.021 | +0.010 | −0.506 | −0.034, 0.057 |
| bench_corr_shrink_block | −0.263 | −0.050, 0.43 | **−0.211** | **−0.532** | −0.060, 0.15 |
| bench_corr_d_hl36 | −0.228 | −0.015, 0.73 | −0.025 | −0.515 | −0.043, 0.071 |
| ref_corr_last120m | −0.213 | — | +0.187 | −0.472 | — |

- Paired with the better (combined) vols, the correlation differences shrink
  and none is significant at 10% except `bench_corr_d_hl36` (p = 0.07) and
  `fastslow` (p = 0.06).
- Combination is the most consistent choice: second to fourth under every
  pairing and the only significant MSE gain.

## Results: covariance and portfolios

`qlike_rel` on four subsets. `daily` has 1,015 origins; `sbg4` has 473
(1974–2013); `full12` has 168 (2000–13).

| cov model | `daily` | diff vs ref, p | 1926–45 | 1946–79 | 1980–99 | 2000–13 | `sbg4` | `core9` (326) | `full12` |
|---|---|---|---|---|---|---|---|---|---|
| **combo_cov_main** | **−0.524** | −0.311, 0.002 | −0.653 | −0.477 | **−0.619** | −0.351 | −0.250 | −0.421 | −0.351 |
| combo_cov_hier | −0.521 | −0.308, 0.003 | −0.635 | −0.478 | −0.612 | −0.358 | −0.251 | −0.423 | −0.358 |
| combo_vol_stack × combo_corr_hier_shrink | −0.515 | −0.302, 0.006 | −0.666 | −0.455 | −0.584 | −0.382 | −0.256 | −0.423 | −0.382 |
| combo_cov_cal | −0.504 | −0.291, 0.006 | −0.584 | −0.473 | −0.582 | −0.373 | −0.254 | −0.416 | −0.373 |
| statespace_2c_daily × mcorr_nls | −0.494 | −0.281, 0.007 | −0.601 | −0.472 | −0.551 | −0.338 | **−0.257** | −0.399 | −0.338 |
| **combo_cov_simple** | −0.492 | −0.280, 0.003 | **−0.721** | **−0.482** | −0.614 | −0.073 | −0.185 | −0.256 | −0.073 |
| combo_vol_cal_rm2006 × combo_corr_hier | −0.475 | −0.262, 0.021 | −0.649 | −0.367 | −0.554 | **−0.419** | −0.251 | **−0.436** | **−0.419** |
| x_cov_simple_fastslow (first `combo_cov_simple`) | −0.463 | −0.251, 0.010 | −0.712 | −0.474 | −0.542 | −0.031 | −0.177 | −0.192 | −0.031 |
| structural_factor_global | −0.386 | −0.174, 0.025 | −0.292 | −0.394 | −0.577 | −0.208 | −0.125 | −0.323 | −0.208 |
| structural_factor_lognormal | −0.379 | −0.166, 0.024 | −0.292 | −0.394 | −0.577 | −0.161 | −0.125 | −0.314 | −0.161 |
| ref_blend × ref_corr_last120m | −0.213 | — | −0.190 | −0.256 | −0.438 | +0.187 | −0.056 | −0.142 | +0.187 |

On `sbg4` every combo beats the reference composite at p = 0.005–0.03. On
`full12` (168 origins) no difference is significant (p ≥ 0.19).

### Portfolio risk prediction (QLIKE / mean realized ÷ predicted variance; common origins, 625 for 60/40)

| cov model | 60/40 | balanced | equity mix | ERC | random |
|---|---|---|---|---|---|
| combo_cov_main | 0.240 / 0.97 | 0.283 / 1.06 | 0.356 / 1.11 | 0.362 / 1.09 | 0.343 / 1.03 |
| combo_cov_hier | 0.240 / 0.96 | 0.282 / 1.05 | 0.356 / 1.10 | 0.361 / 1.08 | 0.342 / 1.02 |
| combo_cov_cal | 0.243 / 0.97 | 0.286 / 1.05 | 0.361 / 1.10 | 0.377 / 1.08 | 0.357 / 1.02 |
| combo_cov_simple | 0.359 / 1.11 | 0.396 / 1.15 | 0.493 / 1.19 | 0.392 / 1.16 | 0.355 / 1.08 |
| statespace_2c_daily × mcorr_nls | **0.223** / 0.86 | **0.240** / 0.91 | **0.279** / 0.91 | **0.326** / 1.09 | **0.299** / 0.99 |
| structural_factor_global | 0.314 / 1.13 | 0.380 / 1.23 | 0.494 / 1.31 | 0.555 / 1.48 | 0.521 / 1.34 |
| ref_blend × ref_corr_last120m | 0.312 / 1.11 | 0.363 / 1.19 | 0.480 / 1.29 | 0.520 / 1.36 | 0.467 / 1.19 |

### Optimized portfolios (common origins)

| cov model | GMV realized var ×1e4 | GMV RV/PV | GMV turnover / gross | long-only GMV realized var ×1e4 | long-only RV/PV | long-only turnover | ERC 10%-target \|vol−10%\| |
|---|---|---|---|---|---|---|---|
| combo_cov_main | 150.8 | 1.41 | 0.065 / 2.37 | 204.4 | 1.25 | 0.029 | 0.032 |
| combo_cov_hier | 150.6 | 1.38 | 0.064 / 2.36 | 204.4 | 1.24 | 0.029 | 0.031 |
| combo_cov_cal | 150.6 | **1.36** | 0.059 / 2.32 | 204.4 | **1.24** | 0.029 | 0.032 |
| combo_cov_simple | **147.4** | 1.69 | 0.067 / 2.49 | 204.8 | 1.34 | 0.032 | 0.032 |
| statespace_2c_daily × mcorr_nls | 196.2 | 1.54 | 0.117 / 2.06 | 225.7 | 1.33 | 0.064 | **0.030** |
| structural_factor_global | 165.6 | 1.93 | 0.043 / 2.26 | 204.4 | 1.70 | 0.023 | 0.037 |
| ref_blend × ref_corr_last120m | 154.3 | 2.40 | **0.034** / 2.49 | 205.3 | 1.76 | **0.010** | 0.038 |

GMV realized/predicted variance by era (1926–45 / 1946–79 / 1980–99 / 2000–13):

| cov model | 1926–45 | 1946–79 | 1980–99 | 2000–13 |
|---|---|---|---|---|
| combo_cov_main | 1.39 | 1.42 | 0.77 | 2.31 |
| combo_cov_simple | 0.96 | 1.58 | 1.10 | 3.64 |
| ref composite | 2.78 | 1.93 | 1.23 | 4.79 |
| structural_factor_global | 1.79 | 1.71 | 1.10 | 3.82 |

- **Realized GMV variance vs the reference composite:**
  - `combo_cov_main` +2.3% [−1.1, +15.3], p = 0.32;
  - `combo_cov_simple` +4.5% [−1.3, +22.0], p = 0.21;
  - `statespace_2c_daily` × `mcorr_nls` −27% [−51, +45], p = 0.30;
  - long-only differences are under 0.5% (except −10% for `statespace`
    × `nls`).
- The `statespace` composite's high GMV variance is driven by a few windows
  (its interval spans ±50%). I did not investigate it further.

### What a planner should know about using the matrix in an optimizer

- **Fixed-weight risk is about right.** For fixed-weight portfolios (policy
  mixes, ERC, random long-only) the combined matrix is close to calibrated on
  average: realized/predicted variance 0.97–1.11. By era it is 0.8–0.9 in
  1980–99 and 1.3–1.4 in 2000–13 for the balanced and random portfolios. The
  under-prediction in 2000–13 comes from the vols, not the correlations.
- **Minimum-variance risk is still under-predicted.** It is by about 40% on
  average (RV/PV 1.41; long-only 1.25) and by a factor of 2.3 in 2000–13. The
  reference matrix under-predicts by 2.4 on average and 4.8 in 2000–13.
  - Combination plus the better-conditioned members (`mcorr`, structural,
    block-shrunk bench) roughly halves the excess.
  - Block shrinkage of the combined matrix (`combo_cov_cal`) helps a little
    more (1.36).
  - No candidate removes the excess. The optimizer still finds directions whose
    forecast variance is too small.
- **Practical rules.**
  - Do not report a minimum-variance or leveraged long–short portfolio's
    predicted vol from this matrix without a haircut. Scale its predicted
    variance up by about 1.5 (2–2.5 in stress eras), or impose long-only or
    gross-exposure limits (long-only GMV is 1.25).
  - The optimized weights churn: 6.5% a month one-way for unconstrained GMV,
    against 3.4% for the reference and 2.9% long-only. Smooth them or rebalance
    on a band.
- **What combination does not change.** It does not lower the realized risk of
  minimum-variance portfolios detectably (+2%, n.s.). What it improves is the
  prediction of that risk.

## Recommendation (made without the holdout)

**Ship `combo_cov_main` = `combo_vol_hier_trim` × `combo_corr_ew`.**

*Volatility: `combo_vol_hier_trim`.*
- **Score.** On `all` it ties the best single model (0.393), with the
  tightest interval against the reference of any model: +26.8% [+18.4, +32.5].
  On `daily` it is 8% behind `statespace_2c_daily`, not significantly.
- **Why not the best single model.** No single model is identifiable ex ante:
  the past-best rule is a disaster (0.52), all 20 models stay in the MCS, and
  the leaders have opposite era profiles (`statespace` 0.29 in 2000–13 but
  third in 1946–99; `ml_glm` second on `daily` but 0.450 on `all`).
- **Why the trimmed family mean.**
  - Family balance makes it insensitive to how many variants a family
    registered.
  - The trim drops the highest and lowest family per asset, so one family that
    breaks after the data correction or a re-run cannot move the forecast much.
  - It needs no estimated parameter.
- **Why not the alternatives.**
  - `combo_vol_ew` is 1% better on `daily` but lets the families with eight
    near-duplicates dominate.
  - Stacking and calibration lose.
- **Cost.** It depends on all 62 models being re-run before every forecast.
  That is operationally heavy, and it inherits their failures. Its
  realized/forecast variance in 2000–13 was 1.24.

*Correlation: `combo_corr_ew`.* It is the only significant improvement on
correlation MSE in the study (+4.7% [+2.1, +6.8]) and is among the best under
the covariance loss with either set of vols. It needs no parameter and is valid
by construction.

*Covariance: `combo_cov_main`.*
- `qlike_rel` −0.524, the best on `daily`; −0.311 vs the reference, p = 0.002.
- Fixed-portfolio risk calibrated to within 10%.
- GMV risk under-predicted by 1.4 rather than 2.4.

**Runner-up a sceptic can rebuild in an afternoon: `combo_cov_simple`.**

- Vol = 12-month-half-life EWMA of squared monthly log excess returns (`12 ×
  EW_12(r²)`), times c, where c is updated every December. c is the shrunk mean
  of realized/forecast variance over all past fully observed 12-month windows,
  with c = 1 + Y/(Y+20) (c_hat − 1). It is about 1.23 today.
- Correlation = 3-year-half-life EW correlation of 21-day return sums
  (`bench_corr_d_hl36`).
- No daily-data mapping beyond the 21-day sums, no model zoo.
- **What it gives up:**
  - vol QLIKE 0.401 vs 0.388 on `daily` (−3%) and 0.435 vs 0.393 on `all`
    (−11%); neither significant;
  - 2000–13 badly (0.585 vs 0.410): the EWMA was slow into and out of 2008;
  - the rates assets (ust_int 0.43 vs 0.33);
  - covariance loss: −0.492 vs −0.524 on `daily` (n.s. between them; both
    beat the reference at p ≤ 0.003), +0.28 worse on `full12`;
  - fixed-portfolio risk under-predicted by 10–20% instead of 0–10%;
  - GMV risk under-predicted by 1.69 instead of 1.41.
- **What it gains:** the best 1926–45 covariance score of any candidate, and the
  lowest realized GMV variance (147 vs 151 ×10⁻⁴, n.s.).
- **Daily-data alternative.** The same recipe on `garch_d_rm2006`
  (`combo_vol_cal_rm2006`) is better on vol (0.390 `daily`, 0.402 `all`, and
  the best calibrated model in 2000–13 at 0.324). It needs the 21-day/1-day
  variance-ratio mapping from the GARCH family, so it is a day's work, not an
  afternoon's.

## Tried and dropped (all scored on the same origins; not registered)

| variant | what | `daily` | `all` | finding |
|---|---|---|---|---|
| `x_stack_models` | shrunk stacking over all 62 single models | 0.363 | 0.394 | best combination on `daily`, but all the gain is 1926–45 (+16% vs EW) and it is −6.5% in 2000–13; vs EW +5.3% (p 0.30) / +0.6% |
| `x_best_model` | pick the past-best single model each December (cautionary) | 0.520 | 0.508 | worse than `ref_last12m` on `daily`; −36% vs EW (p 0.03) |
| `x_best_family` | pick the past-best family mean | 0.444 | 0.456 | −15% vs equal (p 0.02) |
| `x_invq` | inverse past-QLIKE family weights, shrunk | 0.387 | 0.399 | identical to equal weights (past QLIKEs too similar) |
| `x_median` | median of family means | 0.390 | 0.397 | = trimmed mean |
| `x_logspace_hier` | hierarchical mean in log variance | 0.396 | 0.398 | −2.4% / +0.1%, n.s.; biased low in 1926–45 (−5%) |
| `x_ew_cal` | `combo_vol_ew` × c | 0.406 | 0.409 | −6% / −3% vs raw |
| `x_hier_calgroup` | per-asset-group calibration of `combo_vol_hier` | 0.411 | 0.415 | no better than pooled |
| `x_cal_m_hl12_group` | per-group calibration of `bench_vol_m_hl12` | 0.395 | 0.438 | +1.5% vs pooled on `daily`, −0.7% on `all` |
| `cal:<model>` | pooled calibration of each of the 62 single models | — | — | see the calibration section; best: `cal:ref_last12m` 0.416 / 0.479 (+17%, the only significant gain) |
| `x_corr_ew_shrink` | `combo_corr_ew` + 20% block shrinkage | MSE 0.0418 | — | best `qlike_rel` with `ref_blend` vols (−0.316) but corr MSE +1.5% worse than EW and n.s. vs ref |
| `x_corr_fastslow_infl` | `msw_infl` instead of `msw` in fast/slow | MSE 0.0420 | — | slightly better than registered fast/slow (+1.7%), same cov loss |
| `x_corr_dcc_nls` | 0.5 `mcorr_dcc` + 0.5 `mcorr_nls` | MSE 0.0442 | — | inherits DCC's 1929–45 error; `qlike_rel` −0.314 |
| `x_corr_fastslow_shrink` | fast/slow + 20% block shrinkage | MSE 0.0433 | — | no gain |
| `x_cov_simple_fastslow` | first definition of `combo_cov_simple` (corr = `combo_corr_fastslow`) | qlike_rel −0.463 | — | replaced by `bench_corr_d_hl36` (−0.492) for simplicity; disclosed post-hoc change |
| cov composites | `combo_vol_stack` × `combo_corr_hier_shrink`, `combo_vol_cal_rm2006` × `combo_corr_hier`, `statespace_2c_daily` × `mcorr_nls` | −0.515 / −0.475 / −0.494 | — | see the covariance table |

## Run time

Full monthly run, 1,116 origins (1922-01..2014-12), one core, shared machine.
Times are seconds, total / model:

| model | seconds | model | seconds |
|---|---|---|---|
| `combo_vol_ew` | 5.2 / 3.6 | `combo_corr_ew` | 6.2 / 4.4 |
| `combo_vol_hier` | 4.3 / 2.7 | `combo_corr_hier` | 3.8 / 1.8 |
| `combo_vol_hier_trim` | 4.4 / 2.7 | `combo_corr_hier_shrink` | 3.9 / 2.1 |
| `combo_vol_stack` | 12.6 / 11.0 | `combo_corr_fastslow` | 2.7 / 0.6 |
| `combo_vol_hier_cal` | 6.4 / 4.8 | `combo_cov_hier` | 7.2 / 5.1 |
| `combo_vol_stack_cal` | 6.7 / 5.0 | `combo_cov_cal` | 9.3 / 7.4 |
| `combo_vol_cal_m_hl12` | 2.4 / 0.8 | `combo_cov_main` | 7.7 / 5.6 |
| `combo_vol_cal_rm2006` | 2.5 / 0.8 | `combo_cov_simple` | 3.5 / 1.6 |

- All 16 registered combos together take about 90 s.
- The whole `analysis --forecast` run (16 registered combos plus about 80
  in-memory variants, all tables, portfolio scoring) took 4 min 39 s.
- Each component is read once per process. Each December's training targets
  are computed once (about 0.1 s) and cached in the panel's shared cache.

## Tests

`tests/models/test_combo.py`: 20 tests (19 synthetic plus 1 `realdata`), about
17 s. Together with `tests/test_contract.py`: 24 passed.

- **Setup.** The tests use the synthetic panel. Cheap synthetic components are
  run through the harness into a pytest temporary results directory, never the
  real one:
  - two last-N-month window vols in one family and a long-run vol in another;
  - a look-ahead "oracle";
  - a known-scale truth component;
  - two correlation estimators;
  - two deliberately incomplete components (one stops in 2005, one starts in
    1999).
- **Valid output.** Valid harness output for six vol configurations at all
  origins (step 6), including the early monthly-only origin and asset subsets
  in any order. Equal weight equals the mean of variances, and the
  hierarchical mean is computed by hand.
- **Walk-forward.**
  - Training cells use only windows ending by E and equal the project target
    exactly. They are unchanged when data after E are altered.
  - A component that is useless before 2004-12 and perfect after gets exactly
    the shrinkage floor (1−λ)/g (w* = 0) at every E up to 2004-12, and gains
    weight only as its perfect windows end. Its inverse-QLIKE rank is worst
    early.
- **Calibration** recovers a known scale: a component equal to 0.5 × the
  realized value gives c_hat = 2 exactly, c = 1 + λ, and c = 1 with no
  training data.
- **Missing data.** A missing component file, a missing forecast origin and a
  missing training origin each raise `MissingComponentError` naming the
  component and the date, for vol and for corr.
- **Correlation.** Combinations with block shrinkage 0, 0.2 and 1 stay
  symmetric, unit-diagonal and positive definite through the harness. Weighted
  combination is exact. The block target is positive definite and
  block-constant on random matrices.
- **Determinism.** Cold equals warm, bit for bit, in any call order (vol and
  corr). There is no look-ahead: altering the data after t leaves the forecast
  at t unchanged.
- **Covariance.** The composite covariance is positive definite with the
  combo's vols.
- **Registration** stays within budget (8 vol / 4 corr / 4 cov). Default
  member lists exclude every combo and `ref_corr_zero`.
- **`realdata`** (runs when `results/dev` and `VOLMODEL_MAX_DATE` are present):
  every registered combo at 1930, 1990 and 2014 against the stored dev
  forecasts.

## Caveats

- **Order of runs.** Combos must be run after their components. If a component
  is being re-run concurrently, a combo may read its old file. Run the combo
  family last, as `analysis --forecast` does. A missing component fails loudly;
  a stale one cannot be detected.
- **The CLI `--synthetic` flag.** The model is not told which results
  directory the harness is writing to: it reads `volmodel.paths.RESULTS`, or
  `results_dir` if set on the instance. `python -m volmodel.eval.run forecast
  --synthetic` would therefore read the components from `results/dev`, not
  `results/dev/synthetic`. The tests set `results_dir` explicitly. See shared
  code below.
- **Changing membership.** `combo_vol_ew`, `combo_vol_hier(_trim)` and
  `combo_corr_ew/hier` take their members from the registry at run time. Adding
  or removing a model in another family changes them. That is intended ("all
  models"), but it means the scores are tied to the zoo as registered at run
  time: 62 vol, 22 corr + 3 cov.
- **Selection after evaluation.**
  - `combo_cov_main` was registered after the vol and corr evaluation (a choice
    among registered pieces).
  - `combo_cov_simple` was re-paired after it.
  - The other 14 combos were fixed before any combo was scored.
  - The leaders listed in the tables come from the brief.
- **Statistical power.** Almost nothing here is significant against the best
  single models. The protocol's power analysis says a true 15–20% difference
  would usually go undetected. The MCS keeps everything. The recommendation
  rests as much on robustness and simplicity as on score.
- **Early calibration is unreliable.** Every learned quantity estimated before
  about 1935 rests on 1922–28 only, and calibration from that sample is
  actively harmful.
- **Expect movement after the data correction.** The investment-grade credit
  history shifts by 5–11% before 2002. The combos re-estimate everything from
  the view and need no change.

## Shared code (changes I need; I did not edit it)

1. **`eval/harness.run_model`.** It should pass its `results_dir` to the model,
   for example by setting `model.results_dir = results_dir` when the attribute
   exists and is None. Then combos run with `--synthetic` or an explicit
   results directory would read their components from the same place they
   write.
2. **`eval/run.py forecast --all`.** It should run family `combo` after all
   other families, not in parallel with them.
3. **Cache key in `Panel._shared`.** I store per-December training data in the
   panel's shared cache dict (`view._shared`) under a key that includes a
   digest of the data. If `Panel` gets a public cache hook, I would use it.
