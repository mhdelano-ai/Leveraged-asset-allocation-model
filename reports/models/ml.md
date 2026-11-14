# ML family: pooled machine-learning vol forecasts (`ml_*`)

Code: `src/assetforecast/models/ml/` (`features.py`, `learners.py`, `model.py`, `__init__.py`).
Tests: `tests/models/test_ml.py`. All numbers below are from the development
panel (`ASSETFORECAST_MAX_DATE=2014-12-31`), every month-end origin, scored on targets
through 2013-12 origins. Nothing after 2014 was loaded.

## Bottom line

* Pooled learners on a compact set of normalized features beat every reference
  benchmark on QLIKE overall: the best of them, the pooled **linear** gamma GLM
  (`ml_glm`), scores 0.367 on the `daily` subset against 0.505 for the best
  reference (`ref_last12m`), skill **27% (90% CI −0.5% to 43%, DM p = 0.10)**;
  on `all` 0.450 vs 0.537 (`ref_blend`), skill **16% (CI −4% to 33%, p = 0.18)**.
  From 1946 on, where the learner has 70+ years of training data, the skill is
  27% on `daily` (CI 3% to 39%, p = 0.08) and 21% on `all` (CI 4% to 32%, p = 0.06).
* **Nonlinearity adds nothing.** On identical features gradient boosting is
  19% worse than the linear model on `daily` (CI −39% to +10%; DM p = 0.24,
  panel-DM p = 0.07) and 1.5% worse on `all` (n.s.); the random forest is
  worse still (−46% on `daily`, panel-DM p = 0.02). After 1980 boosting is 13%
  worse (CI −20% to +1%). The monotone constraint does not matter (unconstrained
  boosting is 1.4% worse, n.s.). An equal-weight average of linear and boosted
  forecasts is no better than the linear one (−2.8%, n.s.).
* **Macro/implied features add nothing** either: −7% (`ml_glm_macro` vs
  `ml_glm`) and −9% (`ml_gbm_mono_macro` vs `ml_gbm_mono`) on `daily`, n.s.
* What the family really contributes is the **pooled, normalized design**
  (variance relative to the asset's own long-run level, trailing monthly- and
  daily-based variances, a global vol state), not the learner. A HAR-style
  pooled linear model is the right home for it.
* Where it loses: 1926–1945 (`ref_last12m` 0.418 vs `ml_glm` 0.525 on `daily`;
  trees much worse, 0.79–1.05), and on the MSE-of-variance loss
  (`se_var_scaled`, robust but crisis-dominated) every ML model is worse than the
  references (8.6 vs 5.7 for `ml_glm` vs `ref_last12m`, n.s.). The ML forecasts
  are biased **high**: realized/forecast variance ratio 0.79 (`ml_glm`) and 0.86
  (`ml_gbm_mono`) on `daily`, vs 1.00 for `ref_last12m`.

## Models (7 registered; budget 8)

| model | learner | features |
|---|---|---|
| `ml_glm` | gamma GLM, log link, L2 chosen by blocked CV | core |
| `ml_gbm_mono` | LightGBM (gamma), monotone in trailing variances | core |
| `ml_gbm` | LightGBM (gamma), unconstrained | core |
| `ml_rf` | random forest, Poisson-deviance splits | core |
| `ml_glm_macro` | as `ml_glm` | core + macro/implied |
| `ml_gbm_mono_macro` | as `ml_gbm_mono` | core + macro/implied |
| `ml_avg` | mean of `ml_glm` and `ml_gbm_mono` variance forecasts | core |

### Target and loss

For origin `s` and asset `a`, the target is the realized 12-month variance
`V_{a,s}` computed **from the view** with exactly the estimator and coverage
rules of `assetforecast.eval.targets` (overlapping 21-day sums, `n_rows/21`
annualization, daily coverage ≥ 90% of trading days and ≥ 10 returns in each
month; otherwise the monthly proxy `12·mean r_m²`). `realized_targets` is
tested to reproduce `compute_targets` exactly (relative difference 0).

Everything is relative to the asset's expanding long-run level
`LR_{a,s} = log(12 · mean_{u≤s} r_{a,u}²)` (monthly log excess returns, all
history to `s`). The learner predicts the ratio `y = V / exp(LR)`.

Rather than regress `log V` and retransform, every learner is fitted under a
loss minimized by the conditional **mean** of `y`: the gamma deviance, which is
exactly `2 × QLIKE(y, μ)` (GLM, boosting), or the Poisson deviance (forest).
This needs no smearing correction, uses the noisier monthly-proxy targets
consistently (QLIKE is robust to unbiased noise), and trains on the scoring
loss. The log-target alternative was run and dropped (below). The forecast is

    σ²_{a,t} = exp(LR_{a,t}) · clip(ŷ(x_{a,t}), 0.05, 20),    vol = sqrt(σ²).

### Features (`features.py`)

All trailing and causal (tested: the value for month `s` does not change when
later data are added). Variances are annualized 21-day-scale variances, in logs,
minus `LR`:

| feature | definition |
|---|---|
| `lr` | `LR_{a,s}` (the asset's long-run log variance) |
| `hist` | log years of monthly history (how reliable `LR` is) |
| `m3, m12, m60` | `log(12·mean r²)` over the last 3/12/60 months − LR (m60 needs ≥ 36) |
| `d1, d3, d12` | daily-based variance over the last 21/63/252 trading days − LR; NaN without daily data |
| `g_m3, g_m12` | cross-sectional mean of `m3`, `m12` over the assets eligible at `s` (global vol state) |
| `g_d3` | cross-sectional mean of `d3` over assets with daily data |
| `grp_rates, grp_credit, grp_real` | asset-group dummies (equity = baseline) |
| `x_vix` | log VIX (VXO-spliced before 1990; from 1986) |
| `x_baa_aaa` | log Moody's Baa−Aaa spread (1919) |
| `x_term` | long yield − short rate (1871) |
| `x_infl_vol` | log 36-month inflation volatility (1873) |
| `x_sahm` | Sahm unemployment gap (1949) |

**Daily data mapped to the 21-day scale.** Daily features use overlapping
**5-day** sums (which remove most of the pricing-convention effect;
`reports/data/horizon_evidence.md`), annualized by the actual number of daily
rows in the trailing 12 months (~295 before 1952, ~252 after), then rescaled by
a trailing estimate of the 21-day/5-day variance ratio
`log ρ = log[(mean s21²/21)/(mean s5²/5)]` on the same asset's daily data over
the last ≤ 60 months **but never before its last data-tier change**
(`view.tiers`), shrunk by `n/(n+36)` (n months), clipped to ±0.5, and set to 0
with < 12 months of same-tier data. A pre-splice scale is therefore never
carried across a tier change. Limitation: splices *within* a tier (em
VEIEX→EEM, ust_int H.15→GSW, reits VGSIX→VNQ) are not visible in the view; the
60-month window adapts to them within five years.

**Macro features**: the value at month-end `s` is the latest observation whose
`available_date ≤ s`, and NaN if the observation is older than 10 days (VIX) or
130 days (monthly series; macro releases lag up to ~3 months). The five series
were chosen a priori as the standard macro vol predictors (Schwert 1989; Paye
2012; Christiansen, Schmeling and Schrimpf 2012), not by trying candidates.

**Missing values.** Boosting and the forest use native NaN handling. The GLM
replaces a missing daily feature with its monthly analogue (d1, d3 → m3;
d12 → m12) plus a `has_daily` indicator, and a missing macro/global feature by
its **training-sample** mean plus a missing indicator. Standardization uses
training moments. Nothing is fitted on data after the training date.

### Training discipline (`model.py`)

* Re-estimation once a year: at origin `t` the model trained at `τ` = the last
  December month-end ≤ `t` is used. The training set is built from
  `view.asof(τ)` alone: all (origin `s`, asset) with `s + 12 months ≤ τ` (a
  12-month purge by construction), asset eligible at `s` (≥ 36 months).
* Fitted models are cached per (`τ`, data fingerprint), so a cold call and a
  warm call at the same origin give bit-identical forecasts (tested in any
  order; the full rerun of three models reproduced all 8,353 stored forecasts
  exactly).
* **Early-origin rule**: if the training origins span fewer than **20 distinct
  calendar years** (or fewer than 200 rows), the model returns `ref_blend`
  (geometric mean of long-run and 60-month variances). On the real panel the
  first trained model is December 1895 (us_large's monthly history starts
  1871; first targets from 1874), so the fallback only affects origins before
  1896, which are scored only in the `all` subset.
* Training size: 252 rows (1895) → 2,078 (1950) → 8,641 (2013); "years" here
  are calendar years of origins, and the independent information is much
  smaller (overlap and cross-asset correlation; perhaps 30–90 independent years).

### Hyperparameters and why

| learner | setting | reason |
|---|---|---|
| all | min leaf = rows equivalent to **5 independent years** (`5 × rows/years`) | a leaf must not be supportable by a single crisis (1–2 years in every asset) |
| GBM | depth 2 (4 leaves), 150 rounds, learning rate 0.05 | at most pairwise interactions; total shrinkage budget 7.5 – a deliberately small, round, a-priori choice, no early stopping |
| GBM | L2 = 10, row subsampling 0.7, 63 bins, seed 0, deterministic, 1 thread | standard strong regularization; determinism |
| GBM mono | non-decreasing in m3, m12, m60, d1, d3, d12, g_m3, g_m12, g_d3 | higher trailing variance must not lower the forecast; nothing imposed on `lr`, `hist`, groups, macro (economics ambiguous) |
| RF | 200 trees, Poisson splits, `max_features` 0.5, `max_samples` 0.5, seed 0 | Poisson deviance is consistent for the mean and less crisis-dominated than squared error |
| GLM | L2 strength from {1e-4, 1e-3, 1e-2, 0.1} by **blocked 5-fold CV**: contiguous calendar blocks of origins across ALL assets, training excludes origins within 12 months of the block on either side; gamma deviance | the only tuned parameter, chosen inside the model from the view |
| average | equal weights | estimated combination weights rarely beat equal ones with ~90 noisy years |

The CV picked 0.1 (the largest value) in 96 of 119 retrainings. A wider grid
(up to 10) was run as a check and was worse (`mlx_glm_widegrid`, below), so the
boundary is not binding in a harmful direction.

## Results

Scoring: `python -m assetforecast.eval.run score --kind vol --models ml_... ref_...`
plus `assetforecast.eval.inference` (`skill_score` Fieller-EWC 90% intervals,
`dm_test` EWC, `panel_dm_test` Driscoll–Kraay on per-asset losses). All on
origins common to the models compared.

### Overall (QLIKE; 1,051 `daily` origins 1926-06..2013-12, 1,104 `all` origins 1922-01..2013-12)

| model | daily qlike | daily se_var_scaled | daily log_err | all qlike | all se_var_scaled |
|---|---|---|---|---|---|
| **ml_glm** | **0.3669** | 8.59 | 0.169 | **0.4499** | 6.60 |
| ml_avg | 0.3771 | 7.75 | 0.155 | 0.4503 | 6.17 |
| ml_glm_macro | 0.3944 | 15.40 | 0.167 | 0.4696 | 9.30 |
| ml_gbm_mono | 0.4355 | 7.83 | 0.129 | 0.4569 | 4.24 |
| ml_gbm | 0.4417 | 8.04 | 0.135 | 0.4641 | 4.31 |
| ml_gbm_mono_macro | 0.4763 | 10.10 | 0.119 | 0.4790 | 5.16 |
| ml_rf | 0.5346 | 8.40 | 0.204 | 0.4940 | 4.78 |
| ref_last12m | 0.5048 | 5.75 | −0.019 | 0.5795 | 3.71 |
| ref_last60m | 0.5198 | 6.43 | 0.070 | 0.5572 | 3.77 |
| ref_blend | 0.5673 | 5.86 | 0.101 | 0.5370 | 3.48 |
| ref_longrun | 0.8367 | 6.82 | 0.131 | 0.6988 | 3.98 |

### Skill against the best reference (QLIKE)

| model | daily vs ref_last12m: skill % [90% CI], DM p | all vs ref_blend: skill % [90% CI], DM p |
|---|---|---|
| ml_glm | 27.3 [−0.5, 43.2], 0.10 | 16.2 [−4.4, 33.1], 0.18 |
| ml_avg | 25.3 [−5.8, 44.5], 0.15 | 16.1 [−4.5, 32.5], 0.18 |
| ml_glm_macro | 21.9 [−11.6, 42.1], 0.22 | 12.5 [−7.4, 30.7], 0.27 |
| ml_gbm_mono | 13.7 [−31.1, 45.4], 0.54 | 14.9 [3.5, 23.5], 0.046 |
| ml_gbm | 12.5 [−34.0, 44.7], 0.58 | 13.6 [1.8, 22.7], 0.07 |
| ml_gbm_mono_macro | 5.6 [−47.6, 45.1], 0.83 | 10.8 [−1.0, 22.2], 0.13 |
| ml_rf | −5.9 [−70.8, 36.7], 0.83 | 8.0 [−3.6, 16.8], 0.22 |

Sub-periods (best reference re-chosen within each period):

| period | subset | best ref | ml_glm | ml_gbm_mono | ml_avg |
|---|---|---|---|---|---|
| 1946–2013 | daily | ref_last60m | 27.2 [3.1, 38.6] p 0.08 | 24.6 [5.3, 34.8] p 0.06 | 27.7 [6.8, 37.9] p 0.06 |
| 1946–2013 | all | ref_blend | 21.1 [3.5, 31.9] p 0.06 | 16.6 [3.1, 25.2] p 0.06 | 21.0 [6.3, 30.2] p 0.04 |
| 1980–2013 | daily | ref_blend | 33.7 [−3.7, 46.8] p 0.11 | 24.9 [−6.5, 39.0] p 0.13 | 30.7 [−3.3, 44.0] p 0.11 |
| 1926–1945 | daily | ref_last12m | −25.6 [−73.9, 35.2] p 0.37 | −89.7 [−229, 102] p 0.32 | −38.4 [−107, 54] p 0.37 |

### By era (QLIKE; also the "learning curve")

| model | 1926–45 daily | 1946–79 daily | 1980–99 daily | 2000–13 daily | 1926–45 all | 1946–79 all | 1980–99 all | 2000–13 all |
|---|---|---|---|---|---|---|---|---|
| ml_glm | 0.525 | 0.353 | 0.226 | 0.382 | 0.675 | 0.444 | 0.238 | 0.382 |
| ml_avg | 0.578 | 0.335 | 0.233 | 0.405 | 0.676 | 0.431 | 0.244 | 0.405 |
| ml_glm_macro | 0.638 | 0.341 | **0.198** | 0.464 | 0.762 | **0.413** | **0.219** | 0.464 |
| ml_gbm_mono | 0.792 | 0.336 | 0.249 | 0.443 | 0.643 | 0.446 | 0.261 | 0.443 |
| ml_gbm | 0.833 | 0.340 | 0.247 | 0.420 | 0.679 | 0.451 | 0.259 | 0.420 |
| ml_gbm_mono_macro | 0.971 | **0.315** | 0.225 | 0.535 | 0.734 | 0.417 | 0.239 | 0.535 |
| ml_rf | 1.049 | 0.442 | 0.282 | 0.402 | 0.751 | 0.472 | 0.287 | 0.402 |
| ref_last12m | **0.418** | 0.599 | 0.313 | 0.673 | **0.555** | 0.701 | 0.336 | 0.673 |
| ref_last60m | 0.792 | 0.410 | 0.280 | 0.749 | 0.726 | 0.528 | 0.270 | 0.749 |
| ref_blend | 0.958 | 0.471 | 0.313 | 0.618 | 0.729 | 0.515 | 0.288 | 0.618 |
| ref_longrun | 1.475 | 0.720 | 0.576 | 0.600 | 0.896 | 0.725 | 0.488 | 0.600 |
| origins | 235 | 408 | 240 | 168 | 288 | 408 | 240 | 168 |

The learner that forecast 1926–45 was trained on 30–50 years of mostly
monthly-only, pre-1926 data (us_large's Shiller month-average prices, ust_long,
ig_credit) with few or no daily rows, and it then met the Depression. The trees
fare worst there: they cannot extrapolate beyond the training range and had
almost no rows with daily features to learn from (the 1929 and 1930 origins
alone cost `ml_gbm_mono` 4.2 and 1.3 QLIKE points more than `ml_glm`). From 1946
on the boosted and linear models are close (boosting even slightly ahead in
1946–79 on `daily`), and from 1980 the linear model is ahead again. With 70–90
training years the trees still do not overtake the linear model: the learning
curve does not point toward nonlinearity paying off with more data.

### By asset (QLIKE, `daily`)

| model | us_large | us_small | intl_dev | em | ust_int | ust_long | tips | ig_credit | hy_credit | reits | gold | commodities |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ml_glm | **0.322** | **0.299** | **0.259** | **0.296** | 0.362 | 0.337 | **0.278** | **0.309** | **0.634** | 0.410 | 0.331 | 0.244 |
| ml_avg | 0.331 | 0.300 | 0.281 | 0.320 | 0.375 | 0.348 | 0.289 | 0.349 | 0.681 | 0.403 | 0.326 | 0.234 |
| ml_glm_macro | 0.358 | 0.331 | 0.304 | 0.328 | **0.314** | **0.317** | 0.366 | 0.347 | 0.685 | 0.476 | **0.296** | 0.253 |
| ml_gbm_mono | 0.364 | 0.357 | 0.331 | 0.357 | 0.419 | 0.373 | 0.315 | 0.406 | 0.745 | 0.406 | 0.335 | 0.229 |
| ml_gbm | 0.374 | 0.358 | 0.317 | 0.336 | 0.421 | 0.368 | 0.308 | 0.388 | 0.703 | **0.391** | 0.331 | **0.227** |
| ml_gbm_mono_macro | 0.404 | 0.410 | 0.386 | 0.400 | 0.360 | 0.365 | 0.471 | 0.451 | 0.766 | 0.510 | 0.309 | 0.259 |
| ml_rf | 0.481 | 0.471 | 0.320 | 0.329 | 0.399 | 0.394 | 0.312 | 0.388 | 0.638 | 0.395 | 0.360 | 0.252 |
| ref_last12m | 0.390 | 0.402 | 0.517 | 0.473 | 0.712 | 0.678 | 0.508 | 0.616 | 1.062 | 0.592 | 0.420 | 0.406 |
| ref_last60m | 0.496 | 0.403 | 0.513 | 0.410 | 0.531 | 0.513 | 0.566 | 0.446 | 1.362 | 0.603 | 0.401 | 0.415 |
| ref_blend | 0.470 | 0.414 | 0.380 | 0.349 | 0.517 | 0.645 | 0.562 | 0.466 | 1.070 | 0.660 | 0.374 | 0.308 |
| ref_longrun | 0.662 | 0.588 | 0.298 | 0.305 | 0.646 | 1.171 | 0.584 | 0.712 | 0.927 | 0.837 | 0.499 | 0.362 |

`ml_glm` beats every reference on every asset; its margin is smallest on
intl_dev and em, where `ref_longrun` is close (their daily history is short and
calm relative to the long run). The trees do not: `ml_rf` and `ml_gbm*` lose to
`ref_last12m` on us_large and to `ref_longrun` on em. The `all` by-asset table (same ranking; ig_credit and
ust_long worse in the proxy eras: `ml_glm` 0.648 and 0.508) is in the scoreboard
output.

### Linear versus nonlinear on identical features

Skill of `a` over `b` (positive = `a` better), Fieller 90% CI, DM on the pooled
per-origin loss, and the Driscoll–Kraay panel DM on per-asset losses.

| a vs b | daily skill [CI] | DM p | panel p | all skill [CI] | DM p | panel p |
|---|---|---|---|---|---|---|
| ml_gbm_mono vs ml_glm | −18.7 [−39.1, 9.8] | 0.24 | 0.07 | −1.5 [−24.9, 15.3] | 0.89 | 0.34 |
| ml_gbm vs ml_glm | −20.4 [−41.3, 8.2] | 0.21 | 0.07 | −3.1 [−27.6, 14.8] | 0.79 | 0.31 |
| ml_rf vs ml_glm | −45.7 [−79.4, −5.6] | 0.07 | 0.02 | −9.8 [−36.7, 9.8] | 0.42 | 0.07 |
| ml_avg vs ml_glm | −2.8 [−8.6, 5.3] | 0.51 | 0.24 | −0.1 [−5.6, 4.4] | 0.98 | 0.67 |
| ml_gbm vs ml_gbm_mono | −1.4 [−4.9, 1.5] | 0.39 | 0.48 | −1.6 [−4.3, 1.4] | 0.36 | 0.88 |
| ml_glm_macro vs ml_glm | −7.5 [−19.1, 5.0] | 0.30 | 0.47 | −4.4 [−11.6, 4.5] | 0.37 | 0.49 |
| ml_gbm_mono_macro vs ml_gbm_mono | −9.4 [−18.8, 4.8] | 0.22 | 0.37 | −4.8 [−12.3, 5.2] | 0.38 | 0.46 |
| ml_gbm_mono_macro vs ml_glm_macro | −20.8 [−44.7, 13.3] | 0.27 | 0.11 | −2.0 [−24.4, 15.1] | 0.85 | 0.33 |

1946–2013 `daily`: `ml_gbm_mono` vs `ml_glm` −3.5% [−13.4, 8.6], p 0.59;
1980–2013: −13.3% [−19.7, 1.4], p 0.12.

**Verdict: no.** At this horizon, on this panel, nonlinearity and interactions
add nothing detectable over a well-specified pooled linear (log-link) model on
the same features, and the point estimates favour the linear model in every
comparison and every era except 1946–79, where boosting is ahead by about 5%.
The 95%-level evidence is one-sided: the forest is significantly worse
(panel p = 0.02), nothing nonlinear is significantly better.

### What the features say (importance, stability across 119 yearly retrainings)

Standardized GLM coefficients (`ml_glm`; log of the variance ratio per 1 sd):

| feature | 1946 | 1965 | 1980 | 2000 | 2013 |
|---|---|---|---|---|---|
| lr | −1.07 | −0.23 | −0.31 | −0.26 | −0.25 |
| m3 / m12 / m60 | 0.10 / −0.13 / −0.11 | 0.06 / 0.20 / 0.09 | 0.10 / 0.17 / 0.14 | 0.07 / 0.18 / 0.13 | 0.04 / 0.12 / 0.06 |
| d1 / d3 / d12 | 0.03 / −0.06 / 0.13 | 0.04 / 0.02 / 0.18 | 0.05 / 0.01 / 0.16 | 0.07 / 0.04 / 0.14 | 0.10 / 0.07 / 0.22 |
| g_d3 (global daily vol) | 0.36 | 0.29 | 0.23 | 0.17 | 0.21 |
| g_m3 / g_m12 | −0.00 / 0.33 | −0.00 / 0.02 | 0.01 / 0.00 | −0.01 / −0.00 | −0.00 / −0.08 |
| groups rates / credit / real | −0.51 / −0.86 / 0.26 | −0.14 / −0.18 / −0.05 | −0.03 / −0.07 / 0.05 | −0.04 / −0.10 / −0.03 | −0.04 / −0.06 / −0.02 |
| has_daily | 0.62 | −0.02 | 0.11 | 0.04 | 0.06 |

LightGBM gain shares (`ml_gbm_mono`), mean over 20-year blocks of retraining dates:

| feature | 1900–19 | 1920–39 | 1940–59 | 1960–79 | 1980–99 | 2000–13 |
|---|---|---|---|---|---|---|
| m12 | 0.00 | 0.01 | 0.10 | 0.41 | 0.46 | 0.45 |
| g_d3 | 0 | 0.20 | 0.37 | 0.21 | 0.10 | 0.10 |
| lr | 0.41 | 0.28 | 0.13 | 0.11 | 0.13 | 0.12 |
| m3 | 0.05 | 0.03 | 0.04 | 0.09 | 0.12 | 0.10 |
| m60 | 0.08 | 0.04 | 0.01 | 0.05 | 0.07 | 0.08 |
| hist | 0.46 | 0.32 | 0.09 | 0.06 | 0.06 | 0.05 |
| d1 + d3 + d12 | 0 | 0.08 | 0.14 | 0.05 | 0.04 | 0.07 |
| groups | 0 | 0.00 | 0.01 | 0.01 | 0.01 | 0.01 |

What they say:

1. **Persistence at the one-year scale**: the 12-month variances (m12 and the
   daily d12) carry most of the signal; the 1-to-3-month measures add a little.
   Daily-based and monthly-based measures of the same window substitute for each
   other (the GLM splits weight between m12 and d12; the trees use m12).
2. **Mean reversion toward a common level**: the negative `lr` coefficient
   (−0.25 sd) means high-long-run-vol assets are forecast below their own long
   run and low-vol ones above it: pooling shrinks the noisy, era-dependent
   asset long-run levels toward each other.
3. **A global vol state** (`g_d3`, the cross-asset average of recent daily vol)
   gets a large, stable positive weight: when all markets are turbulent, each
   asset's next-year vol is higher than its own history alone says.
4. **Asset groups** matter only while the data are dominated by a few assets and
   eras (1940s); from 1965 on their coefficients are small, i.e. the
   normalization works and the learner is not memorizing asset identity.
5. **Stability**: from 1960 on the ranking and size of the main effects (m12,
   d12, g_d3, lr, m3) are stable (coefficient of variation across retrainings
   0.3–0.6); the global monthly averages, groups, the daily 1- and 3-month
   features and all macro variables flip sign or wander (CV 1–2.4). Before 1946
   everything is unstable: the models are fitting a handful of assets and the
   pre-1926 month-average-price era (`hist` and `lr` dominate the trees, the
   `has_daily` indicator absorbs the level shift between monthly-only and daily
   eras).
6. **Macro**: the term spread (−) and Sahm gap (−) get consistent signs after
   1965 and the Baa–Aaa spread (+), but out of sample they hurt (macro variants
   lose 4–9%); VIX enters only from 1986, flips sign, and the boosted model
   barely uses it. Plausible effects; no forecasting gain over trailing realized
   variance at a 12-month horizon.

## Variants tried and dropped (not registered; QLIKE, same origins)

| variant | what | daily | all | vs parent (daily skill, p) |
|---|---|---|---|---|
| `mlx_ridge_log` | ridge on log y, same prep/CV, Duan smearing retransform | 0.3730 | 0.4554 | vs ml_glm −1.7% [−8.5, 3.1], p 0.59 |
| `mlx_gbm_l2log` | monotone LightGBM on log y (squared error), Duan smearing | 0.4716 | 0.4607 | vs ml_gbm_mono −8.3% [−18.4, 3.6], p 0.21 |
| `mlx_glm_widegrid` | GLM with CV grid {1e-3 … 10} | 0.4540 | 0.4644 | vs ml_glm −23.7% [−51, 11], p 0.23 (loses in 1926–45) |
| `mlx_glm_decay30` | GLM, training rows down-weighted with a 30-year half-life, alpha 0.1 | 0.3691 | 0.4439 | vs ml_glm −0.6% [−3.9, 3.4], p 0.79 (+1.3% on all) |

All four were run once, after the registered set was fixed, as checks on design
choices (log target vs mean-consistent loss; CV grid boundary; level drift). None
changes a conclusion; none was adopted. During development the registered models
were first run at `--step 6` (186 origins) with the same code; no
hyperparameter was changed after looking at any score.

## Overfitting risk: frank assessment

* **Effective sample**: 8,641 training rows by 2013, but 12-month targets
  overlapping by 11 months and cross-asset vol correlations of 0.5–0.8 leave
  perhaps 30–90 independent observations of the thing that matters (the
  vol regime), dominated by 1929–33, 1937, 1973–74, 1987, 2000–02 and 2008. Any
  learner with more than a few effective degrees of freedom fits these episodes.
* **What was done about it**: 5-year minimum leaves, depth-2 trees, a small fixed
  boosting budget, monotone constraints, no tuning on scores, a gamma/QLIKE
  objective that does not over-weight crises the way squared error does, and an
  honest, purged, embargoed walk-forward. Even so, the trees are worse than the
  linear model, and their deficit is concentrated where they must extrapolate
  (1929–30, 2007–08: the worst years of `ml_gbm_mono` vs `ml_glm`). That is the
  signature of fitting the training crises, not of missing structure.
* **Level bias**: all ML forecasts are too high on average (realized/forecast
  variance 0.79 for `ml_glm`, 0.86 for `ml_gbm_mono` on `daily`; by era for
  `ml_glm` 0.61, 0.82, 0.78, 1.01). The in-sample fit is unbiased by construction
  (gamma GLM with intercept), so this is non-stationarity: the pooled level of
  `V/exp(LR)` drifts as the expanding long-run level absorbs the 1930s and the
  1970s–80s. QLIKE tolerates over-forecasting, so the QLIKE ranking flatters the
  ML models relative to `se_var_scaled`, where every ML model is worse than the
  references (not significantly). A 30-year half-life on training rows did not fix
  it (ratio 0.81). Downstream users of these vols should know the forecasts lean
  high.
* **Data-era risk**: the pre-1926 training data (month-average prices, model
  bond returns) differ systematically from the daily era; the GLM absorbs this
  with `has_daily`, the trees with `hist`/`lr` splits. The 1926–45 losses show the
  cost.
* **Multiple testing within the family**: 7 registered models plus 4 checks. The
  `ml_glm` result is the a-priori control, not a winner picked from many.

## Run time

Per model, full monthly run to 2014 (1,116 origins, 119 yearly retrainings),
one core, measured while the machine was heavily shared (load 8–16 on 4 cores):
`ml_rf` 87 s CPU (124 s wall), `ml_glm_macro` 70 s CPU (101 s), `ml_gbm_mono`
39 s CPU (53 s); the whole family (7 models, `--jobs 2`) took 7.6 min of CPU.
Per-origin prediction ~30–50 ms (features rebuilt from the view each call);
training ~0.2 s (GBM) to ~1 s (RF, GLM with 5×4 CV fits) per year. Well inside
the 10-minute budget.

## Caveats

* Splices within a data tier are invisible to the daily scale correction (60-month
  adaptation window).
* Macro series are latest-vintage; the macro variants did not help anyway.
* The 1926–45 era is where any ML model is least reliable; the live forecast is
  made with ~140 calendar years of training rows, the regime in which the
  family scored best, but the level bias above applies.
* The synthetic-panel tests set `min_years = 10` (that panel spans 41 years).

## Reproduce

```
export ASSETFORECAST_MAX_DATE=2014-12-31 OMP_NUM_THREADS=1
python -m assetforecast.eval.run forecast --kind vol --model ml_glm ml_gbm_mono ml_gbm ml_rf ml_glm_macro ml_gbm_mono_macro ml_avg --jobs 2
python -m assetforecast.eval.run score --kind vol --models ml_glm ml_avg ml_gbm_mono ml_gbm ml_rf ml_glm_macro ml_gbm_mono_macro ref_blend ref_last12m ref_last60m ref_longrun --eras 1945-12-31 1979-12-31 1999-12-31
python -m assetforecast.eval.run score --kind vol --subset all --models ...
python -m pytest tests/models/test_ml.py
```
