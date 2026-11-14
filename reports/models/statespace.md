# statespace: latent log-volatility models with explicit measurement noise

Package `src/volmodel/models/statespace/` (`data.py` observations, `kalman.py`
filters, `grid.py` exact filter, `models.py` models, `__init__.py`
registrations). Tests `tests/models/test_statespace.py`. All numbers below are
development runs with `VOLMODEL_MAX_DATE=2014-12-31` (targets to 2013-12),
every month-end origin, stored in `results/dev/forecasts/vol/statespace_*`.

## Idea

Every variance measurement in this project is a noisy reading of a persistent
latent quantity. Let `h_{i,t}` be the log of asset i's variance in calendar
month t (so `E[r_{i,t}^2 | h] = exp(h)`, with `r` the monthly log excess
return). The 12-month target is the sum of the next 12 monthly variances, so

    V_{i,t} = s * sum_{k=1..12} E[exp(h_{i,t+k}) | F_t]
            = s * sum_k exp(m_k + v_k / 2)

with `(m_k, v_k)` the k-step predictive mean and variance of `h` from the
filter. The `v_k / 2` term is the lognormal retransformation: QLIKE rewards
the conditional mean of the variance, not `exp` of the mean log. `s` is a
level calibration (below).

### Measurements

* **y1 (every month, every era):** `y1 = log(r^2 + 0.005 * mbar) - KAPPA`, `mbar` = mean `r^2`
  over the previous 36 months. For normal returns `log z^2` is log chi-square(1):
  mean -1.27, variance 4.93, a long left tail (near-zero months). The small
  offset (half a percent of recent variance, the inlier fix of Fuller 1996 and
  Breidt & Carriquiry 1996) cuts the variance to 3.24 and removes the
  `log 0` tail. KAPPA = E log(z^2 + 0.005) = -1.098, computed by quadrature.
  Measurement equation `y1 = h + e1`, `Var e1 = H1` (estimated; it comes
  out 2.8-3.2, close to the normal-theory 3.24).
* **y2 (months with daily data on at least 80% of the month's rows and at least 15 days):**
  `y2 = log sum_d r_d^2` (1-day realized variance). Its level relative to the 21-day scale is unknown
  and changes at splices, so `y2 = h + c + e2`, where the offset `c` is a state:
  prior N(0, 0.4^2) when daily data start, a random walk with
  standard deviation 0.02 a month, and 0.4^2 of fresh variance whenever the
  data tier changes inside the daily era (`view.tiers`). The filter learns
  `c` from the gap between y2 and y1, so the daily data set the timing of moves
  and y1 sets the level. This replaces an ad hoc variance-ratio correction.
  `H2` is estimated.

### Latent dynamics (state per asset: `[mu, x_1, (x_2), (c)]`)

    h_t = mu + x_{1,t} (+ x_{2,t})        x_{k,t} = phi_k x_{k,t-1} + eta_{k,t},  eta ~ N(0, sigma_k^2)

`mu` is a constant state with a hierarchical prior `N(mubar, tau^2)`.
`mubar` and `tau` are estimated by marginal likelihood (empirical Bayes) in the
pooled models, so an asset with little history is shrunk toward the
cross-section. `x_k` starts from its stationary distribution. The prior is the
same for every asset, and no "related asset" prior is used (see the factor
model).

### Estimation and updating

* The parameters are estimated by Gaussian quasi-maximum likelihood (Kalman
  filter, univariate treatment of the observations). They use every month of
  every asset in `view.asof(E)`, where `E` is the last December (every 5th
  December for the factor models). L-BFGS-B uses a forward-difference gradient
  computed in one batched filter pass. The fixed start values do not depend on
  earlier fits, so a cold call equals a warm one. Fits are cached by a SHA-1
  of `view.asof(E)`.
* The state is updated monthly from `E` to the origin.
* **Level calibration `s`:** `s = mean(r^2 / E_{t-1} exp(h_t))` over the fit sample, pooled over
  assets (each asset's first 36 months excluded). This is the QLIKE-optimal
  constant for the model's own one-step forecasts and equals 1 under a correct
  model. It is the estimated form of the log-chi-square mean correction:
  fat tails given `h` move `E log eps^2` away from KAPPA, and the
  quasi-likelihood filter variances are approximate. Estimates are
  1.01-1.09 for the AR(1) models and 0.92-0.94 for `2c_daily`, where the fast
  component's variance enters `E exp(h)`.

## The eight registered models

| model | dynamics | measurements | pooling | filter |
|---|---|---|---|---|
| `statespace_ar1_asset` | AR(1) | y1 | phi, sigma, H1 per asset (>= 120 months, else start values); vague mu prior N(log(0.15^2/12), 1.5^2) | Kalman |
| `statespace_ar1` | AR(1) | y1 | phi, sigma, H1, mubar, tau pooled | Kalman |
| `statespace_ar1_grid` | AR(1) | r itself, exact `N(0, e^h)` likelihood | `statespace_ar1`'s parameters and each asset's posterior mean of mu at E | point-mass grid (241 points, +-5 sd) |
| `statespace_2c` | slow + fast AR(1) | y1 | pooled | Kalman |
| `statespace_ar1_daily` | AR(1) | y1 + y2 with offset c | pooled | Kalman |
| `statespace_2c_daily` | slow + fast AR(1) | y1 + y2 with offset c | pooled | Kalman |
| `statespace_factor` | `h_i = mu_i + lambda_g(i) f + x_i`, f AR(1) with unit variance, x_i AR(1) | y1, with measurement noise correlated across assets | phi_f, 4 group loadings (universe groups equity/rates/credit/real), phi_x, sigma_x, H1, mubar, tau | joint Kalman (37 states) |
| `statespace_factor_daily` | same | y1 + y2 with offsets | same, plus H2 | joint Kalman |

Factor-model measurement noise: `Cov(e1_i, e1_j) = H1 g(rho_ij)` and
`Cov(e2_i, e2_j) = H2 rho_ij^2`. Here `rho` is the pairwise monthly return
correlation at E, repaired to positive definite, and `g` is the exact
correlation of `log(z_i^2 + d)` for bivariate normals (tabulated, about 0.75 at
rho = 0.95). Without this term a large common return month, a common
chi-square draw, would be read as a common volatility move.

### Fixed hyperparameters and why

| value | reason |
|---|---|
| inlier offset 0.005 x trailing mean | the smallest round value that removes the log-0 tail (8% of chi-square(1) draws lie below 0.01; at 0.005 the mean moves by 0.17 and the variance falls from 4.93 to 3.24); its level effect is absorbed by `s` |
| offset prior and splice reset sd 0.4 | horizon_evidence: 1-day/21-day vol ratios run from 0.69 to 1.20 (log variance -0.74 to +0.36) and splice jumps reach about 0.5 in vol |
| offset random walk 0.02 a month | a drift of 0.3 in log variance over about 20 years, the scale of slow convention changes |
| y2 rule: at least 80% of rows and at least 15 days | rejects partial first months and gappy months |
| calibration burn-in 36 months | the eligibility rule |
| re-estimation yearly (factor: every 5 years) | the protocol maximum; the factor fit costs 3-16 s, so yearly refits would break the 10-minute budget |
| grid 241 points over +-5 stationary sd | spacing 0.04 sd, about 6 points per transition sd at phi = 0.98 |
| factor loading groups | `universe.Asset.group`, fixed in advance |

Nothing was chosen by looking at backtest scores. Every choice above was fixed
before the first scoreboard.

## Estimates

Pooled fits (Kalman, three fit dates). Half-life = ln 0.5 / ln phi in months.
SNR = monthly state-innovation variance / y1 noise variance.

| model, E | phi (half-life) | sigma_eta | stationary sd of h | H1 (sd) | H2 (sd) | SNR | s |
|---|---|---|---|---|---|---|---|
| ar1 1950 | 0.976 (28 m) | 0.22 | 1.00 | 3.21 (1.79) | - | 0.015 | 1.085 |
| ar1 1980 | 0.975 (27 m) | 0.25 | 1.12 | 3.13 (1.77) | - | 0.020 | 1.055 |
| ar1 2013 | 0.982 (38 m) | 0.20 | 1.04 | 3.12 (1.77) | - | 0.012 | 1.083 |
| ar1_daily 1950 | 0.947 (13 m) | 0.33 | 1.04 | 3.06 | 0.34 (0.58) | 0.037 | 1.063 |
| ar1_daily 2013 | 0.940 (11 m) | 0.38 | 1.10 | 2.89 | 0.22 (0.47) | 0.049 | 1.006 |
| 2c_daily 1950 | slow 0.983 (40 m), fast 0.37 (0.7 m) | 0.18 / 0.63 | 0.98 (slow) | 2.83 | 0.072 (0.27) | 0.012 | 0.922 |
| 2c_daily 2013 | slow 0.982 (38 m), fast 0.41 (0.8 m) | 0.20 / 0.51 | 1.05 (slow) | 2.82 | 0.087 (0.29) | 0.014 | 0.925 |
| 2c 1980 / 2013 | slow 0.994 (115 m) / 0.993 (99 m); fast 0.90 (bound, 6.6 m) | 0.12 / 0.31; 0.12 / 0.23 | 1.06 / 1.00 | 3.06 | - | 0.004 | 1.02-1.06 |
| factor 2010 | f 0.953 (14 m); x 0.998 (bound) | x 0.07 | x 1.07 | 3.14 | - | - | 1.095 |
| factor_daily 2010 | f 0.82 (3.4 m); x 0.987 (53 m) | x 0.15 | x 0.91 | 3.00 | 0.25 | - | 1.039 |

Readings:
* **Signal-to-noise.** With monthly data alone the SNR is 0.012-0.020: one
  month's `log r^2` carries almost no information. The steady-state Kalman
  gain is about 0.09 (2013 pooled AR(1)). Each past month's weight in the
  filtered state shrinks by phi(1 - K), about 0.89 a month, so recent data
  have a half-life of about 6 months. The forecast then reverts to `mu` with
  the 3-year half-life of the state.
* **Persistence.** On monthly data the latent log variance has a half-life of
  about 2.5-3 years. With daily data the single AR(1) is faster (11-13 months)
  because it also has to fit month-to-month moves. The two-component model
  separates these into a slow component (half-life 3-3.5 years, stationary sd
  about 1 in log variance) and a very fast one (phi about 0.4) that absorbs
  one-month bursts. The fast component is identified only with daily data.
  On monthly data alone it is not identified: the 1950 fit sent it to phi =
  0.12 with H1 collapsing from 3.2 to 1.0 (it absorbed measurement noise,
  s = 0.37), and the 1980 and 2013 fits pushed phi_fast to its 0.9 bound.
* **Per-asset (unpooled) persistence**, at 2013, ranges from 0.931 (intl_dev) to 0.998 (tips, at
  the bound). The spread is mostly noise: TIPS has 17 years of data.
* **Learnt offsets `c`** (ar1_daily, log 1-day/monthly variance ratio). At 1990:
  us_small -1.30, intl_dev -0.75, reits -0.74, ust_int -0.68, ust_long -0.51,
  hy_credit -0.50, us_large -0.33 (smoothed fund NAVs, yield
  reconstructions, foreign closes). At 2013: -0.04 to +0.10 for most assets,
  hy_credit -0.63, gold -0.23, reits -0.19. The posterior sd is 0.18-0.19
  once learnt. The hy_credit value at 2013 is a known limitation: the 2007
  VWEHX-to-HYG splice keeps tier A, so it is invisible in `view.tiers` and the
  offset only drifts toward the new level at 0.02 a month.

## Results (all monthly origins to 2013-12; references from `results/dev`)

QLIKE (lower is better), on the origins common to all models.

**Overall**

| model | daily subset (1051 origins) | all (1104 origins) |
|---|---|---|
| statespace_2c_daily | **0.3583** | **0.3928** |
| statespace_ar1_daily | 0.4019 | 0.4152 |
| statespace_2c | 0.4155 | 0.4368 |
| statespace_factor | 0.4200 | 0.4495 |
| statespace_factor_daily | 0.4209 | 0.4550 |
| statespace_ar1_grid | 0.4316 | 0.4220 |
| statespace_ar1 | 0.4378 | 0.4396 |
| statespace_ar1_asset | 0.4523 | 0.4482 |
| ref_last12m | 0.5048 | 0.5795 |
| ref_last60m | 0.5198 | 0.5572 |
| ref_blend | 0.5673 | 0.5370 |
| ref_longrun | 0.8367 | 0.6988 |

**Skill against the best reference** (Fieller-EWC 90% CI, DM test with EWC kernel)

| model | daily: skill vs ref_last12m [CI], p | all: skill vs ref_blend [CI], p |
|---|---|---|
| statespace_2c_daily | 29.0% [-3.1, 48.9], p = 0.12 | 26.8% [10.7, 37.1], p = 0.022 |
| statespace_ar1_daily | 20.4% [-28.6, 52.3], p = 0.41 | 22.7% [7.6, 32.8], p = 0.031 |
| statespace_ar1_grid | 14.5% [-24.9, 43.9], p = 0.47 | 21.4% [13.2, 28.2], p = 0.002 |
| statespace_2c | 17.7% [-16.9, 40.0], p = 0.32 | 18.7% [5.8, 27.4], p = 0.034 |
| statespace_ar1 | 13.3% [-29.9, 42.6], p = 0.53 | 18.1% [6.2, 26.5], p = 0.028 |
| statespace_ar1_asset | 10.4% [-38.3, 45.6], p = 0.66 | 16.5% [1.6, 27.5], p = 0.076 |
| statespace_factor | 16.8% [-33.0, 49.7], p = 0.49 | 16.3% [-1.5, 29.7], p = 0.12 |
| statespace_factor_daily | 16.6% [-30.4, 49.8], p = 0.48 | 15.3% [-0.1, 27.0], p = 0.10 |

On the daily subset the best reference is `ref_last12m`. It wins 1926-1945
by a wide margin, and that era dominates the variance of the loss
differential, so the intervals are wide and none is significant. On `all` the
best reference is `ref_blend`, and every pooled single-factor model is
significantly better at 5%.

**By era (daily subset)**

| model | 1926-45 | 1946-79 | 1980-99 | 2000-13 |
|---|---|---|---|---|
| statespace_2c_daily | 0.545 | 0.352 | 0.236 | 0.288 |
| statespace_ar1_daily | 0.723 | 0.360 | 0.243 | 0.282 |
| statespace_2c | 0.637 | 0.356 | 0.251 | 0.485 |
| statespace_factor | 0.745 | 0.323 | 0.266 | 0.421 |
| statespace_factor_daily | 0.743 | 0.382 | 0.236 | 0.328 |
| statespace_ar1_grid | 0.742 | 0.352 | 0.224 | 0.488 |
| statespace_ar1 | 0.754 | 0.350 | 0.245 | 0.485 |
| statespace_ar1_asset | 0.836 | 0.327 | 0.256 | 0.501 |
| ref_last12m | **0.418** | 0.599 | 0.313 | 0.673 |
| ref_last60m | 0.792 | 0.410 | 0.280 | 0.749 |
| ref_blend | 0.958 | 0.471 | 0.313 | 0.618 |
| ref_longrun | 1.475 | 0.720 | 0.576 | 0.600 |
| origins | 235 | 408 | 240 | 168 |

**By era (all subset)**: 2c_daily 0.527 / 0.426 / 0.249 / 0.288. ar1_grid
0.562 / 0.406 / 0.236 / 0.488. ar1 0.592 / 0.423 / 0.255 / 0.485.
ref_blend 0.729 / 0.515 / 0.288 / 0.618. ref_last12m 0.555 / 0.701 / 0.336 /
0.673. (Same era boundaries; the first era starts in 1922.)

**By asset (daily subset)**

| model | us_large | us_small | intl_dev | em | ust_int | ust_long | tips | ig_credit | hy_credit | reits | gold | commod. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2c_daily | 0.287 | 0.328 | 0.256 | 0.338 | 0.396 | 0.339 | 0.320 | 0.242 | 0.421 | 0.264 | 0.311 | 0.222 |
| ar1_daily | 0.344 | 0.383 | 0.260 | 0.291 | 0.347 | 0.322 | 0.257 | 0.248 | 0.436 | 0.274 | 0.337 | 0.222 |
| ar1 | 0.413 | 0.359 | 0.362 | 0.343 | 0.355 | 0.345 | 0.346 | 0.374 | 0.869 | 0.453 | 0.289 | 0.286 |
| ar1_grid | 0.387 | 0.330 | 0.350 | 0.344 | 0.390 | 0.440 | 0.352 | 0.457 | 0.739 | 0.452 | 0.286 | 0.250 |
| factor_daily | 0.317 | 0.400 | 0.289 | 0.317 | 0.434 | 0.388 | 0.379 | 0.303 | 0.464 | 0.285 | 0.306 | 0.265 |
| ref_last12m | 0.390 | 0.402 | 0.517 | 0.473 | 0.712 | 0.678 | 0.508 | 0.616 | 1.062 | 0.592 | 0.420 | 0.406 |
| ref_blend | 0.470 | 0.414 | 0.380 | 0.349 | 0.517 | 0.645 | 0.562 | 0.466 | 1.070 | 0.660 | 0.374 | 0.308 |

The full by-asset tables for all subsets and models are reproducible with the
commands at the end.

### What each ingredient bought (paired, same origins; skill of A over B, Fieller 90% CI, DM p)

| comparison | daily subset | all |
|---|---|---|
| pooling: ar1 vs ar1_asset | +3.2% [-8.7, 8.9], p = 0.52 | +1.9% [-4.7, 7.6], p = 0.60 |
| exact filter: ar1_grid vs ar1 | +1.4% [-4.1, 9.7], p = 0.69 | +4.0% [-1.8, 11.2], p = 0.26 |
| daily obs, AR(1): ar1_daily vs ar1 | +8.2% [-8.9, 24.5], p = 0.39 | +5.6% [-8.7, 17.1], p = 0.48 |
| daily obs, 2 comp.: 2c_daily vs 2c | +13.8% [0.7, 24.5], p = 0.087 | +10.1% [-1.2, 18.8], p = 0.13 |
| 2 components, monthly: 2c vs ar1 | +5.1% [-5.9, 11.1], p = 0.34 | +0.6% [-3.2, 3.9], p = 0.76 |
| 2 components, daily: 2c_daily vs ar1_daily | +10.9% [-10.1, 22.0], p = 0.30 | +5.4% [-3.1, 12.3], p = 0.27 |
| factor, monthly: factor vs ar1 | +4.1% [-8.8, 17.6], p = 0.57 | -2.2% [-13.2, 8.6], p = 0.72 |
| factor, daily: factor_daily vs ar1_daily | -4.7% [-13.0, 5.5], p = 0.38 | -9.6% [-18.0, 0.3], p = 0.11 |

* **The state-space structure is most of the gain.** Even the unpooled
  monthly-only AR(1) beats every reference on `all` (16.5%). A filter with
  the right noise model and mean reversion is what the blends and windows
  approximate.
* **The daily measurement** is the largest single addition once a fast
  component can absorb one-month bursts. 2000-2013 QLIKE falls from 0.485 to
  0.288. The 1946-99 eras gain little, and 1926-45 is mixed.
* **Pooling** helps a little overall. It helps clearly where the
  per-asset fit is poorly identified (reits 0.589 to 0.453; tips 0.416 to
  0.346), but it is not significant.
* **The exact grid filter** is slightly better than the Gaussian Kalman filter
  on log r^2 with the same parameters (4% on `all`, not significant). The
  Gaussian approximation with the log-chi-square mean and variance correction
  and the `s` calibration is adequate; the remaining gap is in the
  monthly-proxy eras.
* **The common factor did not help.** On monthly data the factor
  picked up the slow rates-vol cycle (loadings rates 0.8-1.0, equity 0.5-0.6)
  and left the idiosyncratic part near a unit root. With daily data it became a
  fast common burst factor (phi_f 0.5-0.8). Neither improved on the univariate
  models, and both did worst for short-history assets (below).

### Short-history assets (asset-origins by months of monthly history; QLIKE, all subset, cells common to all models)

| history | ar1_asset | ar1 | ar1_grid | 2c | ar1_daily | 2c_daily | factor | factor_daily | ref_blend | ref_last12m |
|---|---|---|---|---|---|---|---|---|---|---|
| 36-95 months (660 cells) | **0.391** | 0.404 | 0.428 | 0.413 | 0.477 | 0.454 | 0.561 | 0.634 | 0.557 | 0.501 |
| 96-239 months (1548) | 0.423 | 0.410 | 0.408 | 0.427 | 0.400 | 0.401 | 0.413 | 0.436 | 0.482 | 0.649 |
| 240+ months (6001) | 0.433 | 0.420 | 0.403 | 0.415 | 0.357 | **0.340** | 0.395 | 0.373 | 0.520 | 0.559 |

* In their first 3-8 years, the monthly-only state-space models beat the best reference by about
  25% (panel DM vs ref_blend: ar1_asset p = 0.047; others p = 0.13-0.22).
* The daily-data models lose part of that edge early on. The offset `c`
  starts at its prior (0 +- 0.4), and for smoothed sources (us_small 1929-34,
  hy_credit, early reits) the true offset is -0.5 to -1.3. Until it is learnt,
  daily RV pulls `h` down, so forecasts are too low exactly when the asset is
  young. us_small's first years, the 1929-33 crash, score 1.45 (ar1_daily)
  against 0.87 (ar1) and 0.45 (ref_last12m).
* The factor models are worst here. With only 3-8 years of data, the
  factor-relative level `mu_i` is driven by whatever the factor was doing,
  and ig_credit (0.64 to 1.27) and us_small (to 2.3) are badly placed. The
  "related-asset prior" works better as plain empirical-Bayes shrinkage plus
  the asset's own data.

### Where the family loses

* **1926-1945 on the daily subset**, against `ref_last12m` (0.418 vs 0.545 for
  2c_daily and 0.72-0.84 for the others). The Depression's vol changes were
  larger and faster than a pooled half-life of 2.5-3 years allows, and the
  long-run anchor pulls forecasts back too soon. The fast component of the
  daily model recovers part of this.
* **hy_credit** before the daily data (monthly-only models 0.74-0.87),
  and its offset after the 2007 splice (above).
* **ust_int/ust_long** in the daily models are slightly worse than in the monthly
  ones. Rates daily data before about 1990 are yield reconstructions with
  21d/1d ratios up to 2.7 that drift within a tier.
* **log_err** is +0.09 to +0.21 for these models, against about +0.03 for a perfect forecast:
  the forecasts sit above the median outcome. This is what QLIKE asks for under right-skewed
  realized variance; on `se_var_scaled`, the daily models are also best
  (factor_daily 3.09 and ar1_daily 3.29 vs ref_blend 3.48 on `all`).

## Variants tried and dropped

None were scored and dropped. The changes during development were made before
any scoreboard and only for cost: a sequential-update filter and
re-estimation restricted to assets with data, which give identical
likelihoods, and factor refits every 5 years instead of yearly. The inlier
offset 0.005 was chosen over 0.01 on the moment calculation above, not on
scores. The factor models are registered, not dropped, so their negative
result is on record.

## Run time

One core, `OMP_NUM_THREADS=1`. A full monthly run (1,104 origins to 2014,
with every fit) took 40-236 s at low machine load: ar1 82 s, ar1_grid 40 s
(reusing ar1's fits in the same process; standalone it adds ar1's fits, about 80 s), 2c 131 s,
ar1_asset 135 s, ar1_daily 124 s, 2c_daily 236 s, factor 119 s,
factor_daily 193 s. Under heavy machine load (load average 15-17 on 4
cores), the step-6 runs (all fits, 186 origins) took 2-12 minutes. Fits cost 0.4-3.5 s (Kalman) and 3-16 s
(factor) at low load. The per-origin update is about 0.07 s.

## Tests (`tests/models/test_statespace.py`, 31 tests, about 60 s)

Every model is checked at an early origin (monthly data only, below the
pooled-fit threshold, so the start-value path is used), a middle origin and a
late origin. Cold equals warm, in any order. Altering data, tiers and daily
returns after `t` leaves the forecast at `t` unchanged. The estimation-date
rule, the measurement constant KAPPA, the daily and splice flags, and the
batched univariate Kalman filter are checked against a textbook
matrix filter. The multivariate filter gives the same likelihood. The grid
filter is checked for level calibration, and the pooled fit for persistence
on synthetic data. Optimizer iterations are capped at 4 by monkeypatching in
the property tests; one test runs a converged fit.

## Caveats

* QML standard errors are not computed: the pooled likelihood treats assets as
  independent, which is a composite likelihood.
* The offset `c` is learnt slowly by design. Splices that keep their tier are
  invisible to the model.
* `s` is pooled over assets. Assets whose tails differ from the pool's are
  mis-levelled by a few percent.
* The 2c monthly-only model is not identified; its fast component sits at a
  bound or absorbs noise. It is kept as the planned comparison.
* Reproduce: `python -m volmodel.eval.run forecast --kind vol --model statespace_2c_daily`, then
  `python -m volmodel.eval.run score --kind vol --models statespace_2c_daily ref_blend ref_last12m ref_last60m ref_longrun --eras 1945-12-31 1979-12-31 1999-12-31`.
