# bench — the strong simple benchmarks

Package `src/assetforecast/models/bench/` (`core.py` building blocks, `vol.py`,
`corr.py`, `analysis.py` regenerates every table below), tests
`tests/models/test_bench.py`. All numbers: development panel
(`ASSETFORECAST_MAX_DATE=2014-12-31`), every month-end origin, targets = 21-day
realized covariance (version 3). Nothing after 2014 was looked at.

## Answers first

**Volatility.** The best bench vol model, `bench_vol_mr_est` (estimated
recent/long-run blend), has mean QLIKE 0.403 on the `daily` subset vs 0.505
for the best reference (`ref_last12m`): skill +20% (90% CI −15% to +42%,
DM p = 0.27). On `all` (adds monthly-proxy eras): 0.401 vs 0.537 for the best
reference there (`ref_blend`), skill +25% (CI +13% to +33%, DM p = 0.009). The
plain 1-year EWMA (`bench_vol_m_hl12`, monthly returns only) is within 2% of
it: +19% (CI −1% to +34%, p = 0.11) on `daily`, +16% (CI +2% to +28%, p = 0.06)
on `all`. Every bench vol model except the 10-year EWMA beats every reference
in point terms on both subsets.

**Correlation.** `bench_corr_d_hl36` (3-year EW correlation of 21-day sums)
has pooled MSE 0.0406 vs 0.0432 for the best reference
(`ref_corr_last120m`): skill +5.9% (CI −0.5% to +11.3%, p = 0.12). Under the
noise-robust covariance loss (each correlation model paired with the same
vols, `bench_vol_d_hl12`; `qlike_rel`) it is better by 0.071 (p = 0.014), and
the block-shrunk version by 0.108 (p = 0.064).

**What half-life at a 12-month horizon?** About **one year**, for both
monthly and daily inputs, and the curve is **flat from 6 to 36 months**
(none of 6/12/36 is distinguishable: +2% to +13% worse than 12m, p ≥ 0.16) and
then **rises steeply**: 5 years is 15–29% worse (p 0.04–0.09) and 10 years
26–51% worse (p 0.01–0.053). Pooled losses (QLIKE, `daily`): 6m 0.428 / 12m
0.409 / 36m 0.464 / 60m 0.527 / 120m 0.619 (monthly input). By group:
equity and real assets prefer 6–12 months, rates 36 months (flat 12–60),
credit is flat and inconclusive (its best half-life flips between subsets).
The slow decay the literature suggests does not appear here; what does help
is a *two-speed* forecast: a 3-month level blended with the all-history mean
(`mr_fixed`, `mr_est`) beats every single half-life, mostly in 2000–2013.
For correlations the optimum is longer: 36–60 months (MSE flat 36–120,
clearly worse at 6–12).

**Do daily inputs help once mapped to the 21-day scale?** For
**volatility: no, not detectably.** 21-day sums of daily data vs monthly
squared returns at the same half-life: skill between −0.7% and +1.9% at every
half-life, p ≥ 0.32 (pooled DM) and ≥ 0.40 (panel DM). They help equities
(+3% to +10%) and credit, and slightly hurt rates at long half-lives. The
more precise 1-day variance times a trailing 21d/1d ratio adds +7.6% over the
21-day sums at a 6-month half-life (p = 0.17), +1.2% at 12 months (p = 0.80),
−0.9% at 36 months — concentrated in credit (+21% to +34% at 6–12 months, the stale-NAV era
where the ratio correction matters most). The more data a half-life already
averages, the less precision buys. For **correlation: yes, significantly.**
21-day sums beat monthly co-moments at every half-life up to 60 months:
+16% at 6m, +11% at 12m, +4.6% at 36m (CI +1.9% to +7.6%, p = 0.013), +2.7%
at 60m (p = 0.07), +1.5% at 120m (n.s.); the covariance loss agrees.

## Models

Notation (details in the `core` docstring). For month `m` and asset `a`:

* `r_m` — calendar-month log excess return; `r_m^2` is one draw of the target's
  monthly variance.
* `xd_m = D_m · mean_{d∈m} s_d^2 / 21` — the target's own estimator on daily
  data: `s_d` is the 21-trading-day log excess return sum ending on day `d`
  (a missing day inside the daily span counts 0, as in the target), `D_m` the
  panel trading days in the month. Requires ≥ 10 valid sums ending in the month,
  every one of whose 21 days is on or after the asset's first daily return.
  Summed over 12 months it is `n_rows × per-day variance`, the target's
  annualization, correct under the 295-day (pre-1952) and 252-day calendars.
  Verified in the tests against `realized_kday_cov` on the same window.
* `x_m = xd_m` where it exists, else `r_m^2` ("daily where available").
* `x1_m = Σ_{d∈m} r_d^2` — 1-day realized variance (on the 1-DAY scale).
* `EW_h(y) = Σ_m w_m y_m / Σ_m w_m`, `w_m = 0.5^{age_m/h}`, age 0 = the origin
  month, missing months dropped and weights renormalized; `EW_∞` = equal
  weights over all history. Forecast annual variance = 12 × a monthly variance.

| model | forecast of annual variance |
|---|---|
| `bench_vol_m_hl12` / `_hl36` / `_hl120` | `12 EW_h(r^2)`, h = 12 / 36 / 120 months |
| `bench_vol_d_hl12` / `_hl36` | `12 EW_h(x)` |
| `bench_vol_d1r_hl12` | `12 ρ EW_12(x1)`, `ρ = EW_36(xd)/EW_36(x1)`, both on the asset's current tier segment only; falls back to `bench_vol_d_hl12` with < 12 segment months |
| `bench_vol_mr_fixed` | `w · 12EW_3(x) + (1−w) · 12EW_∞(x)`, `w = 0.51` |
| `bench_vol_mr_est` | `k · exp(a + b log 12EW_3(x) + (1−b) log 12EW_∞(x))`, `b ∈ [0,1]` |

`bench_vol_mr_est`: `(a, b)` by pooled OLS of `log(RV12_s / L_s)` on
`log(R_s / L_s)` over all assets and past origins `s` whose 12 months lie
before the last December-end ≤ origin (so parameters change once a year), with
≥ 36 months of history at `s`. `RV12_s = Σ_{m=s+1}^{s+12} x_m` — the target
estimator from the view (daily sums; monthly squared returns before daily
data), never the cached targets. `k = mean(exp(residual))` (Duan smearing) so
the forecast aims at `E[RV]`, which is what QLIKE rewards. Fewer than 120
training rows → `mr_fixed`. Estimated `b`: 0 in the 1880s (one asset),
0.47 by 1930, 0.68–0.72 from 1950 on; `k ≈ 1.4`, `a ≈ −0.1`.

| model | correlation |
|---|---|
| `bench_corr_m_hl36` / `_hl120` | pairwise uncentered EW_h of monthly co-moments `r_i r_j` / sqrt(`r_i^2` `r_j^2`) sums |
| `bench_corr_d_hl36` / `_hl120` | the same with 21-day-sum co-moments `D_m mean(s_i s_j)/21` in months where BOTH assets have ≥ 10 common valid sums, monthly co-moments otherwise |
| `bench_corr_shrink_block` | `δ F + (1−δ) R`, `R` = `bench_corr_d_hl36` before repair, `F` = block target (mean of `R` within each group pair: equity / rates / credit / real from `assetforecast.universe`), `δ` estimated |
| `bench_corr_blend` | `0.5 EW_12 + 0.5 EW_∞` of the 21-day-sum co-moments |

Each pair's variances in the denominator are measured on exactly the months
(and, in daily months, the sums) used for its co-moment, so every pairwise
estimate is a proper correlation in [−1, 1]. Shrinkage intensity
(Schäfer–Strimmer / Ledoit–Wolf for correlation matrices):
`δ = Σ_{i<j} Var(r_ij) / Σ_{i<j} (r_ij − f_ij)^2`, clipped to [0, 1], with
`Var(r_ij) = (1 − r_ij^2)^2 / n_eff,ij`, `n_eff = (Σw)^2/Σw^2` over the pair's
months; entries alone in their block (where `f_ij = r_ij`) are excluded.
Typical `δ`: 0.12–0.35 with 10–12 assets.

### Unequal histories and positive definiteness

Every estimate is **pairwise** (each asset, or pair, uses all the months it
has), then repaired: symmetrize, clip eigenvalues at 1e-3, rescale to unit
diagonal (the same repair as the reference models, so differences are in
estimation). Why pairwise + repair rather than a balanced sample or an EM /
Stambaugh (1997) monotone-missing estimator: the panel's starts run from 1871
to 1997, a balanced sample would discard most of the long histories, and with
exponential weights the long histories matter only for long half-lives; the
EM route assumes a stationary joint distribution across 120 years, which the
half-life results contradict. The repair is rarely needed: checked at the 90
December origins 1925–2014, the unrepaired pairwise matrix is not positive
definite at 0 of them for h = 36, 1 for h = 12 and 4 for h = 120. Pairs
with < 24 common months would get 0 (never triggered: the smallest pair
overlap at those origins is 36 months). Vol: an asset with only monthly data simply
uses `r^2` (the `_d` models reduce to the `_m` models there); the 1-day model
falls back to the 21-day-sum model. Every eligible asset therefore always gets
a finite forecast; a cross-sectional median guard exists but never fired at those origins.

### Why each hyperparameter has its value (all fixed before scoring)

* Half-life grid 6 / 12 / 36 / 60 / 120 months: round numbers spanning "recent
  half-year" to "a decade", as briefed. Registered: monthly 12/36/120 (the
  grid's middle and long end, where the literature points), 21-day sums 12/36,
  1-day×ratio at 12. The whole grid is scored as diagnostics below; the
  registered set was fixed before any score was seen.
* `MIN_SUMS = 10` per month: the target's own per-month coverage rule.
* Ratio half-life 36 months: three years of 21-day sums give a ratio with
  roughly 10% standard error, and a tier change resets it (the view's `tiers`
  are the only splice information it carries).
* Recent component `EW_3` (a quarter): the "current level" in the classic
  two-component (short/long) variance decomposition.
* `w = 0.51`: if the log-variance deviation is AR(1) with a half-life of
  6 months (monthly persistence ≈ 0.9, the order reported for monthly realized
  variance of US equities), the mean over the next 12 months of the expected
  deviation is `φ(1−φ^12)/(12(1−φ))` = 0.51 of today's. Correlation blend
  weight 0.5 by the same argument.
* `mr_est` restrictions: the weights on the two logs sum to one (doubling every
  variance doubles the forecast; QLIKE is scale-free) and `b ∈ [0,1]`; see
  "tried and dropped" for the unrestricted version. 120 training rows minimum,
  36 months history: round numbers matching eligibility.
* Correlation base half-life 36 months for shrinkage: the middle of the grid.
* Block target by `assetforecast.universe` group: the a-priori economic structure
  (stock–bond correlations are nothing like stock–stock ones, so a single
  constant target is the wrong prior; it was scored and is worse).

## Results

Common origins only; QLIKE for vol, MSE of correlations for corr (both the
scoreboard's headline). Skill = % reduction in mean loss vs the best
reference on the same origins, 90% Fieller-EWC interval; DM = EWC
Diebold–Mariano, two-sided. Eras follow the protocol (1926–1945, 1946–1979,
1980–1999, 2000–2013; `pre-1926` = the monthly-only us_large era in `all`).

#### Vol, subset `daily`: 1051 common origins 1926-06..2013-12; skill and DM vs best ref `ref_last12m`

| model | qlike | skill_pct | ci90_lo | ci90_hi | dm_p |
|---|---|---|---|---|---|
| bench_vol_mr_est | 0.403 | 20.207 | -14.958 | 42.295 | 0.273 |
| bench_vol_d1r_hl12 | 0.407 | 19.293 | -2.464 | 33.637 | 0.130 |
| bench_vol_m_hl12 | 0.409 | 18.906 | -0.802 | 34.113 | 0.110 |
| bench_vol_mr_fixed | 0.410 | 18.778 | -16.434 | 39.615 | 0.297 |
| bench_vol_d_hl12 | 0.412 | 18.323 | -5.064 | 34.858 | 0.168 |
| bench_vol_m_hl36 | 0.464 | 8.066 | -30.869 | 33.855 | 0.660 |
| bench_vol_d_hl36 | 0.465 | 7.888 | -32.792 | 34.395 | 0.677 |
| ref_last12m | 0.505 | 0.000 |  |  |  |
| ref_last60m | 0.520 | -2.961 | -40.299 | 23.986 | 0.862 |
| ref_blend | 0.567 | -12.378 | -71.374 | 29.197 | 0.635 |
| bench_vol_m_hl120 | 0.619 | -22.580 | -92.040 | 22.095 | 0.423 |
| ref_longrun | 0.837 | -65.734 | -176.152 | 8.651 | 0.147 |

#### Vol QLIKE by era, subset `daily`

| model | 1926-1945 | 1946-1979 | 1980-1999 | 2000-2013 |
|---|---|---|---|---|
| bench_vol_m_hl12 | 0.477 | 0.378 | 0.229 | 0.649 |
| bench_vol_m_hl36 | 0.728 | 0.379 | 0.248 | 0.611 |
| bench_vol_m_hl120 | 1.015 | 0.584 | 0.325 | 0.568 |
| bench_vol_d_hl12 | 0.509 | 0.400 | 0.239 | 0.555 |
| bench_vol_d_hl36 | 0.731 | 0.396 | 0.253 | 0.565 |
| bench_vol_d1r_hl12 | 0.503 | 0.425 | 0.242 | 0.468 |
| bench_vol_mr_fixed | 0.514 | 0.442 | 0.238 | 0.434 |
| bench_vol_mr_est | 0.617 | 0.383 | 0.237 | 0.388 |
| ref_longrun | 1.475 | 0.720 | 0.576 | 0.600 |
| ref_last12m | 0.418 | 0.599 | 0.313 | 0.673 |
| ref_last60m | 0.792 | 0.410 | 0.280 | 0.749 |
| ref_blend | 0.958 | 0.471 | 0.313 | 0.618 |
| n_origins | 235.000 | 408.000 | 240.000 | 168.000 |

#### Vol QLIKE by asset, subset `daily`

| model | us_large | us_small | intl_dev | em | ust_int | ust_long | tips | ig_credit | hy_credit | reits | gold | commodities |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bench_vol_m_hl12 | 0.377 | 0.336 | 0.463 | 0.383 | 0.453 | 0.483 | 0.477 | 0.525 | 0.960 | 0.568 | 0.322 | 0.330 |
| bench_vol_m_hl36 | 0.406 | 0.362 | 0.388 | 0.339 | 0.421 | 0.432 | 0.505 | 0.434 | 1.068 | 0.636 | 0.377 | 0.338 |
| bench_vol_m_hl120 | 0.513 | 0.456 | 0.322 | 0.311 | 0.507 | 0.577 | 0.548 | 0.399 | 1.020 | 0.775 | 0.438 | 0.365 |
| bench_vol_d_hl12 | 0.353 | 0.330 | 0.342 | 0.344 | 0.482 | 0.457 | 0.433 | 0.477 | 0.940 | 0.509 | 0.368 | 0.277 |
| bench_vol_d_hl36 | 0.396 | 0.357 | 0.328 | 0.330 | 0.454 | 0.424 | 0.509 | 0.416 | 1.017 | 0.594 | 0.393 | 0.320 |
| bench_vol_d1r_hl12 | 0.347 | 0.324 | 0.328 | 0.305 | 0.506 | 0.450 | 0.482 | 0.383 | 0.734 | 0.441 | 0.364 | 0.270 |
| bench_vol_mr_fixed | 0.291 | 0.365 | 0.251 | 0.296 | 0.392 | 0.469 | 0.395 | 0.401 | 0.680 | 0.435 | 0.387 | 0.268 |
| bench_vol_mr_est | 0.329 | 0.337 | 0.255 | 0.309 | 0.398 | 0.380 | 0.313 | 0.352 | 0.651 | 0.398 | 0.349 | 0.250 |
| ref_longrun | 0.662 | 0.588 | 0.298 | 0.305 | 0.646 | 1.171 | 0.584 | 0.712 | 0.927 | 0.837 | 0.499 | 0.362 |
| ref_last12m | 0.390 | 0.402 | 0.517 | 0.473 | 0.712 | 0.678 | 0.508 | 0.616 | 1.062 | 0.592 | 0.420 | 0.406 |
| ref_last60m | 0.496 | 0.403 | 0.513 | 0.410 | 0.531 | 0.513 | 0.565 | 0.446 | 1.362 | 0.603 | 0.401 | 0.415 |
| ref_blend | 0.470 | 0.414 | 0.380 | 0.349 | 0.517 | 0.645 | 0.562 | 0.466 | 1.070 | 0.660 | 0.374 | 0.308 |
| n_cells | 1051.000 | 1015.000 | 337.000 | 236.000 | 625.000 | 625.000 | 168.000 | 409.000 | 349.000 | 326.000 | 473.000 | 276.000 |

#### Vol, subset `all`: 1104 common origins 1922-01..2013-12; skill and DM vs best ref `ref_blend`

| model | qlike | skill_pct | ci90_lo | ci90_hi | dm_p |
|---|---|---|---|---|---|
| bench_vol_mr_est | 0.401 | 25.274 | 13.458 | 32.800 | 0.009 |
| bench_vol_mr_fixed | 0.412 | 23.330 | 12.613 | 30.613 | 0.008 |
| bench_vol_d1r_hl12 | 0.437 | 18.688 | 5.982 | 27.654 | 0.032 |
| bench_vol_d_hl12 | 0.444 | 17.311 | 5.935 | 26.103 | 0.026 |
| bench_vol_m_hl12 | 0.451 | 16.092 | 2.479 | 27.908 | 0.060 |
| bench_vol_d_hl36 | 0.481 | 10.382 | 1.603 | 16.583 | 0.063 |
| bench_vol_m_hl36 | 0.484 | 9.865 | 0.248 | 17.491 | 0.093 |
| ref_blend | 0.537 | 0.000 |  |  |  |
| ref_last60m | 0.557 | -3.774 | -14.092 | 5.342 | 0.487 |
| bench_vol_m_hl120 | 0.568 | -5.701 | -13.796 | 0.438 | 0.127 |
| ref_last12m | 0.579 | -7.915 | -35.218 | 13.117 | 0.546 |
| ref_longrun | 0.699 | -30.146 | -42.862 | -18.387 | 0.001 |

#### Vol QLIKE by era, subset `all`

| model | 1926-1945 | 1946-1979 | 1980-1999 | 2000-2013 | pre-1926 |
|---|---|---|---|---|---|
| bench_vol_m_hl12 | 0.514 | 0.468 | 0.233 | 0.649 | 0.383 |
| bench_vol_m_hl36 | 0.665 | 0.467 | 0.246 | 0.611 | 0.468 |
| bench_vol_m_hl120 | 0.799 | 0.595 | 0.305 | 0.568 | 0.488 |
| bench_vol_d_hl12 | 0.524 | 0.479 | 0.240 | 0.555 | 0.383 |
| bench_vol_d_hl36 | 0.664 | 0.477 | 0.250 | 0.565 | 0.468 |
| bench_vol_d1r_hl12 | 0.522 | 0.495 | 0.241 | 0.468 | 0.383 |
| bench_vol_mr_fixed | 0.499 | 0.456 | 0.241 | 0.434 | 0.374 |
| bench_vol_mr_est | 0.518 | 0.415 | 0.250 | 0.388 | 0.503 |
| ref_longrun | 0.977 | 0.725 | 0.487 | 0.600 | 0.491 |
| ref_last12m | 0.557 | 0.701 | 0.336 | 0.673 | 0.543 |
| ref_last60m | 0.768 | 0.528 | 0.270 | 0.749 | 0.519 |
| ref_blend | 0.774 | 0.515 | 0.288 | 0.618 | 0.502 |
| n_origins | 240.000 | 408.000 | 240.000 | 168.000 | 48.000 |

#### Vol QLIKE by asset, subset `all`

| model | us_large | us_small | intl_dev | em | ust_int | ust_long | tips | ig_credit | hy_credit | reits | gold | commodities |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bench_vol_m_hl12 | 0.362 | 0.336 | 0.413 | 0.389 | 0.443 | 0.522 | 0.477 | 0.625 | 0.919 | 0.480 | 0.322 | 0.355 |
| bench_vol_m_hl36 | 0.389 | 0.362 | 0.339 | 0.337 | 0.416 | 0.504 | 0.505 | 0.621 | 1.025 | 0.531 | 0.377 | 0.397 |
| bench_vol_m_hl120 | 0.490 | 0.456 | 0.283 | 0.307 | 0.494 | 0.594 | 0.548 | 0.678 | 0.982 | 0.634 | 0.438 | 0.443 |
| bench_vol_d_hl12 | 0.339 | 0.330 | 0.318 | 0.354 | 0.469 | 0.507 | 0.433 | 0.608 | 0.900 | 0.438 | 0.368 | 0.341 |
| bench_vol_d_hl36 | 0.380 | 0.357 | 0.292 | 0.328 | 0.446 | 0.499 | 0.509 | 0.615 | 0.978 | 0.502 | 0.393 | 0.392 |
| bench_vol_d1r_hl12 | 0.333 | 0.324 | 0.308 | 0.318 | 0.491 | 0.504 | 0.482 | 0.573 | 0.707 | 0.391 | 0.364 | 0.339 |
| bench_vol_mr_fixed | 0.279 | 0.365 | 0.253 | 0.306 | 0.385 | 0.498 | 0.395 | 0.540 | 0.653 | 0.377 | 0.387 | 0.332 |
| bench_vol_mr_est | 0.315 | 0.337 | 0.269 | 0.321 | 0.386 | 0.440 | 0.313 | 0.490 | 0.631 | 0.362 | 0.349 | 0.316 |
| ref_longrun | 0.632 | 0.588 | 0.263 | 0.301 | 0.618 | 0.931 | 0.584 | 0.909 | 0.895 | 0.680 | 0.499 | 0.480 |
| ref_last12m | 0.374 | 0.402 | 0.540 | 0.534 | 0.672 | 0.791 | 0.508 | 0.721 | 1.010 | 0.509 | 0.420 | 0.457 |
| ref_last60m | 0.477 | 0.403 | 0.437 | 0.396 | 0.518 | 0.580 | 0.565 | 0.722 | 1.304 | 0.509 | 0.401 | 0.444 |
| ref_blend | 0.450 | 0.414 | 0.330 | 0.341 | 0.503 | 0.618 | 0.562 | 0.689 | 1.030 | 0.551 | 0.374 | 0.376 |
| n_cells | 1104.000 | 1015.000 | 433.000 | 259.000 | 693.000 | 1104.000 | 168.000 | 1104.000 | 372.000 | 469.000 | 473.000 | 1015.000 |

#### QLIKE by half-life (months), subset `daily`, common origins 1926-06..2013-12

| input / group | hl6 | hl12 | hl36 | hl60 | hl120 | best | n |
|---|---|---|---|---|---|---|---|
| monthly / overall (pooled) | 0.428 | 0.409 | 0.464 | 0.527 | 0.619 | hl12 | 1051 |
| monthly / equity | 0.373 | 0.373 | 0.381 | 0.407 | 0.448 | hl12 | 2639 |
| monthly / rates | 0.551 | 0.469 | 0.436 | 0.466 | 0.543 | hl36 | 1418 |
| monthly / credit | 0.740 | 0.725 | 0.726 | 0.714 | 0.685 | hl120 | 758 |
| monthly / real | 0.405 | 0.399 | 0.445 | 0.482 | 0.521 | hl12 | 1075 |
| 21d-sum / overall (pooled) | 0.420 | 0.412 | 0.465 | 0.522 | 0.610 | hl12 | 1051 |
| 21d-sum / equity | 0.336 | 0.342 | 0.366 | 0.393 | 0.435 | hl6 | 2639 |
| 21d-sum / rates | 0.521 | 0.465 | 0.447 | 0.480 | 0.555 | hl36 | 1418 |
| 21d-sum / credit | 0.667 | 0.690 | 0.693 | 0.681 | 0.658 | hl120 | 758 |
| 21d-sum / real | 0.386 | 0.387 | 0.435 | 0.475 | 0.521 | hl6 | 1075 |

#### Flatness, subset `daily`: % higher mean QLIKE than the best half-life of the same input (90% CI), DM p-value

| variant | pct_worse | ci90_lo | ci90_hi | dm_p |
|---|---|---|---|---|
| monthly hl6 | 4.659 | -8.072 | 17.914 | 0.517 |
| monthly hl12 | -0.000 |  |  |  |
| monthly hl36 | 13.367 | -4.737 | 36.817 | 0.220 |
| monthly hl60 | 28.615 | 0.542 | 66.255 | 0.094 |
| monthly hl120 | 51.159 | 8.156 | 108.276 | 0.053 |
| 21d-sum hl6 | 1.971 | -9.272 | 14.138 | 0.760 |
| 21d-sum hl12 | -0.000 |  |  |  |
| 21d-sum hl36 | 12.776 | -3.802 | 32.322 | 0.197 |
| 21d-sum hl60 | 26.710 | 1.485 | 57.524 | 0.083 |
| 21d-sum hl120 | 48.016 | 9.586 | 94.070 | 0.045 |

#### QLIKE by half-life (months), subset `all`, common origins 1922-01..2013-12

| input / group | hl6 | hl12 | hl36 | hl60 | hl120 | best | n |
|---|---|---|---|---|---|---|---|
| monthly / overall (pooled) | 0.476 | 0.451 | 0.484 | 0.519 | 0.568 | hl12 | 1104 |
| monthly / equity | 0.369 | 0.363 | 0.367 | 0.390 | 0.429 | hl12 | 2811 |
| monthly / rates | 0.563 | 0.490 | 0.473 | 0.498 | 0.554 | hl36 | 1965 |
| monthly / credit | 0.710 | 0.699 | 0.723 | 0.737 | 0.754 | hl12 | 1476 |
| monthly / real | 0.395 | 0.377 | 0.424 | 0.458 | 0.488 | hl12 | 1957 |
| 21d-sum / overall (pooled) | 0.461 | 0.444 | 0.481 | 0.516 | 0.564 | hl12 | 1104 |
| 21d-sum / equity | 0.334 | 0.334 | 0.353 | 0.378 | 0.417 | hl12 | 2811 |
| 21d-sum / rates | 0.541 | 0.488 | 0.481 | 0.508 | 0.563 | hl36 | 1965 |
| 21d-sum / credit | 0.673 | 0.681 | 0.706 | 0.720 | 0.740 | hl6 | 1476 |
| 21d-sum / real | 0.385 | 0.371 | 0.419 | 0.454 | 0.487 | hl12 | 1957 |

#### Flatness, subset `all`: % higher mean QLIKE than the best half-life of the same input (90% CI), DM p-value

| variant | pct_worse | ci90_lo | ci90_hi | dm_p |
|---|---|---|---|---|
| monthly hl6 | 5.672 | -2.111 | 13.916 | 0.220 |
| monthly hl12 | -0.000 |  |  |  |
| monthly hl36 | 7.422 | -2.918 | 20.586 | 0.239 |
| monthly hl60 | 15.254 | 1.447 | 33.203 | 0.069 |
| monthly hl120 | 25.973 | 7.511 | 49.893 | 0.022 |
| 21d-sum hl6 | 3.761 | -3.552 | 11.626 | 0.385 |
| 21d-sum hl12 | -0.000 |  |  |  |
| 21d-sum hl36 | 8.379 | -1.601 | 20.002 | 0.165 |
| 21d-sum hl60 | 16.175 | 3.227 | 31.435 | 0.042 |
| 21d-sum hl120 | 26.944 | 10.475 | 45.965 | 0.010 |

#### Daily inputs vs monthly, subset `daily` (1051 origins 1926-06..2013-12; 5890 asset-origin cells). skill = % lower mean QLIKE of the first model (90% Fieller CI, pooled series); dm_p = DM on the pooled series; panel_* = Driscoll-Kraay panel DM on asset cells; skill_<group> = point skill on that group's cells

| comparison | skill_pct | ci90_lo | ci90_hi | dm_p | panel_diff | panel_p | skill_equity | skill_rates | skill_credit | skill_real |
|---|---|---|---|---|---|---|---|---|---|---|
| d_hl6 vs m_hl6 | 1.868 | -11.247 | 11.505 | 0.767 | -0.037 | 0.400 | 9.997 | 5.456 | 9.838 | 4.695 |
| d_hl12 vs m_hl12 | -0.720 | -11.969 | 7.965 | 0.894 | -0.021 | 0.535 | 8.216 | 0.848 | 4.869 | 2.943 |
| d_hl36 vs m_hl36 | -0.194 | -5.765 | 4.844 | 0.947 | -0.010 | 0.624 | 3.848 | -2.615 | 4.591 | 2.271 |
| d_hl60 vs m_hl60 | 0.772 | -3.038 | 4.352 | 0.708 | -0.008 | 0.610 | 3.319 | -2.859 | 4.578 | 1.470 |
| d_hl120 vs m_hl120 | 1.374 | -1.097 | 3.764 | 0.322 | -0.007 | 0.561 | 2.961 | -2.292 | 3.973 | 0.137 |
| d1r_hl6 vs d_hl6 | 7.569 | -2.104 | 14.056 | 0.168 | -0.058 | 0.157 | 6.704 | 10.286 | 34.118 | 7.991 |
| d1r_hl12 vs d_hl12 | 1.188 | -8.586 | 8.502 | 0.804 | -0.024 | 0.443 | 2.952 | -2.866 | 21.062 | 6.119 |
| d1r_hl36 vs d_hl36 | -0.863 | -12.527 | 7.768 | 0.875 | 0.015 | 0.324 | -0.202 | -12.106 | -2.209 | 0.827 |
| d1r_hl6 vs m_hl6 | 9.295 | -10.941 | 22.129 | 0.362 | -0.095 | 0.248 | 16.031 | 15.181 | 40.600 | 12.311 |
| d1r_hl12 vs m_hl12 | 0.477 | -18.056 | 13.274 | 0.955 | -0.046 | 0.476 | 10.926 | -1.994 | 24.905 | 8.882 |
| d1r_hl36 vs m_hl36 | -1.059 | -13.749 | 8.173 | 0.857 | 0.005 | 0.855 | 3.653 | -15.038 | 2.483 | 3.079 |

#### Dropped vol variant (unconstrained estimated blend)

| subset / model | qlike | skill_vs_mr_est | ci90_lo | ci90_hi | dm_p |
|---|---|---|---|---|---|
| daily / bench_vol_mr_est | 0.403 | 0.000 |  |  |  |
| daily / diag_vol_mr_est_free | 0.422 | -4.879 | -9.721 | 2.052 | 0.208 |
| daily / bench_vol_mr_fixed | 0.410 | -1.791 | -16.959 | 9.379 | 0.801 |
| all / bench_vol_mr_est | 0.401 | 0.000 |  |  |  |
| all / diag_vol_mr_est_free | 0.472 | -17.711 | -41.936 | 10.539 | 0.276 |
| all / bench_vol_mr_fixed | 0.412 | -2.602 | -10.106 | 5.308 | 0.566 |

#### Correlation, subset `daily`: 1015 common origins 1929-06..2013-12; skill/DM on pooled MSE vs best ref `ref_corr_last120m`; cov_qlike_rel = multivariate QLIKE of `bench_vol_d_hl12` vols x this correlation (1015 origins), cov_diff / cov_dm_p vs `ref_corr_last120m`

| model | mse | skill_pct | ci90_lo | ci90_hi | dm_p | cov_qlike_rel | cov_diff | cov_dm_p |
|---|---|---|---|---|---|---|---|---|
| bench_corr_d_hl36 | 0.0406 | 5.8885 | -0.4857 | 11.2539 | 0.1219 | -0.4797 | -0.0714 | 0.0142 |
| bench_corr_shrink_block | 0.0409 | 5.3057 | -1.4068 | 11.3127 | 0.1746 | -0.5159 | -0.1076 | 0.0636 |
| bench_corr_blend | 0.0411 | 4.8617 | -0.0906 | 8.7514 | 0.1047 | -0.4611 | -0.0528 | 0.0147 |
| bench_corr_d_hl120 | 0.0424 | 1.7863 | -0.7105 | 4.2008 | 0.2169 | -0.4084 | -0.0001 | 0.9944 |
| bench_corr_m_hl36 | 0.0426 | 1.3501 | -5.8756 | 7.1061 | 0.7127 | -0.4570 | -0.0487 | 0.0120 |
| bench_corr_m_hl120 | 0.0430 | 0.3097 | -2.2072 | 2.6341 | 0.8181 | -0.3106 | 0.0977 | 0.2671 |
| ref_corr_last120m | 0.0432 | 0.0000 |  |  |  | -0.4083 | 0.0000 |  |
| ref_corr_longrun | 0.0475 | -10.1193 | -16.3338 | -4.4400 | 0.0104 | 4.8074 | 5.2157 | 0.2239 |
| ref_corr_zero | 0.5079 | -1076.3215 | -2020.9424 | -602.9818 | 0.0000 | 0.3864 | 0.7947 | 0.0000 |

#### Correlation MSE by era

| model | 1926-1945 | 1946-1979 | 1980-1999 | 2000-2013 |
|---|---|---|---|---|
| bench_corr_m_hl36 | 0.0019 | 0.0292 | 0.0711 | 0.0827 |
| bench_corr_m_hl120 | 0.0019 | 0.0279 | 0.0724 | 0.0867 |
| bench_corr_d_hl36 | 0.0017 | 0.0253 | 0.0720 | 0.0791 |
| bench_corr_d_hl120 | 0.0017 | 0.0261 | 0.0737 | 0.0854 |
| bench_corr_shrink_block | 0.0017 | 0.0253 | 0.0730 | 0.0793 |
| bench_corr_blend | 0.0018 | 0.0265 | 0.0714 | 0.0797 |
| ref_corr_longrun | 0.0020 | 0.0296 | 0.0805 | 0.0981 |
| ref_corr_last120m | 0.0019 | 0.0269 | 0.0764 | 0.0842 |
| ref_corr_zero | 0.9024 | 0.5552 | 0.2702 | 0.2655 |
| n_origins | 199.0000 | 408.0000 | 240.0000 | 168.0000 |

#### Correlation MSE by group pair (common pair-origin cells)

| model | credit-credit | credit-equity | credit-rates | credit-real | equity-equity | equity-rates | equity-real | rates-rates | rates-real | real-real |
|---|---|---|---|---|---|---|---|---|---|---|
| bench_corr_m_hl36 | 0.0574 | 0.0637 | 0.0703 | 0.0651 | 0.0263 | 0.1134 | 0.0886 | 0.0184 | 0.0903 | 0.0834 |
| bench_corr_m_hl120 | 0.0621 | 0.0632 | 0.0821 | 0.0657 | 0.0327 | 0.1200 | 0.0870 | 0.0186 | 0.0859 | 0.0773 |
| bench_corr_d_hl36 | 0.0538 | 0.0615 | 0.0695 | 0.0633 | 0.0278 | 0.1083 | 0.0868 | 0.0176 | 0.0868 | 0.0802 |
| bench_corr_d_hl120 | 0.0602 | 0.0623 | 0.0817 | 0.0661 | 0.0338 | 0.1167 | 0.0883 | 0.0191 | 0.0844 | 0.0775 |
| bench_corr_shrink_block | 0.0538 | 0.0648 | 0.0727 | 0.0627 | 0.0281 | 0.1080 | 0.0856 | 0.0161 | 0.0849 | 0.0790 |
| bench_corr_blend | 0.0534 | 0.0628 | 0.0727 | 0.0604 | 0.0276 | 0.1168 | 0.0800 | 0.0164 | 0.0847 | 0.0732 |
| ref_corr_longrun | 0.0729 | 0.0677 | 0.1060 | 0.0703 | 0.0396 | 0.1382 | 0.0927 | 0.0194 | 0.0861 | 0.0781 |
| ref_corr_last120m | 0.0617 | 0.0646 | 0.0816 | 0.0682 | 0.0323 | 0.1153 | 0.0901 | 0.0194 | 0.0859 | 0.0803 |
| ref_corr_zero | 0.3542 | 0.2356 | 0.4751 | 0.1405 | 0.6397 | 0.1357 | 0.1941 | 0.7622 | 0.0995 | 0.1246 |
| n_cells | 349.0000 | 2662.0000 | 1852.0000 | 1962.0000 | 2397.0000 | 4318.0000 | 3797.0000 | 961.0000 | 2654.0000 | 878.0000 |

#### Correlation by half-life (1015 origins)

| input / loss | hl6 | hl12 | hl36 | hl60 | hl120 |
|---|---|---|---|---|---|
| monthly mse | 0.0604 | 0.0502 | 0.0426 | 0.0419 | 0.0430 |
| monthly cov qlike_rel | -0.0348 | -0.3544 | -0.4570 | -0.4404 | -0.3106 |
| 21d-sum mse | 0.0507 | 0.0446 | 0.0406 | 0.0407 | 0.0424 |
| 21d-sum cov qlike_rel | -0.3172 | -0.4515 | -0.4797 | -0.4566 | -0.4084 |

#### Correlation: 21-day-sum vs monthly inputs (skill = % lower MSE; cov_diff < 0 favours the first)

| comparison | skill_mse | ci90_lo | ci90_hi | dm_p | cov_diff | cov_dm_p |
|---|---|---|---|---|---|---|
| d vs m hl6 | 16.0382 | 11.1173 | 21.3443 | 0.0004 | -0.2823 | 0.0137 |
| d vs m hl12 | 11.2163 | 7.0850 | 15.7349 | 0.0009 | -0.0970 | 0.0180 |
| d vs m hl36 | 4.6005 | 1.8560 | 7.6139 | 0.0131 | -0.0227 | 0.0851 |
| d vs m hl60 | 2.6826 | 0.2671 | 5.2817 | 0.0721 | -0.0162 | 0.1374 |
| d vs m hl120 | 1.4812 | -0.6704 | 3.6956 | 0.2372 | -0.0979 | 0.1698 |
| shrink_const vs d_hl36 | -5.2247 | -11.7403 | -1.2435 | 0.0270 | 0.0000 |  |

## Where the bench models lose

* **1926–1945 (`daily`)**: `ref_last12m` (0.418) beats every bench model
  (best `bench_vol_m_hl12` 0.477). In the Depression decade an equal-weight
  last-12-months window (mean age 6 months) is shorter memory than a 12-month
  half-life (mean age ~17 months). `bench_vol_mr_est` is worst of the bench
  set there (0.617) because its weight on the recent level was estimated on
  pre-1926 monthly-only US data (b = 0 to 0.47); it becomes the best model from
  1946 on. Pre-1926 (`all`) it is also worse (0.503) than `mr_fixed` (0.374).
* **intl_dev and em**: `ref_longrun` (0.30 / 0.30 on `daily`) beats the short
  EWMAs (`m_hl12` 0.46 / 0.38) and roughly ties the blends; these assets' daily
  history is short and dominated by 2000–2013.
* **hy_credit**: every model is poor (QLIKE 0.65–1.36). The 2008 jump (21-day
  vol × 7.7 across the HYG splice) is unforecastable from the smoothed VWEHX
  history; `ref_longrun` (0.93) beats `d_hl12`/`m_hl12`/`hl36`.
* **rates**: 36-month half-lives beat 12-month ones; the registered hl12
  models give up ~5% there.
* **Correlation**: in 1980–1999 `bench_corr_m_hl36` (0.0711) and
  `bench_corr_blend` beat `bench_corr_d_hl36` (0.0720); for rates–real and
  real–real pairs the 10-year and all-history estimates (`ref_corr_longrun`,
  `*_hl120`) are as good or better. The MSE gain over `ref_corr_last120m` is
  6% and not significant.

## Tried and dropped (with scores)

| variant | why dropped | score |
|---|---|---|
| `mr_est` with free coefficients on both logs and a lognormal `s²/2` correction (first version) | estimated coefficients explode while only us_large has history (−4.0 on the long-run log in 1885–1900, intercept −21); restricted before it was scored | QLIKE 0.422 vs 0.403 (`daily`, −4.9%, p = 0.21); 0.472 vs 0.401 (`all`, −17.7%, p = 0.28) |
| shrinkage intensity counting entries alone in their block (bug) | those entries have `f_ij = r_ij`, so they inflated the numerator; fixed | MSE 0.0409 before and after (skill vs best ref 5.29% → 5.31%) |
| shrinkage toward a single constant correlation | wrong prior (pulls stock–bond toward stock–stock); block target chosen a priori, constant scored for the record | MSE 5.2% worse than the unshrunk `d_hl36` (CI −11.7% to −1.2%, p = 0.027); δ ≈ 0.05 |
| unregistered grid points: vol monthly hl 6/60, 21-day sums hl 6/60/120, 1-day × ratio hl 6/36; corr hl 6/12/60 for both inputs | diagnostics for the half-life and daily-input questions | full curves in the tables above |

## Run time

Full monthly run, 1,116 origins (1922-01 to 2014-12; earlier origins have
< 2 eligible assets), one process per model, two at a time while other
agents loaded the four cores. Mean model time per origin: vol `m_*` 1.1 ms,
`d_*` 8–9 ms, `d1r` 19 ms, `mr_fixed` 8 ms, `mr_est` 26 ms; corr `m_*` 5 ms,
`d_*` 24–28 ms, `shrink_block` 34 ms, `blend` 60 ms. The slowest model
(`bench_corr_blend`) takes 76 s wall for the whole run, far below the 10-minute
limit. No state is cached between calls (every forecast is recomputed from the
view), so cold and warm calls agree exactly; `mr_est` refits a 2-parameter OLS
on ~9,000 rows per call but its fit only changes once a year (training cut at
the last December-end). `python -m assetforecast.models.bench.analysis` (diagnostic
runs + tables) took about 20 minutes under load.

## Tests

`tests/models/test_bench.py` (48 tests, ~4 s on `make_synthetic_panel(seed=3)`):
every registered model returns a valid forecast (finite, positive vols;
symmetric unit-diagonal PD correlations) at 1979 (monthly-only), 1990 and 2010,
for the eligible set and for an arbitrary reordered subset; cold = warm
(after other origins, in another order) bit for bit; the forecast at 2004-06
is bit-identical when every return and tier after it is altered; the daily
monthly-variance building block matches `realized_kday_cov` on the same
12-month window; `ew_series` equals `ew_mean` at each month; `mr_weight(6)` =
0.51; pairwise correlation recovers a known 0.6 with a late-starting third
asset (effective months 100 = its overlap); shrinkage intensity falls with
the effective sample size and stays in [0, 1]. `tests/test_contract.py`
passes.

## Caveats

* The `_d` vol models equal the `_m` models wherever an asset has no daily
  data, so in the `all` subset their differences are diluted by construction;
  the daily-input tests use the `daily` subset (targets from daily data),
  where inputs may still be partly monthly for long half-lives.
* The 1-day ratio is reset only at TIER changes; same-tier splices (VEIEX→EEM,
  the Treasury source changes of 1971/1977/1985, gold 1998/2004, commodities
  2009, ig_credit 1985) are not visible in a view and the 36-month ratio takes
  about three years to adapt across them. A splice-date accessor on the view
  would allow a clean reset (see below).
* Smoothed-NAV eras (hy_credit VWEHX 1985–2007, reits FRESX/VGSIX) attenuate
  correlations even at 21 days; the 21-day-sum correlation models inherit
  that, and the target, measured the same way, does not penalize it.
* The shrinkage intensity uses the normal-theory sampling variance of a
  correlation; volatility clustering and fat tails make the true sampling
  variance larger, so `δ` is if anything too small.
* `mr_est`'s smearing factor (~1.4) is an average correction for skewed,
  heteroskedastic log residuals; it raises forecasts everywhere, so
  `mr_est` over-forecasts calm periods (its `log_err`, 0.15, is among the highest
  of the set). It still wins on QLIKE, which punishes under-forecasts more.
* Skill intervals on `daily` are wide because 1926–1945 dominates the loss
  variance; on `all` the same point estimates are significant.
* `cov_qlike_rel` of `ref_corr_longrun` reaches ~400 at three 2008 origins
  (a near-singular forecast after eigenvalue clipping), which is why its mean
  is 4.8; read that column through the DM p-values.

## Shared code

No change needed. Nice to have: the splice dates of `segments.csv` exposed
through the view (they are not look-ahead), so a model using 1-day data can
reset its scale at every splice, not only at tier changes.

## Reproduce

```
export ASSETFORECAST_MAX_DATE=2014-12-31 OMP_NUM_THREADS=1
python -m assetforecast.eval.run forecast --kind vol --model bench_vol_m_hl12 bench_vol_m_hl36 bench_vol_m_hl120 \
    bench_vol_d_hl12 bench_vol_d_hl36 bench_vol_d1r_hl12 bench_vol_mr_fixed bench_vol_mr_est --jobs 2
python -m assetforecast.eval.run forecast --kind corr --model bench_corr_m_hl36 bench_corr_m_hl120 \
    bench_corr_d_hl36 bench_corr_d_hl120 bench_corr_shrink_block bench_corr_blend --jobs 2
python -m assetforecast.models.bench.analysis --diag-dir <scratch>/bench_diag --out <scratch>/tables.md
pytest tests/models/test_bench.py
```
