# Structural family (`src/volmodel/models/structural/`)

Forecasts built from the economic structure of an asset class: slow,
**observable exposures** (duration, spread level) times **factor volatilities**
(yield vol, relative spread vol) that are estimated from long histories.
All results are development runs on data to 2014 (`VOLMODEL_MAX_DATE=2014-12-31`),
monthly origins, scored on the 21-day-horizon target; the holdout after 2014 has
not been touched.

## Summary

* **Credit: clear gain from duration-times-spread (DTS).** Investment grade:
  QLIKE 0.506 (lognormal rate part) vs 0.689 for the best reference (ref_blend),
  skill +27% [90% CI +7, +41], DM p = 0.04. High yield: 0.59-0.61 for every
  form vs 0.90 for the best reference (ref_longrun) and 1.03 for ref_blend, a
  skill of about +33% that is not statistically significant (DM p ≈ 0.35; the
  confidence interval is unbounded because 2008 dominates the reference's
  loss). Scaling the spread component by today's spread level is what does it
  (see "What drives the credit gain").
* **Treasuries: the "normal" hypothesis fails; lognormal holds before 2000.**
  Duration × basis-point vol (normal) loses to plain return-history
  extrapolation at every estimation window (ust_int −19%, ust_long −30% vs
  ref_blend). Over 1953-2014 basis-point vol is *less* stable than price vol for
  these ladders. Duration × relative yield vol (lognormal) beats the references
  on ust_int (+16% vs ref_blend) and ust_long (+11% vs ref_blend, +5% vs the
  best reference). That is not significant on its own (p 0.4-0.7). It wins
  1946-1999 by a wide margin and loses 2000-2013, where normal wins. This is the
  regime pattern of "lognormal when yields are high, normal when they are low".
* **TIPS:** the structural prior (real yield vol ≈ κ × nominal bp vol, κ
  shrunk toward 1) beats every reference by a wide margin in the normal form:
  0.303 vs 0.508 for the best reference, skill +40% [+9, +54], DM p = 0.07. It
  fails in the lognormal and CEV forms, which tie TIPS to nominal yields that
  were low after 2008.
* **Pooled over the five structural assets:** `structural_lognormal` +20% vs
  ref_blend [+5, +30], DM p = 0.047. Panel DM (Driscoll-Kraay) p = 0.06 vs
  ref_blend, 0.03 vs ref_last60m and 0.05 vs ref_last12m.
* **Covariance:** the factor model (`structural_factor_global`) has a cov
  QLIKE_rel of −0.387 vs −0.213 for ref_blend × ref_corr_last120m (DM p =
  0.025). It also beats structural vols × ref_corr_last120m (−0.269, p ≈ 0.10),
  so the joint structure adds something beyond the vols. Its **correlations
  alone are not competitive**: MSE 0.0464 vs 0.0432 for ref_corr_last120m
  (−7.5% [−16, −1], p = 0.06), level with ref_corr_longrun (0.0475).

## Models

Common notation: `r_m` is the monthly log excess return of the asset, `y_m`
the month-end yield, and `D_m` the modified duration at the end of month m.

`blend_var(x) = sqrt(12·mean_all(x²) · 12·mean_last60(x²))`, uncentered. This is
exactly ref_blend's estimator, so each structural model differs from ref_blend
**only through the structure**.

Only monthly returns are used. A squared calendar-month return is one draw of
the 21-day-horizon variance, so the autocorrelated pre-1990 daily yield changes
(reports/data/treasuries.md, VR21 up to 2.7) need no correction. The daily
yield series are used only through their month-end levels.

### Vol models (all four share the same code; only β differs)

| model | β (yield-vol elasticity) |
|---|---|
| `structural_normal` | 0: basis-point yield vol is the stable quantity |
| `structural_lognormal` | 1: relative yield vol (σ_y / y) is the stable quantity |
| `structural_sqrt` | ½: an a-priori midpoint (CEV square root) |
| `structural_cev` | estimated walk-forward, re-estimated each January (below) |

**Treasuries** (`ust_int`, `ust_long`):

    n_m = r_m / (D_{m-1} · max(y_{m-1}, 1%)^β)
    σ²_t = D_t² · max(y_t, 1%)^{2β} · blend_var(n)

* D is the modified duration of the asset's own par ladder: 7-10y by 0.5 for
  ust_int and 20-30y by 1 for ust_long. It is computed in closed form at
  today's yield: for a semiannual par bond, D = (1 − (1+y/2)^(−2M))/y. This
  reproduces the panel's `ust_ladder_duration_*` predictors to 0.5% (median
  0.0%), and it works back to 1871 from the monthly long yield.
* Yields:
  * ust_int: the 10y CMT (daily from 1962). Earlier, `macro_long_yield`, which
    is the 10y CMT monthly average from 1953 and long Treasury bonds before.
  * ust_long: the 20y CMT, else the 30y, else the 10y, else `macro_long_yield`.
* **ETF era.** D is multiplied by the trailing 12-month mean of
  `ust_etf_duration_est / ust_ladder_duration`, clipped to [0.75, 1.25], with 1
  before the estimate exists (2003-08). Today's instrument is the ETF. Its
  effective duration differs from the par ladder's, by 0.87-0.95 in 2004-2012
  and by about 1.13 for TLT after 2020. Normalizing ETF returns by the ETF's
  duration makes the yield-vol history consistent across the 2002 splice.
* The 1% yield floor is fixed a priori. Below about 1%, proportional scaling is
  not credible, and the dev sample never reaches it (the minimum 10y is
  1.4%). The live sample does (2020).

**TIPS:**

    σ²_t = D_real,t² · κ² · V_nom,t
    κ² = (n·κ̂² + 24·1) / (n + 24),   κ̂² = mean(u_tips²) / mean(u_int²)

* `u_tips = r_tips / D_real` and `u_int = r_int / D_int` over the n common
  months.
* `V_nom,t = max(y_t,1%)^{2β} · blend_var(n_int)` is the model's own nominal
  10y yield-variance forecast.
* D_real is a 2-15y par ladder at the GSW 10y real yield. Its average maturity
  is 8.5y and its duration 7.1-8.1, in line with TIP's stated effective
  duration of about 7.5. Before real yields exist (1997-98) the real yield is
  taken as 3%.
* The prior κ = 1 means "real-yield vol equals nominal-yield vol". It carries
  the weight of 24 months.
* At first eligibility (2000-01) κ̂ = 0.48 (1997-99 was a quiet period in a
  thin market) and the shrunk κ = 0.70. κ̂ settles at 0.75-0.85 by 2005-2014.

**Credit** (`ig_credit`, `hy_credit`), duration × (rates + DTS spread):

    u_m = r_m / D_c,{m-1}          (D_c: par bond at y_int + s; 12y for IG, 6y for HY)
    g_m = Treasury yield change implied by ust_int returns (ust_long before 1953)
    b   = Σ u g / Σ g²             (all history; absorbs duration-level mismatch and the
                                    usual negative rate/spread correlation)
    z_m = (u_m − b g_m) / s_{m-1}  (DTS: spread shock per unit of spread)
    σ²_t = D_c,t² · [ b² · V_g,t + s_t² · blend_var(z) ]

* V_g,t is the Treasury yield-variance forecast of the same β form.
* The spread s is Moody's Baa − 10y CMT, the monthly average from 1953. Before
  1953 it is the Baa yield minus the long Treasury yield.
* Moody's ends in 2025-12, so a live forecast cannot use it. Where it is stale,
  the live Gilchrist-Zakrajsek spread is rescaled into Baa-10y units by the
  median ratio over the last 60 common months (tested on synthetic data). The
  dead daily Baa series (ends 2016) is not used.
* Spreads are floored at 25bp.
* HY uses the same Baa spread. Under DTS, the HY spread only has to be
  proportional to it.
* Typical values:
  * IG: b ≈ 0.70 and λ = sqrt(blend_var(z)) ≈ 0.2. The spread share of the
    variance runs from 0.2 (2005) to 0.6-0.9 (1970, 2010).
  * HY: b ≈ 0.6-0.8, λ ≈ 0.6-0.8, spread share 0.7-0.95.

**Everything else** (us_large, us_small, intl_dev, em, reits, gold,
commodities) gets `ref_blend`: the family makes no structural vol claim there.

The same fallback applies to any structural asset whose inputs are missing or
non-finite at the origin. In the real panel the inputs always exist, from:

* 1871: yields;
* 1919: spreads;
* 1999: real yields (and before that the assumption described above).

In the synthetic panel there are no predictors, and every asset gets ref_blend
exactly (this is tested).

| asset | first eligible | model used |
|---|---|---|
| ust_long | 1922-01 | structural (yield: NBER long bond → H.15 20y monthly avg → 20y/30y CMT) |
| ust_int | 1956-04 | structural (10y monthly avg → 10y CMT) |
| ig_credit | 1922-01 | structural (spread Baa − long gov't to 1953, Baa − 10y after) |
| hy_credit | 1983-01 | structural |
| tips | 2000-01 | structural (κ prior dominates at the start) |
| other 7 | — | ref_blend |

**CEV β, walk-forward.**

* Each January, use all complete calendar years before the origin's year.
* For each year, regress log mean(u²) on log y at the previous year-end, pooled
  over ust_int and ust_long with asset fixed effects. β = slope/2.
* β is clipped to [0, 1.5] a priori. With fewer than 15 asset-years, β = ½.

The estimates:

| at | 1940 | 1950 | 1960 | 1970 | 1980 | 1990-2005 | 2010 | 2014 |
|---|---|---|---|---|---|---|---|---|
| β | 0.18 | 1.06 | 1.24 | 1.26 | 1.40 | 1.50 (clip) | 1.42 | 1.26 |

History says vol rises *more* than proportionally with the yield level. The
estimate is confounded by the WWII peg (low yields, damped month-average data)
and the Volcker years. It then extrapolates badly to the low-yield 2000s (see
below).

### Covariance models

    Σ_t = B_t F_t B_tᵀ + diag(ψ_t)

Factors are monthly and built from the view:

| factor | definition | history from |
|---|---|---|
| EQ | US large log excess return | 1871 |
| RT | the Treasury rate shock g / y^β (same β as the vol form) | 1919 |
| CS | the DTS spread shock z from investment grade | 1919 |
| CM | commodities | 1926 |
| XU | developed ex-US equity (global variant only) | 1975 |

Each asset loads on an a-priori subset of factors:

| asset | factors |
|---|---|
| us_large | EQ |
| us_small, intl_dev, em, reits | EQ, RT |
| ust_int, ust_long | RT, scaled by D·y^β |
| tips | RT, scaled by D_real·y_int^β |
| ig_credit | RT (scaled by D_c·y^β), CS (scaled by D_c·s) |
| hy_credit | RT (D_c·y^β), CS (D_c·s), EQ |
| gold | RT, CM |
| commodities | CM |

In the global variant, intl_dev loads on XU and em on EQ, XU and RT.

* **Fit.** The coefficient c is estimated by weighted least squares over the
  asset's whole history. The bond regressors are multiplied by the exposure
  known at the start of the month, and both sides are divided by the asset's
  structural scale. Today's loading is c × today's exposure.
* **Factor covariance.** Variances by the ref_blend rule; correlations by the
  ref_corr_last120m rule (last 120 months shrunk halfway to the full history),
  made positive definite.
* **Residual variance.** blend_var of the normalized residual × today's scale²,
  floored at 2% of the systematic variance. No asset is spanned exactly.
* **Positive definite by construction.** Every short-history asset (em from
  1992, TIPS from 2000) gets a full set of correlations from day one.

| model | β | XU factor |
|---|---|---|
| `structural_factor_lognormal` | 1 | no |
| `structural_factor_normal` | 0 | no |
| `structural_factor_global` | 1 | yes: intl_dev on XU; em on EQ, XU, RT |

### Hyperparameters and why

| parameter | value | reason |
|---|---|---|
| window | blend of all history and last 60m | identical to ref_blend, so the comparison isolates the structure |
| ladders | 7-10y, 20-30y | the panel's construction of ust_int / ust_long |
| TIPS ladder | 2-15y | average maturity ≈ TIPS market; duration ≈ TIP's stated |
| IG / HY maturity | 12y / 6y | LQD / HY-index average maturity. Only the level-dependence matters, because b absorbs the level |
| κ prior | 1, weight 24 months | "real-yield vol related to nominal-yield vol"; two years, a round number below the 36-month eligibility |
| yield floor | 1% | no proportional scaling below 1%; not binding in dev |
| spread floor | 25bp | not binding |
| ETF ratio | trailing 12m mean, clip [0.75, 1.25] | smooths a noisy 252-day regression estimate |
| β clip (CEV) | [0, 1.5] | a priori; binding 1990-2009 |
| factor corr rule | 120m shrunk 50% to full | identical to ref_corr_last120m |
| residual floor | 2% of systematic | positive definiteness and no exact spanning |

## Results

`all` = every scored origin, including monthly-proxy targets. `daily` = only
daily-based targets.

### Pooled QLIKE, subset `all` (equal-weight mean over the eligible assets, common origins)

| model | 1926-45 | 1946-79 | 1980-99 | 2000-13 | overall |
|---|---|---|---|---|---|
| ref_blend | 0.729 | 0.515 | 0.288 | 0.618 | 0.537 |
| ref_last12m | 0.555 | 0.701 | 0.336 | 0.673 | 0.579 |
| ref_last60m | 0.726 | 0.528 | 0.270 | 0.749 | 0.557 |
| ref_longrun | 0.896 | 0.725 | 0.487 | 0.600 | 0.699 |
| structural_normal | 0.668 | 0.608 | 0.355 | **0.476** | 0.549 |
| structural_lognormal | 0.657 | 0.438 | 0.252 | 0.592 | **0.478** |
| structural_sqrt | 0.655 | 0.511 | 0.292 | 0.510 | 0.501 |
| structural_cev | 0.659 | 0.425 | 0.246 | 0.688 | 0.487 |
| cov:structural_factor_lognormal | 0.704 | 0.449 | 0.249 | 0.626 | 0.499 |
| cov:structural_factor_normal | 0.715 | 0.645 | 0.369 | 0.531 | 0.586 |
| cov:structural_factor_global | 0.704 | 0.449 | 0.250 | 0.617 | 0.498 |
| n_origins | 288 | 408 | 240 | 168 | 1104 |

### Pooled QLIKE, subset `daily`

| model | 1926-45 | 1946-79 | 1980-99 | 2000-13 | overall |
|---|---|---|---|---|---|
| ref_blend | 0.958 | 0.471 | 0.313 | 0.618 | 0.567 |
| ref_last12m | 0.418 | 0.599 | 0.313 | 0.673 | **0.505** |
| ref_last60m | 0.792 | 0.410 | 0.280 | 0.749 | 0.520 |
| ref_longrun | 1.475 | 0.720 | 0.576 | 0.600 | 0.837 |
| structural_normal | 0.958 | 0.568 | 0.414 | 0.476 | 0.605 |
| structural_lognormal | 0.958 | 0.378 | 0.259 | 0.592 | 0.514 |
| structural_sqrt | 0.958 | 0.456 | 0.321 | 0.510 | 0.546 |
| structural_cev | 0.958 | 0.366 | 0.246 | 0.688 | 0.522 |
| cov:structural_factor_lognormal | 1.109 | 0.367 | 0.256 | 0.626 | 0.549 |
| cov:structural_factor_normal | 1.104 | 0.598 | 0.439 | 0.531 | 0.664 |
| cov:structural_factor_global | 1.109 | 0.367 | 0.256 | 0.617 | 0.547 |
| n_origins | 235 | 408 | 240 | 168 | 1051 |

Two points about the pooled scores:

* They are diluted: 7 of the 12 assets are plain ref_blend.
* On `daily`, ref_last12m wins overall on the strength of 1926-45, where only
  US equities have daily targets and ref_blend (our fallback) does badly.
  Excluding that era, structural_lognormal and structural_cev are best in
  1946-1999.

### QLIKE by asset, subset `all`

| model | us_large | us_small | intl_dev | em | ust_int | ust_long | tips | ig_credit | hy_credit | reits | gold | commodities |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ref_blend | 0.450 | 0.414 | 0.330 | 0.341 | 0.503 | 0.618 | 0.562 | 0.689 | 1.030 | 0.551 | 0.374 | 0.376 |
| ref_last12m | 0.374 | 0.402 | 0.540 | 0.534 | 0.672 | 0.791 | 0.508 | 0.721 | 1.010 | 0.509 | 0.420 | 0.457 |
| ref_last60m | 0.477 | 0.403 | 0.437 | 0.396 | 0.518 | 0.580 | 0.565 | 0.722 | 1.304 | 0.509 | 0.401 | 0.444 |
| ref_longrun | 0.632 | 0.588 | 0.263 | 0.301 | 0.618 | 0.931 | 0.584 | 0.909 | 0.895 | 0.680 | 0.499 | 0.480 |
| structural_normal | = ref_blend | | | | 0.601 | 0.806 | **0.303** | 0.672 | **0.587** | = ref_blend | | |
| structural_lognormal | | | | | **0.423** | 0.551 | 0.845 | 0.506 | 0.597 | | | |
| structural_sqrt | | | | | 0.452 | 0.639 | 0.434 | 0.582 | 0.591 | | | |
| structural_cev | | | | | 0.521 | 0.525 | 1.403 | **0.497** | 0.612 | | | |
| cov:structural_factor_lognormal | 0.445 | 0.481 | 0.369 | 0.341 | 0.510 | **0.519** | 0.780 | 0.541 | 0.603 | 0.602 | 0.374 | 0.378 |
| cov:structural_factor_normal | 0.445 | 0.479 | 0.371 | 0.337 | 0.838 | 0.782 | 0.584 | 0.699 | 0.598 | 0.598 | 0.374 | 0.378 |
| cov:structural_factor_global | 0.445 | 0.481 | 0.326 | 0.356 | 0.510 | 0.519 | 0.780 | 0.541 | 0.603 | 0.602 | 0.374 | 0.378 |

On `daily` the picture is the same:

| asset | best reference | structural_normal | structural_lognormal |
|---|---|---|---|
| ust_int | 0.517 | 0.621 | 0.439 |
| ust_long | 0.513 | 0.963 | 0.508 |
| tips | 0.508 | 0.303 | 0.845 |
| ig_credit | 0.446 | 0.427 | 0.333 |
| hy_credit | 0.927 | 0.609 | 0.615 |

The factor model's equity-beta-plus-residual vols for small caps (0.48 vs
0.41) and REITs (0.60 vs 0.55) are **worse** than the assets' own history.
That answers "only if the structure genuinely adds stability": it does not, so
the vol models keep ref_blend there.

### Structural assets by era, subset `all`

| asset era | ref_blend | ref_last12m | ref_last60m | ref_longrun | s_normal | s_lognormal | s_sqrt | s_cev | cov:factor_lognormal | n |
|---|---|---|---|---|---|---|---|---|---|---|
| ust_int 1946-79 | 0.915 | 1.353 | 0.968 | 1.015 | 1.061 | 0.582 | 0.780 | 0.549 | 0.661 | 285 |
| ust_int 1980-99 | 0.226 | 0.110 | 0.168 | 0.468 | 0.338 | 0.135 | 0.196 | 0.153 | 0.173 | 240 |
| ust_int 2000-13 | 0.200 | 0.320 | 0.255 | 0.160 | 0.196 | 0.566 | 0.259 | 1.001 | 0.737 | 168 |
| ust_long 1926-45 | 0.791 | 1.232 | 0.915 | 0.737 | 0.795 | 0.866 | 0.802 | 0.816 | 0.859 | 288 |
| ust_long 1946-79 | 0.696 | 0.955 | 0.651 | 1.007 | 1.052 | 0.471 | 0.706 | 0.436 | 0.453 | 408 |
| ust_long 1980-99 | 0.366 | 0.110 | 0.173 | 1.136 | 0.735 | 0.210 | 0.437 | 0.125 | 0.204 | 240 |
| ust_long 2000-13 | 0.493 | 0.613 | 0.415 | 0.785 | 0.330 | 0.693 | 0.483 | 0.812 | 0.547 | 168 |
| tips 2000-13 | 0.562 | 0.508 | 0.565 | 0.584 | 0.303 | 0.845 | 0.434 | 1.403 | 0.780 | 168 |
| ig_credit 1926-45 | 0.897 | 0.472 | 1.027 | 0.912 | 0.578 | 0.507 | 0.536 | 0.542 | 0.498 | 288 |
| ig_credit 1946-79 | 0.780 | 1.004 | 0.788 | 1.129 | 1.011 | 0.688 | 0.849 | 0.621 | 0.755 | 408 |
| ig_credit 1980-99 | 0.190 | 0.197 | 0.146 | 0.521 | 0.325 | 0.130 | 0.207 | 0.140 | 0.139 | 240 |
| ig_credit 2000-13 | 0.827 | 1.213 | 0.858 | 0.926 | 0.509 | 0.599 | 0.549 | 0.633 | 0.671 | 168 |
| hy_credit 1980-99 | 0.348 | 0.626 | 0.345 | 0.416 | 0.326 | 0.323 | 0.318 | 0.348 | 0.302 | 204 |
| hy_credit 2000-13 | 1.857 | 1.475 | 2.467 | 1.476 | 0.904 | 0.931 | 0.922 | 0.933 | 0.968 | 168 |

### Significance against the best reference (subset `all`)

The best reference for each asset is chosen ex post on the same origins, which
favours the reference. Skill is the % reduction in mean QLIKE. The CI is the
Fieller-EWC 90% interval; p is the two-sided EWC DM test.

| asset | model | best ref | n | QLIKE | ref | skill % | 90% CI | DM p |
|---|---|---|---|---|---|---|---|---|
| ust_int | structural_normal | ref_blend | 693 | 0.601 | 0.503 | −19.4 | [−29, 6] | 0.14 |
| ust_int | structural_lognormal | ref_blend | 693 | 0.423 | 0.503 | +15.9 | [−80, 42] | 0.56 |
| ust_int | structural_sqrt | ref_blend | 693 | 0.452 | 0.503 | +10.3 | [−11, 19] | 0.25 |
| ust_int | structural_cev | ref_blend | 693 | 0.521 | 0.503 | −3.6 | [−189, 44] | 0.93 |
| ust_long | structural_normal | ref_last60m | 1104 | 0.806 | 0.580 | −39.0 | [−97, 25] | 0.28 |
| ust_long | structural_lognormal | ref_last60m | 1104 | 0.551 | 0.580 | +5.0 | [−16, 24] | 0.66 |
| ust_long | structural_cev | ref_last60m | 1104 | 0.525 | 0.580 | +9.5 | [−17, 31] | 0.50 |
| ust_long | cov:structural_factor_lognormal | ref_last60m | 1104 | 0.519 | 0.580 | +10.5 | [−10, 29] | 0.36 |
| tips | structural_normal | ref_last12m | 168 | 0.303 | 0.508 | **+40.3** | **[9, 54]** | 0.07 |
| tips | structural_sqrt | ref_last12m | 168 | 0.434 | 0.508 | +14.6 | [−41, 48] | 0.51 |
| tips | structural_lognormal | ref_last12m | 168 | 0.845 | 0.508 | −66.2 | [−175, −19] | 0.03 |
| tips | structural_cev | ref_last12m | 168 | 1.403 | 0.508 | −176 | [−387, −86] | 0.004 |
| ig_credit | structural_normal | ref_blend | 1104 | 0.672 | 0.689 | +2.5 | [−33, 35] | 0.90 |
| ig_credit | structural_lognormal | ref_blend | 1104 | 0.506 | 0.689 | **+26.6** | **[7, 41]** | **0.04** |
| ig_credit | structural_cev | ref_blend | 1104 | 0.497 | 0.689 | **+27.8** | **[6, 43]** | 0.05 |
| ig_credit | cov:structural_factor_lognormal | ref_blend | 1104 | 0.541 | 0.689 | +21.5 | [1, 36] | 0.08 |
| hy_credit | structural_normal | ref_longrun | 372 | 0.587 | 0.895 | +34.4 | unbounded | 0.36 |
| hy_credit | structural_lognormal | ref_longrun | 372 | 0.597 | 0.895 | +33.2 | unbounded | 0.34 |
| hy_credit | cov:structural_factor_lognormal | ref_longrun | 372 | 0.603 | 0.895 | +32.6 | unbounded | 0.30 |
| 5 assets pooled | structural_normal | ref_blend | 1104 | 0.720 | 0.679 | −6.0 | [−34, 25] | 0.72 |
| 5 assets pooled | structural_lognormal | ref_blend | 1104 | 0.541 | 0.679 | **+20.3** | **[5, 30]** | **0.047** |
| 5 assets pooled | structural_sqrt | ref_blend | 1104 | 0.600 | 0.679 | +11.7 | [−1, 24] | 0.13 |
| 5 assets pooled | structural_cev | ref_blend | 1104 | 0.554 | 0.679 | +18.5 | [−7, 35] | 0.20 |
| 5 assets pooled | cov:structural_factor_lognormal | ref_blend | 1104 | 0.558 | 0.679 | +17.9 | [2, 28] | 0.07 |

"Unbounded": the Fieller interval is unbounded because the benchmark's mean
loss is not significantly above zero (2008 dominates it).

Panel DM (Driscoll-Kraay EWC, the five structural assets, observation-weighted)
for structural_lognormal:

| reference | mean loss difference | p |
|---|---|---|
| ref_blend | −0.129 | 0.061 |
| ref_last60m | −0.160 | 0.027 |
| ref_last12m | −0.224 | 0.048 |

For structural_normal against the same references, p is between 0.66 and 0.92.

## Does duration × yield vol beat return-history extrapolation?

**Like-for-like test.** For each estimation window W, the reference
`ref_W = sqrt(W(r))` is compared with the duration-scaled versions:

* normal: `D_t · sqrt(W(r/D))`;
* lognormal: `D_t · y_t · sqrt(W(r/(D y)))`.

Everything else is equal, including identical origins. Subset `all`, QLIKE:

| asset | window | ref_W | normal (D × bp vol) | lognormal (D × y × rel. vol) | normal vs ref: skill, DM p |
|---|---|---|---|---|---|
| ust_int | all history | 0.618 | 0.821 | 0.565 | −33%, p 0.13 |
| ust_int | last 60m | 0.518 | 0.570 | 0.440 | −10% [−17, −1], p 0.08 |
| ust_int | last 12m | 0.672 | 0.690 | 0.648 | −3% [−9, −1], p 0.04 |
| ust_int | blend | 0.503 | 0.601 | 0.423 | −19%, p 0.14 |
| ust_long | all history | 0.931 | 1.542 | 0.716 | −66%, p 0.18 |
| ust_long | last 60m | 0.580 | 0.649 | 0.565 | −12% [−22, −2], p 0.06 |
| ust_long | last 12m | 0.791 | 0.810 | 0.797 | −2%, p 0.11 |
| ust_long | blend | 0.618 | 0.806 | 0.551 | −30%, p 0.21 |

**Answer: no, not in the normal form.** Multiplying basis-point vol by
today's duration is worse than extrapolating price vol at every window. The
loss is small at 12 months, where duration barely changes, and large over long
windows, where the structure should have helped most.

The premise fails in this sample. The standard deviation of log annual vol
(monthly returns, 1953-2014) is lowest for relative yield vol, not for
basis-point vol:

| | price vol | bp vol | relative vol (bp / y) |
|---|---|---|---|
| ust_int 1953-2014 | 0.60 | 0.67 | **0.55** |
| ust_long 1953-2014 | 0.57 | 0.71 | **0.54** |
| ust_int 1953-1989 | 0.74 | 0.84 | **0.51** |
| ust_long 1953-1989 | 0.65 | 0.88 | **0.52** |
| ust_int 1990-2014 | **0.23** | 0.23 | 0.43 |
| ust_long 1990-2014 | 0.29 | **0.25** | 0.42 |

For example, ust_long bp vol was about 17bp/yr-equivalent in the 1940s, 210
in 1980-84 and 73 in 2010-14, while price vol was 3%, 16% and 13%. For the
long ladder, duration (∝ ~1/y at high yields) and bp vol (∝ y) move against
each other, so price vol is the more stable product.

The lognormal form, which is the structural statement that fits these
facts, beats the reference at the long windows (blend +16% / +11%, all
history +9% / +23%). None of it is significant individually.

**By era (from the era table above):**

| era | normal | lognormal |
|---|---|---|
| 1946-79 (yields rising 2% → 10%) | loses | wins: ust_int 0.58 vs 0.92, ust_long 0.47 vs 0.70 |
| 1980-99 (yields falling from 15%) | loses | wins against ref_blend: ust_int 0.135 vs 0.226, ust_long 0.21 vs 0.37. ref_last12m is better still (0.110). |
| 2000-13 (yields 1.5-5%) | wins on ust_long: 0.33 vs 0.49 for ref_blend and 0.415 for ref_last60m; ties on ust_int (0.196 vs 0.200, below ref_longrun's 0.160) | fails: ust_int 0.57, ust_long 0.69. TIPS and CEV fail for the same reason. |

**Normal vs lognormal evidence, in one line:** proportional (lognormal)
behaviour dominates 1946-1999 and constant-bp (normal) behaviour dominates the
low-yield 2000s. That is the regime pattern the brief anticipated.

Neither a priori form, nor their midpoint (structural_sqrt: good in no era,
second-best overall), nor the walk-forward CEV captures the switch. The
walk-forward estimate (β ≈ 1.3-1.5) learns the 1946-99 relation and
extrapolates it into the 2000s.

A regime form, such as lognormal above a yield threshold and normal below it,
would almost certainly have won in-sample. I did not build one: the threshold
would have been chosen after seeing these era tables (see "Not done").

**ETF effective-duration adjustment** (on by default, decided a priori):

| | ust_int, all | ust_long, all | ust_int 2000-13 (lognormal) | ust_long 2000-13 (lognormal) |
|---|---|---|---|---|
| with adjustment | 0.423 | 0.551 | 0.566 | 0.693 |
| without | 0.416 | 0.544 | 0.536 | 0.649 |

So in dev the adjustment costs slightly (lognormal blend shown). Within dev the
TLT/ladder ratio was 0.87-1.05. The adjustment exists for the post-2020
regime, where the ratio is about 1.13 (reports/data/treasuries.md). Kept as
specified.

## What drives the credit gain

Each variant was computed in scratch over the same origins. QLIKE, subset
`all`:

| | IG (β=0) | IG (β=1) | HY (β=0) | HY (β=1) |
|---|---|---|---|---|
| ref_blend | 0.689 | 0.689 | 1.030 | 1.030 |
| duration-normalized return only (no decomposition) | 0.790 | — | 0.986 | — |
| rates + spread, spread variance NOT scaled by s_t² | 0.770 | 0.664 | 0.880 | 0.937 |
| rates + spread × s_t² (DTS, registered) | 0.672 | 0.506 | 0.587 | 0.597 |

* **The spread-level scaling does the work.** It lowers QLIKE by 0.10-0.16
  for IG and 0.29-0.34 for HY. That is not significant one asset at a time
  (DM p 0.11-0.37).
* **The gain comes in the high-spread eras:**
  * HY 2000-13: 0.90 vs 1.86 for ref_blend.
  * IG 1926-45: 0.51-0.58 vs 0.90.
  * IG 2000-13: 0.51 vs 0.83.
  Forecasts rise at once when spreads widen: at 2008-12 the IG forecast is
  12.5% vs 7.0% for ref_blend, and HY 20% vs 9.8%.
* **IG's rate part needs the lognormal form before 2000.** IG normal is only
  +2.5%; lognormal is +27%.

## Covariance and correlations

Common origins (1015), daily subset. `qlike_rel` is the headline cov loss
(lower is better; differences equal those of multivariate QLIKE).

| model | 1926-45 | 1946-79 | 1980-99 | 2000-13 | all | DM p vs ref composite |
|---|---|---|---|---|---|---|
| ref_blend × ref_corr_last120m | −0.190 | −0.256 | −0.438 | 0.187 | −0.213 | — |
| ref_blend × ref_corr_longrun | −0.190 | −0.242 | 1.035 | 29.07 | 4.92 | — |
| ref_blend × identity | 0.847 | 0.471 | 0.313 | 0.618 | 0.532 | — |
| structural_lognormal × ref_corr_last120m (scratch) | −0.190 | −0.360 | −0.492 | 0.179 | −0.269 | 0.079 |
| **structural_factor_global** | −0.292 | −0.394 | −0.577 | −0.208 | **−0.387** | **0.025** |
| structural_factor_lognormal | −0.292 | −0.394 | −0.577 | −0.161 | −0.379 | 0.024 |
| structural_factor_normal | −0.294 | −0.166 | −0.469 | −0.258 | −0.278 | 0.50 |

* The factor models beat the best reference composite in every era.
* structural_factor_global also beats "structural vols × ref_corr_last120m"
  by 0.118 (DM p = 0.096). So the factor structure adds to cov QLIKE beyond
  the vols: its matrices are better conditioned, which is what QLIKE rewards.
* Frobenius loss (scaled) is a tie: 4.083 vs 4.089.
* Using ref_corr_longrun produces nearly singular matrices after 1980.

**Correlations alone** (`cov:` correlations vs the correlation references, MSE
over pairs, daily):

| model | 1926-45 | 1946-79 | 1980-99 | 2000-13 | all | mean bias | skill vs ref_corr_last120m |
|---|---|---|---|---|---|---|---|
| ref_corr_last120m | 0.0019 | 0.0269 | 0.0764 | 0.0842 | **0.0432** | −0.003 | — |
| ref_corr_longrun | 0.0020 | 0.0296 | 0.0805 | 0.0981 | 0.0475 | −0.004 | — |
| cov:structural_factor_global | 0.0070 | 0.0270 | 0.0760 | 0.0981 | 0.0464 | −0.037 | −7.5% [−16, −1], p 0.06 |
| cov:structural_factor_lognormal | 0.0070 | 0.0270 | 0.0751 | 0.1023 | 0.0469 | −0.039 | −8.6% [−19, −1], p 0.08 |
| cov:structural_factor_normal | 0.0070 | 0.0267 | 0.0751 | 0.1001 | 0.0464 | −0.036 | −7.5% [−17, 0], p 0.09 |

**The factor correlations are not competitive with ref_corr_last120m.** They
are level with ref_corr_longrun (+2%, p 0.4).

* **Systematic bias.** Correlations are too low on average (bias −0.04): a
  factor model with independent residuals attenuates every correlation not
  fully carried by the factors.
* **Where it loses** (mean squared error, factor vs ref_corr_last120m):

  | pair | factor | ref_corr_last120m |
  |---|---|---|
  | tips–ig_credit | 0.20 | 0.04 |
  | intl_dev–em (fixed by the XU factor) | 0.17 | 0.04 → 0.056 with XU |
  | hy_credit–equities and REITs | 0.06-0.14 | 0.03-0.05 |
  | tips–gold | 0.15 | 0.07 |
  | ust–tips (lognormal) | 0.107 | 0.039 |

  Causes:
  * TIPS's rate exposure is tied to nominal y^β.
  * HY's monthly equity loading is attenuated by the smoothed VWEHX NAV before
    2007.
  * TIPS–IG and credit–equity co-movement in 2008 is not carried by the
    factors.
* **Where it wins:** Treasuries/HY, Treasuries/intl, and US equity/TIPS
  (0.11-0.13 vs 0.13-0.16). These are pairs where a slowly estimated exposure
  is more stable than a 10-year sample correlation.

## Tried and dropped (all reported with scores)

| variant | result | status |
|---|---|---|
| Window sweep (all / 60m / 12m / blend) for normal and lognormal Treasuries | table above. Normal loses at every window; lognormal wins at the long windows | diagnostic only, not registered (window fixed a priori to the blend) |
| No ETF effective-duration adjustment | slightly better in dev (ust_int 0.416 vs 0.423, ust_long 0.544 vs 0.551, lognormal) | kept the a-priori choice; reported |
| Credit without DTS scaling; credit as duration-normalized return only | IG 0.664-0.790, HY 0.880-0.986 (vs 0.506 / 0.587 with DTS) | dropped |
| Factor model with Dimson (one-lag) loadings, for the smoothed NAV histories | corr MSE 0.0487 (worse; p = 0.005 vs ref_corr_last120m); cov −0.363; fixes HY-equity pairs, adds noise elsewhere | dropped |
| Factor model with Dimson + XU factor | corr 0.0476, cov −0.370 | dropped |
| Factor model with XU factor (global) | corr 0.0464, cov −0.387 | **registered** (`structural_factor_global`) |

**Not done, deliberately:** a shifted-lognormal or threshold ("lognormal above
y*, normal below") form, and a walk-forward switch between normal and
lognormal by trailing loss. The era table makes these attractive in hindsight.
Any shift or threshold I chose now would be fitted to the scoreboard, and a
loss-based switch would lag the regime change by most of a decade.

Also not attempted:

* Daily 21-day overlapping sums of duration-normalized returns (more precise
  window estimates). The reference uses monthly data and the comparison was
  meant to be like-for-like.
* A real/inflation factor (breakevens exist only from 1999; TIPS enter through
  the rate factor and their residual).

## Run time and tests

Full monthly runs to 2014 (1,116 origins, one core shared with other agents;
wall time about twice CPU time):

| model | seconds |
|---|---|
| structural_normal | 54 |
| structural_lognormal | 49 |
| structural_sqrt | 51 |
| structural_cev | 88 |
| structural_factor_lognormal | 174 |
| structural_factor_normal | 182 |
| structural_factor_global | 81 |

Each vol call takes 20-40 ms (one month-end resample of the predictors), and
each factor call 70-200 ms. Parameters are re-estimated inside every call from
the view, except β (CEV), which uses only years before the origin's year.

`tests/models/test_structural.py` has 28 tests, which run in 4 s on the
synthetic panel. The synthetic panel is used both as is (no predictors →
fallback path) and augmented with synthetic yields, spreads, real yields and a
daily CMT series (structural path). The tests check:

* every registered model is valid at an early (monthly-only), a middle and a
  late origin;
* covariances are positive definite;
* fallback equals ref_blend exactly;
* structural assets differ from ref_blend and others do not;
* cold, warm and out-of-order calls give the same answer;
* the forecast at t is unchanged when returns, predictors and later-published
  vintages after t are scrambled;
* the closed-form duration is correct;
* the vol scales with today's duration;
* the GZ fallback is used when the Moody's series stops;
* `estimate_beta` recovers β = 0 and β = 1;
* the TIPS prior.

`tests/test_contract.py` passes as well.

## Caveats

* **Seven of twelve assets are ref_blend.** Pooled scores understate what
  happens on the five structural assets and inherit ref_blend's weak
  1926-45 equity era.
* **The best references move by asset and era.** ref_last12m is excellent in
  1980-99 Treasuries (0.110), better than any structural form.
* **Pre-1962 Treasury and pre-1980 credit data are month-average or index
  reconstructions.** Vol is damped by about 15%, and this affects the
  references equally. The WWII peg drives the CEV estimate.
* **The HY gain rests mostly on 2008.** It is large but not significant;
  the Fieller interval is unbounded.
* **TIPS has 168 scored origins.** The +40% for structural_normal is the
  family's strongest single-asset result, but it is one 14-year episode. It
  shares the low-yield dependence of the nominal forecast in the other forms.
* **Live-forecast inputs not exercised by dev data:**
  * Moody's ending in 2025 (the GZ path is tested only synthetically);
  * yields below the 1% floor (2020);
  * TLT's duration ratio above 1.
* **No change was needed in shared code.** One observation: `models._load_all`
  imports every family, so one family's broken in-progress package once made
  `python -m volmodel.eval.run forecast` fail for everyone
  (`ImportError ... condcorr.embed`). The family's runs used the harness
  directly in the meantime. Isolating import errors per family in `_load_all`
  would make the CLI robust.
