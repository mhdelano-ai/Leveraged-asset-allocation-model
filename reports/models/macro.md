# Macro family: volatility conditioned on macro state and implied volatility

Package `src/volmodel/models/macro/` (`features.py`, `model.py`, `__init__.py`), tests
`tests/models/test_macro.py`. Every number here is from the development sample
(`VOLMODEL_MAX_DATE=2014-12-31`): origins 1922-01 to 2013-12 with observed targets, every month-end.

**Question.** Beyond the asset's own return history, do observable state variables predict the
LEVEL of next-year volatility?
**Answer on this sample.** No, not detectably. On top of a good own-history base, the macro state
variables and option-implied vol change pooled QLIKE by a few percent either way. No group-level
difference is significant at 5%. The best model of the family is the own-history base with
scale-homogeneous slopes (`macro_base_h`), and adding state variables to it makes it slightly worse
(-7%, p = 0.14). The models with state variables still beat the reference benchmarks after 1946,
but they get that from the own-history base, not from the state variables.

## Models (7 registered, kind `vol`)

| model | content |
|---|---|
| `macro_base` | own-history base: pooled QLIKE (Gamma-GLM) regression per asset group |
| `macro_state` | base + macro state variables (pre-specified set per group) |
| `macro_state_ip` | `macro_state` + industrial-production growth (sensitivity check: IP is latest-vintage and heavily revised) |
| `macro_implied` | base + option-implied variance |
| `macro_full` | base + macro + implied |
| `macro_base_h` | base with own-history slopes constrained to sum to 1 (added after the first full run; see "Tried and dropped / changed") |
| `macro_full_h` | `macro_full` on the homogeneous base |

### Definition

Inputs (all are 21-day-horizon variances, annualized; no 1-day vol is used anywhere, so no
1-day/21-day ratio has to be carried across splices):

* `v_slow`: an EWMA of squared monthly log excess returns, x 12. Half-life 60 months, using all history.
* `v12`: the variance over the last 12 months. In the daily era it uses the target's own estimator (overlapping 21-day sums,
  uncentered, x n_rows/21), when the asset has >= 90% daily coverage and >= 150 rows. Otherwise it is
  12 x mean of the squared monthly returns.
* `v3`: the variance over the last 63 trading days, from the same 21-day estimator (annualized with the trailing-year row count).
  Without daily data it is the last 3 squared monthly returns. If a window is thin, the fallback order is `v3` <- `v12` <- `v_slow`.
* Training target `y`: the next-12-month variance by the same rule (the daily estimator, or the monthly
  proxy, which needs all 12 months). The test suite checks that it equals `realized_kday_cov` exactly. A row is used
  only if its window ends by the fit date.

Stage 1 (base), group g = equity {us_large, us_small, intl_dev, em, reits}, credit {ig, hy}, rates
{ust_int, ust_long, tips}, real {gold, commodities}:

    log E[y] = a_g + b1 log v_slow + b2 log v12 + b3 log v3          (homogeneous: b1 = 1 - b2 - b3)

Stage 2 (state), with the stage-1 forecast as a fixed offset:

    log E[y] = base(s, a) + d0 + sum_k c_k z_k ,   z_k standardized on the training rows

Both stages minimize `nu * sum_i w_i [y_i/mu_i + log mu_i] + sum (d0^2 + c_k^2) / (2 tau^2)`.
Stage 1 has no penalty. The bracket is QLIKE up to constants (the Gamma log-likelihood, log link), so
the fit targets E[y] and is consistent under noisy, unbiased training targets.
`w_i = 1/(12 m_s)` (m_s = the group's assets at month s), so each group-year counts once, because of the 12-month
overlap and common shocks. `nu = 1/phi`, where phi = the weighted mean of (y/mu_base - 1)^2 on the stage-2 rows. The
penalty is therefore a N(0, tau^2) prior weighed against roughly n_years observations. Keeping the base fixed in stage 2
stops a short sample (MOVE 2003-13, GVZ 2008-13) from re-fitting the own-history slopes.

Pre-specified state sets (economic rationale; never searched):

| group | macro | implied |
|---|---|---|
| equity | log Baa-Aaa spread, term spread, log infl. vol 36m, Sahm gap | log VIX^2 (VXO-spliced 1986-89) |
| credit | log Baa-Aaa, term spread, Sahm gap, long-yield level | log VIX^2 |
| rates | long-yield level, log infl. vol, abs(12m policy-rate change), term spread | log (MOVE x ladder duration)^2 |
| real | log infl. vol, inflation yoy, real short rate, abs(dollar yoy) | log GVZ^2 (gold), log OVX^2 (commodities) |

`macro_state_ip` adds IP yoy to every group. Implied regressors are centered per asset (the instrument level and
its variance risk premium differ by asset) and scaled by the pooled sd.

Predictor values are those known at each date: the latest observation with `available_date <= s`, read from the
view's own long frame. Monthly series count as stale (missing) after 190 days. Daily series are the mean of the last 21
published closes and count as stale after 31 days. The log credit spread is Moody's Baa-Aaa. Only at the forecast
date, if that is stale (Moody's ends 2025-12), it is replaced by the GZ spread, mapped by a log-log regression
on the months where both series are published. History is never imputed.

### Fallback chain

At each origin, each asset's information set is the group's candidate variables that are fresh at the origin.
The stage-2 model for that set is trained only on rows where all of them existed. If that leaves fewer than 60
training months, the variable with the shortest training history is dropped and the step is repeated, down to the
empty set (the base). If the group's base itself has fewer than 60 training months (rates and credit before 1927,
commodities 1929-34), the fixed geometric mean of `v_slow` and `v12` is used instead. The sets actually used by
`macro_full` at December origins:

| asset | information sets over time |
|---|---|
| us_large / us_small | cs,ts,ivol (1925/29-54) -> + sahm (1955-90) -> + vix (1991-) |
| intl_dev, reits | cs,ts,ivol,sahm (1974/77-90) -> + vix (1991-) |
| em | all five from its start (1992) |
| ig_credit | FIXED (1925-26) -> cs,ts,ly (1927-54) -> + sahm (1955-90) -> + vix (1991-) |
| hy_credit | cs,ts,sahm,ly (1983-90) -> + vix (1991-) |
| ust_long / ust_int | FIXED (ust_long 1925-26) -> ly,ivol,dpol,ts (to 2007) -> + move_d (2008-) |
| tips | ly,ivol,dpol,ts (2000-07) -> + move_d (2008-) |
| gold / commodities | (commodities FIXED 1929-34) ivol,infl,rsr -> + usd (1979-) -> + iv_own (2013-) |

At the live origin (2026-10): Moody's is dead, so the credit spread comes from GZ. The October-2025 CPI and
unemployment prints that were never published are absorbed by the 190-day staleness rule (the previous print is
used), and a series that stays missing simply drops out of the set. The tests check each case on a synthetic panel
with real series names. VXHYG, VXLQD, VXIEF, VXEFA, VXEEM, RVX, VIX1Y and the VIX term slope are not used: in the
dev sample they have fewer than 60 training months (VIX1Y: 84 months, and its early history was computed after the
fact) or long holes. They are listed here as untested, not as rejected.

### Hyperparameters and why

| value | reason |
|---|---|
| tau = 0.25 | prior: a 1-sd move in one state variable shifts expected variance by up to ~+-25% (vol +-12%); 2-sd bounds +-50%. Fixed before any run |
| 60-month minimum training | five years before a regressor may enter; the prior does the rest |
| v_slow half-life 60 months | a business cycle; also the window of `ref_last60m` |
| 63 trading days for v3 | one quarter: the most recent information, on the target's 21-day scale |
| 190 / 31 days staleness | covers the pre-1947 CPI lag (M+3) plus one missing print / one month of daily closes |
| 21-close average for implied vol | the implied track recommends averages over single prints |
| z clipped at +-4 sd | guards against prints like OVX 3.25 (April 2020) |
| yearly re-estimation at 31 December | protocol |

## Results (QLIKE, lower is better; every month-end)

### Overall, common origins

| model | daily (1051 origins) | all (1104) |
|---|---|---|
| **macro_base_h** | **0.418** | **0.474** |
| macro_full_h | 0.434 | 0.507 |
| macro_full | 0.524 | 0.527 |
| macro_state | 0.528 | 0.532 |
| macro_state_ip | 0.534 | 0.527 |
| macro_base | 0.541 | 0.535 |
| macro_implied | 0.543 | 0.536 |
| ref_last12m | 0.505 | 0.580 |
| ref_last60m | 0.520 | 0.557 |
| ref_blend | 0.567 | 0.537 |
| ref_longrun | 0.837 | 0.699 |

Skill against the best reference (Fieller-EWC 90% interval; DM-EWC p-value):

| model | daily vs ref_last12m | all vs ref_blend |
|---|---|---|
| macro_base | -7.2% [-81, 47], p 0.83 | +0.3% [-22, 25], p 0.98 |
| macro_state | -4.6% [-72, 46], p 0.88 | +1.0% [-21, 27], p 0.94 |
| macro_state_ip | -5.7% [-75, 46], p 0.86 | +1.8% [-19, 26], p 0.89 |
| macro_implied | -7.5% [-82, 46], p 0.83 | +0.1% [-22, 25], p 0.99 |
| macro_full | -3.8% [-73, 47], p 0.91 | +1.9% [-21, 28], p 0.89 |
| macro_base_h | +17.2% [-25, 43], p 0.41 | +11.8% [-4, 28], p 0.21 |
| macro_full_h | +13.9% [-31, 43], p 0.52 | +5.6% [-14, 28], p 0.64 |

Descriptive, because the era cut was chosen after seeing the era table: from 1946 on, every model in the family is
about 25% better than `ref_last60m` on the daily subset (macro_full 25.6% [5, 37], p 0.06; macro_base_h 25.6%
[0, 37], p 0.10). On `all`, macro_base_h is 20.9% better than ref_blend ([2, 33], p 0.08).

### By era

| model | daily 1926-45 | 1946-79 | 1980-99 | 2000-13 | all <1926 | all 1926-45 | all 1946-79 | all 1980-99 | all 2000-13 |
|---|---|---|---|---|---|---|---|---|---|
| macro_base_h | 0.731 | 0.345 | 0.245 | 0.407 | 0.382 | 0.841 | 0.426 | 0.252 | 0.407 |
| macro_full_h | 0.787 | 0.324 | 0.249 | 0.473 | 0.385 | 0.944 | 0.425 | 0.257 | 0.473 |
| macro_base | 1.268 | 0.367 | 0.222 | 0.404 | 0.366 | 1.005 | 0.512 | 0.231 | 0.404 |
| macro_state | 1.203 | 0.341 | 0.217 | 0.484 | 0.368 | 1.005 | 0.470 | 0.230 | 0.484 |
| macro_implied | 1.268 | 0.367 | 0.226 | 0.407 | 0.366 | 1.005 | 0.512 | 0.234 | 0.407 |
| macro_full | 1.203 | 0.341 | 0.228 | 0.441 | 0.368 | 1.005 | 0.470 | 0.239 | 0.441 |
| ref_last12m | 0.418 | 0.599 | 0.313 | 0.673 | 0.543 | 0.557 | 0.701 | 0.336 | 0.673 |
| ref_last60m | 0.792 | 0.410 | 0.280 | 0.749 | 0.519 | 0.768 | 0.528 | 0.270 | 0.749 |
| ref_blend | 0.958 | 0.471 | 0.313 | 0.618 | 0.502 | 0.774 | 0.515 | 0.288 | 0.618 |

### By asset (daily subset)

| model | us_large | us_small | intl_dev | em | ust_int | ust_long | tips | ig | hy | reits | gold | comm |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| macro_base_h | 0.367 | 0.335 | 0.272 | 0.300 | 0.367 | 0.345 | 0.308 | 0.379 | 0.689 | 0.428 | 0.338 | 0.268 |
| macro_full_h | 0.391 | 0.375 | 0.311 | 0.351 | 0.356 | 0.341 | 0.330 | 0.411 | 0.708 | 0.494 | 0.354 | 0.342 |
| macro_base | 0.430 | 0.474 | 0.254 | 0.308 | 0.352 | 0.465 | 0.253 | 0.360 | 0.719 | 0.410 | 0.328 | 0.258 |
| macro_state | 0.427 | 0.491 | 0.298 | 0.351 | 0.324 | 0.402 | 0.241 | 0.429 | 0.767 | 0.507 | 0.369 | 0.287 |
| macro_full | 0.432 | 0.491 | 0.285 | 0.341 | 0.329 | 0.395 | 0.275 | 0.383 | 0.689 | 0.457 | 0.369 | 0.287 |
| ref_last12m | 0.390 | 0.402 | 0.517 | 0.473 | 0.712 | 0.678 | 0.508 | 0.616 | 1.062 | 0.592 | 0.420 | 0.406 |
| ref_last60m | 0.496 | 0.403 | 0.513 | 0.410 | 0.531 | 0.513 | 0.566 | 0.446 | 1.362 | 0.603 | 0.401 | 0.415 |
| ref_blend | 0.470 | 0.414 | 0.380 | 0.349 | 0.517 | 0.645 | 0.562 | 0.466 | 1.070 | 0.660 | 0.374 | 0.308 |

(Per-asset rows are over each asset's own origins, so they are not comparable across assets.)

### Incremental value of the state variables (the question this family answers)

These are pooled per-(origin, asset) QLIKE cells, restricted to the origins where the two forecasts differ. The test is
a Driscoll-Kraay EWC panel DM test; skill = 1 - loss/loss_base, so positive means the state variables help.

| comparison | group | subset | from | skill | DM t | p |
|---|---|---|---|---|---|---|
| macro_state vs macro_base | equity | daily | 1926 | -5.7% | +0.83 | 0.42 |
| | credit | daily / all | 1979 / 1928 | -11.3% / +11.2% | +1.06 / -1.21 | 0.32 / 0.24 |
| | rates | daily / all | 1961 / 1927 | +10.7% / -1.1% | -0.81 / +0.11 | 0.43 / 0.91 |
| | real | daily | 1974 | -12.2% | +1.56 | 0.15 |
| | ALL | daily / all | | -3.5% / -0.6% | | 0.59 / 0.90 |
| macro_implied vs macro_base | equity | daily | 1991 | -0.4% | +0.10 | 0.93 |
| | credit | daily | 1991 | -3.2% | +2.08 | 0.09 |
| | rates | daily | 2008-12 | -4.4% | +1.56 | 0.19 |
| | ALL | daily | 1991 | -1.4% | +0.78 | 0.47 |
| macro_full vs macro_state (implied on top of macro) | equity / credit / rates | daily | 1991 / 1991 / 2008 | +3.6% / +11.7% / -11.5% | | 0.80 / 0.40 / 0.71 |
| | ALL | daily | 1991 | +5.0% | -0.53 | 0.62 |
| macro_state_ip vs macro_state (adding IP) | ALL | daily / all | | -0.6% / +0.5% | | 0.11 / 0.67 |
| macro_full_h vs macro_base_h | equity | daily | 1926 | -11.0% | +1.84 | 0.08 |
| | credit / rates / real | daily | | -5.1% / +1.1% / -11.9% | | 0.47 / 0.88 / 0.24 |
| | ALL | daily / all | | -7.2% / -7.2% | +1.53 / +1.41 | 0.14 / 0.17 |

The only consistent positive is rates in the daily era (+10% on the non-homogeneous base, from inflation vol,
policy activity and yield level), and it disappears on the homogeneous base. Implied vol adds nothing for equities.
For credit it is slightly worse (p 0.09). For rates (MOVE x duration, 61 origins) the evidence is too short to say.

### Coefficients (macro_full, standardized; posterior mean, posterior z in brackets)

At the 2013 fit: equity cs +0.04 (0.2), ts -0.08 (-0.5), ivol +0.08 (0.4), sahm -0.05 (-0.2), vix +0.05 (0.3);
credit cs +0.07, ts -0.08, ly -0.07, sahm -0.01, vix +0.03; rates ly -0.01, ivol +0.04, dpol +0.07, ts +0.01,
move_d +0.07 (0.3); real infl +0.08 (0.3), usd +0.03, iv_own -0.03. In the longer macro-only fits (2007, 692
equity months): ts -0.14 (-1.5), ivol -0.12 (-1.3), cs +0.06 (0.7); credit ts -0.24 (-1.7); rates dpol +0.12
(0.8), ivol -0.05, ly +0.07; real infl +0.09. No coefficient reaches |z| = 2 in any yearly fit.

Signs against the economics:
* These match: credit spread + for equity (weak); term spread - (a flat or inverted curve precedes higher equity and
  credit vol); policy-rate activity + for bond vol; inflation level + for gold and commodities; VIX and
  MOVE x duration + (both small).
* These do not: inflation volatility - for equity and (before 2008) for bonds; the Sahm gap - for equity and credit. The Sahm
  gap is reasonable once one notes that `v3` already holds the recession vol and vol mean-reverts after the
  Sahm trigger. The bond-vol relationship with yield level and inflation vol that we expected is not visible
  incrementally, because the base's `v_slow` already tracks the high-vol 1970s-80s.
* The IP coefficient changes sign across groups and fits (equity -0.10 in 1950, credit +0.17 in 1965). Read anything
  that leans on it with suspicion: it is latest-vintage data.

### Implied vol scale against the 21-day target

Ratio of mean implied variance (21-close average at the origin) to mean realized next-12-month variance, on the origins where both exist:

| implied | asset | from | IV^2 / next-yr var | vol ratio | corr(log IV^2, log y) | corr(log v3, log y) |
|---|---|---|---|---|---|---|
| VIX | us_large | 1986 | 1.82 | 1.35 | 0.57 | 0.50 |
| VIX | us_small / intl_dev / reits | 1986 | 1.11 / 1.43 / 1.27 | | 0.51 / 0.50 / 0.22 | 0.33 / 0.39 / 0.50 |
| MOVE x duration | ust_int / ust_long / tips | 2002 | 1.16 / 1.36 / 1.29 | 1.08 / 1.17 / 1.13 | 0.54 / 0.61 / 0.35 | 0.32 / 0.20 / 0.23 |
| GVZ | gold | 2008 | 1.91 | 1.38 | -0.04 | -0.11 |
| OVX | commodities | 2007 | 3.98 | 1.99 | 0.44 | 0.39 |

VIX overstates the 21-day realized variance of the S&P 500 by 82% on average (a 35% vol premium). That makes it a
biased level forecast, so it is only used as a centred regressor. MOVE x duration is close to scale after duration
conversion. Raw correlations with next-year variance are higher for implied vol than for `v3`. The incremental
coefficient is still small, because implied vol and the own-history terms share most of that information and the
dev sample holds only about 20 independent years of VIX.

## Run time

CPU time for a full monthly run (1116 origins to 2014) is about 0.7 to 1 minute per model. All seven
models together used about 6 CPU-minutes. Wall-clock time was 160-330 s per model, because the machine was shared at a
load of about 17. Fits are cached yearly in the panel's shared cache.

## Tried and dropped / changed

* **One-stage fit** (base slopes re-estimated jointly with the state variables on each information set's
  sample): dropped before scoring. In the 2013 fit, the gold group on 80 months gave a `v_slow` slope of -2.6 and an
  intercept of -11.7.
* **All-asset pooled fallback** when a group's base had fewer than 60 months: dropped before scoring. In 1922 it gave
  ig_credit a 16% vol forecast, because it was fitted on us_large. Replaced by the fixed geometric mean.
* **Homogeneous base (`macro_base_h`, `macro_full_h`)**: added after the first full monthly run. That run showed the
  non-homogeneous base failing in 1926-45 (QLIKE 1.27 vs 0.42 for ref_last12m): the free intercept, fitted on
  low-vol tier-D 1871-1925 data, pulls forecasts toward a fixed group level. Constraining the slopes to sum to 1 is the
  standard scale-invariance restriction, but it was chosen after seeing that result, so its edge is partly
  selected. Both versions stay registered and reported.
* **Step-6 quick run**: on the 186 step-6 origins, every macro model beat every reference. The monthly run
  reversed this on the daily subset. The difference is mostly which 1930s origins happen to be sampled, and it is
  a reminder of how noisy the comparison is.

## Caveats

* The headline differences are driven by 1926-45 (the 1929-33 vol explosion). Before 1946, ref_last12m is hard to beat.
* Pre-1926 data are tier D (month-average prices; vol understated by about 15%). The base intercept partly learns that level.
* The base slopes add up to less than 1 (e.g. 0.79 for equity), which implies strong mean reversion to a group
  level. This works after 1946 and fails in the 1930s.
* Implied vol series other than VIX/VXO and MOVE barely enter in dev: GVZ and OVX only for the 2013-14 fits.
  Their behaviour in the holdout is untested.
* Nothing in this family is significantly different from its own base. The power calculation in
  `eval.inference` says effects under about 15-20% cannot be detected, so "no detectable incremental value" is the
  honest reading, not "no value".
