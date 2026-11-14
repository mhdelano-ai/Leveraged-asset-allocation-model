# condcorr — correlations conditioned on economic state

Package `src/volmodel/models/condcorr/`, tests `tests/models/test_condcorr.py`.
All numbers below are from the development sample (`VOLMODEL_MAX_DATE=2014-12-31`,
targets to origin 2013-12); nothing after 2014 was loaded.

## The question

Can observable conditions tell us which stock–bond correlation regime the next
12 months will be in, faster and for a better reason than a slow average? The
stock–bond correlation is the one number in the matrix that has changed sign
(realized 12-month us_large–ust_int correlation: about −0.2..+0.1 in 1955–65,
+0.3..+0.7 in 1968–1996, −0.7..+0.1 from 2000). Every model in the family
leaves the rest of the matrix at a sound unconditional estimate and moves only
a block.


## Models (6 registered, kind `corr`)

| model | what moves | conditioning information |
|---|---|---|
| `condcorr_sb_base` | equity x rates | recent 12-month realized correlation only (the "baseline" for incremental value) |
| `condcorr_sb_state` | equity x rates | recent 12m corr + 63-day corr + CPI inflation (yoy) + inflation vol (36m) + long yield |
| `condcorr_sb_macro` | equity x rates | inflation, inflation vol, long yield only |
| `condcorr_msw` | equity x rates | two-state Markov switching on (us_large, ust_int), filtered probabilities |
| `condcorr_msw_infl` | equity x rates | same, transition probabilities depend on inflation |
| `condcorr_stress` | equity x HY credit, equity x REITs | recent 12m corr + equity vol ratio (63d / 10y) + Baa–Aaa spread |

### Common structure

* **Base matrix** `C0` = `ref_corr_last120m` (imported read-only from
  `volmodel.models.ref`). Chosen a priori because it is the "long run plus
  recent" estimate the brief asks us to start from; it is also the best of the
  three references on every subset.
* **Anchor pairs.** Stock–bond models move `(us_large, ust_int)` and
  `(us_large, ust_long)`; the stress model moves `(us_large, hy_credit)` and
  `(us_large, reits)`. For the stock–bond block both anchor pairs move by the
  same Fisher-z shift `dz` (the mean of the predicted shifts), because the two
  pairs are near-duplicates and different shifts would be amplified through the
  nearly collinear anchors.
* **Embedding** (`embed.py`, anchor-factor representation). With `A` the anchor
  assets, `B = C_AA^{-1} C_A` (loadings of every asset on the anchors),
  `R = C0 − C_A' C_AA^{-1} C_A` (Schur complement, PSD):
  `C1 = B' C_AA* B + R`, `C = diag(C1)^{-1/2} C1 diag(C1)^{-1/2}`, where `C_AA*`
  is the anchor block with the target correlations. Anchors hit their targets
  exactly; every other pair moves only through its loadings on the anchors
  (TIPS and IG credit inherit most of a stock–bond move, intl/EM equity most of
  it through their equity loading, gold almost none); the residual structure
  `R` is untouched. `C` is PSD by construction; if the result's minimum
  eigenvalue were below `1e-4` (or `0.5 × min eig C0`) the anchor change is
  scaled back by bisection. In a check every 6 months 1962–2014 on the real
  panel the scale-back never bound (λ = 1 at all 104 origins for
  `sb_state`, `msw_infl`, `stress`); minimum eigenvalue 0.001 (the base's
  own floor). Tests: 20 random block matrices with extreme targets (±0.99),
  random asset subsets/orders at early/mid/late origins.
* **Information-set ladder and fallback.** (1) a predictor not observable at
  the origin (series not yet started, dead, or > 190 days stale) is dropped;
  (2) if fewer than 120 training months remain, the predictor with the fewest
  observations is dropped, repeatedly; (3) with fewer than 120 training months
  even for the intercept-only model, or if the anchor assets are not both
  eligible, the model returns the base matrix unchanged. Training starts with
  windows opening in 1952 (after the March 1951 Treasury–Fed Accord ended the
  bond-yield peg), so on the real panel every model equals
  `ref_corr_last120m` before 1962 (stock–bond), and before ~1983/1991 for the
  stress pairs (HY from 1980, REITs from 1972 monthly). The 63-day correlation
  needs daily bond data (1962+) and enters the stock–bond fit from 1972. Macro
  series (1871+) never bind after 1952. On the synthetic test panel, which has
  no macro series unless injected, the models degrade to the recent-correlation
  rung.

### Regression models (`stockbond.py`, `regression.py`)

Target for training origin `s`, pair `p`:
`y_ps = z(ρ̂_p(s, s+12m]) − z(ref_p(s))`, `z` = Fisher z (|ρ| clipped at
0.98), `ρ̂` = realized correlation from the view with exactly the target's
estimator (overlapping 21-day sums when both assets pass the daily coverage
rules, else the uncentered correlation of the 12 monthly returns; verified
identical to `volmodel.eval.targets` to 1e-16), `ref_p(s)` = the base's pair
value (half last-120m, half all-history monthly correlation). The model
predicts the baseline's error, so all coefficients measure incremental value:

`y = b0 + b_rec·(z(ρ12) − z_ref) + b_sht·(z(ρ63) − z_ref) + g1·infl_yoy + g2·infl_vol36 + g3·long_yield + e`

* Pooled over the two core pairs, re-estimated each December on windows fully
  observed by then (`s + 12m ≤ Dec`), applied monthly with current predictors.
* Predictors standardized on the training sample; **all slopes ≥ 0** (economics:
  persistence; high/volatile inflation and high yields → supply/monetary shocks
  dominate → stocks and bonds fall together).
* **Shrinkage**: Gaussian prior N(0, τ²) on every standardized coefficient and
  the intercept, τ = 0.2 (a 1-sd move in a state variable is a priori unlikely
  to move z by more than ~0.4); noise scale `s2` = mean square of `y`;
  observation weights `1/(12 × n_pairs)` (twelve overlapping windows ≈ one
  independent year; two near-identical pairs ≈ one). Solved as bounded least
  squares (`scipy.optimize.lsq_linear`, BVLS).
* Stress model: same machinery, pairs (us_large, hy_credit), (us_large, reits),
  predictors recent 12m corr, `eqvol_ratio` = ½ log(63-day / 10-year variance of
  us_large 1-day returns), Baa–Aaa spread; all ≥ 0.

### Markov-switching models (`markov.py`)

Observations per month for (us_large, ust_int) (ust_long when ust_int is not
eligible): non-overlapping ~5-day sums of daily log excess returns within the
month when both have ≥ 10 daily returns (4–5 vectors), else the monthly return;
each standardized by the asset's trailing 12-month RMS monthly return (scaled to
the block length) and winsorized at ±4. Emission N(0, Σ_k) with **variances
common to both states** and a state-specific correlation ρ_k (first version had
free variances: EM then found volatility regimes, with ρ = 0.33 vs 0.15 in both
states; that version was never scored and is replaced). EM with Hamilton filter
and Kim smoother, months 1952 → last December, refit yearly, deterministic
start (ρ = +0.4 / −0.3). Priors fixed in advance: transition pseudo-counts 59
stay / 1 switch per state (mean duration 60 months, weight of 5 years); 10
pseudo-observations of an uncorrelated pair per state. TVTP variant:
`logit p_kk(t) = a_k + c_k·infl_std(t−1)`, `c_hi ≥ 0`, `c_lo ≤ 0`, N(0,1) prior
on `c`. Forecast: **filtered** probability at the origin, propagated
`π_{t+h} = π_t P^h`, h = 1..12, mixture `Σ̄ = Σ_k w̄_k Σ_k`, ρ = Σ̄₁₂/√(Σ̄₁₁Σ̄₂₂);
the Fisher-z gap to the base's us_large–ust_int value is applied to both
anchor pairs. Fits are cached by a hash of the data used, so cold and warm
calls agree exactly (tested).

## Results (dev sample; common origins; skill = % reduction in mean loss vs `ref_corr_last120m`, Fieller-EWC 90% CI; DM = HAC Diebold–Mariano p)

### Correlation MSE

| model | daily (1015 origins, 1929–2013) | stock_bond subset (625, 1961–2013) | skill stock_bond [90% CI], DM p | sbg4 (473, 1974–) | full12 (168, 2000–) |
|---|---|---|---|---|---|
| condcorr_msw_infl | **0.0429** | **0.0925** | +19.1% [−15.5, +34.1], p 0.25 | **0.0777** | **0.0773** |
| condcorr_msw | 0.0444 | 0.0990 | +13.4% [−25.4, +30.5], p 0.41 | 0.0799 | 0.0782 |
| condcorr_sb_state | 0.0455 | 0.1100 | +3.8% [−16.4, +18.4], p 0.68 | 0.0817 | 0.0872 |
| ref_corr_last120m | 0.0432 | 0.1143 | — | 0.0887 | 0.0842 |
| condcorr_sb_base | 0.0437 | 0.1155 | −1.0% [−10.0, +14.2], p 0.87 | 0.0874 | 0.0857 |
| condcorr_stress | 0.0432 | 0.1179 | −3.1% [−6.3, +3.4], p 0.30 | 0.0904 | 0.0840 |
| condcorr_sb_macro | 0.0454 | 0.1201 | −5.1% [−23.1, +11.0], p 0.56 | 0.0865 | 0.0856 |
| ref_corr_longrun | 0.0475 | 0.1430 | | 0.1024 | 0.0981 |
| ref_corr_zero | 0.5079 | 0.1700 | | 0.2589 | 0.2655 |

Daily-subset skill vs `ref_corr_last120m`: msw_infl +0.7% [−7.5, +7.7];
msw −2.8%; sb_base −1.1%; stress −0.1%; sb_state −5.3% [−14.6, +2.8]; sb_macro −5.1%.
(Before 1962 every model equals the base, and the 1926–45 era is all
us_large–us_small, which nobody changes.)

**Stock–bond pairs** (`by_pair`, MSE, 625 origins 1961–2013 except where noted):

| pair | sb_base | sb_state | sb_macro | msw | msw_infl | stress | last120m | longrun | zero |
|---|---|---|---|---|---|---|---|---|---|
| us_large~ust_int | 0.1155 | 0.1100 | 0.1201 | 0.0990 | **0.0925** | 0.1179 | 0.1143 | 0.1430 | 0.1700 |
| us_large~ust_long | 0.1021 | **0.0866** | 0.0969 | 0.0976 | 0.0906 | 0.1105 | 0.1079 | 0.1476 | 0.1763 |
| us_small~ust_int | 0.1045 | 0.1084 | 0.1150 | 0.0880 | **0.0822** | 0.1016 | 0.0986 | 0.1245 | 0.1256 |
| intl_dev~ust_int (337, 1985–) | 0.1629 | 0.1605 | 0.1609 | 0.1344 | 0.1285 | 0.1571 | 0.1528 | 0.1826 | **0.1278** |
| us_large~tips (168, 2000–) | 0.1375 | 0.1619 | 0.1314 | 0.1252 | 0.1212 | 0.1347 | 0.1302 | 0.1267 | **0.1013** |
| us_large~ig_credit (409, 1979–) | 0.0914 | 0.0976 | 0.0889 | **0.0751** | 0.0756 | 0.0898 | 0.0888 | 0.0891 | 0.1831 |

Skills on us_large~ust_long: sb_state +19.7% [−2.2, +37.7] (p 0.13), msw_infl
+16.0% [−10.8, +33.1], sb_macro +10.2% [−3.5, +25.0]. Nothing is significant
at 10%.

us_large~ust_int MSE by era (1946–79 / 1980–99 / 2000–13): last120m 0.057 /
0.104 / 0.204; sb_base 0.059 / 0.100 / 0.212; sb_state 0.072 / 0.092 / 0.185;
sb_macro 0.077 / 0.096 / 0.211; msw 0.084 / 0.089 / 0.132; msw_infl 0.071 /
0.091 / 0.122; longrun 0.064 / 0.104 / 0.301; zero 0.107 / 0.235 / 0.159.
Mean error (forecast − realized) on that pair: last120m +0.01, sb_base +0.04,
sb_state +0.10, sb_macro +0.13, msw −0.04, msw_infl −0.05.

### Covariance (`ref_blend` vols × each corr model, i.e. `Composite`), `qlike_rel`

Built as `diag(v) C diag(v)` from the stored `ref_blend` vol and corr
forecasts (identical to `Composite`), scored with `volmodel.eval.losses.cov_losses`.

| model | daily (1015) | stock_bond (625) | full12 (168, 2000–) |
|---|---|---|---|
| +condcorr_msw | **−0.2135** | 0.4192 | 0.1678 |
| +condcorr_stress | −0.2132 | 0.4391 | 0.1760 |
| +ref_corr_last120m | −0.2128 | 0.4332 | 0.1872 |
| +condcorr_msw_infl | −0.2122 | **0.4187** | 0.1713 |
| +condcorr_sb_base | −0.2119 | 0.4497 | 0.1753 |
| +condcorr_sb_state | −0.1934 | 0.5201 | **0.1559** |
| +condcorr_sb_macro | −0.1817 | 0.5425 | 0.1919 |
| +ref_corr_zero | 0.5318 | 0.4377 | 0.6178 |
| +ref_corr_longrun | 4.92 | 0.4435 | 29.07 |

Differences vs `ref_blend+ref_corr_last120m` (QLIKE, DM p): daily — msw −0.0007
(0.81), msw_infl +0.0006 (0.86), sb_state +0.019 (0.24), sb_macro +0.031 (0.10);
stock_bond — msw −0.014 (0.41), msw_infl −0.015 (0.44), sb_state +0.087
(0.10), sb_macro +0.109 (0.09); full12 — sb_state −0.031 (0.29), msw −0.019
(0.17). The regression models' covariance losses come from 1962–79, where
they predicted strongly positive stock–bond correlations (+0.4..+0.6) in
years that realized near zero; QLIKE punishes an over-confident high
correlation in a two-asset matrix hard. (`ref_corr_longrun`'s huge values come
from near-singular forecasts in a few windows; not our concern.)

### Portfolios (`volmodel.eval.portfolio.score_portfolios`, same composites, common origins)

| | p6040 qlike | p6040 var_ratio | p_balanced qlike | gmv_long real_var | erc_vt |vol−10%| |
|---|---|---|---|---|---|
| sb_macro | **0.2949** | 1.045 | **0.3497** | 0.0205 | **0.0359** |
| ref_corr_longrun | 0.3029 | 1.094 | 0.3581 | 0.0206 | 0.0379 |
| stress | 0.3098 | 1.101 | 0.3572 | 0.0205 | 0.0378 |
| msw_infl | 0.3117 | 1.137 | 0.3644 | 0.0205 | 0.0380 |
| ref_corr_last120m | 0.3120 | 1.112 | 0.3629 | 0.0205 | 0.0377 |
| msw | 0.3170 | 1.139 | 0.3691 | 0.0205 | 0.0384 |
| sb_state | 0.3196 | 1.091 | 0.3747 | 0.0205 | 0.0365 |
| sb_base | 0.3203 | 1.107 | 0.3713 | 0.0205 | 0.0380 |
| ref_corr_zero | 0.3216 | 1.204 | 1.0639 | 0.0239 | 0.0759 |

60/40 QLIKE skill vs last120m (625 origins): sb_macro +5.5% [+1.1, +14.5],
DM p 0.04 (the only interval excluding zero in this report; with six models
and several portfolios it is within what chance produces); msw_infl +0.1%;
msw −1.6%; sb_state −2.4% [−12.6, +25.7]; sb_base −2.7%. Every model
under-predicts 60/40 risk (var_ratio 1.05–1.14): the vol forecasts, not the
correlation, dominate a 60/40's risk (equity is ~90% of its variance), so the
stock–bond correlation moves 60/40 QLIKE by only a few percent. By era, 60/40
QLIKE 2000–13: longrun 0.522, zero 0.547, sb_macro 0.549, stress 0.557,
msw_infl 0.558, last120m 0.564, msw 0.572, sb_base 0.596, sb_state 0.635.
GMV realized variance is identical to 4 digits across all condcorr models.

### Stress model

Pairs it moves (MSE, skill vs last120m): us_large~hy_credit 0.0294 vs 0.0312
(+5.8% [−43, +34]); us_large~reits 0.0566 vs 0.0588 (+3.7% [−43, +36]);
us_small~hy_credit +10.3% [−2.2, +27.5]. Coefficients are small: equity-vol
ratio 0.00–0.07 per sd, spread 0.00–0.04, recent corr 0.00–0.09, intercept
+0.07..+0.13 (the base under-predicts equity–credit correlation, which rose
over the sample). Its main effect is a level correction, not stress
conditioning. It loses slightly on the stock–bond pairs (−3%) because HY
and REIT anchors drag some of the rates block with them through the loadings.

## The estimated relationship and the evidence behind it

**Coefficient path** of `sb_state` (standardized slopes, Fisher z per 1 sd,
refit each December; `n_years` = sum of weights ≈ independent years):

| Dec | n_years | b0 | recent12 | short63 | infl_yoy | infl_vol | long_yield |
|---|---|---|---|---|---|---|---|
| 1964 | 10.9 | −0.07 | 0.05 | – | 0.00 | 0.15 | 0.17 |
| 1970 | 16.9 | 0.00 | 0.03 | – | 0.00 | 0.11 | 0.19 |
| 1976 | 13.8 | 0.09 | 0.00 | 0.02 | 0.05 | 0.00 | 0.03 |
| 1986 | 23.8 | 0.16 | 0.00 | 0.02 | 0.00 | 0.02 | 0.07 |
| 1996 | 33.8 | 0.16 | 0.00 | 0.01 | 0.00 | 0.00 | 0.07 |
| 2002 | 39.8 | 0.09 | 0.00 | 0.12 | 0.01 | 0.02 | 0.07 |
| 2008 | 45.8 | 0.06 | 0.00 | 0.10 | 0.03 | 0.00 | 0.08 |
| 2014 | 51.8 | 0.02 | 0.00 | 0.12 | 0.04 | 0.00 | 0.09 |

(n_years drops in 1974 when the 63-day correlation enters and the sample is
restricted to windows where it exists.) The **long yield** is the only state
variable with a stable positive coefficient throughout (0.03–0.19 per sd; in
native units roughly +0.03..+0.05 in z per 1 pp of yield after 1980). The
inflation level contributes a little only after 2000; inflation volatility was
large in the 1960s fits and is at its zero bound since 2008. The 12-month
recent correlation adds nothing beyond the base's own recent window (its
coefficient is at zero from 1974); the 63-day correlation is the second
stable predictor (0.10–0.12 per sd since 2002).

**Full-sample (1952–2014) unpenalized OLS with Newey–West(18) t-stats**
(deviation from the baseline, us_large~ust_int; standardized): long yield
+0.13 (t 3.7), inflation vol −0.07 (t −2.1), inflation +0.04 (t 0.8); with
the recent terms, 63-day corr +0.13 (t 2.3), long yield +0.08 (t 1.8),
R² 0.26. Sub-samples: in 1952–79 every single macro predictor has t ≤ 1.1 for
ust_int (≤ 2.5 for ust_long); in 1980–2013 long yield +0.21 (t 5.4), inflation
+0.14 (t 2.4), **inflation volatility −0.10 (t −1.7) — the wrong sign**
(2008–09: oil-driven CPI swings with strongly negative stock–bond correlation).
Level regressions of realized z on the long yield alone: slope +9.4 per unit
(t 6.0, R² 0.34).

**How many episodes?** In the training sample (1952–2013) there are two and a
half: a near-zero/negative regime 1952–65 (low yields, low inflation), a
positive regime 1966–1999 (rising then falling but high yields), and the
negative regime from 2000. The yield level fell almost monotonically from 1981
to 2013 while the correlation went from +0.5 to −0.4: that is ONE slow
co-movement, and the high t-statistics come from it (NW standard errors with
18 lags cannot fix a sample with one trend). The 1950s–60s episode is
consistent with the yield story (low yields, low correlation), the 1966–75
episode only partly (yields rose in 1966; the realized correlation stayed near
zero until 1968 and dropped back to 0 in 1971–72). Effectively the evidence for
"state → correlation" rests on 2, at most 3, independent episodes. The
predictor that the theory names first (inflation volatility) gets the wrong
sign in the period with most data. I would not trust the regression
coefficients out of sample; the Markov models' advantage comes from reading
the correlation itself, quickly, not from macro state.

## How fast the switch around 2000 was recognized

Forecasts of corr(us_large, ust_int) (month-ends). Realized next-12-month
correlation: 1999-12 +0.13, 2000-03 −0.22, 2000-06 −0.61, 2000-12 −0.55.

| origin | longrun | last120m | sb_base | sb_state | sb_macro | msw | msw_infl |
|---|---|---|---|---|---|---|---|
| 1999-12 | 0.27 | 0.33 | 0.41 | 0.37 | 0.37 | 0.26 | 0.18 |
| 2000-06 | 0.28 | 0.33 | 0.41 | 0.26 | 0.37 | 0.06 | 0.04 |
| 2000-12 | 0.27 | 0.29 | 0.36 | 0.21 | 0.32 | 0.01 | −0.00 |
| 2001-12 | 0.25 | 0.19 | 0.18 | 0.17 | 0.22 | −0.02 | −0.04 |
| 2002-12 | 0.20 | 0.07 | −0.07 | −0.39 | 0.05 | −0.05 | −0.08 |

First forecast below zero after mid-1999: msw_infl 2000-12, msw 2001-12,
sb_state 2002-05, sb_base 2002-12, ref_corr_last120m 2006-12,
ref_corr_longrun never; sb_macro never stays below zero until 2007. The Markov
filter dropped the probability of the positive regime from 0.66 (Dec 1999) to
0.13 (Jun 2000) and below 0.01 by Dec 2000, i.e. **about 6 months** after the
switch, versus **about 6–7 years** for the 10-year/long-run blend. The cost is
false alarms: the same filter flipped to the low-correlation regime in late
1987 (crash; probability 0.03 in Dec 1987) and in mid-1998 (LTCM;
0.001 in Dec 1998), forecasting ≈0 when the next 12 months realized +0.65 and
+0.36. The neg-regime correlation was estimated near 0 until 2002 (the 1950s
regime), so the first years of the new regime were forecast at ≈0 rather than
−0.4; by 2014 the regimes are +0.40 / −0.30.

## Where the models lose

* 1946–1979 on the stock–bond pair: all conditional models are worse than the
  base (msw 0.084, sb_state 0.072, base 0.057). The Markov states were not
  separated until ~1970 and the regressions' early fits (10–15 years of data)
  extrapolated the 1950s–60s yield relationship to predict +0.4..+0.6 in
  1966–72, when realized correlations were ≈0. This is what makes `sb_state`
  and `sb_macro` lose on covariance QLIKE.
* The regressions are biased high on the stock–bond pair (+0.10, +0.13 mean
  error) — the intercept and the slow fall of yields keep them positive into
  the 2000s.
* Markov false alarms in 1987 and 1998 (above).
* Other pairs that inherit the move through loadings: TIPS (2000–13) and
  intl_dev–ust_int are best forecast by **zero**; every stock–bond model hurts
  `us_large~tips` (sb_state −24%).
* `ref_corr_zero` beats every model on the stock–bond subset in 2000–13
  (MSE 0.159; only the Markov models, 0.122–0.132, beat zero there).

## Tried and dropped (all reported)

* Uniform Fisher-z shift of the whole equity × rates block, PD by bisection on
  the shift (first embedding): needed scale-back at 2008 and 2014 (λ ≈ 0.78–0.83)
  because IG credit, highly correlated with Treasuries, did not move with them.
  Replaced by the anchor-factor embedding before any scoring. Step-6
  scores of that version (stock_bond MSE): sb_state 0.109, sb_base 0.116,
  sb_macro 0.117 (essentially the same as now).
* Markov model with state-specific variances: states became volatility
  regimes (ρ 0.33 / 0.15); replaced on design grounds before scoring.
* Not attempted: gold / commodity blocks (item 4 of the brief was conditional
  on 1–3 being solid; the regression evidence is not), Student-t emissions to
  stop the 1987/1998 false alarms (would be tuning to two known events), a
  combination of the Markov and regression forecasts.

## Run time (full monthly, 1116 origins to 2014-12, on a machine with load ≈ 17 on 4 cores)

sb_base 187 s, sb_macro 197 s, sb_state 434 s, stress 176 s, msw 275 s,
msw_infl 465 s (≈ 60–400 ms per origin; Markov fits are yearly and cached).
All under the 10-minute limit even under contention.

## Tests

`tests/models/test_condcorr.py`, 51 tests, ~20 s: every model valid (finite,
symmetric, unit diagonal, |ρ| ≤ 1, PD) at an early (monthly-only assets), a
middle and a late origin, also on random asset subsets and orders; the
conditional part is active at the late origin; cold = warm in any order;
altering data and predictors after `t` leaves the forecast at `t` unchanged;
fallback equals `ref_corr_last120m` before the training minimum; realized
series equal `compute_targets` exactly; predictors respect publication dates;
PD of the embedding for 20 random block matrices with extreme targets;
embedding leaves unrelated blocks untouched; ridge sign constraint and
shrinkage; EM recovers two simulated correlation regimes.
`tests/test_contract.py` passes.

## Caveats

* No condcorr model is significantly better than `ref_corr_last120m` on any
  correlation subset (best: `msw_infl`, +19% on the stock–bond subset, CI
  [−16%, +34%]). Realized 12-month correlations have SE ≈ 0.2; with two and a
  half regimes, the dev sample cannot establish the claim.
* Macro series are latest vintage (CPI is not revised; fine).
* `condcorr_sb_*` and `condcorr_stress` are cheap; the Markov models cache fits
  in a module-level dict keyed by a hash of the data (bounded at 512 entries).
* The holdout (2015+) contains the 2021–23 inflation episode in which the
  stock–bond correlation turned positive again: the decisive out-of-sample test
  of whether `msw`/`msw_infl` (fast, data-driven) or the yield/inflation
  regressions (slow, state-driven) recognize a switch in the other direction.

