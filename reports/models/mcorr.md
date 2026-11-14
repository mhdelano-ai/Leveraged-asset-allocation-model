# mcorr — unconditional-structure estimators of the correlation matrix

Family package `src/volmodel/models/mcorr/` (`core.py` estimators, `dcc.py`
corrected DCC, `__init__.py` models and registry), tests
`tests/models/test_mcorr.py`. Eight `corr` models are registered. All
results below are development results (`VOLMODEL_MAX_DATE=2014-12-31`,
origins 1922-01 .. 2014-12, scored origins 1929-06 .. 2013-12); nothing
after 2014 was looked at.

## The problem and the common design

At a 12-month horizon the estimation error of a correlation matrix is the
dominant problem: a pair's realized 21-day correlation has a standard error
of about 0.2, an asset enters the forecast set with only 36 months of
history, and histories range from 1871 (us_large) to 1997 (tips). Every
model here is a classical remedy for estimation error. To make the
comparisons between them controlled, all share one data convention, fixed
before any scoring:

* **Input: monthly (calendar-month) log excess returns** (`view.excess_monthly`).
  A calendar-month return is one non-overlapping ~21-trading-day sum, i.e. the
  target's own horizon; it is immune to the asynchronous-close and stale-price
  biases that distort 1-day correlations by up to several tenths
  (reports/data/horizon_evidence.md), and it exists in every era (the 1920s
  have monthly data only). Daily data are not used at all.
* **Weighting: exponential, half-life 120 months** (`EW_HALFLIFE`). A round
  10 years, the window of `ref_corr_last120m`; chosen a priori. The
  equal-weight alternative is reported as a sensitivity (it is worse; it was
  not used to choose).
* **Uncentered second moments**, as in the target and the references.
* **Effective sample sizes in independent months**: Kish's
  `n = (Σw)² / Σw²` over the months a pair shares (≈ 346 months for a long
  pair under the 10-year half-life, 36 for an asset that just became eligible).
  Never days.
* Every output is symmetric, unit-diagonal and positive definite for any
  requested asset list, in any order, at every origin (the earliest scored
  origins have 2-5 assets, monthly only).

## Models

Notation: `X` is the T×N matrix of monthly returns (NaN before an asset
starts), `w_t = 2^{-(T-t)/120}`.

**`mcorr_pairwise`** — the unequal-history *control* (its stored forecasts
turn out identical in score to `bench_corr_m_hl120`; it is kept because it
is the control arm of the Stambaugh comparison). Each pair on its own
common sample: `r_ij = Σ w x_i x_j / sqrt(Σ w x_i² · Σ w x_j²)` (sums over
months where both exist; < 24 common months → 0), then the PD repair of the
references (eigenvalues clipped at 1e-3, rescaled to unit diagonal). This is
`ref_corr_longrun` with the 10-year exponential weights.

**`mcorr_stambaugh`** — maximum likelihood under monotone missingness
(Anderson 1957; Stambaugh 1997), weighted, uncentered. Assets are ordered by
first observation. The first asset's second moment is its full-sample
weighted mean square. Each further asset `k` is regressed by WLS (no
intercept) on all assets before it over its own sample:
`β_k = (Z'WZ)⁻¹Z'Wy`, `s²_k = Σw e²/Σw`; then
`Σ[k,prev] = β_k' Σ[prev,prev]`, `Σ[k,k] = s²_k + β_k' Σ[prev,prev] β_k`.
Long-history assets keep their full-history moments; each shorter asset is
attached by regression on everything already in. PD by construction.

**`mcorr_block`** — block equicorrelation of `mcorr_stambaugh`: every
off-diagonal entry replaced by the mean of its block-pair cell, blocks from
`volmodel.universe` groups (equity 4, rates 3, credit 2, real 3). This equals
the average of the matrix over all within-block relabelings, hence PD. Ten
numbers instead of 66.

**`mcorr_shrink_block`** — `mcorr_stambaugh` shrunk toward `mcorr_block`
with a pair-specific empirical-Bayes (Ledoit-Wolf / James-Stein) intensity:

    v_ij    = (1 − r_ij²)² / n_ij                       (normal-theory sampling variance)
    τ²      = max(0, mean_ij[(r_ij − T_ij)² − v_ij])     (true dispersion around the target)
    δ_ij    = v_ij / (v_ij + τ²)
    r*_ij   = T_ij + (1 − δ_ij)(r_ij − T_ij)

`n_ij` = Kish size of the pair's common sample. A pair of 1926 assets is
barely shrunk; a pair involving an asset with 3 years of data is shrunk
hard. PD repair only if needed (rare). Everything is estimated from the view.

**`mcorr_nls`** — Ledoit & Wolf (2020) analytical nonlinear shrinkage of the
eigenvalues of `mcorr_stambaugh` (their kernel formula, p ≤ n branch), with
`n` = harmonic mean of the pair effective sizes (the `n` whose `1/n` is the
average sampling-variance factor across pairs), then rescaled to unit
diagonal.

**`mcorr_fa3`** — 3-factor model of `mcorr_stambaugh` by iterated
principal-axis factoring (principal components of `R` with communalities on
the diagonal, iterated to a fixed point; Heywood guard 0.995):
`R_fa = LL' + diag(1 − h)`. The number of factors is cut to the Ledermann
bound (k with `(p−k)² ≥ p+k`), so with 2 assets the model returns `R` and
with 3 assets it fits one factor. Three factors = equity, rates,
real/inflation, fixed a priori.

**`mcorr_factor2`** — observed factors: equity = `us_large` and rates =
`ust_long` (the two longest monthly histories, 1871 and 1919). Factor second
moments by monotone ML; every other asset regressed (WLS, its own sample) on
both factors; `Σ = B Σ_f B' + D`, a factor asset has loading 1 on itself and
no residual. Fills every pair from the factors and is PD by construction.
Falls back to `mcorr_stambaugh` if no factor has 36 months.

**`mcorr_dcc`** — corrected DCC (Aielli 2013) on monthly returns.
Standardization by a slow predictable vol: RiskMetrics monthly EWMA,
`h_t = 0.97 h_{t−1} + 0.03 r²_{t−1}`, `e_t = r_t/√h_t`. Recursion

    Q_{t+1} = (1 − a − b) S_{t+1} + a (Q*_t^{1/2} e_t)(Q*_t^{1/2} e_t)' + b Q_t

with correlation targeting: `S` = `mcorr_shrink_block` on the standardized
returns, recomputed every December from data through that month and held for
the next year (a slowly moving long-run level, as in DCC-MIDAS), so only
`(a, b)` describe the dynamics. Estimation: composite likelihood summed over
all pairs (Pakel, Shephard, Sheppard and Engle), each pair on its own sample
from the January after both assets have 36 months — unequal histories never
force truncation to 1997. `(a, b)` are re-estimated once a year on all data
through the latest December (Nelder-Mead from a fixed start, so a cold call
gives the same answer); with < 120 pair-months, or no likelihood gain over
`a = b = 0`, the model is static. Forecast: the average expected correlation
over the 12 months. Since `E_t Q_{t+h} = S + (a+b)^{h−1}(Q_{t+1} − S)`,

    C = (1 − κ) S_T + κ R_{t+1},     κ = mean_{h=1..12} (a+b)^{h−1}

a convex combination of two PD matrices (`S_T` = the target with data
through the origin; `R_{t+1}` = next month's conditional correlation,
repaired if pairs started at different dates made it indefinite).

Hyperparameters (all fixed in advance): half-life 120 months (round, = the
reference window); 24 months minimum pair overlap (as the references); PD
floor 1e-3 (as the references); 3 factors (equity/rates/real); factor assets
us_large and ust_long (longest histories); EWMA 0.97 (RiskMetrics monthly);
36-month activation (= eligibility); 120 pair-months minimum for dynamics
(10 years); horizon 12 months (the target). Nothing was changed after
scoring except one bug fix in `mcorr_dcc` (below).

## How to read the results

Correlation loss: `mse` of the forecast correlation against the realized
21-day correlation, mean over the pairs with daily-based targets at each
origin (only `us_large`–`us_small` before 1962). Covariance: each
correlation forecast is paired with the stored `ref_blend` vols,
`H = diag(v) C diag(v)` — identical to `Composite(ref_blend, corr)` — and
scored by `qlike_rel` and by `volmodel.eval.portfolio.score_portfolios`
(minimum-variance realized variance, Engle-Colacito; and risk-prediction
QLIKE of the 20 random long-only portfolios per origin). Only correlations
differ between the `cb_*` composites. Common origins only (1015 for the full
sample). Skill = % reduction of mean loss vs the best reference
(`ref_corr_last120m` in every view), 90% Fieller-EWC interval; DM = EWC
Diebold-Mariano p-value (two-sided). Script and full output:
scratch `evaluate.py` / `sub.py` / `history.py` (numbers reproduced below).

## Results

### Correlation loss (daily subset, 1015 common origins 1929-06..2013-12)

| model | mse | bias | 1929–45 | 1946–79 | 1980–99 | 2000–13 | skill vs last120m [90% CI] | DM p |
|---|---|---|---|---|---|---|---|---|
| mcorr_stambaugh | 0.0430 | −0.006 | 0.0046 | 0.0280 | 0.0700 | 0.0865 | +0.4 [−4.5, 4.1] | 0.88 |
| mcorr_pairwise | 0.0430 | +0.001 | 0.0019 | 0.0279 | 0.0724 | 0.0867 | +0.3 [−2.2, 2.6] | 0.82 |
| mcorr_fa3 | 0.0431 | −0.007 | 0.0047 | 0.0281 | 0.0703 | 0.0864 | +0.1 [−4.7, 3.8] | 0.96 |
| **ref_corr_last120m** | 0.0432 | −0.003 | 0.0019 | 0.0269 | 0.0764 | 0.0842 | — | — |
| mcorr_shrink_block | 0.0432 | −0.004 | 0.0046 | 0.0275 | 0.0712 | 0.0870 | −0.0 [−5.0, 4.0] | 0.99 |
| mcorr_nls | 0.0440 | −0.018 | 0.0074 | 0.0292 | 0.0703 | 0.0860 | −2.0 [−9.7, 3.5] | 0.56 |
| mcorr_factor2 | 0.0461 | −0.027 | 0.0047 | 0.0276 | 0.0702 | 0.1056 | −6.7 [−16.9, 2.1] | 0.20 |
| mcorr_dcc | 0.0466 | −0.016 | 0.0231 | 0.0291 | 0.0703 | 0.0830 | −8.0 [−35.3, 13.2] | 0.54 |
| ref_corr_longrun | 0.0475 | −0.004 | 0.0020 | 0.0296 | 0.0805 | 0.0981 | | |
| mcorr_block | 0.0544 | −0.006 | 0.0046 | 0.0287 | 0.0986 | 0.1127 | −26.0 [−37.6, −14.0] | 0.006 |
| ref_corr_zero | 0.5079 | −0.556 | 0.9024 | 0.5552 | 0.2702 | 0.2655 | | |

Origins from 1946 (816) and from 1980 (408), where the matrices are larger:

| model | mse ≥1946 | skill [90% CI] | mse ≥1980 | skill [90% CI] |
|---|---|---|---|---|
| mcorr_dcc | 0.0523 | +1.7 [−4.0, 6.0] | 0.0755 | +5.1 [−0.6, 9.8] |
| mcorr_stambaugh | 0.0524 | +1.6 [−2.3, 5.1] | 0.0768 | +3.6 [−1.4, 8.4] |
| mcorr_nls | 0.0530 | +0.5 [−4.5, 4.7] | 0.0768 | +3.6 [−1.5, 8.0] |
| mcorr_fa3 | 0.0525 | +1.4 [−2.4, 4.8] | 0.0769 | +3.4 [−1.5, 8.1] |
| mcorr_shrink_block | 0.0526 | +1.2 [−2.9, 5.1] | 0.0777 | +2.4 [−2.3, 7.4] |
| mcorr_pairwise | 0.0531 | +0.3 [−2.2, 2.7] | 0.0783 | +1.7 [−1.4, 5.0] |
| ref_corr_last120m | 0.0532 | — | 0.0796 | — |
| mcorr_factor2 | 0.0562 | −5.5 [−15.1, 3.4] | 0.0848 | −6.5 [−20.7, 6.0] |
| mcorr_block | 0.0665 | −25.0 [−36.3, −12.8] | 0.1044 | −31.1 [−45.3, −17.1] |

Named subsets (like-for-like eras; mse): stock_bond (us_large, ust_int;
625 origins from 1961-12): **mcorr_dcc 0.1115**, ref_last120m 0.1143,
fa3 0.1164, block 0.1176, pairwise 0.1186, shrink_block 0.1204,
stambaugh 0.1208, nls 0.1218, ref_longrun 0.1430; in 2000–13 alone dcc
0.172 vs ref_last120m 0.204. sbg4 (473 from 1974-08): **dcc 0.0845**,
ref_last120m 0.0887, others 0.089–0.091, block 0.106. core9 (326 from 1986-11):
**dcc 0.0813**, nls 0.0858, fa3 0.0869, stambaugh 0.0870, ref_last120m
0.0880, pairwise 0.0881, shrink_block 0.0883, factor2 0.0919, block 0.124.
full12 (168, 2000–13): **dcc 0.0830**, ref_last120m 0.0842, nls 0.0860,
fa3/stambaugh 0.0865, pairwise 0.0867, shrink_block 0.0870, factor2 0.106,
block 0.113.

### Covariance (ref_blend vols + each correlation), qlike_rel, daily subset

| model | qlike_rel | 1929–45 | 1946–79 | 1980–99 | 2000–13 | diff vs cb_last120m | DM p |
|---|---|---|---|---|---|---|---|
| mcorr_nls | −0.301 | −0.329 | −0.295 | −0.521 | +0.030 | −0.089 | 0.029 |
| mcorr_dcc | −0.298 | −0.268 | −0.275 | −0.527 | −0.063 | −0.086 | 0.015 |
| mcorr_shrink_block | −0.295 | −0.323 | −0.292 | −0.519 | +0.051 | −0.082 | 0.027 |
| mcorr_stambaugh | −0.280 | −0.323 | −0.286 | −0.506 | +0.105 | −0.068 | 0.038 |
| mcorr_fa3 | −0.266 | −0.320 | −0.284 | −0.498 | +0.172 | −0.053 | 0.11 |
| mcorr_block | −0.250 | −0.323 | −0.286 | −0.345 | +0.059 | −0.037 | 0.48 |
| mcorr_factor2 | −0.246 | −0.321 | −0.287 | −0.453 | +0.235 | −0.033 | 0.34 |
| **ref_corr_last120m** | −0.213 | −0.190 | −0.256 | −0.438 | +0.187 | — | — |
| mcorr_pairwise | −0.104 | −0.168 | −0.252 | −0.504 | +0.904 | +0.109 | 0.23 |
| ref_corr_zero | +0.532 | +0.847 | +0.471 | +0.313 | +0.618 | | |
| ref_corr_longrun | +4.922 | −0.190 | −0.242 | +1.035 | +29.07 | | |

(`qlike_rel` can be negative, so a ratio skill is not defined; differences
and DM tests are reported. From 1980: dcc −0.336 vs ref −0.181, p = 0.047;
nls −0.294, p = 0.078; shrink_block −0.285, p = 0.093.) On the full12
subset (2000–13) the best are the shrinkage variants (see dropped variants):
lw_common −0.089, nls +0.030, shrink_block +0.051, vs ref +0.187.

### Portfolios (ref_blend vols + each correlation)

GMV realized variance (`w'Sw`, lower is better; ×10⁴) and risk-prediction
QLIKE of 20 random long-only portfolios:

| model | GMV var all | 1926–45 | 1946–79 | 1980–99 | 2000–13 | GMV skill ≥1946 [90%] | GMV skill ≥1980 [90%] | random-port QLIKE | skill ≥1946 [90%] |
|---|---|---|---|---|---|---|---|---|---|
| ref_corr_last120m | 154 | 501 | 103 | 27 | 52 | — | — | 0.4672 | — |
| mcorr_pairwise | 157 | 500 | 104 | 23 | 75 | −5.8 [−16.3, 4.0] | −19.5 [−47.6, 44.9] | 0.4694 | −0.8 [−1.6, 0.9] |
| mcorr_dcc | 158 | 537 | 102 | 23 | 40 | +5.8 [+0.8, 12.0] | +19.2 [+0.4, 24.6] | 0.4847 | −2.3 [−5.2, 3.9] |
| mcorr_shrink_block | 159 | 533 | 103 | 22 | 49 | +2.4 [−0.4, 5.5] | +9.1 [−2.0, 17.3] | 0.4666 | +1.2 [−2.1, 3.2] |
| mcorr_stambaugh | 159 | 533 | 103 | 23 | 47 | +2.8 [−0.5, 6.5] | +10.6 [−5.4, 17.0] | 0.4679 | +0.8 [−1.3, 2.0] |
| mcorr_block | 159 | 533 | 103 | 23 | 46 | +2.9 [−3.1, 8.6] | +12.6 [−24.4, 23.7] | 0.4682 | +0.6 [−3.7, 4.3] |
| mcorr_nls | 162 | 550 | 103 | 23 | 45 | +3.4 [−0.4, 7.4] | +12.5 [−4.5, 16.7] | 0.4742 | −0.5 [−1.7, 1.7] |
| mcorr_fa3 | 161 | 534 | 103 | 24 | 59 | −1.2 [−6.1, 3.6] | −3.8 [−21.4, 33.4] | 0.4684 | +0.6 [−1.5, 1.8] |
| mcorr_factor2 | 161 | 534 | 103 | 23 | 57 | −0.1 [−3.9, 3.9] | −0.1 [−16.0, 31.0] | 0.4946 | −8.0 [−17.6, 16.4] |
| ref_corr_longrun | 197 | 501 | 106 | 72 | 235 | | | 0.4682 | |
| ref_corr_zero | 239 | 901 | 107 | 51 | 45 | | | 0.8997 | |

Full-sample GMV skills vs ref_corr_last120m are all small and negative
(−2 to −5%, every CI covers 0) because 1926–45, where the realized GMV
variance is 5–20 times larger than later and only two to five assets exist,
dominates the mean; that era is the Stambaugh failure described below.
Long-only GMV realized variance is identical to 3 decimals for every model
(0.0205–0.0206) — the long-only constraint removes nearly all of the
difference. Predicted-vs-realized GMV variance ratio (mean RV/PV): 1.9–2.1
for the shrinkage models, 2.4 for ref_corr_last120m, 3.5 for mcorr_pairwise,
60 for ref_corr_longrun: every model under-predicts the risk of its own
minimum-variance portfolio, shrinkage least. Fixed policies (60/40,
balanced, equity): all correlation models within ±2% of each other except
dcc and factor2 (3–5% worse); 60/40 risk is best predicted by
ref_corr_longrun (0.303) and worst by dcc (0.326). Turnover of GMV weights
(one-way per month): factor2 1.9%, block 2.2%, shrink_block 2.6%,
stambaugh 2.9%, last120m 3.4%, dcc 4.1%, pairwise 4.2%, longrun 9.2%.

## Findings

**1. Little survives as correlation skill; much survives as covariance
skill.** On the correlation MSE every sensible estimator ties
`ref_corr_last120m` (skills within ±2%, intervals ±4%): with ~0.2 noise per
realized pair, a 12-month correlation target can rank only gross
differences (equal weights vs 10-year weights: −10%; block structure: −26%).
Where correlations are *used* — a covariance matrix for optimizers — the ML
and shrinkage estimators are clearly better: `qlike_rel` improves by
0.07–0.09 (DM p 0.015–0.04) over the best reference, mostly because they are
better conditioned (ML is PD by construction; the reference's pairwise
estimate + eigenvalue clipping produces near-singular matrices after 1997,
see `mcorr_pairwise` and `ref_corr_longrun` in 2000–13).

**2. Unequal histories: monotone ML beats pairwise + repair — except where
the long history is not homogeneous.** Controlled comparison
(`mcorr_stambaugh` vs `mcorr_pairwise`, same weights, same data):
* covariance `qlike_rel` −0.280 vs −0.104 (2000–13: +0.105 vs +0.904); GMV
  variance 2000–13 47 vs 75 ×10⁻⁴; GMV turnover 2.9% vs 4.2%;
* correlation MSE tied overall; by age of the younger asset in the pair:
  72–120 months 0.0767 vs 0.0833, 120–240 months 0.0778 vs 0.0825 (DM p
  0.21 and 0.14), > 240 months 0.0818 vs 0.0812;
* but in 1929–45 (36–72 months bin, `us_large`–`us_small` only) ML is
  significantly worse (0.0507 vs 0.0482 in that bin, p = 0.03; era MSE
  0.0046 vs 0.0019). Cause, verified: ML attaches `us_small` (1926) to
  `us_large`'s full history back to 1871, whose pre-1926 segment is
  Shiller's monthly-average-price series with 11% vol (vs 20–33% after
  1926). The regression implies `Σ_SS = s² + β²Σ_LL` with the low
  long-sample `Σ_LL`, so the implied correlation drops (1929-12: 0.83 ML vs
  0.905 pairwise; dropping the pre-1926 months restores 0.905 exactly).
  Monotone ML assumes the long-history assets' second moments are stable
  over the extra history; a spliced, smoothed early segment violates that.
  After 1946 the 10-year weights make this negligible.

**3. Structure: shrinking toward blocks helps a little; imposing blocks or
two observed factors hurts.** Pure block equicorrelation (`mcorr_block`) is
26% worse than the reference on correlations (the "real" block mixes REITs,
gold and commodities; credit pairs differ) yet still better than the
reference on `qlike_rel` and GMV variance — structure buys conditioning.
Two observed factors (`mcorr_factor2`) lose 6–7% on correlations and lose
badly after 2000 (misses intl–em, ig–hy, gold–commodities co-movement
beyond the two factors; bias −0.027). Three statistical factors
(`mcorr_fa3`) are equal to the ML matrix on correlations but worse in
covariance use after 2000. The best use of structure is as a shrinkage
target: `mcorr_shrink_block` (average intensity δ ≈ 0.19 in 1985–2000, 0.08 in 2014) and nonlinear
eigenvalue shrinkage `mcorr_nls` give the best `qlike_rel` among static
models (−0.295, −0.301) at no cost in correlation MSE.

**4. How much dynamics survive to 12 months.** The monthly cDCC estimates
are stable after 1960: `a ≈ 0.025–0.032`, `a + b` rising from 0.86 (1960s)
to 0.94 (1980s) and 0.97–0.99 (2000s). The 12-month averaging factor is
`κ = mean (a+b)^{h−1}` = 0.50 (1960s), 0.60–0.75 (1980s), 0.85–0.93
(2000s): at a monthly sampling the conditional deviation from the slowly
moving target is persistent, and most of it survives the 12-month average.
The deviation itself is modest: the forecast differs from its own static
target (`mcorr_shrink_block`) by 0.03 on average per pair in 1950–80,
0.05 in the 1990s, 0.08 in the 2000s (it caught the stock–bond sign change
early: `us_large`–`ust_int` −0.37 at 2002-12 vs +0.08 for the static
target). This is the best model on every like-for-like subset (stock_bond,
sbg4, core9, full12) and from 1980 (correlation skill +5.1% [−0.6, 9.8],
`qlike_rel` −0.336 vs −0.181, p = 0.047; GMV variance skill +19% [+0.4,
24.6]). It loses in 1929–45 (era MSE 0.023 vs 0.002): with only 3 pairs and
3–6 years, the 1927–30 estimates hit the persistence bound (`a + b` = 0.995)
and the forecast extrapolated a short-lived deviation; 1932–55 estimates
are `a + b` ≈ 0.48 (κ ≈ 0.16, practically static). It also predicts fixed
policy-portfolio risk (60/40, balanced) 3–5% worse than the static models.

## Variants tried and dropped (scores on the same 1015 origins)

| variant | corr mse | cov qlike_rel | why dropped / what it shows |
|---|---|---|---|
| `mcorr_dcc` v1: forecast `S_T + κ(R_{t+1} − S_used)` | 0.0467 | **+0.008** (2000–13: +1.72) | BUG FIX, not tuning: with `S_T ≠ S_used` the sum is not PD; the 1e-8 repair floor produced near-singular matrices. Replaced by the convex combination above (corr mse 0.0466, `qlike_rel` −0.298). The only post-scoring change. |
| RMT eigenvalue clipping of `mcorr_stambaugh` (Laloux et al.), same `n` | 0.1066 (bias −0.17) | −0.072 | Pushes all correlations toward the mean; with p ≤ 12 almost every eigenvalue is below the Marchenko-Pastur edge. NLS chosen a priori instead. |
| Ledoit-Wolf common intensity toward blocks | 0.0431 | **−0.317** (best) | Registered instead: pair-specific intensity (a priori, for unequal histories). The common intensity is better on qlike_rel (−0.317 vs −0.295) and 2000–13 (−0.089 vs +0.051); corr mse equal. |
| `mcorr_shrink_block`, equal weights (no half-life) | 0.0506 | −0.287 | Equal weights lose 17% on correlations (ignore the drift in stock–bond correlation) but give the best random-portfolio QLIKE (0.460) and balanced-policy QLIKE. |
| `mcorr_stambaugh`, equal weights | 0.0503 | −0.277 | as above |
| `mcorr_pairwise`, equal weights | 0.0475 | +4.92 | = `ref_corr_longrun` exactly (check of the implementation). |

## Run times (full monthly run 1922–2014, 1116 origins, one core)

Model time per full run (machine shared by ~12 agents; lightly loaded /
loaded): pairwise 2.7 / 5.0 s, stambaugh 3.6 / 8.2 s, block 5.2 / 12.6 s,
shrink_block 5.6 / 11.1 s, nls 4.5 / 8.0 s, fa3 9.7 / 17.9 s, factor2
6.3 / 12.0 s, dcc ≈ 110–150 s CPU (210–365 s wall under load; yearly
re-estimation of `(a, b)` ≈ 1–2 s, monthly filtering ≈ 0.1 s). All well
under the 10-minute budget. A
cold call at a 2014 origin: dcc ≈ 2 s, others < 10 ms.

## Tests

`tests/models/test_mcorr.py` (34 tests, ~25 s, synthetic panel): every
registered model returns a valid matrix (symmetric, unit diagonal, PD,
|r| ≤ 1, exact asset order) at an early monthly-only origin with 3 assets,
a middle and a late origin, and for an arbitrary asset subset without the
factor assets; the same origin gives identical output cold, warm and in any
order; altering all data after the origin leaves the forecast unchanged;
unit tests of monotone ML (equals the weighted sample matrix without
missing data; recovers the truth with staggered starts; column-order
invariant), block projection PD, shrinkage intensity ordering by sample
size, nonlinear shrinkage, factor recovery and Ledermann bound, the
observed-factor model, and the DCC κ formula and static limit.

## Caveats

* Correlation targets exist only where both assets have daily data; before
  1962 that is `us_large`–`us_small` alone, so the 1929–45 era is a 2×2
  problem and dominates portfolio means.
* Monthly calendar returns of smoothed NAV sources (high yield before 2007,
  early REIT and TIPS funds) still understate co-movement somewhat at one
  month; no correction is applied (the target shares part of it).
* The normal-theory sampling variance `(1 − r²)²/n` understates the
  variance under fat tails; shrinkage intensities are therefore on the low
  side.
* `mcorr_shrink_block` flaw found after scoring (left unchanged, to avoid
  tuning on the scoreboard): τ² is averaged over ALL pairs, including pairs
  that are alone in their block cell (where the target equals the estimate,
  deviation 0). This biases τ² toward 0 and the intensity toward 1; with 5–6
  assets (mostly singleton cells) τ² is often 0 (e.g. at 1935-12 and
  1960-12) and the model then equals `mcorr_block`; in 1929–45 both equal
  `mcorr_stambaugh` on the only scored pair. τ² should be estimated over
  pairs in multi-pair cells only.
* Possible improvement not tried (would be a new a-priori variant): monotone
  ML on returns standardized by each era's own volatility, which would
  remove most of the pre-1926 distortion in finding 2.
