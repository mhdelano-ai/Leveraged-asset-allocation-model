# Final evaluation: development period and holdout

Scope: every stored model (70 volatility, 27 correlation, 9 covariance; 88 single models plus 16
combinations and 2 registered pairs) re-run over the full, corrected history into `results/full/`, and
scored once against the targets. The four covariance candidates and their roles were fixed in
`reports/SELECTION.md` before any post-2013 origin was scored. Nothing here re-selects anything.

Reproduce (about 2.5 minutes on one core):

```
export OMP_NUM_THREADS=1 VOLMODEL_RESULTS=$PWD/results/full      # no VOLMODEL_MAX_DATE
python scripts/score_final.py                                     # writes results/final/
```

**Periods** are defined by forecast origin. The **development period** is origins up to 2013-12, whose
windows end by 2014-12. The **holdout** is origins from 2014-01 to 2025-10: 142 overlapping 12-month
windows, about 12 independent years, and the last origin with a fully observed window. The two periods
are reported separately everywhere. Within every table, models are compared on identical cells (the
origin × asset or origin × pair cells where every model in that table has a forecast and a target), and
the table states how many.

**Inference** uses the `volmodel.eval.inference` defaults:
- Diebold-Mariano tests with the EWC variance and a t(B) reference;
- Fieller 90% skill intervals;
- the Götze-Künsch Model Confidence Set.

On the holdout, B is clipped at its floor of 4, so every holdout interval uses t(4) quantiles and is
wide by construction. "Skill" is the percentage reduction in mean loss; positive means the first-named
model is better.

## The answer

**Yes, the selected model held up out of sample, but the holdout shows much less than the development
period did.**

- **Volatility.** The primary (`combo_cov_main`, whose volatilities are `combo_vol_hier_trim`) had the
  lowest volatility QLIKE of the four candidates on the holdout: 0.353, against 0.383 for the reference.
  - It beat both pre-named references: +7.8% versus `ref_blend` [90% CI +2.6, +29.0], p = 0.03, and
    +35.9% versus `ref_last12m` [+17.8, +57.4], p = 0.06.
  - Its margin over `ref_blend`, the best reference on the holdout, shrank from +31.7% in development to
    +7.8%. The reference improved a lot in the calmer holdout; the primary improved less.
- **Covariance.** On the covariance loss the primary was ahead of the reference by about a third of its
  development margin, and that is not significant on the holdout: `qlike_rel` −0.623 against −0.502,
  difference −0.121 [−0.322, +0.079], p = 0.27.
- **Runner-up.** The primary was ahead of the simple runner-up on volatility (+10.6% [−7.6, +30.1],
  p = 0.18) and tied it on covariance (−0.623 against −0.623).
- **Best single models.** The best-single pairing (`pair_cov_single`, built on `statespace_2c_daily`)
  led development (QLIKE 0.360 against 0.389 for the primary). On the holdout it was the worst of the
  four candidates on volatility (0.415, worse than the reference).
- **Model Confidence Set.** The holdout cannot rank the four: the 90% MCS keeps all four on volatility and
  on covariance loss, as expected.

| daily subset | dev QLIKE (1,051 origins, 5,890 cells) | holdout QLIKE (142, 1,704) | primary skill, dev [90% CI], p | primary skill, holdout [90% CI], p |
|---|---|---|---|---|
| **primary** `combo_cov_main` | 0.389 | **0.353** | — | — |
| runner-up `combo_cov_simple` | 0.402 | 0.395 | +3.2 [−16.3, +21.0], 0.76 | +10.6 [−7.6, +30.1], 0.18 |
| best single `pair_cov_single` | **0.360** | 0.415 | −7.8 [−21.9, +7.3], 0.36 | +15.0 [unbounded, +23.7], 0.36 |
| reference `pair_cov_ref` (= `ref_blend` vols) | 0.569 | 0.383 | +31.7 [+19.2, +38.6], 0.012 | +7.8 [+2.6, +29.0], 0.034 |
| `ref_last12m` | 0.505 | 0.550 | +23.0 [−13.5, +46.5], 0.24 | +35.9 [+17.8, +57.4], 0.059 |

| covariance `qlike_rel`, daily | dev (1,015 origins) | holdout (142) | primary − model, dev [90% CI], p | primary − model, holdout [90% CI], p |
|---|---|---|---|---|
| primary | **−0.524** | **−0.623** | — | — |
| runner-up | −0.492 | −0.623 | −0.032 [−0.125, +0.060], 0.55 | −0.001 [−0.073, +0.071], 0.99 |
| best single | −0.489 | −0.575 | −0.036 [−0.076, +0.005], 0.15 | −0.048 [−0.123, +0.027], 0.24 |
| reference | −0.208 | −0.502 | −0.316 [−0.472, −0.160], 0.002 | −0.121 [−0.322, +0.079], 0.27 |

**What is statistically supported.**
- The primary beats the reference matrix's volatilities on both periods.
- On the development period it beats the reference covariance (p = 0.002) and the reference correlation.
- Fixed-weight portfolio risk is not systematically mis-predicted in either period.

**What is not supported.**
- Any ranking among the primary, the runner-up and the best single model, in either period.
- A covariance-loss advantage over the reference on the holdout.
- Most per-asset holdout comparisons; their Fieller intervals are often unbounded.

## The five pre-registered expectations

| # | expectation (SELECTION.md) | outcome | numbers (holdout unless stated) |
|---|---|---|---|
| 1 | primary vol QLIKE below the best reference; a reversal larger than the development gap would mean overfitting | **met** | vs `ref_blend` (the best reference on the holdout) 0.353 vs 0.383, +7.8% [+2.6, +29.0], p = 0.03; vs `ref_last12m` 0.353 vs 0.550, +35.9% [+17.8, +57.4], p = 0.06. No reversal. The gap over `ref_blend` narrowed from +31.7% (dev) to +7.8%. |
| 2 | primary `qlike_rel` below `pair_cov_ref` | **met in point estimate, not significant** | −0.623 vs −0.502, difference −0.121 [−0.322, +0.079], p = 0.27 (dev −0.316, p = 0.002) |
| 3 | fixed-weight portfolios: realized/predicted variance averages in [0.75, 1.35] | **met** (point estimates; intervals far wider than the band) | 60/40 1.05 [0.40, 1.70]; balanced 1.16 [0.30, 2.02]; risk parity 1.25 [0.44, 2.06]; random long-only 1.19 [0.36, 2.01]. Dev 0.97 / 1.06 / 1.10 / 1.03. |
| 4 | minimum-variance portfolios realize more variance than predicted, factor > 1.2 | **not met** | unconstrained GMV 0.83 [0.10, 1.57] (dev 1.43); long-only GMV 1.08 [0.29, 1.87] (dev 1.27). The development-period under-prediction did not appear on the holdout; the intervals include both 0.8 and 1.4. |
| 5 | with the bridge, Var(z) of policy portfolios nearer 1 than without | **not met in point estimate; inconclusive** | holdout Var(z), identity → `vr_market`: 60/40 0.59 → 0.48; balanced 0.67 → 0.55; equity mix 0.55 → 0.45. Both bridges are below 1 because the holdout years were calm relative to the forecasts, and the bridge only raises predicted annual variance. The non-overlapping χ² interval for 60/40 with `vr_market` is [0.26, 1.22] from about 12 years. In development the bridge moved balanced (1.14 → 0.93) and equity (1.19 → 0.97) nearer 1, and 60/40 slightly away (1.09 → 0.88). |

## Verification (done before trusting any table)

**Independent recomputation.** Short code in `score_final.py` (`kday_var`, `_independent_window`,
`qlike_var`, `qlike_rel_independent`) reads `panel.daily` and the cash rate directly. It forms log excess
returns and computes:
- overlapping 21-day sums with missing days as 0, annualized by the window's row count;
- coverage by the 90% / 10-per-month rule;
- the monthly proxy where coverage fails.

It does not call `targets` or `losses`. Results (`verification.csv`):

| check | cases | max relative difference |
|---|---|---|
| realized vol, 4 assets × 12 origins (1931-06 ... 2021-12) | 44 | < 3e-11 |
| proxy-vs-daily classification | 44 | identical |
| QLIKE of the primary from the stored parquet | 43 | < 3e-11 |
| stored cov diagonal = `combo_vol_hier_trim` vol | 43 | 0 |
| us_large–ust_int correlation (1975, 2008, 2021) | 3 | < 3e-11 |
| `qlike_rel` of the primary (1990, 9 assets; 2016, 12 assets) | 2 | < 3e-11 |

- The 12 origins are 1931-06, 1938-12, 1949-06, 1962-12, 1974-06, 1987-09, 1999-12, 2008-08, 2013-12,
  2016-06, 2020-02 and 2021-12. Three are pre-1952: their windows hold 301, 300 and 283 rows (Saturday
  trading). Three are holdout origins.
- The independent proxy decision agreed with the stored flag in every case.

**Coverage** (`coverage.csv`).
- All 106 stored models have all 1,258 expected origins (1922-01 ... 2026-10, the live origin).
- At every origin their asset set equals the harness eligibility (36 monthly returns).
- There are no non-finite values.
- Each model has all 142 holdout origins, and every one is scorable.

**Look-ahead.**
- `scripts/leak_test.py`'s `check()` was re-run without `VOLMODEL_MAX_DATE`, with holdout-era origins
  added through a scratch wrapper (the script has no CLI option for origins). The origins were 1951-03,
  2008-09, 2013-12, 2016-06, 2020-02, 2022-06 and 2025-10.
- Models covered: the four candidates; their components `combo_vol_hier_trim`, `combo_vol_cal_m_hl12`,
  `combo_corr_ew`, `statespace_2c_daily`, `mcorr_nls`, `ref_blend`, `ref_corr_last120m` and
  `bench_corr_d_hl36`; and `ml_glm` and `har_pooled_logcvx_d`.
- Result: 14 models × 7 origins, 98 checks, 0 failing, maximum relative difference 0.
- Separately, the stored forecasts of the four candidates at 2016-06, 2020-02, 2022-06 and 2025-10 are
  bit-identical to forecasts computed from `load_panel(max_date=origin)`.
- Not covered: the other 50+ vol components inside the combination were leak-tested only on development
  origins (`reports/models/leak_test_dev.csv`).

**Sanity of holdout targets** (realized 21-day-horizon vol):
- us_large: 33.5% in calendar 2020, against 7.3% in 2017.
- ust_long: 20.9% in 2022, against 7.1% in 2017.
- ust_int: 10.9% in 2022.
- hy_credit: 19% for the windows containing March 2020.

These are as expected.

**Pipeline check.** The script reproduces the committed development numbers of `reports/models/combo.md`
to the third decimal: `combo_vol_hier_trim` 0.388/0.389 (daily), `combo_cov_main` −0.524, and
`combo_corr_ew` +4.5–4.7% versus `ref_corr_last120m`. The `summarize()` means equal the script's own means
(max difference 3e-15).

## 1. Volatility leaderboard (all 70 models; `vol_leaderboard.csv`, `vol_leaderboard_asset.csv`)

Daily subset, common cells.

- **Development:** 1,051 origins, 5,890 cells. The best reference is `ref_last12m` (0.505).
- **Holdout:** 142 origins, 1,704 cells. The best reference is `ref_blend` (0.383).

| rank | development | QLIKE | holdout | QLIKE (skill vs `ref_blend` [CI], p) |
|---|---|---|---|---|
| 1 | statespace_2c_daily | 0.360 | garch_d_gjr | 0.318 (+17.0 [−21.5, +34.2], 0.22) |
| 2 | ml_glm | 0.364 | garch_d_garch11 | 0.322 (+15.8 [−11.9, +36.3], 0.18) |
| 3 | ml_avg | 0.377 | bench_vol_m_hl120 | 0.329 (+13.9 [−86, +33.6], 0.48) |
| 4 | combo_vol_ew | 0.383 | regime_ms2_weekly | 0.339 |
| 5 | combo_vol_hier | 0.387 | structural_normal | 0.348 |
| 7 | combo_vol_hier_trim (primary) | 0.389 | combo_vol_hier_trim (primary, rank 7) | 0.353 (+7.8 [+2.6, +29.0], 0.03) |

**The development ranking did not carry over.**
- The Spearman correlation between the 70 models' development and holdout ranks is **−0.26**.
- Several holdout leaders were near the bottom in development: `bench_vol_m_hl120`, a 10-year EWMA, was
  68th of 70, and `structural_normal` was 67th.
- The development leaders fell back on the holdout: `ml_glm` was 43rd (0.390) and `statespace_2c_daily`
  was 56th (0.415).
- `garch_d_rm2006`, the base of the "cheap" calibrated combo, was third from last on the holdout (0.541;
  realized/forecast variance 1.48).

The family medians on the holdout are combo 0.357, regime 0.358, macro 0.368, garch 0.373, bench 0.379,
ml 0.382, statespace 0.391, har 0.403, ref 0.432 and structural 0.453.

Only 4 of the 70 models beat `ref_blend` at p < 0.10 on the holdout: the trimmed family mean (p = 0.03),
`regime_bocpd_eb` (p = 0.03), `combo_vol_hier` (p = 0.05) and `combo_vol_ew` (p = 0.06). 37 of the 70 have a positive point estimate. This is the pattern that motivated choosing a
combination: with no ex-ante identifiable winner, the combination was 7th of 70 on the holdout and among
the leaders in development.

**Level calibration** (mean realized/forecast variance; 1 = QLIKE-calibrated):
- primary: 1.11 in development (1.50 in 1926–45, 0.98 in 1946–79, 0.87 in 1980–99, 1.25 in 2000–13) and
  1.05 on the holdout;
- reference: 1.37 and 1.17;
- `ref_last12m`: 1.54 and 1.59.

## 2. Correlation and covariance leaderboards (`corr_leaderboard.csv`, `cov_leaderboard.csv`)

**Correlation MSE**, daily pairs, common cells: development 1,015 origins and 21,830 pair-cells; holdout
142 origins and 9,372 pair-cells.
- **Development.** `combo_corr_ew` is +4.5% versus `ref_corr_last120m` [+2.0, +6.5], p = 0.014. That is
  still the only significant gain apart from `combo_corr_hier` (+3.3%, p = 0.035) and `bench_corr_blend`
  (+5.0% [+0.1, +8.9], p = 0.10).
- **Holdout.** `combo_corr_ew` is +1.2% [−4.2, +7.8], p = 0.68, rank 8 of 27.
  - The holdout leaders are the regime-switching stock–bond models: `condcorr_msw_infl` +8.0%
    [−17.9, +25.4], then `bench_corr_blend` and `condcorr_msw`.
  - No correlation model beats the reference significantly on the holdout.
  - `bench_corr_d_hl36` (the runner-up's correlation) is 20th (−6.5%).

**Covariance `qlike_rel`**, daily set: development 1,015 origins; holdout 142.
- In development every combination and `pair_cov_single` beat `pair_cov_ref` at p ≤ 0.01.
- On the holdout none does: the primary's p is 0.27, and the best p among all nine models is 0.14.
- Ranks on the holdout: `combo_cov_main` and `combo_cov_simple` −0.623, `combo_cov_hier` −0.612,
  `combo_cov_cal` −0.580, `pair_cov_single` −0.575, `pair_cov_ref` −0.502, then the three structural
  factor models (−0.36 to −0.48).
- `frob_scaled` separates nothing in either period (all p > 0.12).

## 3. The four candidates head to head (`h2h_*.csv`, `mcs.csv`)

Volatility QLIKE by era (daily subset, development) and on the holdout:

| model | 1926–45 (235) | 1946–79 (408) | 1980–99 (240) | 2000–13 (168) | holdout (142) |
|---|---|---|---|---|---|
| primary | 0.637 | **0.338** | **0.213** | 0.415 | **0.353** |
| runner-up | 0.532 | 0.348 | 0.238 | 0.584 | 0.395 |
| best single | 0.534 | 0.351 | 0.253 | **0.294** | 0.415 |
| reference | 0.958 | 0.472 | 0.319 | 0.620 | 0.383 |
| `ref_last12m` | **0.418** | 0.599 | 0.313 | 0.671 | 0.550 |

On the like-for-like `full12` subset (all 12 assets, origins 1997+), 2000–13 against the holdout:
- primary 0.415 → 0.353;
- best single 0.294 → 0.415;
- reference 0.620 → 0.383.

The holdout rewarded slower forecasts.

Covariance `qlike_rel` by era (development):
- primary −0.656 / −0.478 / −0.621 / −0.342;
- reference −0.191 / −0.255 / −0.424 / +0.197.

The primary beats the reference at p ≤ 0.005 in 1946–79 and 1980–99. The other eras are not significant.

**Holdout, per asset**, primary versus reference (`h2h_vol_asset.csv`):
- The primary is clearly ahead on ust_long (QLIKE 0.17 vs 0.29), ig_credit (0.59 vs 0.82) and gold
  (0.14 vs 0.18).
- It is slightly behind on reits, em, intl_dev and hy_credit.
- No per-asset difference is significant. Twelve independent years per asset are too few, and several
  Fieller intervals are unbounded.
- In development the primary beats the reference on every asset (significantly on us_small, ust_long,
  ig_credit, gold and commodities).

**Model Confidence Set** (90%, Götze-Künsch, block 36):

| loss | development | holdout |
|---|---|---|
| vol QLIKE, daily | keeps all 4 (reference p = 0.11) | keeps all 4 |
| cov `qlike_rel`, daily | keeps 3; **excludes the reference** (p = 0.094) | keeps all 4 |

As expected, the holdout MCS keeps everything. In development the MCS excluded the reference matrix on
covariance loss.

**Implied correlations of the candidates** (`h2h_corr.csv`, holdout MSE):
- primary 0.0746, reference 0.0755, best single 0.0771, runner-up 0.0804;
- no difference is significant;
- the bias is −0.04 for the primary: forecast correlations were too low on average, mainly from stock–bond.

## 4. Portfolios (`portfolio_summary.csv`, `portfolio_dm.csv`; mean with HAC 90% interval)

Realized/predicted variance (1 = unbiased):

| portfolio | primary dev | primary holdout | reference dev | reference holdout | runner-up holdout | best single holdout |
|---|---|---|---|---|---|---|
| 60/40 | 0.97 | 1.05 [0.40, 1.70] | 1.11 | 1.14 | 1.13 | 1.17 |
| balanced | 1.06 | 1.16 [0.30, 2.02] | 1.19 | 1.23 | 1.16 | 1.27 |
| equity mix | 1.11 | 1.08 [0.29, 1.86] | 1.29 | 1.14 | 1.08 | 1.21 |
| risk parity (ERC) | 1.10 | 1.25 [0.44, 2.06] | 1.36 | 1.34 | 1.27 | 1.34 |
| inverse-vol | 1.10 | 1.29 [0.35, 2.22] | 1.38 | 1.38 | 1.27 | 1.40 |
| equal weight | 1.04 | 1.21 [0.31, 2.10] | 1.17 | 1.24 | 1.20 | 1.29 |
| random long-only (mean of 20) | 1.03 | 1.19 [0.36, 2.01] | 1.19 | 1.23 | 1.18 | 1.27 |
| GMV (unconstrained) | 1.43 | 0.83 [0.10, 1.57] | 2.43 | 1.04 | 1.18 | 1.06 |
| GMV (long-only) | 1.27 | 1.08 [0.29, 1.87] | 1.78 | 1.18 | 1.31 | 1.33 |

**Portfolio QLIKE.**
- Holdout, primary vs reference: 60/40 0.328 vs 0.341 (p = 0.53); balanced 0.426 vs 0.444 (p = 0.34);
  random 0.419 vs 0.428 (p = 0.73).
- Development, the same comparisons: 0.241 vs 0.313 (p = 0.10); 0.283 vs 0.363 (p = 0.12); 0.343 vs 0.468
  (p = 0.03).

**Minimum-variance realized variance** (Engle–Colacito), primary vs reference:
- Holdout: GMV 0.0012 vs 0.0014, p = 0.27 (about 3.5% vs 3.7% vol); long-only 0.0024 vs 0.0025.
- Development: no difference (p ≥ 0.14).

**10%-vol target error**, |realized − 10%|: primary 3.2 pp (dev) and 3.3 pp (holdout); reference 3.8 and
3.5 pp.

**Turnover**, one-way, per month:
- primary: GMV 6.6% → 8.6% (dev → holdout); long-only GMV 3.0% → 6.4%; ERC 1.2%;
- reference: GMV 3.4% → 5.2%;
- best single: GMV 12% → 19%.

## 5. Bridge calibration of the primary (`bridge_calibration.csv`, `bridge_tails.csv`)

z is the 12-month log excess return divided by its predicted annual sigma, sample-demeaned within each
period. Intervals:
- var_z: 24-month moving-block bootstrap;
- non-overlapping variance: χ² with the degrees of freedom of one calendar offset;
- tail shares: circular 24-month block bootstrap of origins.

The tail parameter s² is walk-forward. Its median is 0.80 overall and 0.59 at holdout origins.

Var(z), identity → `vr_market`:

| | development | holdout (about 12 independent years) |
|---|---|---|
| 60/40 | 1.09 [0.82, 1.43] → 0.88 [0.66, 1.14] | 0.59 [0.27, 1.20] → 0.48 [0.21, 0.97] |
| balanced | 1.14 → 0.93 | 0.67 → 0.55 |
| equity mix | 1.19 → 0.97 | 0.55 → 0.45 |
| random long-only, pooled | 1.35 → 1.17 | 0.84 → 0.71 |
| single assets (range) | 0.99–1.58 → 0.89–1.44 | 0.43 (us_large)–1.64 (commodities) → 0.35–1.22 |

On the holdout, Var(z) differs by asset class:
- Below 1 for equities and credit spreads. Steady equity years and the post-2020 calm meant realized
  annual returns varied less than predicted.
- Above 1 for ust_int (1.34), ust_long (1.44), ig_credit (1.54) and commodities (1.64). 2022 and
  2021–22 commodities are the cause.

Tail shares, `vr_market` (expected under the mixture in brackets after the observed share and its
interval):

| | development, assets pooled | holdout, assets pooled | holdout, policy portfolios pooled |
|---|---|---|---|
| beyond ±1σ | 0.31 [0.27, 0.34] (0.24) | 0.21 [0.13, 0.29] (0.26) | 0.10 [0.03, 0.20] (0.26) |
| beyond ±2σ | 0.063 [0.044, 0.087] (0.054) | 0.031 [0.009, 0.057] (0.053) | 0.014 [0, 0.038] (0.053) |
| below mixture 5% | 0.073 [0.050, 0.100] (0.05) | 0.033 [0.005, 0.072] (0.05) | 0.035 [0, 0.092] (0.05) |
| below mixture 1% | 0.020 [0.009, 0.034] (0.01) | 0.008 [0, 0.021] (0.01) | 0 [0, 0] (0.01) |

What twelve years can show. The holdout contains about twelve independent annual returns per portfolio.
- The χ² interval for a variance estimated from 12 observations runs from roughly 0.5× to 2× the
  estimate.
- One 1%-quantile breach is expected every 100 years.
- So the holdout cannot confirm or reject the bridge or the tail mixture. It can only rule out gross
  failure, and there was none: no policy portfolio fell below its 1% quantile.

In development the asset tails were somewhat heavier than the mixture predicts: 7.3% below the 5% quantile
and 2.0% below the 1% quantile. For policy portfolios they were close (6.2% / 1.6%).

## 6. What the data correction changed (`data_correction.csv`)

Pre-fix forecasts with pre-fix targets (`results/dev/`) were compared with post-fix forecasts and targets
(`results/full/`), on the same development origins.

- **Vol forecasts** moved by 1.5% on average in log terms. The model averages range from 0.7% to 3.9%; the
  largest single cell moved 54% (`statespace_factor_daily`).
- **Development QLIKE** changed by a median of +0.07% (daily) and +0.5% (all), with a range of −3.0% to
  +3.4%. The rank correlation of the 70 models before and after is 0.99 (daily) and 0.996 (all).
  - Largest moves: `garch_d_*` worsened by 2–3.4% (`garch_d_rm2006` dropped from rank 14 to 21 on daily);
    `statespace_2c` improved by 3%.
- **Candidates.**
  - `statespace_2c_daily` is still first on daily (0.358 → 0.360).
  - `combo_vol_hier_trim` moved from 6th to 7th on daily (0.388 → 0.389) and from 2nd to 1st on all
    (0.393 → 0.396).
  - The skill of the leaders against the references changed by less than 1 point.
- **Correlation MSE** changed by less than 0.6%. `combo_corr_ew` versus the reference went from +4.7% to
  +4.5%; it is still significant.
- **Covariance `qlike_rel`** changed by less than 0.002 (`combo_cov_main` −0.5242 → −0.5241).

No family conclusion in `reports/models/*.md` changes. The correction mattered for the integrity of the
holdout (finding 1 of the audit), not for the scores.

## 7. Holdout episodes (`episodes_vol.csv`, `episodes_worst.csv`, `episodes_corr.csv`)

Realized vol divided by the primary's forecast at December origins (calendar-year windows):

| window | us_large | us_small | intl_dev | em | ust_int | ust_long | tips | ig_credit | hy_credit | reits | gold | commod. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2017 (calm) | 0.56 | 0.52 | 0.54 | 0.54 | 0.51 | 0.56 | 0.48 | 0.49 | 0.40 | 0.41 | 0.53 | 0.57 |
| 2019 | 0.75 | 0.69 | 0.74 | 0.86 | 0.96 | 1.19 | 0.80 | 1.13 | 0.65 | 0.64 | 0.83 | 0.60 |
| 2020 | 2.42 | 2.62 | 2.48 | 1.73 | 1.06 | 1.30 | 1.78 | 2.96 | 3.02 | 2.86 | 1.13 | 1.81 |
| Mar 2020–Feb 2021 (origin 2020-02) | 1.40 | 1.67 | 1.39 | 1.19 | 0.59 | 0.92 | 1.43 | 2.21 | 2.00 | 1.72 | 1.04 | 1.18 |
| 2021 | 0.56 | 0.54 | 0.43 | 0.54 | 0.87 | 1.04 | 0.83 | 0.74 | 0.29 | 0.69 | 0.76 | 0.95 |
| 2022 | 1.36 | 1.12 | 1.50 | 1.19 | 1.92 | 1.75 | 1.95 | 2.27 | 1.71 | 1.45 | 0.96 | 1.43 |
| 2024 (calm) | 0.62 | 0.61 | 0.60 | 0.66 | 0.71 | 0.70 | 0.55 | 0.59 | 0.35 | 0.64 | 0.85 | 0.68 |

For example, us_large in 2020: the forecast was 13.8%, the realized vol 33.5%. For ust_long in 2022: the
forecast was 12.0%, the realized vol 20.9%.

- **Calm years.** Realized vol was about half the forecast in every asset. That is the cost of a forecast
  calibrated to mean variance.
- **2020.** The crisis landed 2–3× above forecast in equities and credit, and the forecast was too high in
  2021 after it.
- **2022.** Rates and credit vol ran 1.7–2.3× the forecast.
- **Worst misses** were all in credit and all around 2020: hy_credit realized 19% against a 6.3% forecast
  for the 2020 window, then 2.9% against 10.4% for the window from 2020-11. ig_credit realized 16.6%
  against 5.6%.
- **Share of holdout windows with realized vol above 1.5× the forecast:** 2% (gold) to 14% (ig_credit).

**Stock–bond correlation** (us_large with ust_int; ust_long is similar). Every candidate missed the sign
change.

| origin | realized | primary | runner-up | best single | reference |
|---|---|---|---|---|---|
| 2019-12 | −0.48 | −0.17 | −0.30 | −0.13 | −0.13 |
| 2020-12 | +0.08 | −0.18 | −0.33 | −0.15 | −0.12 |
| 2021-12 | **+0.46** | −0.14 | −0.29 | −0.15 | −0.09 |
| 2022-06 | +0.50 | −0.04 | −0.19 | −0.11 | −0.03 |
| 2022-12 | +0.39 | +0.06 | +0.01 | −0.03 | +0.04 |
| 2023-06 | +0.65 | +0.05 | +0.01 | −0.01 | +0.07 |
| 2023-12 | −0.04 | +0.11 | +0.09 | +0.04 | +0.10 |

- The primary crossed zero only at the end of 2022, a year after the realized correlation turned positive.
- The runner-up (3-year EW correlation) stayed most negative.
- None of the four forecast a positive correlation above +0.11 during 2021–23.
- On the `stock_bond` subset the primary's holdout correlation MSE is 0.087, against 0.096 for the
  reference (p = 0.38).

## 8. Forecast accuracy a planner can use (`accuracy.csv`)

The tables use log(realized vol / forecast vol) for the primary, where the realized value is the 12-month
21-day-horizon vol.

| sample | cells | mean | sd | central two-thirds of realized/forecast | central 90% |
|---|---|---|---|---|---|
| full sample (1922–2025 origins) | 9,913 | −0.15 | 0.40 | **−40% to +22%** | −53% to +73% |
| last 30 years (1995-11 to 2025-10) | 4,270 | −0.11 | 0.37 | **−35% to +22%** | −46% to +79% |
| holdout | 1,704 | −0.15 | 0.38 | −39% to +19% | −49% to +78% |

Plain statements:
- Two years in three, realized vol over the next 12 months landed between about two-thirds of the
  forecast and a fifth above it.
- One year in twenty it was more than 1.7–1.8× the forecast.
- The distribution is skewed. The typical (median) year comes in about 10–15% below the forecast, but the
  forecast is right on average in VARIANCE (realized/forecast variance ≈ 1.05–1.11). The bad years more
  than make up the gap.

**Noise split.** Part of that dispersion is measurement noise in the realized 12-month vol itself. By
simulation of iid normal days, the standard deviation of log realized vol is 0.17 for a 252-day window
(0.16 for 295 days) and 0.21 for the monthly proxy. Subtracting that variance:
- The forecast error proper has sd ≈ 0.33 (last 30 years; 0.36 on the full sample).
- Noise accounts for about 21% of the variance of the log ratio.
- If the error were lognormal, the two-thirds band for the TRUE vol around the forecast would be about
  −35% to +24% (last 30 years).

The noise figure assumes conditionally normal days. Fat tails and within-year vol changes raise it to
about 0.18–0.19 (`targets` docstring), which would lower the forecast-error sd slightly.

By asset (last 30 years, two-thirds band):
- tightest: ust_long −25% to +28% and ust_int −34% to +11%;
- widest: hy_credit −51% to +31% and tips −40% to +41%.

## Surprises and caveats

1. **The development ranking of single volatility models reversed on the holdout** (Spearman −0.26).
   - The best development single model was the worst of the four candidates on the holdout.
   - A 10-year EWMA ranked 68th of 70 in development and 3rd on the holdout.
   - This is what the selection rationale anticipated, but it also means the development scoreboard says
     little about which single model to trust.
2. **The reference got much better out of sample** (0.569 → 0.383; on `full12`, 0.620 → 0.383). The
   primary still beat it, by a quarter of the development margin.
3. **The minimum-variance under-prediction vanished** on the holdout (GMV 0.83). In development it was
   1.43 (2.40 in 2000–13), and the pre-registered expectation of > 1.2 was not met. The interval is
   [0.10, 1.57], so this is not evidence that the optimizer problem is solved.
4. **The bridge moved Var(z) further from 1** for every policy portfolio on the holdout. Var(z) without
   the bridge was already 0.55–0.67, because the holdout was calmer than its forecasts. Twelve years
   cannot separate the two bridges.
5. **Stock–bond correlation** turned positive in 2021–23, and no candidate forecast it within a year.
6. **The holdout is short.** EWC degrees of freedom are 4, and many per-asset Fieller intervals are
   unbounded. Treat every holdout p-value between 0.03 and 0.10 as weak.
7. **The pairs duplicate their component.** `pair_cov_ref`'s vols are exactly `ref_blend`, so their
   volatility rows coincide.

## Files in `results/final/`

| file | contents |
|---|---|
| `coverage.csv` | per stored model: origins, first/last, missing/extra origins, asset-set mismatches, non-finite count, holdout origins |
| `vol_leaderboard.csv` | 70 vol models × period × era × subset (daily, all, stock_bond, sbg4, core9, full12): QLIKE, R/F, n origins/cells, skill vs best reference [90% Fieller], DM p, rank |
| `vol_leaderboard_asset.csv` | 70 vol models × asset × period: QLIKE, R/F, n proxy, skill vs best reference, DM p |
| `corr_leaderboard.csv` | 27 corr models × period × era × subset: pairwise MSE, bias, skill vs best reference, DM p |
| `cov_leaderboard.csv` | 9 cov models × period × era × subset: `qlike_rel`, `frob_scaled`, difference vs `pair_cov_ref` [90% CI], DM p, Frobenius skill |
| `h2h_vol.csv` | four candidates + `ref_last12m`, `ref_blend`: QLIKE and R/F by subset, period, era |
| `h2h_vol_pairwise.csv` | all pairs of the six: skill [CI], DM p by subset, period, era |
| `h2h_vol_asset.csv` | the six per asset and period, with the primary's skill against each |
| `h2h_cov_pairwise.csv` | the four candidates pairwise: `qlike_rel` difference [CI], p; Frobenius skill |
| `h2h_corr.csv` | the candidates' implied correlations: MSE, bias, skill vs the reference |
| `mcs.csv` | 90% MCS for the candidates, vol QLIKE and cov `qlike_rel`, development and holdout |
| `portfolio_summary.csv` | four candidates × portfolio × metric × period/era: mean with HAC 90% interval |
| `portfolio_dm.csv` | primary vs each candidate: portfolio QLIKE, GMV realized variance, abs log variance ratio |
| `bridge_calibration.csv` | `calibration_summary` for identity and `vr_market` per asset and portfolio, by period |
| `bridge_tails.csv` | tail shares (±1σ, ±2σ, below mixture 5% / 1%) with block-bootstrap intervals and expected values |
| `expectations.csv` | the five pre-registered expectations, status and evidence |
| `episodes_vol.csv` | per asset at holdout December origins and 2020-02: realized vs each candidate, ratios |
| `episodes_worst.csv` | worst holdout miss per asset, top 15 overall, share of windows with realized above 1.5× forecast |
| `episodes_corr.csv` | us_large–ust_int / ust_long forecasts vs realized, 2018-12 to 2023-12 |
| `dashboard_vol.csv` | per asset per origin (1922-01 to 2026-10, including live): realized vol, proxy flag, four candidates, 10th–90th percentile of the nine family means |
| `dashboard_corr.csv` | six key pairs per origin: realized and the four candidates' correlations |
| `dashboard_portfolio.csv` | 60/40 and balanced per origin: realized vol and each candidate's predicted vol |
| `accuracy.csv` | log(realized/forecast vol) of the primary by asset and sample, with quantiles, bands and noise split |
| `data_correction.csv` | development scores pre vs post fix per model (vol, corr, cov), rank changes, forecast movement |
| `verification.csv` | independent recomputation vs stored targets and losses |
| `summary.json` | run metadata, coverage summary, sanity values, noise estimates, s² range, expectation statuses, headline numbers |
