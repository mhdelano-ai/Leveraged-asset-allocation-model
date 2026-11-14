# Modeling protocol

How models are developed and judged in this package, what is frozen, and
what may still count as evidence. `ARCHITECTURE.md` defines the targets, the
data and the interfaces.

## Read first

1. `ARCHITECTURE.md`: the targets, the two views of the panel, the predictor
   store, the two evaluation engines, the results layout.
2. `src/assetforecast/models/base.py` (covariance models) or
   `src/assetforecast/returns/models/base.py` (return-side forecasters): the
   interfaces and the rules every model obeys.
3. `reports/SELECTION.md`: the covariance model choice, committed before its
   holdout was scored.
4. `reports/data/panel.md`, `reports/data/horizon_evidence.md`: what is in the
   panel and why 1-day volatility cannot be trusted.
5. The module docstrings of `eval/targets.py`, `eval/losses.py`,
   `eval/inference.py`, and of `returns/config.py`, `returns/run.py`,
   `returns/evaluation/walkforward.py`, `returns/evaluation/inference.py`.

## Where the evidence stands: both holdouts are spent

| side | developed on | frozen | holdout (scored once) | record |
|---|---|---|---|---|
| covariance | outcomes known by 2014-12-31 (origins to 2013-12) | the model choice in `reports/SELECTION.md`, committed before the scoring script existed | origins 2014-01 to 2025-10 | `results/final/`, `reports/final_evaluation.md` |
| returns (`assetpred`) | outcomes known by the end of 2014 | the model lists, test families and scoring rules at `assetpred` commit `7577b13` | origins 2015-01 onward (to 2025-10 at 12 months), at 1, 3 and 12 months | `results/archive/assetpred/` (`dashboard.html`, `scores/frozen.*`, `strategy/`) |

Both were scored once and both are spent: every month up to the merge has now
been seen. Development continues on data through 2014
(`ASSETFORECAST_MAX_DATE=2014-12-31`) so that new work is comparable with the
recorded development results, but nothing in 2015-2026 is a holdout any more.
A comparison on those years describes a model; it does not test it.

During `assetpred`'s development 75 configurations were tried
(`results/archive/assetpred/dev/trials_{mean,dist,cov}.csv`: 21, 15 and 39).
Its adjusted tests account for the frozen models, not for all 75.

### The replication and `REPLICATION_LABEL`

The six frozen mean models (`ridge_z, combo_z, gbm_z, rf_z, ens_z,
gbm_z.price`), the robustness rows (`gbm_z.nohind, combo_z.nohind`) and the
baselines were re-run unchanged at 12 months on the package's 12-asset return
panel (`python -m assetforecast.returns.run`). The strategy test was re-run
with the package covariance as the risk model. Every output of these runs,
and of the distribution scores, carries

> `REPLICATION_LABEL` = "Replication on the 12-asset panel, frozen models
> unchanged; both holdouts already spent, so not out-of-sample evidence"

The periods in `returns.config.PERIODS` (`dev`, `holdout` = the covariance
side's 2014-01..2025-10, `holdout_ap` = `assetpred`'s 2015-01..2025-10, decades)
are labels for comparing with the original results row by row, not
out-of-sample samples. The expected-return interval and the distribution
model `dist_hm_af` were designed after both holdouts had been seen; their
scores are descriptive too.

### `FROZEN_AT`: what can still be evidence

`returns.config.FROZEN_AT = "2026-11-14"` is the date the return models were
merged into this package and frozen again. **For the return side, only
forecasts whose whole 12-month target window lies after `FROZEN_AT` count as
out-of-sample evidence**: origins from `OOS_ORIGIN_START` (2026-11-30) on,
period `oos` in `config.PERIODS`. The first such outcome is known at the end of
November 2027; for years, the evidence will be thin. The published forecast
at origin 2026-10-31 is not evidence: its window began before `FROZEN_AT`.
The covariance model was chosen on the same day; the same reasoning applies
to its forecasts.

## What may and may not change in the frozen mean models

Frozen (do not change; if a change seems necessary, report it instead):

- the model definitions and their registered settings
  (`returns/models/ml_mean.py`, `returns/models/baselines.py`), the model
  lists (`BASELINES`, `FROZEN_MEAN`, `ROBUSTNESS`) and the test families
  (`SPA_FAMILIES`);
- the feature definitions (`returns/features/`) and the pinned feature list
  (`FROZEN_FEATURE_COLUMNS`);
- the values in `returns/config.py` that set what a model computes or how it
  is judged: `QUANTILES`, `OOS_START`, `REFIT_EVERY`, `MIN_ASSET_HISTORY`,
  `SEED`, `ASSET_GROUP`, `RETURN_DROP_TIERS`, `RETURN_START`, the periods;
- the walk-forward engine's purge and refit schedule, the scoring rules and
  the tests (`returns/evaluation/`), and the strategy rules and costs;
- on the covariance side: `combo_cov_main`, its component models, the
  `vr_market` bridge, `eval/`, `bridge/` and `scripts/score_final.py`.

Allowed:

- changes the data or the universe force, each listed where it is made and
  re-verified (so far: the asset ids; `ust_int` on the 7-year zero yield;
  `us_small` with no carry or value; `us_large` as the price features'
  equity reference; the pooling groups keyed by the new ids; a 10bp one-way
  cost for `us_small`, a new judgement);
- a bug fix that makes the code do what was frozen, listed, with the
  replication rerun and the change in its outputs reported;
- imports, paths, logging, error messages and docstrings.

A new idea is a new model under a new name, outside the frozen families. Its
scores on any period before `FROZEN_AT` are development evidence only; to be
tested it needs data after `FROZEN_AT`, or a new holdout declared in
`returns/config.py` before the idea is tried. Rerunning the frozen models with
a change and comparing on 2014-2026 would turn those years into a tuning
sample.

## Developing a covariance model

### What the data forces on every model

- **Unequal histories.** Models are called with `assets = view.eligible(36)`:
  an asset appears once it has 36 months of *monthly* history, which may be
  years before it has any daily data. First monthly / first daily data:
  us_large 1871/1926, us_small 1926/1926, intl_dev 1975/1986, em 1989/1994,
  ust_int 1953/1962, ust_long 1919/1962, tips 1997/1997, ig_credit 1919/1980,
  hy_credit 1980/1985, reits 1972/1986, gold 1971/1971, commodities 1926/1991.
  Every eligible asset must get a finite forecast: fall back, pool or shrink.
- **Daily returns are inputs, not the target.** The ratio of 1-day to 21-day
  volatility ranges from 0.69 (smoothed fund NAVs, early Treasury quotes,
  pre-2000 small caps) to 1.20 (US-listed ETFs) and changes at the splice
  dates in `data/processed/segments.csv` (`view.tiers` gives the tier of each
  asset-month). Map daily data to the 21-day scale in a way that adapts across
  those changes using only the view; never carry a pre-splice scale across a
  splice.
- **The calendar.** About 295 trading days a year before 1952, about 252
  after. Annualize by actual days per year (the target does).
- **The target is noisy.** A realized 12-month vol has about 17% standard
  error and a realized correlation about 0.2. QLIKE (vol) and `qlike_rel`
  (covariance) are the headline losses because they stay valid under that
  noise.
- **Predictors** (`view.predictors_wide()`, `view.predictor_latest()`, catalog
  in `data/processed/predictors_catalog.csv`) start at very different dates,
  several have holes or are dead, and macro series are latest-vintage. A model
  must work when a predictor is missing, including at the live origin.

### Rules (binding)

1. **Development data.** Run with
   `export ASSETFORECAST_MAX_DATE=2014-12-31 OMP_NUM_THREADS=1`; results go to
   `results/dev/`. Do not read post-2014 values by any other route while
   developing.
2. **No tuning on the scoreboard.** Every hyperparameter is fixed in advance
   for a stated reason or estimated inside the model from the view
   (walk-forward). A family registers at most **8** models. What you try and
   drop goes in your report with its score. Losing to a reference benchmark is
   a result.
3. **No look-ahead.** Only the view. Training on realized targets may use
   origin `s` only if `s + 12 months <= view.asof_date`; derive training
   targets from the view (the estimator of `eval.targets.realized_kday_cov`, or
   monthly squared returns), never from cached target files.
   `scripts/leak_test.py` checks this mechanically.
4. **Cost.** Use `--jobs` <= 2 and `--step 6` while developing; re-estimate at
   most once a year inside the model and update state monthly. A full monthly
   run to 2014 must finish in under 10 minutes on one core.
5. **Ownership.** A family owns `src/assetforecast/models/<family>/`,
   `tests/models/test_<family>.py` and `reports/models/<family>.md`.

### Running and scoring

```
export ASSETFORECAST_MAX_DATE=2014-12-31 OMP_NUM_THREADS=1
python -m assetforecast.eval.run forecast --kind vol --model my_model --step 6     # quick
python -m assetforecast.eval.run forecast --kind vol --model my_model              # every month-end
python -m assetforecast.eval.run score --kind vol --models my_model ref_blend ref_last12m ref_last60m ref_longrun
```

For significance use `assetforecast.eval.inference` (`dm_test`,
`skill_score`, `pairwise_dm`, `panel_dm_test`) on
`assetforecast.eval.losses.loss_matrix(...)`; the defaults are sized for
overlapping 12-month losses. Expect few significant differences: a true 20%
improvement in mean loss is detected about half the time with 90 years of
data. Report on the `daily` subset and on `all`, overall, by asset, and for
the eras 1926-1945, 1946-1979, 1980-1999 and 2000-2013.

## Return-side runs

```
python -m assetforecast.returns.run [--max-date D] [--jobs N]          # replication
python -m assetforecast.returns.distribution [--max-date D] --score    # expected returns, distributions
python -m assetforecast.returns.evaluation.strategy [--max-date D]     # strategy test
```

Scoring uses `returns/evaluation/` (Clark-West and DM with `1.5h`
Newey-West lags, SPA over the frozen family, MCS, Kupiec and calibration
tests, FDR across the strategy grid), never `eval/`.

## Tests

Fast, on synthetic panels: every registered model returns a valid forecast at
an early, a middle and a late origin; the same origin gives the same answer
cold or warm and in any order; a forecast at `t` does not change when data
after `t` are altered. Real-data tests are marked `realdata`, slow ones
`slow`, and network ones `network` (deselected by default).

## Report (`reports/models/<family>.md`)

Each model's definition with formulas; why each hyperparameter has its value;
the results against the reference models; significance against the best
reference; everything tried and dropped; run time; caveats.
