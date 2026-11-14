# Modeling protocol

Rules for anyone adding a forecasting model. ARCHITECTURE.md defines the
target, the data and the interfaces; this file defines how models are
developed and judged so that the final comparison means something.

## Read first

1. `ARCHITECTURE.md` — the target (21-day-horizon covariance over the next 12
   months), the Panel, storage formats.
2. `src/assetforecast/models/base.py` — the three interfaces and the rules every
   model must obey.
3. `reports/data/panel.md` and `reports/data/horizon_evidence.md` — what is in
   the panel and why 1-day volatility cannot be trusted.
4. `src/assetforecast/models/ref/__init__.py` — the reference benchmarks, and the
   pattern for a family package.
5. The module docstrings of `src/assetforecast/eval/targets.py`, `losses.py`,
   `inference.py` and the CLI `python -m assetforecast.eval.run --help`.

## What the data forces on every model

- **Unequal histories.** Models are called with `assets = view.eligible(36)`:
  an asset appears once it has 36 months of *monthly* history, which may be
  years before it has any daily data. First monthly / first daily data:
  us_large 1871/1926, us_small 1926/1926, intl_dev 1975/1986, em 1989/1994,
  ust_int 1953/1962, ust_long 1919/1962, tips 1997/1997, ig_credit 1919/1980,
  hy_credit 1980/1985, reits 1972/1986, gold 1971/1971, commodities 1926/1991.
  Every eligible asset must get a finite forecast: fall back, pool or shrink.
- **Daily returns are inputs, not the target.** The ratio of 1-day to 21-day
  volatility ranges from 0.69 (smoothed fund NAVs, early Treasury quotes,
  pre-2000 small caps) to 1.20 (US-listed ETFs), and it changes at the splice
  dates listed in `data/processed/segments.csv` (`view.tiers` gives the tier
  of each asset-month). A model that uses daily data must map it to the
  21-day scale in a way that adapts across those changes using only the view
  — for example a trailing estimate of the 21-day/1-day variance ratio, or
  multi-day sums — and must not carry a pre-splice scale across a splice.
- **The calendar.** About 295 trading days a year before 1952, about 252
  after. Annualize by actual days per year (the target does).
- **The target is noisy.** A realized 12-month vol has about 17% standard
  error and a realized correlation about 0.2. Losses for a single origin mean
  little; only long averages do. QLIKE (vol) and `qlike_rel` (covariance) are
  the headline losses because they stay valid under that noise; the
  scoreboard flags the ones that do not.
- **Predictors** (`view.predictors_wide()`, `view.predictor_latest()`,
  catalog in `data/processed/predictors_catalog.csv`) start at very different
  dates, several have multi-year holes or are dead, and macro series are
  latest-vintage (industrial production is the most revised). A model must
  work when a predictor is missing, including at the live origin.

## Development rules (binding)

1. **Holdout.** Run every command with
   `export ASSETFORECAST_MAX_DATE=2014-12-31 OMP_NUM_THREADS=1`. The panel then
   ends in 2014 and results go to `results/dev/`. Everything after 2014 is a
   holdout that is scored once, at the end, by the orchestrator. Do not load
   the panel without that variable and do not read post-2014 values from
   `data/processed` by any other route.
2. **No tuning on the scoreboard.** The longest assets have about 90 years of
   targets, the shortest under 20, and the scores are noisy: try enough
   variants and one wins by luck. Therefore every hyperparameter is either
   fixed in advance for a stated reason (theory, the literature, a round
   number) or estimated inside the model from the view (walk-forward). A
   family registers at most **8** models. If you try something and drop it,
   it goes in your report with its score. Losing to a reference benchmark is
   a result, not a failure — report it.
3. **No look-ahead.** Only the view. A model that trains on realized targets
   may use origin `s` only if `s + 12 months <= view.asof_date`. Derive
   training targets from the view itself (the same estimator as
   `assetforecast.eval.targets.realized_kday_cov`, or monthly squared returns),
   never from the cached target files.
4. **Cost.** Four cores are shared by about a dozen agents. Use `--jobs` ≤ 2
   and `--step 6` while developing; re-estimate parameters at most once a
   year inside the model and update state monthly. A full monthly run (about
   1,100 origins to 2014) must finish in under 10 minutes on one core.
5. **Ownership.** You own `src/assetforecast/models/<family>/`,
   `tests/models/test_<family>.py` and `reports/models/<family>.md`. Touch
   nothing else and run no git write commands. Keep scratch files in your own
   subfolder of the scratch directory.

## Running and scoring

```
export ASSETFORECAST_MAX_DATE=2014-12-31 OMP_NUM_THREADS=1
python -m assetforecast.eval.run forecast --kind vol --model my_model --step 6     # quick
python -m assetforecast.eval.run forecast --kind vol --model my_model              # every month-end
python -m assetforecast.eval.run score --kind vol --models my_model ref_blend ref_last12m ref_last60m ref_longrun
```

The reference models are already stored in `results/dev/` for every origin.
For significance use `assetforecast.eval.inference` (`dm_test`, `skill_score`,
`pairwise_dm`, `panel_dm_test`) on `assetforecast.eval.losses.loss_matrix(...)`;
the defaults are sized for overlapping 12-month losses. Expect few
significant differences: a true 20% improvement in mean loss is detected
about half the time with 90 years of data.

Report results on the `daily` subset (assets whose target comes from daily
data) and on `all` (which adds the monthly-proxy eras), overall, by asset, and
for the eras 1926–1945, 1946–1979, 1980–1999 and 2000–2013.

## Tests

Fast, on `make_synthetic_panel()`: every registered model returns a valid
forecast at an early, a middle and a late origin (the early one has assets
with monthly data only); the same origin gives the same answer whether the
model is called cold or after other origins, and in any order; and a forecast
at `t` does not change when the data after `t` are altered.

## Report (`reports/models/<family>.md`)

Each model's definition with formulas; why each hyperparameter has its value;
the results tables above against the reference models; significance against
the best reference; everything tried and dropped; run time; caveats.
