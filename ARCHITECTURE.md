# assetforecast: architecture and conventions

At each month-end the package forecasts, for 12 asset classes over the next
12 months:

1. the **covariance matrix** of returns (the selected covariance model, bridged
   to 12-month returns);
2. the **expected return** of each asset (its expanding historical mean, with a
   90% interval);
3. the **predictive distribution** of each asset's and each policy portfolio's
   12-month return (quantiles built from 1 and 2 and an empirical shape).

Two research projects were merged to get here: the covariance project
(formerly `volmodel`) and `assetpred`, which tested whether machine learning
can predict expected returns and found that nothing beat the historical mean.
`assetpred`'s frozen models survive as a replication (`returns/`), not as a
forecast. This file is the contract that keeps the pieces compatible; read it
before writing code. `MODELING_PROTOCOL.md` says how models are developed and
judged and what is frozen.

## What is forecast

Returns are **excess over cash** (T-bills), in **logs**: `r = log(1+R) - log(1+Rf)`.
Every target covers the window `(t, t + 12 calendar months]` after a month-end
origin `t`. A row or forecast dated `t` uses only what was known at the close
of `t`.

### Covariance: two targets

1. **Monthly-horizon covariance** (`cov_rv`, published as `cov_21d`): the
   annualized covariance of **overlapping 21-trading-day** log excess returns
   realized in the window. This is what the covariance models forecast and
   what is scored most heavily.
   - Realized matrix: Σ = (252/21) × mean over the window of s·sᵀ, where s is
     the vector of 21-day log excess return sums ending on each day (all 21
     days inside the window), uncentered. One estimator gives variances and
     correlations together and is positive semi-definite by construction.
   - Where an asset has no daily data in the window, its variance is
     12 × mean of squared calendar-month log excess returns: the same quantity
     measured less precisely. QLIKE and MSE rank forecasts consistently under
     noisy unbiased proxies (Patton 2011).
   - Why not daily returns: measured 1-day volatility depends on how a source
     is priced. Smoothed fund NAVs (high yield, REITs, early Treasury quotes)
     understate it by up to 30%; US-listed ETFs of foreign or illiquid assets
     overstate it by up to 20%; the two conventions meet at splice dates. At a
     21-day horizon the sources agree (`reports/data/horizon_evidence.md`).
   - Models may use daily returns as inputs but must map them to this target.
2. **Annual-return covariance** (`cov_annual`): the covariance of the 12-month
   log excess return itself, obtained from `cov_rv` by the **bridge**
   (`assetforecast.bridge`, selected bridge `vr_market`), which estimates
   variance ratios with uncertainty. It differs from (1) when returns are
   serially correlated beyond a month. Validated by calibration, not by horse
   race (there are few independent 12-month observations).

The selected model is `combo_cov_main` with the bridge `vr_market`
(`reports/SELECTION.md`). `forecast.py` publishes it at the latest origin;
`covhistory.py` rebuilds it at every origin for the return side, and
`forecast.py` asserts that both give the same matrix at the published origin.

Annualization: 252 trading days; 12 months.

### Expected returns (`returns/expected.py`)

`mean_12m` = 12 × the mean monthly log excess return over the asset's whole
history through `t`, on the return panel (below). It is identical to the
`hist_mean` benchmark of the replication, including its eligibility rule (60
months of returns). It is not a model forecast: the replication of the frozen
mean models found nothing that beats it. `se_12m` comes from an
equal-weighted-cosine long-run variance (`eval.inference.long_run_variance`,
`B = n/48` terms clipped to `[4, 0.4 n^(2/3)]`) and the 90% interval is
`mean_12m ± t_B(0.95)·se_12m`; the module docstring has the simulation behind
the rule. The interval covers the sampling error of an average, not a mean
that has shifted. The published `mean_12m_arith = exp(mean_12m + ½ cov_annual_ii) − 1`
is the simple-return counterpart.

### Predictive distributions (`returns/distribution.py`, model `dist_hm_af`)

`q_k(t, a) = m(t, a) + s(t, a)·Z_k(t)` with `m` the expected return,
`s = sqrt(cov_annual_ii + se²)`, and `Z_k` the 9 quantiles (1, 5, 10, 25, 50,
75, 90, 95, 99%) of past standardized outcomes `z = (y − m)/s` pooled across
assets over origins whose window closed by `t`, recentred, with the 1/5/95/99%
quantiles floored at a normal of the same spread. Policy portfolios
(`eval.portfolio.POLICIES`, weights renormalized over the assets present at
each origin) get the mean `w'm`, the variance `w'Σw`, the mean's standard error
`Σ|w_i|·se_i` (mean errors treated as perfectly correlated; it enters the scale
as `sqrt(w'Σw + se_p²)`), and a shape pooled over the portfolios' own past `z`,
which keeps their spread (below 1) and floors the tails at a normal of that
spread; the asset-pooled shape applied to portfolios is the
sensitivity model `dist_hm_af.assetshape`. Portfolio log returns are
approximated by `w'r`, as in the covariance side's portfolio tests. This model
was designed after both holdouts had been seen; its scores are descriptive.

## Universe

`assetforecast.universe.ASSETS`: 12 ids used as column names everywhere:
`us_large, us_small, intl_dev, em, ust_int, ust_long, tips, ig_credit,
hy_credit, reits, gold, commodities`. `universe.Asset.group` puts REITs in
"real"; the mean models' pooling groups (`returns.config.ASSET_GROUP`) put them
in "equity", as `assetpred` did. The two are different tables on purpose.

## Data

Everything flows through `assetforecast.data.panel.Panel` (see its
docstring). `load_panel()` reads the real panel; `make_synthetic_panel()`
returns one with identical conventions for tests. `Panel.asof(t)` is the only
way models see data, and it removes everything dated or *published* after `t`.

Files in `data/processed/` (committed; `data/raw/` is a gitignored download cache):

| file | content |
|---|---|
| `assets/<id>_daily.parquet` | index `date`; columns `ret` (simple total return), `source`, `tier` |
| `assets/<id>_monthly.parquet` | index month-end `date`; same columns; full history incl. the daily era (compounded daily) |
| `assets/cash_daily.parquet`, `assets/cash_monthly.parquet` | T-bill simple return, column `ret`, plus `source`, `tier` |
| `predictors/<track>.parquet` | per-track predictor files, long: `obs_date, available_date, series, value` (native units) |
| `predictors/returns.parquet`, `predictors/returns_meta.csv` | the return side's 73 predictors (the `rp_` track) and their metadata |
| `predictors.parquet`, `predictors_catalog.csv` | the single predictor store that `load_panel()` reads (every track, standard units) and its catalog |
| `panel_daily.parquet`, `panel_monthly.parquet`, `cash_daily.parquet` (col `cash`), `cash_monthly.parquet`, `tiers_monthly.parquet` | assembled wide panel |
| `segments.csv`, `segments/` | asset, start, end, freq, source, tier, notes: the provenance of every spliced segment |
| `features.parquet`, `feature_meta.csv` | the return side's point-in-time feature panel (below) |
| `managed_futures_monthly.parquet`, `managed_futures_meta.json` | managed futures, outside the panel and read only by the diversifier study: AQR's time-series-momentum factor (`tsmom_xr`, simple excess, hypothetical, before fees) and an equal-weight composite of live funds (`funds_ret`, simple total, after fees), from `data/sources/managed_futures.py` |
| `diversifier_vehicles.json` | funds that give the diversifier sleeve's exposures and their expense ratios: a hand-recorded outside lookup with its date and sources, shown in the diversifier report; no loader rebuilds it |
| `raw_manifest.json` | what `data/fetch.py` downloaded, when, with a hash (written by the downloader) |

The panel's daily calendar is the US equity trading calendar (the dates of the
cash series). A return dated off that calendar is compounded into the next
calendar day; a calendar day on which an asset did not trade is NaN.

Quality tiers: **A** daily ETF price (market close), total return · **B**
daily open-end fund NAV, index, or reconstructed total return · **C** monthly
index total return · **D** monthly model-based reconstruction (bond returns
from yields, month-average prices, ...). Parameters fitted on overlaps (blend
weights, splice factors, selection rules) use only data up to 2014-12-31. ETF
total returns are built from close plus validated distributions
(`data.sources.distributions`), not from Yahoo's adjusted close.

### One panel, two views

- **Covariance side: every tier.** The covariance models and the bridge see
  the whole history, tier D included (the bridge classifies the month-average
  eras itself). `covhistory` refuses to run with `ASSETFORECAST_DROP_TIERS` set.
- **Return side: without tier D, from 1926-07-31**
  (`returns.inputs.load_return_panel()`, which passes
  `drop_tiers=config.RETURN_DROP_TIERS` explicitly). Month-average prices
  smooth monthly returns (they add first-order autocorrelation of about 0.25
  and cut measured volatility by about 15%); a 12-month sum of such returns,
  momentum and reversal features computed from them, and a historical mean's
  standard error would all inherit the artefact, and `assetpred`'s models
  were never trained on such data. The cost is history: on the return side
  Treasuries start in 1962-02 (before that they are tier D). Never set
  `ASSETFORECAST_DROP_TIERS` globally: the covariance side would silently
  change.

`returns.inputs` also gives `excess_returns(panel)` (monthly log excess
returns through `observed_through(panel)`, the last complete month) and
`return_predictors(panel)` (the `rp_` series as a wide month-end panel).

### The predictor store

One long store, `predictors.parquet` (`obs_date, available_date, series,
value`), built by `python -m assetforecast.data.assemble` from the track files.
`Panel.asof(t)` keeps rows with `available_date <= t`. The catalog
(`predictors_catalog.csv`) has `series, group, unit, start, end, live, notes`
and, for the return track, `family, scope, asset, lag_months, realtime,
hindsight` (blank or False for the other tracks).

The **`rp_` track** (`data/sources/return_predictors.py`) holds the 73
valuation/carry, macro and sentiment series of `assetpred` as `rp_<name>`.
Each row is already lagged to the month-end at which it could have been
known, so `obs_date = available_date =` that month-end; values are in native
units (scale 1) because the feature formulas assume them. Point-in-time
conventions (`realtime`): `market` (never revised), `vintage` (rebuilt from
ALFRED vintages), `lagged_revised` (today's data shifted by `lag_months`, at
the actual first-release month where vintages exist; revisions remain a small
leak). The **hindsight flag** marks series estimated over the full sample
(factor models, full-sample regressions, principal components): even lagged,
their history embeds later information. The `.nohind` robustness rows of the
replication drop every feature built on them.

The default build is offline: `python -m
assetforecast.data.sources.return_predictors --from-committed` migrates
`results/archive/assetpred/data/predictors_assetpred.parquet`; `--rebuild`
downloads every source again through `data/fetch.py` (network; ALFRED vintages
take long). Run the assembler afterwards. Adding `rp_` series does not change
any covariance forecast (`tests/data/test_return_predictors.py` checks models on
every predictor access path).

### The return side's feature panel (`returns/features/`)

`python -m assetforecast.returns.features.build` writes `features.parquet`:
(date, asset) index, 84 float columns in `assetpred`'s order, one row per
asset-month with a return on the return panel. Families: price, valuation_carry,
macro, sentiment; scope asset or global; a `hindsight` column in
`feature_meta.csv`. The frozen models read the 82 columns pinned in
`returns.config.FROZEN_FEATURE_COLUMNS` (all but the Baker-Wurgler sentiment
pair, which ended in 2019), whatever the panel's end date. Each asset's own
carry and value sit in two generic slots, `carry` and `value`; the per-asset
definitions are `CARRY_DEF` and `VALUE_DEF` in `returns/features/predictors.py`.
No credible measure exists for `us_small`, `em`, `hy_credit` and `reits`
(both slots NaN) or for the value of `commodities`. The changes the new
universe forced (ids, `ust_int` on the 7-year zero, `us_large` as the equity
reference of the price features) are listed in that module and compared
feature by feature in `reports/data/feature_port.md`
(`python -m assetforecast.returns.features.port_report`). Daily-volatility
features jump where an asset's daily source changes (high yield's fund NAV,
spliced to HYG in 2007-04, has 0.45 times its 1-day volatility); this is
reported, not corrected, because correcting it would change a frozen
definition.

### Rules for data code

- Loaders live in `src/assetforecast/data/sources/<track>.py`, are plain Python
  HTTP (reproducible by `python -m assetforecast.data.sources.<track>`), cache
  raw downloads under `data/raw/`, and never need an API key. The return track
  downloads through `data/fetch.py`, which records every file in
  `data/processed/raw_manifest.json`.
- Total returns include distributions. Prefer the longest credible history;
  splice the highest-tier source in each era and record every segment. Never
  fabricate or interpolate returns; a gap is NaN. Check splices on the overlap
  and record the numbers in the segment notes.
- `available_date` is when a value could first have been known (publication
  lag included). Market prices: same day. Monthly macro: be conservative.
- Network facts: `fred.stlouisfed.org` times out (use the DBnomics mirror
  `api.db.nomics.world` or primary sources); the Yahoo chart API works with a
  browser User-Agent; Ken French, Shiller, the Fed GSW curves, CBOE, the World
  Bank pink sheet, Damodaran, LBMA and AQR respond; Stooq does not. Outbound
  HTTPS uses a configured proxy; never disable TLS verification.

## Models

**Covariance models** (`assetforecast/models/<family>/`). Interfaces and rules
are in `models/base.py` (read its docstring). Each family's `__init__`
registers its models with `@register(kind, name)` for `kind` in vol, corr, cov;
`models.base._load_all` imports every subpackage of `models/` automatically,
which is why nothing else may live there. Names are `snake_case`, prefixed by
family. No look-ahead: training on realized targets requires
`s + 12 months <= asof`. Every eligible asset (`view.eligible(36)`) gets a
finite forecast.

**Return-side mean models** (`assetforecast/returns/models/`): the forecaster
interface (`base.py`), the registry (discovery scans `returns/models`), the
baselines (`hist_mean, vol_ewma, hist_quantile, vol_ewma_fhs, zero`) and the
standardized-deviation ML models (`ml_mean.py`: `ridge_z, combo_z, gbm_z, rf_z,
ens_z, gbm_z.price` and the robustness rows `gbm_z.nohind, combo_z.nohind`).
They are frozen (see `MODELING_PROTOCOL.md`) and run only as the replication.
Constants are in `returns/config.py`.

**Bridge** (`bridge/`): 21-day to 12-month covariance; variance ratios
re-estimated once a year from the view at the last December month-end.

## Evaluation: two engines, two inference modules

Both are kept because they do different jobs. Do not mix them.

| | covariance side | return side |
|---|---|---|
| walk-forward | `eval/harness.py`: at each origin `panel.asof(t)`, the model gets the view and the eligible assets; one file per model in `<results>/forecasts/{vol,corr,cov}/<model>.parquet`; resumable | `returns/evaluation/walkforward.py`: supervised models on the feature panel; the purge (`s + h <= t`) is enforced by the engine; refit every 12 months on an expanding window; cached by a fingerprint of data, settings and code in `results/returns/forecasts/<model>/h12.parquet` |
| targets and losses | `eval/targets.py` (realized 21-day covariance), `eval/losses.py` (QLIKE, `qlike_rel`, ...), `eval/portfolio.py` | `returns/schema.make_targets` (12-month sums), `returns/evaluation/{scoring,metrics}.py` (CRPS, pinball, R² against `hist_mean`, coverage) |
| inference | `eval/inference.py`: DM (HAC and bootstrap), panel DM, pairwise DM, MCS, Mincer-Zarnowitz, skill scores for overlapping 12-month losses, fixed-b EWC variance with Student-t references | `returns/evaluation/inference.py`: Clark-West, DM with Newey-West `1.5h` lags, SPA, MCS, Kupiec and calibration tests, FDR; used by scoring and the strategy test |
| used by | `eval/run.py`, `scripts/score_final.py` (the frozen covariance evaluation) | `returns/run.py` (replication), `returns/distribution.py` (scoring), `returns/evaluation/strategy.py` |

A new covariance model is run and scored with the left column; anything scored
as a forecast of returns (means, quantiles, distributions, portfolio rules)
with the right. Overlapping windows make 12-month errors serially correlated:
every test in both modules allows for it.

## Running it

`python -m assetforecast.pipeline [--max-date D] [--jobs N] [--skip STEP ...] [--rebuild-covariance]`
runs, each as its own process: `covariance` (reuses `results/full/` unless
`--rebuild-covariance`; otherwise `scripts/run_all.sh`), `covhistory`,
`features`, `replication`, `distribution` (with scoring), `strategy`,
`forecast`, `dashboard`. `--dry-run` prints the commands. The frozen
covariance evaluation (`scripts/score_final.py`) is not a step.

The published forecast, `python -m assetforecast.forecast`, writes
`forecasts/<origin>/` and `forecasts/latest/`: `cov_annual.csv` (feed this to
an optimizer), `cov_21d.csv`, `corr.csv`, `vols.csv`, `portfolios.csv`,
`expected_returns.csv`, `distribution_assets.csv`,
`distribution_portfolios.csv` (column `shape`: `portfolios` = published,
`assets` = sensitivity), `shape.csv` and `forecast.json` (everything, with a
statement of method per block and the consistency checks).

It also writes `frontier.csv` and `frontier_points.csv` (`assetforecast.frontier`,
also `python -m assetforecast.frontier --dir <folder>`): the long-only efficient
frontier those files imply, with cash below full investment and borrowing on
Robinhood margin up to `MAX_LEVERAGE` above it (`data/processed/margin_rates.json`: the published
rate schedule and its date, downloaded by `python -m assetforecast.data.sources.margin_rates --refresh`; the cost is the rate of tier `MARGIN_TIER` on a
360-day year, less the T-bill rate of the origin month, recorded in the
`frontier.borrow` block of `forecast.json`), in expected compound
growth against volatility, with a band from the 90% intervals of the means. It
is arithmetic on the published covariance and expected returns, computed from the
rounded files themselves; it is not a model, selects nothing and has no
out-of-sample evidence of its own. The dashboard's frontier section shows it.

## Results layout

| path | contents | committed |
|---|---|---|
| `results/full/` | covariance harness store: every model, every origin, and the cached targets | no (rebuild: `scripts/run_all.sh`, about 30 minutes on 4 cores) |
| `results/final/` | the frozen covariance evaluation, development and the 2014-2025 holdout (`scripts/score_final.py`) | yes, never rewritten |
| `results/covariance/` | `cov_history.parquet` (`origin, asset_i, asset_j, cov_21d, cov_annual`), `cov_history_origins.csv`, `meta.json` | yes |
| `results/returns/forecasts/<model>/h12.parquet` + `.json` | replication forecasts (`returns.schema.FORECAST_COLUMNS`) | yes |
| `results/returns/covariances/{af_primary,sample}/h12.parquet` | the package covariance in the return side's layout, and the static comparator | yes |
| `results/returns/scores/` | `replication.{parquet,csv,json}`, `replication_cumdsse.parquet` | yes |
| `results/returns/expected/` | `history.parquet`: mean and interval at every origin | yes |
| `results/returns/distribution/` | `history.parquet`, `history_portfolios.parquet`, `shape_history.parquet`, `residuals.parquet`, `scores.parquet`, metadata | yes |
| `results/returns/strategy/` | `metrics.parquet`, `summary.csv`, `summary_by_period.csv`, `meta.json` (`returns.parquet`, `weights.parquet` regenerate in a minute and are not committed) | partly |
| `results/archive/assetpred/` | the `assetpred` project's results and inputs as merged: its 1-, 3- and 12-month findings (`scores/`, `strategy/`; its HTML dashboard was removed, recoverable from git history), its development trial logs, and the fixtures the port is checked against | yes, never rewritten |
| `results/horizons/` | score tables of the horizon study (`assetforecast.horizons`: returns, volatility and correlations at 1 month to 10 years, descriptive only); its report is `reports/horizon_study.md`, its page `dashboard/horizons.html` | yes |
| `results/diversifiers/` | tables of the diversifier study (`assetforecast.diversifiers`: each non-stock asset and managed futures against world stocks, descriptive only), and of an equal-weight sleeve of intermediate Treasuries, gold and managed futures beside stocks, with and without borrowing (`sleeve.csv`); report `reports/diversifiers.md` | yes |
| `results/allocation/` | tables of the allocation study (`assetforecast.allocation`: 10 sleeve rules and 8 portfolio rules replayed month by month on fixed, historical and forecast inputs, how well each input carried over, and today's weights; descriptive only); report `reports/allocation.md`, page `dashboard/allocation.html` (`assetforecast.allocation_page`) | yes |
| `results/dev/` | every development run (and the horizon study's forecast caches, `results/dev/horizons/`) | no |
| `forecasts/` | the published forecast per origin and `latest/`; `three_fund/` holds its restatement for two declared composites (world stocks, world bonds) and cash, with the mixes into the borrowed range (`python -m assetforecast.threefund`) | yes |
| `dashboard/` | the published dashboard page, its data and the one-page summary (`summary.html`) | yes |

Every output of the replication, the distribution scores and the strategy
test carries `returns.config.REPLICATION_LABEL`.

## The development switch

`ASSETFORECAST_MAX_DATE=D` makes `load_panel()` end at `D` (no returns,
predictors or targets after it) and moves outputs under `results/dev/`:
`paths.RESULTS` (the covariance harness store, unless `ASSETFORECAST_RESULTS`
points elsewhere), `RESULTS_COV`/`COV_HISTORY`, `RESULTS_RETURNS` and
`FEATURES`/`FEATURE_META`. `RESULTS_FULL`, `RESULTS_FINAL`, `ARCHIVE_AP` and
`FORECASTS_OUT` never move. `RESULTS_RETURNS` ignores `ASSETFORECAST_RESULTS`
on purpose (that variable points at the covariance harness store).

`assetforecast.paths` reads the variable once, when it is first imported, and
most modules import it at import time. So: CLIs take `--max-date D`, which sets
the variable before `paths` is imported (or reloads `paths`, in `covhistory`
and `forecast`); the return-side runners refuse to start if `paths` was set
up for the other mode; and the pipeline runs every step as a fresh process
with the variable set (or removed) in its environment. In a long-lived
process, set it before the first import of `assetforecast`.

## Working conventions

- Python 3.13, pandas 3, numpy 2. Month-end frequency alias is `"ME"`.
- Tests live in `tests/` mirroring the source (`tests/returns/`, `tests/data/`,
  ...). There is no `__init__.py` in the test folders, so every test file name
  must be unique. Markers: `network` (deselected by default), `slow`,
  `realdata` (needs `data/processed`).
- Report honestly: what works, what was verified and how, what is weak or
  unverified, and any number that surprised you.
