# assetforecast — architecture and conventions

The goal: at any month-end, forecast the **covariance matrix of 12 asset
classes over the next 12 months**, for use as long-run planning inputs. Many
people (and agents) work on this in parallel; this file is the contract that
keeps their pieces compatible. Read it before writing code.

## What is being forecast

Returns are **excess over cash** (T-bills), in **logs**: `r = log(1+R) − log(1+Rf)`.

Two targets, both for the window `(t, t + 12 calendar months]` where `t` is a
month-end origin:

1. **Monthly-horizon covariance** (`cov_rv`) — the annualized covariance of
   **overlapping 21-trading-day** log excess returns realized in the window.
   This is what models forecast and what is scored most heavily.
   - Realized matrix: Σ = (252/21) × mean over the window of s·sᵀ, where s is
     the vector of 21-day log excess return sums ending on each day (all 21
     days inside the window), uncentered. One estimator gives variances and
     correlations together and is positive semi-definite by construction.
   - Where an asset has no daily data in the window, its variance is
     12 × mean of squared calendar-month log excess returns — the **same
     quantity** measured less precisely, not a different one. QLIKE and MSE
     rank forecasts consistently under noisy unbiased proxies (Patton 2011).
   - Why not daily returns: measured 1-day volatility depends on how a source
     is priced, not only on risk. Smoothed fund NAVs (high yield, REITs, early
     Treasury quotes) understate it by up to 30%; US-listed ETFs of foreign or
     illiquid assets overstate it by up to 20%; and the two conventions meet at
     splice dates, creating artificial jumps. At a 21-day horizon the sources
     agree. See `reports/data/horizon_evidence.md`.
   - The price is a noisier target (about 16% standard error on a realized
     vol, versus about 5% from daily returns), which costs statistical power
     but not validity.
   - Models may use daily returns as INPUTS — they carry the most timely
     information — but must map them to this target, and must not assume a
     1-day vol equals a 21-day vol for any asset or era.
2. **Annual-return covariance** (`cov_annual`) — the covariance of the
   12-month log excess return itself. It differs from (1) when returns are
   serially correlated beyond a month. It is obtained from `cov_rv` by the
   **bridge** (`assetforecast.bridge`), which estimates variance ratios with
   uncertainty. There are few independent 12-month observations, so this is
   validated by calibration (are standardized 12-month returns unit-variance?)
   rather than by horse race.

Annualization: 252 trading days; 12 months.

## Universe

`assetforecast.universe.ASSETS` — 12 ids used as column names everywhere:
`us_large, us_small, intl_dev, em, ust_int, ust_long, tips, ig_credit,
hy_credit, reits, gold, commodities`.

## Data

Everything flows through `assetforecast.data.panel.Panel` (see its docstring).
`load_panel()` reads the real panel; `make_synthetic_panel()` returns one with
identical conventions for tests. `Panel.asof(t)` is the only way models see
data, and it removes everything dated or *published* after `t`.

Files in `data/processed/` (committed; `data/raw/` is a gitignored download cache):

| file | content |
|---|---|
| `assets/<id>_daily.parquet` | index `date`; columns `ret` (simple total return), `source`, `tier` |
| `assets/<id>_monthly.parquet` | index month-end `date`; same columns; full history incl. the daily era (compounded daily) |
| `assets/cash_daily.parquet`, `assets/cash_monthly.parquet` | T-bill simple return, column `ret`, plus `source`, `tier` |
| `predictors/<group>.parquet` | per-track predictor files, long: `obs_date, available_date, series, value` (native units) |
| `predictors.parquet`, `predictors_catalog.csv` | assembled predictors in standard units (rates, yields, spreads and vols in decimals), and their catalog: series, group, unit, start, end, live, notes. This is what `load_panel()` reads |
| `panel_daily.parquet`, `panel_monthly.parquet`, `cash_daily.parquet` (col `cash`), `cash_monthly.parquet`, `tiers_monthly.parquet` | assembled wide panel |
| `segments.csv` | asset, start, end, freq, source, tier, notes — the provenance of every spliced segment |

The panel's daily calendar is the US equity trading calendar (the dates of the
cash series). A return dated off that calendar is compounded into the next
calendar day; a calendar day on which an asset did not trade is NaN.

Quality tiers: **A** daily price of an exchange-traded fund (market close),
total return · **B** daily open-end mutual-fund NAV, index, or reconstructed
total return · **C** monthly index total return · **D** monthly model-based
reconstruction (bond returns from yields, month-average prices, …). Every
NAV-to-ETF splice is therefore a B→A tier change. Parameters fitted on overlaps
(blend weights, splice factors, selection rules) may use only data up to
2014-12-31, the start of the holdout. ETF total returns are built from close
plus validated distributions (`assetforecast.data.sources.distributions`), not from
Yahoo's adjusted close, which lists some distributions that were never paid.

Rules for data code:
- Loaders live in `src/assetforecast/data/sources/<track>.py`, are plain Python
  HTTP (reproducible by `python -m assetforecast.data.sources.<track>`), cache raw
  downloads under `data/raw/<track>/`, and never need an API key.
- Total returns include distributions. Prefer the longest *credible* history;
  splice the highest-tier source available in each era and record every
  segment. Never fabricate or interpolate returns; a gap is NaN.
- Splices must be checked on the overlap (correlation, vol ratio, tracking
  error) and the numbers recorded in `segments.csv` notes.
- `available_date` for a predictor is when it could first have been known
  (publication lag included). Market prices: same day. Monthly macro: be
  conservative.
- Known network facts in this environment: `fred.stlouisfed.org` times out —
  use the DBnomics mirror (`api.db.nomics.world`, providers `FED`, `BLS`, …)
  or primary sources. Yahoo chart API (`query1.finance.yahoo.com/v8/finance/chart/<sym>?period1=..&period2=..&interval=1d&events=div,splits`,
  with a browser User-Agent) works and returns `adjclose`. Ken French library,
  Shiller (`www.econ.yale.edu/~shiller/data/ie_data.xls`), Fed GSW curves
  (`www.federalreserve.gov/data/yield-curve-tables/feds200628.csv`), CBOE CDN,
  World Bank pink sheet, Damodaran, LBMA, AQR datasets all respond. Stooq is
  behind a JavaScript challenge. Outbound HTTPS uses a proxy that is already
  configured; never disable TLS verification.

## Models

Interfaces and rules are in `assetforecast/models/base.py` (read the module
docstring). Each model family is a subpackage `assetforecast/models/<family>/`
whose `__init__` registers its models with `@register(kind, name)`; the
registry imports all subpackages automatically.

- Model names are `snake_case`, prefixed by family (`har_pooled_log`, `garch_gjr_t`).
- No look-ahead. Training on realized targets requires `s + 12 months <= asof`.
- Forecast every asset passed in `assets`. Handle unequal histories and
  monthly-only history gracefully (fall back, shrink, pool) — never raise or
  return NaN for an eligible asset.

## Evaluation

`assetforecast.eval` owns targets, the walk-forward harness, losses, significance
tests and portfolio tests. Forecasts are stored one file per model so that
anyone can add a model without touching anyone else's results:

- `results/forecasts/vol/<model>.parquet` — long: `origin, asset, vol`
- `results/forecasts/corr/<model>.parquet` — long: `origin, asset_i, asset_j, corr`
- `results/forecasts/cov/<model>.parquet` — long: `origin, asset_i, asset_j, cov`

Origins are month-ends. A forecast at origin `t` is scored only when its
12-month window is fully observed. Overlapping windows make errors serially
correlated: all inference uses HAC or block bootstrap with block ≥ 12 months.

## Working in parallel

- Touch only the files your brief assigns you. If you need a change in shared
  code (`universe.py`, `data/panel.py`, `models/base.py`, `eval/*` once it
  exists), do not edit it: describe the change you need in your final report.
- Do not run `git` write commands (add/commit/push/checkout/stash). The
  orchestrator commits.
- Tests go in `tests/` mirroring the source path; mark with `network`, `slow`,
  `realdata` as appropriate. Run only your own tests, plus `tests/test_contract.py`.
- `pip install` what you need and name it in your report so it reaches `pyproject.toml`.
- Python 3.13, pandas 3, numpy 2. Month-end frequency alias is `"ME"`.
- Report honestly: what works, what was verified and how, what is weak or
  unverified, and any number that surprised you.
