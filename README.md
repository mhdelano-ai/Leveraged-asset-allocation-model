# assetforecast: risk, expected return and the range of outcomes for twelve asset classes

At each month-end this package publishes, for twelve asset classes and three policy portfolios, over the next twelve months:

- a **covariance forecast** (the selected model: a trimmed average of nine families of volatility models with an average of 25 correlation forecasts, bridged to 12-month returns);
- an **expected excess return**: each asset's long-run historical mean, with a 90% interval for that mean;
- a **predictive distribution** of next year's return built from the two.

They are planning inputs for a leveraged allocation model. The package does no position sizing. Returns are log returns in excess of T-bills.

**Read the report first: [reports/REPORT.md](reports/REPORT.md).** The dashboard (`dashboard/`) shows the same numbers. To view it, open `dashboard/index.html` in any browser: one self-contained file with the data included, it works offline.

## Current forecast: <!-- gen:window -->November 2026 to October 2027<!-- /gen:window --> (data through <!-- gen:data_through -->2026-10-30<!-- /gen:data_through -->)
<!-- gen:readme_forecast -->
| asset | expected excess return (90% interval) | 12-month volatility | 5% quantile | 95% quantile |
|---|---|---|---|---|
| US large-cap equity | +6.5% (3.0 to 9.9) | 15.9% | −20.6% | 32.7% |
| US small-cap equity | +7.7% (3.3 to 12.0) | 21.5% | −29.0% | 43.2% |
| International developed equity | +4.7% (0.0 to 9.4) | 16.8% | −24.0% | 32.6% |
| Emerging-market equity | +5.3% (−0.4 to 10.9) | 23.2% | −34.2% | 43.5% |
| Intermediate Treasuries (~7y) | +1.5% (−0.3 to 3.4) | 7.7% | −11.6% | 14.3% |
| Long Treasuries (~20y+) | +1.5% (−1.3 to 4.2) | 13.5% | −21.5% | 23.8% |
| TIPS (broad) | +2.3% (−0.3 to 5.0) | 5.7% | −7.6% | 12.0% |
| Investment-grade corporates | +1.7% (0.5 to 2.9) | 8.3% | −12.4% | 15.4% |
| High-yield corporates | +3.2% (1.3 to 5.2) | 6.3% | −7.6% | 13.8% |
| US REITs | +5.9% (2.2 to 9.5) | 18.2% | −25.0% | 35.9% |
| Gold | +3.8% (−1.8 to 9.4) | 24.5% | −37.8% | 44.2% |
| Broad commodities (futures TR) | +3.2% (−0.3 to 6.7) | 19.3% | −29.6% | 35.1% |
| *60/40 (US large cap / intermediate Treasuries)* | +4.5% | 10.3% | −12.5% | 20.2% |
| *Balanced 12-asset mix* | +4.4% | 10.7% | −13.4% | 20.9% |
| *Global equity mix* | +6.0% | 16.0% | −20.3% | 30.3% |
<!-- /gen:readme_forecast -->
Expected return and quantiles are of the 12-month log excess return, in percent; the quantiles include the mean. Portfolio rows (italic) use the published portfolio shape; `forecasts/latest/` also has an asset-shape version as a sensitivity check, the full covariance matrix, the 21-day-basis figures and the arithmetic-return equivalents. Stock–Treasury correlation: <!-- gen:sb_corr -->+0.10<!-- /gen:sb_corr -->.

## How far to trust it

- **Volatilities and correlations** were tested on 2014–2025 forecasts scored once after the model was chosen: they beat a simple reference; the whole-matrix advantage was not significant.
- **Expected returns** are historical means because nothing beat them. In the original return-forecasting project's own holdout (forecasts from 2015, at 1, 3 and 12 months) no machine-learning model beat the historical mean, and in a portfolio test the model forecasts did worse than historical means. The interval covers only the sampling error of the average.
- **The predictive distributions** and the 12-month re-run of the frozen models on this package's panel were produced after both holdouts had been seen: they describe, they do not test. Only forecasts made from <!-- gen:oos_from -->November 2026<!-- /gen:oos_from --> on count as new evidence.

## Where things are

| path | contents |
|---|---|
| `reports/REPORT.md` | the report: numbers, how to use them, the evidence, limits |
| `forecasts/latest/` | `cov_annual.csv` (feed this to an optimizer), `cov_21d.csv`, `corr.csv`, `vols.csv`, `portfolios.csv`, `expected_returns.csv`, `distribution_assets.csv`, `distribution_portfolios.csv`, `shape.csv`, `forecast.json` |
| `results/final/` | the covariance model's development and holdout scores (frozen) |
| `results/covariance/` | the selected covariance forecast at every past month-end |
| `results/returns/` | the 12-month replication of the frozen return models, expected-return and distribution history, distribution scores, strategy test (all labelled as not out-of-sample) |
| `results/archive/assetpred/` | the original return-forecasting project's results, unmodified |
| `reports/SELECTION.md`, `reports/final_evaluation.md` | the covariance model choice (committed before its holdout was scored) and its evaluation |
| `reports/models/`, `reports/data/` | one report per model family and per data source; the data audit; the feature port |
| `dashboard/` | `index.html` (the page, self-contained: open it in a browser), `data/*.csv` and `datasets.json`, all written by `python -m assetforecast.dashboard`; d3 vendored in `vendor/` |
| `src/assetforecast/` | `data/`, `models/`, `eval/`, `bridge/` (covariance side); `returns/` (expected returns, distributions, replication, strategy test); `covhistory.py`, `forecast.py`, `pipeline.py`, `dashboard/` |
| `scripts/` | `run_all.sh`, `score_final.py`, `make_report_figures.py`, `make_report_tables.py`, `leak_test.py` |

## Reproduce

```
pip install -e ".[dev]"
python -m assetforecast.pipeline --jobs 4      # covariance models (reuses results/full if present), covariance history,
                                               # features, replication, distributions, strategy test, forecast, dashboard data
python scripts/make_report_figures.py          # -> reports/figures/
python scripts/make_report_tables.py           # -> generated blocks in reports/REPORT.md and this README
python -m pytest -m "not network"              # the test suite
```

`results/full/` is not committed: on a fresh clone the pipeline's first step (about 30 minutes on 4 cores) must run before the rest. `python -m assetforecast.pipeline --list` shows the steps; `--max-date D` makes a development run under `results/dev/`. Rebuilding the data from source and rerunning the frozen covariance evaluation are described in the report, section 8.

## Main caveats

- **Expected returns are averages, not forecasts.** Their intervals are wide relative to the differences between many assets, and they assume the mean has not shifted. Treat them as a weak prior.
- **Wide single-year errors.** Over the last 30 years, realized volatility landed between 35% below and 22% above the forecast two years in three. Crises are not anticipated.
- **Optimizers find the matrix's weak spots.** In development, minimum-variance portfolios realized 1.43 times their predicted variance. Haircut optimized or leveraged portfolios by about 1.5, or constrain the optimizer.
- **Portfolio tails rest on few episodes.** Use the more severe of the published and asset-shape quantiles.
- **Correlations, especially stock–bond, shift faster than the forecast adapts.**
