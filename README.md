# volmodel: one-year covariance forecasts for a 12-asset universe

This repository forecasts, at each month-end, the covariance matrix of twelve asset classes over the next 12 months. The forecasts are long-run planning inputs for a leveraged asset-allocation model.

- Returns are log returns in excess of T-bills.
- The selected model is a trimmed average of nine families of volatility models, combined with an average of 25 correlation forecasts.
- It was validated walk-forward from 1922, with a 2014–2025 holdout that was scored only after the model was chosen.

**Read the report first: [reports/REPORT.md](reports/REPORT.md)** (about 15 minutes).

## Current forecast: November 2026 – October 2027 (data through 2026-10-30)

| asset | 12-month return volatility |
|---|---|
| US large cap | 15.9% |
| US small cap | 21.5% |
| International developed equity | 16.8% |
| Emerging-market equity | 23.2% |
| Treasuries 7–10y | 7.7% |
| Treasuries 20–30y | 13.5% |
| TIPS | 5.7% |
| Investment-grade credit | 8.3% |
| High-yield credit | 6.3% |
| US REITs | 18.2% |
| Gold | 24.5% |
| Commodities | 19.3% |

- Stock–Treasury correlation: +0.10. A 60/40 portfolio: 10.3%.
- These are forecasts of the *mean* of next year's variance. Most years come in below them; a few come in far above.
- The full matrix, the 21-day-basis figures and the downside quantiles are in `forecasts/latest/`.

## Where things are

| path | contents |
|---|---|
| `reports/REPORT.md` | the final report: numbers, how to use them, validation, findings, limits |
| `forecasts/latest/` | `cov_annual.csv` (feed this to an optimizer), `cov_21d.csv`, `corr.csv`, `vols.csv`, `portfolios.csv`, `forecast.json` |
| `results/final/` | development and holdout scores; `dashboard_vol.csv`, `dashboard_corr.csv` and `dashboard_portfolio.csv` hold forecast against realized for every month |
| `reports/SELECTION.md` | the model choice and expectations, committed before the holdout was scored |
| `reports/final_evaluation.md` | the full evaluation |
| `reports/models/`, `reports/data/` | one report per model family and per data source; the data audit |
| `dashboard/` | the page and data files behind the published dashboard |
| `src/volmodel/` | `data/` (loaders, panel), `models/<family>/`, `eval/` (targets, losses, tests), `bridge/`, `forecast.py` |
| `scripts/` | `run_all.sh`, `score_final.py`, `make_report_figures.py`, `make_dashboard_data.py`, `leak_test.py` |
| `ARCHITECTURE.md`, `MODELING_PROTOCOL.md` | definitions and the rules the work followed |

## Reproduce

```
pip install -e ".[dev]"
# optional data rebuild (needs internet, no API keys):
for t in cash_macro us_equity intl_equity treasuries tips credit reits real_assets implied; do python -m volmodel.data.sources.$t; done
python -m volmodel.data.assemble
scripts/run_all.sh 4                                         # every model, full history -> results/full/ (about 30 min on 4 cores)
export OMP_NUM_THREADS=1 VOLMODEL_RESULTS=$PWD/results/full
python scripts/score_final.py                                # -> results/final/ (about 2.5 min)
python -m volmodel.forecast                                  # -> forecasts/<date>/ and forecasts/latest/
python scripts/make_report_figures.py                        # -> reports/figures/
python -m pytest -m "not network"                            # the test suite
```

`results/full/` is not committed, so `scripts/run_all.sh` must be run before scoring or forecasting on a fresh clone.

## Main caveats

- **No expected returns.** The quantiles exclude the mean; supply your own.
- **Wide single-year errors.** Over the last 30 years, realized volatility landed between 35% below and 22% above the forecast two years in three. Crises are not anticipated: high yield realized 19% against a 6.3% forecast in 2020.
- **Optimizers find the matrix's weak spots.** In development, minimum-variance portfolios realized 1.43 times their predicted variance. Haircut optimized or leveraged portfolios by about 1.5, or constrain the optimizer.
- **Correlations, especially stock–bond, shift faster than the forecast adapts.** The backtest can only detect large differences between models: the development and holdout rankings of 70 models correlated at −0.26.
