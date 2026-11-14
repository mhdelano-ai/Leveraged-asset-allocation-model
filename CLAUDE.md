# assetpred

Research into whether asset-class excess returns are predictable out of sample
at 1, 3 and 12 months, using machine learning over price, valuation/carry,
macro and sentiment predictors. Research only: no portfolio decision hangs on it.

## Layout
- `src/assetpred/config.py` — universe, horizons, quantile grid, walk-forward settings
- `src/assetpred/schema.py` — the data contract (read its docstring first) and loaders
- `src/assetpred/data/` — downloads (`fetch.py`), asset returns, raw predictors
- `src/assetpred/features/` — point-in-time feature panel
- `src/assetpred/models/` — forecaster interface, baselines, ML models
- `src/assetpred/evaluation/` — walk-forward harness, scoring, strategy test
- `src/assetpred/dashboard/` — static dashboard generator
- `data/raw/` cached downloads (ignored); `data/processed/` and `results/` are committed

## Rules
- No look-ahead. A row dated t contains only what was knowable at the close of
  month t. Lag at the source; scaling, imputation and tuning are fit on the
  training window only.
- The bar is the expanding historical mean. Report results that fail to beat it
  as plainly as ones that do.
- Every number in a report or dashboard is produced by code in this repo.
- Install: `pip install -e .[dev]`. Tests: `pytest` (network tests are deselected by default).
