# assetforecast

One package that forecasts, for 12 asset classes over the next 12 months, the
covariance matrix (the selected model `combo_cov_main` + bridge `vr_market`),
the expected return (each asset's expanding historical mean, with a 90%
interval) and the predictive distribution. The return side (`returns/`)
holds `assetpred`'s frozen mean models, re-run only as a replication. Read
`ARCHITECTURE.md` (contract) and `MODELING_PROTOCOL.md` (what is frozen, what
counts as evidence) before changing anything.

## Layout
- `src/assetforecast/paths.py`, `universe.py`: locations, the 12 asset ids
- `data/`: loaders (`sources/`), downloader (`fetch.py`), `assemble.py`, `panel.py` (`Panel.asof`)
- `models/<family>/`, `eval/`, `bridge/`: covariance models, their harness, losses and inference, the 12-month bridge
- `covhistory.py`: the selected covariance at every origin
- `returns/`: config, inputs, features, mean models, walk-forward, scoring, inference, strategy test, `expected.py`, `distribution.py`
- `horizons/`: the descriptive horizon study (1 month to 10 years; `reports/horizon_study.md`, `dashboard/horizons.html`, `results/horizons/`); changes nothing that is frozen or published
- `threefund.py`: the published forecast restated for world stocks, world bonds and cash (`forecasts/three_fund/`); arithmetic on declared composites, not a model
- `diversifiers.py`: the descriptive diversifier study (which assets have diversified world stocks; `reports/diversifiers.md`, `results/diversifiers/`). Managed futures are not in the 12-asset panel; they come from `data/sources/managed_futures.py` (`data/processed/managed_futures_monthly.parquet`)
- `forecast.py`: the published forecast (`forecasts/<origin>/`, `forecasts/latest/`); `frontier.py`: the efficient frontier it implies; `pipeline.py`: everything in order
- `data/processed/`, `results/` (except `results/full/`, `results/dev/`), `forecasts/` are committed; `data/raw/` is a download cache

## Rules
- No look-ahead. A row or forecast dated t uses only what was known at the close of month t.
  Lag at the source; fit scaling, imputation and tuning on the training window only.
- Both holdouts are spent. Replication outputs carry `REPLICATION_LABEL`; only forecasts whose
  whole 12-month window lies after `FROZEN_AT` are out-of-sample evidence for the return side.
- Do not change frozen model code or definitions (`returns/models/`, `returns/features/`,
  `returns/evaluation/`, `returns/config.py` values, the covariance models, `bridge/`, `eval/`,
  `scripts/score_final.py`) or anything under `results/archive/` and `results/final/`.
- The bar for expected returns is the expanding historical mean. Report results that fail to
  beat it as plainly as ones that do.
- Every number in a report or dashboard is produced by code in this repo.
- Never set `ASSETFORECAST_DROP_TIERS` globally; development runs use
  `ASSETFORECAST_MAX_DATE` (or `--max-date`), outputs go under `results/dev/`.

## Install, run, test
- Install: `pip install -e ".[dev]"`
- Run: `python -m assetforecast.pipeline` (reuses `results/full/`; `--rebuild-covariance` reruns it, about 30 min)
- Test: `pytest` (network tests are deselected by default; `-m "not slow"` for a quick pass)
