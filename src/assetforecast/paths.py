import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
ASSET_DIR = PROCESSED / "assets"
PREDICTOR_DIR = PROCESSED / "predictors"
# Development runs (ASSETFORECAST_MAX_DATE set) write under results/dev so that
# the full-sample directories only ever hold full-sample runs.
_DEV = bool(os.environ.get("ASSETFORECAST_MAX_DATE"))

# Covariance harness store (every model, every origin). Not committed.
RESULTS_FULL = ROOT / "results" / "full"
RESULTS = Path(os.environ["ASSETFORECAST_RESULTS"]) if os.environ.get("ASSETFORECAST_RESULTS") else (
    ROOT / "results" / "dev" if _DEV else RESULTS_FULL)
FORECASTS = RESULTS / "forecasts"
REPORTS = ROOT / "reports"

# Frozen covariance evaluation (development and the 2014-2025 holdout).
RESULTS_FINAL = ROOT / "results" / "final"
# Per-origin 21-day and 12-month covariance of the selected model.
RESULTS_COV = ROOT / "results" / ("dev/covariance" if _DEV else "covariance")
COV_HISTORY = RESULTS_COV / "cov_history.parquet"
# Expected returns, distributions, the mean-model replication and the strategy
# test. Deliberately ignores ASSETFORECAST_RESULTS, which points at the
# covariance harness store.
RESULTS_RETURNS = ROOT / "results" / ("dev/returns" if _DEV else "returns")
# The assetpred project's results and inputs as they stood when it was merged
# in: the source of its 1-, 3- and 12-month findings and the fixtures against
# which the ported code is checked. Never rewritten.
ARCHIVE_AP = ROOT / "results" / "archive" / "assetpred"

FEATURES = PROCESSED / "features.parquet"
FEATURE_META = PROCESSED / "feature_meta.csv"
FORECASTS_OUT = ROOT / "forecasts"
DASHBOARD_DATA = ROOT / "dashboard" / "data"
