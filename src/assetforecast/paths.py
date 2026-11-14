import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
ASSET_DIR = PROCESSED / "assets"
PREDICTOR_DIR = PROCESSED / "predictors"
# Development runs (ASSETFORECAST_MAX_DATE set) write to results/dev so that the
# final full-sample results directory only ever holds full-sample runs.
RESULTS = Path(os.environ["ASSETFORECAST_RESULTS"]) if os.environ.get("ASSETFORECAST_RESULTS") else (
    ROOT / "results" / "dev" if os.environ.get("ASSETFORECAST_MAX_DATE") else ROOT / "results")
FORECASTS = RESULTS / "forecasts"
REPORTS = ROOT / "reports"
