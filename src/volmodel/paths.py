from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
ASSET_DIR = PROCESSED / "assets"
PREDICTOR_DIR = PROCESSED / "predictors"
RESULTS = ROOT / "results"
FORECASTS = RESULTS / "forecasts"
REPORTS = ROOT / "reports"
