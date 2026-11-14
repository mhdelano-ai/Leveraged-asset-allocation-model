"""Shared constants. Every module reads the universe, horizons and paths from here."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"              # cached downloads, not committed
PROCESSED = ROOT / "data" / "processed"  # committed panels
RESULTS = ROOT / "results"               # committed forecasts and scores

# Asset id -> description. Ids are the column names of every returns panel and
# the `asset` level of every (date, asset) index.
ASSETS = {
    "us_eq": "US equities",
    "dev_eq": "Developed ex-US equities",
    "em_eq": "Emerging-market equities",
    "ust_int": "Intermediate US Treasuries",
    "ust_long": "Long US Treasuries",
    "tips": "US TIPS",
    "ig_credit": "US long-duration investment-grade corporate bonds",
    "hy_credit": "US high-yield corporate bonds",
    "reits": "US equity REITs",
    "gold": "Gold",  # from 2004-12 (GLD) unless an LBMA price file is supplied; see data/assets_sources.py
    "commodities": "Diversified commodity futures",
}

HORIZONS = (1, 3, 12)  # months ahead
QUANTILES = (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)
FEATURE_FAMILIES = ("price", "valuation_carry", "macro", "sentiment")

# Walk-forward evaluation
OOS_START = "1990-01-31"     # first forecast origin that is scored
REFIT_EVERY = 12             # months between refits; expanding window
MIN_ASSET_HISTORY = 60       # months of an asset's own returns before it is forecast
SEED = 20261114
