"""The horizon study: how predictable are returns, volatility and correlations
at forecast horizons from 1 month to 10 years?

The package publishes everything at 12 months. This study asks what changes
when the horizon is opened up. It is a DESCRIPTIVE study (``LABEL``): both
holdouts were spent before it was designed, nothing in it is selected, and it
changes neither the frozen models nor the published forecast.

    python -m assetforecast.horizons.run          # everything, then the report and the page

Modules
-------
``mlcopies``  the frozen learners behind ``ridge_z``, ``gbm_z`` and ``ens_z``,
              re-aimed at each horizon under new names (``hz_ridge``, ``hz_gbm``,
              ``hz_ens``); the frozen registrations and their results are untouched
``signals``   single-signal predictive regressions and their equal-weight
              combination (``sig_*``), and the benchmark (the expanding mean)
``risk``      realized volatility and correlation at each horizon, simple
              forecasters of both, their scores
``value``     what a single-asset sleeve that sizes its position with these
              forecasts would have earned, after costs
``score``     skill tables for the return forecasters
``report``    ``reports/horizon_study.md`` and its figures
``page``      ``dashboard/horizons.html``

Periods (by forecast origin t and horizon h, see ``period_mask``)
-----------------------------------------------------------------
``early``  origins before 1990-01 (simple forecasters only; the ML copies start in 1990)
``dev``    origins from 1990-01 whose whole window closes by 2014-12-31: the
           years on which this package's models were developed
``after``  origins from 2015-01: the spent holdout years, shown as description
``full``   every scored origin from 1990-01

Outputs: forecast caches under ``results/dev/horizons/`` (not committed), the
score tables under ``results/horizons/`` (committed). With
``ASSETFORECAST_MAX_DATE`` set, both go under ``results/dev/horizons/``.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

HORIZONS = (1, 3, 6, 12, 24, 36, 60, 120)   # months
LABEL = ("Descriptive horizon study; both holdouts were spent before it was designed, so no "
         "number here is out-of-sample evidence and nothing was selected on it")

ML_START = "1990-01-31"       # returns.config.OOS_START: first origin of the ML copies
EARLY_START = "1960-01-31"    # first origin of the simple forecasters
DEV_END = "2014-12-31"        # development data end (MODELING_PROTOCOL.md)
AFTER_START = "2015-01-31"
PERIODS = ("early", "dev", "after", "full")
MIN_INDEP = 3.0               # fewer independent windows than this: a cell is shown as "thin"

_ROOT = Path(__file__).resolve().parents[3]
_DEV = bool(os.environ.get("ASSETFORECAST_MAX_DATE"))
STORE = _ROOT / "results" / "dev" / "horizons"                      # caches, not committed
TABLES = STORE / "tables" if _DEV else _ROOT / "results" / "horizons"   # score tables
REPORT = _ROOT / "reports" / ("horizon_study_dev.md" if _DEV else "horizon_study.md")
FIGURES = _ROOT / "reports" / "figures" / "horizons"
PAGE = _ROOT / "dashboard" / "horizons.html"


def period_mask(origins, h: int, period: str) -> np.ndarray:
    """Boolean mask over month-end ``origins`` for ``period`` at horizon ``h``."""
    o = pd.DatetimeIndex(origins)
    end = o + pd.offsets.MonthEnd(int(h))
    ml, dev_end, after = (pd.Timestamp(x) for x in (ML_START, DEV_END, AFTER_START))
    if period == "early":
        return np.asarray(o < ml)
    if period == "dev":
        return np.asarray((o >= ml) & (end <= dev_end))
    if period == "after":
        return np.asarray(o >= after)
    if period == "full":
        return np.asarray(o >= ml)
    raise KeyError(period)
