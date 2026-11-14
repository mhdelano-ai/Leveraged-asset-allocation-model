"""Expected returns: each asset's expanding historical mean, with an honest interval.

What is published as the expected return
----------------------------------------
``mean_12m`` at origin t for asset a is twelve times the mean monthly log
excess return of a over every month of its history through t, on the
return-side panel (``inputs.load_return_panel``: from 1926-07, without the
month-average tier-D eras). It is identical to the ``hist_mean`` benchmark of
the mean-model replication (``returns.models.baselines.HistoricalMean``),
including its eligibility rule: an asset gets a value only once it has
``config.MIN_ASSET_HISTORY`` (60) months of returns. The assetpred project and
its replication found that nothing beat this number out of sample; it is the
expected return, not a forecast that is believed to be good.

``mean_12m`` is the expected 12-month LOG excess return. The arithmetic
counterpart a planner may want, ``exp(mean_12m + 0.5 var) - 1``, needs the
12-month variance and is computed where the covariance forecast is at hand
(``forecasts/.../expected_returns.csv``, column ``mean_12m_arith``).

Standard error and interval
---------------------------
``se_12m = 12 * sqrt(LRV / n)`` with LRV the long-run variance of the monthly
excess returns from the equal-weighted cosine estimator
(``assetforecast.eval.inference.long_run_variance``, kernel ``"ewc"``) with
``B = EWC_TERMS_RULE(n)`` cosine terms, and the interval is
``mean_12m +/- t_B(0.95) * se_12m`` (Student-t with B degrees of freedom, the
estimator's fixed-b reference distribution, which pays for the noise in the
variance estimate; with 30 years of data B is small and the interval is
correspondingly wider than a normal one).

The rule was chosen on simulated data before any interval on the real panel
was looked at (scratch Monte Carlo, 1500 replications per cell, MC s.e. about
0.8 points; ``tests/returns/test_expected.py`` re-checks two cells). Coverage
of the nominal 90% interval for the mean of a monthly series and the average
width relative to the iid-normal interval, at n = 357 / 448 / 622 / 1204 months
(TIPS, EM, intl equity, the 1926 assets today):

==========================  ======================  ======================  ======================
design (monthly)            iid s.e., normal        EWC B=min(n/4,.4n^2/3)  EWC B=n/48 (chosen)
==========================  ======================  ======================  ======================
iid normal                  91 / 89 / 90 / 91       91 / 89 / 90 / 91 x1.03 90 / 88 / 90 / 91 x1.03-1.11
stoch. vol, t(5) shocks     90 / 89 / 89 / 89       90 / 90 / 89 / 89 x1.02 91 / 88 / 90 / 90 x1.03-1.11
 + AR(1) 0.10               87 / 86 / 86 / 87       90 / 91 / 90 / 91 x1.13 90 / 90 / 89 / 91 x1.14-1.23
 + AR(1) 0.20               82 / 83 / 81 / 82       91 / 90 / 90 / 90 x1.25 91 / 90 / 90 / 90 x1.26-1.37
 + persistent mean (1)      84 / 83 / 83 / 85       88 / 87 / 88 / 89 x1.13 90 / 88 / 87 / 90 x1.16-1.23
 + slow price reversion (2) 90 / 91 / 91 / 91       92 / 92 / 92 / 93 x1.06 91 / 92 / 90 / 92 x1.05-1.14
==========================  ======================  ======================  ======================

(1) AR(1) 0.08 returns plus an expected-return component that is AR(1) with
    coefficient 0.98 and a monthly R^2 of 0.25% (about 3-4% at 12 months).
(2) AR(1) 0.08 returns plus the change of a price deviation that is AR(1) 0.97.
Newey-West with a fixed-b reference behaved like the first EWC rule.
Real monthly first-order autocorrelations run from -0.03 (TIPS) to +0.14
(US small caps). The iid interval under-covers by up to 9 points once returns
are serially correlated or the expected return drifts; the chosen rule
(``assetforecast.eval.inference.default_ewc_terms(n, h=12)``: B = n/48 clipped
to [4, 0.4 n^(2/3)], i.e. only cosine periods of 8 years and longer) holds
87-91% in every design at every length, at the price of intervals 3-11% wider
than iid when returns really are iid. With 357 months (TIPS) B = 7, so the
interval uses t(7): 1.89 standard errors instead of 1.64.

What the interval does NOT cover: a mean that has changed over the sample
(regime shifts, the secular fall in yields behind 40 years of bond returns),
survivorship of the asset class itself, and the choice of start date. It is
the sampling uncertainty of an average of the history at hand, nothing more.

Functions
---------
``expected_returns(panel, origin=None, level=0.90)``  the table at one origin
``history(panel_or_xr, ...)``                         the same at every origin (long)
``mean_se(x)``                                        mean, se and df of one monthly series
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from ..eval.inference import default_ewc_terms, long_run_variance
from . import config
from .inputs import excess_returns, observed_through

__all__ = ["HORIZON", "LEVEL", "EXPECTED_COLUMNS", "HISTORY_COLUMNS", "ewc_terms", "mean_se",
           "expected_returns", "expected_from_xr", "history", "write_history", "load_history"]

HORIZON = 12
LEVEL = 0.90
EXPECTED_COLUMNS = ["mean_12m", "se_12m", "ci90_lo", "ci90_hi", "n_months", "first_month"]
HISTORY_COLUMNS = ["origin", "asset", "mean_12m", "se_12m", "ci90_lo", "ci90_hi", "df",
                   "n_months", "first_month"]
HISTORY_FILE = "history.parquet"


def ewc_terms(n: int) -> int:
    """B = n/48 clipped to [4, 0.4 n^(2/3)] (``default_ewc_terms(n, h=12)``), chosen by simulation.

    Only cosine terms with periods of at least 8 years enter, so slow drifts in
    the expected return (the main threat to a mean's interval) are counted;
    see the module docstring for the Monte Carlo."""
    return int(default_ewc_terms(n, h=12))


def mean_se(x) -> tuple[float, float, float, int]:
    """(mean, se of the mean, t degrees of freedom, n) of a monthly series (NaN dropped).

    The series is treated as contiguous (the return panel has no interior gaps;
    NaN can only lead or trail an asset's history)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 3:
        return (float(x.mean()) if n else np.nan), np.nan, np.nan, n
    r = long_run_variance(x, kernel="ewc", ewc_terms=ewc_terms(n))
    return float(x.mean()), float(math.sqrt(r.lrv / n)), float(r.df), n


def _ci_names(level: float) -> tuple[str, str]:
    k = int(round(100 * level))
    return f"ci{k}_lo", f"ci{k}_hi"


def expected_from_xr(xr: pd.DataFrame, origin=None, level: float = LEVEL,
                     horizon: int = HORIZON,
                     min_history: int = config.MIN_ASSET_HISTORY) -> pd.DataFrame:
    """``expected_returns`` on a wide monthly log excess return frame.

    Uses the rows of ``xr`` dated at or before ``origin`` (default: the last
    row). Assets with fewer than ``min_history`` months are left out.
    """
    if origin is not None:
        xr = xr.loc[:pd.Timestamp(origin)]
    lo_name, hi_name = _ci_names(level)
    rows = {}
    for a in xr.columns:
        s = xr[a].dropna()
        if len(s) < min_history:
            continue
        m, se, df, n = mean_se(s.to_numpy())
        c = stats.t.ppf(0.5 + level / 2.0, df)
        rows[a] = {"mean_12m": horizon * m, "se_12m": horizon * se,
                   lo_name: horizon * (m - c * se), hi_name: horizon * (m + c * se),
                   "n_months": n, "first_month": s.index[0].strftime("%Y-%m")}
    cols = ["mean_12m", "se_12m", lo_name, hi_name, "n_months", "first_month"]
    out = pd.DataFrame.from_dict(rows, orient="index", columns=cols)
    out.index.name = "asset"
    out["n_months"] = out["n_months"].astype(int)
    return out


def expected_returns(panel, origin=None, level: float = LEVEL) -> pd.DataFrame:
    """Expected 12-month log excess return of every eligible asset at ``origin``.

    ``panel``: the return-side panel (``inputs.load_return_panel()``).
    ``origin``: a month end (default: the panel's last complete month).
    Returns a frame indexed by asset (universe order) with columns
    ``mean_12m, se_12m, ci90_lo, ci90_hi, n_months, first_month``
    (the interval columns are named after ``level``). ``mean_12m`` equals the
    ``hist_mean`` benchmark's mean at that origin.
    """
    xr = excess_returns(panel)
    t = observed_through(panel) if origin is None else pd.Timestamp(origin)
    return expected_from_xr(xr, t, level)


def history(xr, level: float = LEVEL, horizon: int = HORIZON,
            min_history: int = config.MIN_ASSET_HISTORY, start=None) -> pd.DataFrame:
    """``expected_returns`` at every origin, long (``HISTORY_COLUMNS``).

    ``xr``: wide monthly log excess returns, or a return-side Panel. One row per
    (origin, asset) with at least ``min_history`` months through the origin.
    ``df`` is the Student-t degrees of freedom behind the interval.
    """
    if not isinstance(xr, pd.DataFrame):
        xr = excess_returns(xr)
    lo_name, hi_name = _ci_names(level)
    recs = []
    for a in xr.columns:
        s = xr[a].dropna()
        if len(s) < min_history:
            continue
        v = s.to_numpy()
        first = s.index[0].strftime("%Y-%m")
        for k in range(min_history, len(s) + 1):
            t = s.index[k - 1]
            if start is not None and t < pd.Timestamp(start):
                continue
            m, se, df, n = mean_se(v[:k])
            c = stats.t.ppf(0.5 + level / 2.0, df)
            recs.append((t, a, horizon * m, horizon * se, horizon * (m - c * se),
                         horizon * (m + c * se), df, n, first))
    cols = ["origin", "asset", "mean_12m", "se_12m", lo_name, hi_name, "df", "n_months", "first_month"]
    out = pd.DataFrame(recs, columns=cols)
    order = {a: i for i, a in enumerate(xr.columns)}
    out = out.sort_values(["origin", "asset"], key=lambda c: c.map(order) if c.name == "asset" else c)
    return out.reset_index(drop=True)


def write_history(hist: pd.DataFrame, root=None) -> Path:
    import json
    if root is None:
        from ... import paths
        root = paths.RESULTS_RETURNS / "expected"
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    p = root / HISTORY_FILE
    hist.to_parquet(p, index=False)
    meta = {"what": "expanding historical mean of 12-month log excess returns at every origin "
                    "(= hist_mean), EWC standard error and Student-t 90% interval",
            "ewc_terms_rule": ewc_terms.__doc__.strip().splitlines()[0],
            "min_history_months": config.MIN_ASSET_HISTORY,
            "panel": f"return panel from {config.RETURN_START}, tiers {config.RETURN_DROP_TIERS} dropped",
            "first_origin": str(hist["origin"].min().date()), "last_origin": str(hist["origin"].max().date()),
            "n_rows": int(len(hist))}
    (root / "history.json").write_text(json.dumps(meta, indent=1))
    return p


def load_history(path=None) -> pd.DataFrame:
    if path is None:
        from ... import paths
        path = paths.RESULTS_RETURNS / "expected" / HISTORY_FILE
    return pd.read_parquet(path)
