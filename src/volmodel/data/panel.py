"""The Panel: the one object every model and every test receives.

A Panel holds returns, cash and predictors. ``Panel.asof(t)`` returns a copy
that contains nothing dated (or published) after ``t``; the backtest harness
only ever hands models such truncated views, which is how look-ahead is
prevented structurally rather than by convention.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from ..paths import PROCESSED
from ..universe import ASSET_IDS

PREDICTOR_COLUMNS = ["obs_date", "available_date", "series", "value"]


@dataclass(frozen=True)
class Panel:
    """Returns are SIMPLE total returns in decimals; NaN where an asset has no data.

    daily         business-day index, columns are asset ids
    monthly       calendar month-end index, columns are asset ids. Covers at
                  least the daily span and extends earlier where only monthly
                  data exists. Where daily data exists, monthly == compounded daily.
    cash_daily    simple daily T-bill return, same index as ``daily``
    cash_monthly  simple monthly T-bill return, same index as ``monthly``
    tiers         monthly index, data-quality tier letter (A-D) per asset, NaN if no data
    predictors    long frame: obs_date, available_date, series, value
    asof_date     the truncation date of this view (None for the full panel)

    Predictors carry one vintage per (series, obs_date). If a file supplies
    several, the earliest-published one is what a view shows, so nothing is
    ever visible before it was first available.
    """

    daily: pd.DataFrame
    monthly: pd.DataFrame
    cash_daily: pd.Series
    cash_monthly: pd.Series
    tiers: pd.DataFrame
    predictors: pd.DataFrame
    asof_date: pd.Timestamp | None = None
    # shared between a panel and all views cut from it; holds lazily built caches
    _shared: dict = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self):
        if "root_predictors" not in self._shared:
            p = self.predictors
            if len(p) and not p["available_date"].is_monotonic_increasing:
                p = p.sort_values("available_date", kind="stable").reset_index(drop=True)
                object.__setattr__(self, "predictors", p)
            self._shared["root_predictors"] = p

    # ------------------------------------------------------------------ views
    def asof(self, t) -> "Panel":
        """Everything known at the close of ``t``. Never returns later data."""
        t = pd.Timestamp(t)
        p = self.predictors
        k = int(p["available_date"].searchsorted(t, side="right")) if len(p) else 0
        return replace(
            self,
            daily=self.daily.loc[:t],
            monthly=self.monthly.loc[:t],
            cash_daily=self.cash_daily.loc[:t],
            cash_monthly=self.cash_monthly.loc[:t],
            tiers=self.tiers.loc[:t],
            predictors=p.iloc[:k],
            asof_date=t,
        )

    # ---------------------------------------------------------------- returns
    @property
    def excess_daily(self) -> pd.DataFrame:
        """Daily LOG excess returns: log(1+R) - log(1+Rf)."""
        return np.log1p(self.daily).sub(np.log1p(self.cash_daily), axis=0)

    @property
    def excess_monthly(self) -> pd.DataFrame:
        """Monthly LOG excess returns."""
        return np.log1p(self.monthly).sub(np.log1p(self.cash_monthly), axis=0)

    # ------------------------------------------------------------- predictors
    def _wide_cache(self):
        """(values, available_date) wide arrays for the root panel, built once."""
        c = self._shared
        if "wide" not in c:
            p = c["root_predictors"]
            if len(p) == 0:
                c["wide"] = (pd.DataFrame(), None)
            else:
                p = p.drop_duplicates(["series", "obs_date"], keep="first")  # earliest vintage
                vals = p.pivot(index="obs_date", columns="series", values="value").sort_index()
                avail = p.pivot(index="obs_date", columns="series", values="available_date")
                avail = avail.reindex(index=vals.index, columns=vals.columns)
                c["wide"] = (vals, avail.values.astype("datetime64[ns]"))
        return c["wide"]

    def predictors_wide(self) -> pd.DataFrame:
        """Predictors as a wide frame indexed by obs_date, one column per series.

        On an ``asof`` view only values already published by ``asof_date``
        appear; everything else is NaN or absent. Cheap to call repeatedly.
        """
        vals, avail = self._wide_cache()
        if vals.empty or self.asof_date is None:
            return vals.copy()
        t = self.asof_date
        n = int(vals.index.searchsorted(t, side="right"))
        out = vals.iloc[:n].where(avail[:n] <= np.datetime64(t, "ns"))
        return out.dropna(how="all")

    def predictor_latest(self, max_age_days: int | None = None) -> pd.Series:
        """Most recent published value of every series as of this view.

        ``max_age_days`` drops series whose latest observation is older than
        that (dead or discontinued series), measured from ``asof_date``.
        """
        w = self.predictors_wide()
        if w.empty:
            return pd.Series(dtype=float)
        last = w.ffill().iloc[-1]
        if max_age_days is not None:
            ref = self.asof_date if self.asof_date is not None else w.index[-1]
            age = pd.Series({c: (ref - w[c].last_valid_index()).days if w[c].notna().any() else np.inf
                             for c in w.columns})
            last = last[age <= max_age_days]
        return last.dropna()

    # ------------------------------------------------------------ eligibility
    def eligible(self, min_months: int = 36) -> list[str]:
        """Assets with at least ``min_months`` of monthly history in this view."""
        n = self.monthly.notna().sum()
        return [a for a in self.monthly.columns if n[a] >= min_months]

    def has_daily(self, min_days: int = 252) -> list[str]:
        n = self.daily.notna().sum()
        return [a for a in self.daily.columns if n[a] >= min_days]


def load_panel() -> Panel:
    """Load the assembled real-data panel from data/processed."""
    daily = pd.read_parquet(PROCESSED / "panel_daily.parquet")
    monthly = pd.read_parquet(PROCESSED / "panel_monthly.parquet")
    cash_d = pd.read_parquet(PROCESSED / "cash_daily.parquet")["cash"]
    cash_m = pd.read_parquet(PROCESSED / "cash_monthly.parquet")["cash"]
    tiers = pd.read_parquet(PROCESSED / "tiers_monthly.parquet")
    pfile = PROCESSED / "predictors.parquet"
    if pfile.exists():
        pred = pd.read_parquet(pfile)[PREDICTOR_COLUMNS]
        pred["obs_date"] = pd.to_datetime(pred["obs_date"])
        pred["available_date"] = pd.to_datetime(pred["available_date"])
    else:
        pred = pd.DataFrame(columns=PREDICTOR_COLUMNS)
    cols = [a for a in ASSET_IDS if a in daily.columns or a in monthly.columns]
    return Panel(
        daily=daily.reindex(columns=cols),
        monthly=monthly.reindex(columns=cols),
        cash_daily=cash_d.reindex(daily.index),
        cash_monthly=cash_m.reindex(monthly.index),
        tiers=tiers.reindex(columns=cols),
        predictors=pred,
    )
