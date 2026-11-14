"""The Panel: the one object every model and every test receives.

A Panel holds returns, cash and predictors. ``Panel.asof(t)`` returns a copy
that contains nothing dated (or published) after ``t``; the backtest harness
only ever hands models such truncated views, which is how look-ahead is
prevented structurally rather than by convention.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from ..paths import PROCESSED, PREDICTOR_DIR
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
    """

    daily: pd.DataFrame
    monthly: pd.DataFrame
    cash_daily: pd.Series
    cash_monthly: pd.Series
    tiers: pd.DataFrame
    predictors: pd.DataFrame
    asof_date: pd.Timestamp | None = None

    # ------------------------------------------------------------------ views
    def asof(self, t) -> "Panel":
        """Everything known at the close of ``t``. Never returns later data."""
        t = pd.Timestamp(t)
        p = self.predictors
        return replace(
            self,
            daily=self.daily.loc[:t],
            monthly=self.monthly.loc[:t],
            cash_daily=self.cash_daily.loc[:t],
            cash_monthly=self.cash_monthly.loc[:t],
            tiers=self.tiers.loc[:t],
            predictors=p[p["available_date"] <= t],
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
    def predictors_wide(self) -> pd.DataFrame:
        """Predictors as a wide frame indexed by obs_date (latest vintage per obs).

        On an ``asof`` view this contains only values already published.
        """
        p = self.predictors
        if p.empty:
            return pd.DataFrame()
        p = p.sort_values(["series", "obs_date", "available_date"])
        p = p.drop_duplicates(["series", "obs_date"], keep="last")
        return p.pivot(index="obs_date", columns="series", values="value").sort_index()

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
    files = sorted(PREDICTOR_DIR.glob("*.parquet"))
    if files:
        pred = pd.concat([pd.read_parquet(f)[PREDICTOR_COLUMNS] for f in files], ignore_index=True)
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
