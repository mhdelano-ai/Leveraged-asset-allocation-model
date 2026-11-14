"""Validation of the asset panel: splice overlaps, ETF/fund comparisons, daily vs monthly.

Writes one row per comparison to data/processed/asset_validation.csv with
monthly statistics over the common months: correlation, annualised means and
vols of both series, mean difference, vol ratio, tracking error (annualised sd
of the difference) and first-order autocorrelations.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import assets_sources as S


def compare(a: pd.Series, b: pd.Series) -> dict:
    j = pd.concat([a, b], axis=1, keys=["a", "b"]).dropna()
    if len(j) < 3:
        return {"n": len(j)}
    x, y = j["a"], j["b"]
    return {
        "start": j.index.min().date(), "end": j.index.max().date(), "n": len(j),
        "corr": x.corr(y),
        "mean_a": x.mean() * 12, "mean_b": y.mean() * 12, "mean_diff": (x - y).mean() * 12,
        "vol_a": x.std() * np.sqrt(12), "vol_b": y.std() * np.sqrt(12),
        "vol_ratio": x.std() / y.std(),
        "tracking_error": (x - y).std() * np.sqrt(12),
        "ac1_a": x.autocorr(), "ac1_b": y.autocorr(),
    }


def validate(panel, cash, segs, cash_segs, validators, daily=None, daily_meta=None, path=None) -> pd.DataFrame:
    rows = []
    end = panel.index.max()
    for asset, seglist in {**segs, "cash": cash_segs}.items():
        for s0, s1 in zip(seglist[:-1], seglist[1:]):
            rows.append({"asset": asset, "check": "splice_overlap", "a": s0.key, "b": s1.key,
                         **compare(s0.series[:end], s1.series[:end])})
    for asset, tickers in validators.items():
        for t in tickers:
            try:
                etf = S.yahoo_monthly_tr(t)
            except Exception as e:  # noqa: BLE001 - validation must not stop the build
                rows.append({"asset": asset, "check": "etf", "a": "final", "b": t, "note": f"unavailable: {e}"})
                continue
            rows.append({"asset": asset, "check": "etf", "a": "final", "b": t, **compare(panel[asset], etf[:end])})
    if daily is not None:
        for asset in daily.columns:
            m = S.compound_monthly(daily[asset])
            m = m[m.index > daily[asset].first_valid_index() + pd.offsets.MonthEnd(0)]  # drop partial first month
            src = "daily_compounded"
            if daily_meta is not None:
                srcs = daily_meta.loc[daily_meta["asset"] == asset, "source"].tolist()
                src = "daily_compounded: " + " -> ".join(srcs)
            rows.append({"asset": asset, "check": "daily_vs_monthly", "a": "monthly_panel", "b": src,
                         **compare(panel[asset], m)})
    out = pd.DataFrame(rows)
    if path is not None:
        out.round(6).to_csv(path, index=False)
    return out
