"""Run the horizon study and write its tables, report and page.

    python -m assetforecast.horizons.run [--jobs N] [--skip-ml] [--no-report]

Steps: the ML copies (cached; about 25 minutes on 4 cores the first time), the
signal forecasters, the return scores, the risk forecasts and scores, the
economic tests, then ``report`` and ``page``. Tables go to ``horizons.TABLES``
as CSV, each with the study label in ``meta.json`` beside them.
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd

from . import HORIZONS, LABEL, STORE, TABLES


def check_copies(ml: pd.DataFrame) -> dict:
    """At h = 12 a copy must equal the frozen model's committed forecast."""
    from .. import paths
    from .mlcopies import COPIES
    out = {}
    for new, old in COPIES.items():
        p = paths.RESULTS_RETURNS / "forecasts" / old / "h12.parquet"
        a = ml[(ml["model"] == new) & (ml["horizon"] == 12)].set_index(["date", "asset"])["mean"]
        if not p.exists() or a.empty:
            continue
        b = pd.read_parquet(p).set_index(["date", "asset"])["mean"]
        j = pd.concat([a, b], axis=1, keys=["a", "b"])
        out[new] = {"frozen": old, "rows": int(len(j)), "unmatched": int(j.isna().any(axis=1).sum()),
                    "max_abs_diff": float((j["a"] - j["b"]).abs().max())}
    return out


def blank_thin(df: pd.DataFrame) -> pd.DataFrame:
    """No test statistic where there are too few independent windows to estimate one."""
    cols = [c for c in ("cw_t", "cw_p", "dm_t", "dm_p", "t", "p") if c in df.columns]
    df = df.copy()
    df.loc[df["thin"].astype(bool), cols] = np.nan
    return df


def save(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Blank the thin cells and write ``TABLES/<name>.csv``."""
    df = blank_thin(df)
    df.to_csv(TABLES / f"{name}.csv", index=False, float_format="%.6g")
    return df


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m assetforecast.horizons.run")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--skip-ml", action="store_true", help="use the cached ML copies as they are")
    ap.add_argument("--no-report", action="store_true")
    a = ap.parse_args(argv)

    from ..data.panel import load_panel
    from ..returns import inputs, run as rrun
    from . import mlcopies, risk, score, signals, value

    if not a.skip_ml:
        print(mlcopies.run(HORIZONS, a.jobs).to_string(index=False))
    features, xr = rrun.load_inputs(None)
    ml = mlcopies.load(HORIZONS)
    fc = pd.concat([signals.run(features, xr), ml], ignore_index=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    STORE.mkdir(parents=True, exist_ok=True)

    sc = save(score.score(fc, xr), "return_scores")

    panel = load_panel()
    vol, corr = risk.run(panel)
    vol.to_parquet(STORE / "vol.parquet", index=False)
    corr.to_parquet(STORE / "corr.parquet", index=False)
    sv, scr = save(risk.score_vol(vol), "vol_scores"), save(risk.score_corr(corr), "corr_scores")

    rp = inputs.load_return_panel()
    simple = rp.monthly.loc[:xr.index.max()]
    sl = save(value.sleeves(fc, vol, simple, rp.cash_monthly), "sleeves")
    mv = save(value.minvar(vol, corr), "minvar")

    meta = {
        "label": LABEL, "horizons": list(HORIZONS),
        "returns_through": str(xr.index.max().date()),
        "return_models": sorted(set(fc["model"])), "signals": list(signals.SIGNALS),
        "copies_equal_frozen_at_h12": check_copies(ml),
        "risk": {"k": list(risk.K), "min_days": risk.MIN_DAYS,
                 "vol_rows": int(len(vol)), "corr_rows": int(len(corr))},
        "value": {"gamma": value.GAMMA, "cap": value.CAP},
        "rows": {"return_scores": len(sc), "vol_scores": len(sv), "corr_scores": len(scr),
                 "sleeves": len(sl), "minvar": len(mv)},
    }
    (TABLES / "meta.json").write_text(json.dumps(meta, indent=1))
    print(LABEL)
    print(json.dumps(meta["copies_equal_frozen_at_h12"], indent=1))
    if not a.no_report:
        from . import page, report
        print(report.build())
        print(page.build())
    return 0


if __name__ == "__main__":
    sys.exit(main())
