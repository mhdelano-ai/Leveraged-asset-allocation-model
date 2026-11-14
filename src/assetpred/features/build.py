"""Build the point-in-time feature panel.

    python -m assetpred.features.build

writes ``data/processed/features.parquet`` (long, (date, asset)) and
``data/processed/feature_meta.csv``. :func:`build_features` is a pure function
of the inputs, so the truncation test can rebuild from inputs cut at any month.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from .. import config, schema
from .common import Feat, Inputs, load_inputs
from .predictors import predictor_features
from .price import price_features, realised_variance_us_eq

META_COLUMNS = ["name", "family", "scope", "hindsight", "first_date", "description"]


def all_features(inp: Inputs) -> list[Feat]:
    return price_features(inp) + predictor_features(inp, realised_variance_us_eq(inp))


def build_features(inp: Inputs) -> tuple[pd.DataFrame, pd.DataFrame]:
    feats = all_features(inp)
    names = [f.name for f in feats]
    if len(set(names)) != len(names):
        raise ValueError("duplicate feature names")
    unknown_src = {s for f in feats for s in f.sources} - set(inp.predictor_meta["name"])
    if unknown_src:
        raise ValueError(f"features reference unknown predictors {sorted(unknown_src)}")

    rets = inp.returns
    assets = list(rets.columns)
    stacked = rets.stack().dropna()  # one row per month with a return
    index = pd.MultiIndex.from_arrays(
        [pd.DatetimeIndex(stacked.index.get_level_values(0), name="date"),
         pd.Index(stacked.index.get_level_values(1).astype(str), name="asset")])
    dates = rets.index
    di = dates.get_indexer(index.get_level_values("date"))
    ai = pd.Index(assets).get_indexer(index.get_level_values("asset"))

    cols, meta = {}, []
    hs = inp.hindsight
    for f in feats:
        v = f.values
        if isinstance(v, pd.DataFrame):
            arr = v.reindex(index=dates, columns=assets).to_numpy(dtype=float)[di, ai]
        else:
            arr = v.reindex(dates).to_numpy(dtype=float)[di]
        arr[~np.isfinite(arr)] = np.nan
        cols[f.name] = arr
        meta.append({"name": f.name, "family": f.family, "scope": f.scope,
                     "hindsight": bool(set(f.sources) & hs), "description": f.description})
    out = pd.DataFrame(cols, index=index).sort_index()
    meta = pd.DataFrame(meta)
    first = {c: out[c].dropna().index.get_level_values("date").min() for c in out}
    meta["first_date"] = meta["name"].map(lambda c: "" if pd.isna(first[c]) else first[c].strftime("%Y-%m-%d"))
    return out, meta[META_COLUMNS]


def coverage(features: pd.DataFrame, dates=("1990-01-31", "2005-01-31")) -> pd.DataFrame:
    """Number of non-missing features per asset on selected dates (and the last date)."""
    last = features.index.get_level_values("date").max()
    rows = {}
    for d in [*map(pd.Timestamp, dates), last]:
        x = features.xs(d, level="date")
        rows[d.strftime("%Y-%m")] = x.notna().sum(axis=1)
    return pd.DataFrame(rows).reindex(list(config.ASSETS))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.parse_args(argv)
    features, meta = build_features(load_inputs())
    features.to_parquet(schema.FEATURES)
    meta.to_csv(schema.FEATURE_META, index=False)
    print(f"wrote {schema.FEATURES} {features.shape} and {schema.FEATURE_META}")
    print(meta.groupby(["family", "scope"]).size().to_string())
    print(coverage(features).to_string())


if __name__ == "__main__":
    main()
