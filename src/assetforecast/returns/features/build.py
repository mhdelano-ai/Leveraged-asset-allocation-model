"""Build the point-in-time feature panel.

    python -m assetforecast.returns.features.build [--max-date D]

writes ``paths.FEATURES`` (long, (date, asset)) and ``paths.FEATURE_META``.
Ported from ``assetpred.features.build``; the inputs now come from the return
panel (``assetforecast.returns.inputs.load_return_panel``: the package panel
without its tier-D months, from 1926-07). :func:`build_features` is a pure
function of the inputs, so the truncation tests can rebuild from inputs cut at
any month.

``--max-date D`` sets ``ASSETFORECAST_MAX_DATE`` before ``paths`` is imported and
builds from the panel as it stood at D. It writes to the same ``paths.FEATURES``
(that path does not move under the development switch): rows dated <= D equal
the full build's, but later rows are absent until the full build is rerun.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from .. import config
from .common import Feat, Inputs
from .predictors import predictor_features
from .price import price_features, realised_variance_us_large

META_COLUMNS = ["name", "family", "scope", "hindsight", "first_date", "description"]


def all_features(inp: Inputs) -> list[Feat]:
    return price_features(inp) + predictor_features(inp, realised_variance_us_large(inp))


def build_features(inp: Inputs) -> tuple[pd.DataFrame, pd.DataFrame]:
    unknown_assets = set(inp.returns.columns) - set(config.ASSETS)
    if unknown_assets:
        raise ValueError(f"returns hold ids outside the universe: {sorted(unknown_assets)}")
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
            extra = set(v.columns) - set(config.ASSETS)
            if extra:  # would be dropped by the reindex below
                raise ValueError(f"feature {f.name} has columns outside the universe: {sorted(extra)}")
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


def load_feature_inputs(max_date=None) -> Inputs:
    """Inputs of the feature build from the return panel (as of ``max_date``)."""
    from ..inputs import load_return_panel
    from .common import inputs_from_panel
    return inputs_from_panel(load_return_panel(max_date))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-date", default=None, help="build from the panel as it stood at this date")
    args = ap.parse_args(argv)
    if args.max_date:
        os.environ["ASSETFORECAST_MAX_DATE"] = args.max_date
    from ... import paths  # after the development switch is set

    features, meta = build_features(load_feature_inputs(args.max_date))
    paths.FEATURES.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(paths.FEATURES)
    meta.to_csv(paths.FEATURE_META, index=False)
    print(f"wrote {paths.FEATURES} {features.shape} and {paths.FEATURE_META}")
    print(meta.groupby(["family", "scope"]).size().to_string())
    print(coverage(features, dates=("1990-01-31", "2014-12-31")).to_string())


if __name__ == "__main__":
    main()
