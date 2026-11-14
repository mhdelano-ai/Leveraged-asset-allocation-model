"""Per-origin history of the selected covariance forecast, 21-day and 12-month.

    python -m assetforecast.covhistory [--model combo_cov_main] [--bridge vr_market] [--max-date D]
                                       [--truncation-origins N]

``forecast.py`` publishes the selected model's covariance at the LATEST origin
only. The return side (distributions centred on historical means, the
strategy test) needs it at every past origin. This module rebuilds it with the
package's own forecast path, origin by origin:

* ``cov_21d``    the stored forecast of the selected model
                 (``<results>/forecasts/cov/<model>.parquet``): annualized
                 covariance of 21-trading-day log excess returns;
* ``cov_annual`` ``get_bridge(bridge).cov_annual(cov_21d, panel.asof(origin))``:
                 the covariance of the 12-month log excess return, exactly as
                 ``forecast.build`` computes it at its origin.

The panel is the FULL panel with no quality tiers dropped: the bridge
estimates its variance ratios from the whole monthly history, including the
month-average (tier D) eras that it classifies itself. A non-empty
``ASSETFORECAST_DROP_TIERS`` in the environment therefore makes ``build``
refuse rather than silently produce a different history.

How the bridge's parameters evolve
----------------------------------
``LongRunBridge`` re-estimates its variance ratios at most once a year: at
origin ``t`` from the view truncated at the last December month-end ``<= t``
(the *bridge anchor*), cached by a fingerprint of the monthly data up to the
anchor and by the asset set. Within a calendar year the ratios change only
when the eligible asset set changes; the 12-month matrix still changes every
month because ``cov_21d`` does. One bridge instance is reused over all
origins, so each (anchor, asset set) is estimated once; this is what
``forecast.py`` (fresh instance, one origin) and ``scripts/score_final.py``
(one instance over all origins) compute, because the cached estimate depends
only on the view, never on the call order ("cold equals warm").

Tail dispersion ``s2`` (``bridge.vol_uncertainty``) is recorded per origin
under the two conventions the package uses: ``tail_s2_origin`` on the view at
the origin (the convention of the old scale-mixture quantiles, which
``forecast.py`` no longer publishes; ``results/final`` keeps them) and
``tail_s2_anchor`` on the view at the bridge anchor (what
``score_final.stage_bridge`` uses).

Outputs (``write`` / the command):

    results/covariance/cov_history.parquet        origin, asset_i, asset_j, cov_21d, cov_annual
                                                  (full square matrix per origin, asset order as stored)
    results/covariance/cov_history_origins.csv    origin, n_assets, assets, bridge_anchor,
                                                  bridge_tau2_vr, tail_s2_origin, tail_s2_anchor
    results/covariance/meta.json                  model, bridge, data and code identity, tiers, checks
    results/returns/covariances/af_primary/h12.parquet   cov_annual in the return side's COV_COLUMNS

Public API::

    build(model, bridge, results_dir) -> (hist_df, origins_df)
    history_from_forecasts(stored, panel, bridge) -> (hist_df, origins_df)
    write(hist_df, origins_df, path)
    load(kind="annual" | "21d", path) -> {origin: asset x asset DataFrame}
    to_cov_columns(hist_df, name="af_primary", horizon=12) -> DataFrame[COV_COLUMNS]
    check_matrices(hist_df) -> dict ; truncation_check(hist_df, origins, ...) -> DataFrame
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import paths
from .bridge import _anchor, get_bridge, vol_uncertainty
from .data.panel import Panel, load_panel
from .eval import harness

PRIMARY, BRIDGE = "combo_cov_main", "vr_market"
HIST_COLUMNS = ["origin", "asset_i", "asset_j", "cov_21d", "cov_annual"]
ORIGIN_COLUMNS = ["origin", "n_assets", "assets", "bridge_anchor", "bridge_tau2_vr",
                  "tail_s2_origin", "tail_s2_anchor"]
ORIGINS_FILE = "cov_history_origins.csv"
META_FILE = "meta.json"
COV_NAME = "af_primary"
_DROP_ENV = "ASSETFORECAST_DROP_TIERS"
_MAX_ENV = "ASSETFORECAST_MAX_DATE"
_PD_REL_TOL = 1e-12   # min eigenvalue / mean diagonal below this counts as not positive definite


class CovHistoryError(RuntimeError):
    """The history could not be built as the package would have built it."""


# ------------------------------------------------------------------ build
def _refuse_dropped_tiers() -> None:
    v = os.environ.get(_DROP_ENV, "")
    if v:
        raise CovHistoryError(
            f"{_DROP_ENV}={v!r} is set: the covariance history must be built on the full panel "
            "(the bridge needs the month-average history). Unset it and run again.")


def history_from_forecasts(stored: dict, panel: Panel, bridge: str = BRIDGE,
                           min_months: int = harness.MIN_MONTHS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Bridge every stored 21-day forecast to 12 months with the panel as of its origin.

    ``stored``: {origin: asset x asset DataFrame} as ``harness.load_forecasts``
    returns it. Each forecast's asset set must equal ``view.eligible(min_months)``
    (the harness contract); a mismatch raises.
    """
    b = get_bridge(bridge)
    blocks, rows, s2_memo = [], [], {}
    for o in sorted(stored):
        o = pd.Timestamp(o)
        C = stored[o]
        view = panel.asof(o)
        assets = list(C.index)
        elig = view.eligible(min_months)
        if sorted(assets) != sorted(elig):
            raise CovHistoryError(f"origin {o.date()}: stored assets {assets} != eligible {elig}")
        A = b.cov_annual(C, view)
        est = b.estimate(view, assets)          # cached: the estimate cov_annual just used
        an = _anchor(view)
        if an not in s2_memo:
            s2_memo[an] = float(vol_uncertainty(view.asof(an))["s2"])
        n = len(assets)
        blocks.append(pd.DataFrame({
            "origin": o,
            "asset_i": np.repeat(assets, n).astype(str),
            "asset_j": np.tile(assets, n).astype(str),
            "cov_21d": C.to_numpy(float).ravel(),
            "cov_annual": A.loc[assets, assets].to_numpy(float).ravel(),
        }))
        rows.append({"origin": o, "n_assets": n, "assets": "|".join(assets), "bridge_anchor": an,
                     "bridge_tau2_vr": float(est.tau2_vr),
                     "tail_s2_origin": float(vol_uncertainty(view)["s2"]),
                     "tail_s2_anchor": s2_memo[an]})
    if not blocks:
        raise CovHistoryError("no stored forecasts to bridge")
    hist = pd.concat(blocks, ignore_index=True)[HIST_COLUMNS]
    origins = pd.DataFrame(rows)[ORIGIN_COLUMNS]
    return hist, origins


def build(model: str = PRIMARY, bridge: str = BRIDGE, results_dir: Path | None = None
          ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The selected model's 21-day forecast and its 12-month bridge at every stored origin.

    ``results_dir`` defaults to ``paths.RESULTS_FULL`` (the covariance harness
    store). Under ``ASSETFORECAST_MAX_DATE`` the panel and the origins stop there.
    ``hist_df.attrs["meta"]`` carries the identity recorded by ``write``.
    """
    _refuse_dropped_tiers()
    results_dir = Path(results_dir) if results_dir is not None else paths.RESULTS_FULL
    t0 = time.time()
    max_date = os.environ.get(_MAX_ENV) or None
    panel = load_panel(max_date=max_date, drop_tiers="")
    src = harness.forecast_path("cov", model, results_dir)
    stored = harness.load_forecasts("cov", model, results_dir)
    if max_date:
        stored = {o: c for o, c in stored.items() if o <= pd.Timestamp(max_date)}
    if not stored:
        raise CovHistoryError(f"no stored forecasts for {model!r} in {results_dir}")
    last = max(stored)
    view = panel.asof(last)
    if view.daily.index.max() < last - pd.Timedelta(days=5):   # the check forecast.build makes
        raise CovHistoryError(f"the panel ends {view.daily.index.max().date()}, "
                              f"before the latest forecast origin {last.date()}")
    hist, origins = history_from_forecasts(stored, panel, bridge)
    hist.attrs["meta"] = _meta(model, bridge, results_dir, src, panel, max_date, time.time() - t0)
    return hist, origins


# ------------------------------------------------------------------ identity
def _sha256(p: Path) -> str | None:
    p = Path(p)
    if not p.exists():
        return None
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def _git_head() -> str | None:
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=paths.ROOT, capture_output=True,
                           text=True, timeout=10)
        return r.stdout.strip() or None
    except Exception:
        return None


def _meta(model, bridge, results_dir, src, panel, max_date, seconds) -> dict:
    from .eval.targets import panel_fingerprint
    here = Path(__file__).resolve().parent
    tiers = panel.tiers.stack().value_counts().sort_index()
    model_meta = src.with_suffix(".meta.json")
    return {
        "model": model,
        "bridge": bridge,
        "returns": "log returns in excess of T-bills",
        "cov_21d": "stored forecast: annualized covariance of overlapping 21-trading-day returns",
        "cov_annual": f"get_bridge({bridge!r}).cov_annual(cov_21d, panel.asof(origin)): "
                      "covariance of the 12-month return",
        "bridge_reestimation": "once a year, from the view at the last December month-end <= origin "
                               "(bridge_anchor), per eligible asset set",
        "tail_s2": {"tail_s2_origin": "vol_uncertainty(panel.asof(origin)); forecast.py's tail_model.s2",
                    "tail_s2_anchor": "vol_uncertainty(panel.asof(bridge_anchor)); score_final.stage_bridge's s2"},
        "panel": {
            "drop_tiers": "",
            "tiers_dropped": [],
            "note": "full panel, no quality tiers dropped; ASSETFORECAST_DROP_TIERS was unset at build",
            "env_ASSETFORECAST_DROP_TIERS": os.environ.get(_DROP_ENV) or None,
            "max_date": max_date,
            "asset_months_by_tier": {str(k): int(v) for k, v in tiers.items()},
            "monthly_start": str(panel.monthly.index.min().date()),
            "monthly_end": str(panel.monthly.index.max().date()),
            "daily_end": str(panel.daily.index.max().date()),
            "fingerprint": panel_fingerprint(panel),
        },
        "source": {
            "results_dir": str(results_dir),
            "forecast_file": str(src.relative_to(paths.ROOT)) if src.is_relative_to(paths.ROOT) else str(src),
            "forecast_sha256_16": _sha256(src),
            "forecast_run_at": (json.loads(model_meta.read_text()).get("run_at")
                                if model_meta.exists() else None),
        },
        "code": {
            "git_head": _git_head(),
            "covhistory_sha256_16": _sha256(Path(__file__)),
            "bridge_sha256_16": _sha256(here / "bridge" / "__init__.py"),
            "panel_sha256_16": _sha256(here / "data" / "panel.py"),
        },
        "build_seconds": round(seconds, 1),
    }


# ------------------------------------------------------------------ checks
def _matrices(hist: pd.DataFrame, col: str) -> dict:
    out = {}
    for o, g in hist.groupby("origin", sort=True):
        names = list(dict.fromkeys(g["asset_i"]))
        n = len(names)
        if len(g) != n * n or list(g["asset_j"].iloc[:n]) != names:
            w = g.pivot(index="asset_i", columns="asset_j", values=col).reindex(index=names, columns=names)
        else:
            w = pd.DataFrame(g[col].to_numpy(float).reshape(n, n), index=names, columns=names)
        out[pd.Timestamp(o)] = w
    return out


def check_matrices(hist: pd.DataFrame) -> dict:
    """Symmetry and positive definiteness of every matrix; raises nothing, reports all."""
    res = {}
    for col in ("cov_21d", "cov_annual"):
        asym, rel_min, n_bad, worst = 0.0, np.inf, 0, None
        for o, M in _matrices(hist, col).items():
            a = M.to_numpy(float)
            if not np.isfinite(a).all():
                n_bad += 1
                continue
            asym = max(asym, float(np.abs(a - a.T).max() / np.abs(np.diag(a)).max()))
            r = float(np.linalg.eigvalsh((a + a.T) / 2).min() / np.diag(a).mean())
            if r < rel_min:
                rel_min, worst = r, o
            if r <= _PD_REL_TOL:
                n_bad += 1
        res[col] = {"max_rel_asymmetry": asym, "min_eig_over_mean_diag": rel_min,
                    "worst_origin": str(worst.date()) if worst is not None else None,
                    "n_not_pd": n_bad}
    res["n_origins"] = int(hist["origin"].nunique())
    res["all_pd"] = all(res[c]["n_not_pd"] == 0 for c in ("cov_21d", "cov_annual"))
    return res


def truncation_check(hist: pd.DataFrame, origins, bridge: str = BRIDGE) -> pd.DataFrame:
    """Recompute ``cov_annual`` at ``origins`` from a panel LOADED truncated at the origin.

    ``load_panel(max_date=origin)`` (nothing after the origin is read), a fresh
    bridge instance (no cache carried over), the stored ``cov_21d``. Returns
    the largest absolute and relative difference per origin.
    """
    _refuse_dropped_tiers()
    C21, CA = _matrices(hist, "cov_21d"), _matrices(hist, "cov_annual")
    rows = []
    for o in origins:
        o = pd.Timestamp(o)
        p = load_panel(max_date=o, drop_tiers="")
        A = get_bridge(bridge).cov_annual(C21[o], p.asof(o)).to_numpy(float)
        ref = CA[o].to_numpy(float)
        d = np.abs(A - ref)
        rows.append({"origin": o, "panel_end": p.monthly.index.max(), "n_assets": len(ref),
                     "max_abs_diff": float(d.max()), "max_rel_diff": float((d / np.abs(np.diag(ref)).max()).max())})
    return pd.DataFrame(rows)


def compare_latest(hist: pd.DataFrame, forecast_dir: Path | None = None) -> dict:
    """Last origin's matrices against the published ``forecasts/latest`` (rounded to 8 decimals there)."""
    d = Path(forecast_dir) if forecast_dir is not None else paths.FORECASTS_OUT / "latest"
    meta = json.loads((d / "forecast.json").read_text())
    last = hist["origin"].max()
    out = {"history_last_origin": str(pd.Timestamp(last).date()), "published_origin": meta["origin"]}
    for col, f in (("cov_annual", "cov_annual.csv"), ("cov_21d", "cov_21d.csv")):
        pub = pd.read_csv(d / f, index_col=0)
        M = _matrices(hist[hist["origin"] == last], col)[pd.Timestamp(last)]
        same_labels = list(pub.index) == list(M.index) and list(pub.columns) == list(M.columns)
        out[col] = {"same_labels": same_labels,
                    "max_abs_diff": float(np.abs(M.reindex_like(pub).to_numpy() - pub.to_numpy()).max())}
    return out


def compare_dashboard_vol(hist: pd.DataFrame, path: Path | None = None) -> dict:
    """sqrt(diag cov_21d) against ``results/final/dashboard_vol.csv`` ``primary`` (4 decimals there)."""
    p = Path(path) if path is not None else paths.RESULTS_FINAL / "dashboard_vol.csv"
    dv = pd.read_csv(p)
    d = hist[hist["asset_i"] == hist["asset_j"]]
    mine = pd.DataFrame({"origin": pd.DatetimeIndex(d["origin"]).strftime("%Y-%m"), "asset": d["asset_i"].to_numpy(),
                         "vol": np.sqrt(d["cov_21d"].to_numpy())})
    m = mine.merge(dv[["origin", "asset", "primary"]], on=["origin", "asset"], how="outer", indicator=True)
    both = m[m["_merge"] == "both"]
    diff = (both["vol"].round(4) - both["primary"]).abs()
    return {"n_cells": int(len(both)), "only_history": int((m["_merge"] == "left_only").sum()),
            "only_dashboard": int((m["_merge"] == "right_only").sum()),
            "n_primary_missing": int(both["primary"].isna().sum()),
            "max_abs_diff_after_rounding": float(diff.max()),
            "n_cells_differ_after_rounding": int((diff > 1e-9).sum()),
            "max_abs_diff_unrounded": float((both["vol"] - both["primary"]).abs().max())}


# ------------------------------------------------------------------ files
def write(hist_df: pd.DataFrame, origins_df: pd.DataFrame, path: Path | None = None,
          checks: dict | None = None) -> Path:
    """Write the history, the per-origin facts and ``meta.json`` next to each other."""
    path = Path(path) if path is not None else paths.COV_HISTORY
    path.parent.mkdir(parents=True, exist_ok=True)
    hist = hist_df[HIST_COLUMNS].copy()
    hist["origin"] = pd.to_datetime(hist["origin"]).astype("datetime64[ns]")
    hist.to_parquet(path, index=False)
    o = origins_df[ORIGIN_COLUMNS].copy()
    for c in ("origin", "bridge_anchor"):
        o[c] = pd.DatetimeIndex(o[c]).strftime("%Y-%m-%d")
    o.to_csv(path.parent / ORIGINS_FILE, index=False, float_format="%.10g")
    meta = dict(hist_df.attrs.get("meta", {}))
    meta.update({
        "n_origins": int(hist["origin"].nunique()),
        "first_origin": str(hist["origin"].min().date()), "last_origin": str(hist["origin"].max().date()),
        "n_rows": int(len(hist)),
        "assets_per_origin": {str(k): int(v) for k, v in origins_df["n_assets"].value_counts().sort_index().items()},
        "first_origin_by_asset": {a: str(g["origin"].min().date())
                                  for a, g in hist[hist["asset_i"] == hist["asset_j"]].groupby("asset_i")},
        "files": {"history": path.name, "origins": ORIGINS_FILE},
        "checks": checks if checks is not None else check_matrices(hist),
        "written": pd.Timestamp.now().isoformat(timespec="seconds"),
    })
    (path.parent / META_FILE).write_text(json.dumps(meta, indent=1, default=str))
    return path


def load(kind: str = "annual", path: Path | None = None) -> dict:
    """{origin: asset x asset DataFrame} of ``cov_annual`` (``kind="annual"``) or ``cov_21d``."""
    col = {"annual": "cov_annual", "21d": "cov_21d"}.get(kind)
    if col is None:
        raise ValueError("kind must be 'annual' or '21d'")
    path = Path(path) if path is not None else paths.COV_HISTORY
    hist = pd.read_parquet(path, columns=["origin", "asset_i", "asset_j", col])
    return _matrices(hist, col)


def to_cov_columns(hist_df: pd.DataFrame, name: str = COV_NAME, horizon: int = 12) -> pd.DataFrame:
    """``cov_annual`` in the return side's layout (``returns.schema.COV_COLUMNS``)."""
    from .returns.schema import COV_COLUMNS
    if int(horizon) != 12:
        raise ValueError("the bridge produces the 12-month covariance only")
    df = pd.DataFrame({
        "model": name, "horizon": np.int64(horizon),
        "date": pd.to_datetime(hist_df["origin"]).to_numpy(),
        "asset_i": hist_df["asset_i"].astype(str).to_numpy(),
        "asset_j": hist_df["asset_j"].astype(str).to_numpy(),
        "cov": hist_df["cov_annual"].to_numpy(float),
    })
    return df[COV_COLUMNS]


def cov_columns_path(name: str = COV_NAME, horizon: int = 12) -> Path:
    return paths.RESULTS_RETURNS / "covariances" / name / f"h{int(horizon):02d}.parquet"


# ------------------------------------------------------------------ CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default=PRIMARY)
    ap.add_argument("--bridge", default=BRIDGE)
    ap.add_argument("--results", default=None, help="covariance harness store (default results/full)")
    ap.add_argument("--max-date", default=None, help="development cut-off; outputs go under results/dev")
    ap.add_argument("--truncation-origins", type=int, default=6,
                    help="number of origins re-computed from a truncated panel (0 to skip)")
    a = ap.parse_args(argv)
    if a.max_date:
        os.environ[_MAX_ENV] = str(pd.Timestamp(a.max_date).date())
        importlib.reload(paths)   # RESULTS_COV / RESULTS_RETURNS move under results/dev
    hist, origins = build(a.model, a.bridge, a.results)
    checks = check_matrices(hist)
    if not checks["all_pd"]:
        raise CovHistoryError(f"matrices not positive definite: {checks}")
    if a.truncation_origins:
        ors = sorted(hist["origin"].unique())
        pick = sorted({ors[int(i)] for i in np.linspace(0, len(ors) - 1, a.truncation_origins)})
        tc = truncation_check(hist, pick, a.bridge)
        checks["truncation"] = {"origins": [str(pd.Timestamp(o).date()) for o in tc["origin"]],
                                "max_abs_diff": float(tc["max_abs_diff"].max())}
        print(tc.to_string(index=False))
    if not os.environ.get(_MAX_ENV):
        for k, fn in (("latest_forecast", compare_latest), ("dashboard_vol", compare_dashboard_vol)):
            try:
                checks[k] = fn(hist)
            except FileNotFoundError as exc:
                checks[k] = {"skipped": str(exc)}
    p = write(hist, origins, checks=checks)
    cp = cov_columns_path()
    cp.parent.mkdir(parents=True, exist_ok=True)
    to_cov_columns(hist).to_parquet(cp, index=False)
    print(json.dumps(checks, indent=1, default=str))
    print(f"{hist['origin'].nunique()} origins -> {p} and {cp}")


if __name__ == "__main__":
    main()
