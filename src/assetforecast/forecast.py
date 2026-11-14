"""Produce the current forecast: one command, a handful of plain files.

    python -m assetforecast.forecast [--model combo_cov_main] [--bridge vr_market] [--out forecasts]

Reads the selected covariance model's stored forecast at its latest origin
(run ``scripts/run_all.sh`` first so every model is up to date), bridges it to
the covariance of 12-month returns, and writes

    forecasts/<origin>/forecast.json   everything below, plus metadata
    forecasts/<origin>/vols.csv        per asset: 21-day-horizon vol, 12-month-return vol,
                                       variance ratio, the range across model families,
                                       recent and long-run realized vol, tail quantiles
    forecasts/<origin>/corr.csv        forecast correlation matrix
    forecasts/<origin>/cov_annual.csv  covariance of 12-month log excess returns
    forecasts/<origin>/cov_21d.csv     annualized covariance of 21-day log excess returns
    forecasts/<origin>/portfolios.csv  the same risk figures for the policy portfolios
    forecasts/latest/                  a copy of the newest origin

All returns are log returns in excess of T-bills. Quantiles are of the
12-month return MINUS ITS MEAN: this project does not forecast expected
returns, so add your own.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from .bridge import ScaleMixture, get_bridge, predictive_quantiles, vol_uncertainty
from .data.panel import load_panel
from .eval import harness
from .eval.portfolio import POLICIES
from .eval.targets import realized_kday_cov
from .paths import RESULTS, ROOT
from .universe import BY_ID

PRIMARY, BRIDGE = "combo_cov_main", "vr_market"


def _family(model: str) -> str:
    return model.split("_", 1)[0]


def family_range(origin, assets, results_dir) -> pd.DataFrame:
    """10th/50th/90th percentile, across model families, of each family's mean vol forecast."""
    rows = {}
    for m in harness.stored_models("vol", results_dir):
        fam = _family(m)
        if fam in ("combo", "scratch"):
            continue
        f = harness.load_forecasts("vol", m, results_dir).get(origin)
        if f is not None:
            rows.setdefault(fam, []).append(f.reindex(assets))
    fam_mean = pd.DataFrame({k: pd.concat(v, axis=1).mean(axis=1) for k, v in rows.items()})
    q = fam_mean.quantile([0.1, 0.5, 0.9], axis=1).T
    q.columns = ["family_p10", "family_median", "family_p90"]
    q["n_families"] = fam_mean.notna().sum(axis=1)
    return q


def realized_context(panel, origin, assets) -> pd.DataFrame:
    """What actually happened: last-12-month and since-inception vol at the 21-day horizon."""
    x = panel.excess_daily.loc[:origin, assets]
    last = x.loc[origin - pd.offsets.MonthEnd(12) + pd.Timedelta(days=1):]
    cov, _ = realized_kday_cov(last.to_numpy(float), 21)
    xm = panel.excess_monthly.loc[:origin, assets]
    return pd.DataFrame({
        "realized_last_12m": np.sqrt(np.diag(cov)),
        "realized_long_run": np.sqrt(12.0 * (xm ** 2).mean()),
        "history_starts": [panel.monthly[a].first_valid_index().strftime("%Y-%m") for a in assets],
    }, index=assets)


def build(model: str = PRIMARY, bridge: str = BRIDGE, results_dir: Path | None = None) -> dict:
    results_dir = Path(results_dir) if results_dir else RESULTS
    panel = load_panel()
    stored = harness.load_forecasts("cov", model, results_dir)
    origin = max(stored)
    cov_rv = stored[origin]
    assets = list(cov_rv.index)
    view = panel.asof(origin)
    if view.daily.index.max() < origin - pd.Timedelta(days=5):
        raise RuntimeError(f"the panel ends {view.daily.index.max().date()}, before the forecast origin {origin.date()}")

    cov_a = get_bridge(bridge).cov_annual(cov_rv, view)
    v21, v12 = np.sqrt(np.diag(cov_rv)), np.sqrt(np.diag(cov_a))
    corr = cov_rv / np.outer(v21, v21)
    s2 = float(vol_uncertainty(view)["s2"])
    mix = ScaleMixture(s2)

    q_assets = predictive_quantiles(cov_a, None, s2=s2)
    vols = pd.DataFrame({
        "name": [BY_ID[a].name for a in assets],
        "group": [BY_ID[a].group for a in assets],
        "vol_21d_horizon": v21,
        "vol_12m_return": v12,
        "variance_ratio": (v12 / v21) ** 2,
        "q05_12m_ex_mean": q_assets["q0.05_mixture"].to_numpy(),
        "q01_12m_ex_mean": q_assets["q0.01_mixture"].to_numpy(),
    }, index=assets)
    vols = vols.join(family_range(origin, assets, results_dir)).join(realized_context(panel, origin, assets))
    vols.index.name = "asset"

    W = pd.DataFrame({k: pd.Series(p["weights"]) for k, p in POLICIES.items()}).T.reindex(columns=assets).fillna(0.0)
    W = W.div(W.sum(axis=1), axis=0)
    qp = predictive_quantiles(cov_a, W, s2=s2)
    ports = pd.DataFrame({
        "vol_21d_horizon": np.sqrt(np.einsum("pi,ij,pj->p", W.to_numpy(), cov_rv.to_numpy(), W.to_numpy())),
        "vol_12m_return": qp["sigma"].to_numpy(),
        "q05_12m_ex_mean": qp["q0.05_mixture"].to_numpy(),
        "q01_12m_ex_mean": qp["q0.01_mixture"].to_numpy(),
    }, index=W.index)
    ports.index.name = "portfolio"

    meta = {
        "origin": origin.strftime("%Y-%m-%d"),
        "window": f"{(origin + pd.Timedelta(days=1)).strftime('%Y-%m-%d')} to {(origin + pd.offsets.MonthEnd(12)).strftime('%Y-%m-%d')}",
        "data_through": view.daily.index.max().strftime("%Y-%m-%d"),
        "model": model, "bridge": bridge,
        "returns": "log returns in excess of T-bills",
        "tail_model": {"type": "lognormal scale mixture", "s2": round(s2, 4), "kurtosis": round(mix.kurtosis, 2),
                       "q05_in_sigmas": round(mix.quantile(0.05), 3), "q01_in_sigmas": round(mix.quantile(0.01), 3)},
        "note": "Quantiles are of the 12-month return minus its mean; expected returns are not forecast here.",
    }
    return {"meta": meta, "vols": vols, "corr": corr, "cov_annual": cov_a, "cov_21d": cov_rv,
            "portfolios": ports, "weights": W}


def write(result: dict, out: Path) -> Path:
    d = Path(out) / result["meta"]["origin"]
    d.mkdir(parents=True, exist_ok=True)
    result["vols"].round(5).to_csv(d / "vols.csv")
    result["corr"].round(4).to_csv(d / "corr.csv")
    result["cov_annual"].round(8).to_csv(d / "cov_annual.csv")
    result["cov_21d"].round(8).to_csv(d / "cov_21d.csv")
    result["portfolios"].round(5).to_csv(d / "portfolios.csv")
    payload = {
        **result["meta"],
        "assets": json.loads(result["vols"].round(5).reset_index().to_json(orient="records")),
        "portfolios": json.loads(result["portfolios"].round(5).reset_index().to_json(orient="records")),
        "portfolio_weights": json.loads(result["weights"].round(4).to_json(orient="index")),
        "corr": {"assets": list(result["corr"].index), "matrix": result["corr"].round(4).to_numpy().tolist()},
        "cov_annual": {"assets": list(result["cov_annual"].index),
                       "matrix": result["cov_annual"].round(8).to_numpy().tolist()},
    }
    (d / "forecast.json").write_text(json.dumps(payload, indent=1))
    latest = Path(out) / "latest"
    if latest.exists():
        shutil.rmtree(latest)
    shutil.copytree(d, latest)
    return d


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default=PRIMARY)
    ap.add_argument("--bridge", default=BRIDGE)
    ap.add_argument("--results", default=None, help="results directory holding the stored forecasts")
    ap.add_argument("--out", default=str(ROOT / "forecasts"))
    a = ap.parse_args(argv)
    res = build(a.model, a.bridge, a.results)
    d = write(res, Path(a.out))
    pd.set_option("display.width", 200)
    print(f"forecast for {res['meta']['window']} (model {a.model}, bridge {a.bridge}) -> {d}")
    print(res["vols"][["vol_21d_horizon", "vol_12m_return", "family_p10", "family_p90",
                       "realized_last_12m", "realized_long_run"]].round(3).to_string())


if __name__ == "__main__":
    main()
