"""Produce the current forecast: one command, a handful of plain files.

    python -m assetforecast.forecast [--model combo_cov_main] [--bridge vr_market]
                                     [--results DIR] [--out DIR] [--max-date D]

Reads the selected covariance model's stored forecast at its latest origin
(``results/full``, written by ``scripts/run_all.sh``), bridges it to the
covariance of 12-month returns, checks that this is the matrix the covariance
history (``python -m assetforecast.covhistory``) holds for that origin, adds
the expected returns and the predictive distributions of the return side, and
writes

    forecasts/<origin>/forecast.json              everything below, plus metadata and the method
    forecasts/<origin>/vols.csv                   per asset: 21-day-horizon vol, 12-month-return vol,
                                                  variance ratio, the range across model families,
                                                  recent and long-run realized vol
    forecasts/<origin>/corr.csv                   forecast correlation matrix
    forecasts/<origin>/cov_annual.csv             covariance of 12-month log excess returns
    forecasts/<origin>/cov_21d.csv                annualized covariance of 21-day log excess returns
    forecasts/<origin>/portfolios.csv             the same risk figures for the policy portfolios
    forecasts/<origin>/expected_returns.csv       expected 12-month log excess return (the expanding
                                                  historical mean) with its standard error, 90% interval
                                                  and the arithmetic counterpart mean_12m_arith
    forecasts/<origin>/distribution_assets.csv    predictive quantiles of each asset's 12-month log
                                                  excess return (q01..q99), mean, scale, sd
    forecasts/<origin>/distribution_portfolios.csv  the same for the policy portfolios: rows with
                                                  shape="portfolios" are the published ones, rows with
                                                  shape="assets" a sensitivity check
    forecasts/<origin>/shape.csv                  the pooled standardized shapes behind the quantiles
    forecasts/<origin>/frontier.csv               the efficient frontier these numbers imply, long-only with
                                                  cash and no borrowing (``assetforecast.frontier``)
    forecasts/<origin>/frontier_points.csv        the assets, policy portfolios and named portfolios on
                                                  the same axes
    forecasts/latest/                             a copy of the newest origin

All returns are log returns in excess of T-bills. The method of every block
is stated in ``forecast.json``; the definitions are in
``assetforecast.returns.expected`` and ``assetforecast.returns.distribution``.

``--max-date D`` (or ``ASSETFORECAST_MAX_DATE``) makes a development forecast:
the latest stored origin on or before D, the panel cut at D, the covariance
history under ``results/dev/covariance``, and output under
``results/dev/forecasts`` (never ``forecasts/``).
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from . import paths
from .bridge import get_bridge
from .data.panel import load_panel
from .eval import harness
from .eval.portfolio import POLICIES
from .eval.targets import realized_kday_cov
from .universe import BY_ID

PRIMARY, BRIDGE = "combo_cov_main", "vr_market"
_MAX_ENV = "ASSETFORECAST_MAX_DATE"
# The bridged matrix computed here and the covariance history's matrix for the
# same origin come from the same code on the same data and must agree to
# floating-point noise (relative to the largest variance).
COV_TOL = 1e-12

EXPECTED_METHOD = (
    "mean_12m is the expected 12-month log excess return: 12 x the mean monthly log excess return "
    "over the asset's whole history through the origin on the return panel (from 1926-07, "
    "month-average tier-D eras excluded), identical to the hist_mean benchmark. It is not a model "
    "forecast: no model beat it out of sample in the assetpred study or its replication here. "
    "se_12m is its standard error from an equal-weighted cosine long-run variance (robust to serial "
    "correlation); ci90_lo/ci90_hi is mean_12m +/- t_B(0.95) se_12m with B the number of cosine terms. "
    "The interval covers sampling error of the average only, not a mean that has shifted over time. "
    "mean_12m_arith = exp(mean_12m + 0.5 cov_annual_ii) - 1 is the arithmetic (simple) 12-month "
    "excess return implied by a lognormal with that mean and the published 12-month variance.")
DISTRIBUTION_METHOD = (
    "Quantile q_k = mean + pscale x Z_k. mean is mean_12m (above); pscale = sqrt(cov_annual_ii + "
    "se_12m^2), the published 12-month variance plus the mean's own uncertainty (portfolios: "
    "sqrt(w'cov_annual w + (sum|w| se)^2)); Z_k is the shape: quantiles of past standardized "
    "outcomes z = (y - mean)/pscale at origins whose 12-month window had closed by the origin, "
    "pooled across assets (for portfolios: across the policy portfolios), recentred to mean zero "
    "with the spread kept, and with the 1/5/95/99% quantiles floored at a normal of the same "
    "standard deviation. 'scale' is sqrt(cov_annual_ii) alone, 'sd' = pscale x sd(Z). Portfolio "
    "log returns are approximated by w'r (as in the covariance side's portfolio tests). The "
    "distribution model was designed after both holdouts had been seen: its past scores describe "
    "it and are not out-of-sample evidence.")


def family_range(origin, assets, results_dir) -> pd.DataFrame:
    """10th/50th/90th percentile, across model families, of each family's mean vol forecast."""
    rows = {}
    for m in harness.stored_models("vol", results_dir):
        fam = m.split("_", 1)[0]
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


# ------------------------------------------------------------------ consistency with the history
class ForecastConsistencyError(RuntimeError):
    """The published covariance and the covariance history disagree."""


def check_against_history(origin, cov_21d: pd.DataFrame, cov_annual: pd.DataFrame,
                          path: Path | None = None, tol: float = COV_TOL) -> dict:
    """Assert that the covariance history holds exactly these matrices at ``origin``.

    The return side (distributions, strategy test) reads the history; the
    published files come from here. Both must be the same numbers. Raises
    ``ForecastConsistencyError`` when the history is missing, lacks the origin,
    has other assets or order, or differs by more than ``tol`` x the largest
    variance. Returns the differences found.
    """
    from . import covhistory
    path = Path(path) if path is not None else paths.COV_HISTORY
    if not path.exists():
        raise ForecastConsistencyError(
            f"no covariance history at {path}: run `python -m assetforecast.covhistory` first")
    out = {"history": str(path.relative_to(paths.ROOT)) if path.is_relative_to(paths.ROOT) else str(path)}
    for kind, mine in (("21d", cov_21d), ("annual", cov_annual)):
        hist = covhistory.load(kind, path)
        o = pd.Timestamp(origin)
        if o not in hist:
            raise ForecastConsistencyError(
                f"the covariance history {path} has no origin {o.date()} (last {max(hist).date()}): "
                "rerun `python -m assetforecast.covhistory` after the covariance run")
        H = hist[o]
        if list(H.index) != list(mine.index) or list(H.columns) != list(mine.columns):
            raise ForecastConsistencyError(
                f"cov_{kind} at {o.date()}: assets differ from the history "
                f"({list(mine.index)} vs {list(H.index)})")
        a, b = mine.to_numpy(float), H.to_numpy(float)
        diff = float(np.abs(a - b).max())
        rel = diff / float(np.abs(np.diag(a)).max())
        if not rel <= tol:
            raise ForecastConsistencyError(
                f"cov_{kind} at {o.date()} differs from the covariance history by {diff:.3g} "
                f"({rel:.3g} of the largest variance, tolerance {tol:g}); the history is stale or was "
                "built from other inputs: rerun `python -m assetforecast.covhistory`")
        out[f"cov_{kind}_max_abs_diff"] = diff
    return out


# ------------------------------------------------------------------ return side
def return_side(origin, cov_annual: pd.DataFrame, cov_history_path: Path | None = None,
                max_date: str | None = None) -> dict:
    """Expected returns and predictive distributions at ``origin``.

    The location is ``returns.expected.expected_returns`` on the return panel;
    the scale is ``cov_annual`` (the matrix published here); the shapes are
    ``returns.distribution.shape_info`` on the standardized residuals of every
    earlier origin of the covariance history whose window has closed.
    """
    from . import covhistory
    from .returns import config as rconfig
    from .returns import distribution as D
    from .returns import expected as E
    from .returns.inputs import excess_returns, load_return_panel, observed_through

    o = pd.Timestamp(origin)
    rpanel = load_return_panel(max_date=max_date)
    if observed_through(rpanel) < o:
        raise RuntimeError(f"the return panel ends {observed_through(rpanel).date()}, "
                           f"before the forecast origin {o.date()}")
    xr = excess_returns(rpanel).loc[:o]

    exp = E.expected_returns(rpanel, o)
    var = pd.Series(np.diag(cov_annual.to_numpy(float)), index=cov_annual.index)
    exp["mean_12m_arith"] = np.exp(exp["mean_12m"] + 0.5 * var.reindex(exp.index)) - 1.0
    exp.insert(0, "group", exp.index.map({a: BY_ID[a].group for a in exp.index}))
    exp.insert(0, "name", exp.index.map({a: BY_ID[a].name for a in exp.index}))

    hist = covhistory.load("annual", cov_history_path)
    hist = {t: m for t, m in hist.items() if t <= o}
    hist[o] = cov_annual
    exp_hist = E.history(xr)
    za = D.standardized_residuals(xr, hist, expected=exp_hist)
    zp = D.portfolio_residuals(xr, hist, expected=exp_hist)
    sa, sp = D.shape_info(za, o), D.shape_info(zp, o)
    if not (np.isfinite(sa["q"]).all() and np.isfinite(sp["q"]).all()):
        raise RuntimeError(f"no distribution shape at {o.date()} (too few closed windows)")
    assets_df, own = D.distribution_at(o, xr, cov_annual, {"assets": sa, "portfolios": sp}, expected=exp_hist)
    _, alt = D.distribution_at(o, xr, cov_annual, {"assets": sa}, expected=exp_hist)
    own.insert(1, "shape", "portfolios")
    own.insert(1, "model", D.MODEL)
    alt.insert(1, "shape", "assets")
    alt.insert(1, "model", D.MODEL_ASSETSHAPE)
    ports_df = pd.concat([own, alt], ignore_index=True)

    shape = pd.DataFrame({
        "tau": D.TAUS, "z_assets": sa["q"], "z_portfolios": sp["q"],
        "n_resid_assets": sa["n_rows"], "n_resid_portfolios": sp["n_rows"],
        "first_origin_assets": sa["first_origin"].strftime("%Y-%m-%d"),
        "last_origin_assets": sa["last_origin"].strftime("%Y-%m-%d"),
        "first_origin_portfolios": sp["first_origin"].strftime("%Y-%m-%d"),
        "last_origin_portfolios": sp["last_origin"].strftime("%Y-%m-%d"),
    })

    def facts(s):
        return {"sd_z": round(float(s["sd"]), 4), "mean_removed": round(float(s["mean"]), 4),
                "n_residuals": int(s["n_rows"]), "n_origins": int(s["n_dates"]),
                "n_series": int(s["n_series"]), "n_eff": round(float(s["n_eff"]), 1),
                "n_eff_q01": round(float(s["n_eff_q01"]), 1), "n_eff_q05": round(float(s["n_eff_q05"]), 1),
                "episodes_q01": int(s["episodes_q01"]), "episodes_q05": int(s["episodes_q05"]),
                "first_origin": s["first_origin"].strftime("%Y-%m-%d"),
                "last_origin": s["last_origin"].strftime("%Y-%m-%d")}

    meta = {
        "expected_returns": {
            "method": EXPECTED_METHOD, "level": 0.90,
            "panel": f"return panel from {rconfig.RETURN_START}, tiers {rconfig.RETURN_DROP_TIERS} dropped",
            "min_history_months": rconfig.MIN_ASSET_HISTORY,
        },
        "distribution": {
            "method": DISTRIBUTION_METHOD, "model": D.MODEL, "sensitivity_model": D.MODEL_ASSETSHAPE,
            "tail_floor": list(D.TAIL_FLOOR), "min_residual_origins": D.MIN_RESID_DATES,
            "shape_assets": facts(sa), "shape_portfolios": facts(sp),
            "label": rconfig.REPLICATION_LABEL,
            "out_of_sample_from": (f"Only forecasts whose whole 12-month window lies after FROZEN_AT "
                                   f"({rconfig.FROZEN_AT}), i.e. origins from {rconfig.OOS_ORIGIN_START} "
                                   "on, count as out-of-sample evidence for the return side."),
        },
    }
    return {"expected": exp, "dist_assets": assets_df, "dist_portfolios": ports_df,
            "shape": shape, "meta": meta}


# ------------------------------------------------------------------ build / write
def build(model: str = PRIMARY, bridge: str = BRIDGE, results_dir: Path | None = None,
          cov_history: Path | None = None, max_date: str | None = None) -> dict:
    results_dir = Path(results_dir) if results_dir else paths.RESULTS
    max_date = max_date or os.environ.get(_MAX_ENV) or None
    panel = load_panel(max_date=max_date)
    stored = harness.load_forecasts("cov", model, results_dir)
    if max_date:
        stored = {o: c for o, c in stored.items() if o <= pd.Timestamp(max_date)}
    if not stored:
        raise RuntimeError(f"no stored {model!r} forecasts in {results_dir}"
                           + (f" on or before {max_date}" if max_date else ""))
    origin = max(stored)
    cov_rv = stored[origin]
    assets = list(cov_rv.index)
    view = panel.asof(origin)
    if view.daily.index.max() < origin - pd.Timedelta(days=5):
        raise RuntimeError(f"the panel ends {view.daily.index.max().date()}, before the forecast origin {origin.date()}")

    cov_a = get_bridge(bridge).cov_annual(cov_rv, view)
    checks = {"covariance_history": check_against_history(origin, cov_rv, cov_a, cov_history)}
    v21, v12 = np.sqrt(np.diag(cov_rv)), np.sqrt(np.diag(cov_a))
    corr = cov_rv / np.outer(v21, v21)

    vols = pd.DataFrame({
        "name": [BY_ID[a].name for a in assets],
        "group": [BY_ID[a].group for a in assets],
        "vol_21d_horizon": v21,
        "vol_12m_return": v12,
        "variance_ratio": (v12 / v21) ** 2,
    }, index=assets)
    vols = vols.join(family_range(origin, assets, results_dir)).join(realized_context(panel, origin, assets))
    vols.index.name = "asset"

    W = pd.DataFrame({k: pd.Series(p["weights"]) for k, p in POLICIES.items()}).T.reindex(columns=assets).fillna(0.0)
    W = W.div(W.sum(axis=1), axis=0)
    Wa = W.to_numpy()
    ports = pd.DataFrame({
        "vol_21d_horizon": np.sqrt(np.einsum("pi,ij,pj->p", Wa, cov_rv.to_numpy(), Wa)),
        "vol_12m_return": np.sqrt(np.einsum("pi,ij,pj->p", Wa, cov_a.to_numpy(), Wa)),
    }, index=W.index)
    ports.index.name = "portfolio"

    rs = return_side(origin, cov_a, cov_history, max_date)
    # the distribution's portfolio scale is w' cov_annual w with the same policy weights
    own = rs["dist_portfolios"][rs["dist_portfolios"]["shape"] == "portfolios"].set_index("portfolio")
    gap = float((own["scale"] - ports["vol_12m_return"].reindex(own.index)).abs().max())
    if not gap < 1e-12:
        raise ForecastConsistencyError(f"distribution portfolio scale differs from vol_12m_return by {gap:.3g}")
    checks["portfolio_scale_vs_vol_12m_return_max_abs_diff"] = gap

    meta = {
        "origin": origin.strftime("%Y-%m-%d"),
        "window": f"{(origin + pd.Timedelta(days=1)).strftime('%Y-%m-%d')} to {(origin + pd.offsets.MonthEnd(12)).strftime('%Y-%m-%d')}",
        "data_through": view.daily.index.max().strftime("%Y-%m-%d"),
        "model": model, "bridge": bridge,
        "returns": "log returns in excess of T-bills",
        "note": ("Covariance: the selected model's forecast, bridged to 12-month returns. Expected return: "
                 "the expanding historical mean, not a model forecast. Quantiles: "
                 "distribution_assets.csv and distribution_portfolios.csv (they include the mean)."),
        **({"max_date": max_date, "development": True} if max_date else {}),
        **rs["meta"],
        "checks": checks,
    }
    return {"meta": meta, "vols": vols, "corr": corr, "cov_annual": cov_a, "cov_21d": cov_rv,
            "portfolios": ports, "weights": W, "expected": rs["expected"],
            "dist_assets": rs["dist_assets"], "dist_portfolios": rs["dist_portfolios"], "shape": rs["shape"]}


def _records(df: pd.DataFrame, digits: int) -> list:
    return json.loads(df.round(digits).to_json(orient="records"))


def write(result: dict, out: Path, latest: bool = True) -> Path:
    d = Path(out) / result["meta"]["origin"]
    d.mkdir(parents=True, exist_ok=True)
    result["vols"].round(5).to_csv(d / "vols.csv")
    result["corr"].round(4).to_csv(d / "corr.csv")
    result["cov_annual"].round(8).to_csv(d / "cov_annual.csv")
    result["cov_21d"].round(8).to_csv(d / "cov_21d.csv")
    result["portfolios"].round(5).to_csv(d / "portfolios.csv")
    result["expected"].round(5).to_csv(d / "expected_returns.csv")
    result["dist_assets"].round(5).to_csv(d / "distribution_assets.csv", index=False)
    result["dist_portfolios"].round(5).to_csv(d / "distribution_portfolios.csv", index=False)
    result["shape"].round(5).to_csv(d / "shape.csv", index=False)
    meta = dict(result["meta"])
    blocks = {k: meta.pop(k) for k in ("expected_returns", "distribution", "checks")}
    payload = {
        **meta,
        "assets": _records(result["vols"].reset_index(), 5),
        "portfolios": _records(result["portfolios"].reset_index(), 5),
        "portfolio_weights": json.loads(result["weights"].round(4).to_json(orient="index")),
        "corr": {"assets": list(result["corr"].index), "matrix": result["corr"].round(4).to_numpy().tolist()},
        "cov_annual": {"assets": list(result["cov_annual"].index),
                       "matrix": result["cov_annual"].round(8).to_numpy().tolist()},
        "expected_returns": {**blocks["expected_returns"],
                             "assets": _records(result["expected"].reset_index(), 5)},
        "distribution": {**blocks["distribution"],
                         "assets": _records(result["dist_assets"], 5),
                         "portfolios": _records(result["dist_portfolios"], 5),
                         "shape": _records(result["shape"], 5)},
        "checks": blocks["checks"],
    }
    (d / "forecast.json").write_text(json.dumps(payload, indent=1))
    from . import frontier
    frontier.publish(d)      # frontier.csv, frontier_points.csv and the "frontier" block, from the files above
    if latest:
        lt = Path(out) / "latest"
        if lt.exists():
            shutil.rmtree(lt)
        shutil.copytree(d, lt)
    return d


def main(argv=None):
    global paths
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default=PRIMARY)
    ap.add_argument("--bridge", default=BRIDGE)
    ap.add_argument("--results", default=None,
                    help="covariance harness store holding the stored forecasts (default paths.RESULTS)")
    ap.add_argument("--out", default=None,
                    help="output root (default forecasts/, or results/dev/forecasts under --max-date)")
    ap.add_argument("--max-date", default=None, help="development forecast: latest origin on or before D")
    a = ap.parse_args(argv)
    if a.max_date:
        os.environ[_MAX_ENV] = str(pd.Timestamp(a.max_date).date())
    if bool(os.environ.get(_MAX_ENV)) != paths._DEV:
        paths = importlib.reload(paths)   # COV_HISTORY and RESULTS move under results/dev
    max_date = os.environ.get(_MAX_ENV) or None
    out = Path(a.out) if a.out else (paths.ROOT / "results" / "dev" / "forecasts" if max_date
                                     else paths.FORECASTS_OUT)
    res = build(a.model, a.bridge, a.results, max_date=max_date)
    d = write(res, out)
    pd.set_option("display.width", 200)
    print(f"forecast for {res['meta']['window']} (model {a.model}, bridge {a.bridge}) -> {d}")
    print(res["vols"][["vol_21d_horizon", "vol_12m_return", "family_p10", "family_p90",
                       "realized_last_12m", "realized_long_run"]].round(3).to_string())
    print(res["expected"][["mean_12m", "ci90_lo", "ci90_hi", "mean_12m_arith"]].round(3).to_string())


if __name__ == "__main__":
    main()
