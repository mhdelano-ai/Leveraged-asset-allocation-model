"""The published forecast restated for three holdings: world stocks, world bonds, cash.

    python -m assetforecast.threefund [--dir DIR]

Reads a published forecast folder (default ``forecasts/latest``) and writes
``forecasts/three_fund/<origin>/`` and ``forecasts/three_fund/latest/``:
``weights.csv``, ``cov_annual.csv``, ``cov_21d.csv``, ``vols.csv``,
``expected_returns.csv`` (with the sensitivity column described below),
``mixes.csv``, ``forecast.json`` and the frontier files of
``assetforecast.frontier`` (cash below full investment, borrowing above it).

Nothing is forecast here. The two composites are fixed-weight mixes of the 12
assets, so their covariance is ``W cov W'`` of the published matrix, exactly.
Like the frontier, this is arithmetic on the published forecast: it selects
nothing and has no out-of-sample evidence of its own.

The composites (``COMPOSITES``) are declared judgements, not estimates
-----------------------------------------------------------------------
The panel has no world-stock or world-bond series, so each is a constant mix
of asset classes it does have:

``world_stocks``  US large 53%, US small 9%, developed ex-US 28%, emerging 10%:
                  roughly the regional split of a float-weighted all-world
                  index in the mid-2020s (US about 62%).
``world_bonds``   intermediate Treasuries 45%, investment-grade corporates
                  25%, T-bills 30%. The panel's bonds are all long (about 7
                  years and more) and it has no short, mortgage-backed or
                  non-US bonds, so a mix of them alone swings about 40% more
                  than a world bond fund hedged to dollars; the bill share
                  stands in for the short maturities. An unhedged world bond
                  fund carries currency risk that is absent here.

How close they are (``CHECK``, a one-off outside comparison, not reproducible
from this repo's loaders): calendar-year total returns of the two Vanguard
funds, from month-end adjusted closes retrieved from Alpha Vantage on
2026-11-15, against the composites'. World stocks vs VT, 2009-2025:
correlation 0.997, composite ahead by 0.45 points a year on average, standard
deviation of the difference 1.3 points. World bonds vs BNDW, 2019-2025:
correlation 0.985, composite ahead by 0.4 points, standard deviation of the
difference 1.3 points (3.7 points for a mix without the bill share). The bill
share was chosen after seeing that comparison.

The weights are today's, applied to every year. A real world index held less
in the US for most of its history, so the composite is "today's world fund,
had it existed", not the history of the index.

Expected return
---------------
``mean_12m`` is the package's expected return (``returns.expected``: the
expanding mean of monthly log excess returns, with its 90% interval) of the
composite's own return series on the return panel: each month the simple
returns of the components that exist are mixed with the weights renormalized
over them (the convention of the package's policy portfolios). Before 1975
"world stocks" is therefore US stocks only, and before 1962 "world bonds" is
corporates and bills only. ``mean_12m_full_mix`` is the same mean over the months in
which every component exists: a sensitivity, shown beside it.

``mixes.csv``: the mix with the most expected growth at each volatility of
``MIX_VOLS``, as shares of capital in stocks, bonds, cash and borrowing, into
the borrowed range, and the growth-optimal mix (see :func:`mixes`).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import frontier, paths

CASH = "cash"     # T-bills inside a composite: no excess return, no risk
COMPOSITES = {
    "world_stocks": {"us_large": 0.53, "us_small": 0.09, "intl_dev": 0.28, "em": 0.10},
    "world_bonds": {"ust_int": 0.45, "ig_credit": 0.25, CASH: 0.30},
}
NAMES = {"world_stocks": "World stocks (composite)", "world_bonds": "World bonds (composite)"}
POLICIES = {"100/0": {"world_stocks": 1.0}, "80/20": {"world_stocks": 0.8, "world_bonds": 0.2},
            "60/40": {"world_stocks": 0.6, "world_bonds": 0.4},
            "40/60": {"world_stocks": 0.4, "world_bonds": 0.6}, "0/100": {"world_bonds": 1.0}}
MIX_VOLS = (0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.25, 0.30)
OUT = paths.FORECASTS_OUT / "three_fund"
LABEL = ("Arithmetic on the published forecast for two declared composites and cash; not a model, "
         "selects nothing, no out-of-sample evidence of its own")


def weight_matrix(assets: list[str]) -> pd.DataFrame:
    W = pd.DataFrame(0.0, index=list(COMPOSITES), columns=assets)
    for c, w in COMPOSITES.items():
        risky = {a: x for a, x in w.items() if a != CASH}
        if abs(sum(w.values()) - 1.0) > 1e-12 or set(risky) - set(assets):
            raise ValueError(f"{c}: weights must sum to 1 over known assets and cash")
        W.loc[c, list(risky)] = list(risky.values())
    return W


def aggregate(cov: pd.DataFrame) -> pd.DataFrame:
    W = weight_matrix(list(cov.index))
    return pd.DataFrame(W.to_numpy() @ cov.to_numpy() @ W.to_numpy().T, index=W.index, columns=W.index)


def composite_simple(simple: pd.DataFrame, cash: pd.Series, full_mix: bool = False) -> pd.DataFrame:
    """Monthly simple returns of the composites (weights renormalized over the components present)."""
    out = {}
    simple = simple.assign(**{CASH: cash.reindex(simple.index)})
    for c, w in COMPOSITES.items():
        r, ws = simple[list(w)], pd.Series(w)
        have = r.notna()
        tot = have.mul(ws, axis=1).sum(axis=1)
        mix = r.fillna(0.0).mul(ws, axis=1).sum(axis=1) / tot.where(tot > 0)
        out[c] = mix.where(have.all(axis=1)) if full_mix else mix
    return pd.DataFrame(out)


def composite_xr(simple: pd.DataFrame, cash: pd.Series, full_mix: bool = False) -> pd.DataFrame:
    """Monthly log excess returns of the composites from monthly simple returns."""
    c = composite_simple(simple, cash, full_mix)
    return np.log1p(c).sub(np.log1p(cash.reindex(simple.index)), axis=0)


def past_losses(x, comp: pd.DataFrame, cash: pd.Series, spread: float, start=None) -> dict:
    """What holding ``x`` (rebalanced monthly, borrowing at T-bills plus ``spread``) did at its worst.

    ``worst_12m``: the lowest 12-month total return; ``max_drawdown``: the
    deepest fall from a previous high, both from month-end values of
    ``comp`` (composite simple returns) since ``start``. Month-end values
    miss falls inside a month, and a margin loan can be called before a
    month ends, so a levered mix can do worse than this.
    """
    c = comp.dropna()
    if start is not None:
        c = c.loc[pd.Timestamp(start):]
    rf = cash.reindex(c.index)
    x = np.asarray(x, float)
    r = rf + (c.sub(rf, axis=0) * x).sum(axis=1) - spread / 12.0 * max(float(x.sum()) - 1.0, 0.0)
    v = (1.0 + r.clip(lower=-1.0)).cumprod()
    return {"worst_12m": float((v / v.shift(12) - 1.0).min()),
            "max_drawdown": float((v / v.cummax() - 1.0).min()), "from": c.index[0].strftime("%Y-%m")}


def expected(origin) -> pd.DataFrame:
    from .returns import expected as ex, inputs
    panel = inputs.load_return_panel()
    simple = panel.monthly.loc[:pd.Timestamp(origin)]
    e = ex.expected_from_xr(composite_xr(simple, panel.cash_monthly), origin)
    f = ex.expected_from_xr(composite_xr(simple, panel.cash_monthly, full_mix=True), origin)
    e["mean_12m_full_mix"] = f["mean_12m"]
    e["first_month_full_mix"] = f["first_month"]
    e.insert(0, "name", [NAMES[a] for a in e.index])
    return e


def _grid(inp, step: float = 0.0025) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Every long-only mix of the two composites up to ``frontier.MAX_LEVERAGE``: weights, volatility, growth."""
    g = np.arange(0.0, frontier.MAX_LEVERAGE + step / 2, step)
    X = np.array([(s, b) for s in g for b in g if s + b <= frontier.MAX_LEVERAGE + 1e-9])
    var = np.einsum("ni,ij,nj->n", X, inp.cov, X)
    net = X @ frontier.drift(inp) - inp.spread * np.maximum(X.sum(axis=1) - 1.0, 0.0)
    return X, np.sqrt(var), net - 0.5 * var


def mixes(d: Path) -> pd.DataFrame:
    """The mix with the most growth at each volatility, and the mix with the most growth of all.

    Up to the volatility of world stocks these are rows of the published
    levered frontier (``frontier.at_volatility``). Above it, where the package's
    frontier stops, they come from a grid over every long-only mix up to
    ``frontier.MAX_LEVERAGE`` (weights in steps of 0.25%, volatility within
    0.1 point of the target). ``most_growth`` is the grid's growth-optimal mix:
    past it more borrowing lowers expected compound growth. Each row also
    carries what that mix did at its worst (:func:`past_losses`) over the whole
    return panel (``_all``) and over the months in which every component of both
    composites exists (``_full_mix``).
    """
    from .returns import inputs
    inp = frontier.load_inputs(d)
    top = float(np.sqrt(np.diag(inp.cov)).max())
    X, vol, gr = _grid(inp)
    origin = json.loads((d / "forecast.json").read_text())["origin"]
    panel = inputs.load_return_panel()
    simple = panel.monthly.loc[:pd.Timestamp(origin)]
    eras = {"all": composite_simple(simple, panel.cash_monthly),
            "full_mix": composite_simple(simple, panel.cash_monthly, full_mix=True)}

    def row(label, x):
        s = frontier.stats(inp, x)
        inv = float(np.sum(x))
        out = {"mix": label, "vol": s["vol"], **{a: float(w) for a, w in zip(inp.assets, x)},
               "cash": max(1 - inv, 0.0), "borrowed": max(inv - 1, 0.0),
               "growth": s["growth"], "growth_lo": s["growth_lo"], "growth_hi": s["growth_hi"]}
        for k, comp in eras.items():
            h = past_losses(x, comp, panel.cash_monthly, inp.spread)
            out.update({f"worst_12m_{k}": h["worst_12m"], f"max_drawdown_{k}": h["max_drawdown"],
                        f"from_{k}": h["from"]})
        return out

    rows = []
    for k, w in POLICIES.items():
        rows.append(row(k, [w.get(a, 0.0) for a in inp.assets]))
    for v in MIX_VOLS:
        if v <= top + 1e-9:
            rows.append(row(f"vol {v:.0%}", frontier.at_volatility(inp, v, True)))
            continue
        near = np.abs(vol - v) <= 0.001
        if near.any():
            rows.append(row(f"vol {v:.0%}", X[np.flatnonzero(near)[gr[near].argmax()]]))
    rows.append(row("most_growth", X[gr.argmax()]))
    return pd.DataFrame(rows)


def publish(src: Path | None = None) -> Path:
    src = Path(src) if src else paths.FORECASTS_OUT / "latest"
    meta = json.loads((src / "forecast.json").read_text())
    origin = meta["origin"]
    d = OUT / origin
    d.mkdir(parents=True, exist_ok=True)
    cov = pd.read_csv(src / "cov_annual.csv", index_col=0)
    c21 = pd.read_csv(src / "cov_21d.csv", index_col=0)
    A, A21 = aggregate(cov), aggregate(c21)
    weight_matrix(list(cov.index)).round(4).to_csv(d / "weights.csv")
    A.round(8).to_csv(d / "cov_annual.csv")
    A21.round(8).to_csv(d / "cov_21d.csv")
    sd, sd21 = np.sqrt(np.diag(A)), np.sqrt(np.diag(A21))
    pd.DataFrame({"name": [NAMES[a] for a in A.index], "vol_21d_horizon": sd21, "vol_12m_return": sd,
                  "corr_with_other": [A.iloc[0, 1] / (sd[0] * sd[1])] * 2}, index=A.index
                 ).rename_axis("asset").round(5).to_csv(d / "vols.csv")
    e = expected(origin)
    e["mean_12m_arith"] = np.expm1(e["mean_12m"] + 0.5 * np.diag(A))
    num = e.select_dtypes("float").columns
    e[num] = e[num].round(5)
    e.to_csv(d / "expected_returns.csv")
    (d / "forecast.json").write_text(json.dumps({
        "origin": origin, "window": meta.get("window"), "source_forecast": str(src.name),
        "model": meta.get("model"), "bridge": meta.get("bridge"), "returns": meta.get("returns"),
        "note": LABEL, "composites": COMPOSITES, "portfolio_weights": POLICIES}, indent=1))
    frontier.publish(d)
    mixes(d).round(5).to_csv(d / "mixes.csv", index=False)
    latest = OUT / "latest"
    if latest.exists():
        shutil.rmtree(latest)
    shutil.copytree(d, latest)
    return d


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m assetforecast.threefund", description=__doc__.split("\n")[0])
    ap.add_argument("--dir", type=Path, default=None)
    d = publish(ap.parse_args(argv).dir)
    pd.set_option("display.width", 200)
    for f in ("vols.csv", "expected_returns.csv", "mixes.csv"):
        print(f"\n{f}\n" + pd.read_csv(d / f).to_string(index=False))
    pts = pd.read_csv(d / "frontier_points.csv")
    print("\nfrontier_points.csv\n" + pts[["point", "kind", "vol", "growth", "growth_lo", "growth_hi", "invested",
                                          "w_world_stocks", "w_world_bonds", "gap"]].to_string(index=False))
    b = json.loads((d / "forecast.json").read_text())["frontier"]["borrow"]
    print(f"\nborrowing: {b['margin_rate']:.2%} = T-bills {b['tbill_rate']:.2%} + {b['spread']:.2%}")
    print(LABEL)
    return 0


if __name__ == "__main__":
    sys.exit(main())
