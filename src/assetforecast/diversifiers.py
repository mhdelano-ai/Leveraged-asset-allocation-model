"""The diversifier study: which assets have really diversified stocks?

    python -m assetforecast.diversifiers

Writes ``results/diversifiers/`` (``summary.csv``, ``episodes.csv``,
``rolling_corr.csv``, ``mixes.csv``, ``sleeve.csv``, ``sleeve_corr.csv``, ``meta.json``), ``reports/diversifiers.md``
and ``reports/figures/diversifiers/``. A DESCRIPTIVE study (``LABEL``): it
measures the past, forecasts nothing, selects nothing and changes neither the
frozen models nor the published forecast.

Stocks are ``threefund``'s world-stock composite (US only before 1975). The
candidates are the panel's eight assets that are not stocks, and managed
futures, which the 12-asset panel does not hold and which enter from
``data.sources.managed_futures`` as two series:

``mf_tsmom``  AQR's time-series-momentum factor, 1985 on: hypothetical, before
              fees and trading costs.
``mf_funds``  equal-weight composite of live managed-futures funds, 2007-03 on:
              after fees; a survivor list.

All returns are monthly log excess returns over T-bills on the return panel
(no tier D), as everywhere on the return side.

What is measured, for each candidate against stocks, on each window
-------------------------------------------------------------------
``corr_1m, corr_12m``   correlation of monthly and of overlapping 12-month returns
``beta``                slope of the monthly regression on stocks
``roll_min, roll_max``  lowest and highest correlation over any ``ROLL`` months
``down_corr``           correlation over the months in which stocks fell
``worst_mean``          mean monthly return in stocks' worst ``WORST`` of months
``capture``             the candidate's summed return over the months of stocks'
                        bear markets (``bear_markets``), as a share of stocks' own
``bears, bears_down, worst_bear``  the bear markets the candidate's history covers
                        in full, in how many it lost money, and its worst return in one
``mean, se, vol``       12 x mean, its standard error (``returns.expected.mean_se``),
                        annualized volatility
``alpha, alpha_t``      12 x mean of ``y - beta x`` on SIMPLE excess returns and its
                        t-ratio (the same standard error; beta's own error is
                        ignored). A small holding added to stocks raises the
                        Sharpe ratio exactly when alpha is positive.

The verdict (``verdict``) applies thresholds that were fixed before any table
was computed. They are judgements, not estimates:

``stock-like``   correlation above ``LOW_CORR`` at 1 or 12 months
``unreliable``   low on average, but above ``MAX_ROLL`` in some ``ROLL``-month
                 stretch, or losing more than ``MAX_CAPTURE`` of stocks' fall in
                 bear markets
``diluter``      low, stable and held up in bear markets, but no return of its
                 own beyond its stock exposure (alpha <= 0): it lowers risk the
                 way cash does
``diversifier``  low, stable, held up in bear markets, alpha > 0; "(unproven)"
                 when alpha is within ``T_SIG`` standard errors of zero
``short``        under ``ROLL`` + 60 months of history: stability cannot be judged

``mixes.csv``: what replacing 20% of an all-stock holding with the candidate
(``swap``) did to growth, volatility and the worst fall, on common windows.
Nothing borrows.

The sleeve (``sleeve.csv``, ``sleeve_corr.csv``)
------------------------------------------------
``SLEEVE`` holds the three assets the verdicts leave standing (intermediate
Treasuries, gold, managed futures), rebalanced monthly to equal weights, or to
weights proportional to 1/volatility measured over the same window (so those
are in-sample). The members were picked on the history the sleeve is then
measured on, which flatters every number. ``sleeve_table`` gives the sleeve
alone and mixes of stocks and sleeve at ``SLEEVE_STOCKS``, the sleeve taking
what stocks leave of the capital.

``vehicles()`` reads ``data/processed/diversifier_vehicles.json``: plain funds
that give these exposures and their fees, a hand-recorded outside lookup with its
date and sources (not reproducible from a loader, like ``threefund.CHECK``).

``mf_tsmom_net`` in the tables is ``mf_tsmom`` less ``drag``: the gap, measured
on the months both exist, between the paper factor scaled to the funds'
volatility and the funds themselves, scaled back to the factor's volatility.
Subtracting a constant changes the mean and alpha only.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

from . import paths, threefund
from .returns import inputs
from .returns.expected import mean_se

LABEL = ("Descriptive diversifier study; it measures the past on data that were all seen "
         "before it was designed, selects nothing and is not out-of-sample evidence")
STOCKS = "world_stocks"
CANDIDATES = ["ust_int", "ust_long", "tips", "ig_credit", "hy_credit", "reits", "gold", "commodities",
              "mf_tsmom", "mf_tsmom_net", "mf_funds"]
NAMES = {"ust_int": "Intermediate Treasuries", "ust_long": "Long Treasuries", "tips": "TIPS",
         "ig_credit": "Investment-grade corporates", "hy_credit": "High-yield corporates",
         "reits": "US REITs", "gold": "Gold", "commodities": "Broad commodities",
         "mf_tsmom": "Managed futures: paper trend factor, before fees",
         "mf_tsmom_net": "Managed futures: paper trend factor, less measured drag",
         "mf_funds": "Managed futures: live funds, after fees"}
WINDOWS = {"own": None, "since_1985": "1985-01-31", "since_2007": "2007-03-31"}
ROLL = 120            # months in the rolling correlation
WORST = 0.10          # share of months counted as stocks' worst
BEAR = 0.20           # a bear market: stocks fall this much from a high (month-end, excess of cash)
LOW_CORR = 0.30
MAX_ROLL = 0.50
MAX_CAPTURE = 0.20
T_SIG = 1.645
MIX_SHARE = 0.20
SLEEVE = {"since_1985": ["ust_int", "gold", "mf_tsmom_net"], "since_2007": ["ust_int", "gold", "mf_funds"]}
SLEEVE_STOCKS = (1.0, 0.8, 0.6, 0.4)   # stock shares of the stocks-and-sleeve mixes
VEHICLES = paths.PROCESSED / "diversifier_vehicles.json"

_DEV = bool(os.environ.get("ASSETFORECAST_MAX_DATE"))
OUT = paths.ROOT / "results" / ("dev/diversifiers" if _DEV else "diversifiers")
REPORT = paths.REPORTS / ("diversifiers_dev.md" if _DEV else "diversifiers.md")
FIGURES = paths.REPORTS / "figures" / "diversifiers"


# ------------------------------------------------------------------ inputs
def measured_drag(tsmom: pd.Series, funds: pd.Series) -> dict:
    """Annual gap between the paper factor and the live funds at the factor's volatility."""
    j = pd.concat([tsmom, funds], axis=1, keys=["t", "f"], sort=True).dropna()
    k = j["f"].std() / j["t"].std()
    gap_at_fund_vol = 12 * (k * j["t"].mean() - j["f"].mean())
    return {"months": int(len(j)), "vol_ratio_funds_to_factor": float(k),
            "gap_at_fund_vol": float(gap_at_fund_vol), "drag": float(gap_at_fund_vol / k)}


def load_inputs(mf: pd.DataFrame | None = None) -> tuple[pd.Series, pd.DataFrame, pd.Series, dict]:
    """(stocks, candidates, cash, drag): monthly log excess returns and the simple T-bill return."""
    from .data.sources import managed_futures
    rp = inputs.load_return_panel()
    xr = inputs.excess_returns(rp)
    cash = rp.cash_monthly.loc[:xr.index.max()]
    stocks = threefund.composite_xr(rp.monthly.loc[:xr.index.max()], cash)[STOCKS]
    mf = managed_futures.load() if mf is None else mf
    mf = mf.loc[:xr.index.max()]
    c = xr.copy()
    c["mf_tsmom"] = np.log1p(mf["tsmom_xr"])
    c["mf_funds"] = np.log1p(mf["funds_ret"]) - np.log1p(cash.reindex(mf.index))
    drag = measured_drag(c["mf_tsmom"], c["mf_funds"])
    c["mf_tsmom_net"] = c["mf_tsmom"] - drag["drag"] / 12
    return stocks, c[CANDIDATES], cash, drag


# ------------------------------------------------------------------ measures
def bear_markets(x: pd.Series, depth: float = BEAR) -> pd.DataFrame:
    """Falls of at least ``depth`` in the index of log returns ``x``, peak to trough.

    A fall starts at the highest month-end since the last trough and ends at
    its lowest month-end before a rise of ``depth`` from that low. An unfinished
    fall at the end of the sample is closed at its low so far."""
    lvl = x.dropna().cumsum()
    lo, up = np.log1p(-depth), np.log1p(depth)
    rows, peak, peak_d, trough, trough_d = [], 0.0, lvl.index[0] - pd.offsets.MonthEnd(1), None, None
    for d, v in lvl.items():
        if trough is None:
            if v > peak:
                peak, peak_d = v, d
            elif v - peak <= lo:
                trough, trough_d = v, d
        elif v < trough:
            trough, trough_d = v, d
        elif v - trough >= up:
            rows.append((peak_d, trough_d, np.expm1(trough - peak)))
            peak, peak_d, trough, trough_d = v, d, None, None
    if trough is not None:
        rows.append((peak_d, trough_d, np.expm1(trough - peak)))
    return pd.DataFrame(rows, columns=["peak", "trough", "stocks"])


def episode_returns(stocks: pd.Series, cands: pd.DataFrame, depth: float = BEAR) -> pd.DataFrame:
    """Simple excess return of every candidate from each bear market's peak to its trough
    (NaN unless the candidate has every month of it)."""
    ep = bear_markets(stocks, depth)
    for c in cands:
        ep[c] = [np.expm1(cands[c].loc[p + pd.offsets.MonthEnd(1):t].sum(skipna=False))
                 for p, t in zip(ep["peak"], ep["trough"])]
    return ep


def bear_mask(index: pd.DatetimeIndex, ep: pd.DataFrame) -> pd.Series:
    m = pd.Series(False, index=index)
    for p, t in zip(ep["peak"], ep["trough"]):
        m.loc[p + pd.offsets.MonthEnd(1):t] = True
    return m


def stats(y: pd.Series, x: pd.Series, bear: pd.Series) -> dict:
    """Every measure of one candidate ``y`` against stocks ``x`` (log excess returns)."""
    j = pd.concat([y, x], axis=1, keys=["y", "x"], sort=True).dropna()
    n = len(j)
    if n < 36:
        return {"n": n}
    y, x = j["y"], j["x"]
    beta = float(np.cov(y, x)[0, 1] / x.var())
    y12, x12 = y.rolling(12).sum().dropna(), x.rolling(12).sum().dropna()
    roll = y.rolling(ROLL).corr(x).dropna()
    worst = x <= x.quantile(WORST)
    b = bear.reindex(j.index).fillna(False).astype(bool)
    ys, xs = np.expm1(y), np.expm1(x)
    bs = float(np.cov(ys, xs)[0, 1] / xs.var())
    a, a_se, _, _ = mean_se(ys - bs * xs)
    m, se, _, _ = mean_se(y)
    return {
        "n": n, "start": str(j.index[0].date()), "end": str(j.index[-1].date()),
        "corr_1m": float(y.corr(x)), "corr_12m": float(y12.corr(x12)), "beta": beta,
        "roll_min": float(roll.min()) if len(roll) else np.nan,
        "roll_max": float(roll.max()) if len(roll) else np.nan,
        "down_corr": float(y[x < 0].corr(x[x < 0])),
        "worst_mean": float(y[worst].mean()), "worst_stocks": float(x[worst].mean()),
        "bear_months": int(b.sum()),
        "capture": float(y[b].sum() / x[b].sum()) if b.any() else np.nan,
        "mean": 12 * m, "se": 12 * se, "vol": float(y.std() * np.sqrt(12)),
        "sharpe": float(12 * m / (y.std() * np.sqrt(12))),
        "alpha": 12 * a, "alpha_se": 12 * a_se, "alpha_t": float(a / a_se),
    }


def verdict(s: dict) -> str:
    if s.get("n", 0) < 36:
        return ""
    if max(s["corr_1m"], s["corr_12m"]) > LOW_CORR:
        return "stock-like"
    if s["n"] < ROLL + 60:
        return "short"
    if s["roll_max"] > MAX_ROLL or s["capture"] > MAX_CAPTURE:
        return "unreliable"
    if s["alpha"] <= 0:
        return "diluter"
    return "diversifier" if s["alpha_t"] >= T_SIG else "diversifier (unproven)"


def summary(stocks: pd.Series, cands: pd.DataFrame) -> pd.DataFrame:
    ep = episode_returns(stocks, cands)
    bear = bear_mask(stocks.dropna().index, ep)
    rows = []
    for w, start in WINDOWS.items():
        for c in cands:
            y = cands[c] if start is None else cands[c].loc[start:]
            if start is not None and (y.first_valid_index() is None or y.first_valid_index() > pd.Timestamp(start)):
                continue                      # a common window holds only assets that span it
            s = stats(y, stocks, bear)
            e = ep.loc[ep["peak"] >= (pd.Timestamp(start) if start else ep["peak"].min()), c].dropna()
            s.update(bears=int(len(e)), bears_down=int((e < 0).sum()),
                     worst_bear=float(e.min()) if len(e) else np.nan)
            rows.append({"window": w, "asset": c, **s, "verdict": verdict(s)})
    return pd.DataFrame(rows)


def rolling_corr(stocks: pd.Series, cands: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({c: cands[c].rolling(ROLL).corr(stocks) for c in cands}).dropna(how="all")


def _path_stats(r: pd.Series, cash: pd.Series) -> dict:
    """Growth over cash, volatility and worst fall of a monthly simple total return."""
    lx = np.log1p(r) - np.log1p(cash.reindex(r.index))
    lvl = np.log1p(r).cumsum()
    dd = np.expm1((lvl - np.maximum(lvl.cummax(), 0.0)).min())
    return {"growth": float(12 * lx.mean()), "vol": float(lx.std() * np.sqrt(12)),
            "sharpe": float(12 * lx.mean() / (lx.std() * np.sqrt(12))), "worst_fall": float(dd)}


def mixes(stocks: pd.Series, cands: pd.DataFrame, cash: pd.Series, share: float = MIX_SHARE) -> pd.DataFrame:
    """All stocks against ``share`` moved to the candidate (swap). Rebalanced monthly."""
    rows = []
    for w, start in WINDOWS.items():
        if start is None:
            continue
        have = [c for c in cands if cands[c].first_valid_index() <= pd.Timestamp(start)]
        j = pd.concat([stocks.rename("x"), cands[have]], axis=1, sort=True).loc[start:].dropna()
        rf = cash.reindex(j.index)
        tot = lambda lx: np.expm1(lx + np.log1p(rf))           # noqa: E731  simple total return
        rs = tot(j["x"])
        base = {"window": w, "start": str(j.index[0].date()), "end": str(j.index[-1].date()), "n": len(j)}
        rows.append({**base, "asset": "stocks only", "kind": "base", **_path_stats(rs, rf)})
        rows.append({**base, "asset": "cash", "kind": "swap", **_path_stats((1 - share) * rs + share * rf, rf)})
        for c in have:
            rc = tot(j[c])
            rows.append({**base, "asset": c, "kind": "swap", **_path_stats((1 - share) * rs + share * rc, rf)})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ the sleeve
def sleeve_weights(x: pd.DataFrame, scheme: str) -> pd.Series:
    """Constant weights over the columns of ``x`` (log excess returns): equal, or 1/volatility."""
    if scheme == "equal":
        return pd.Series(1.0 / x.shape[1], index=x.columns)
    if scheme == "inv_vol":
        w = 1.0 / x.std()
        return w / w.sum()
    raise KeyError(scheme)


def sleeve_table(stocks: pd.Series, cands: pd.DataFrame, cash: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(rows, correlations) of the sleeve on each window of ``SLEEVE``; see the module docstring.

    Row kinds: ``sleeve`` (the sleeve alone, with its correlation with stocks, its weights and
    its return in the window's bear markets) and ``mix`` (stocks and sleeve, summing to 100%)."""
    ep_all = bear_markets(stocks)
    rows, corr = [], []
    for w, cols in SLEEVE.items():
        j = pd.concat([stocks.rename(STOCKS), cands[cols]], axis=1, sort=True).loc[WINDOWS[w]:].dropna()
        rf, S = cash.reindex(j.index), np.expm1(j)             # S: simple excess returns
        base = {"window": w, "start": str(j.index[0].date()), "end": str(j.index[-1].date()), "n": len(j)}
        c = j.corr()
        corr += [{"window": w, "a": a, "b": b, "corr": float(c.loc[a, b])} for a in c.index for b in c.columns]
        ep = ep_all[ep_all["peak"] >= j.index[0] - pd.offsets.MonthEnd(1)]
        tot = lambda xs: (1 + xs) * (1 + rf) - 1               # noqa: E731  simple total return
        for scheme in ("equal", "inv_vol"):
            wt = sleeve_weights(j[cols], scheme)
            sl = S[cols] @ wt
            lx = np.log1p(sl)
            bears = [float(np.expm1(lx.loc[p + pd.offsets.MonthEnd(1):t].sum())) for p, t in zip(ep["peak"], ep["trough"])]
            head = {**base, "scheme": scheme}
            rows.append({**head, "kind": "sleeve", "stocks": 0.0, "sleeve": 1.0, **_path_stats(tot(sl), rf),
                         "corr_stocks": float(lx.corr(j[STOCKS])), "bears": len(bears),
                         "bears_down": int(sum(b < 0 for b in bears)), "worst_bear": min(bears) if bears else np.nan,
                         "weights": json.dumps({k: round(float(v), 4) for k, v in wt.items()})})
            for ws in SLEEVE_STOCKS:
                rows.append({**head, "kind": "mix", "stocks": ws, "sleeve": 1 - ws,
                             **_path_stats(tot(ws * S[STOCKS] + (1 - ws) * sl), rf)})
    return pd.DataFrame(rows), pd.DataFrame(corr)


def vehicles(path=None) -> dict:
    v = json.loads((VEHICLES if path is None else path).read_text())
    for r in v["vehicles"]:
        if r["kind"] != "outright" or not 0 < r["fee"] < 0.03:
            raise ValueError(f"vehicle {r.get('symbol')}: bad kind or fee")
    return v


def _sleeve_report(sv: pd.DataFrame, sc: pd.DataFrame, meta: dict) -> list[str]:
    L = ["## An equal-weight sleeve of the three that passed", "",
         "Intermediate Treasuries, gold and managed futures, a third each, rebalanced monthly; the stock share is then "
         "chosen separately. From 1985 managed futures are the paper factor less the measured drag; from March 2007 "
         "they are the live funds, which is the window to trust. The three were picked on this same history, so "
         "every number below is flattered by that choice.", ""]
    for w in SLEEVE:
        d = sv[sv["window"] == w]
        e = d[d["scheme"] == "equal"]
        c = sc[sc["window"] == w].pivot(index="a", columns="b", values="corr")
        names = [STOCKS, *SLEEVE[w]]
        lab = {STOCKS: "Stocks", "ust_int": "Treasuries", "gold": "Gold", "mf_tsmom_net": "Managed futures",
               "mf_funds": "Managed futures"}
        b = e.iloc[0]
        L += [f"### {b['start'][:7]} to {b['end'][:7]} ({b['n']} months)", "",
              _table(["Correlation", *[lab[n] for n in names]],
                     [[lab[a], *[_num(c.loc[a, x]) for x in names]] for a in names]), ""]
        rows = []
        for scheme, title in (("equal", "a third each"), ("inv_vol", "by 1/volatility")):
            s = d[(d["scheme"] == scheme) & (d["kind"] == "sleeve")].iloc[0]
            wt = json.loads(s["weights"])
            rows.append([f"Sleeve, {title}", " / ".join(f"{100 * wt[k]:.0f}%" for k in SLEEVE[w]), _pct(s["growth"]),
                         _pct(s["vol"], 1, False), _num(s["sharpe"]), _num(s["corr_stocks"]),
                         f"{s['bears_down']:.0f} of {s['bears']:.0f} (worst {_pct(s['worst_bear'], 0)})"])
        L += [_table(["", "Treasuries / gold / managed futures", "Growth", "Vol", "Sharpe", "Corr with stocks",
                      "Stock bear markets it fell in"], rows), ""]
        rows = []
        for _, r in e[e["kind"] == "mix"].iterrows():
            rows.append([f"{r['stocks']:.0%} stocks / {r['sleeve']:.0%} sleeve", _pct(r["growth"]),
                         _pct(r["vol"], 1, False), _num(r["sharpe"]), _pct(r["worst_fall"], 0)])
        L += [_table(["Equal-weight sleeve", "Growth over T-bills", "Vol", "Sharpe", "Worst fall"], rows), ""]
    L += ["The sleeve lowers growth as well as risk: it earned less than stocks. "
          "The 1/volatility weights use each window's own volatility.", ""]
    return L


def _vehicle_report(v: dict) -> list[str]:
    rows = lambda kind: [[r["symbol"], r["name"], r["exposure"], r["holds"], f"{100 * r['fee']:.2f}%", r["checked"],  # noqa: E731
                          r["note"]] for r in v["vehicles"] if r["kind"] == kind]
    head = ["Fund", "Name", "Exposure", "Holds", "Fee a year", "Fee read from", "Note"]
    fee = {r["symbol"]: r["fee"] for r in v["vehicles"]}
    cheap, tried = np.mean([fee["IAUM"], fee["VGIT"], fee["SDMF"]]), np.mean([fee["IAUM"], fee["VGIT"], fee["DBMF"]])
    return ["## What it costs to hold", "",
            f"Expense ratios looked up on {v['retrieved']} (`data/processed/diversifier_vehicles.json`): a hand-recorded "
            "outside lookup, mostly from secondary sites. Confirm on the issuer's page before use.", "",
            _table(head, rows("outright")), "",
            f"A third each in IAUM, VGIT and SDMF costs {100 * cheap:.2f}% a year; with DBMF, which has a record since "
            f"2019, {100 * tried:.2f}%.", "",
            "None of these funds is in the panel, and the study's returns are not theirs.", ""]


# ------------------------------------------------------------------ report
def _pct(v, d=1, sign=True):
    return "" if v is None or not np.isfinite(v) else f"{100 * v:{'+' if sign else ''}.{d}f}%"


def _num(v, d=2):
    return "" if v is None or not np.isfinite(v) else f"{v:+.{d}f}"


def _table(head: list[str], rows: list[list[str]]) -> str:
    return "\n".join(["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
                     + ["| " + " | ".join(r) + " |" for r in rows])


def figure(rc: pd.DataFrame) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIGURES.mkdir(parents=True, exist_ok=True)
    show = {"ust_int": "Intermediate Treasuries", "ig_credit": "Investment-grade corporates", "gold": "Gold",
            "commodities": "Broad commodities", "mf_tsmom": "Managed futures (paper factor)",
            "mf_funds": "Managed futures (live funds)"}
    fig, ax = plt.subplots(figsize=(9, 4.6))
    for c, lab in show.items():
        s = rc[c].dropna()
        ax.plot(s.index, s, label=lab, lw=2.2 if c.startswith("mf_") else 1.2,
                ls="--" if c == "mf_funds" else "-")
    ax.axhline(0, color="0.4", lw=0.8)
    for y in (LOW_CORR, MAX_ROLL):
        ax.axhline(y, color="0.7", lw=0.8, ls=":")
    ax.set_ylabel(f"correlation with stocks, trailing {ROLL // 12} years")
    ax.set_ylim(-0.8, 1.0)
    ax.legend(ncol=2, fontsize=8, frameon=False, loc="upper left")
    ax.set_title("No fixed number: correlation with stocks over rolling 10-year windows", fontsize=10)
    fig.tight_layout()
    p = FIGURES / "rolling_corr.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p.name


def write_report(sm: pd.DataFrame, ep: pd.DataFrame, mx: pd.DataFrame, meta: dict, fig: str,
                 sv: pd.DataFrame, sc: pd.DataFrame) -> str:
    own = sm[sm["window"] == "own"].set_index("asset")
    L = [f"# Which assets have diversified stocks?", "", f"*{LABEL}.*", "",
         f"Written by `python -m assetforecast.diversifiers`; returns through {meta['returns_through']}. "
         "Stocks are the world-stock composite of the three-fund restatement (US stocks only before 1975). "
         "Returns are monthly, over T-bills. Managed futures are not one of the package's 12 assets; "
         "they enter here from `data/sources/managed_futures.py` as a paper trend-following factor "
         f"(AQR, {meta['managed_futures']['tsmom']['start'][:4]} on, before fees and costs) and as a composite of "
         f"{len(meta['managed_futures']['funds'])} live funds (2007 on, after fees, survivors only).", "",
         "## Verdict on each asset's whole history", "",
         f"Thresholds, fixed before the tables were computed: correlation at most {LOW_CORR} at 1 and 12 months; "
         f"never above {MAX_ROLL} over any {ROLL // 12} years; in stocks' bear markets (falls of {BEAR:.0%} or more) "
         f"the asset loses at most {MAX_CAPTURE:.0%} of what stocks lose; and a positive return of its own after "
         "its stock exposure is removed (alpha).", ""]
    rows = [[NAMES[a], r["start"][:4], _num(r["corr_1m"]), _num(r["corr_12m"]),
             f"{_num(r['roll_min'])} to {_num(r['roll_max'])}", _pct(r["capture"], 0),
             f"{r['bears_down']:.0f} of {r['bears']:.0f} (worst {_pct(r['worst_bear'], 0)})",
             f"{_pct(r['alpha'])} ({_num(r['alpha_t'], 1)})", f"**{r['verdict']}**"] for a, r in own.iterrows()]
    L += [_table(["Asset", "From", "Corr 1m", "Corr 12m", f"Range over {ROLL // 12}y", "Share of stocks' bear-market loss",
                  "Bear markets it fell in", "Alpha a year (t)", "Verdict"], rows), "",
          "\"Share of stocks' bear-market loss\": the asset's summed return over all bear-market months in its "
          "history divided by stocks' summed return over the same months; negative means it rose while stocks fell. That sum lets a gain in one bear market offset a loss in another; "
          "\"Bear markets it fell in\" counts them one by one (all are listed at the end). "
          "Windows differ by asset, so rows are not like for like; the common-window tables below are.", "",
          f"![rolling correlation](figures/diversifiers/{fig})", ""]

    d = meta["drag"]
    L += ["## Managed futures: paper against live", "",
          f"On the {d['months']} months both series exist, the live funds ran at {d['vol_ratio_funds_to_factor']:.2f} "
          f"times the paper factor's volatility and earned {_pct(d['gap_at_fund_vol'], 1, False)} a year less than the "
          f"factor scaled to that volatility. At the factor's own volatility that is a drag of {_pct(d['drag'], 1, False)} "
          "a year (fees, trading costs, and whatever the paper rules flatter). The \"less measured drag\" row subtracts "
          "it from every month of the paper factor. Their monthly correlation with each other is "
          f"{meta['managed_futures']['overlap_corr_monthly']:.2f}.", ""]

    for w, title in (("since_1985", "Common window from 1985"), ("since_2007", "Common window from March 2007 (live funds exist)")):
        t = sm[sm["window"] == w].set_index("asset")
        m = mx[mx["window"] == w]
        L += [f"## {title}", "", f"{t['n'].max()} months, to {t['end'].max()}"
              + (f" (the paper trend factor ends {t.loc['mf_tsmom', 'end']})." if t.loc["mf_tsmom", "end"] != t["end"].max() else "."), ""]
        rows = [[NAMES[a], _num(r["corr_1m"]), _num(r["corr_12m"]), _num(r["down_corr"]),
                 _pct(r["worst_mean"]), _pct(r["capture"], 0), _pct(r["mean"]), _pct(r["vol"], 1, False),
                 f"{_pct(r['alpha'])} ({_num(r['alpha_t'], 1)})", r["verdict"]] for a, r in t.iterrows()]
        L += [_table(["Asset", "Corr 1m", "Corr 12m", "Corr when stocks fall", "Mean month in stocks' worst 10%",
                      "Share of bear-market loss", "Return a year", "Vol", "Alpha a year (t)", "Verdict"], rows), "",
              f"Stocks' own mean in their worst 10% of months: {_pct(t['worst_stocks'].iloc[0])}.", ""]
        base = m[m["kind"] == "base"].iloc[0]
        rows = [["All stocks", _pct(base["growth"]), _pct(base["vol"], 1, False), _num(base["sharpe"]),
                 _pct(base["worst_fall"], 0)]]
        for a in ["cash", *[c for c in CANDIDATES if c in set(m["asset"])]]:
            s = m[(m["asset"] == a) & (m["kind"] == "swap")].iloc[0]
            rows.append([NAMES.get(a, "Cash (T-bills)"), _pct(s["growth"]), _pct(s["vol"], 1, False), _num(s["sharpe"]),
                         _pct(s["worst_fall"], 0)])
        L += [f"What a {MIX_SHARE:.0%} holding did to an all-stock portfolio (growth is over T-bills, a year):", "",
              _table(["", "Growth", "Vol", "Sharpe", "Worst fall"], rows), "",
              f"{1 - MIX_SHARE:.0%} stocks, {MIX_SHARE:.0%} the asset, rebalanced monthly. "
              f"Every row uses the same {base['n']} months, {base['start']} to {base['end']}.", ""]

    cols = [c for c in CANDIDATES if c != "mf_tsmom_net"]
    short = {"ust_int": "Tsy 7y", "ust_long": "Tsy long", "tips": "TIPS", "ig_credit": "IG", "hy_credit": "HY",
             "reits": "REITs", "gold": "Gold", "commodities": "Cmdty", "mf_tsmom": "MF paper", "mf_funds": "MF funds"}
    rows = [[f"{p:%Y-%m} to {t:%Y-%m}", _pct(s, 0), *[_pct(r[c], 0) for c in cols]]
            for p, t, s, r in zip(ep["peak"], ep["trough"], ep["stocks"], ep.to_dict("records"))]
    L += ["## Every bear market in stocks", "",
          f"Month-end peak to trough of stocks over T-bills, falls of {BEAR:.0%} or more (a fall ends when stocks have "
          f"risen {BEAR:.0%} from the low). Each cell is that asset's return over T-bills across the same months; "
          "blank where the asset's history does not cover it.", "",
          _table(["Peak to trough", "Stocks", *[short[c] for c in cols]], rows), "",
          *_sleeve_report(sv, sc, meta), *_vehicle_report(vehicles()),
          "## Caveats", "",
          "- The paper trend factor is a backtest of rules published in 2012. Its 1985-2011 return is what the rules "
          "would have earned before costs, not what anyone earned. The drag is estimated on one "
          f"{d['months']}-month stretch and applied to all of it.",
          "- The fund composite holds only funds alive today, which flatters its return. Its correlation with stocks "
          "is less exposed to that.",
          "- Bear markets since 1985 number few; a share of bear-market loss rests on a handful of episodes.",
          "- The thresholds are judgements. Each column is in `results/diversifiers/summary.csv` for anyone who would "
          "draw the lines elsewhere.",
          "- Nothing here is a forecast. The published 12-month forecast still covers the 12 assets only.", ""]
    REPORT.write_text("\n".join(L))
    return str(REPORT)


def run() -> dict:
    from .data.sources import managed_futures
    stocks, cands, cash, drag = load_inputs()
    sm, ep = summary(stocks, cands), episode_returns(stocks, cands)
    rc, mx = rolling_corr(stocks, cands), mixes(stocks, cands, cash)
    sv, sc = sleeve_table(stocks, cands, cash)
    OUT.mkdir(parents=True, exist_ok=True)
    sm.to_csv(OUT / "summary.csv", index=False, float_format="%.6g")
    ep.to_csv(OUT / "episodes.csv", index=False, float_format="%.6g")
    rc.to_csv(OUT / "rolling_corr.csv", float_format="%.4f")
    mx.to_csv(OUT / "mixes.csv", index=False, float_format="%.6g")
    sv.to_csv(OUT / "sleeve.csv", index=False, float_format="%.6g")
    sc.to_csv(OUT / "sleeve_corr.csv", index=False, float_format="%.4f")
    meta = {"label": LABEL, "returns_through": str(stocks.dropna().index[-1].date()), "stocks": STOCKS,
            "thresholds": {"low_corr": LOW_CORR, "max_roll": MAX_ROLL, "max_capture": MAX_CAPTURE, "t_sig": T_SIG,
                           "roll_months": ROLL, "worst_share": WORST, "bear_depth": BEAR, "mix_share": MIX_SHARE},
            "sleeve": {"members": SLEEVE, "stock_shares": list(SLEEVE_STOCKS)},
            "vehicles_retrieved": vehicles()["retrieved"],
            "drag": drag, "managed_futures": json.loads(managed_futures.META.read_text())}
    (OUT / "meta.json").write_text(json.dumps(meta, indent=1))
    meta["report"] = write_report(sm, ep, mx, meta, figure(rc), sv, sc)
    return meta


def main(argv=None) -> int:
    meta = run()
    print(LABEL)
    print(pd.read_csv(OUT / "summary.csv").query("window == 'own'")[
        ["asset", "start", "corr_1m", "corr_12m", "roll_max", "capture", "alpha", "alpha_t", "verdict"]
    ].round(3).to_string(index=False))
    print(meta["report"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
