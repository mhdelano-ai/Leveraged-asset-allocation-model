"""The one-page summary: ``dashboard/summary.html``, a single printable sheet in plain language.

    python -m assetforecast.dashboard.summary        # also run by python -m assetforecast.dashboard

Four takeaways for a reader who knows markets but not this project: what the next twelve
months may look like, how far to trust it, what it means for a portfolio, how the forecast
is made. One self-contained file (no scripts, no data fetched), light by default and dark
when the viewer's device is; it prints on one Letter or A4 sheet in the light look.

No statistic is estimated here. :func:`facts` selects rows from files other steps wrote and
converts units; :func:`render` prints them. Sources:

* ``forecasts/latest/``  ``expected_returns.csv``, ``distribution_assets.csv``,
                         ``distribution_portfolios.csv``, ``frontier.csv``, ``frontier_points.csv``,
                         ``shape.csv``, ``forecast.json``
* ``results/final/``     ``accuracy.csv``, ``portfolio_summary.csv``, ``summary.json`` (frozen)
* ``data/processed/``   ``cash_monthly.parquet`` (the T-bill return of the origin month, a year)
* the dashboard's own builders ``verdicts``, ``snapshot`` and ``history_vol`` (``data.py``)

Units
-----
The report states returns as 12-month LOG returns in excess of T-bills. This page states the
same forecast as ordinary (simple) percent gains above cash, so its figures are higher than
the report's:

average        ``exp(drift) - 1`` with ``drift = mean_12m + 0.5 variance``, the expected simple
               return of a lognormal. For one asset this is ``mean_12m_arith`` of
               ``expected_returns.csv``; for a portfolio the drift is the ``drift`` column of
               ``frontier_points.csv`` and ``frontier.csv`` (weights times the assets' drifts).
range          ``exp(q) - 1`` of the published quantiles ``q`` (a quantile of the simple return
               is that transform of the quantile of the log return). "9 years in 10" is the
               5% to 95% span, "half of all years" the 25% to 75% span. Portfolios use the
               published portfolio shape.
swing          a volatility, unchanged: ``vol_12m_return`` and the frontier's ``vol``.
frontier band  ``exp(drift -/+ h) - 1`` with ``h`` the half-width of the published growth band.
swing excess   the square root of a realized/predicted VARIANCE ratio (``portfolio_summary.csv``
               and :data:`VARIANCE_HAIRCUT`).

Recommended mixes
-----------------
The four mixes are rows of the published frontier with cash (``frontier.csv``, curve
``with_cash``; nothing is borrowed): the points one fifth, two fifths, three fifths and four
fifths of the way, in volatility, from all cash to the top of the frontier. Nothing is
optimized here. Their bad year (1 in 20) follows the report's rule for optimized portfolios
(``reports/REPORT.md``, section 2):

    exp(center + sqrt(k vol^2 + (sum_i |w_i| se_i)^2) z) - 1

with ``center = sum_i w_i mean_12m_i``, ``k`` =
:data:`VARIANCE_HAIRCUT`, ``se`` the standard errors of the means and ``z`` the 5% point of the
asset-pooled shape (``shape.csv``, column ``z_assets``). They are arithmetic on the forecast,
not tested strategies, and the page says so beside them.
"""
from __future__ import annotations

import argparse
import html
import math
from pathlib import Path

import pandas as pd

from .. import paths
from ..universe import ASSETS, BY_ID, HORIZON_MONTHS
from . import data as dd

OUT = paths.ROOT / "dashboard" / "summary.html"
TITLE = "Asset Outlook Sheet"
HISTORY_FROM = "1990-01"       # first forecast date of the forecast-against-actual chart
HISTORY_ASSET = "us_large"
PEAK_GAP_YEARS = 5             # the two labelled peaks of that chart are at least this far apart
VARIANCE_HAIRCUT = 1.5         # reports/REPORT.md section 2: scale the predicted variance of optimized portfolios
HAIRCUT_MODEL, HAIRCUT_PORTFOLIO, HAIRCUT_PERIOD = "combo_cov_main", "gmv", "dev"

# Plain-language names. Text about what a thing is, never a result.
ASSET_NAMES = {
    "us_large": "US large companies", "us_small": "US small companies",
    "intl_dev": "Developed markets outside the US", "em": "Emerging markets",
    "ust_int": "Treasuries, mid-term (~7 years)", "ust_long": "Treasuries, long-term (20+ years)",
    "tips": "Inflation-protected Treasuries", "ig_credit": "Corporate bonds, high grade",
    "hy_credit": "Corporate bonds, high yield", "reits": "US real estate (REITs)",
    "gold": "Gold", "commodities": "Commodities",
}
GROUP_OF = {"equity": "Stocks", "rates": "Bonds", "credit": "Bonds", "real": "Real assets"}
GROUP_WORD = {"Stocks": "stocks", "Bonds": "bonds", "Real assets": "real assets"}
PORTFOLIO_NAMES = {"p6040": "60/40", "p_balanced": "Balanced mix", "p_equity": "Global stock mix"}
PORTFOLIO_GROUP = "Portfolios"
MIX_NAMES = ("Cautious", "Moderate", "Growth", "Aggressive")     # the four recommended mixes, by rising swing
MIX_TAU = 0.05                                                   # their bad year: the 1-in-20 outcome
MIX_MIN_WEIGHT = 0.005                                           # holdings below half a percent are not listed
HOLDING_NAMES = {
    "us_large": "US large stocks", "us_small": "US small stocks", "intl_dev": "developed ex-US stocks",
    "em": "emerging-market stocks", "ust_int": "mid-term Treasuries", "ust_long": "long-term Treasuries",
    "tips": "inflation-protected Treasuries", "ig_credit": "high-grade bonds", "hy_credit": "high-yield bonds",
    "reits": "REITs", "gold": "gold", "commodities": "commodities",
}


# --------------------------------------------------------------------------- units and text

def simple(x: float) -> float:
    """Log return -> simple return."""
    return math.expm1(float(x))


def pct(x: float, dp: int = 1, sign: bool = True) -> str:
    """A fraction as a percent with a true minus sign: 0.0802 -> '+8.0%'."""
    v = round(100 * float(x), dp)
    body = f"{abs(v):.{dp}f}%"
    return ("−" + body) if v < 0 else (("+" + body) if sign and v > 0 else body)


def num(x: float, dp: int = 1) -> str:
    return f"{float(x):.{dp}f}"


def years(first: str, last: str) -> str:
    return f"{str(first)[:4]}–{str(last)[:4]}"


def esc(s) -> str:
    return html.escape(str(s), quote=True)


# --------------------------------------------------------------------------- facts

def _mix(weights: dict) -> str:
    """'55% stocks, 33% bonds, 12% real assets' from a portfolio's weights."""
    share: dict[str, float] = {}
    for a, w in weights.items():
        g = GROUP_OF[BY_ID[a].group]
        share[g] = share.get(g, 0.0) + float(w)
    return ", ".join(f"{round(100 * w):d}% {GROUP_WORD[g]}" for g, w in share.items() if round(100 * w) > 0)


def _ranges(q: pd.Series) -> dict:
    """The 9-in-10 and half-of-all-years ranges of one row of published quantiles, as simple returns."""
    return {"lo": simple(q.q05), "hi": simple(q.q95), "mid_lo": simple(q.q25), "mid_hi": simple(q.q75)}


def outlook(s: dd.Sources) -> list[dict]:
    """One row per asset and policy portfolio: average, the two ranges, all as simple returns."""
    e = s.read("expected_returns.csv").set_index("asset")
    d = s.read("distribution_assets.csv").set_index("asset")
    rows = []
    for a in ASSETS:
        rows.append({"id": a.id, "kind": "asset", "group": GROUP_OF[a.group], "name": ASSET_NAMES[a.id], "detail": "",
                     "avg": float(e.loc[a.id, "mean_12m_arith"]), **_ranges(d.loc[a.id])})
    dp = s.read("distribution_portfolios.csv")
    dp = dp[dp["shape"] == "portfolios"].set_index("portfolio")
    pts = s.read("frontier_points.csv")
    drift = pts[pts.kind == "policy"].set_index("point").drift
    weights = s.json("forecast.json")["portfolio_weights"]
    for p, name in PORTFOLIO_NAMES.items():
        rows.append({"id": p, "kind": "portfolio", "group": PORTFOLIO_GROUP, "name": name, "detail": _mix(weights[p]),
                     "avg": simple(drift[p]), **_ranges(dp.loc[p])})
    return rows


def _window_label(origin: str) -> str:
    """The 12 months after month-end ``origin`` ('2008-09' -> '2008–09'; a calendar year -> '2009')."""
    y, m = int(origin[:4]), int(origin[5:7])
    return str(y + 1) if m == 12 else f"{y}–{(y + 1) % 100:02d}"


def _end_year(origin: str) -> float:
    """The end of the 12-month window after ``origin``, as a decimal year."""
    return int(origin[:4]) + 1 + int(origin[5:7]) / 12.0


def history(s: dd.Sources) -> dict:
    """Forecast and realized 12-month swing of one asset, by the END of the window they describe."""
    h = dd.history_vol(s)
    h = h[(h.asset == HISTORY_ASSET) & (h.origin >= HISTORY_FROM)].sort_values("origin")
    forecast = [(_end_year(o), float(f)) for o, f in zip(h.origin, h.forecast)]
    done = h[h.realized.notna()]
    actual = [(_end_year(o), float(r)) for o, r in zip(done.origin, done.realized)]
    top = done.loc[done.realized.idxmax()]
    far = done[(done.origin.str[:4].astype(int) - int(top.origin[:4])).abs() >= PEAK_GAP_YEARS]
    peaks = [top] + ([far.loc[far.realized.idxmax()]] if len(far) else [])
    peaks = sorted(({"t": _end_year(p.origin), "actual": float(p.realized), "forecast": float(p.forecast),
                     "label": _window_label(p.origin)} for p in peaks), key=lambda p: p["t"])
    return {"asset": ASSET_NAMES[HISTORY_ASSET], "forecast": forecast, "actual": actual, "peaks": peaks}


def trust(s: dd.Sources) -> dict:
    v = dd.verdicts(s).set_index("question")
    snap = dd.snapshot(s).iloc[0]
    acc = s.read("accuracy.csv", "final")
    acc = acc[(acc["sample"] == "last30y") & (acc.asset == "pooled")].iloc[0]
    vol, corr, mean = v.loc["cov_vol"], v.loc["cov_corr"], v.loc["mean_orig"]
    return {
        "vol_status": vol.status, "vol_skill": float(vol.skill), "corr_status": corr.status, "corr_skill": float(corr.skill),
        "tested": years(vol.first_origin, vol.last_origin),
        "mean_models": int(v.loc["mean_orig_h12"].n_models), "mean_tests": int(mean.n_cells), "mean_pass": int(mean.n_pass),
        "mean_from": str(snap.return_panel_from).split()[-1], "n_assets": int(snap.er_n), "n_zero": int(snap.er_n_zero),
        "oos_from": str(snap.oos_from), "first_outcome": str(snap.first_oos_outcome),
        "below": abs(float(acc.two_thirds_lo_pct)), "above": float(acc.two_thirds_hi_pct),
        "history": history(s),
    }


def cash_rate(s: dd.Sources, origin: str) -> float:
    """What T-bills earn at the origin: 12 times the month's T-bill return."""
    c = s.read("cash_monthly.parquet", "processed")["cash"]
    return float(12 * c.loc[pd.Timestamp(origin)])


def _with_cash(s: dd.Sources) -> pd.DataFrame:
    """The published frontier with cash (nothing borrowed), in order from all cash."""
    wc = s.read("frontier.csv")
    return wc[wc.curve == "with_cash"].sort_values("i").reset_index(drop=True)


def frontier(s: dd.Sources) -> dict:
    wc = _with_cash(s)
    half_lo, half_hi = wc.growth - wc.growth_lo, wc.growth_hi - wc.growth
    curve = [{"vol": float(r.vol), "avg": simple(r.drift), "lo": simple(r.drift - lo), "hi": simple(r.drift + hi),
              "invested": float(r.invested)}
             for r, lo, hi in zip(wc.itertuples(), half_lo, half_hi)]
    pts = s.read("frontier_points.csv").set_index("point")
    policies, gaps = [], []
    for p, name in PORTFOLIO_NAMES.items():
        own, best = pts.loc[p], pts.loc[f"match_{p}"]
        policies.append({"id": p, "name": name, "vol": float(own.vol), "avg": simple(own.drift)})
        gaps.append({"id": p, "gap": simple(best.drift) - simple(own.drift), "invested": float(best.invested),
                     "includes_zero": bool(best.gap_lo <= 0 <= best.gap_hi)})
    ps = s.read("portfolio_summary.csv", "final")
    ratio = ps[(ps.model == HAIRCUT_MODEL) & (ps.portfolio == HAIRCUT_PORTFOLIO) & (ps.metric == "var_ratio")
               & (ps.period == HAIRCUT_PERIOD)]["mean"].iloc[0]
    return {
        "curve": curve, "top": curve[-1], "policies": policies, "gaps": gaps, "mixes": mixes(s),
        "gap_min": min(g["gap"] for g in gaps), "gap_max": max(g["gap"] for g in gaps),
        "n_gap_zero": sum(g["includes_zero"] for g in gaps),
        "var_ratio": float(ratio), "swing_excess": math.sqrt(float(ratio)) - 1.0,
        "haircut": VARIANCE_HAIRCUT, "haircut_swing": math.sqrt(VARIANCE_HAIRCUT) - 1.0,
    }


def mixes(s: dd.Sources) -> list[dict]:
    """The recommended mixes: evenly spaced rows of the published frontier with cash (module docstring)."""
    wc = _with_cash(s)
    e = s.read("expected_returns.csv").set_index("asset")
    shape = s.read("shape.csv")
    z = float(shape[(shape.tau - MIX_TAU).abs() < 1e-9].z_assets.iloc[0])
    out = []
    for k, name in enumerate(MIX_NAMES, 1):
        r = wc.iloc[round(k * (len(wc) - 1) / (len(MIX_NAMES) + 1))]
        w = {a.id: float(r[f"w_{a.id}"]) for a in ASSETS}
        invested = float(r.invested)
        center = sum(w[a] * float(e.mean_12m[a]) for a in w)
        se = sum(abs(w[a]) * float(e.se_12m[a]) for a in w)
        scale = math.sqrt(VARIANCE_HAIRCUT * float(r.vol) ** 2 + se ** 2)
        groups: dict[str, float] = {g: 0.0 for g in GROUP_WORD}
        for a, x in w.items():
            groups[GROUP_OF[BY_ID[a].group]] += x
        held = sorted(((a, x) for a, x in w.items() if x >= MIX_MIN_WEIGHT), key=lambda t: -t[1])
        out.append({"no": k, "name": name, "vol": float(r.vol), "avg": simple(r.drift), "bad": simple(center + scale * z),
                    "invested": invested, "cash": max(1.0 - invested, 0.0),
                    "weights": w, "groups": groups, "holdings": [(HOLDING_NAMES[a], x) for a, x in held]})
    return out


def machine(s: dd.Sources) -> dict:
    fj = s.json("forecast.json")
    return {
        "n_assets": len(fj["assets"]), "since": min(a["history_starts"] for a in fj["assets"])[:4],
        "n_families": int(fj["assets"][0]["n_families"]),
        "n_models": int(s.json("summary.json", "final")["coverage"]["models"]["vol"]),
    }


def facts(s: dd.Sources | None = None) -> dict:
    """Everything the page shows, in the units of the module docstring."""
    s = s or dd.Sources()
    snap = dd.snapshot(s).iloc[0]
    through = pd.Timestamp(snap.data_through)
    return {
        "window": str(snap.window), "origin": str(snap.origin), "cash_rate": cash_rate(s, str(snap.origin)),
        "data_through": f"{through.day} {through.strftime('%B %Y')}",
        "outlook": outlook(s), "trust": trust(s), "frontier": frontier(s), "machine": machine(s),
    }


# --------------------------------------------------------------------------- charts

def _nice_span(lo: float, hi: float, step: float) -> tuple[float, float]:
    return math.floor(lo / step) * step, math.ceil(hi / step) * step


def _ticks(lo: float, hi: float, step: float) -> list[float]:
    n = round((hi - lo) / step)
    return [lo + i * step for i in range(n + 1)]


def range_chart(rows: list[dict]) -> str:
    """The outlook rows: one bar per asset on a shared percent axis (HTML, so it reflows on a phone)."""
    lo, hi = _nice_span(100 * min(r["lo"] for r in rows), 100 * max(r["hi"] for r in rows), 20)
    pos = lambda x: f"{(100 * x - lo) / (hi - lo) * 100:.2f}"            # noqa: E731
    bar = lambda cls, a, b: (f'<i class="{cls}" style="left:{pos(a)}%;'  # noqa: E731
                             f'width:{float(pos(b)) - float(pos(a)):.2f}%"></i>')
    ticks = _ticks(lo, hi, 20)
    axis = "".join(f'<span style="left:{(t - lo) / (hi - lo) * 100:.2f}%">{pct(t / 100, 0)}</span>' for t in ticks)
    grid = f"background-size:{100 / (len(ticks) - 1):.4f}% 100%"
    out = ['<div class="rng">',
           '<div class="rng-row rng-head"><span class="rng-g"></span><span class="rng-n"></span>'
           f'<div class="rng-axis" aria-hidden="true">{axis}</div>'
           '<span class="rng-a">Average</span><span class="rng-s">9 years in 10</span></div>']
    last = None
    for r in rows:
        first = r["group"] != last
        last = r["group"]
        cls = "rng-row" + (" rng-first" if first else "") + (" rng-port" if r["kind"] == "portfolio" else "")
        span = f'{pct(r["lo"])} to {pct(r["hi"])}'
        tip = (f'{r["name"]}: average {pct(r["avg"])}; 9 years in 10 between {span}; '
               f'half of all years between {pct(r["mid_lo"])} and {pct(r["mid_hi"])}')
        out.append(
            f'<div class="{cls}" title="{esc(tip)}">'
            f'<span class="rng-g">{esc(r["group"]) if first else ""}</span>'
            f'<span class="rng-n">{esc(r["name"])}</span>'
            f'<div class="rng-plot" style="{grid}"><i class="rng-zero" style="left:{pos(0)}%"></i>'
            f'{bar("rng-out", r["lo"], r["hi"])}{bar("rng-in", r["mid_lo"], r["mid_hi"])}'
            f'<i class="rng-dot" style="left:{pos(r["avg"])}%"></i></div>'
            f'<span class="rng-a">{pct(r["avg"])}</span><span class="rng-s">{span}</span></div>')
    out.append("</div>")
    mixes = " ".join(f'{esc(r["name"])}: {esc(r["detail"])}.' for r in rows if r["detail"])
    out.append(f'<p class="fine">{mixes} Their ranges come from how such mixes have behaved, the assets\' from single assets.</p>')
    return "\n".join(out)


def _path(points: list[tuple[float, float]]) -> str:
    return "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in points)


def history_chart(h: dict) -> str:
    """Forecast against actual swing, two lines with direct labels."""
    W, H, L, R, T, B = 460, 150, 30, 62, 14, 20
    t0, t1 = h["forecast"][0][0], h["forecast"][-1][0]
    top = max(max(v for _, v in h["actual"]), max(v for _, v in h["forecast"]))
    ymax = math.ceil(top / 10) * 10
    X = lambda t: L + (t - t0) / (t1 - t0) * (W - L - R)                 # noqa: E731
    Y = lambda v: T + (1 - v / ymax) * (H - T - B)                       # noqa: E731
    g = [f'<svg class="chart" viewBox="0 0 {W} {H}" role="img" aria-label="Forecast and actual yearly swing of '
         f'{esc(h["asset"])}: the forecast follows the level and misses the peaks">']
    for v in _ticks(0, ymax, 10):
        g.append(f'<line class="grid{" base" if v == 0 else ""}" x1="{L}" x2="{W - R}" y1="{Y(v):.1f}" y2="{Y(v):.1f}"/>'
                 f'<text class="tick" x="{L - 6}" y="{Y(v) + 3.5:.1f}" text-anchor="end">{v:.0f}%</text>')
    for yr in range(math.ceil(t0 / 5) * 5, int(t1) + 1, 5):
        g.append(f'<text class="tick" x="{X(yr):.1f}" y="{H - 6}" text-anchor="middle">{yr}</text>')
    g.append(f'<path class="line-b" d="{_path([(X(t), Y(v)) for t, v in h["actual"]])}"/>')
    g.append(f'<path class="line-a" d="{_path([(X(t), Y(v)) for t, v in h["forecast"]])}"/>')
    for p in h["peaks"]:
        g.append(f'<circle class="ring-b" cx="{X(p["t"]):.1f}" cy="{Y(p["actual"]):.1f}" r="3.5"/>'
                 f'<text class="note" x="{X(p["t"]):.1f}" y="{Y(p["actual"]) - 8:.1f}" text-anchor="middle">{esc(p["label"])}</text>')
    (ta, va), (tf, vf) = h["actual"][-1], h["forecast"][-1]
    ya, yf = Y(va), Y(vf)
    if abs(ya - yf) < 13:                                                # keep the two end labels apart
        mid = (ya + yf) / 2
        ya, yf = (mid + 6.5, mid - 6.5) if va < vf else (mid - 6.5, mid + 6.5)
    g.append(f'<circle class="dot-a" cx="{X(tf):.1f}" cy="{Y(vf):.1f}" r="4"/>'
             f'<text class="lab" x="{X(tf) + 8:.1f}" y="{yf + 4:.1f}">Forecast</text>'
             f'<circle class="dot-b" cx="{X(ta):.1f}" cy="{Y(va):.1f}" r="4"/>'
             f'<text class="lab muted" x="{X(tf) + 8:.1f}" y="{ya + 4:.1f}">Actual</text>')
    g.append("</svg>")
    return "".join(g)


def frontier_chart(b: dict) -> str:
    """The best mix at each level of swing, with cash below full investment and nothing borrowed."""
    W, H, L, R, T, B = 460, 190, 34, 60, 8, 34
    curve = b["curve"]
    xmax = math.ceil(100 * curve[-1]["vol"] / 5) * 5
    ymax = math.ceil(100 * max(c["hi"] for c in curve) / 5) * 5
    X = lambda v: L + 100 * v / xmax * (W - L - R)                       # noqa: E731
    Y = lambda v: T + (1 - 100 * v / ymax) * (H - T - B)                 # noqa: E731
    g = [f'<svg class="chart" viewBox="0 0 {W} {H}" role="img" aria-label="Expected gain above cash against the '
         'size of yearly swings for the best mix at each level of risk, with cash and nothing borrowed">']
    for v in _ticks(0, ymax, 5):
        g.append(f'<line class="grid{" base" if v == 0 else ""}" x1="{L}" x2="{W - R}" y1="{Y(v / 100):.1f}" y2="{Y(v / 100):.1f}"/>'
                 f'<text class="tick" x="{L - 6}" y="{Y(v / 100) + 3.5:.1f}" text-anchor="end">{v:.0f}%</text>')
    for v in _ticks(0, xmax, 5):
        g.append(f'<text class="tick" x="{X(v / 100):.1f}" y="{H - B + 14}" text-anchor="middle">{v:.0f}%</text>')
    g.append(f'<text class="axis" x="{W - R}" y="{H - 4}" text-anchor="end">Typical yearly swing →</text>')
    band = [(X(c["vol"]), Y(c["hi"])) for c in curve] + [(X(c["vol"]), Y(c["lo"])) for c in reversed(curve)]
    g.append(f'<path class="band" d="{_path(band)} Z"/>')
    g.append(f'<path class="line-a" d="{_path([(X(c["vol"]), Y(c["avg"])) for c in curve])}"/>')
    for m in b["mixes"]:                                                 # the recommended mixes, numbered as in the table
        x, y = X(m["vol"]), Y(m["avg"])
        g.append(f'<line class="lead" x1="{x:.1f}" x2="{x:.1f}" y1="{y - 4:.1f}" y2="{y - 10:.1f}"/>'
                 f'<circle class="dot-a" cx="{x:.1f}" cy="{y:.1f}" r="3.5"/>'
                 f'<circle class="bub" cx="{x:.1f}" cy="{y - 18:.1f}" r="8"/>'
                 f'<text class="bub-t" x="{x:.1f}" y="{y - 14.5:.1f}" text-anchor="middle">{m["no"]}</text>')
    g.append(f'<circle class="ring-a" cx="{X(0):.1f}" cy="{Y(0):.1f}" r="3.5"/>'
             f'<text class="lab muted" x="{X(0) + 7:.1f}" y="{Y(0) - 22:.1f}">All cash</text>')
    side = {"p6040": ("end", -7), "p_balanced": ("start", 7), "p_equity": ("start", 7)}
    for p in b["policies"]:
        anchor, dx = side[p["id"]]
        x, y = X(p["vol"]), Y(p["avg"])
        g.append(f'<circle class="dot-c" cx="{x:.1f}" cy="{y:.1f}" r="4"/>'
                 f'<text class="lab" x="{x + dx:.1f}" y="{y + 14:.1f}" text-anchor="{anchor}">{esc(p["name"])}</text>')
    g.append("</svg>")
    return "".join(g)


def _funding(m: dict) -> str:
    inv = round(100 * m["invested"])
    return f"{inv}% invested, {100 - inv}% cash" if inv < 100 else "fully invested"


def mix_table(b: dict) -> str:
    """The recommended mixes: what each holds (a bar by asset group and a list), its average, swing and bad year."""
    out = ['<div class="mixes">',
           '<div class="mix mix-top"><div class="legend" aria-hidden="true">'
           '<span><i class="k-sq k-stocks"></i>stocks</span><span><i class="k-sq k-bonds"></i>bonds</span>'
           '<span><i class="k-sq k-real"></i>real assets</span><span><i class="k-line"></i>100% of your money</span></div>'
           '<span class="mix-k">Average</span><span class="mix-k">Swing</span><span class="mix-k">Bad year</span></div>']
    seg = {"Stocks": "seg-stocks", "Bonds": "seg-bonds", "Real assets": "seg-real"}
    for m in b["mixes"]:
        bar = "".join(f'<i class="{seg[g]}" style="width:{100 * x:.2f}%"></i>' for g, x in m["groups"].items() if x >= MIX_MIN_WEIGHT)
        hold = ", ".join(f"{round(100 * x)}% {n}" for n, x in m["holdings"])
        tip = f'{m["name"]}: {_funding(m)}. {hold}.'
        out.append(
            f'<div class="mix" title="{esc(tip)}">'
            f'<div class="mix-name"><b class="mix-no">{m["no"]}</b><h3>{esc(m["name"])}</h3><small>{esc(_funding(m))}</small></div>'
            f'<span class="mix-v">{pct(m["avg"])}</span><span class="mix-v">{pct(m["vol"], 1, sign=False)}</span>'
            f'<span class="mix-v">{pct(m["bad"])}</span>'
            f'<div class="mix-bar" style="--own:100.00%">{bar}<b></b></div>'
            f'<p class="mix-hold">{esc(hold)}</p></div>')
    out.append("</div>")
    return "\n".join(out)


# --------------------------------------------------------------------------- sentences

def _tested(status: str, skill: float, span: str) -> tuple[str, str, str]:
    """(chip class, chip text, sentence) for a forecast scored once on years the model had not seen."""
    if status == "pass":
        return ("chip-on", "Tested", f"Scored once on {span} forecasts the model had never seen, its forecast error was "
                f"{num(abs(skill))}% lower than a simple benchmark's.")
    side = "ahead of" if skill >= 0 else "behind"
    return ("chip-off", "Unproven", f"On {span} forecasts the model had never seen, it finished {num(abs(skill))}% "
            f"{side} a simple benchmark, a gap chance alone could produce.")


def trust_items(t: dict) -> list[tuple[str, str, str, str]]:
    """(chip class, chip text, heading, sentence) for the four trust rows."""
    vol, corr = _tested(t["vol_status"], t["vol_skill"], t["tested"]), _tested(t["corr_status"], t["corr_skill"], t["tested"])
    beat = ("none did reliably better" if t["mean_pass"] == 0
            else f"a model did better in {t['mean_pass']} of {t['mean_tests']} tests")
    peaks = " and ".join(p["label"].replace("\u2013", "\u2013\u2060") for p in t["history"]["peaks"])   # no break after the dash
    return [
        (*vol[:2], "How bumpy each asset will be", vol[2]),
        (*corr[:2], "How assets move together", corr[2]),
        ("chip-off", "Average only", "How much each asset will earn",
         f"A long-run average since {t['mean_from']}, not a prediction. {t['mean_models']} forecasting models were "
         f"tested against it and {beat}. For {t['n_zero']} of the {t['n_assets']} assets the data cannot rule out "
         "an average of zero above cash."),
        ("chip-no", "Not predicted", "When a crisis starts",
         f"The forecast follows the general level of risk and reacts only after a shock, as {peaks} show below."),
    ]


# --------------------------------------------------------------------------- page

FONTS = "https://fonts.googleapis.com/css2?family=Archivo:wdth,wght@62..125,100..900&display=swap"

# Two looks from one sheet. Light: white, deep navy, bright blue, light type with generous space.
# Dark: black, white, aluminium greys, one red accent, wide heavy headings.
LIGHT = {
    "bg": "#ffffff", "panel": "#f3f6f9", "ink": "#001e50", "ink-2": "#3d4f72", "ink-3": "#6a7890",
    "rule": "#dfe4ea", "rule-strong": "#001e50", "grid": "#e6eaf0",
    "hi": "#0040c5", "band-in": "#00b0f0", "band-out": "#c4eafb", "line-b": "#76859c", "shade": "#e9f1fb",
    "g-stocks": "#0040c5", "g-bonds": "#0a9bd6", "g-real": "#d98a1f",
    "chip-ink": "#ffffff",
    "head-weight": "300", "head-stretch": "100%", "head-track": "-0.015em", "label-weight": "600",
    "h1-size": "clamp(1.75rem, 4.4vw, 2.75rem)",
}
DARK = {
    "bg": "#000000", "panel": "#141414", "ink": "#ffffff", "ink-2": "#c9c9c9", "ink-3": "#8f8f8f",
    "rule": "#2e2e2e", "rule-strong": "#ffffff", "grid": "#262626",
    "hi": "#f50537", "band-in": "#b3b3b3", "band-out": "#474747", "line-b": "#8c8c8c", "shade": "#1c1c1c",
    "g-stocks": "#f50537", "g-bonds": "#4a90cc", "g-real": "#b98a3a",
    "chip-ink": "#ffffff",
    "head-weight": "600", "head-stretch": "125%", "head-track": "0", "label-weight": "500",
    "h1-size": "clamp(1.5rem, 3.5vw, 2.125rem)",
}


def _tokens(t: dict) -> str:
    return " ".join(f"--{k}: {v};" for k, v in t.items())


CSS = """
/* Layout: one sheet. Masthead, a full-width range chart, two columns (trust | mixes), a five-step strip. */
:root { color-scheme: light; /*LIGHT*/ }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { color-scheme: dark; /*DARK*/ } }
:root[data-theme="dark"] { color-scheme: dark; /*DARK*/ }
@media print { :root:root:root { color-scheme: light; /*LIGHT*/ } }
*, *::before, *::after { box-sizing: border-box; }
html { font-size: 16px; -webkit-text-size-adjust: 100%; }
body { margin: 0; background: var(--bg); color: var(--ink); font-family: Archivo, "Helvetica Neue", Helvetica, Arial, sans-serif;
  font-size: 0.875rem; line-height: 1.45; -webkit-print-color-adjust: exact; print-color-adjust: exact;
  padding-block: 2.5rem 3rem; padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)); }
.sheet { max-width: 61rem; margin: 0 auto; display: flex; flex-direction: column; gap: 2.25rem; }
h1, h2, h3, p { margin: 0; }
h1, h2, .big { font-weight: var(--head-weight); font-stretch: var(--head-stretch); letter-spacing: var(--head-track); }
h1 { font-size: var(--h1-size); line-height: 1.1; text-wrap: balance; }
h2 { font-size: 1.5rem; line-height: 1.15; text-wrap: balance; }
h3 { font-size: 0.875rem; font-weight: 600; }
.eyebrow { font-size: 0.6875rem; font-weight: var(--label-weight); letter-spacing: 0.14em; text-transform: uppercase; color: var(--ink-2); }
.lede { color: var(--ink-2); max-width: 68ch; }
.lede-row { display: flex; flex-wrap: wrap; align-items: baseline; justify-content: space-between; gap: 0.5rem 1.5rem; }
small, .fine { font-size: 0.75rem; color: var(--ink-3); }
.num, .rng-a, .rng-s, .big { font-variant-numeric: tabular-nums; }

.mast { display: grid; grid-template-columns: minmax(0, 1.5fr) minmax(0, 1fr); gap: 1.5rem 3rem; align-items: end;
  padding-bottom: 1.75rem; border-bottom: 1px solid var(--rule-strong); }
.mast-text { display: flex; flex-direction: column; gap: 0.875rem; min-width: 0; }
.mast-key { display: flex; flex-direction: column; gap: 0.625rem; min-width: 0; }
.keys { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 1rem; }
.key { display: flex; flex-direction: column; gap: 0.125rem; border-top: 1px solid var(--rule); padding-top: 0.5rem; min-width: 0; }
.key .big { font-size: 1.625rem; line-height: 1.1; }
.key:first-child { border-top-color: var(--hi); }
.key span { font-size: 0.75rem; color: var(--ink-2); }

section { display: flex; flex-direction: column; gap: 0.875rem; min-width: 0; }
.sec-head { display: flex; flex-direction: column; gap: 0.375rem; }
.legend { display: flex; flex-wrap: wrap; gap: 0.375rem 1.25rem; font-size: 0.75rem; color: var(--ink-2); }
.legend span { display: inline-flex; align-items: center; gap: 0.4rem; }
.legend i { display: inline-block; height: 0.5rem; border-radius: 0.25rem; }
.k-out { width: 1.75rem; background: var(--band-out); }
.k-in { width: 1.25rem; background: var(--band-in); }
.k-dot { width: 0.625rem; height: 0.625rem !important; border-radius: 50% !important; background: var(--hi); }

.rng { display: flex; flex-direction: column; }
.rng-row { display: grid; grid-template-columns: 6.25rem minmax(0, 15rem) minmax(0, 1fr) 4rem 9.5rem; column-gap: 0.875rem;
  align-items: center; min-height: 1.5rem; }
.rng-first { border-top: 1px solid var(--rule); }
.rng-g { font-size: 0.6875rem; font-weight: var(--label-weight); letter-spacing: 0.1em; text-transform: uppercase; color: var(--ink-3); }
.rng-n { min-width: 0; line-height: 1.2; }
.rng-n small { white-space: nowrap; }
.rng-port .rng-n { font-weight: 600; }
.rng-a { text-align: right; font-weight: 600; }
.rng-s { text-align: right; color: var(--ink-2); white-space: nowrap; }
.rng-head { min-height: 1.375rem; font-size: 0.6875rem; color: var(--ink-3); }
.rng-head .rng-a, .rng-head .rng-s { font-weight: 400; color: var(--ink-3); }
.rng-axis { position: relative; height: 1rem; }
.rng-axis span { position: absolute; top: 0; transform: translateX(-50%); white-space: nowrap; font-variant-numeric: tabular-nums; }
.rng-plot { position: relative; align-self: stretch; min-height: 1.5rem;
  background-image: linear-gradient(to right, var(--grid) 1px, transparent 1px); border-right: 1px solid var(--grid); }
.rng-plot i { position: absolute; top: 50%; display: block; }
.rng-zero { top: 0 !important; bottom: 0; width: 1px; background: var(--ink-3); }
.rng-out { height: 0.5rem; margin-top: -0.25rem; border-radius: 0.25rem; background: var(--band-out); }
.rng-in { height: 0.5rem; margin-top: -0.25rem; border-radius: 0.25rem; background: var(--band-in); }
.rng-dot { width: 0.625rem; height: 0.625rem; margin: -0.3125rem 0 0 -0.3125rem; border-radius: 50%; background: var(--hi);
  box-shadow: 0 0 0 2px var(--bg); }

.duo { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 2.25rem 3rem; align-items: start; }
.items { display: flex; flex-direction: column; margin: 0; padding: 0; list-style: none; }
.items li { display: grid; grid-template-columns: 6.75rem minmax(0, 1fr); gap: 0.875rem; padding-block: 0.625rem; border-top: 1px solid var(--rule); }
.items p { color: var(--ink-2); font-size: 0.8125rem; }
.chip { justify-self: start; align-self: start; display: inline-flex; align-items: center; gap: 0.35rem; padding: 0.15rem 0.55rem;
  border: 1px solid var(--ink); border-radius: 999px; font-size: 0.6875rem; font-weight: 600; letter-spacing: 0.04em; white-space: nowrap; }
.chip-on { background: var(--hi); border-color: var(--hi); color: var(--chip-ink); }
.chip-no { border-style: dashed; color: var(--ink-2); border-color: var(--ink-3); }
figure { margin: 0; display: flex; flex-direction: column; gap: 0.375rem; }
figcaption { font-size: 0.75rem; color: var(--ink-2); }
figcaption b { color: var(--ink); font-weight: 600; }
.chart { display: block; width: 100%; height: auto; overflow: visible; font-family: inherit; }
.chart text { font-size: 10.5px; fill: var(--ink-2); font-variant-numeric: tabular-nums; }
.chart .tick, .chart .axis { fill: var(--ink-3); }
.chart .lab { fill: var(--ink); font-weight: 600; }
.chart .lab.muted { fill: var(--ink-2); }
.chart .note { fill: var(--ink-2); }
.chart .grid { stroke: var(--grid); stroke-width: 1; fill: none; }
.chart .grid.base { stroke: var(--ink-3); }
.chart .line-a { fill: none; stroke: var(--hi); stroke-width: 2; stroke-linejoin: round; stroke-linecap: round; }
.chart .line-a.dash { stroke-dasharray: 1 5; stroke-width: 2.5; }
.chart .line-b { fill: none; stroke: var(--line-b); stroke-width: 1.5; stroke-linejoin: round; }
.chart .band { fill: var(--shade); stroke: none; }
.chart .lead { stroke: var(--ink-3); stroke-width: 1; fill: none; }
.chart .dot-a { fill: var(--hi); stroke: var(--bg); stroke-width: 2; }
.chart .dot-b { fill: var(--line-b); stroke: var(--bg); stroke-width: 2; }
.chart .dot-c { fill: var(--ink); stroke: var(--bg); stroke-width: 2; }
.chart .ring-a { fill: var(--bg); stroke: var(--hi); stroke-width: 2; }
.chart .ring-b { fill: var(--bg); stroke: var(--line-b); stroke-width: 1.5; }
.chart .bub { fill: var(--ink); stroke: none; }
.chart .bub-t { fill: var(--bg); font-size: 10px; font-weight: 700; }

.mixes { display: flex; flex-direction: column; }
.mix { display: grid; grid-template-columns: minmax(0, 1fr) 3.75rem 3.25rem 4rem; column-gap: 0.75rem; row-gap: 0.3rem;
  align-items: baseline; padding-block: 0.5rem; border-top: 1px solid var(--rule); }
.mix-top { padding-block: 0 0.3rem; border-top: 0; font-size: 0.6875rem; color: var(--ink-3); }
.mix-top .legend { font-size: 0.6875rem; gap: 0.25rem 0.75rem; }
.mix-k, .mix-v { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
.mix-v { font-weight: 600; }
.mix-name { display: flex; flex-wrap: wrap; align-items: baseline; gap: 0.1rem 0.5rem; min-width: 0; }
.mix-name small { white-space: nowrap; }
.mix-no { display: inline-flex; align-items: center; justify-content: center; width: 1.15rem; height: 1.15rem; border-radius: 50%;
  background: var(--ink); color: var(--bg); font-size: 0.6875rem; font-weight: 700; align-self: center; }
.mix-bar { grid-column: 1 / -1; position: relative; display: flex; gap: 2px; height: 0.5rem; }
.mix-bar::before { content: ""; position: absolute; inset: 0 auto 0 0; width: var(--own); background: var(--grid); border-radius: 0.25rem; }
.mix-bar i { position: relative; display: block; height: 100%; }
.mix-bar i:first-of-type { border-radius: 0.25rem 0 0 0.25rem; }
.mix-bar i:last-of-type { border-radius: 0 0.25rem 0.25rem 0; }
.mix-bar b { position: absolute; top: -0.2rem; bottom: -0.2rem; left: var(--own); width: 1px; background: var(--ink); }
.seg-stocks, .k-stocks { background: var(--g-stocks); }
.seg-bonds, .k-bonds { background: var(--g-bonds); }
.seg-real, .k-real { background: var(--g-real); }
.legend .k-sq { width: 0.625rem; height: 0.5rem; border-radius: 2px; }
.legend .k-line { width: 1px; height: 0.8rem; border-radius: 0; background: var(--ink); }
.mix-hold { grid-column: 1 / -1; font-size: 0.75rem; color: var(--ink-2); }
.caution { background: var(--panel); padding: 0.75rem 0.875rem; font-size: 0.8125rem; color: var(--ink-2); border-left: 2px solid var(--hi); }
.caution b { color: var(--ink); font-weight: 600; }

.flow { display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); margin: 0; padding: 0; list-style: none; border-top: 1px solid var(--rule-strong); }
.flow li { position: relative; display: flex; flex-direction: column; gap: 0.25rem; padding: 0.75rem 1.25rem 0 0; min-width: 0; }
.flow li + li { padding-left: 1.25rem; border-left: 1px solid var(--rule); }
.flow li + li::before { content: ""; position: absolute; left: -0.3125rem; top: 0.9rem; width: 0.5rem; height: 0.5rem;
  border-top: 1px solid var(--ink); border-right: 1px solid var(--ink); transform: rotate(45deg); background: var(--bg); }
.flow .step { display: flex; align-items: baseline; flex-wrap: wrap; gap: 0 0.5rem; }
.flow .big { font-size: 1.375rem; line-height: 1.1; }
.flow p { font-size: 0.75rem; color: var(--ink-2); }

footer { border-top: 1px solid var(--rule); padding-top: 0.75rem; display: flex; flex-direction: column; gap: 0.25rem; }
footer p { font-size: 0.6875rem; color: var(--ink-3); max-width: none; }

@media screen and (max-width: 760px) {
  .mast, .duo { grid-template-columns: minmax(0, 1fr); }
  .rng-row { grid-template-columns: minmax(0, 1fr) auto; row-gap: 0.125rem; padding-block: 0.375rem; }
  .rng-g { grid-column: 1 / -1; }
  .rng-g:empty, .rng-head .rng-n, .rng-head .rng-a, .rng-head .rng-s { display: none; }
  .rng-n { grid-column: 1; }
  .rng-a { grid-column: 2; grid-row: 2; }
  .rng-row:not(.rng-head) .rng-g:empty + .rng-n, .rng-row:not(.rng-head) .rng-g:empty ~ .rng-a { grid-row: 1; }
  .rng-plot, .rng-axis { grid-column: 1 / -1; }
  .rng-s { grid-column: 1 / -1; text-align: left; font-size: 0.75rem; }
  .flow { grid-template-columns: minmax(0, 1fr); border-top: 0; }
  .flow li, .flow li + li { padding: 0.75rem 0; border-left: 0; border-top: 1px solid var(--rule); }
  .flow li + li::before { display: none; }
}
@page { margin: 7mm 10mm; }
@media print {
  html { font-size: 10.4px; }
  body { padding: 0; line-height: 1.34; }
  .sheet { max-width: none; gap: 1.15rem; }
  h1 { font-size: 2.3rem; }
  h2 { font-size: 1.35rem; }
  .lede { max-width: none; }
  .mast { padding-bottom: 0.9rem; gap: 1rem 2.5rem; }
  .mast-text { gap: 0.6rem; }
  section { gap: 0.5rem; break-inside: avoid; }
  .screen-only { display: none; }                 /* the curve behind the four mixes: the sheet keeps the table */
  .mix { padding-block: 0.4rem; row-gap: 0.25rem; }
  .sec-head { gap: 0.2rem; }
  .rng-row, .rng-plot { min-height: 1.2rem; }
  .duo { gap: 2.25rem; }
  .items li { padding-block: 0.32rem; }
  .caution { padding: 0.55rem 0.75rem; }
  .flow li { padding-top: 0.5rem; }
  footer { padding-top: 0.5rem; }
}
"""


def render(f: dict) -> str:
    t, b, m = f["trust"], f["frontier"], f["machine"]
    p6040 = next(r for r in f["outlook"] if r["id"] == "p6040")
    h = t["history"]
    items = "".join(f'<li><span class="chip {cls}">{esc(chip)}</span><div><h3>{esc(head)}</h3><p>{esc(text)}</p></div></li>'
                    for cls, chip, head, text in trust_items(t))
    zero = ("for all of them" if b["n_gap_zero"] == len(b["gaps"]) else f"for {b['n_gap_zero']} of {len(b['gaps'])}")
    body = f"""
<main class="sheet">
<header class="mast">
  <div class="mast-text">
    <p class="eyebrow">Asset-class outlook &middot; {esc(f["window"])}</p>
    <h1>The size of the swings can be forecast. The gains cannot.</h1>
    <p class="lede">Each month this model looks {m["n_assets"]} asset classes a year ahead: what each has earned above
    cash on average, how wide the range of outcomes is, and how much of that deserves belief.
    Market data through {esc(f["data_through"])}.</p>
  </div>
  <div class="mast-key">
    <p class="eyebrow">A 60/40 portfolio over the next 12 months, above cash</p>
    <div class="keys">
      <div class="key"><b class="big">{pct(p6040["avg"])}</b><span>on average</span></div>
      <div class="key"><b class="big">{pct(p6040["lo"])}</b><span>a bad year<br>(1 in 20 is worse)</span></div>
      <div class="key"><b class="big">{pct(p6040["hi"])}</b><span>a good year<br>(1 in 20 is better)</span></div>
    </div>
  </div>
</header>

<section class="outlook">
  <div class="sec-head">
    <p class="eyebrow">What next year may look like</p>
    <h2>Wide ranges around modest averages</h2>
    <div class="lede-row">
      <p class="lede">Gain or loss above cash over the 12 months, in ordinary percent. Cash itself yields about
      {pct(f["cash_rate"], 1, sign=False)} today.</p>
      <div class="legend" aria-hidden="true">
        <span><i class="k-out"></i>9 years in 10</span>
        <span><i class="k-in"></i>half of all years</span>
        <span><i class="k-dot"></i>average</span>
      </div>
    </div>
  </div>
  {range_chart(f["outlook"])}
</section>

<div class="duo">
<section class="trust">
  <div class="sec-head">
    <p class="eyebrow">How far to trust it</p>
    <h2>The risk side is tested. The return side is an average.</h2>
  </div>
  <ul class="items">{items}</ul>
  <figure>
    <figcaption><b>{esc(h["asset"])}: forecast against actual yearly swing,</b> by 12-month period.</figcaption>
    {history_chart(h)}
    <figcaption>Across all assets over the last 30 years, the actual swing landed between {num(t["below"], 0)}% below
    and {num(t["above"], 0)}% above the forecast two years in three.</figcaption>
  </figure>
</section>

<section class="best">
  <div class="sec-head">
    <p class="eyebrow">Recommended mixes at four risk levels</p>
    <h2>Four mixes, from cautious to aggressive</h2>
  </div>
  <figure class="screen-only">
    <figcaption><b>Expected gain above cash for the best mix at each level of risk,</b> with the rest in cash and
    nothing borrowed. Shaded: how far off the averages could be.</figcaption>
    {frontier_chart(b)}
  </figure>
  {mix_table(b)}
  <p class="caution"><b>These are the forecast&rsquo;s best mixes, not tested strategies.</b> At equal risk their edge over
  the plain portfolios is {num(100 * b["gap_min"])} to {num(100 * b["gap_max"])} points a year, and {zero} it could be zero.
  None of them borrows. Optimizer-built portfolios swung {num(100 * b["swing_excess"], 0)}% more than forecast in back-tests, so each bad year
  assumes swings {num(100 * b["haircut_swing"], 0)}% above the forecast.</p>
</section>
</div>

<section class="machine">
  <div class="sec-head">
    <p class="eyebrow">How the forecast is made</p>
    <h2>Many models for risk, one long average for return</h2>
  </div>
  <ol class="flow">
    <li><div class="step"><b class="big">{esc(m["since"])}</b><h3>Market history</h3></div><p>Prices for {m["n_assets"]} asset classes; the oldest series starts in {esc(m["since"])}.</p></li>
    <li><div class="step"><b class="big">{m["n_models"]}</b><h3>Risk models compared</h3></div><p>Each estimates how bumpy every asset will be.</p></li>
    <li><div class="step"><b class="big">{m["n_families"]}</b><h3>Families averaged</h3></div><p>The models&rsquo; families are blended, so no single blind spot decides.</p></li>
    <li><div class="step"><b class="big">{esc(t["mean_from"])}</b><h3>Average returns added</h3></div><p>Each asset&rsquo;s average gain above cash over its recorded history.</p></li>
    <li><div class="step"><b class="big">{HORIZON_MONTHS} mo</b><h3>A range for the year</h3></div><p>Risk and average combine into the ranges above, every month.</p></li>
  </ol>
</section>

<footer>
  <p>Gains are ordinary yearly percentages above cash (T-bills). The full report states the same forecast as compounded
  returns, so its figures read lower. Ranges are built from how these assets have behaved in the past;
  crises are not anticipated. Forecasts made from {esc(t["oos_from"])} on are the first that can count as new evidence for
  the return side; the first is judged in {esc(t["first_outcome"])}.</p>
  <p>Planning inputs, not investment advice. Every figure is produced by code in the repository from the forecast dated
  {esc(f["origin"])}: see reports/REPORT.md and forecasts/latest.</p>
</footer>
</main>
"""
    css = CSS.replace("/*LIGHT*/", _tokens(LIGHT)).replace("/*DARK*/", _tokens(DARK))
    return "\n".join([
        "<!doctype html>",
        "<!-- Generated by python -m assetforecast.dashboard.summary (src/assetforecast/dashboard/summary.py). Do not edit. -->",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">',
        '<meta name="claude:viewport-fit-sides" content="cover">',
        f"<title>{TITLE}</title>",
        f'<link rel="stylesheet" href="{FONTS}">',
        f"<style>{css}</style>",
        "</head>",
        "<body>",
        body.strip("\n"),
        "</body>",
        "</html>",
        "",
    ])


def build(out: Path = OUT, s: dd.Sources | None = None) -> Path:
    """Write the page. Its only inputs are the files named in the module docstring."""
    out.write_text(render(facts(s)), encoding="utf-8")
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m assetforecast.dashboard.summary", description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=OUT, help="where to write the page")
    a = ap.parse_args(argv)
    p = build(a.out)
    print(f"wrote {p} ({p.stat().st_size / 1e3:.0f} kB)")


if __name__ == "__main__":
    main()
