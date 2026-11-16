"""The allocation questionnaire: one balanced mix, scaled with T-bills or leverage to a drawdown budget.

    python -m assetforecast.questionnaire

Writes ``results/questionnaire/`` (``answers.csv``, ``curve.csv``, ``funds.csv``,
``inputs.csv``, ``cov.csv``, ``meta.json``), ``reports/questionnaire.md`` and
``dashboard/questionnaire.html`` (``questionnaire_page``). ARITHMETIC UNDER DECLARED
RULES (``LABEL``): it reads the diversifier and allocation studies and the published
forecast, fits nothing, optimizes nothing on returns, selects no model and changes
nothing frozen or published. The rules are judgements, written down here before the
tables were computed; they are not advice and have no out-of-sample evidence.

Risk decides, returns do not
----------------------------
The allocation study's reading: risk carries over from one period to the next and can
be forecast; returns cannot, and rules that optimized on them did not beat fixed
weights. So the mix is set by risk alone, and how much of it to hold by the fall the
goal can sit through. Expected growth is shown beside each point with its interval,
and is not an input.

The mix
-------
World stocks, intermediate Treasuries, gold and managed futures, each weighted by
1/volatility on the whole past (:func:`balanced`; the allocation study's reading for
weights, applied to all four). Its three diversifiers are in the proportions of the
allocation study's sleeve (``SLEEVE_RULE``), so every portfolio here is two numbers:
a share of capital in stocks and a share in that sleeve.

The questions
-------------
``GOALS``     what the money is for. Each goal declares ``loss_limit``, its drawdown
              budget: the largest fall from a previous high it can sit through.
              ``near`` (spent within 3 years) has none and is answered with T-bills.
``ACCOUNTS``  the account's tax treatment. It decides which ``ROUTES`` can carry
              exposure above 100% of capital, and the tax notes shown. Taxes themselves
              are not modelled.
``ROUTES``    ``stacked``: funds that hold stocks and add a diversifier with futures,
              so the financing is inside the fund (counted at the T-bill rate; their
              fees are shown, not charged). The only route in a retirement account,
              which cannot borrow, and the default in a taxable one. ``margin``: a
              margin loan (the published frontier's, ``frontier.borrow_terms``), in a
              taxable account only.

The path (``curve.csv``, one per route)
---------------------------------------
``scaled``  the mix at ``STEP``, 2 ``STEP``, ... times capital. Below 1 the rest is
            T-bills; above 1 the route carries it. It stops at the most the route can
            carry: ``frontier.MAX_LEVERAGE`` times capital on margin; with stacked
            funds, where the funds of the lookup no longer fit in the capital
            (:func:`stacked_plan`; they stack a diversifier only on top of stocks, and
            the mix holds few stocks).
``tilted``  at that limit there is one way left to take more risk: stocks in place of
            diversifiers. Stocks rise in steps of ``STEP`` to 100% of capital, each
            with the most sleeve the route can still carry. Borrowing is never used to
            hold more stocks than the capital.

The rule: the point furthest along the route's path whose worst fall in the whole
replay is no deeper than the goal's ``loss_limit``.

Per point: ``vol``, its volatility on the allocation study's ``fcst`` matrix at the
last origin (the package's 12-month forecast for stocks, Treasuries and gold; managed
futures, which the forecast does not cover, at their last-``RECENT``-months volatility
and correlations); the replay; and, for display only, ``growth`` with its band
(``frontier.stats``: the expected log return over T-bills of the rebalanced mix, net of
the borrowing spread, and the 90% intervals of the means added as if perfectly
correlated). The means are each holding's historical mean (``returns.expected``):
stocks from ``threefund.expected``, Treasuries and gold from the published forecast,
managed futures from the live funds only (2007 on, after fees), the window the
diversifier study says to trust.

The replay is the allocation study's: the sleeve's walk-forward return (weights set
at each month-end from returns through it, held over the next month, from ``BURN``
months after the common start), beside stocks, rebalanced monthly, borrowing at
T-bills plus the spread. Each point is replayed at its own fixed shares of stocks and
sleeve; for ``scaled`` points those are today's mix, whose stock share has moved
little over the replay (``meta.json``: ``mix_stock_share_range``). Same series, same
statistics (``diversifiers._path_stats``); ``menu.csv`` of the allocation study is
reproduced where the two share a mix.

Funds
-----
``funds.csv``: one way to hold each answer with the funds of
``data/processed/diversifier_vehicles.json`` (a hand-recorded fee lookup). With
nothing above 100%, or on margin: a stock fund and the cheapest recorded fund per
diversifier. With stacked funds: the cheapest recorded mix of stacked and plain funds
that gives the diversifiers exactly and fits in the capital (:func:`stacked_plan`); it
may leave stocks up to ``STOCK_TOL`` short of the target, which is reported
(``stocks_held``). No stock fund's fee is in the lookup.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import linprog

from . import allocation as A
from . import diversifiers, frontier, paths, threefund
from .returns.expected import expected_from_xr

LABEL = ("Arithmetic under declared rules on the diversifier and allocation studies and the published "
         "forecast; the rules are judgements, not advice, and nothing here is out-of-sample evidence")
ROUTES = {   # how exposure above 100% of capital is carried
    "stacked": {"name": "Stacked funds",
                "detail": "Funds that hold stocks and add a diversifier with futures. No loan and no margin call."},
    "margin": {"name": "Margin loan", "detail": "Borrowed from the broker. It costs more and can be called."},
}
ACCOUNTS = {   # routes: the ways this account can carry more than 100%, the default first
    "taxable": {"name": "Taxable brokerage", "detail": "Gains, interest and distributions are taxed as they come.",
                "routes": ["stacked", "margin"]},
    "traditional": {"name": "Tax-deferred", "detail": "Traditional IRA or 401(k): taxed as income on withdrawal.",
                    "routes": ["stacked"]},
    "roth": {"name": "Roth", "detail": "Roth IRA or Roth 401(k): qualified withdrawals are untaxed.",
             "routes": ["stacked"]},
}
GOALS = {   # loss_limit: the drawdown budget, the largest fall from a previous high the goal can sit through
    "near": {"name": "Spending within 3 years", "detail": "A purchase, a reserve.", "loss_limit": None},
    "medium": {"name": "Spending in 3 to 10 years", "detail": "A house, tuition.", "loss_limit": 0.20},
    "income": {"name": "Retirement income, drawn now", "detail": "Withdrawals have begun.",
               "loss_limit": threefund.LOSS_LIMITS[0]},
    "growth": {"name": "Growth, untouched for 10 years or more", "detail": "Retirement that is still far off.",
               "loss_limit": threefund.LOSS_LIMITS[1]},
}
NOTES = {   # shown with the answer; statements about tax treatment, not modelled
    "taxable": [
        "Both ways of holding more than 100% of capital are open in this account: funds that carry the exposure "
        "inside, or a margin loan.",
        "Taxes are not modelled. Stock index funds defer most of their gain until sold. The diversifiers are taxed "
        "harder: Treasury interest is ordinary income (free of state tax), funds holding gold bullion are taxed as "
        "collectibles, and funds that use futures pay out most of their return as taxable distributions.",
        "The replay rebalances every month. In this account a sale can realise a gain."],
    "traditional": [
        "A retirement account cannot take a margin loan. Exposure above 100% of capital can only come from funds "
        "that carry it inside.",
        "Nothing is taxed until withdrawal, and withdrawals are taxed as ordinary income whatever was held. No "
        "holding is tax-favoured over another, and rebalancing costs no tax."],
    "roth": [
        "A retirement account cannot take a margin loan. Exposure above 100% of capital can only come from funds "
        "that carry it inside.",
        "Qualified withdrawals are untaxed. No holding is tax-favoured over another, and rebalancing costs no tax."],
}
SLEEVE_RULE = "invvol_hist"          # the allocation study's rule whose proportions the mix's three diversifiers have
STEP = 0.05                          # steps along the path: of exposure when scaling, of the stock share when tilting
STOCK_TOL = STEP / 2                 # how far short of its stock target a stacked-fund plan may fall
EXPOSURE = {"ust_int": "intermediate Treasuries", "gold": "gold", "mf": "managed futures"}   # names in the fund lookup
STOCK_FUND = "A world stock fund"
CASH_FUND = "T-bills or a money-market fund"
ASSETS, STOCKS, SLEEVE, NAMES = A.ASSETS, A.STOCKS, A.SLEEVE, A.NAMES

_DEV = bool(os.environ.get("ASSETFORECAST_MAX_DATE"))
OUT = paths.ROOT / "results" / ("dev/questionnaire" if _DEV else "questionnaire")
REPORT = paths.REPORTS / ("questionnaire_dev.md" if _DEV else "questionnaire.md")


# ------------------------------------------------------------------ the mix and the replay
def balanced(cov: pd.DataFrame) -> pd.Series:
    """The mix: the four holdings by 1/volatility on ``cov`` (the whole past's covariance)."""
    return A.inv_vol(cov.loc[ASSETS, ASSETS])


def sleeve_replay(x: pd.DataFrame, rule: str = SLEEVE_RULE) -> pd.Series:
    """The sleeve's monthly simple excess return, weights set at each month-end from returns through it."""
    S, out = np.expm1(x), {}
    for i in range(A.BURN - 1, len(x) - 1):
        w = A.sleeve_weights(rule, A.estimates(S, i, None))
        out[x.index[i + 1]] = float(w @ S.loc[x.index[i + 1], SLEEVE])
    return pd.Series(out)


def mix_stats(stocks: pd.Series, sleeve: pd.Series, cash: pd.Series, a: float, b: float, spread: float) -> dict:
    """Path statistics of ``a`` stocks + ``b`` sleeve (simple excess returns), per period of the allocation study.

    What ``a + b`` leaves of the capital is in T-bills; what it exceeds the capital by costs ``spread`` over them."""
    r = a * stocks.reindex(sleeve.index) + b * sleeve - spread / 12 * max(a + b - 1.0, 0.0)
    row = {}
    for p, start in A.PERIODS.items():
        rp = r.loc[start:] if start else r
        rf = cash.reindex(rp.index)
        row.update({f"{k}_{p}": v for k, v in diversifiers._path_stats((1 + rp) * (1 + rf) - 1, rf).items()})
    return row


def weights(stocks: float, sleeve: float, ws: pd.Series) -> np.ndarray:
    return pd.Series({STOCKS: stocks, **(sleeve * ws).to_dict()})[ASSETS].to_numpy(float)


def _grid(lo: float, hi: float, step: float = STEP) -> np.ndarray:
    return np.round(np.arange(lo, hi + 1e-9, step), 6)


def expected_four(x_funds: pd.Series, origin: pd.Timestamp, forecast_dir=None) -> pd.DataFrame:
    """Each holding's historical mean and 90% interval (module docstring), with where it was read."""
    d = paths.FORECASTS_OUT / str(origin.date()) if forecast_dir is None else forecast_dir
    pub = pd.read_csv(d / "expected_returns.csv", index_col=0)
    if json.loads((d / "forecast.json").read_text())["origin"] != str(origin.date()):
        raise ValueError(f"{d} is not the forecast of {origin.date()}")
    cols = ["mean_12m", "ci90_lo", "ci90_hi", "first_month", "n_months"]
    e = pd.concat([threefund.expected(origin).loc[[STOCKS], cols], pub.loc[["ust_int", "gold"], cols],
                   expected_from_xr(x_funds.dropna().to_frame("mf"), origin)[cols]])
    e["source"] = ["threefund.expected", "published forecast", "published forecast", "live funds only"]
    return e.loc[ASSETS]


# ------------------------------------------------------------------ funds, and what a route can carry
def vehicles(path=None) -> dict:
    v = diversifiers.vehicles(path)
    for r in v["vehicles"]:
        if r["kind"] == "stacked" and not (0 < r["stocks_per_dollar"] <= 1.5 and 0 < r["exposure_per_dollar"] <= 1.5):
            raise ValueError(f"vehicle {r['symbol']}: stacked funds need stocks_per_dollar and exposure_per_dollar")
    return v


def _fund(veh: dict, asset: str, stacked: bool) -> dict:
    pool = [r for r in veh["vehicles"] if r["exposure"] == EXPOSURE[asset] and r["kind"] == ("stacked" if stacked else "outright")]
    if stacked:
        return min(pool, key=lambda r: (-r["exposure_per_dollar"] / r["stocks_per_dollar"], r["fee"]))
    return min(pool, key=lambda r: r["fee"])


def stacked_plan(stocks: float, sleeve: float, ws: pd.Series, veh: dict) -> dict | None:
    """Shares of capital in stacked and plain funds that hold ``stocks`` and ``sleeve`` with no loan; None if none fits.

    A small linear program. Per diversifier, dollars in its stacked fund (which carries that
    diversifier and stocks) and in its plain fund; dollars in a plain stock fund; and the stock
    shortfall. The diversifiers are met exactly, the dollars sum to at most the capital, stocks
    are never above the target, and the plan with the least shortfall, then the least recorded
    fees, is taken. It fits if the shortfall is at most ``STOCK_TOL``."""
    f, o, n = [_fund(veh, a, True) for a in SLEEVE], [_fund(veh, a, False) for a in SLEEVE], len(SLEEVE)
    cost = [r["fee"] for r in f] + [r["fee"] for r in o] + [0.0, 1.0]
    A_eq, b_eq = np.zeros((n + 1, 2 * n + 2)), np.zeros(n + 1)
    for i, a in enumerate(SLEEVE):
        A_eq[i, i], A_eq[i, n + i], b_eq[i] = f[i]["exposure_per_dollar"], 1.0, sleeve * float(ws[a])
        A_eq[n, i] = f[i]["stocks_per_dollar"]
    A_eq[n, 2 * n], A_eq[n, 2 * n + 1], b_eq[n] = 1.0, 1.0, stocks
    A_ub = np.concatenate([np.ones(2 * n + 1), [0.0]])[None, :]
    r = linprog(cost, A_ub=A_ub, b_ub=[1.0], A_eq=A_eq, b_eq=b_eq, bounds=[(0, None)] * (2 * n + 2), method="highs")
    if r.status != 0 or r.x[-1] > STOCK_TOL + 1e-9:
        return None
    z = np.where(r.x < 1e-9, 0.0, r.x)
    return {"stacked": dict(zip(SLEEVE, z[:n])), "plain": dict(zip(SLEEVE, z[n:2 * n])), "stock_fund": float(z[2 * n]),
            "short": float(z[2 * n + 1])}


def carries(route: str, stocks: float, sleeve: float, ws: pd.Series, veh: dict) -> bool:
    """Whether ``route`` can hold these shares of capital."""
    if stocks + sleeve > frontier.MAX_LEVERAGE + 1e-9:
        return False
    if stocks + sleeve <= 1.0 + 1e-9 or route == "margin":
        return True
    return stacked_plan(stocks, sleeve, ws, veh) is not None


def fund_plan(d: dict, ws: pd.Series, veh: dict) -> list[dict]:
    """Funds and their shares of capital for a decision ``d`` (module docstring, Funds).

    Columns ``stocks`` and ``exposure`` are what each line carries, as shares of capital."""
    def line(r, role, w, stocks=0.0, exposure=0.0, note=None):
        return {"symbol": r["symbol"], "name": r["name"], "role": role, "weight": float(w), "stocks": float(stocks),
                "exposure": float(exposure), "fee": r["fee"], "note": r["note"] if note is None else note}

    stock = {"symbol": "", "name": STOCK_FUND, "fee": np.nan, "note": "no stock fund's fee is in the lookup"}
    rows = []
    if d["financing"] == "stacked":
        p = stacked_plan(d["stocks"], d["sleeve"], ws, veh)
        if p is None:
            raise ValueError("the stacked funds need more than the capital there is")
        if p["stock_fund"] > 0:
            rows.append(line(stock, STOCKS, p["stock_fund"], stocks=p["stock_fund"]))
        for a in SLEEVE:
            if p["stacked"][a] > 0:
                r, k = _fund(veh, a, True), p["stacked"][a]
                rows.append(line(r, a, k, k * r["stocks_per_dollar"], k * r["exposure_per_dollar"], r["holds"]))
            if p["plain"][a] > 0:
                rows.append(line(_fund(veh, a, False), a, p["plain"][a], exposure=p["plain"][a]))
    else:
        if d["stocks"] > 0:
            rows.append(line(stock, STOCKS, d["stocks"], stocks=d["stocks"]))
        for a in SLEEVE:
            w = d["sleeve"] * float(ws[a])
            if w > 0:
                rows.append(line(_fund(veh, a, False), a, w, exposure=w))
    if d["cash"] > 5e-4:
        rows.append({"symbol": "", "name": CASH_FUND, "role": "cash", "weight": float(d["cash"]), "stocks": 0.0,
                     "exposure": 0.0, "fee": np.nan, "note": ""})
    return rows


def _fees(rows: list[dict]) -> float:
    return float(sum(r["weight"] * r["fee"] for r in rows if np.isfinite(r["fee"])))


# ------------------------------------------------------------------ the path
def path(route: str, w_stocks: float, ws: pd.Series, veh: dict) -> list[tuple[str, float, float]]:
    """(segment, stocks, sleeve) along a route: the mix scaled as far as the route carries it, then tilted to stocks."""
    pts = []
    for k in _grid(STEP, frontier.MAX_LEVERAGE):
        s, d = float(k * w_stocks), float(k * (1.0 - w_stocks))
        if not carries(route, s, d, ws, veh):
            break
        pts.append(("scaled", s, d))
    for s in _grid(STEP, 1.0):
        if s <= pts[-1][1] + 1e-9:
            continue
        most = next((d for d in _grid(STEP, frontier.MAX_LEVERAGE - s)[::-1] if carries(route, s, d, ws, veh)), None)
        if most is not None:
            pts.append(("tilted", float(s), float(most)))
    return pts


def curve(inp: frontier.Inputs, ws: pd.Series, stocks: pd.Series, sleeve: pd.Series, cash: pd.Series,
          spreads: dict, routes: dict) -> pd.DataFrame:
    """Every point an answer can be, per route, in order (``i``), with its forecast risk, its replay and its growth."""
    rows = []
    for route, pts in routes.items():
        for i, (segment, a, b) in enumerate(pts):
            sp = spreads[route] if a + b > 1.0 + 1e-9 else 0.0
            w = weights(a, b, ws)
            fs = frontier.stats(inp, w, sp)
            rows.append({"route": route, "i": i, "segment": segment, "exposure": a + b, "stocks": a, "sleeve": b,
                         "cash": max(1.0 - a - b, 0.0), "borrowed": max(a + b - 1.0, 0.0), "spread": sp, "vol": fs["vol"],
                         "growth": fs["growth"], "growth_lo": fs["growth_lo"], "growth_hi": fs["growth_hi"],
                         **{f"w_{k}": float(v) for k, v in zip(ASSETS, w)}, **mix_stats(stocks, sleeve, cash, a, b, sp)})
    return pd.DataFrame(rows)


def decide(route: str, goal: str, cv: pd.DataFrame) -> dict:
    """The point of ``route``'s path for a goal: the furthest whose worst fall is within the goal's limit."""
    limit = GOALS[goal]["loss_limit"]
    none = {"segment": "cash", "i": -1, "stocks": 0.0, "sleeve": 0.0, "cash": 1.0, "borrowed": 0.0, "financing": "none"}
    if limit is None:
        return none
    p = cv[cv["route"] == route]
    ok = p[p["worst_fall_all"] >= -limit - 1e-12]
    if ok.empty:
        return none
    r = ok.loc[ok["i"].idxmax()]
    return {"segment": r["segment"], "i": int(r["i"]), "stocks": float(r["stocks"]), "sleeve": float(r["sleeve"]),
            "cash": float(r["cash"]), "borrowed": float(r["borrowed"]), "financing": route if r["borrowed"] > 1e-9 else "none"}


# ------------------------------------------------------------------ wording
def _p(v: float) -> str:
    return f"{100 * v:.0f}%"


def _times(k: float) -> str:
    return f"{k:.2f}".rstrip("0").rstrip(".") + " times capital"


def headline(d: dict) -> str:
    if d["cash"] >= 1.0:
        return "T-bills"
    s = f"{_p(d['stocks'])} stocks and {_p(d['sleeve'])} diversifiers"
    if d["cash"] > 5e-4:
        return s + f", with {_p(d['cash'])} in T-bills"
    if d["borrowed"] > 5e-4:
        return s + (f", on a margin loan of {_p(d['borrowed'])} of capital" if d["financing"] == "margin"
                    else f", {_p(d['borrowed'])} of capital carried inside stacked funds")
    return s


def reason(goal: str, d: dict, row: dict, cap: dict) -> str:
    """Why this point. ``cap``: the end of the route's scaled segment (its exposure and its worst fall)."""
    limit = GOALS[goal]["loss_limit"]
    if limit is None:
        return "Money needed within 3 years has no room for a fall, so it stays in T-bills."
    s, fall = f"The goal can sit through a fall of {_p(limit)}. ", f"{100 * row['worst_fall_all']:.0f}%"
    if d["segment"] == "tilted":
        return (s + f"Scaled to the most this account can carry, {_times(cap['exposure'])}, the mix's worst fall in the "
                f"replay was {100 * cap['worst_fall_all']:.0f}%, still inside the budget. From there stocks take the place of diversifiers "
                f"until the budget is used: {_p(d['stocks'])} stocks, with a worst fall of {fall}.")
    k = d["stocks"] + d["sleeve"]
    how = f"at {_p(k)} of capital with the rest in T-bills" if k < 1 - 1e-9 else f"at {_times(k)}"
    return s + f"The mix is held {how}, the most whose worst fall in the replay stayed within that ({fall})."


# ------------------------------------------------------------------ run
def build() -> dict:
    x, cash, _, _ = A.load_inputs()
    origin = x.index[-1]
    S = np.expm1(x)
    est = A.estimates(S, len(x) - 1, A.forecast_cov([origin]).get(origin))
    if "fcst" not in est:
        raise ValueError(f"no covariance forecast at {origin.date()}")
    cov, mix = est["fcst"][1].loc[ASSETS, ASSETS], balanced(est["hist"][1])
    ws = A.sleeve_weights(SLEEVE_RULE, est)
    if not np.allclose(mix[SLEEVE] / mix[SLEEVE].sum(), ws[SLEEVE]):
        raise ValueError("the mix's diversifiers are not in the sleeve's proportions")
    past = [float(balanced(A.estimates(S, i, None)["hist"][1])[STOCKS]) for i in range(A.BURN - 1, len(x) - 1)]
    borrow = frontier.borrow_terms(origin)
    spreads = {"margin": float(borrow["spread"]), "stacked": 0.0}
    exp = expected_four(diversifiers.load_inputs()[1]["mf_funds"], origin)
    inp = frontier.Inputs(cov, exp, {}, spreads["margin"])      # used for its arithmetic (stats); nothing is optimized
    veh = vehicles()
    cv = curve(inp, ws, S[STOCKS], sleeve_replay(x), cash, spreads,
               {r: path(r, float(mix[STOCKS]), ws, veh) for r in ROUTES})
    caps = {r: cv[(cv["route"] == r) & (cv["segment"] == "scaled")].iloc[-1][["exposure", "worst_fall_all", "vol"]].to_dict()
            for r in ROUTES}

    keep = ["exposure", "spread", "vol", "growth", "growth_lo", "growth_hi", *[f"w_{a}" for a in ASSETS],
            *[c for c in cv.columns if c.endswith(("_all", "_live"))]]
    answers, funds = [], []
    for acc, spec in ACCOUNTS.items():
        for route in spec["routes"]:
            for goal in GOALS:
                d = decide(route, goal, cv)
                row = {"account": acc, "route": route, "goal": goal, "loss_limit": GOALS[goal]["loss_limit"], **d}
                if d["cash"] < 1.0:
                    row.update(cv[(cv["route"] == route) & (cv["i"] == d["i"])].iloc[0][keep].to_dict())
                else:
                    row.update(exposure=0.0, spread=0.0, vol=0.0, growth=0.0, growth_lo=0.0, growth_hi=0.0,
                               **{f"w_{a}": 0.0 for a in ASSETS})
                plan = fund_plan(d, ws, veh)
                row.update(fund_fees=_fees(plan), loan_cost=row["spread"] * d["borrowed"],
                           stocks_held=sum(r["stocks"] for r in plan))
                row.update(headline=headline(d), reason=reason(goal, d, row, caps[route]))
                answers.append(row)
                funds += [{"account": acc, "route": route, "goal": goal, **r} for r in plan]

    meta = {"label": LABEL, "origin": str(origin.date()), "returns_through": str(origin.date()),
            "first_holding_month": str(x.index[A.BURN].date()), "live_from": A.LIVE, "assets": ASSETS, "names": NAMES,
            "accounts": ACCOUNTS, "routes": ROUTES, "goals": GOALS, "notes": NOTES,
            "settings": {"mix_rule": "1/volatility, whole past", "step": STEP, "stock_tol": STOCK_TOL,
                         "max_leverage": frontier.MAX_LEVERAGE, "spreads": spreads, "recent": A.RECENT},
            "mix": {a: float(mix[a]) for a in ASSETS}, "sleeve_weights": {a: float(ws[a]) for a in SLEEVE},
            "mix_stock_share_range": [min(past), max(past)], "caps": caps, "borrow": borrow,
            "vehicles_retrieved": veh["retrieved"],
            "covariance": {"model": A.covhistory.PRIMARY, "bridge": A.covhistory.BRIDGE}}
    inputs = exp.assign(name=[NAMES[a] for a in exp.index], vol=np.sqrt(np.diag(cov))).rename_axis("asset").reset_index()
    return {"answers": pd.DataFrame(answers), "curve": cv, "funds": pd.DataFrame(funds), "inputs": inputs,
            "cov": cov, "meta": meta}


TABLES = ("answers", "curve", "funds", "inputs")
STALE = ("ladder.csv", "frontier.csv", "points.csv")       # written by the first version, which drew an optimizer's frontier


def run() -> dict:
    res = build()
    OUT.mkdir(parents=True, exist_ok=True)
    for f in STALE:
        (OUT / f).unlink(missing_ok=True)
    for t in TABLES:
        res[t].to_csv(OUT / f"{t}.csv", index=False, float_format="%.6g")
    res["cov"].to_csv(OUT / "cov.csv", float_format="%.8g")
    (OUT / "meta.json").write_text(json.dumps(res["meta"], indent=1))
    return res


def main(argv=None) -> int:
    res = run()
    print(LABEL)
    pd.set_option("display.width", 220)
    print(res["answers"][["account", "route", "goal", "segment", "stocks", "sleeve", "cash", "borrowed", "vol",
                          "worst_fall_all", "growth"]].round(3).to_string(index=False))
    if not _DEV:
        from . import questionnaire_page
        print(questionnaire_page.write_report(), questionnaire_page.write_page())
    return 0


if __name__ == "__main__":
    sys.exit(main())
