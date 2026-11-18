"""The allocation questionnaire: stocks sized to a drawdown budget, the diversifiers for the rest.

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
be forecast; returns cannot, and since the live managed-futures funds began no rule that
optimized on them beat fixed weights. So nothing here is chosen on an expected return.
Expected growth is shown beside each point with its interval, and is not an input.

Stocks, then diversifiers
-------------------------
Stocks        World stocks cause the large falls, so the goal's drawdown budget sets the
              stock share: the largest, in steps of ``STEP``, whose worst fall in the
              whole replay is no deeper than the budget. The fall is measured on the
              whole portfolio, diversifiers included, so the budget holds for what is
              actually held.
Diversifiers  The sleeve the diversifier study left (intermediate Treasuries, gold and
              managed futures, in the fixed proportions of ``SLEEVE_RULE``: 1/volatility
              on the whole past) has held up when stocks fell, so it takes the rest of
              the capital.

Nothing borrows: every portfolio holds at most 100% of capital. So every portfolio is one
number, the share of capital in stocks, with the sleeve in the rest, and there is one
path (``curve.csv``): every stock share from 0 to 100%.

The questions
-------------
``GOALS``     what the money is for. Each goal declares ``loss_limit``, its drawdown
              budget: the largest fall from a previous high it can sit through.
              ``near`` (spent within 3 years) has none and is answered with T-bills,
              as is any goal whose budget no point of the path fits.
``ACCOUNTS``  the account's tax treatment. It does not change the mix; it decides the
              tax notes shown (``NOTES``). Taxes themselves are not modelled.

Per point: ``vol``, its volatility on the allocation study's ``fcst`` matrix at the
last origin (the package's 12-month forecast for stocks, Treasuries and gold; managed
futures, which the forecast does not cover, at their last-``RECENT``-months volatility
and correlations); the replay; and, for display only, ``growth`` with its band
(``frontier.stats``: the expected log return over T-bills of the rebalanced mix, and
the 90% intervals of the means added as if perfectly correlated). The means are each
holding's historical mean (``returns.expected``): stocks from ``threefund.expected``,
Treasuries and gold from the published forecast, managed futures from the live funds
only (2007 on, after fees), the window the diversifier study says to trust.

The replay is the allocation study's: the sleeve's walk-forward return (weights set
at each month-end from returns through it, held over the next month, from ``BURN``
months after the common start), beside stocks, rebalanced monthly. Each point is
replayed at its own fixed shares of stocks and sleeve. Same series, same statistics
(``diversifiers._path_stats``); ``menu.csv`` of the allocation study is reproduced where
the two share a mix.

Funds
-----
``answers.csv`` also carries ``fund_fees`` (recorded expense ratios times shares of
capital) and ``growth_all_net`` (the replay's growth less ``fund_fees``; the replay
charges no fees).

``funds.csv``: one way to hold each answer with the funds of
``data/processed/diversifier_vehicles.json`` (a hand-recorded fee lookup): a stock fund
and the cheapest recorded fund per diversifier. No stock fund's fee is in the lookup.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

from . import allocation as A
from . import diversifiers, frontier, paths, threefund
from .returns.expected import expected_from_xr

LABEL = ("Arithmetic under declared rules on the diversifier and allocation studies and the published "
         "forecast; the rules are judgements, not advice, and nothing here is out-of-sample evidence")
ACCOUNTS = {
    "taxable": {"name": "Taxable brokerage", "detail": "Gains, interest and distributions are taxed as they come."},
    "traditional": {"name": "Tax-deferred", "detail": "Traditional IRA or 401(k): taxed as income on withdrawal."},
    "roth": {"name": "Roth", "detail": "Roth IRA or Roth 401(k): qualified withdrawals are untaxed."},
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
        "Taxes are not modelled. Stock index funds defer most of their gain until sold. The diversifiers are taxed "
        "harder: Treasury interest is ordinary income (free of state tax), funds holding gold bullion are taxed as "
        "collectibles, and funds that use futures pay out most of their return as taxable distributions.",
        "The replay rebalances every month. In this account a sale can realise a gain."],
    "traditional": [
        "Nothing is taxed until withdrawal, and withdrawals are taxed as ordinary income whatever was held. No "
        "holding is tax-favoured over another, and rebalancing costs no tax."],
    "roth": [
        "Qualified withdrawals are untaxed. No holding is tax-favoured over another, and rebalancing costs no tax."],
}
SLEEVE_RULE = "invvol_hist"          # the weights of the three diversifiers among themselves
STEP = 0.05                          # steps of the stock share
EXPOSURE = {"ust_int": "intermediate Treasuries", "gold": "gold", "mf": "managed futures"}   # names in the fund lookup
STOCK_FUND = "A world stock fund"
CASH_FUND = "T-bills or a money-market fund"
ASSETS, STOCKS, SLEEVE, NAMES = A.ASSETS, A.STOCKS, A.SLEEVE, A.NAMES

_DEV = bool(os.environ.get("ASSETFORECAST_MAX_DATE"))
OUT = paths.ROOT / "results" / ("dev/questionnaire" if _DEV else "questionnaire")
REPORT = paths.REPORTS / ("questionnaire_dev.md" if _DEV else "questionnaire.md")


# ------------------------------------------------------------------ the replay
def sleeve_replay(x: pd.DataFrame, rule: str = SLEEVE_RULE) -> pd.Series:
    """The sleeve's monthly simple excess return, weights set at each month-end from returns through it."""
    S, out = np.expm1(x), {}
    for i in range(A.BURN - 1, len(x) - 1):
        w = A.sleeve_weights(rule, A.estimates(S, i, None))
        out[x.index[i + 1]] = float(w @ S.loc[x.index[i + 1], SLEEVE])
    return pd.Series(out)


def mix_stats(stocks: pd.Series, sleeve: pd.Series, cash: pd.Series, a: float, b: float) -> dict:
    """Path statistics of ``a`` stocks + ``b`` sleeve (simple excess returns), per period of the allocation study.

    ``a + b`` is at most 1; what it leaves of the capital is in T-bills."""
    if a + b > 1.0 + 1e-9:
        raise ValueError("nothing borrows: stocks and sleeve hold at most 100% of capital")
    return A.period_stats(a * stocks.reindex(sleeve.index) + b * sleeve, cash)


def weights(stocks: float, sleeve: float, ws: pd.Series) -> np.ndarray:
    return pd.Series({STOCKS: stocks, **(sleeve * ws).to_dict()})[ASSETS].to_numpy(float)


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


# ------------------------------------------------------------------ funds
def vehicles(path=None) -> dict:
    return diversifiers.vehicles(path)


def _fund(veh: dict, asset: str) -> dict:
    """The cheapest recorded fund for a diversifier."""
    return min((r for r in veh["vehicles"] if r["exposure"] == EXPOSURE[asset]), key=lambda r: r["fee"])


def fund_plan(d: dict, ws: pd.Series, veh: dict) -> list[dict]:
    """Funds and their shares of capital for a decision ``d`` (module docstring, Funds)."""
    rows = []
    if d["stocks"] > 0:
        rows.append({"symbol": "", "name": STOCK_FUND, "role": STOCKS, "weight": float(d["stocks"]), "fee": np.nan,
                     "note": "no stock fund's fee is in the lookup"})
    for a in SLEEVE:
        w = d["sleeve"] * float(ws[a])
        if w > 0:
            r = _fund(veh, a)
            rows.append({"symbol": r["symbol"], "name": r["name"], "role": a, "weight": float(w), "fee": r["fee"],
                         "note": r["note"]})
    if d["cash"] > 5e-4:
        rows.append({"symbol": "", "name": CASH_FUND, "role": "cash", "weight": float(d["cash"]), "fee": np.nan, "note": ""})
    return rows


def _fees(rows: list[dict]) -> float:
    return float(sum(r["weight"] * r["fee"] for r in rows if np.isfinite(r["fee"])))


# ------------------------------------------------------------------ the path
def path() -> list[tuple[float, float]]:
    """(stocks, sleeve): every stock share, with the sleeve in the rest of the capital."""
    return [(float(s), float(round(1.0 - s, 6))) for s in np.round(np.arange(0.0, 1.0 + 1e-9, STEP), 6)]


def curve(inp: frontier.Inputs, ws: pd.Series, stocks: pd.Series, sleeve: pd.Series, cash: pd.Series) -> pd.DataFrame:
    """Every point an answer can be, in order (``i``), with its forecast risk, its replay and its growth."""
    rows = []
    for i, (a, b) in enumerate(path()):
        w = weights(a, b, ws)
        fs = frontier.stats(inp, w)
        rows.append({"i": i, "stocks": a, "sleeve": b, "vol": fs["vol"],
                     "growth": fs["growth"], "growth_lo": fs["growth_lo"], "growth_hi": fs["growth_hi"],
                     **{f"w_{k}": float(v) for k, v in zip(ASSETS, w)}, **mix_stats(stocks, sleeve, cash, a, b)})
    return pd.DataFrame(rows)


def decide(goal: str, cv: pd.DataFrame) -> dict:
    """The point of the path for a goal: the largest stock share whose worst fall is within the goal's limit."""
    limit = GOALS[goal]["loss_limit"]
    none = {"i": -1, "stocks": 0.0, "sleeve": 0.0, "cash": 1.0}
    if limit is None:
        return none
    ok = cv[cv["worst_fall_all"] >= -limit - 1e-12]
    if ok.empty:
        return none
    r = ok.loc[ok["i"].idxmax()]
    return {"i": int(r["i"]), "stocks": float(r["stocks"]), "sleeve": float(r["sleeve"]),
            "cash": max(1.0 - float(r["stocks"]) - float(r["sleeve"]), 0.0)}


# ------------------------------------------------------------------ wording
def _p(v: float) -> str:
    return f"{100 * v:.0f}%"


def headline(d: dict) -> str:
    if d["cash"] >= 1.0:
        return "T-bills"
    parts = [f"{_p(v)} {k}" for k, v in (("stocks", d["stocks"]), ("diversifiers", d["sleeve"])) if v > 5e-4]
    return " and ".join(parts) + (f", with {_p(d['cash'])} in T-bills" if d["cash"] > 5e-4 else "")


def reason(goal: str, d: dict, row: dict) -> str:
    """Why this point: one sentence for the stocks, one for the diversifiers."""
    limit = GOALS[goal]["loss_limit"]
    if limit is None:
        return "Money needed within 3 years has no room for a fall, so it stays in T-bills."
    if d["cash"] >= 1.0:
        return f"The goal can sit through a fall of {_p(limit)}, and no mix of stocks and diversifiers stayed within that."
    return (f"The goal can sit through a fall of {_p(limit)}. Stocks: {_p(d['stocks'])}, the most whose worst fall in the "
            f"replay stayed within that ({100 * row['worst_fall_all']:.0f}%, measured with the diversifiers). "
            f"Diversifiers: the rest of the capital, {_p(d['sleeve'])}.")


# ------------------------------------------------------------------ run
def build() -> dict:
    x, cash, _, _ = A.load_inputs()
    origin = x.index[-1]
    S = np.expm1(x)
    est = A.estimates(S, len(x) - 1, A.forecast_cov([origin]).get(origin))
    if "fcst" not in est:
        raise ValueError(f"no covariance forecast at {origin.date()}")
    cov, ws = est["fcst"][1].loc[ASSETS, ASSETS], A.sleeve_weights(SLEEVE_RULE, est)
    exp = expected_four(diversifiers.load_inputs()[1]["mf_funds"], origin)
    inp = frontier.Inputs(cov, exp, {})      # used for its arithmetic (stats); nothing is optimized
    veh = vehicles()
    cv = curve(inp, ws, S[STOCKS], sleeve_replay(x), cash)

    keep = ["vol", "growth", "growth_lo", "growth_hi", *[f"w_{a}" for a in ASSETS],
            *[c for c in cv.columns if c.endswith(("_all", "_live"))]]
    answers, funds = [], []
    for goal in GOALS:
        d = decide(goal, cv)
        row = {"goal": goal, "loss_limit": GOALS[goal]["loss_limit"], **d}
        if d["cash"] < 1.0:
            row.update(cv[cv["i"] == d["i"]].iloc[0][keep].to_dict())
        else:
            row.update(vol=0.0, growth=0.0, growth_lo=0.0, growth_hi=0.0, **{f"w_{a}": 0.0 for a in ASSETS})
        plan = fund_plan(d, ws, veh)
        row["fund_fees"] = _fees(plan)
        if d["cash"] < 1.0:                # the replay charges no fees: take them off
            row["growth_all_net"] = row["growth_all"] - row["fund_fees"]
        row.update(headline=headline(d), reason=reason(goal, d, row))
        answers.append(row)
        funds += [{"goal": goal, **r} for r in plan]

    meta = {"label": LABEL, "origin": str(origin.date()), "returns_through": str(origin.date()),
            "first_holding_month": str(x.index[A.BURN].date()), "live_from": A.LIVE, "assets": ASSETS, "names": NAMES,
            "accounts": ACCOUNTS, "goals": GOALS, "notes": NOTES,
            "settings": {"sleeve_rule": SLEEVE_RULE, "sleeve_rule_name": A.RULE_NAMES[SLEEVE_RULE], "step": STEP,
                         "recent": A.RECENT},
            "sleeve_weights": {a: float(ws[a]) for a in SLEEVE},
            "least_risk_stocks": float(cv.loc[cv["vol"].idxmin(), "stocks"]),
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
    print(res["answers"][["goal", "stocks", "sleeve", "cash", "vol", "worst_fall_all", "growth"]].round(3).to_string(index=False))
    if not _DEV:
        from . import questionnaire_page
        print(questionnaire_page.write_report(), questionnaire_page.write_page())
    return 0


if __name__ == "__main__":
    sys.exit(main())
