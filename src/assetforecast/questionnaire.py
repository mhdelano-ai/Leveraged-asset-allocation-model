"""The allocation questionnaire: an account's tax treatment and goal -> stocks, then the diversifier sleeve.

    python -m assetforecast.questionnaire

Writes ``results/questionnaire/`` (``answers.csv``, ``curve.csv``, ``funds.csv``,
``inputs.csv``, ``cov.csv``, ``meta.json``), ``reports/questionnaire.md`` and
``dashboard/questionnaire.html`` (``questionnaire_page``). ARITHMETIC UNDER DECLARED
RULES (``LABEL``): it reads the diversifier and allocation studies and the published
forecast, fits nothing, optimizes nothing, selects no model and changes nothing frozen
or published. The rules are judgements, written down here before the tables were
computed; they are not advice and have no out-of-sample evidence.

Risk decides, returns do not
----------------------------
The allocation study's reading: risk carries over from one period to the next and can
be forecast; returns cannot, and rules that optimized on them did not beat fixed
weights. So nothing here is chosen on an expected return. Every answer is a point on
one curve of mixes (``curve.csv``), and the goal's tolerance for loss picks the point.
Expected growth is shown beside each point with its interval, and is not an input.

The two questions
-----------------
``ACCOUNTS``  the account's tax treatment. It decides how exposure above 100% of
              capital can be held (``financing``): a margin loan in a taxable account
              (the published frontier's loan, ``frontier.borrow_terms``); in a
              retirement account, where no loan is allowed, only through funds that
              carry it inside (``stacked``, costed at the T-bill rate; their fees are
              shown, not charged). It also decides the tax notes shown. Taxes
              themselves are not modelled.
``GOALS``     what the money is for. Each goal declares ``loss_limit``: the largest
              fall from a previous high the goal can sit through. ``near`` (spent
              within 3 years) is outside the model and is answered with T-bills.

The curve
---------
``unlevered``  world stocks at every share from 0 to 100% in steps of ``STEP``, the
               rest in the sleeve the diversifier study left: intermediate Treasuries,
               gold and managed futures, weighted by ``SLEEVE_RULE`` (1/volatility on
               the whole past, the allocation study's reading for the weights inside
               the sleeve) at the last origin.
``margin``,    100% stocks with the sleeve added on top, from ``STEP`` of capital up to
``stacked``    what that financing can carry (:func:`overlay_caps`): a margin loan up
               to ``frontier.MAX_LEVERAGE`` times capital; stacked funds until the
               funds of the lookup use all the capital.

Per point: ``vol``, its volatility on the allocation study's ``fcst`` matrix at the
last origin (the package's 12-month forecast for stocks, Treasuries and gold; managed
futures, which the forecast does not cover, at their last-``RECENT``-months volatility
and correlations); the replay (below); and, for display only, ``growth`` with its band
(``frontier.stats``: the expected log return over T-bills of the rebalanced mix, net of
the borrowing spread, and the 90% intervals of the means added as if perfectly
correlated). The means are each holding's historical mean (``returns.expected``):
stocks from ``threefund.expected``, Treasuries and gold from the published forecast,
managed futures from the live funds only (2007 on, after fees), the window the
diversifier study says to trust.

The rule
--------
1. Stock share: the largest on the ``unlevered`` part whose worst fall in the whole
   replay is no deeper than the goal's ``loss_limit``.
2. Backfill: the rest of the capital is the sleeve. That is the point on the curve.
3. Nothing left to backfill: at 100% stocks the sleeve goes on top instead,
   ``OVERLAY`` of capital (the diversifier study's overlay), on the account's part of
   the curve, if that point also stays within the limit. ``OVERLAY`` is a declared
   amount: the loss limit alone does not pin it down.

The replay is the allocation study's: the sleeve's walk-forward return (weights set
at each month-end from returns through it, held over the next month, from ``BURN``
months after the common start), beside stocks, rebalanced monthly, borrowing at
T-bills plus the spread. Same series, same statistics (``diversifiers._path_stats``);
``menu.csv`` of the allocation study is reproduced at its mixes.

Funds
-----
``funds.csv``: one way to hold each answer with the funds of
``data/processed/diversifier_vehicles.json`` (a hand-recorded fee lookup). Outright:
the cheapest recorded fund per exposure. Stacked: per exposure the fund carrying the
most of it per dollar of stocks (ties: the cheaper), sized to the sleeve weight; what
capital is left goes to a plain stock fund. No stock fund's fee is in the lookup.
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
    "taxable": {"name": "Taxable brokerage", "detail": "Gains, interest and distributions are taxed as they come.",
                "financing": "margin"},
    "traditional": {"name": "Tax-deferred", "detail": "Traditional IRA or 401(k): taxed as income on withdrawal.",
                    "financing": "stacked"},
    "roth": {"name": "Roth", "detail": "Roth IRA or Roth 401(k): qualified withdrawals are untaxed.",
             "financing": "stacked"},
}
FINANCING = {"margin": "a margin loan", "stacked": "funds that carry the exposure inside"}
GOALS = {   # loss_limit: the largest fall from a previous high the goal can sit through (None: outside the model)
    "near": {"name": "Spending within 3 years", "detail": "A purchase, a reserve.", "loss_limit": None},
    "medium": {"name": "Spending in 3 to 10 years", "detail": "A house, tuition.", "loss_limit": 0.20},
    "income": {"name": "Retirement income, drawn now", "detail": "Withdrawals have begun.",
               "loss_limit": threefund.LOSS_LIMITS[0]},
    "growth": {"name": "Growth, untouched for 10 years or more", "detail": "Retirement that is still far off.",
               "loss_limit": threefund.LOSS_LIMITS[1]},
}
NOTES = {   # shown with the answer; statements about tax treatment, not modelled
    "taxable": [
        "A margin loan is available in this account, so the sleeve can sit on top of a full stock position. Funds "
        "that carry the exposure inside are available here too.",
        "Taxes are not modelled. Stock index funds defer most of their gain until sold. The sleeve is taxed harder: "
        "Treasury interest is ordinary income (free of state tax), funds holding gold bullion are taxed as "
        "collectibles, and managed-futures funds pay out most of their return as taxable distributions.",
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
SLEEVE_RULE = "invvol_hist"          # the weights inside the sleeve
STEP = 0.05                          # steps of the stock share, and of the sleeve on top
OVERLAY = diversifiers.OVERLAY       # sleeve added on top of 100% stocks, share of capital
EXPOSURE = {"ust_int": "intermediate Treasuries", "gold": "gold", "mf": "managed futures"}   # names in the fund lookup
STOCK_FUND = "A world stock fund"
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


def mix_stats(stocks: pd.Series, sleeve: pd.Series, cash: pd.Series, a: float, b: float, spread: float) -> dict:
    """Path statistics of ``a`` stocks + ``b`` sleeve (simple excess returns), per period of the allocation study."""
    r = a * stocks.reindex(sleeve.index) + b * sleeve - spread / 12 * max(a + b - 1.0, 0.0)
    row = {}
    for p, start in A.PERIODS.items():
        rp = r.loc[start:] if start else r
        rf = cash.reindex(rp.index)
        row.update({f"{k}_{p}": v for k, v in diversifiers._path_stats((1 + rp) * (1 + rf) - 1, rf).items()})
    return row


# ------------------------------------------------------------------ the curve
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


def curve(inp: frontier.Inputs, ws: pd.Series, stocks: pd.Series, sleeve: pd.Series, cash: pd.Series,
          spreads: dict, caps: dict) -> pd.DataFrame:
    """Every mix an answer can be (module docstring, The curve), with its forecast risk, its replay and its growth."""
    def row(segment, a, b, sp):
        w = weights(a, b, ws)
        fs = frontier.stats(inp, w, sp)
        return {"segment": segment, "stocks": float(a), "sleeve": float(b), "borrowed": float(max(a + b - 1.0, 0.0)),
                "spread": sp if a + b > 1 else 0.0, "vol": fs["vol"], "growth": fs["growth"],
                "growth_lo": fs["growth_lo"], "growth_hi": fs["growth_hi"],
                **{f"w_{k}": float(v) for k, v in zip(ASSETS, w)}, **mix_stats(stocks, sleeve, cash, a, b, sp)}

    rows = [row("unlevered", a, 1.0 - a, 0.0) for a in _grid(0.0, 1.0)]
    for fin, sp in spreads.items():
        rows += [row(fin, 1.0, b, sp) for b in _grid(STEP, caps[fin])]
    return pd.DataFrame(rows)


def stock_share(unlevered: pd.DataFrame, limit: float) -> float | None:
    """The largest stock share of the unlevered curve whose worst fall is no deeper than ``limit``; None if none is."""
    ok = unlevered[unlevered["worst_fall_all"] >= -limit - 1e-12]
    return None if ok.empty else float(ok["stocks"].max())


def decide(account: str, goal: str, cv: pd.DataFrame) -> dict:
    """The point of the curve ``cv`` for one pair of answers (the rule of the module docstring)."""
    fin, limit = ACCOUNTS[account]["financing"], GOALS[goal]["loss_limit"]
    a = None if limit is None else stock_share(cv[cv["segment"] == "unlevered"], limit)
    if a is None:
        return {"stocks": 0.0, "sleeve": 0.0, "cash": 1.0, "borrowed": 0.0, "financing": "none"}
    if a < 1.0:
        return {"stocks": a, "sleeve": 1.0 - a, "cash": 0.0, "borrowed": 0.0, "financing": "none"}
    top = cv[(cv["segment"] == fin) & np.isclose(cv["sleeve"], OVERLAY)]
    if len(top) and top["worst_fall_all"].iloc[0] >= -limit:
        return {"stocks": 1.0, "sleeve": OVERLAY, "cash": 0.0, "borrowed": OVERLAY, "financing": fin}
    return {"stocks": 1.0, "sleeve": 0.0, "cash": 0.0, "borrowed": 0.0, "financing": "none"}


def on_curve(cv: pd.DataFrame, d: dict) -> pd.Series:
    """The row of the curve a decision is."""
    seg = d["financing"] if d["borrowed"] > 0 else "unlevered"
    return cv[(cv["segment"] == seg) & np.isclose(cv["stocks"], d["stocks"]) & np.isclose(cv["sleeve"], d["sleeve"])].iloc[0]


# ------------------------------------------------------------------ funds
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


def overlay_caps(ws: pd.Series, veh: dict) -> dict:
    """How much sleeve each financing can put on top of 100% stocks, on the ``STEP`` grid.

    A margin loan: up to ``frontier.MAX_LEVERAGE`` times capital. Stacked funds: until the
    funds that carry the sleeve use all the capital (no loan can buy more of them)."""
    top = frontier.MAX_LEVERAGE - 1.0
    use = sum(float(ws[a]) / _fund(veh, a, True)["exposure_per_dollar"] for a in SLEEVE)   # capital per unit of sleeve
    down = lambda v: float(np.floor(v / STEP + 1e-9) * STEP)                                # noqa: E731
    return {"margin": down(top), "stacked": down(min(top, 1.0 / use))}


def fund_plan(d: dict, ws: pd.Series, veh: dict, stacked: bool) -> list[dict]:
    """Funds and their shares of capital for a decision ``d`` (module docstring, Funds).

    Columns ``stocks`` and ``exposure`` are what each line carries, as shares of capital."""
    if d["cash"] >= 1.0:
        return [{"symbol": "", "name": "T-bills or a money-market fund", "role": "cash", "weight": 1.0,
                 "stocks": 0.0, "exposure": 0.0, "fee": np.nan, "note": "outside the model"}]
    rows = []
    for a in SLEEVE:
        w = d["sleeve"] * float(ws[a])
        if w <= 0:
            continue
        r = _fund(veh, a, stacked)
        if stacked:
            k = w / r["exposure_per_dollar"]
            rows.append({"symbol": r["symbol"], "name": r["name"], "role": a, "weight": k,
                         "stocks": k * r["stocks_per_dollar"], "exposure": w, "fee": r["fee"], "note": r["holds"]})
        else:
            rows.append({"symbol": r["symbol"], "name": r["name"], "role": a, "weight": w, "stocks": 0.0,
                         "exposure": w, "fee": r["fee"], "note": r["note"]})
    used = sum(r["weight"] for r in rows) if stacked else 0.0
    plain = max(1.0 - used, 0.0) if stacked else d["stocks"]
    if used > 1.0 + 1e-9:
        raise ValueError("the stacked funds need more than the capital there is")
    if plain > 0:
        rows.insert(0, {"symbol": "", "name": STOCK_FUND, "role": STOCKS, "weight": plain, "stocks": plain,
                        "exposure": 0.0, "fee": np.nan, "note": "no stock fund's fee is in the lookup"})
    return rows


def _fees(rows: list[dict]) -> float:
    return float(sum(r["weight"] * r["fee"] for r in rows if np.isfinite(r["fee"])))


# ------------------------------------------------------------------ wording
def _p(v: float) -> str:
    return f"{100 * v:.0f}%"


def headline(d: dict) -> str:
    if d["cash"] >= 1.0:
        return "T-bills"
    if d["borrowed"] > 0:
        how = "on a margin loan" if d["financing"] == "margin" else "carried inside stacked funds"
        return f"{_p(d['stocks'])} stocks, plus {_p(d['sleeve'])} diversifiers {how}"
    if d["sleeve"] <= 0:
        return f"{_p(d['stocks'])} stocks"
    return f"{_p(d['stocks'])} stocks, {_p(d['sleeve'])} diversifiers"


def reason(goal: str, d: dict, row: dict) -> str:
    limit = GOALS[goal]["loss_limit"]
    if limit is None:
        return ("Money needed within 3 years is outside this model. Every mix of stocks and the sleeve has fallen by "
                "more than such a goal can make up in time.")
    s = (f"The goal can sit through a fall of {_p(limit)}. {_p(d['stocks'])} is the largest stock share whose worst fall "
         f"in the replay stayed within that ({100 * row['worst_fall_unlevered']:.1f}%, with the rest in the sleeve).")
    if d["borrowed"] > 0:
        s += (f" At 100% stocks nothing is left to backfill, so the sleeve goes on top, {_p(d['sleeve'])} of capital; "
              f"that position's worst fall was {100 * row['worst_fall_all']:.0f}%.")
    return s


# ------------------------------------------------------------------ run
def build() -> dict:
    x, cash, _, _ = A.load_inputs()
    origin = x.index[-1]
    S = np.expm1(x)
    est = A.estimates(S, len(x) - 1, A.forecast_cov([origin]).get(origin))
    if "fcst" not in est:
        raise ValueError(f"no covariance forecast at {origin.date()}")
    cov, ws = est["fcst"][1].loc[ASSETS, ASSETS], A.sleeve_weights(SLEEVE_RULE, est)
    borrow = frontier.borrow_terms(origin)
    spreads = {"margin": float(borrow["spread"]), "stacked": 0.0}
    exp = expected_four(diversifiers.load_inputs()[1]["mf_funds"], origin)
    inp = frontier.Inputs(cov, exp, {}, spreads["margin"])      # used for its arithmetic (stats); nothing is optimized
    veh = vehicles()
    caps = overlay_caps(ws, veh)
    cv = curve(inp, ws, S[STOCKS], sleeve_replay(x), cash, spreads, caps)
    unl = cv[cv["segment"] == "unlevered"]

    keep = ["spread", "vol", "growth", "growth_lo", "growth_hi", *[f"w_{a}" for a in ASSETS],
            *[c for c in cv.columns if c.endswith(("_all", "_live"))]]
    answers, funds = [], []
    for acc in ACCOUNTS:
        for goal in GOALS:
            d = decide(acc, goal, cv)
            row = {"account": acc, "goal": goal, "loss_limit": GOALS[goal]["loss_limit"], **d}
            if d["cash"] < 1.0:
                row.update(on_curve(cv, d)[keep].to_dict())
                row["worst_fall_unlevered"] = float(unl.loc[np.isclose(unl["stocks"], d["stocks"]), "worst_fall_all"].iloc[0])
            else:
                row.update(spread=0.0, vol=0.0, growth=0.0, growth_lo=0.0, growth_hi=0.0, **{f"w_{a}": 0.0 for a in ASSETS})
            plan = fund_plan(d, ws, veh, stacked=d["financing"] == "stacked")
            row.update(fund_fees=_fees(plan), loan_cost=row["spread"] * d["borrowed"], stocks_held=sum(r["stocks"] for r in plan))
            if d["financing"] == "margin":      # the other way to hold the same position in this account
                row["stacked_fees"] = _fees(fund_plan(d, ws, veh, stacked=True))
            row.update(headline=headline(d), reason=reason(goal, d, row))
            answers.append(row)
            funds += [{"account": acc, "goal": goal, **r} for r in plan]

    meta = {"label": LABEL, "origin": str(origin.date()), "returns_through": str(origin.date()),
            "first_holding_month": str(x.index[A.BURN].date()), "live_from": A.LIVE, "assets": ASSETS, "names": NAMES,
            "accounts": ACCOUNTS, "goals": GOALS, "notes": NOTES, "financing": FINANCING,
            "settings": {"sleeve_rule": SLEEVE_RULE, "sleeve_rule_name": A.RULE_NAMES[SLEEVE_RULE], "step": STEP,
                         "overlay": OVERLAY, "overlay_caps": caps, "max_leverage": frontier.MAX_LEVERAGE,
                         "spreads": spreads, "recent": A.RECENT},
            "sleeve_weights": {a: float(ws[a]) for a in SLEEVE}, "borrow": borrow,
            "least_risk_stocks": float(unl.loc[unl["vol"].idxmin(), "stocks"]),
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
    print(res["answers"][["account", "goal", "stocks", "sleeve", "borrowed", "vol", "growth", "worst_fall_all"]]
          .round(3).to_string(index=False))
    if not _DEV:
        from . import questionnaire_page
        print(questionnaire_page.write_report(), questionnaire_page.write_page())
    return 0


if __name__ == "__main__":
    sys.exit(main())
