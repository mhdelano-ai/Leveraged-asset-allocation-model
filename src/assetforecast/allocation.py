"""The allocation study: what should set the weights of stocks and the diversifier sleeve?

    python -m assetforecast.allocation

Writes ``results/allocation/`` (``sleeve_rules.csv``, ``portfolio_rules.csv``,
``risk_accuracy.csv``, ``persistence.csv``, ``means.csv``, ``today.csv``,
``menu.csv``, ``weights.csv``, ``paths.csv``, ``meta.json``), ``reports/allocation.md`` and
``dashboard/allocation.html`` (``allocation_page``). A DESCRIPTIVE study
(``LABEL``): it replays rules on a history that was all seen before the rules
were written, selects no model and changes nothing frozen or published.

The question
------------
The diversifier study left four holdings: world stocks (``threefund``'s
composite) and a sleeve of intermediate Treasuries, gold and managed futures.
Weights can be set from three kinds of input:

``none``    fixed weights, no estimate at all (a third each; 60/40)
``hist``    everything measured on the whole past to date (expanding window)
``recent``  everything measured on the last ``RECENT`` months
``fcst``    the package's covariance forecast (``covhistory``: ``combo_cov_main``
            bridged to 12 months), aggregated to the composite. It does not cover
            managed futures: their volatility and their correlations with the
            other three are the ``recent`` estimate.

For expected returns the package's own forecast IS the expanding historical
mean (``returns.expected``; nothing tried beat it), so ``hist`` means are "the
forecast" of returns and ``recent`` means are the return-chasing alternative.

Every rule below was fixed before any of them was run. At each month-end
origin t from ``BURN`` months after the common start, a rule sees returns
through t only and the covariance forecast made at t; its weights are held
over month t+1.

Sleeve rules (three assets, long-only, sum to 1)
------------------------------------------------
``equal``                      a third each
``invvol_{hist,recent,fcst}``  proportional to 1/volatility
``minvar_{hist,recent,fcst}``  least variance
``maxsharpe_hist``             most mean/volatility, means and covariance both ``hist``
``maxsharpe_recent``           the same on ``recent``
``maxsharpe_mix``              ``hist`` means, ``fcst`` covariance

Portfolio rules (stocks and the three, plus cash)
-------------------------------------------------
``fixed``                      ``BASE_STOCKS`` stocks, the rest the equal sleeve; never changed
``target_{hist,recent,fcst}``  the same proportions, scaled each month to ``TARGET_VOL``
                               on that covariance (exposure capped at 100% of capital)
``riskonly_fcst``              1/volatility over all four on ``fcst``, scaled to ``TARGET_VOL``
``opt_hist``, ``opt_recent``   most mean/volatility over all four, scaled to ``TARGET_VOL``
``opt_mix``                    ``hist`` means, ``fcst`` covariance

Nothing borrows: a scaled rule never holds more than 100% of capital, and
below 100% the rest is in T-bills. The scaled rules were first run with the
cap at twice capital, borrowing at a margin loan's spread; when the package
dropped leverage they were rerun once with the cap at 100%, and nothing else
was changed. Trading costs are not charged; ``turnover``
(one-way, a year, from the change in target weights) shows what they would
scale with. ``sharpe_diff`` and its 90% interval compare each rule with the
no-estimate rule of its table by a paired circular block bootstrap
(``BOOT_BLOCK`` months, ``BOOT_N`` draws, fixed seed).

What the inputs are worth
-------------------------
``risk_accuracy.csv``  each covariance source's forecast at t of the volatility
                       of each holding and of the fixed mix, and of the
                       correlations among stocks, Treasuries and gold, against
                       what the next 12 months realized (``qlike`` for
                       volatility, mean squared error for correlation; lower is
                       better). ``fcst21`` is the forecast's 21-day matrix: a
                       check added after the first run showed the 12-month
                       matrix standing above month-to-month volatility; no rule
                       uses it.
``menu.csv``           stocks beside the sleeve at the mixes of ``MENU``, for the
                       sleeves of ``MENU_SLEEVES``: today's
                       weights, forecast volatility, and the replay.
``persistence.csv``    over each holding's own full history: the correlation
                       between its volatility in one ``PERSIST``-month block and
                       in the next, and the same for its mean return and its
                       return per unit of risk. Non-overlapping blocks.
``means.csv``          each holding's historical mean with the package's 90%
                       interval (``returns.expected``), and its return per unit
                       of risk in the two halves of the common window.

Managed futures are one spliced series (``mf``): the live funds from 2007-03;
before that the paper trend factor scaled to the funds' volatility, less the
measured gap (``diversifiers.measured_drag``). The scale and the gap are
measured on 2007-2026, so every rule that reads managed futures' pre-2007
mean or volatility reads a number nobody had at the time.
"""
from __future__ import annotations

import itertools
import json
import os
import sys

import numpy as np
import pandas as pd

from . import covhistory, diversifiers, paths, threefund
from .returns.expected import expected_from_xr

LABEL = ("Descriptive allocation study; it replays weighting rules on a history that was all seen "
         "before the rules were written, selects nothing and is not out-of-sample evidence")
STOCKS = "world_stocks"
SLEEVE = ["ust_int", "gold", "mf"]
ASSETS = [STOCKS, *SLEEVE]
NAMES = {STOCKS: "World stocks", "ust_int": "Intermediate Treasuries", "gold": "Gold", "mf": "Managed futures"}
START = "1985-01-31"     # first month all four exist
LIVE = "2007-03-31"      # first month of the live managed-futures funds
BURN = 60                # months of history before the first weights are set
RECENT = 60              # months in the "recent" window
PERSIST = 60             # block length of the persistence table
TARGET_VOL = 0.10        # annual volatility the scaled portfolio rules aim at
BASE_STOCKS = 0.60       # stock share of the fixed portfolio
BOOT_N, BOOT_BLOCK, BOOT_SEED = 2000, 12, 20261116
SOURCES = ("hist", "recent", "fcst")
SLEEVE_RULES = ["equal", *[f"{k}_{s}" for k in ("invvol", "minvar") for s in SOURCES],
                "maxsharpe_hist", "maxsharpe_recent", "maxsharpe_mix"]
PORTFOLIO_RULES = ["fixed", "target_hist", "target_recent", "target_fcst", "riskonly_fcst",
                   "opt_hist", "opt_recent", "opt_mix"]
RULE_NAMES = {
    "equal": "A third each", "invvol_hist": "1/volatility, whole past", "invvol_recent": "1/volatility, last 5 years",
    "invvol_fcst": "1/volatility, forecast", "minvar_hist": "Least variance, whole past",
    "minvar_recent": "Least variance, last 5 years", "minvar_fcst": "Least variance, forecast",
    "maxsharpe_hist": "Optimized: past returns, past risk", "maxsharpe_recent": "Optimized: last 5 years of both",
    "maxsharpe_mix": "Optimized: past returns, forecast risk",
    "fixed": "Fixed 60% stocks / 40% sleeve", "target_hist": "60/40 scaled to 10% risk, whole past",
    "target_recent": "60/40 scaled to 10% risk, last 5 years", "target_fcst": "60/40 scaled to 10% risk, forecast",
    "riskonly_fcst": "1/volatility over all four, forecast", "opt_hist": "Optimized: past returns, past risk",
    "opt_recent": "Optimized: last 5 years of both", "opt_mix": "Optimized: past returns, forecast risk"}
RULE_INPUTS = {   # (what sets the return input, what sets the risk input)
    "equal": ("none", "none"), "fixed": ("none", "none"), "riskonly_fcst": ("none", "fcst"),
    **{f"{k}_{s}": ("none", s) for k in ("invvol", "minvar", "target") for s in SOURCES},
    "maxsharpe_hist": ("hist", "hist"), "maxsharpe_recent": ("recent", "recent"), "maxsharpe_mix": ("hist", "fcst"),
    "opt_hist": ("hist", "hist"), "opt_recent": ("recent", "recent"), "opt_mix": ("hist", "fcst")}
PERIODS = {"all": None, "live": LIVE}
MENU = ((0.4, 0.6), (0.6, 0.4), (0.8, 0.2), (1.0, 0.0))   # (stocks, sleeve) shares of capital
MENU_SLEEVES = ("invvol_hist", "equal")

_DEV = bool(os.environ.get("ASSETFORECAST_MAX_DATE"))
OUT = paths.ROOT / "results" / ("dev/allocation" if _DEV else "allocation")
REPORT = paths.REPORTS / ("allocation_dev.md" if _DEV else "allocation.md")


# ------------------------------------------------------------------ inputs
def spliced_mf(tsmom: pd.Series, funds: pd.Series, drag: dict) -> pd.Series:
    """Live funds where they exist; before, the paper factor at the funds' volatility less the gap."""
    pre = drag["vol_ratio_funds_to_factor"] * tsmom - drag["gap_at_fund_vol"] / 12
    return funds.combine_first(pre)


def load_inputs() -> tuple[pd.DataFrame, pd.Series, dict, dict]:
    """(log excess returns of ``ASSETS``, simple T-bill return, drag, each holding's full history)."""
    stocks, cands, cash, drag = diversifiers.load_inputs()
    mf = spliced_mf(cands["mf_tsmom"], cands["mf_funds"], drag)
    full = {STOCKS: stocks.dropna(), "ust_int": cands["ust_int"].dropna(), "gold": cands["gold"].dropna(),
            "mf": mf.dropna()}
    x = pd.concat(full, axis=1, sort=True)[ASSETS].loc[START:].dropna()
    return x, cash, drag, full


def forecast_cov(origins, hist: dict | None = None) -> dict:
    """{origin: 3x3 12-month covariance of stocks, Treasuries and gold} from the package's forecast.

    The stock composite's weights are renormalized over the equity assets the
    forecast holds at that origin (``threefund``'s convention)."""
    hist = covhistory.load("annual") if hist is None else hist
    ws = threefund.COMPOSITES[STOCKS]
    out = {}
    for o in origins:
        C = hist.get(pd.Timestamp(o))
        if C is None or not {"ust_int", "gold"} <= set(C.index):
            continue
        have = [a for a in ws if a in C.index]
        W = pd.DataFrame(0.0, index=[STOCKS, "ust_int", "gold"], columns=C.index)
        W.loc[STOCKS, have] = np.array([ws[a] for a in have]) / sum(ws[a] for a in have)
        W.loc["ust_int", "ust_int"] = W.loc["gold", "gold"] = 1.0
        out[pd.Timestamp(o)] = pd.DataFrame(W.to_numpy() @ C.to_numpy() @ W.to_numpy().T, index=W.index, columns=W.index)
    return out


def estimates(S: pd.DataFrame, i: int, fc: pd.DataFrame | None) -> dict:
    """{source: (annual mean, annual covariance)} from simple excess returns ``S[:i+1]`` and the forecast at i.

    ``fcst`` has no mean. Its managed-futures row is the ``recent`` volatility
    and correlations, the latter applied to the forecast volatilities."""
    a, r = S.iloc[:i + 1], S.iloc[max(0, i + 1 - RECENT):i + 1]
    est = {"hist": (12 * a.mean(), 12 * a.cov()), "recent": (12 * r.mean(), 12 * r.cov())}
    if fc is not None:
        rc = est["recent"][1]
        sd = np.sqrt(np.diag(rc))
        corr = pd.DataFrame(rc.to_numpy() / np.outer(sd, sd), index=rc.index, columns=rc.columns)
        f3 = fc.loc[fc.index, fc.index]
        s3 = np.sqrt(np.diag(f3))
        corr.loc[f3.index, f3.index] = f3.to_numpy() / np.outer(s3, s3)
        vol = pd.Series(sd, index=rc.index)
        vol[f3.index] = s3
        est["fcst"] = (None, _psd(corr.mul(vol, axis=0).mul(vol, axis=1)))
    return est


def _psd(C: pd.DataFrame) -> pd.DataFrame:
    """``C`` with any negative eigenvalue raised to a small positive one (a mixed-source matrix can have one)."""
    w, v = np.linalg.eigh((C.to_numpy() + C.to_numpy().T) / 2)
    if w.min() > 1e-10:
        return C
    return pd.DataFrame((v * np.maximum(w, 1e-8)) @ v.T, index=C.index, columns=C.columns)


# ------------------------------------------------------------------ weights
def inv_vol(cov: pd.DataFrame) -> pd.Series:
    w = 1.0 / np.sqrt(pd.Series(np.diag(cov), index=cov.index))
    return w / w.sum()


def _best_subset(cov: pd.DataFrame, direction) -> pd.Series | None:
    """Long-only weights summing to 1: over every subset, solve ``w ~ inv(cov) direction`` and keep the
    all-positive solution that scores best (exact for least variance and for most mean/volatility)."""
    names, best, best_score = list(cov.index), None, -np.inf
    d = pd.Series(direction, index=names)
    for k in range(1, len(names) + 1):
        for sub in itertools.combinations(names, k):
            sub = list(sub)
            C = cov.loc[sub, sub].to_numpy()
            raw = np.linalg.solve(C, d[sub].to_numpy())
            if raw.sum() <= 0 or (raw <= 0).any():
                continue
            w = raw / raw.sum()
            score = float(w @ d[sub].to_numpy() / np.sqrt(w @ C @ w))
            if score > best_score:
                best, best_score = pd.Series(0.0, index=names).add(pd.Series(w, index=sub), fill_value=0.0), score
    return best


def min_var(cov: pd.DataFrame) -> pd.Series:
    return _best_subset(cov, np.ones(len(cov)))[cov.index]


def max_sharpe(mean: pd.Series, cov: pd.DataFrame) -> pd.Series:
    """Most mean/volatility, long-only; least variance when no mean is positive."""
    w = _best_subset(cov, mean[cov.index].to_numpy()) if (mean > 0).any() else None
    return (min_var(cov) if w is None else w)[cov.index]


def sleeve_weights(rule: str, est: dict) -> pd.Series:
    kind, _, src = rule.partition("_")
    sub = lambda s: est[s][1].loc[SLEEVE, SLEEVE]                      # noqa: E731
    if rule == "equal":
        return pd.Series(1.0 / len(SLEEVE), index=SLEEVE)
    if kind == "invvol":
        return inv_vol(sub(src))
    if kind == "minvar":
        return min_var(sub(src))
    if kind == "maxsharpe":
        m, c = ("hist", "fcst") if src == "mix" else (src, src)
        return max_sharpe(est[m][0][SLEEVE], sub(c))
    raise KeyError(rule)


def _scaled(w: pd.Series, cov: pd.DataFrame, target: float = TARGET_VOL) -> pd.Series:
    """``w`` (summing to 1) scaled to ``target`` volatility on ``cov``, never above 100% of capital."""
    vol = float(np.sqrt(w @ cov.loc[w.index, w.index] @ w))
    return w * min(target / vol, 1.0)


def base_weights() -> pd.Series:
    w = pd.Series((1 - BASE_STOCKS) / len(SLEEVE), index=ASSETS)
    w[STOCKS] = BASE_STOCKS
    return w


def portfolio_weights(rule: str, est: dict) -> pd.Series:
    """Shares of capital in ``ASSETS``, summing to at most 1; the rest is T-bills."""
    kind, _, src = rule.partition("_")
    if rule == "fixed":
        return base_weights()
    if kind == "target":
        return _scaled(base_weights(), est[src][1])
    if rule == "riskonly_fcst":
        return _scaled(inv_vol(est["fcst"][1].loc[ASSETS, ASSETS]), est["fcst"][1])
    if kind == "opt":
        m, c = ("hist", "fcst") if src == "mix" else (src, src)
        return _scaled(max_sharpe(est[m][0][ASSETS], est[c][1].loc[ASSETS, ASSETS]), est[c][1])
    raise KeyError(rule)


# ------------------------------------------------------------------ walk-forward
def walk_forward(x: pd.DataFrame, fc: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(monthly simple excess return of every rule, the weights it held), by holding month.

    ``weights`` rows: rule, date (the holding month), one column per asset."""
    S = np.expm1(x)
    rets, wrows = {r: {} for r in SLEEVE_RULES + PORTFOLIO_RULES}, []
    for i in range(BURN - 1, len(x) - 1):
        t, nxt = x.index[i], x.index[i + 1]
        est = estimates(S, i, fc.get(t))
        if "fcst" not in est:
            raise ValueError(f"no covariance forecast at {t.date()}")
        for r in SLEEVE_RULES:
            w = sleeve_weights(r, est)
            rets[r][nxt] = float(w @ S.loc[nxt, SLEEVE])
            wrows.append({"rule": r, "date": nxt, **w.reindex(ASSETS).to_dict()})
        for r in PORTFOLIO_RULES:
            w = portfolio_weights(r, est)
            rets[r][nxt] = float(w @ S.loc[nxt, ASSETS])
            wrows.append({"rule": r, "date": nxt, **w.to_dict()})
    return pd.DataFrame(rets), pd.DataFrame(wrows)


def _sharpe(lx: np.ndarray) -> float:
    return float(np.sqrt(12) * lx.mean() / lx.std(ddof=1))


def sharpe_diff(a: pd.Series, b: pd.Series, n: int = BOOT_N, block: int = BOOT_BLOCK, seed: int = BOOT_SEED) -> dict:
    """Sharpe of ``a`` less Sharpe of ``b`` (log excess returns) with a paired circular block bootstrap interval."""
    A, B = a.to_numpy(), b.to_numpy()
    T = len(A)
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, T, size=(n, -(-T // block)))
    idx = ((starts[:, :, None] + np.arange(block)[None, None, :]) % T).reshape(n, -1)[:, :T]
    d = np.array([_sharpe(A[j]) - _sharpe(B[j]) for j in idx])
    return {"sharpe_diff": _sharpe(A) - _sharpe(B), "diff_lo": float(np.quantile(d, 0.05)),
            "diff_hi": float(np.quantile(d, 0.95))}


def rule_table(rets: pd.DataFrame, weights: pd.DataFrame, cash: pd.Series, rules: list[str], base: str) -> pd.DataFrame:
    """One row per rule and period: the path statistics, what it held, and its Sharpe against ``base``."""
    rows = []
    for p, start in PERIODS.items():
        R = rets.loc[start:] if start else rets
        rf = cash.reindex(R.index)
        lx = np.log1p(R)
        for r in rules:
            w = weights[weights["rule"] == r].set_index("date")[ASSETS].loc[R.index[0]:].fillna(0.0)
            st = diversifiers._path_stats((1 + R[r]) * (1 + rf) - 1, rf)
            v12 = lx[r].rolling(12).std() * np.sqrt(12)
            row = {"period": p, "start": str(R.index[0].date()), "end": str(R.index[-1].date()), "n": len(R), "rule": r,
                   "name": RULE_NAMES[r], "return_input": RULE_INPUTS[r][0], "risk_input": RULE_INPUTS[r][1], **st,
                   "worst_12m": float(np.expm1((lx[r] + np.log1p(rf)).rolling(12).sum().min())),
                   "vol_12m_lo": float(v12.min()), "vol_12m_hi": float(v12.max()),
                   "turnover": float(12 * w.diff().abs().sum(axis=1).iloc[1:].mean() / 2),
                   "exposure": float(w.sum(axis=1).mean()), "exposure_max": float(w.sum(axis=1).max()),
                   **{f"w_{a}": float(w[a].mean()) for a in ASSETS},
                   **{f"w_{a}_lo": float(w[a].min()) for a in ASSETS}, **{f"w_{a}_hi": float(w[a].max()) for a in ASSETS}}
            row.update(sharpe_diff(lx[r], lx[base]) if r != base else {"sharpe_diff": 0.0, "diff_lo": np.nan, "diff_hi": np.nan})
            rows.append(row)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ what the inputs are worth
def risk_accuracy(x: pd.DataFrame, fc: dict, fc21: dict | None = None) -> pd.DataFrame:
    """Each covariance source's forecast at t against the next 12 months, by quantity.

    ``qlike`` (volatilities): mean of v/f - log(v/f) - 1 with v the realized and f
    the forecast variance. ``mse`` (correlations): mean squared error. ``ratio``:
    mean forecast volatility over mean realized volatility. Realized values are
    from the 12 monthly returns after t (variance uncentred), so origins overlap.
    With ``fc21`` a source ``fcst21`` is added: the forecast's 21-day matrix in place
    of its 12-month one."""
    S, base = np.expm1(x), base_weights()
    pairs = [(STOCKS, "ust_int"), (STOCKS, "gold"), ("ust_int", "gold")]
    acc = {}
    for i in range(BURN - 1, len(x) - 12):
        est = estimates(S, i, fc.get(x.index[i]))
        if fc21 is not None:
            est["fcst21"] = estimates(S, i, fc21.get(x.index[i]))["fcst"]
        nx = x.iloc[i + 1:i + 13]
        mix = np.log1p(S.iloc[i + 1:i + 13] @ base)
        real = {**{a: 12 * float((nx[a] ** 2).mean()) for a in ASSETS}, "mix": 12 * float((mix ** 2).mean())}
        rc = nx.corr()
        for s in [k for k in (*SOURCES, "fcst21") if k in est]:
            C = est[s][1]
            f = {**{a: float(C.loc[a, a]) for a in ASSETS}, "mix": float(base @ C.loc[ASSETS, ASSETS] @ base)}
            for k in f:
                acc.setdefault((s, "vol", k), []).append((f[k], real[k]))
            for a, b in pairs:
                acc.setdefault((s, "corr", f"{a}|{b}"), []).append(
                    (float(C.loc[a, b] / np.sqrt(C.loc[a, a] * C.loc[b, b])), float(rc.loc[a, b])))
    rows = []
    for (s, kind, k), v in acc.items():
        f, r = np.array(v).T
        row = {"source": s, "kind": kind, "item": k, "n": len(f), "corr_with_realized": float(np.corrcoef(f, r)[0, 1])}
        if kind == "vol":
            row.update(qlike=float((r / f - np.log(r / f) - 1).mean()),
                       ratio=float(np.sqrt(f).mean() / np.sqrt(r).mean()))
        else:
            row.update(mse=float(((f - r) ** 2).mean()), mean_forecast=float(f.mean()), mean_realized=float(r.mean()))
        rows.append(row)
    return pd.DataFrame(rows)


def persistence(full: dict, block: int = PERSIST) -> pd.DataFrame:
    """Does one block of ``block`` months tell the next? Per holding on its whole history, and pooled.

    Non-overlapping blocks counted back from the last month. ``corr_vol``,
    ``corr_mean`` and ``corr_sharpe`` are correlations between a block's value and
    the next block's; the pooled row stacks the pairs of all holdings after
    removing each holding's own average, so it asks about changes within a
    holding, not about differences between holdings."""
    rows, pooled = [], {"vol": [], "mean": [], "sharpe": []}
    for a, s in full.items():
        s = s.dropna()
        k = len(s) // block
        b = s.iloc[len(s) - k * block:].to_numpy().reshape(k, block)
        v = {"vol": b.std(axis=1, ddof=1) * np.sqrt(12), "mean": 12 * b.mean(axis=1)}
        v["sharpe"] = v["mean"] / v["vol"]
        row = {"asset": a, "start": str(s.index[len(s) - k * block].date()), "blocks": k, "pairs": k - 1}
        for q, z in v.items():
            row[f"corr_{q}"] = float(np.corrcoef(z[:-1], z[1:])[0, 1]) if k >= 4 else np.nan
            row[f"{q}_lo"], row[f"{q}_hi"] = float(z.min()), float(z.max())
            pooled[q].append(np.column_stack([z[:-1], z[1:]]) - z.mean())
        rows.append(row)
    P = {q: np.vstack(z) for q, z in pooled.items()}
    rows.append({"asset": "pooled", "blocks": sum(r["blocks"] for r in rows), "pairs": len(P["vol"]),
                 **{f"corr_{q}": float(np.corrcoef(z[:, 0], z[:, 1])[0, 1]) for q, z in P.items()}})
    return pd.DataFrame(rows)


def means_table(x: pd.DataFrame) -> pd.DataFrame:
    """Each holding's historical mean on the common window with the package's 90% interval, and by half."""
    e = expected_from_xr(x, x.index[-1])
    lo, hi = [c for c in e.columns if c.startswith("ci") and c.endswith("_lo")][0], \
             [c for c in e.columns if c.startswith("ci") and c.endswith("_hi")][0]
    vol = x.std() * np.sqrt(12)
    h = len(x) // 2
    sh = lambda d: 12 * d.mean() / (d.std() * np.sqrt(12))            # noqa: E731
    return pd.DataFrame({"name": pd.Series(NAMES), "mean": e["mean_12m"], "mean_lo": e[lo], "mean_hi": e[hi], "vol": vol,
                         "sharpe": e["mean_12m"] / vol, "sharpe_lo": e[lo] / vol, "sharpe_hi": e[hi] / vol,
                         "sharpe_first_half": sh(x.iloc[:h]), "sharpe_second_half": sh(x.iloc[h:]),
                         "first_half_end": str(x.index[h - 1].date())}).loc[ASSETS].rename_axis("asset").reset_index()


def today(x: pd.DataFrame, fc: dict) -> pd.DataFrame:
    """What every rule holds at the last origin, with its volatility on the forecast covariance."""
    S, t = np.expm1(x), x.index[-1]
    est = estimates(S, len(x) - 1, fc.get(t))
    C = est["fcst"][1].loc[ASSETS, ASSETS]
    rows = []
    for table, rules, fn in (("sleeve", SLEEVE_RULES, sleeve_weights), ("portfolio", PORTFOLIO_RULES, portfolio_weights)):
        for r in rules:
            w = fn(r, est).reindex(ASSETS).fillna(0.0)
            rows.append({"table": table, "rule": r, "name": RULE_NAMES[r], "origin": str(t.date()), **w.to_dict(),
                         "cash": max(1 - float(w.sum()), 0.0), "vol_fcst": float(np.sqrt(w @ C @ w))})
    vol = {s: np.sqrt(np.diag(est[s][1].loc[ASSETS, ASSETS])) for s in SOURCES}
    for j, a in enumerate(ASSETS):
        rows.append({"table": "inputs", "rule": a, "name": NAMES[a], "origin": str(t.date()),
                     "mean_hist": float(est["hist"][0][a]), "mean_recent": float(est["recent"][0][a]),
                     **{f"vol_{s}": float(vol[s][j]) for s in SOURCES}})
    return pd.DataFrame(rows)


def menu(x: pd.DataFrame, rets: pd.DataFrame, cash: pd.Series, fc: dict) -> pd.DataFrame:
    """Stocks beside the sleeve at each mix of ``MENU``, for each sleeve of ``MENU_SLEEVES``.

    Per mix: what it holds at the last origin, its volatility there on the forecast
    covariance, and what the replay did (the sleeve's walk-forward return beside
    stocks, the rest in T-bills) on each period."""
    S = np.expm1(x)
    est = estimates(S, len(x) - 1, fc.get(x.index[-1]))
    C = est["fcst"][1].loc[ASSETS, ASSETS]
    rows = []
    for sl in MENU_SLEEVES:
        ws = sleeve_weights(sl, est)
        for a, b in MENU:
            w = pd.Series({STOCKS: a, **(b * ws).to_dict()})[ASSETS]
            r = a * S[STOCKS].reindex(rets.index) + b * rets[sl]
            row = {"sleeve_rule": sl, "stocks": a, "sleeve": b, "cash": max(1 - a - b, 0.0),
                   **{f"w_{k}": float(v) for k, v in w.items()}, "vol_fcst": float(np.sqrt(w @ C @ w))}
            for p, start in PERIODS.items():
                rp = r.loc[start:] if start else r
                rf = cash.reindex(rp.index)
                st = diversifiers._path_stats((1 + rp) * (1 + rf) - 1, rf)
                row.update({f"{k}_{p}": v for k, v in st.items()})
            rows.append(row)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ run
def run() -> dict:
    x, cash, drag, full = load_inputs()
    fc = forecast_cov(x.index)
    rets, weights = walk_forward(x, fc)
    sl = rule_table(rets, weights, cash, SLEEVE_RULES, "equal")
    pf = rule_table(rets, weights, cash, PORTFOLIO_RULES, "fixed")
    fc21 = forecast_cov(x.index, covhistory.load("21d"))
    ra, ps, mt, td = risk_accuracy(x, fc, fc21), persistence(full), means_table(x), today(x, fc)
    mn = menu(x, rets, cash, fc)
    lvl = np.log1p(rets).cumsum()
    OUT.mkdir(parents=True, exist_ok=True)
    sl.to_csv(OUT / "sleeve_rules.csv", index=False, float_format="%.6g")
    pf.to_csv(OUT / "portfolio_rules.csv", index=False, float_format="%.6g")
    ra.to_csv(OUT / "risk_accuracy.csv", index=False, float_format="%.6g")
    ps.to_csv(OUT / "persistence.csv", index=False, float_format="%.6g")
    mt.to_csv(OUT / "means.csv", index=False, float_format="%.6g")
    td.to_csv(OUT / "today.csv", index=False, float_format="%.6g")
    mn.to_csv(OUT / "menu.csv", index=False, float_format="%.6g")
    weights.to_csv(OUT / "weights.csv", index=False, float_format="%.4f", date_format="%Y-%m-%d")
    lvl.rename_axis("date").to_csv(OUT / "paths.csv", float_format="%.5f", date_format="%Y-%m-%d")
    meta = {"label": LABEL, "returns_through": str(x.index[-1].date()), "common_start": str(x.index[0].date()),
            "first_holding_month": str(rets.index[0].date()), "live_from": LIVE, "assets": ASSETS, "names": NAMES,
            "settings": {"burn": BURN, "recent": RECENT, "persist": PERSIST, "target_vol": TARGET_VOL,
                         "base_stocks": BASE_STOCKS, "max_exposure": 1.0,
                         "boot": {"n": BOOT_N, "block": BOOT_BLOCK, "seed": BOOT_SEED}},
            "covariance": {"model": covhistory.PRIMARY, "bridge": covhistory.BRIDGE},
            "sleeve_rules": SLEEVE_RULES, "portfolio_rules": PORTFOLIO_RULES, "rule_names": RULE_NAMES,
            "rule_inputs": RULE_INPUTS, "drag": drag}
    (OUT / "meta.json").write_text(json.dumps(meta, indent=1))
    return meta


def main(argv=None) -> int:
    meta = run()
    print(LABEL)
    pd.set_option("display.width", 220)
    for f in ("sleeve_rules.csv", "portfolio_rules.csv"):
        d = pd.read_csv(OUT / f)
        print(d[["period", "rule", "growth", "vol", "sharpe", "worst_fall", "turnover", "sharpe_diff", "diff_lo",
                 "diff_hi"]].round(3).to_string(index=False))
    if not _DEV:
        from . import allocation_page
        print(allocation_page.write_report(), allocation_page.write_page())
    print(meta["returns_through"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
