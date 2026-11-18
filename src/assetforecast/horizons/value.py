"""Economic value: what would an investor who used these forecasts have earned?

Sleeves (returns and volatility, one asset at a time)
-----------------------------------------------------
At each origin ``t`` a tranche sets the weight in one asset against cash,

    w = clip((mu + sigma2 / 2) / (GAMMA * sigma2), 0, CAP)

with ``mu`` the forecast annualized log excess return over the next ``h``
months and ``sigma2`` the forecast annualized variance (the 21-day measure of
``risk``), and holds it for ``h`` months. The position in a month is the mean
of the tranches still alive (the last ``h`` origins that have a forecast; the
newest must exist), so a forecast is judged over its own horizon while returns
stay monthly. The monthly return is ``w (R - Rf)`` less the one-way trading
cost of the package's strategy test (``strategy.COST_BP``) on the change in
weight and its financing spread on any weight above 1. ``GAMMA`` and ``CAP``
are that test's settings (risk aversion 5; the Campbell-Thompson cap 1.5).

Four specifications per (asset, horizon): the baseline (``hist_mean`` and the
``naive`` variance), a return model with the naive variance, the benchmark
return with a variance model, and both. ``d_cer`` is the gain in annualized
certainty-equivalent return, ``12 (mean - GAMMA/2 variance)`` of the monthly
return, over the baseline on the months both exist; ``t`` tests it (utility
differences, Bartlett long-run variance with 1.5 h lags). ``pooled`` is the
equal-weight average of the asset sleeves.

Minimum variance (volatility and correlation together)
------------------------------------------------------
At each origin the long-only minimum-variance portfolio (``eval.portfolio``)
of the assets with a full forecast is built from ``D R D`` with volatilities
``D`` and correlations ``R`` each from ``naive`` or ``half``, and its variance
over the next ``h`` months is measured with the realized 21-day covariance.
``ratio`` is realized volatility relative to the all-naive portfolio (below 1:
the forecasts lowered risk); ``dm_t`` tests the variance difference.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..eval.portfolio import gmv_long_only_weights, regularize
from ..returns.evaluation import inference
from ..returns.evaluation.strategy import COST_BP, FINANCING_SPREAD
from ..returns.evaluation.strategy_rules import RuleSettings
from . import AFTER_START, DEV_END, MIN_INDEP, ML_START, PERIODS, period_mask

GAMMA = RuleSettings().gamma
CAP = RuleSettings().timing_cap
BASE = ("hist_mean", "naive")
MU_MODELS = ("sig_combo", "hz_ens")
VAR_MODELS = ("half", "har")


def _hold_mask(months, period: str) -> np.ndarray:
    m = pd.DatetimeIndex(months)
    ml, dev_end, after = (pd.Timestamp(x) for x in (ML_START, DEV_END, AFTER_START))
    return np.asarray({"early": m <= ml, "dev": (m > ml) & (m <= dev_end), "after": m >= after,
                       "full": m > ml}[period])


def _positions(w: pd.DataFrame, h: int) -> pd.DataFrame:
    """Mean of the live tranches (last h origins), where the newest exists."""
    pos = w.rolling(h, min_periods=1).mean()
    return pos.where(w.notna())


def _sleeve(pos: pd.DataFrame, rx: pd.DataFrame) -> pd.DataFrame:
    """Monthly net return of holding ``pos`` (set at the previous month end)."""
    held = pos.shift(1)
    cost = pd.Series({a: COST_BP[a] * 1e-4 for a in pos.columns})
    trade = (held - held.shift(1).fillna(0.0)).abs() * cost
    fin = (held - 1.0).clip(lower=0.0) * FINANCING_SPREAD / 12.0
    return held * rx - trade - fin


def _cer(r: np.ndarray) -> float:
    return float(12.0 * (r.mean() - 0.5 * GAMMA * r.var()))


def _compare(ra: pd.Series, rb: pd.Series, h: int) -> dict:
    j = pd.concat([ra, rb], axis=1, keys=["a", "b"]).dropna()
    if len(j) < 24:
        return {}
    a, b = j["a"].to_numpy(), j["b"].to_numpy()
    u = (a - 0.5 * GAMMA * (a - a.mean()) ** 2) - (b - 0.5 * GAMMA * (b - b.mean()) ** 2)
    t, p1, _, n = inference.mean_test(u, h)
    return {"cer": _cer(a), "cer_base": _cer(b), "d_cer": _cer(a) - _cer(b), "t": t, "p": p1,
            "mean_ann": 12 * a.mean(), "vol_ann": float(np.sqrt(12) * a.std()),
            "n": n, "n_indep": n / h, "thin": n / h < MIN_INDEP}


def sleeves(fc: pd.DataFrame, vol: pd.DataFrame, simple: pd.DataFrame, cash: pd.Series,
            periods=PERIODS) -> pd.DataFrame:
    """``fc``: return forecasts (model, horizon, date, asset, mean); ``vol``: the
    k = 21 rows of ``risk.run``; ``simple``, ``cash``: monthly simple returns."""
    rx = simple.sub(cash.reindex(simple.index), axis=0)
    months, assets = rx.index, list(rx.columns)
    vol = vol[vol["k"] == 21]
    rows = []
    for h in sorted(set(fc["horizon"]) & set(vol["horizon"])):
        mu = {m: g.pivot(index="date", columns="asset", values="mean").reindex(
                  index=months, columns=assets) * (12.0 / h)
              for m, g in fc[fc["horizon"] == h].groupby("model") if m in (BASE[0], *MU_MODELS)}
        vh = vol[vol["horizon"] == h]
        var = {m: vh.pivot(index="date", columns="asset", values=m).reindex(
                   index=months, columns=assets) for m in (BASE[1], *VAR_MODELS)}
        specs = [BASE, *[(m, BASE[1]) for m in MU_MODELS if m in mu],
                 *[(BASE[0], v) for v in VAR_MODELS],
                 *[(m, v) for m in MU_MODELS if m in mu for v in VAR_MODELS]]
        ret, wavg, turn = {}, {}, {}
        for m, v in specs:
            w = ((mu[m] + 0.5 * var[v]) / (GAMMA * var[v])).clip(0.0, CAP)
            pos = _positions(w, h)
            ret[(m, v)] = _sleeve(pos, rx)
            wavg[(m, v)], turn[(m, v)] = pos, pos.diff().abs()
        for spec in specs[1:]:
            r, rb = ret[spec], ret[BASE]
            both = r.notna() & rb.notna()
            series = {a: (r[a], rb[a]) for a in assets}
            series["pooled"] = (r.where(both).mean(axis=1), rb.where(both).mean(axis=1))
            for per in periods:
                sel = _hold_mask(months, per)
                for a, (x, y) in series.items():
                    c = _compare(x[sel], y[sel], h)
                    if not c:
                        continue
                    if a == "pooled":
                        ok = both.to_numpy() & sel[:, None]
                        wa = float(np.nanmean(wavg[spec].shift(1).to_numpy()[ok]))
                        wb = float(np.nanmean(wavg[BASE].shift(1).to_numpy()[ok]))
                        tu = float(12 * np.nanmean(turn[spec].shift(1).to_numpy()[ok]))
                    else:
                        ok = both[a].to_numpy() & sel
                        wa = float(wavg[spec][a].shift(1)[ok].mean())
                        wb = float(wavg[BASE][a].shift(1)[ok].mean())
                        tu = float(12 * turn[spec][a].shift(1)[ok].mean())
                    rows.append({"horizon": h, "asset": a, "period": per, "mu": spec[0],
                                 "var": spec[1], **c, "weight": wa, "weight_base": wb,
                                 "turnover_ann": tu})
    return pd.DataFrame(rows)


def _cube(g: pd.DataFrame, col: str, dates, assets) -> np.ndarray:
    """dates x N x N correlation array from long pair rows (diagonal 1)."""
    N = len(assets)
    out = np.full((len(dates), N, N), np.nan)
    w = g.pivot(index="date", columns="pair", values=col).reindex(dates)
    ai = {a: i for i, a in enumerate(assets)}
    for p in w.columns:
        a, b = p.split("|")
        out[:, ai[a], ai[b]] = out[:, ai[b], ai[a]] = w[p].to_numpy()
    out[:, np.arange(N), np.arange(N)] = 1.0
    return out


def minvar(vol: pd.DataFrame, corr: pd.DataFrame, periods=PERIODS) -> pd.DataFrame:
    vol, corr = vol[vol["k"] == 21], corr[corr["k"] == 21]
    assets = list(pd.unique(vol["asset"]))
    combos = [("naive", "naive"), ("half", "naive"), ("naive", "half"), ("half", "half")]
    rows = []
    for h in sorted(set(corr["horizon"])):
        vh, ch = vol[vol["horizon"] == h], corr[corr["horizon"] == h]
        dates = pd.DatetimeIndex(sorted(set(ch["date"])))
        V = {m: vh.pivot(index="date", columns="asset", values=m).reindex(index=dates, columns=assets).to_numpy()
             for m in ("target", "naive", "half", "published") if m in vh}
        C = {m: _cube(ch, m, dates, assets) for m in ("target", "naive", "half", "published") if m in ch}
        use = combos + ([("published", "published")] if "published" in V and "published" in C else [])
        rv = {c: np.full(len(dates), np.nan) for c in use}
        nass = np.zeros(len(dates), dtype=int)
        for t in range(len(dates)):
            ok = np.isfinite(V["target"][t]) & np.isfinite(V["naive"][t]) & np.isfinite(V["half"][t])
            idx = np.flatnonzero(ok)
            while len(idx) >= 3:      # drop assets until every pair is complete
                sub = np.ix_(idx, idx)
                bad = ~(np.isfinite(C["target"][t][sub]) & np.isfinite(C["naive"][t][sub])
                        & np.isfinite(C["half"][t][sub]))
                if not bad.any():
                    break
                idx = np.delete(idx, bad.sum(axis=0).argmax())
            if len(idx) < 3:
                continue
            sub = np.ix_(idx, idx)
            sd = np.sqrt(V["target"][t][idx])
            S = C["target"][t][sub] * np.outer(sd, sd)
            nass[t] = len(idx)
            for vm, cm in use:
                f = V[vm][t][idx]
                R = C[cm][t][sub]
                if not (np.isfinite(f).all() and np.isfinite(R).all()):
                    continue
                d = np.sqrt(f)
                w = gmv_long_only_weights(regularize(R * np.outer(d, d)))
                rv[(vm, cm)][t] = w @ S @ w
        base = rv[combos[0]]
        for c in use[1:]:
            for per in periods:
                sel = period_mask(dates, h, per) & np.isfinite(rv[c]) & np.isfinite(base)
                if sel.sum() < 12:
                    continue
                t, p1, _ = inference.diebold_mariano(rv[c][sel], base[sel], h)
                n = int(sel.sum())
                rows.append({"horizon": h, "vol": c[0], "corr": c[1], "period": per,
                             "vol_realized": float(np.sqrt(rv[c][sel].mean())),
                             "vol_realized_naive": float(np.sqrt(base[sel].mean())),
                             "ratio": float(np.sqrt(rv[c][sel].mean() / base[sel].mean())),
                             "dm_t": t, "dm_p": p1, "n": n, "n_indep": n / h,
                             "thin": n / h < MIN_INDEP, "assets_avg": float(nass[sel].mean())})
    return pd.DataFrame(rows)
