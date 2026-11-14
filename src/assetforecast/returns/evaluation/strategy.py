"""Economic value of the 12-month forecasts: a small, pre-declared set of portfolio rules.

    python -m assetforecast.returns.evaluation.strategy
    python -m assetforecast.returns.evaluation.strategy --max-date 2014-12-31

Ported from ``assetpred.evaluation.strategy`` (and ``strategy_rules``,
``strategy_stats``) as a REPLICATION: the rules, the accounting and the tests
are unchanged; the run is labelled with ``config.REPLICATION_LABEL`` because
both holdouts were already spent when it was made, so it is not out-of-sample
evidence. What the new setting forced, and nothing else:

* the universe (``assets``) and the one-way cost table (``cost_bp``) are
  arguments, so the same code runs on assetpred's eleven ids (the
  reproduction check) and on the package's twelve;
* 12-month horizon only (the staggered-sleeve machinery stays; the h=1-only
  ``vs_static`` comparisons are gone);
* the risk model is the package's covariance forecast, ``af_primary``
  (``covhistory``, read from ``<results>/covariances/af_primary/h12.parquet``);
  the static comparator ``sample`` comes from ``static_cov``;
* the periods are a parameter (``config.PERIODS``: ``dev``, ``holdout``,
  ``holdout_ap``), and the development switch is ``--max-date``.

Question 1 (mean forecasts). The risk model is held fixed (``--cov-model``)
and each rule is run twice, once on a model's expected returns and once on
``hist_mean``'s; only the expected returns differ. Question 2 (risk
forecasts). Expected returns are held at ``hist_mean`` and each rule is run
on each dynamic covariance model (``--risk-models``) and on the static
``sample`` covariance. Question 3 (static reference). 1/N of the eligible
assets and 60/40 (US large caps / intermediate Treasuries), monthly
rebalanced, reported as levels only.

Rules (fixed before looking at any result; see ``strategy_rules.py``)
-------------------------------------------------------------------
gamma = 5 (middle of the 1-10 range of DeMiguel, Garlappi & Uppal 2009; Campbell &
Thompson 2008 use 3 for a single equity index), long only, at most 100% of
capital in one asset and 200% gross (Reg-T 50% initial margin). Residual in
T-bills; borrowing pays T-bill + ``FINANCING_SPREAD``.

* ``mv``      constrained plug-in mean-variance: max w'mu - gamma/2 w'Sigma w.
              Long-only and the caps are the robustness device (Jagannathan &
              Ma 2003: constraints act as shrinkage); exposure varies with the
              level of mu, so this rule rewards aggregate timing.
* ``mv_vt``   the same objective direction at a fixed 10% ex-ante volatility
              (max w'mu s.t. w'Sigma w <= 0.10^2/12): rewards only the
              cross-asset allocation, not the decision how much risk to take.
* ``timing``  per-asset timing, no correlations, no optimizer: 1/N of capital
              to each eligible asset, each sleeve at the Campbell-Thompson
              weight mu_i / (gamma sigma_i^2) clipped to [0, 1.5]. Position
              risk is proportional to the forecast Sharpe ratio.
* ``vt_rp``   (risk test only) inverse-volatility weights scaled to 10%
              ex-ante volatility (Moreira & Muir 2017 style volatility
              management), same caps.

mu is the forecast mean of the h-month excess log return converted to a
monthly simple excess return, (mean + Sigma_ii / 2) / h, with Sigma the risk
model's h-month covariance (so the conversion term is identical for every
mean model); the monthly risk input is Sigma / h.

Horizon. The 12-month forecast made at each origin sets a sleeve held for 12
months; the portfolio is the average of the 12 most recent sleeves
(Jegadeesh-Titman staggering), so it trades monthly, has no overlapping-return
problem in the monthly P&L, and months enter only once all 12 sleeves exist.
Inference on those series uses HAC lags for horizon 12. The static reference
portfolios use no forecast and hold their monthly target (no staggering),
over the same holding months as the forecast rules.

Costs (one-way, per unit of turnover, see ``COST_BP``) and financing are
charged in the "net" series; "gross" has neither. Turnover is measured
against drifted weights. The first month's build-up is charged.

Timing. Weights formed at the close of month t use forecasts with origin t
(and, for the drift, returns through t) and earn the return of month t+1.
Eligible set at t: assets with a forecast at t from both the model and
``hist_mean``, a covariance at t, and in the universe (``all`` or ``ex_hy``).

Metrics (annualized; on monthly excess returns over T-bills). Levels
(benchmark = "none"): ann_mean, ann_vol, sharpe, cer (gamma = 5), max_drawdown
(of total-return wealth), turnover_ann (sum of one-way |dw|), avg_leverage,
cost/financing drag, ex_ante_vol_avg, sd_standardized (sd of return / ex-ante
vol; 1 = risk model calibrated), vol_track_rmse_log (RMSE of log rolling-12m
realized vol / 10%), vol_of_vol, contrib_ann_<asset> (gross P&L attribution).
Comparisons (benchmark = the other strategy; family ``mean``: model vs
hist_mean, ``risk``: dynamic vs sample covariance): d_sharpe / d_cer with HAC
se, t, p, 95% CI and stationary-bootstrap CI and p
(``strategy_stats.diff_test``), BH q-values across models x rules
(``*_q_bh``), n_eff (months x var / long-run var of the return difference),
d_max_drawdown (> 0: shallower), d_vol_track_rmse_log (< 0: better),
d_contrib_ann_<asset>, and breakeven_cost_bp: the uniform one-way cost at which
the net CER gain is zero (0 = no gain even before trading costs).

Outputs under ``paths.RESULTS_RETURNS / "strategy"``: ``metrics.parquet``
(tidy: strategy, family, rule, mean_model, cov_model, universe, horizon,
period, basis, benchmark, metric, value), ``returns.parquet`` (monthly series
of every strategy), ``weights.parquet`` (held weights by formation date),
``summary.csv`` (headline comparisons, net, full sample),
``summary_by_period.csv`` (the same with a leading ``period`` column) and
``meta.json`` (all assumptions, the replication label and sample
diagnostics). The run also writes the static comparator to
``<results>/covariances/sample/h12.parquet``.

Periods. Every run has ``full`` and the decades (by holding month, at least 24
months). The periods in ``periods`` (default ``config.PERIODS``' ``dev``,
``holdout`` and ``holdout_ap``, each a (first, last) range of origins) are by
sleeve formation date: a holding month belongs to a period only if every
sleeve in its position was formed at an origin inside the range, i.e. the
oldest sleeve, formation - 11 months, is >= first and the newest, the
formation date, is <= last. Months whose sleeves straddle a boundary are only
in ``full``. A period needs at least 12 months. Every metric, test, CI,
break-even and BH q-value is computed per period. BH q-values (``*_q_bh``)
cover only config.FROZEN_MEAN; the robustness rows (config.ROBUSTNESS) are in
the comparison but reported individually.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config, schema
from . import inference as inf
from . import static_cov
from . import strategy_rules as rules
from . import strategy_stats as st
from .walkforward import load_forecasts

# assetforecast.paths and assetforecast.returns.inputs are imported inside the
# functions that need them: ``main`` sets ASSETFORECAST_MAX_DATE first.

log = logging.getLogger("assetforecast.returns.evaluation.strategy")

HORIZON = 12
ASSETS = tuple(config.ASSETS)
BENCH_MEAN = "hist_mean"
STATIC_COV = static_cov.NAME           # "sample"
PRIMARY_COV = "af_primary"             # covhistory.to_cov_columns(name="af_primary")
FROZEN_MEAN = tuple(config.FROZEN_MEAN)
# Mean models in the comparison by default: the frozen six and the robustness
# rows (same learners without full-sample "hindsight" features). Only the
# frozen six form the Benjamini-Hochberg family; the others are reported
# individually.
DEFAULT_MEAN = (*FROZEN_MEAN, *config.ROBUSTNESS)
BH_FAMILY = FROZEN_MEAN
DEFAULT_RISK = (PRIMARY_COV,)
UNIVERSES = {"all": (), "ex_hy": ("hy_credit",)}
SUBPERIODS = tuple((k, a, b) for k, (a, b) in config.DECADES.items())
# Periods by sleeve formation date: name -> (first origin, last origin).
PERIODS = {k: tuple(config.PERIODS[k]) for k in ("dev", "holdout", "holdout_ap")}
SUMMARY_PERIODS = ("holdout", "holdout_ap", "dev", "full")
BOOT_REPS = 999
SIXTY_FORTY = ("us_large", "ust_int")

# One-way proportional cost per unit of turnover, basis points. Judgement for
# a mid-sized investor implementing each class with the cheapest liquid
# vehicle (index futures / ETFs / on-the-run bonds) over 1990-2014, i.e.
# above today's ETF quoted half-spreads (~1-5bp) to allow for the 1990s,
# market impact and the absence of ETFs for much of the period. Bond costs
# follow the ordering of Edwards, Harris & Piwowar (2007) and Bao et al.
# (2011): Treasuries << IG corporates < high yield. The mutual-fund series
# trade at NAV with no spread, but a monthly trading rule cannot be run
# through the funds themselves (round-trip restrictions); the cost stands for
# a tradable proxy. Synthetic GSW bonds and TIPS are costed like the
# corresponding cash bonds / ETFs.
#
# assetpred's table, on its ids (used by the reproduction check):
COST_BP_AP = {
    "us_eq": 5, "dev_eq": 10, "em_eq": 25, "ust_int": 3, "ust_long": 5, "tips": 10,
    "ig_credit": 20, "hy_credit": 40, "reits": 20, "gold": 10, "commodities": 10,
}
# The same costs carried over by asset to the package ids (us_eq -> us_large,
# dev_eq -> intl_dev, em_eq -> em); us_small is new.
COST_BP = {
    "us_large": 5, "us_small": 10, "intl_dev": 10, "em": 25, "ust_int": 3, "ust_long": 5,
    "tips": 10, "ig_credit": 20, "hy_credit": 40, "reits": 20, "gold": 10, "commodities": 10,
}
# Costs that are new judgements made at the merge, not carried over from
# assetpred's pre-declared table. Recorded in every run's meta.json.
COST_NEW_JUDGEMENTS = {
    "us_small": ("10bp one-way: new asset at the merge, a judgement made then (twice US large caps, "
                 "for the wider spreads and higher impact of small-cap futures/ETFs, especially in "
                 "the 1990s); not part of assetpred's pre-declared table"),
}
# Borrowing cost over T-bills on gross exposure above 100% (annual): roughly
# the average 3m LIBOR - T-bill (TED) spread 1990-2014, the rate at which
# futures/swap-implemented leverage is financed.
FINANCING_SPREAD = 0.005


@dataclass(frozen=True)
class Spec:
    rule: str
    mean_model: str     # "none" for rules that ignore expected returns
    cov_model: str      # "none" for static rules
    universe: str
    horizon: int

    @property
    def name(self) -> str:
        return f"{self.rule}|{self.mean_model}|{self.cov_model}|{self.universe}|h{self.horizon}"


# ----------------------------------------------------------------- inputs
@dataclass
class Inputs:
    returns: pd.DataFrame            # simple total returns, month-end index, columns = assets
    cash: pd.Series
    origins: dict                    # h -> DatetimeIndex of forecast origins
    mean: dict                       # (model, h) -> (T_h, N) log-excess mean forecasts
    cov: dict                        # (model, h) -> (T_h, N, N) covariance of h-month excess log returns
    assets: tuple = ASSETS


def _months_after(d: pd.DatetimeIndex, k: int) -> pd.DatetimeIndex:
    return d + pd.offsets.MonthEnd(k)


def panel_returns(max_date=None) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """(simple total returns, simple T-bill returns, excess log returns) of the
    return panel (``inputs.load_return_panel``: tier D dropped, from
    ``config.RETURN_START``), through its last complete month."""
    from .. import inputs
    panel = inputs.load_return_panel(max_date)
    xr = inputs.excess_returns(panel)
    R = panel.monthly.reindex(index=xr.index, columns=list(xr.columns))
    rf = panel.cash_monthly.reindex(xr.index)
    return R, rf, xr


def forecast_origins(fc: pd.DataFrame, h: int, last) -> pd.DatetimeIndex:
    """Origins of ``hist_mean``'s h-month forecasts whose target is realized by ``last``."""
    f = fc[fc.horizon == h]
    o = pd.DatetimeIndex(sorted(f.loc[f.model == BENCH_MEAN, "date"].unique()))
    # only origins whose h-month target is realized within the loaded returns
    o = o[_months_after(o, h) <= last]
    if len(o) and not (o == pd.date_range(o[0], periods=len(o), freq="ME")).all():
        raise ValueError(f"h={h}: forecast origins are not consecutive months")
    return o


def _check_ids(found, assets, what: str) -> None:
    unknown = sorted(set(map(str, found)) - set(assets))
    if unknown:
        raise ValueError(f"{what}: asset ids not in the universe {list(assets)}: {unknown}")


def load_inputs(results_dir: Path, mean_models, cov_models, returns: pd.DataFrame, cash: pd.Series,
                assets=ASSETS, horizon: int = HORIZON) -> Inputs:
    """Forecasts from ``<results_dir>/forecasts/<model>/h12.parquet`` and
    covariances from ``<results_dir>/covariances/<model>/h12.parquet``, on the
    origins of ``hist_mean`` whose target is realized within ``returns``."""
    assets = tuple(assets)
    h = int(horizon)
    R = returns.reindex(columns=list(assets))
    rf = cash
    last = min(R.index.max(), rf.index.max())
    models = list(dict.fromkeys([BENCH_MEAN, *mean_models]))
    fc = load_forecasts(models, [h], results_dir)
    fc["date"] = pd.to_datetime(fc["date"])
    _check_ids(fc["asset"].unique(), assets, "forecasts")
    origins, mean, cov = {}, {}, {}
    aidx = {a: i for i, a in enumerate(assets)}
    f = fc[fc.horizon == h]
    o = forecast_origins(fc, h, last)
    origins[h] = o
    for m in models:
        w = f[f.model == m].pivot(index="date", columns="asset", values="mean")
        mean[(m, h)] = w.reindex(index=o, columns=list(assets)).to_numpy()
    for c in cov_models:
        path = Path(results_dir) / "covariances" / c / f"h{h:02d}.parquet"
        df = pd.read_parquet(path, columns=["date", "asset_i", "asset_j", "cov"])
        df["date"] = pd.to_datetime(df["date"])
        _check_ids(pd.unique(df[["asset_i", "asset_j"]].to_numpy().ravel()), assets, f"covariances {c}")
        df = df[df.date.isin(o)]
        gap = o.difference(pd.DatetimeIndex(df.date.unique()))
        if len(gap):
            raise ValueError(f"covariances {c}: no matrix at {len(gap)} forecast origins, "
                             f"first {gap[0].date()}")
        arr = np.full((len(o), len(assets), len(assets)), np.nan)
        di = o.get_indexer(df.date)
        arr[di, df.asset_i.map(aidx).to_numpy(), df.asset_j.map(aidx).to_numpy()] = df["cov"].to_numpy()
        cov[(c, h)] = arr
    return Inputs(R, rf, origins, mean, cov, assets)


# ----------------------------------------------------------------- weights
def target_weights(spec: Spec, inp: Inputs, s: rules.RuleSettings = rules.RuleSettings(),
                   sixty_forty=SIXTY_FORTY) -> np.ndarray:
    """(T_h, N) target weights at each origin of horizon h (one sleeve)."""
    h = spec.horizon
    A = list(inp.assets)
    o = inp.origins[h]
    T, N = len(o), len(A)
    hm = inp.mean[(BENCH_MEAN, h)]
    excl = np.array([a in UNIVERSES[spec.universe] for a in A])
    W = np.zeros((T, N))
    if spec.rule == "sixty_forty":
        eq, bond = sixty_forty
        W[:, A.index(eq)] = 0.6
        W[:, A.index(bond)] = 0.4
        return W
    m = hm if spec.mean_model in ("none", BENCH_MEAN) else inp.mean[(spec.mean_model, h)]
    S_all = inp.cov[(spec.cov_model, h)] if spec.cov_model != "none" else None
    func = rules.RULE_FUNCS[spec.rule]
    for k in range(T):
        elig = np.isfinite(m[k]) & np.isfinite(hm[k]) & ~excl
        if S_all is not None:
            elig &= np.isfinite(np.diagonal(S_all[k]))
            S = S_all[k] / h
            mu = np.where(elig, (m[k] + 0.5 * np.diagonal(S_all[k])) / h, np.nan)
        else:
            S = None
            mu = np.where(elig, m[k], np.nan)
        W[k] = func(mu, S, s) if S is not None else func(mu, None, s)
    return W


def held_weights(W: np.ndarray, h: int) -> tuple[np.ndarray, np.ndarray]:
    """Average of the h most recent sleeves; returns (held, valid mask)."""
    if h == 1:
        return W, np.ones(len(W), bool)
    T = len(W)
    cs = np.concatenate([np.zeros((1, W.shape[1])), np.cumsum(W, axis=0)])
    held = np.full_like(W, np.nan)
    valid = np.arange(T) >= h - 1
    held[valid] = (cs[h:] - cs[:-h]) / h
    return held, valid


# ----------------------------------------------------------------- backtest
def simulate(held: np.ndarray, formation: pd.DatetimeIndex, returns: pd.DataFrame, cash: pd.Series,
             cost_bp=None, spread: float = FINANCING_SPREAD, cov_m: np.ndarray | None = None,
             assets=None) -> pd.DataFrame:
    """Monthly P&L of weights ``held[k]`` formed at ``formation[k]`` and held over the next month.

    ``assets``: the column order of ``held`` (default ``returns.columns``);
    ``cost_bp``: one-way cost per asset (default ``COST_BP``)."""
    assets = list(returns.columns) if assets is None else list(assets)
    cost_bp = COST_BP if cost_bp is None else cost_bp
    month = _months_after(formation, 1)
    R = returns.reindex(index=month, columns=assets).to_numpy()
    rf = cash.reindex(month).to_numpy()
    if np.isnan(rf).any():
        raise ValueError("missing T-bill return in a holding month")
    if (np.isnan(R) & (held != 0)).any():
        raise ValueError("a held asset has no return in its holding month")
    X = np.nan_to_num(R) - rf[:, None]
    gross = (held * X).sum(1)
    tot = rf + gross
    prev = np.zeros_like(held)
    cont = np.r_[False, formation[1:] == _months_after(formation[:-1], 1)]
    drift = held[:-1] * (1 + np.nan_to_num(R[:-1])) / (1 + tot[:-1])[:, None]
    prev[1:][cont[1:]] = drift[cont[1:]]
    dw = np.abs(held - prev)
    c = np.array([cost_bp[a] for a in assets]) * 1e-4
    turnover = dw.sum(1)
    cost = dw @ c
    lev = held.sum(1)
    fin = spread / 12 * np.maximum(lev - 1, 0)
    out = pd.DataFrame({"formation": formation, "date": month, "rf": rf, "gross": gross,
                        "net": gross - cost - fin, "turnover": turnover, "cost": cost,
                        "financing": fin, "leverage": lev})
    contrib = pd.DataFrame(held * X, columns=[f"c_{a}" for a in assets])
    out = pd.concat([out, contrib], axis=1)
    if cov_m is not None:
        C = np.nan_to_num(cov_m)
        out["ex_ante_vol"] = np.sqrt(np.einsum("ti,tij,tj->t", held, C, held) * 12)
    else:
        out["ex_ante_vol"] = np.nan
    return out


def run_spec(spec: Spec, inp: Inputs, s: rules.RuleSettings = rules.RuleSettings(), cost_bp=None,
             sixty_forty=SIXTY_FORTY):
    h = spec.horizon
    W = target_weights(spec, inp, s, sixty_forty)
    if spec.rule in rules.RULES_STATIC:
        # no forecast, so no sleeves: hold the monthly target, over the same
        # holding months as the staggered forecast rules
        held, valid = W, np.arange(len(W)) >= h - 1
    else:
        held, valid = held_weights(W, h)
    o = inp.origins[h]
    cov_m = inp.cov[(spec.cov_model, h)][valid] / h if spec.cov_model != "none" else None
    ret = simulate(held[valid], o[valid], inp.returns, inp.cash, cost_bp=cost_bp, cov_m=cov_m,
                   assets=inp.assets)
    wts = pd.DataFrame(held[valid], index=o[valid], columns=list(inp.assets))
    return ret, wts


# ----------------------------------------------------------------- specs
def build_specs(mean_models, cov_model, risk_models, horizon: int = HORIZON) -> list[tuple[str, Spec]]:
    """(family, spec) for every strategy. Family: mean / risk / static."""
    out = {}
    h = int(horizon)
    for u in UNIVERSES:
        for r in rules.RULES_MEAN:
            for m in [BENCH_MEAN, *mean_models]:
                out.setdefault(Spec(r, m, cov_model, u, h), "mean")
            for c in [STATIC_COV, *risk_models]:
                out.setdefault(Spec(r, BENCH_MEAN, c, u, h), "risk")
        for c in [STATIC_COV, *risk_models]:
            out.setdefault(Spec("vt_rp", "none", c, u, h), "risk")
    for u in UNIVERSES:
        out[Spec("ew", "none", "none", u, h)] = "static"
    out[Spec("sixty_forty", "none", "none", "all", h)] = "static"
    return [(f, s) for s, f in out.items()]


def comparisons(specs, mean_models, cov_model, risk_models) -> list[tuple[str, Spec, Spec]]:
    have = {s for _, s in specs}
    pairs = []
    for _, s in specs:
        if s.rule in rules.RULES_MEAN and s.cov_model == cov_model and s.mean_model in mean_models:
            pairs.append(("mean", s, Spec(s.rule, BENCH_MEAN, cov_model, s.universe, s.horizon)))
        if s.cov_model in risk_models and s.mean_model in (BENCH_MEAN, "none"):
            pairs.append(("risk", s, Spec(s.rule, s.mean_model, STATIC_COV, s.universe, s.horizon)))
    return [p for p in pairs if p[1] in have and p[2] in have and p[1] != p[2]]


# ----------------------------------------------------------------- metrics
def period_masks(formation, h: int, periods=None) -> dict[str, np.ndarray]:
    """{period: mask over holding months} for ranges of sleeve formation dates.

    A holding month formed at ``formation`` holds the sleeves formed at
    formation - (h-1) .. formation months; it belongs to period (first, last)
    if all of them do: oldest >= first and formation <= last. Months whose
    sleeves straddle a boundary belong to neither side."""
    periods = PERIODS if periods is None else periods
    formation = pd.DatetimeIndex(formation)
    oldest = _months_after(formation, -(h - 1)) if h > 1 else formation
    return {name: np.asarray((oldest >= pd.Timestamp(a)) & (formation <= pd.Timestamp(b)))
            for name, (a, b) in periods.items()}


def _periods(dates: pd.Series, formation, h: int, periods=None, subperiods=SUBPERIODS):
    yield "full", np.ones(len(dates), bool)
    for name, a, b in subperiods:
        m = ((dates >= a) & (dates <= b)).to_numpy()
        if m.sum() >= 24:
            yield name, m
    for name, m in period_masks(formation, h, periods).items():
        if m.sum() >= 12:
            yield name, m


def level_metrics(ret: pd.DataFrame, s: rules.RuleSettings, h: int = HORIZON, periods=None,
                  assets=None) -> list[tuple]:
    assets = [c[2:] for c in ret.columns if c.startswith("c_")] if assets is None else list(assets)
    rows = []
    for per, m in _periods(ret.date, ret.formation, h, periods):
        r = ret[m]
        for basis in ("gross", "net"):
            x = r[basis].to_numpy()
            tot = r.rf.to_numpy() + x
            vals = {
                "n_months": len(x), "ann_mean": 12 * x.mean(), "ann_vol": x.std(ddof=1) * math.sqrt(12),
                "sharpe": st.sharpe(x), "cer": st.cer(x, s.gamma), "max_drawdown": st.max_drawdown(tot),
                "skew": pd.Series(x).skew(), "ex_kurt": pd.Series(x).kurt(),
                "turnover_ann": 12 * r.turnover.mean(), "avg_leverage": r.leverage.mean(),
                "cost_drag_ann": 12 * r.cost.mean(), "financing_drag_ann": 12 * r.financing.mean(),
            }
            ev = r.ex_ante_vol.to_numpy()
            if np.isfinite(ev).all() and (ev > 0).all():
                vals["ex_ante_vol_avg"] = ev.mean()
                vals["sd_standardized"] = float(np.std(x / (ev / math.sqrt(12)), ddof=1))
            rm, vv = st.vol_tracking(x, s.vol_target)
            vals["vol_track_rmse_log"] = rm
            vals["vol_of_vol"] = vv
            if basis == "gross":   # P&L attribution: annualized contribution of each asset
                for a in assets:
                    vals[f"contrib_ann_{a}"] = 12 * r[f"c_{a}"].mean()
            rows += [(per, basis, k, float(v)) for k, v in vals.items()]
    return rows


def compare(a: pd.DataFrame, b: pd.DataFrame, h: int, s: rules.RuleSettings, rng_seed: int,
            boot_cache: dict, periods=None, assets=None) -> list[tuple]:
    assets = [c[2:] for c in a.columns if c.startswith("c_")] if assets is None else list(assets)
    j = a.merge(b, on="date", suffixes=("_a", "_b"))
    if not (j.formation_a == j.formation_b).all():
        raise ValueError("compared strategies have different formation dates")
    rows = []
    for per, m in _periods(j.date, j.formation_a, h, periods):
        r = j[m]
        n = len(r)
        key = (n, h)
        if key not in boot_cache:
            rng = np.random.default_rng(rng_seed + n * 31 + h)
            boot_cache[key] = st.stationary_bootstrap_indices(n, BOOT_REPS, inf.block_length(h, n), rng)
        idx = boot_cache[key]
        for basis in ("gross", "net"):
            d = st.diff_test(r[f"{basis}_a"].to_numpy(), r[f"{basis}_b"].to_numpy(), s.gamma, h, idx)
            d["d_turnover_ann"] = 12 * (r.turnover_a.mean() - r.turnover_b.mean())
            d["d_max_drawdown"] = (st.max_drawdown(r.rf_a + r[f"{basis}_a"])
                                   - st.max_drawdown(r.rf_b + r[f"{basis}_b"]))
            ra, rb = (st.vol_tracking(r[f"{basis}_{k}"].to_numpy(), s.vol_target)[0] for k in "ab")
            d["d_vol_track_rmse_log"] = ra - rb
            if basis == "gross":
                for x in assets:
                    d[f"d_contrib_ann_{x}"] = 12 * (r[f"c_{x}_a"].mean() - r[f"c_{x}_b"].mean())
            if basis == "net":
                d["breakeven_cost_bp"] = st.breakeven(r.gross_a, r.gross_b, r.turnover_a, r.turnover_b,
                                                      r.financing_a, r.financing_b, s.gamma)
            rows += [(per, basis, k, float(v)) for k, v in d.items()]
    return rows


def diagnostics(inp: Inputs, h: int = HORIZON) -> dict:
    A = list(inp.assets)
    o = inp.origins[h]
    months = _months_after(o, 1)
    xr = np.log1p(inp.returns).sub(np.log1p(inp.cash), axis=0).reindex(months)
    out = {"first_formation_month": str(months.min().date()), "last_month": str(months.max().date()),
           "n_origins": int(len(o))}
    elig = np.isfinite(inp.mean[(BENCH_MEAN, h)])
    out["eligible_assets_first"] = [a for a, e in zip(A, elig[0]) if e]
    out["eligible_assets_last"] = [a for a, e in zip(A, elig[-1]) if e]
    out["avg_eligible_assets"] = float(elig.sum(1).mean())
    out["asset_entry"] = {a: str(o[np.argmax(elig[:, i])].date()) for i, a in enumerate(A) if elig[:, i].any()}
    # effective number of independent assets (participation ratio of correlation eigenvalues)
    for label, cols in (("all", out["eligible_assets_first"]), (f"all_{len(A)}_common_window", A)):
        sub = xr[cols].dropna()
        lam = np.linalg.eigvalsh(sub.corr().to_numpy())
        out[f"effective_n_assets_{label}"] = float(lam.sum() ** 2 / (lam ** 2).sum())
        out[f"effective_n_assets_{label}_months"] = int(len(sub))
    out["ac1_monthly_excess"] = {a: float(xr[a].dropna().autocorr(1)) for a in A}
    return out


# ----------------------------------------------------------------- driver
_INP: Inputs | None = None
_JOB_KW: dict = {}


def _init(inp, kw):
    global _INP
    _INP = inp
    _JOB_KW.update(kw)


def _job(spec: Spec):
    return spec, run_spec(spec, _INP, **_JOB_KW)


def write_sample_cov(results_dir: Path, xr: pd.DataFrame, origins, horizon: int = HORIZON) -> Path:
    """The static comparator at ``origins`` -> ``<results_dir>/covariances/sample/hXX.parquet`` (+ .json)."""
    df = static_cov.sample_cov_history(xr, horizon, origins)
    path = Path(results_dir) / "covariances" / STATIC_COV / f"h{int(horizon):02d}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    path.with_suffix(".json").write_text(json.dumps({
        "model": STATIC_COV, "horizon": int(horizon), "rows": len(df),
        "first_origin": str(df["date"].min().date()) if len(df) else None,
        "last_origin": str(df["date"].max().date()) if len(df) else None,
        "min_history": static_cov.MIN_HISTORY, "mean_adjust": True,
        "assets": list(map(str, xr.columns)),
        "source": "assetforecast.returns.evaluation.static_cov.sample_cov_history",
    }, indent=1))
    return path


def run(results_dir: Path, mean_models=DEFAULT_MEAN, cov_model: str = PRIMARY_COV, risk_models=DEFAULT_RISK,
        *, assets=ASSETS, cost_bp=None, periods=None, sixty_forty=SIXTY_FORTY,
        returns: pd.DataFrame | None = None, cash: pd.Series | None = None, build_sample: bool = True,
        max_date: str | None = None, horizon: int = HORIZON, seed: int = config.SEED, jobs: int = 3,
        out_dir: Path | None = None, cost_notes: dict | None = None, label: str = config.REPLICATION_LABEL,
        extra_meta: dict | None = None) -> dict:
    """Run every strategy, compare, and write the outputs.

    ``returns``/``cash`` default to the return panel (``panel_returns``);
    pass them to run on other inputs (the archived assetpred panel). With
    ``build_sample`` the static ``sample`` covariance is computed from those
    returns at the forecast origins and written under ``results_dir`` first;
    otherwise it is read from there like every other covariance."""
    t0 = time.time()
    s = rules.RuleSettings()
    h = int(horizon)
    assets = tuple(assets)
    cost_bp = dict(COST_BP if cost_bp is None else cost_bp)
    cost_notes = dict(COST_NEW_JUDGEMENTS if cost_notes is None else cost_notes)
    periods = dict(PERIODS if periods is None else periods)
    missing = [a for a in assets if a not in cost_bp]
    if missing:
        raise ValueError(f"no trading cost for {missing}")
    max_date = max_date or os.environ.get("ASSETFORECAST_MAX_DATE") or None
    if returns is None or cash is None:
        returns, cash, xr = panel_returns(max_date)
    else:
        xr = schema.excess_log_returns(returns, cash)
    risk_models = [c for c in risk_models if c != STATIC_COV]
    cov_models = list(dict.fromkeys([cov_model, STATIC_COV, *risk_models]))
    if build_sample:
        last = min(returns.index.max(), cash.index.max())
        fc = load_forecasts([BENCH_MEAN], [h], results_dir)
        fc["date"] = pd.to_datetime(fc["date"])
        o = forecast_origins(fc, h, last)
        write_sample_cov(results_dir, xr.reindex(columns=list(assets)), o, h)
    inp = load_inputs(results_dir, mean_models, cov_models, returns, cash, assets, h)
    if max_date is not None:
        o = inp.origins[h]
        if len(o) and _months_after(o[-1:], h)[0] > pd.Timestamp(max_date):
            raise RuntimeError("max-date run would use a target realized after max_date")
    specs = build_specs(mean_models, cov_model, risk_models, h)
    fam = {sp: f for f, sp in specs}
    log.info("%d strategies, loading took %.1fs", len(specs), time.time() - t0)

    job_kw = {"cost_bp": cost_bp, "sixty_forty": sixty_forty}
    results = {}
    todo = [sp for _, sp in specs]
    if jobs > 1:
        import multiprocessing as mp
        with ProcessPoolExecutor(jobs, mp_context=mp.get_context("fork"), initializer=_init,
                                 initargs=(inp, job_kw)) as ex:
            for sp, res in ex.map(_job, todo, chunksize=4):
                results[sp] = res
    else:
        for sp in todo:
            results[sp] = run_spec(sp, inp, s, **job_kw)
    log.info("backtests done %.1fs", time.time() - t0)

    cols = ["strategy", "family", "rule", "mean_model", "cov_model", "universe", "horizon"]

    def ident(sp, family):
        return [sp.name, family, sp.rule, sp.mean_model, sp.cov_model, sp.universe, sp.horizon]

    rows = []
    for sp, (ret, _) in results.items():
        for per, basis, k, v in level_metrics(ret, s, sp.horizon, periods, assets):
            rows.append([*ident(sp, fam[sp]), per, basis, "none", k, v])
    boot_cache = {}
    pairs = comparisons(specs, mean_models, cov_model, risk_models)
    for family, a, b in pairs:
        for per, basis, k, v in compare(results[a][0], results[b][0], a.horizon, s, seed, boot_cache,
                                       periods, assets):
            rows.append([*ident(a, family), per, basis, b.name, k, v])
    metrics = pd.DataFrame(rows, columns=[*cols, "period", "basis", "benchmark", "metric", "value"])
    metrics = _add_fdr(metrics)
    log.info("metrics done %.1fs", time.time() - t0)

    rets, wts = [], []
    for sp, (ret, w) in results.items():
        r = ret.copy()
        for c, v in zip(cols, ident(sp, fam[sp])):
            r[c] = v
        rets.append(r)
        wl = w.rename_axis("formation").reset_index().melt("formation", var_name="asset", value_name="weight")
        wl = wl[wl.weight != 0]
        for c, v in zip(cols, ident(sp, fam[sp])):
            wl[c] = v
        wts.append(wl)
    rets = pd.concat(rets, ignore_index=True)
    wts = pd.concat(wts, ignore_index=True)
    wts["date"] = _months_after(pd.DatetimeIndex(wts.formation), 1)
    for df in (rets, wts, metrics):
        for c in ("strategy", "family", "rule", "mean_model", "cov_model", "universe"):
            df[c] = df[c].astype("category")

    out = Path(out_dir) if out_dir is not None else Path(results_dir) / "strategy"
    out.mkdir(parents=True, exist_ok=True)
    metrics.to_parquet(out / "metrics.parquet", index=False, compression="zstd")
    # human-readable headline: full-sample comparisons
    summary_table(metrics).to_csv(out / "summary.csv", float_format="%.6g")
    # the same with a period column
    summary_by_period(metrics, periods=_summary_order(periods)).to_csv(out / "summary_by_period.csv",
                                                                       float_format="%.6g")
    rets[[*cols, "formation", "date", "rf", "gross", "net", "turnover", "cost", "financing", "leverage",
          "ex_ante_vol", *[f"c_{a}" for a in assets]]].to_parquet(out / "returns.parquet", index=False,
                                                                  compression="zstd")
    wts[[*cols, "formation", "date", "asset", "weight"]].to_parquet(out / "weights.parquet", index=False,
                                                                    compression="zstd")
    meta = {
        "label": label, "max_date": max_date, "results_dir": str(results_dir), "assets": list(assets),
        "mean_models": list(mean_models), "bench_mean": BENCH_MEAN, "cov_model": cov_model,
        "risk_models": risk_models, "static_cov": STATIC_COV, "horizons": [h],
        "universes": {k: list(v) for k, v in UNIVERSES.items()}, "sixty_forty": list(sixty_forty),
        "rule_settings": asdict(s), "cost_bp_one_way": cost_bp, "cost_new_judgements": cost_notes,
        "financing_spread_annual": FINANCING_SPREAD,
        "boot_reps": BOOT_REPS, "seed": seed, "subperiods": SUBPERIODS,
        "n_strategies": len(specs), "n_comparisons": len(pairs),
        "periods": {k: list(v) for k, v in periods.items()},
        "period_rule": ("periods are ranges of sleeve formation dates (origins): a holding month belongs to "
                        "(first, last) if every sleeve in its position was formed in it, i.e. formation - "
                        f"{h - 1} months >= first and formation <= last; straddling months are only in "
                        "'full'; decades are by holding month"),
        "static_rule": "ew and sixty_forty hold their monthly target (no sleeves) over the same months",
        "bh_family": list(BH_FAMILY),
        "not_in_bh_family": [m for m in mean_models if m not in BH_FAMILY],
        "diagnostics": diagnostics(inp, h), "runtime_s": round(time.time() - t0, 1),
        **(extra_meta or {}),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=1, default=str))
    log.info("wrote %s (%.1fs)", out, time.time() - t0)
    return {"metrics": metrics, "returns": rets, "weights": wts, "meta": meta}


def _add_fdr(metrics: pd.DataFrame) -> pd.DataFrame:
    """Benjamini-Hochberg q-values for the mean-forecast comparisons, across
    the frozen models (``BH_FAMILY``) x rules within (universe, horizon,
    period, basis). Robustness rows get no q-value."""
    extra = []
    sel = metrics[(metrics.family == "mean") & metrics.metric.isin(["d_sharpe_p", "d_cer_p"])
                  & metrics.mean_model.isin(BH_FAMILY)]
    for _, g in sel.groupby(["universe", "horizon", "period", "basis", "metric"]):
        q = inf.fdr(g.value.to_numpy(), "bh")
        e = g.copy()
        e["metric"] = e.metric.str.replace("_p", "_q_bh", regex=False)
        e["value"] = q
        extra.append(e)
    return pd.concat([metrics, *extra], ignore_index=True) if extra else metrics


def summary_table(metrics: pd.DataFrame, period: str = "full", basis: str = "net",
                  horizon: int | None = None) -> pd.DataFrame:
    """Headline: each comparison's d_sharpe, d_cer and p-values (net, full sample by default)."""
    m = metrics[(metrics.period == period) & (metrics.basis == basis) & (metrics.benchmark != "none")]
    if horizon is not None:
        m = m[m.horizon == horizon]
    keep = ["d_sharpe", "d_sharpe_p", "d_sharpe_boot_p", "d_cer", "d_cer_p", "d_cer_ci_lo", "d_cer_ci_hi",
            "d_cer_boot_p", "d_cer_q_bh", "breakeven_cost_bp", "d_turnover_ann", "n_eff"]
    t = m[m.metric.isin(keep)].pivot_table(index=["family", "horizon", "universe", "rule", "mean_model",
                                                  "cov_model"],
                                           columns="metric", values="value", observed=True)
    return t.reindex(columns=keep).round(4)


def _summary_order(periods) -> tuple[str, ...]:
    """Holdout-type periods first, in SUMMARY_PERIODS order where listed, then 'full'."""
    known = [p for p in SUMMARY_PERIODS if p in periods]
    return (*known, *[p for p in periods if p not in known], "full")


def summary_by_period(metrics: pd.DataFrame, basis: str = "net", periods=SUMMARY_PERIODS) -> pd.DataFrame:
    """``summary_table`` for each period in ``periods`` present (in that order),
    with a leading ``period`` index level."""
    parts = {p: summary_table(metrics, period=p, basis=basis, horizon=None)
             for p in periods if (metrics.period == p).any()}
    return pd.concat(parts, names=["period"])


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m assetforecast.returns.evaluation.strategy",
                                 description=__doc__.split("\n")[0])
    ap.add_argument("--max-date", default=None,
                    help="development run: sets ASSETFORECAST_MAX_DATE (returns truncated there, inputs "
                         "and outputs under results/dev/returns)")
    ap.add_argument("--results-dir", default=None, help="results root (default paths.RESULTS_RETURNS)")
    ap.add_argument("--mean-models", nargs="+", default=list(DEFAULT_MEAN),
                    help="default: config.FROZEN_MEAN + config.ROBUSTNESS")
    ap.add_argument("--cov-model", default=PRIMARY_COV, help="risk model held fixed in the mean-forecast test")
    ap.add_argument("--risk-models", nargs="+", default=list(DEFAULT_RISK),
                    help="dynamic covariance models compared with the static 'sample' (mean = hist_mean)")
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    if args.max_date:
        if "assetforecast.paths" in sys.modules and \
                os.environ.get("ASSETFORECAST_MAX_DATE") != args.max_date:
            raise RuntimeError("assetforecast.paths was imported before --max-date could set "
                               "ASSETFORECAST_MAX_DATE; run this module as a script")
        os.environ["ASSETFORECAST_MAX_DATE"] = args.max_date
    from ... import paths
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(asctime)s %(name)s %(message)s")
    results_dir = Path(args.results_dir) if args.results_dir is not None else paths.RESULTS_RETURNS
    mean_models = [m for m in args.mean_models if m != BENCH_MEAN]
    res = run(results_dir, mean_models, args.cov_model, args.risk_models, max_date=args.max_date,
              jobs=args.jobs)
    met = res["metrics"]
    print(config.REPLICATION_LABEL)
    for p in ("holdout", "holdout_ap", "dev"):
        if (met.period == p).any():
            a, b = PERIODS[p]
            print(f"\n{p.upper()}: positions all of whose sleeves were formed at origins in [{a}, {b}] (net):")
            print(summary_table(met, period=p).to_string())
    print("\nFULL SAMPLE (net):")
    print(summary_table(met).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
