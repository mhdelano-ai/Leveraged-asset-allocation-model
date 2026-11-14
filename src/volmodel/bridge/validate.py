"""Validation of the bridges (produces the tables of reports/models/bridge.md).

    export VOLMODEL_MAX_DATE=2014-12-31 OMP_NUM_THREADS=1
    python -m volmodel.bridge.validate [--out FILE] [--step 1]

Inputs: the stored reference forecasts (``ref_blend`` vols x
``ref_corr_last120m`` correlations) as ``cov_rv``, and an ORACLE ``cov_rv``:
the realized 21-day target matrix of the same window (daily-based assets
only), which removes vol-forecast error and isolates the bridge. Each bridge
maps ``cov_rv`` to ``cov_annual`` with ``panel.asof(origin)``; calibration of
standardized 12-month returns uses ``volmodel.eval.portfolio``
(``annual_z_scores``, ``calibration_summary``: sample-demeaned variance of z,
24-month moving-block bootstrap 90% interval, non-overlapping chi-square
interval, shares beyond 1 and 2 sigma).

The oracle is not a perfect forecast of the annual variance even when the
bridge is right: the realized matrix is estimated from the same window as
the return, so z = R / sqrt(RV) behaves like a t-ratio. Its null value of
E[z^2] (VR = 1, iid normal days) is computed by simulation here and every
oracle variance is reported relative to it.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
from scipy.stats import norm

from ..data.panel import load_panel
from ..eval.harness import load_forecasts
from ..eval.portfolio import POLICIES, RealizedCache, annual_z_scores, calibration_summary
from ..eval.targets import realized_kday_cov
from ..universe import ASSET_IDS
from . import (BRIDGES, ScaleMixture, convention_labels, estimate_long_run, get_bridge,
               market_mask, standardized_returns, variance_ratio, vol_uncertainty, _anchor)

ERAS = {"1926-1945": ("1926-01-01", "1945-12-31"), "1946-1979": ("1946-01-01", "1979-12-31"),
        "1980-1999": ("1980-01-01", "1999-12-31"), "2000-2013": ("2000-01-01", "2013-12-31")}
SINGLE = {f"a_{a}": {"weights": {a: 1.0}, "required": (a,)} for a in ASSET_IDS}
POL = {**POLICIES, **SINGLE}
KEY_PAIRS = [("us_large", "ust_long"), ("us_large", "ust_int"), ("us_large", "intl_dev"), ("us_large", "em"),
             ("us_large", "us_small"), ("us_large", "hy_credit"), ("us_large", "reits"), ("ig_credit", "hy_credit"),
             ("ust_int", "ig_credit"), ("gold", "commodities"), ("us_large", "commodities"), ("ust_long", "tips")]


def md(df: pd.DataFrame, floatfmt: int = 3) -> str:
    d = df.copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else f"{v:.{floatfmt}f}")
    head = "| " + " | ".join(map(str, [d.index.name or ""] + list(d.columns))) + " |"
    sep = "|" + "---|" * (len(d.columns) + 1)
    rows = ["| " + " | ".join(map(str, [i] + list(r))) + " |" for i, r in zip(d.index, d.to_numpy())]
    return "\n".join([head, sep, *rows])


# ------------------------------------------------------------------ inputs
def reference_cov(vol_model="ref_blend", corr_model="ref_corr_last120m") -> dict:
    vol, corr = load_forecasts("vol", vol_model), load_forecasts("corr", corr_model)
    out = {}
    for o, v in vol.items():
        if o in corr:
            c = corr[o].loc[v.index, v.index].to_numpy()
            out[o] = pd.DataFrame(np.outer(v, v) * c, v.index, v.index)
    return out


def oracle_cov(cache: RealizedCache, origins) -> dict:
    out = {}
    for o in origins:
        rw = cache.get(o)
        if len(rw.assets) >= 1 and rw.complete:
            out[o] = pd.DataFrame(rw.cov, list(rw.assets), list(rw.assets))
    return out


def market_window_filter(cov: dict, era: str = "market") -> dict:
    """Drop assets whose 12-month evaluation window contains non-market months."""
    out = {}
    for o, C in cov.items():
        months = pd.date_range(pd.Timestamp(o) + pd.offsets.MonthEnd(1), periods=12, freq="ME")
        keep = [a for a in C.index if market_mask(months, [a], era).all()]
        if keep:
            out[o] = C.loc[keep, keep]
    return out


def bridge_all(cov_rv: dict, panel, names) -> tuple[dict, dict]:
    out, secs = {}, {}
    for name in names:
        b = get_bridge(name)
        t0 = time.perf_counter()
        out[name] = {o: b.cov_annual(C, panel.asof(o)) for o, C in cov_rv.items()}
        secs[name] = time.perf_counter() - t0
    return out, secs


# --------------------------------------------------------- oracle baseline
def oracle_null(n_sim: int = 20000, n_days: int = 252, k: int = 21, seed: int = 0) -> dict:
    """E[z^2] (and tail shares) of z = R / sqrt(RV_21) for iid normal days, VR = 1.

    RV_21 is the target estimator on the same window (uncentered); R the sum of
    the window. Also a stochastic-volatility version (log-vol AR(1), daily
    persistence 0.99, unconditional sd 0.4) since real windows have vol changes.
    """
    rng = np.random.default_rng(seed)
    res = {}
    for label in ("iid", "sv"):
        z = np.empty(n_sim)
        for i in range(n_sim):
            if label == "iid":
                x = rng.standard_normal((n_days, 1))
            else:
                h = np.empty(n_days)
                h[0] = rng.standard_normal() * 0.4
                e = rng.standard_normal(n_days) * 0.4 * np.sqrt(1 - 0.99 ** 2)
                for t in range(1, n_days):
                    h[t] = 0.99 * h[t - 1] + e[t]
                x = (np.exp(h) * rng.standard_normal(n_days))[:, None]
            S, _ = realized_kday_cov(x, k)
            z[i] = x.sum() / np.sqrt(S[0, 0])
        z = z - z.mean()
        res[label] = {"Ez2": float((z ** 2).mean()), "gt1": float((np.abs(z) > 1).mean()),
                      "gt2": float((np.abs(z) > 2).mean())}
    return res


# -------------------------------------------------------------- summaries
def calib_table(z: pd.DataFrame, ports=("a_*", "p6040", "p_balanced", "p_equity", "ew", "invvol", "erc",
                                         "gmv_long", "gmv", "random_pooled"), scale: float = 1.0, drop=()) -> pd.DataFrame:
    cs = calibration_summary(z)
    cs = cs[cs["portfolio"].map(lambda p: p not in drop and
                                any(p == q or (q.endswith("*") and p.startswith(q[:-1])) for q in ports))]
    cs = cs.assign(var_z=cs.var_z / scale, ci_lo=cs.ci_lo / scale, ci_hi=cs.ci_hi / scale,
                   chi2_lo=cs.chi2_lo / scale, chi2_hi=cs.chi2_hi / scale)
    return cs


def pivot_var(cs: pd.DataFrame, models) -> pd.DataFrame:
    rows = {}
    for port, g in cs.groupby("portfolio", sort=False):
        g = g.set_index("model")
        r = {"n": int(g["n"].max())}
        for m in models:
            if m in g.index:
                r[m] = f"{g.at[m, 'var_z']:.2f} [{g.at[m, 'ci_lo']:.2f}, {g.at[m, 'ci_hi']:.2f}]"
        rows[port] = r
    out = pd.DataFrame(rows).T
    out.index.name = "portfolio"
    return out


def tail_table(z: pd.DataFrame, s2_by_origin: pd.Series, models, ports_prefix=("a_", "p6040", "p_balanced", "p_equity",
                                                                          "ew", "erc", "invvol", "gmv_long")) -> pd.DataFrame:
    """Pooled (assets + policy + optimized portfolios) tail shares of sample-demeaned z.

    Each (model, portfolio) series is demeaned by its own full-sample mean.
    ``scaled`` divides z additionally by its own sample sd (shape-only test).
    Predicted lower-tail probabilities use the normal quantile and the mixture
    quantile at each origin's walk-forward s2.
    """
    d = z[(z.metric == "z")].copy()
    d["portfolio"] = d["portfolio"].astype(str)
    d["model"] = d["model"].astype(str)
    d = d[d.portfolio.map(lambda p: any(p.startswith(q) for q in ports_prefix))]
    rows = []
    for m in models:
        g = d[d.model == m].copy()
        if g.empty:
            continue
        g["dev"] = g["value"] - g.groupby("portfolio")["value"].transform("mean")
        g["sd"] = g.groupby("portfolio")["dev"].transform(lambda v: np.sqrt((v ** 2).mean()))
        s2 = s2_by_origin.reindex(pd.DatetimeIndex(g["origin"])).to_numpy()
        for kind, zz in (("raw", g["dev"].to_numpy()), ("scaled", (g["dev"] / g["sd"]).to_numpy())):
            q05m = np.array([ScaleMixture(s).quantile(0.05) for s in s2]) if kind else None
            q01m = np.array([ScaleMixture(s).quantile(0.01) for s in s2])
            rows.append({"model": m, "z": kind, "n": len(zz),
                         "gt1": np.mean(np.abs(zz) > 1), "gt2": np.mean(np.abs(zz) > 2),
                         "lt_q05_normal": np.mean(zz < norm.ppf(0.05)), "lt_q05_mix": np.mean(zz < q05m),
                         "lt_q01_normal": np.mean(zz < norm.ppf(0.01)), "lt_q01_mix": np.mean(zz < q01m),
                         "kurt": float(np.mean(zz ** 4) / np.mean(zz ** 2) ** 2)})
    return pd.DataFrame(rows)


def vr_tables(panel) -> dict:
    out = {}
    for era in ("all", "market", "strict", "recent"):
        e = estimate_long_run(panel, era=era, cross=False)
        out[era] = e
    t = pd.DataFrame({
        "n_mkt": out["market"].n, "raw_mkt": out["market"].vr_raw, "se_mkt": out["market"].vr_se,
        "se_iid": out["market"].vr_se_iid, "shrunk_mkt": out["market"].vr,
        "raw_all": out["all"].vr_raw, "shrunk_all": out["all"].vr,
        "raw_strict": out["strict"].vr_raw, "shrunk_strict": out["strict"].vr,
        "raw_recent": out["recent"].vr_raw, "se_recent": out["recent"].vr_se, "shrunk_recent": out["recent"].vr,
    })
    t.index.name = "asset"
    taus = {era: e.tau2_vr for era, e in out.items()}
    return {"table": t, "tau2": taus}


def vr_by_convention(panel) -> pd.DataFrame:
    """Raw VR per asset and convention label (only labels with >= 60 months)."""
    x = panel.excess_monthly
    U = standardized_returns(x)
    rows = []
    for a in x.columns:
        u = U[a].dropna()
        lab = convention_labels(u.index, a)
        for lb in np.unique(lab):
            uu = u[lab == lb]
            if len(uu) < 60:
                continue
            v, se, _, n = variance_ratio(uu.to_numpy())
            rows.append({"asset": a, "convention": lb, "start": uu.index[0].date(), "end": uu.index[-1].date(),
                         "n": n, "vr": v, "se": se})
        for name, lo, hi in (("market, pre-1985", None, "1984-12-31"), ("market, 1985+", "1985-01-01", None)):
            m = (lab == "market") & (u.index >= pd.Timestamp(lo or "1800")) & (u.index <= pd.Timestamp(hi or "2100"))
            uu = u[m]
            if len(uu) >= 60:
                v, se, _, n = variance_ratio(uu.to_numpy())
                rows.append({"asset": a, "convention": name, "start": uu.index[0].date(), "end": uu.index[-1].date(),
                             "n": n, "vr": v, "se": se})
    return pd.DataFrame(rows).set_index("asset")


def vr_local_mean(panel, window: int = 120) -> pd.Series:
    """Diagnostic: VR of standardized returns demeaned by a centered 120-month moving
    average (removes decade-scale drifts in the mean). Not bias-corrected for the
    moving-average filter (which lowers it by roughly 2 x 11 x 1/120 ~ 0.1 extra)."""
    U = standardized_returns(panel.excess_monthly)
    M = market_mask(U.index, list(U.columns), "market")
    out = {}
    for j, a in enumerate(U.columns):
        u = U[a].where(M[:, j]).dropna()
        if len(u) < 2 * window:
            out[a] = np.nan
            continue
        loc = u.rolling(window, center=True, min_periods=window // 2).mean()
        out[a] = variance_ratio((u - loc).to_numpy())[0]
    return pd.Series(out)


def walk_forward_vr(panel, era="market", years=(1950, 1970, 1990, 2000, 2007, 2014)) -> pd.DataFrame:
    rows = {}
    for y in years:
        v = panel.asof(pd.Timestamp(f"{y}-12-31"))
        e = estimate_long_run(v, list(v.eligible(36)), era=era, cross=False)
        rows[y] = e.vr
    return pd.DataFrame(rows)


def _corr12(x: pd.DataFrame, a: str, b: str) -> tuple[float, float, int]:
    """Model-free check: centered correlation of monthly returns and of overlapping
    12-month sums, on the months where both assets are market-priced."""
    m = market_mask(x.index, [a, b], "market").all(axis=1) & x[[a, b]].notna().all(axis=1).to_numpy()
    d = x.loc[m, [a, b]]
    if len(d) < 60:
        return np.nan, np.nan, len(d)
    c1 = float(d.corr().iloc[0, 1])
    s = d.rolling(12).sum().dropna()
    # keep only windows of 12 consecutive calendar months
    span = (d.index.to_series().shift(-0) - d.index.to_series().shift(11)).reindex(s.index)
    s = s[(span.dt.days < 370).to_numpy()]
    return c1, float(s.corr().iloc[0, 1]), len(d)


def cross_table(panel):
    e = estimate_long_run(panel, era="market", cross=True)
    x = panel.excess_monthly
    rows = []
    for a, b in KEY_PAIRS:
        c1, c12, nn = _corr12(x, a, b)
        rows.append({"pair": f"{a}-{b}", "n": int(e.n_pair.at[a, b]), "rho_month_std": e.rho0.at[a, b],
                     "rho_annual_std": e.rho_annual_raw.at[a, b], "gap": e.gap_raw.at[a, b],
                     "se": e.gap_se.at[a, b], "t": e.gap_raw.at[a, b] / e.gap_se.at[a, b],
                     "gap_shrunk": e.gap.at[a, b], "corr_1m_raw": c1, "corr_12m_raw": c12})
    t = pd.DataFrame(rows).set_index("pair")
    iu = np.triu_indices(len(e.assets), 1)
    g, se = e.gap_raw.to_numpy()[iu], e.gap_se.to_numpy()[iu]
    ok = np.isfinite(se)
    tt = g[ok] / se[ok]
    gs = e.gap.to_numpy()[iu][ok]
    summ = {"tau2_gap": e.tau2_gap, "n_pairs": int(ok.sum()), "share_|t|>2": float(np.mean(np.abs(tt) > 2)),
            "mean_t": float(np.mean(tt)), "mean_gap_raw": float(np.mean(g[ok])),
            "mean_abs_gap_raw": float(np.mean(np.abs(g[ok]))), "mean_abs_gap_shrunk": float(np.mean(np.abs(gs))),
            "max_abs_gap_shrunk": float(np.max(np.abs(gs))),
            "largest_shrunk": ", ".join(f"{e.assets[i]}-{e.assets[j]} {e.gap.iat[i, j]:+.3f}"
                                        for i, j in sorted(zip(*iu), key=lambda ij: -abs(e.gap.iat[ij[0], ij[1]]))[:6])}
    return t, summ


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--n-sim", type=int, default=4000)
    args = ap.parse_args(argv)
    T0 = time.perf_counter()
    lines = []
    P = lambda *s: (print(*s, flush=True), lines.append(" ".join(map(str, s))))

    panel = load_panel()
    ref = reference_cov()
    origins = sorted(ref)
    if args.step > 1:
        origins = [o for o in origins if (o.year * 12 + o.month) % args.step == 0]
        ref = {o: ref[o] for o in origins}
    cache = RealizedCache(panel)
    orc = oracle_cov(cache, origins)
    names = list(BRIDGES)

    P("## Variance ratios at the 2014-12 view\n")
    vt = vr_tables(panel)
    P(md(vt["table"]))
    P("\ntau^2 by era:", {k: round(v, 4) for k, v in vt["tau2"].items()})
    P("\n## Raw VR by pricing convention and period\n")
    P(md(vr_by_convention(panel)))
    P("\n## Local-mean diagnostic (VR around a centered 120m moving mean, market months)\n")
    P(md(pd.DataFrame({"vr_local": vr_local_mean(panel), "vr_raw_market": vt["table"]["raw_mkt"]})))
    P("\n## Walk-forward shrunk VR (market era) at selected year-ends\n")
    P(md(walk_forward_vr(panel)))
    P("\n## Lead-lag (cross) terms, market era, 2014 view\n")
    ct, cs_ = cross_table(panel)
    P(md(ct))
    P("\n", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in cs_.items()})

    P("\n## Oracle null (VR = 1)\n")
    null = oracle_null(n_sim=args.n_sim)
    P(null)
    base = null["iid"]["Ez2"]

    t1 = time.perf_counter()
    A_ref, secs = bridge_all(ref, panel, names)
    A_orc, _ = bridge_all(orc, panel, names)
    P("\nbridge seconds per", len(ref), "origins (reference input):", {k: round(v, 1) for k, v in secs.items()})

    zr = annual_z_scores(panel, A_ref, cache=cache, n_random=10, policies=POL)
    zo = annual_z_scores(panel, A_orc, cache=cache, n_random=10, policies=POL)
    A_ref_m = {m: market_window_filter(d) for m, d in A_ref.items()}
    A_orc_m = {m: market_window_filter(d) for m, d in A_orc.items()}
    zrm = annual_z_scores(panel, A_ref_m, cache=cache, n_random=10, policies=POL)
    zom = annual_z_scores(panel, A_orc_m, cache=cache, n_random=10, policies=POL)

    for title, z, scale in (("REFERENCE input, all windows", zr, 1.0), ("REFERENCE input, market-priced windows", zrm, 1.0),
                            ("ORACLE input, all daily windows (var_z / oracle null)", zo, base),
                            ("ORACLE input, market-priced daily windows (var_z / oracle null)", zom, base)):
        P(f"\n## Calibration: {title}\n")
        # unconstrained GMV on a realized (oracle) 12-asset matrix from ~11 independent
        # 21-day sums is near-singular by construction: not a test of the bridge
        P(md(pivot_var(calib_table(z, scale=scale, drop=("gmv",) if scale != 1.0 else ()), names)))
    for title, z, scale in (("REFERENCE, market windows", zrm, 1.0), ("ORACLE, market windows", zom, base)):
        P(f"\n## By era (origin): {title} - var_z [90% block-bootstrap CI]\n")
        rows = []
        for era, (lo, hi) in ERAS.items():
            zz = z[(z.origin >= lo) & (z.origin <= hi)]
            if zz.empty:
                continue
            cs = calib_table(zz, ports=("a_*", "p6040", "p_balanced", "random_pooled", "gmv_long", "erc"), scale=scale)
            # pooled single assets: average var over a_* series
            a = cs[cs.portfolio.str.startswith("a_")].groupby("model")[["var_z", "ci_lo", "ci_hi"]].median()
            for m in names:
                r = {"era": era, "model": m}
                if m in a.index:
                    r["assets(median)"] = f"{a.at[m, 'var_z']:.2f}"
                for p in ("p6040", "p_balanced", "random_pooled", "erc", "gmv_long"):
                    g = cs[(cs.model == m) & (cs.portfolio == p)]
                    if len(g):
                        g = g.iloc[0]
                        r[p] = f"{g.var_z:.2f} [{g.ci_lo:.2f},{g.ci_hi:.2f}]"
                rows.append(r)
        P(md(pd.DataFrame(rows).set_index("era")))

    # tails (reference input, market windows): s2 walk-forward, re-estimated each December
    P("\n## Tails: reference input, market windows\n")
    s2 = {}
    memo = {}
    for o in origins:
        v = panel.asof(o)
        an = _anchor(v)
        if an not in memo:
            memo[an] = vol_uncertainty(v.asof(an))["s2"]
        s2[o] = memo[an]
    s2 = pd.Series(s2)
    P("s2 walk-forward: min %.3f median %.3f max %.3f, last %.3f" % (s2.min(), s2.median(), s2.max(), s2.iloc[-1]))
    vu = vol_uncertainty(panel)
    P("vol_uncertainty(2014 view):", {k: round(v, 4) if isinstance(v, float) else v for k, v in vu.items()})
    mix = ScaleMixture(float(s2.median()))
    P("mixture at median s2: kurtosis %.2f, P(|X|>1) %.3f, P(|X|>2) %.3f, q05 %.3f (normal %.3f), q01 %.3f (normal %.3f)"
      % (mix.kurtosis, mix.share_beyond(1), mix.share_beyond(2), mix.quantile(.05), norm.ppf(.05),
         mix.quantile(.01), norm.ppf(.01)))
    tt = tail_table(zrm, s2, names)
    P(md(tt.set_index("model")))
    P("\n## Tails: oracle input, market windows (shape relative to oracle null)\n")
    P("oracle null:", null)
    P(md(tail_table(zom, s2 * 0, names).set_index("model")))
    P(f"\nvalidation run time {time.perf_counter() - T0:.0f}s (bridging+scoring {time.perf_counter() - t1:.0f}s)")
    if args.out:
        with open(args.out, "w") as f:
            f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
