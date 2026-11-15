"""Write ``reports/horizon_study.md`` and its figures from the tables in ``horizons.TABLES``.

    python -m assetforecast.horizons.report

Every number in the report is read from those tables; this module computes
nothing but counts and formats.
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from . import FIGURES, HORIZONS, LABEL, MIN_INDEP, REPORT, TABLES

ASSET_ORDER = ["us_large", "us_small", "intl_dev", "em", "ust_int", "ust_long", "tips",
               "ig_credit", "hy_credit", "reits", "gold", "commodities"]
HL = {1: "1m", 3: "3m", 6: "6m", 12: "1y", 24: "2y", 36: "3y", 60: "5y", 120: "10y"}
PER = {"dev": "1990 to 2014", "after": "2015 on"}


def load() -> dict:
    t = {n: pd.read_csv(TABLES / f"{n}.csv") for n in
         ("return_scores", "vol_scores", "corr_scores", "sleeves", "minvar")}
    t["meta"] = json.loads((TABLES / "meta.json").read_text())
    return t


def _fmt(v, kind="f3"):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return ""
    return {"f3": f"{v:+.3f}", "f2": f"{v:+.2f}", "p": f"{v:.2f}", "pct": f"{100 * v:+.1f}%", "pct2": f"{100 * v:+.2f}%",
            "pct0": f"{100 * v:.0f}%", "bp": f"{1e4 * v:+.0f}", "r3": f"{v:.3f}"}[kind]


def md_table(df: pd.DataFrame, kind="f3", index_name="") -> str:
    cols = [HL.get(c, str(c)) for c in df.columns]
    out = ["| " + " | ".join([index_name, *cols]) + " |", "|" + "---|" * (len(cols) + 1)]
    for i, r in df.iterrows():
        name = " / ".join(map(str, i)) if isinstance(i, tuple) else str(i)
        out.append("| " + " | ".join([name, *[_fmt(x, kind) if not isinstance(x, str) else x
                                              for x in r]]) + " |")
    return "\n".join(out)


def by_h(df, value, rows, **where) -> pd.DataFrame:
    """Pivot horizon into columns after filtering; thin cells become NaN."""
    m = np.ones(len(df), dtype=bool)
    for k, v in where.items():
        m &= df[k].isin(v) if isinstance(v, (list, tuple)) else (df[k] == v)
    d = df[m].copy()
    d.loc[d["thin"].astype(bool), value] = np.nan
    p = d.pivot_table(index=rows, columns="horizon", values=value, dropna=False)
    return p.reindex(columns=[h for h in HORIZONS if h in p.columns])


def asset_map(df, value, **where) -> pd.DataFrame:
    p = by_h(df, value, "asset", **where)
    return p.reindex([a for a in ASSET_ORDER if a in p.index])


def heat(panels: list, path, vmax: float, fmt="{:+.2f}", note="") -> None:
    """Row of asset x horizon heatmaps. Blue: better than the naive forecast; red: worse."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("div", ["#c0392b", "#f0efec", "#2a78d6"])
    cmap.set_bad("#ffffff")
    fig, axes = plt.subplots(1, len(panels), figsize=(5.6 * len(panels), 5.0), squeeze=False)
    for ax, (title, p) in zip(axes[0], panels):
        a = p.to_numpy(dtype=float)
        ax.imshow(np.ma.masked_invalid(a), cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(p.shape[1]), [HL.get(c, c) for c in p.columns])
        ax.set_yticks(range(p.shape[0]), p.index)
        ax.set_title(title, fontsize=11, loc="left")
        ax.tick_params(length=0, labelsize=9)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_xticks(np.arange(-.5, p.shape[1]), minor=True)
        ax.set_yticks(np.arange(-.5, p.shape[0]), minor=True)
        ax.grid(which="minor", color="white", linewidth=2)
        ax.tick_params(which="minor", length=0)
        for i in range(a.shape[0]):
            for j in range(a.shape[1]):
                if np.isfinite(a[i, j]):
                    ax.text(j, i, fmt.format(a[i, j]), ha="center", va="center", fontsize=7.5,
                            color="white" if abs(a[i, j]) > 0.6 * vmax else "#0b0b0b")
                else:
                    ax.text(j, i, "·", ha="center", va="center", fontsize=8, color="#898781")
    fig.text(0.01, 0.01, f"Blue: beats the naive forecast. Red: worse. Colour clipped at ±{vmax:g}. "
             f"Dot: no forecast or fewer than {MIN_INDEP:g} independent windows. {note}",
             fontsize=8, color="#52514e")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def both_positive(dev: pd.DataFrame, after: pd.DataFrame) -> tuple[int, int]:
    """(cells positive in both periods, cells scored in both)."""
    d, a = dev.align(after)
    ok = d.notna() & a.notna()
    return int(((d > 0) & (a > 0) & ok).sum().sum()), int(ok.sum().sum())


def persist_list(dev: pd.DataFrame, after: pd.DataFrame) -> pd.DataFrame:
    """Per asset: the horizons with positive skill in both periods (a description, not a pick)."""
    d, a = dev.align(after)
    rows = {}
    for asset in d.index:
        ok = [HL[h] for h in d.columns if d.loc[asset, h] > 0 and a.loc[asset, h] > 0]
        rows[asset] = ", ".join(ok) if ok else "none"
    return pd.Series(rows)


def build() -> str:
    t = load()
    rs, vs, cs, sl, mv, meta = (t[k] for k in ("return_scores", "vol_scores", "corr_scores",
                                                "sleeves", "minvar", "meta"))
    rmods = [m for m in ("sig_combo", "hz_ridge", "hz_gbm", "hz_ens") if m in set(rs["model"])]
    L = []
    w = L.append
    w("# Horizon study: what is predictable from 1 month to 10 years\n")
    w(f"> {LABEL}.\n")
    w("Generated by `python -m assetforecast.horizons.run` (tables in `results/horizons/`, "
      f"returns through {meta['returns_through']}). The interactive version is "
      "`dashboard/horizons.html`.\n")

    # ------------------------------------------------------------------ returns
    r_dev = {m: asset_map(rs, "r2", model=m, period="dev") for m in rmods}
    r_aft = {m: asset_map(rs, "r2", model=m, period="after") for m in rmods}
    pooled = {p: by_h(rs, "r2", "model", asset="pooled", period=p, model=rmods) for p in PER}
    pooled_p = {p: by_h(rs, "cw_p", "model", asset="pooled", period=p, model=rmods) for p in PER}
    v_dev = {k: asset_map(vs, "skill", k=k, model="half", period="dev") for k in (21, 1)}
    v_aft = {k: asset_map(vs, "skill", k=k, model="half", period="after") for k in (21, 1)}
    c_dev = {k: asset_map(cs, "skill", k=k, model="half", period="dev") for k in (21, 1)}
    c_aft = {k: asset_map(cs, "skill", k=k, model="half", period="after") for k in (21, 1)}

    w("## The short answer\n")
    nb = {m: both_positive(r_dev[m], r_aft[m]) for m in rmods}
    s12 = [h for h in HORIZONS if h <= 12]
    lng = [h for h in (36, 60) if h in pooled["dev"].columns]
    ni = by_h(rs, "n_indep", "model", asset="pooled", period="dev", model=rmods)
    rng = lambda d, hs: f"{_fmt(d[hs].min().min(), 'pct')} to {_fmt(d[hs].max().max(), 'pct')}"
    spool = sl[sl["asset"] == "pooled"]
    ens_money = by_h(spool, "d_cer", "period", mu="hz_ens", var="naive", period="dev").iloc[0]
    w("**Returns: nothing reliable at any horizon, and a hint at 3 to 5 years that cannot be "
      "checked.** Up to 12 months the pooled out-of-sample R squared against the historical mean "
      f"is {rng(pooled['dev'], s12)} in 1990 to 2014 and {rng(pooled['after'], s12)} after 2015 "
      f"across the {len(rmods)} forecasters: noise around zero. In 1990 to 2014 it rises with the "
      "horizon for two of them ("
      + "; ".join(f"`{m}` " + ", ".join(f"{_fmt(pooled['dev'].loc[m, h], 'pct')} at {HL[h]}" for h in lng)
                  for m in ("hz_ridge", "hz_ens") if m in pooled["dev"].index)
      + f"), mostly from equities, but that rests on about {ni[lng].min().min():.0f} to "
      f"{ni[lng].max().max():.0f} non-overlapping windows, the years after 2015 are too few to "
      "confirm it, and a sleeve trading on `hz_ens` at those horizons earned "
      + " and ".join(f"{_fmt(ens_money[h], 'pct2')} a year at {HL[h]}" for h in lng)
      + " over the baseline. Cells where a forecaster beat the mean in both periods: "
      + "; ".join(f"`{m}` {a} of {b}" for m, (a, b) in nb.items())
      + " (one in four would be expected from coin flips).\n")
    pv = {k: {p: by_h(vs, "skill", "model", k=k, asset="pooled", period=p) for p in PER} for k in (21, 1)}
    pw = {k: {p: by_h(vs, "win", "model", k=k, asset="pooled", period=p) for p in PER} for k in (21, 1)}
    pc = {k: {p: by_h(cs, "skill", "model", k=k, asset="pooled", period=p) for p in PER} for k in (21, 1)}
    sh = [h for h in (1, 3, 6, 12) if h in pv[1]["dev"].columns]
    lg = [h for h in (36, 60) if h in pv[1]["dev"].columns]
    w("**Volatility is predictable out to about a year, and much less beyond.** With daily data the "
      f"blended forecast (`half`) removes {_fmt(pv[1]['dev'].loc['half', sh].min(), 'pct0')} to "
      f"{_fmt(pv[1]['dev'].loc['half', sh].max(), 'pct0')} of the naive forecast's loss at 1 to 12 "
      f"months in 1990 to 2014 and {_fmt(pv[1]['dev'].loc['half', lg].min(), 'pct0')} to "
      f"{_fmt(pv[1]['dev'].loc['half', lg].max(), 'pct0')} at 3 to 5 years. After 2015 the average "
      "gain disappears at most horizons "
      f"({_fmt(pv[1]['after'].loc['half', sh].min(), 'pct0')} to "
      f"{_fmt(pv[1]['after'].loc['half', sh].max(), 'pct0')} at 1 to 12 months), yet the forecast "
      "still has the lower loss in "
      f"{_fmt(pw[1]['after'].loc['half', sh].min(), 'pct0')} to "
      f"{_fmt(pw[1]['after'].loc['half', sh].max(), 'pct0')} of months: the average is dominated by the "
      "windows that contain the crash of March 2020, when every forecast built on recent calm was "
      "far too low.\n")
    ch = [h for h in pc[1]["dev"].columns]
    w("**Correlations are modestly predictable at every horizon up to about 3 years.** `half` "
      f"removes {_fmt(pc[1]['dev'].loc['half', ch].min(), 'pct0')} to "
      f"{_fmt(pc[1]['dev'].loc['half', ch].max(), 'pct0')} of the naive squared error in 1990 to 2014 "
      f"and {_fmt(pc[1]['after'].loc['half', ch].min(), 'pct0')} to "
      f"{_fmt(pc[1]['after'].loc['half', ch].max(), 'pct0')} after 2015 (daily measure), "
      "and unlike volatility this held up in both periods.\n")
    sp = sl[(sl["asset"] == "pooled")]
    def cer(mu, var, per):
        return by_h(sp, "d_cer", "period", mu=mu, var=var, period=per).iloc[0]
    both_dev, both_aft = cer("sig_combo", "half", "dev"), cer("sig_combo", "half", "after")
    w("**Combining the two does not make the whole model better.** A sleeve that sizes each "
      "position with a return forecast and a volatility forecast gained "
      f"{_fmt(both_dev.min(), 'pct')} to {_fmt(both_dev.max(), 'pct')} a year in certainty-equivalent "
      f"return over the baseline in 1990 to 2014 and {_fmt(both_aft.min(), 'pct')} to "
      f"{_fmt(both_aft.max(), 'pct')} after 2015, depending on the horizon (`sig_combo` with `half`): "
      "small, and with the sign changing between periods and horizons. The volatility regression "
      "`har` added "
      f"{_fmt(cer('hist_mean', 'har', 'dev')[[h for h in HORIZONS if 3 <= h <= 60]].min(), 'pct')} to "
      f"{_fmt(cer('hist_mean', 'har', 'dev')[[h for h in HORIZONS if 3 <= h <= 60]].max(), 'pct')} a year "
      "at 3 months to 5 years in 1990 to 2014 and "
      f"{_fmt(cer('hist_mean', 'har', 'after').min(), 'pct')} to "
      f"{_fmt(cer('hist_mean', 'har', 'after').max(), 'pct')} after 2015. What the study supports "
      "is a split: keep expected returns at the historical mean at every horizon, and treat risk "
      "as the forecastable part, mainly at short horizons.\n")

    # ------------------------------------------------------------------ design
    w("## What was run\n")
    w(f"Horizons: {', '.join(HL[h] for h in HORIZONS)}. Forecasts are made at every month end; "
      "a window is scored once it has closed. Two periods are shown side by side: **1990 to 2014** "
      "(origins from 1990-01 whose window closed by 2014-12, the years this package's models were "
      "developed on) and **2015 on** (the spent holdout years). Neither is a test. A cell with "
      f"fewer than {MIN_INDEP:g} non-overlapping windows is left blank; at 10 years that is every "
      "cell after 2015 and most cells before.\n")
    w("| quantity | forecasters | the bar | skill |\n|---|---|---|---|\n"
      "| return | `sig_combo` (mean of nine single-signal regressions: carry, value, momentum, "
      "trend, reversal, term spread, default spread, CAPE yield, bill rate) and the frozen learners "
      "re-aimed at each horizon under new names (`hz_ridge`, `hz_gbm`, `hz_ens`) | the expanding "
      "historical mean | out-of-sample R squared; Clark-West test |\n"
      "| volatility | `trail_3/12/60` (last 3, 12, 60 months), `half` (mean of `trail_12` and the "
      "long-run level), `har` (log regression on the three trailing levels) | the long-run "
      "level of the asset's whole history | share of QLIKE loss removed; Diebold-Mariano test |\n"
      "| correlation | `trail_12/36/60`, `half` (mean of `trail_36` and the long-run level) | "
      "the pair's long-run correlation | share of squared error removed; Diebold-Mariano test |\n")
    eq = meta.get("copies_equal_frozen_at_h12", {})
    if eq:
        w("Check: at 12 months the re-aimed learners must reproduce the frozen forecasts. Largest "
          "absolute difference: "
          + ", ".join(f"`{k}` vs `{v['frozen']}` {v['max_abs_diff']:.1e} over {v['rows']} rows"
                      for k, v in eq.items()) + ".\n")
    w("Risk is measured two ways. The **21-day measure** is the package's target definition "
      "(overlapping 21-day returns; price sources agree at that span) but it is very noisy in "
      "short windows, which pulls measured skill toward zero. The **daily measure** is precise "
      "but depends on how each source is priced. Read them together.\n")

    # ------------------------------------------------------------------ returns
    w("## 1. Returns\n")
    for p in PER:
        w(f"Pooled out-of-sample R squared against the historical mean, {PER[p]}:\n")
        w(md_table(pooled[p], "pct", "model") + "\n")
        w(f"Clark-West one-sided p-value (model better), {PER[p]}; not adjusted for the "
          f"{pooled_p[p].notna().sum().sum()} cells:\n")
        w(md_table(pooled_p[p], "p", "model") + "\n")
    top = "hz_ens" if "hz_ens" in rmods else rmods[-1]
    heat([(f"sig_combo, {PER['dev']}", r_dev["sig_combo"]), (f"sig_combo, {PER['after']}", r_aft["sig_combo"]),
          (f"{top}, {PER['dev']}", r_dev[top]), (f"{top}, {PER['after']}", r_aft[top])],
         FIGURES / "returns_r2.png", 0.3, note="Out-of-sample R squared against the historical mean.")
    w("![Return predictability by asset and horizon](figures/horizons/returns_r2.png)\n")
    w("Horizons at which a forecaster beat the historical mean in **both** periods, by asset "
      "(a description; with this many cells some agreement is expected by chance):\n")
    pl = pd.DataFrame({m: persist_list(r_dev[m], r_aft[m]) for m in rmods})
    w("| asset | " + " | ".join(f"`{m}`" for m in rmods) + " |\n|" + "---|" * (len(rmods) + 1))
    for a, r in pl.iterrows():
        w(f"| {a} | " + " | ".join(r) + " |")
    w("")
    sig = rs[(rs["asset"] == "pooled") & rs["model"].str.startswith("sig_")
             & (rs["model"] != "sig_combo")]
    w("\nThe single signals, pooled R squared, " + PER["dev"] + ":\n")
    w(md_table(by_h(sig, "r2", "model", period="dev"), "pct", "signal") + "\n")
    w("and " + PER["after"] + ":\n")
    w(md_table(by_h(sig, "r2", "model", period="after"), "pct", "signal") + "\n")

    # ------------------------------------------------------------------ risk
    w("## 2. Volatility\n")
    for k, name in ((1, "daily measure"), (21, "21-day measure")):
        for p in PER:
            w(f"Share of the naive forecast's QLIKE loss removed, pooled, {name}, {PER[p]}:\n")
            w(md_table(pv[k][p], "pct0", "model") + "\n")
        w(f"Share of months on which `half` had the lower loss than naive, {name}:\n")
        w(md_table(pd.DataFrame({PER[p]: pw[k][p].loc["half"] for p in PER}).T, "pct0", "period") + "\n")
    heat([(f"daily, {PER['dev']}", v_dev[1]), (f"daily, {PER['after']}", v_aft[1]),
          (f"21-day, {PER['dev']}", v_dev[21]), (f"21-day, {PER['after']}", v_aft[21])],
         FIGURES / "vol_skill.png", 0.6, note="Volatility forecaster `half`; share of QLIKE loss removed.")
    w("![Volatility predictability by asset and horizon](figures/horizons/vol_skill.png)\n")
    w("`published` is the selected model `combo_cov_main`, scored here at its own 12-month "
      "horizon for reference. `har` is unstable beyond 2 years: there are too few closed windows "
      "to fit it.\n")

    w("## 3. Correlations\n")
    for k, name in ((1, "daily measure"), (21, "21-day measure")):
        for p in PER:
            w(f"Share of the naive squared error removed, pooled over pairs, {name}, {PER[p]}:\n")
            w(md_table(pc[k][p], "pct0", "model") + "\n")
    heat([(f"daily, {PER['dev']}", c_dev[1]), (f"daily, {PER['after']}", c_aft[1]),
          (f"21-day, {PER['dev']}", c_dev[21]), (f"21-day, {PER['after']}", c_aft[21])],
         FIGURES / "corr_skill.png", 0.5,
         note="Correlation forecaster `half`; average over the pairs that contain the asset.")
    w("![Correlation predictability by asset and horizon](figures/horizons/corr_skill.png)\n")

    # ------------------------------------------------------------------ value
    w("## 4. Is it worth money?\n")
    w(f"Single-asset sleeves against cash (risk aversion {meta['value']['gamma']:g}, weight between "
      f"0 and {meta['value']['cap']:g}, the package's trading costs and financing spread). Each "
      "forecast is held for its own horizon. Gain in annualized certainty-equivalent return over "
      "the baseline (historical mean, long-run volatility), equal-weight average of the 12 "
      "sleeves:\n")
    for p in PER:
        d = sp[sp["period"] == p]
        w(f"{PER[p]}:\n")
        w(md_table(by_h(d, "d_cer", ["mu", "var"]), "pct", "return / volatility forecast") + "\n")
        w("t-statistic of the gain:\n")
        w(md_table(by_h(d, "t", ["mu", "var"]), "f2", "return / volatility forecast") + "\n")
    w("Long-only minimum-variance portfolio built from forecast volatilities and correlations; "
      "realized volatility relative to the same portfolio built from the naive forecasts (below "
      "1: the forecasts lowered risk):\n")
    for p in PER:
        w(f"{PER[p]}:\n")
        w(md_table(by_h(mv[mv["period"] == p], "ratio", ["vol", "corr"]), "r3",
                   "volatility / correlation forecast") + "\n")

    # ------------------------------------------------------------------ map
    w("## 5. The map by asset\n")
    w("Horizons with positive skill in both periods, per asset. Nothing is picked from this "
      "table; it shows where the two periods agree.\n")
    mp = pd.DataFrame({
        "return (`sig_combo`)": persist_list(r_dev["sig_combo"], r_aft["sig_combo"]),
        "volatility (`half`, daily)": persist_list(v_dev[1], v_aft[1]),
        "volatility (`half`, 21-day)": persist_list(v_dev[21], v_aft[21]),
        "correlation (`half`, daily)": persist_list(c_dev[1], c_aft[1]),
        "correlation (`half`, 21-day)": persist_list(c_dev[21], c_aft[21])})
    w("| asset | " + " | ".join(mp.columns) + " |\n|" + "---|" * (mp.shape[1] + 1))
    for a, r in mp.iterrows():
        w(f"| {a} | " + " | ".join(r) + " |")
    w("")

    w("\n## Limits\n")
    w("- Both holdouts were spent before this study. The 2015 on columns are a description of "
      "years already seen, and the forecasters were chosen by someone who knew the literature "
      "on those years.\n"
      "- Nothing is adjusted for the number of cells. With 12 assets, 8 horizons, several "
      "forecasters and two periods, isolated good cells are expected by chance.\n"
      "- Long horizons have almost no independent evidence: 1990 to 2014 holds two and a half "
      "non-overlapping 10-year windows at best, and fewer once a model needs training data.\n"
      "- QLIKE punishes forecasts that are too low much more than ones that are too high, so one "
      "crisis that follows a calm stretch (March 2020) dominates a 12-year average. The win "
      "rates are shown beside it for that reason.\n"
      "- The daily risk measure shifts level where an asset's price source changes "
      "(`reports/data/horizon_evidence.md`); the 21-day measure does not, but is noisy.\n"
      "- The sleeves hold each forecast to its horizon. An investor who re-forecasts every month "
      "would trade more and could do better or worse.\n"
      "- The frozen models and the published 12-month forecast are unchanged. A model at a new "
      "horizon would have to be declared now and judged on forecasts made after "
      f"`FROZEN_AT`; at 1 month that evidence accumulates 12 times faster than at 12.\n")
    REPORT.write_text("\n".join(L))
    return str(REPORT)


if __name__ == "__main__":
    print(build())
    sys.exit(0)
