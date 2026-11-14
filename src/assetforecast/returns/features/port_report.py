"""Old-versus-new comparison of the feature panel (assetpred's archived panel
against the one rebuilt on the 12-asset return panel).

    python -m assetforecast.returns.features.port_report

writes ``reports/data/feature_port.md``. Every table is computed here from
``paths.FEATURES``, the archived ``results/archive/assetpred/data`` files and
``data/processed/segments.csv``; nothing is typed by hand.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ... import paths
from .. import config
from .build import coverage

# assetpred id -> package id (us_small is new and has no counterpart)
OLD_TO_NEW = {"us_eq": "us_large", "dev_eq": "intl_dev", "em_eq": "em"}
JUMP_WINDOW = 24  # months either side of a daily-source change


def _load():
    d = paths.ARCHIVE_AP / "data"
    old = pd.read_parquet(d / "features.parquet")
    old_meta = pd.read_csv(d / "feature_meta.csv", keep_default_na=False)
    new = pd.read_parquet(paths.FEATURES)
    meta = pd.read_csv(paths.FEATURE_META, keep_default_na=False)
    old = old.rename(index=OLD_TO_NEW, level="asset")
    return old, old_meta, new, meta


def _fmt(df: pd.DataFrame, digits: int = 3) -> str:
    def cell(v):
        if isinstance(v, (float, np.floating)):
            return "" if not np.isfinite(v) else f"{v:.{digits}f}"
        return str(v)
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join([df.index.name or ""] + cols) + " |",
             "|" + "---|" * (len(cols) + 1)]
    for i, r in df.iterrows():
        lines.append("| " + " | ".join([str(i)] + [cell(v) for v in r]) + " |")
    return "\n".join(lines)


def global_equality(old, new, meta) -> pd.DataFrame:
    """Predictor-only global features (all but vrp): max |new - old| and NaN mismatches on shared dates."""
    names = meta.loc[(meta.scope == "global") & (meta.family != "price") & (meta.name != "vrp"), "name"]
    on, oo = new.groupby(level="date").first(), old.groupby(level="date").first()
    d = on.index.intersection(oo.index)
    rows = {}
    for c in names:
        a, b = on.loc[d, c], oo.loc[d, c]
        rows[c] = {"max_abs_diff": float((a - b).abs().max()) if (a.notna() & b.notna()).any() else np.nan,
                   "nan_mismatch": int((a.isna() != b.isna()).sum()), "n_shared": int((a.notna() & b.notna()).sum())}
    return pd.DataFrame(rows).T.rename_axis("feature")


def _pair_stats(a: pd.Series, b: pd.Series) -> dict:
    ok = a.notna() & b.notna()
    n = int(ok.sum())
    if n < 3:
        return {"n": n, "corr": np.nan, "mean_diff": np.nan, "mad": np.nan, "sd_ratio": np.nan}
    x, y = a[ok], b[ok]
    sd = y.std()
    return {"n": n, "corr": float(x.corr(y)) if x.std() > 0 and sd > 0 else np.nan,
            "mean_diff": float((x - y).mean()), "mad": float((x - y).abs().mean()),
            "sd_ratio": float(x.std() / sd) if sd > 0 else np.nan}


def asset_feature_stats(old, new, meta) -> pd.DataFrame:
    """Per (asset-scope feature, asset): stats of new vs old over shared (date, asset) rows."""
    idx = new.index.intersection(old.index)
    rows = []
    for c in meta.loc[meta.scope == "asset", "name"]:
        a, b = new.loc[idx, c], old.loc[idx, c]
        for asset in config.ASSETS:
            if asset == "us_small":
                continue
            m = idx.get_level_values("asset") == asset
            rows.append({"feature": c, "asset": asset, **_pair_stats(a[m], b[m])})
    return pd.DataFrame(rows)


def global_price_stats(old, new, meta) -> pd.DataFrame:
    names = list(meta.loc[(meta.scope == "global") & ((meta.family == "price") | (meta.name == "vrp")), "name"])
    on, oo = new.groupby(level="date").first(), old.groupby(level="date").first()
    d = on.index.intersection(oo.index)
    return pd.DataFrame({c: _pair_stats(on.loc[d, c], oo.loc[d, c]) for c in names}).T.rename_axis("feature")


def return_stats(old_returns: pd.DataFrame, old_cash: pd.Series, panel) -> pd.DataFrame:
    """Monthly log excess returns, new panel vs archived, per asset."""
    from ..schema import excess_log_returns
    from ..inputs import excess_returns
    xo = excess_log_returns(old_returns, old_cash).rename(columns=OLD_TO_NEW)
    xn = excess_returns(panel)
    rows = {}
    for a in config.ASSETS:
        if a not in xo:
            continue
        s = _pair_stats(xn[a], xo[a])
        s["new_first"] = xn[a].first_valid_index().strftime("%Y-%m")
        s["old_first"] = xo[a].first_valid_index().strftime("%Y-%m")
        rows[a] = s
    return pd.DataFrame(rows).T.rename_axis("asset")[["old_first", "new_first", "n", "corr", "sd_ratio", "mad"]]


def daily_source_changes(panel, segments: pd.DataFrame) -> pd.DataFrame:
    """Every change of daily source inside the return panel, with the ratio of
    daily realised vol to monthly-return vol in the ``JUMP_WINDOW`` months on
    either side (a smoothed NAV has a low ratio) and the median ``vol_d_1m``
    feature before and after."""
    from ..inputs import excess_returns, observed_through
    t = observed_through(panel)
    xm = excess_returns(panel)
    daily = np.log1p(panel.daily.loc[:t])
    feats = pd.read_parquet(paths.FEATURES, columns=["vol_d_1m"])["vol_d_1m"]
    seg = segments[segments.asset.isin(config.ASSETS)].copy()
    seg["start"] = pd.to_datetime(seg["start"])
    rows = []
    for a, g in seg.groupby("asset", sort=False):
        g = g.sort_values("start").reset_index(drop=True)
        for i in range(1, len(g)):
            prev, cur = g.iloc[i - 1], g.iloc[i]
            if prev["freq"] != "D" or cur["freq"] != "D":
                continue
            b = cur["start"] + pd.offsets.MonthEnd(0)  # month of the change
            if b < pd.Timestamp(config.RETURN_START) or b > t:
                continue

            def side(lo, hi):
                r = daily[a].loc[lo:hi].dropna()
                m = xm[a].loc[lo + pd.offsets.MonthEnd(0):hi].dropna()
                vd = np.sqrt(252 * (r ** 2).mean())
                vm = m.std() * np.sqrt(12)
                f = feats.xs(a, level="asset").loc[lo:hi]
                return vd / vm if vm > 0 else np.nan, f.median()

            before = side(b - pd.offsets.MonthEnd(JUMP_WINDOW), b - pd.offsets.MonthEnd(1))
            after = side(b + pd.offsets.MonthBegin(1), b + pd.offsets.MonthEnd(JUMP_WINDOW))
            rows.append({"asset": a, "month": b.strftime("%Y-%m"), "from": prev["source"], "to": cur["source"],
                         "dvol/mvol before": before[0], "dvol/mvol after": after[0],
                         "vol_d_1m median before": before[1], "vol_d_1m median after": after[1],
                         "jump (after/before)": after[1] / before[1] if before[1] else np.nan})
    return pd.DataFrame(rows).set_index("asset")


def first_dates(old, new, cols=("carry", "value", "vol_d_1m", "mom_12_1", "dd_max")) -> pd.DataFrame:
    out = {}
    for a in config.ASSETS:
        r = {}
        for c in cols:
            for tag, f in (("old", old), ("new", new)):
                s = f[c].xs(a, level="asset").dropna() if a in f.index.get_level_values("asset") else pd.Series(dtype=float)
                r[f"{c} {tag}"] = s.index.min().strftime("%Y-%m") if len(s) else "-"
        out[a] = r
    return pd.DataFrame(out).T.rename_axis("asset")


def render() -> str:
    from ..inputs import load_return_panel
    old, old_meta, new, meta = _load()
    d = paths.ARCHIVE_AP / "data"
    panel = load_return_panel()
    seg = pd.read_csv(paths.PROCESSED / "segments.csv")
    eq = global_equality(old, new, meta)
    st = asset_feature_stats(old, new, meta)
    gp = global_price_stats(old, new, meta)
    gp["n"] = gp["n"].astype(int).astype(str)
    rs = return_stats(pd.read_parquet(d / "asset_returns.parquet"), pd.read_parquet(d / "cash.parquet")["cash"], panel)
    jumps = daily_source_changes(panel, seg)
    cov = coverage(new, dates=("1990-01-31", "2014-12-31"))

    corr = st.pivot(index="feature", columns="asset", values="corr").reindex(
        index=list(meta.loc[meta.scope == "asset", "name"]), columns=[a for a in config.ASSETS if a != "us_small"])
    corr.index.name = "feature"
    mdiff = st.pivot(index="feature", columns="asset", values="mean_diff").reindex(index=corr.index, columns=corr.columns)
    mdiff.index.name = "feature"
    worst = st.dropna(subset=["corr"]).sort_values("corr").head(15).set_index("feature")
    n_bad = int(((eq.max_abs_diff > 1e-12) | (eq.nan_mismatch > 0)).sum())

    out = [
        "# Feature panel: assetpred's archived panel vs the rebuild on the 12-asset panel",
        "",
        "Generated by `python -m assetforecast.returns.features.port_report`. Old = "
        "`results/archive/assetpred/data/features.parquet` (11 assets, old ids mapped "
        f"{', '.join(f'{k}->{v}' for k, v in OLD_TO_NEW.items())}); new = `data/processed/{paths.FEATURES.name}`. "
        "Comparisons use shared (date, asset) rows or shared dates. `us_small` is new and has no counterpart.",
        "",
        f"Old panel: {len(old)} rows, {old.index.get_level_values('date').min().date()}..{old.index.get_level_values('date').max().date()}. "
        f"New panel: {len(new)} rows, {new.index.get_level_values('date').min().date()}..{new.index.get_level_values('date').max().date()}.",
        "",
        "## 1. Features that depend only on predictors",
        "",
        f"{len(eq)} global valuation/macro/sentiment features (all but `vrp`): {len(eq) - n_bad} equal the archived values "
        f"to 1e-12 on shared dates with identical missing patterns; {n_bad} do not. Largest difference: "
        f"{eq.max_abs_diff.max():.2e}.",
        "",
        "## 2. The asset returns underneath (monthly log excess return, new vs old)",
        "",
        "Every price feature inherits these differences. `sd_ratio` = new sd / old sd; `mad` = mean |new - old|.",
        "",
        _fmt(rs),
        "",
        "## 3. Asset-scope features: correlation of new with old, per asset",
        "",
        "Over shared (date, asset) rows. Includes the slots `carry` and `value`: their formulas are unchanged except "
        "`ust_int` (7-year zero instead of 5-year), so for other assets any difference comes only from which rows exist.",
        "",
        _fmt(corr),
        "",
        "## 4. Asset-scope features: mean level difference (new - old), per asset",
        "",
        _fmt(mdiff, 4),
        "",
        "The 15 lowest (feature, asset) correlations:",
        "",
        _fmt(worst[["asset", "n", "corr", "mean_diff", "sd_ratio"]]),
        "",
        "## 5. Global price-derived features and `vrp`",
        "",
        "These read `us_large` where assetpred read the CRSP value-weighted market (`us_eq`), and `ust_int` "
        "(now a 7-10y ladder, was a 5y par bond); the breadth and cross-sectional ranks change with the universe.",
        "",
        _fmt(gp),
        "",
        "## 6. First non-missing month, old vs new",
        "",
        _fmt(first_dates(old, new)),
        "",
        "## 7. Daily-volatility features where the daily source changes",
        "",
        f"Each change of daily source inside the return panel (`data/processed/segments.csv`). `dvol/mvol` = "
        f"annualised daily realised vol / annualised sd of monthly excess returns over the {JUMP_WINDOW} months "
        "before or after the month of the change (smoothed NAVs and stale quotes give low values; non-synchronous "
        "closes high ones). `jump` = median `vol_d_1m` after / before; it mixes the source change with genuine "
        "changes in volatility. Not corrected: that would change a frozen definition.",
        "",
        _fmt(jumps),
        "",
        "## 8. Coverage: non-missing features per asset (of 84)",
        "",
        _fmt(cov.rename_axis("asset"), 0),
        "",
        "Blank: the asset has no return that month (no row).",
        "",
    ]
    return "\n".join(out)


def main(argv=None) -> None:
    out = paths.REPORTS / "data" / "feature_port.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render())
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
