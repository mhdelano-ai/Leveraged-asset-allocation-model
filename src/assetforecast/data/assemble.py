"""Assemble the per-track asset and predictor files into the panel that ``load_panel`` reads.

Run ``python -m assetforecast.data.assemble`` (idempotent). It reads whatever exists in
``data/processed/assets/`` and ``data/processed/predictors/`` and writes, in
``data/processed/``:

* ``panel_daily.parquet``   wide simple returns on the US equity trading calendar
  (the dates of ``assets/cash_daily.parquet``) from the first date any asset has
  daily data. A return dated off that calendar is COMPOUNDED into the next calendar
  day; a calendar day without a return is NaN (the next return spans the gap).
* ``panel_monthly.parquet`` calendar month-end index. Months fully covered by an
  asset's daily data are compounded daily returns; earlier months come from the
  track's monthly file. Only complete months appear.
* ``cash_daily.parquet``, ``cash_monthly.parquet`` (column ``cash``).
* ``tiers_monthly.parquet`` tier letter per asset-month (worst tier of the month's
  daily sources in daily eras).
* ``segments.csv`` provenance of every spliced segment, derived from the ``source``
  column of the files, with notes from the tracks.
* ``predictors.parquet`` and ``predictors_catalog.csv``: all predictor files in
  standard units (rates, yields, spreads, breakevens, vols as decimals).

It also writes ``reports/data/panel.md`` and ``reports/data/horizon_evidence.md``
(``--no-evidence`` skips the latter).
"""
from __future__ import annotations

import argparse
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.tseries.holiday import (AbstractHolidayCalendar, GoodFriday, Holiday, USLaborDay,
                                    USMartinLutherKingJr, USMemorialDay, USPresidentsDay,
                                    USThanksgivingDay, nearest_workday, sunday_to_monday)

from ..paths import ASSET_DIR, PREDICTOR_DIR, PROCESSED, RAW, REPORTS
from ..universe import ASSET_IDS, BY_ID

TIER_ORDER = "ABCD"
SEP = " ||| "
SEGMENT_COLUMNS = ["asset", "start", "end", "freq", "source", "tier", "notes"]
PREDICTOR_COLUMNS = ["obs_date", "available_date", "series", "value"]
CATALOG_COLUMNS = ["series", "group", "unit", "freq", "start", "end", "live", "notes"]
HORIZONS = [1, 2, 3, 5, 10, 21, 63]
PAIRS = [("us_large", "intl_dev"), ("us_large", "em"), ("us_large", "ust_long"),
         ("us_large", "hy_credit"), ("us_large", "reits"), ("ust_int", "ig_credit"),
         ("ust_long", "tips"), ("gold", "commodities"), ("us_large", "us_small"),
         # extra pairs where the pricing convention also differs
         ("intl_dev", "em"), ("ig_credit", "hy_credit"), ("ust_int", "ust_long"),
         ("hy_credit", "reits"), ("intl_dev", "ust_long")]
LIVE_DAYS = {"D": 60, "M": 100}


# ============================================================== small helpers
def _dt_index(idx) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(idx).as_unit("ns").normalize()


def worst_tier(tiers) -> str | float:
    t = [x for x in tiers if isinstance(x, str) and x in TIER_ORDER]
    return max(t, key=TIER_ORDER.index) if t else np.nan


def _runs(labels: pd.Series) -> list[tuple[object, pd.Timestamp, pd.Timestamp, int]]:
    """Contiguous runs of equal non-null labels: (label, first index, last index, n)."""
    s = labels.dropna()
    if s.empty:
        return []
    brk = s.ne(s.shift())
    gid = brk.cumsum()
    out = []
    for _, g in s.groupby(gid, sort=True):
        out.append((g.iloc[0], g.index[0], g.index[-1], len(g)))
    return out


class _NYSEHolidays(AbstractHolidayCalendar):
    """Approximate NYSE full-day holidays, used only to decide whether the month
    containing the last calendar date is complete."""
    rules = [
        Holiday("NewYearsDay", month=1, day=1, observance=sunday_to_monday),
        USMartinLutherKingJr, USPresidentsDay, GoodFriday, USMemorialDay,
        Holiday("Juneteenth", month=6, day=19, start_date="2022-01-01", observance=nearest_workday),
        Holiday("July4th", month=7, day=4, observance=nearest_workday),
        USLaborDay, USThanksgivingDay,
        Holiday("Christmas", month=12, day=25, observance=nearest_workday),
    ]


def month_is_complete(last_date: pd.Timestamp) -> bool:
    """True if ``last_date`` is the last NYSE trading day of its month."""
    last_date = pd.Timestamp(last_date)
    me = last_date + pd.offsets.MonthEnd(0)
    rest = pd.bdate_range(last_date + pd.Timedelta(days=1), me)
    if len(rest) == 0:
        return True
    hol = _NYSEHolidays().holidays(rest[0], rest[-1])
    return bool(rest.difference(hol).empty)


# ============================================================== inputs
@dataclass
class AssetFiles:
    asset: str
    daily: pd.DataFrame | None
    monthly: pd.DataFrame | None


def _read_asset(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    d = pd.read_parquet(path)
    d.index = _dt_index(d.index)
    d.index.name = "date"
    d = d[["ret", "source", "tier"]].copy()
    d["ret"] = d["ret"].astype(float)
    d["source"] = d["source"].astype(object)
    d["tier"] = d["tier"].astype(object)
    if not d.index.is_monotonic_increasing:
        d = d.sort_index()
    if d.index.has_duplicates:
        raise ValueError(f"{path.name}: duplicate dates")
    return d


def load_inputs(asset_dir: Path = ASSET_DIR) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, AssetFiles]]:
    cash_d = _read_asset(asset_dir / "cash_daily.parquet")
    cash_m = _read_asset(asset_dir / "cash_monthly.parquet")
    if cash_d is None or cash_m is None:
        raise FileNotFoundError("cash_daily/cash_monthly are required (they define the calendar)")
    assets = {}
    for a in ASSET_IDS:
        d = _read_asset(asset_dir / f"{a}_daily.parquet")
        m = _read_asset(asset_dir / f"{a}_monthly.parquet")
        if d is None and m is None:
            continue
        assets[a] = AssetFiles(a, d, m)
    return cash_d, cash_m, assets


# ============================================================== calendar folding
@dataclass
class FoldResult:
    frame: pd.DataFrame          # ret, source, tier on calendar dates (only dates with a return)
    n_input: int
    n_off_calendar: int          # returns dated off the calendar, compounded into next day
    n_dropped_after_end: int     # off-calendar returns after the last calendar day (dropped)
    n_boundary_moves: int        # folded returns that moved into a different month
    off_examples: list = field(default_factory=list)


def fold_to_calendar(d: pd.DataFrame, cal: pd.DatetimeIndex) -> FoldResult:
    """Compound every return dated off ``cal`` into the next calendar day."""
    idx = d.index
    on = idx.isin(cal)
    pos = cal.searchsorted(idx, side="left")
    beyond = pos >= len(cal)
    keep = ~beyond
    tgt = cal[np.minimum(pos, len(cal) - 1)]
    x = d.loc[keep].copy()
    x["target"] = tgt[keep]
    off_mask = ~on[keep]
    moved_month = (x.index.to_period("M") != x["target"].dt.to_period("M")) & off_mask
    lr = np.log1p(x["ret"])
    g = x.assign(lr=lr).groupby("target", sort=True)
    out = pd.DataFrame({
        "ret": np.expm1(g["lr"].sum(min_count=1)),
        "source": g["source"].last(),
        "tier": g["tier"].agg(worst_tier),
    })
    out.index.name = "date"
    ex = [(str(a.date()), str(b.date())) for a, b in zip(x.index[off_mask][:5], x["target"][off_mask][:5])]
    return FoldResult(out, len(d), int(off_mask.sum()), int(beyond.sum()), int(moved_month.sum()), ex)


# ============================================================== assembly
@dataclass
class Assembled:
    daily: pd.DataFrame
    monthly: pd.DataFrame
    cash_daily: pd.Series
    cash_monthly: pd.Series
    tiers: pd.DataFrame
    daily_source: pd.DataFrame           # source label per calendar day with a return
    daily_tier: pd.DataFrame
    monthly_source: pd.DataFrame         # source per month (track label pre-daily; last daily source)
    daily_months: dict[str, pd.PeriodIndex]
    folds: dict[str, FoldResult]
    checks: dict


def assemble(cash_d: pd.DataFrame, cash_m: pd.DataFrame, assets: dict[str, AssetFiles]) -> Assembled:
    cal = cash_d.index
    if not (cal.is_monotonic_increasing and cal.is_unique):
        raise ValueError("cash calendar must be strictly increasing")
    folds = {a: fold_to_calendar(f.daily, cal) for a, f in assets.items() if f.daily is not None}
    order = [a for a in ASSET_IDS if a in assets]

    # ---------------- daily
    first_daily = min(f.frame.index[0] for f in folds.values())
    dcal = cal[cal >= first_daily]
    daily = pd.DataFrame(index=dcal, columns=order, dtype=float)
    dsrc = pd.DataFrame(index=dcal, columns=order, dtype=object)
    dtier = pd.DataFrame(index=dcal, columns=order, dtype=object)
    for a, f in folds.items():
        daily[a] = f.frame["ret"].reindex(dcal)
        dsrc[a] = f.frame["source"].reindex(dcal)
        dtier[a] = f.frame["tier"].reindex(dcal)
    daily.index.name = "date"

    # ---------------- monthly
    per = cal.to_period("M")
    cal_s = pd.Series(cal, index=cal)
    m_first = cal_s.groupby(per).min()
    m_last = cal_s.groupby(per).max()
    last_month = per[-1] if month_is_complete(cal[-1]) else per[-1] - 1

    first_monthly = []
    for a in order:
        f = assets[a]
        if f.monthly is not None:
            first_monthly.append(f.monthly.index[0].to_period("M"))
        if a in folds:
            first_monthly.append(folds[a].frame.index[0].to_period("M"))
    mper = pd.period_range(min(first_monthly), last_month, freq="M")
    midx = mper.to_timestamp(how="end").normalize()

    monthly = pd.DataFrame(index=mper, columns=order, dtype=float)
    tiers = pd.DataFrame(index=mper, columns=order, dtype=object)
    msrc = pd.DataFrame(index=mper, columns=order, dtype=object)
    daily_months = {}
    mcheck = {}
    for a in order:
        f = assets[a]
        dm = pd.PeriodIndex([], freq="M")
        if a in folds:
            fr = folds[a].frame
            fd, ld = fr.index[0], fr.index[-1]
            ok = (m_first >= fd) & (m_last <= ld)
            dm = ok.index[ok.values]
            dm = dm[(dm >= mper[0]) & (dm <= last_month)]
            r = fr["ret"]
            comp = np.expm1(np.log1p(r).groupby(r.index.to_period("M")).sum(min_count=1)).reindex(dm)
            mt = fr["tier"].groupby(fr.index.to_period("M")).agg(worst_tier).reindex(dm)
            ms = fr["source"].groupby(fr.index.to_period("M")).last().reindex(dm)
            monthly.loc[dm, a] = comp.values
            tiers.loc[dm, a] = mt.values
            msrc.loc[dm, a] = ms.values
            # checks vs the track's own monthly file, and the effect of folding
            raw = f.daily["ret"]
            unf = np.expm1(np.log1p(raw).groupby(raw.index.to_period("M")).sum(min_count=1)).reindex(dm)
            chk = {"n_daily_months": len(dm)}
            if f.monthly is not None:
                tm = f.monthly["ret"].copy()
                tm.index = tm.index.to_period("M")
                common = dm.intersection(tm.index)
                d1 = (comp.reindex(common) - tm.reindex(common)).abs()
                d2 = (unf.reindex(common) - tm.reindex(common)).abs()
                chk.update(n_compared=len(common),
                           max_abs_vs_track=float(d1.max()) if len(d1) else np.nan,
                           worst_month_vs_track=str(d1.idxmax()) if len(d1) and d1.max() > 0 else "",
                           max_abs_unfolded_vs_track=float(d2.max()) if len(d2) else np.nan,
                           n_months_gt_1e8_vs_track=int((d1 > 1e-8).sum()))
            fe = (comp - unf).abs()
            chk.update(n_months_moved_by_folding=int((fe > 1e-12).sum()),
                       max_abs_folding_effect=float(fe.max()) if len(fe) else 0.0,
                       worst_month_folding=str(fe.idxmax()) if len(fe) and fe.max() > 1e-12 else "")
            mcheck[a] = chk
        daily_months[a] = dm
        if f.monthly is not None:
            tm = f.monthly.copy()
            tm.index = tm.index.to_period("M")
            first_d = dm[0] if len(dm) else None
            pre = tm.index[(tm.index >= mper[0]) & (tm.index <= last_month)]
            if first_d is not None:
                pre = pre[pre < first_d]
            monthly.loc[pre, a] = tm.loc[pre, "ret"].values
            tiers.loc[pre, a] = tm.loc[pre, "tier"].values
            msrc.loc[pre, a] = tm.loc[pre, "source"].values
    tiers = tiers.where(monthly.notna())
    for x in (monthly, tiers, msrc):
        x.index = midx
        x.index.name = "date"

    # ---------------- cash
    cash_daily = cash_d["ret"].reindex(dcal).rename("cash")
    cm = cash_m["ret"].copy()
    cm.index = cm.index.to_period("M")
    cash_monthly = cm.reindex(mper).rename("cash")
    cash_monthly.index = midx
    cash_monthly.index.name = "date"
    cd_comp = np.expm1(np.log1p(cash_d["ret"]).groupby(per).sum())
    common = cd_comp.index.intersection(cm.index)
    common = common[common <= last_month]
    cash_gap = float((cd_comp.reindex(common) - cm.reindex(common)).abs().max())

    checks = {"monthly_vs_track": mcheck, "cash_compound_gap": cash_gap,
              "last_month": str(last_month), "calendar_end": str(cal[-1].date())}
    return Assembled(daily, monthly, cash_daily, cash_monthly, tiers, dsrc, dtier, msrc,
                     daily_months, folds, checks)


# ============================================================== segments
def _norm_freq(f) -> str:
    f = str(f).strip().lower()
    return "D" if f.startswith("d") else "M" if f.startswith("m") else f.upper()


def _parse_date(x) -> pd.Timestamp | None:
    try:
        return pd.Timestamp(str(x).strip().strip("*"))
    except Exception:
        return None


def _md_segment_rows(path: Path, default_asset: str | None) -> list[dict]:
    """Parse the 'segments' table(s) of a track report (any table with start/end/source/tier)."""
    rows = []
    lines = path.read_text().splitlines()
    i = 0
    while i < len(lines):
        ln = lines[i].strip()
        if ln.startswith("|") and i + 1 < len(lines) and re.match(r"^\|[\s\-|:]+\|$", lines[i + 1].strip()):
            hdr = [h.strip().lower() for h in ln.strip("|").split("|")]
            j = i + 2
            body = []
            while j < len(lines) and lines[j].strip().startswith("|"):
                body.append(lines[j].strip().strip("|").split("|"))
                j += 1
            if {"start", "end", "source", "tier"} <= set(hdr) and ("freq" in hdr or "segment" in hdr):
                for b in body:
                    if len(b) > len(hdr) and hdr[-1] in ("notes", "why"):
                        b = b[: len(hdr) - 1] + ["|".join(b[len(hdr) - 1:])]   # '|' inside the notes
                    b = [c.strip() for c in b]
                    if len(b) != len(hdr):
                        continue
                    r = dict(zip(hdr, b))
                    asset = r.get("asset", default_asset)
                    fr = r.get("freq") or r.get("segment")
                    s, e = _parse_date(r["start"]), _parse_date(r["end"])
                    if asset is None or s is None or e is None:
                        continue
                    fq = _norm_freq(fr)
                    if fq == "M":
                        s, e = s + pd.offsets.MonthEnd(0), e + pd.offsets.MonthEnd(0)
                    rows.append(dict(asset=asset.strip("`"), start=s, end=e, freq=fq,
                                     source=re.sub(r"\*\*", "", r["source"]), tier=r["tier"].strip("*"),
                                     notes=r.get("notes") or r.get("why") or "", origin=path.name))
            i = j
        else:
            i += 1
    return rows


def track_segment_rows() -> pd.DataFrame:
    """Every segment row the tracks publish (CSV files first, then report tables)."""
    rows = []
    csvs = sorted((PROCESSED / "segments").glob("*.csv")) + sorted(RAW.glob("*/segments_*.csv"))
    for p in csvs:
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        for r in df.to_dict("records"):
            fq = _norm_freq(r["freq"])
            s, e = pd.Timestamp(r["start"]), pd.Timestamp(r["end"])
            rows.append(dict(asset=r["asset"], start=s, end=e, freq=fq, source=str(r["source"]),
                             tier=str(r["tier"]), notes=str(r.get("notes", "") or ""), origin=p.name))
    defaults = {"reits.md": "reits", "tips.md": "tips", "cash_macro.md": "cash"}
    for p in sorted((REPORTS / "data").glob("*.md")):
        if p.name in ("panel.md", "horizon_evidence.md"):
            continue
        rows += _md_segment_rows(p, defaults.get(p.name))
    df = pd.DataFrame(rows, columns=["asset", "start", "end", "freq", "source", "tier", "notes", "origin"])
    return df


# Overlap statistics quoted from the track reports where the track's own segment notes lack them
# (ARCHITECTURE: splice checks must be recorded in segments.csv notes). Keyed by (asset, source prefix).
EXTRA_NOTES = {
    ("reits", "nareit"): "vs FRESX M 1986-12..1996-05: corr 0.948, vol ratio 0.985, TE 3.9%/yr (reits.md)",
    ("reits", "yahoo:FRESX"): "vs Nareit AER M 1986-12..1996-05: corr 0.948, vol ratio 0.985, TE 3.9%/yr; "
                              "daily AC1 0.12-0.32, ~12% zero days (stale NAV)",
    ("reits", "yahoo:VGSIX"): "vs Nareit AER M 1996-06..2004-09: corr 0.997, vol ratio 1.009, TE 1.1%/yr; "
                              "vs VNQ D 2004-09..2026-11: corr 0.994, vol ratio 1.022",
    ("reits", "yahoo:VNQ"): "vs Nareit AER M 2004-10..2026-10: corr 0.998, vol ratio 1.014, TE 1.5%/yr",
    ("gold", "bundesbank"): "vs IFS London month-end M 1971-09..1998-12: corr 0.989, vol ratio 1.022, TE 3.2%/yr",
    ("gold", "ft:GC.1"): "vs GLD D 2004-11..2026-11: corr 0.890, vol ratio 1.005; M corr 0.993, vol ratio 1.001; "
                         "vs IFS M 1999-2004: corr 0.985, vol ratio 0.999",
    ("gold", "ft:GLD"): "vs COMEX front month M 2004-2026: corr 0.993, vol ratio 1.001",
    ("commodities", "aqr:"): "AQR-EW vs BCOM TR M 1991-2025: corr 0.953, vol ratio 0.908 (0.88 1991-98, 0.75 "
                             "1999-2006, 0.97 2007-15, 0.95 2016-25)",
    ("commodities", "ft:BCOM (excess"): "BCOM ER + RF vs BCOMTR D 2009-2026: corr 1.000, vol ratio 1.000, TE 0.1%/yr",
    ("commodities", "ft:BCOMTR"): "DBC vs BCOM TR M 2006-2026: vol ratio 1.168 (DBC is energy-heavy)",
    ("tips", "yahoo:PRRIX"): "vs TIP 2004-2006 overlap: daily corr 0.940 (3-day 0.975), vol ratio 0.989; monthly "
                             "corr 0.989, vol ratio 0.973, TE 0.8%/yr",
    ("tips", "yahoo:TIP"): "vs VIPSX M 2004-2026: corr 0.994, vol ratio 0.987 (TIP = VIPSX x 1.013)",
    ("cash", "nber_m13002"): "NYC prime commercial paper minus 1.28pp; ret = yield/1200; model-based (cash_macro.md)",
    ("cash", "nber_m13029a"): "yield on 3-6m Treasury notes/certificates; ret = yield/1200 (cash_macro.md)",
    ("cash", "french_rf"): "Ken French RF (1m T-bill); daily = monthly RF spread geometrically over trading days",
    ("cash", "h15_tbill4w"): "H.15 4-week bill vs French RF M 2001-09..2026-09: mean diff +1.5bp/yr, corr 0.993",
}


def build_segments(A: Assembled, assets: dict[str, AssetFiles], cash_d: pd.DataFrame,
                   cash_m: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Segments derived from the source column of the files actually assembled."""
    track = track_segment_rows()
    issues = []
    derived = []

    def add(asset, s, e, freq, src, tier):
        derived.append(dict(asset=asset, start=s, end=e, freq=freq, source=src, tier=tier))

    for a in A.daily.columns:
        f = assets[a]
        dm = A.daily_months[a]
        if f.monthly is not None:
            tm = f.monthly.copy()
            tm.index = tm.index.to_period("M")
            pre = tm.index[tm.index <= A.monthly.index[-1].to_period("M")]
            if len(dm):
                pre = pre[pre < dm[0]]
            lab = tm.loc[pre, "source"] + SEP + tm.loc[pre, "tier"]
            lab.index = pre.to_timestamp(how="end").normalize()
            for l, s, e, n in _runs(lab):
                src, tier = l.split(SEP)
                add(a, s, e, "M", src, tier)
        if a in A.folds:
            fr = A.folds[a].frame
            lab = fr["source"] + SEP + fr["tier"]
            for l, s, e, n in _runs(lab):
                src, tier = l.split(SEP)
                add(a, s, e, "D", src, tier)
    # cash: monthly pre-daily runs and daily runs
    lab = cash_m["source"] + SEP + cash_m["tier"]
    lab = lab[lab.index < cash_d.index[0]]
    for l, s, e, n in _runs(lab):
        src, tier = l.split(SEP)
        add("cash", s, e, "M", src, tier)
    for l, s, e, n in _runs(cash_d["source"] + SEP + cash_d["tier"]):
        src, tier = l.split(SEP)
        add("cash", s, e, "D", src, tier)

    seg = pd.DataFrame(derived)
    notes = []
    for r in seg.itertuples():
        t = track[(track.asset == r.asset) & (track.freq == r.freq)]
        best, best_ov = None, pd.Timedelta(0)
        for tr in t.itertuples():
            ov = min(r.end, tr.end) - max(r.start, tr.start)
            exact_src = tr.source == r.source
            if ov >= pd.Timedelta(0) and (best is None or (exact_src, ov) > (best.source == r.source, best_ov)):
                best, best_ov = tr, ov
        if best is None:
            notes.append("")
            if r.asset != "cash":
                issues.append(f"{r.asset} {r.freq} segment {r.source!r} ({r.start.date()}..{r.end.date()}): "
                              "no matching segment row published by the track")
            continue
        n = best.notes if isinstance(best.notes, str) else ""
        if best.source != r.source:
            n = (n + f" [track label: {best.source}]").strip()
        tol = pd.Timedelta(days=5 if r.freq == "D" else 35)
        if abs(best.start - r.start) > tol or abs(best.end - r.end) > tol:
            msg = (f"{r.asset} {r.freq} {r.source!r}: file span {r.start.date()}..{r.end.date()} vs "
                   f"track row ({best.origin}) {best.start.date()}..{best.end.date()}")
            issues.append(msg)
        if best.tier != r.tier:
            issues.append(f"{r.asset} {r.freq} {r.source!r}: tier in file {r.tier} vs track row {best.tier}")
        notes.append(n)
    for i, r in enumerate(seg.itertuples()):
        for (a, pre), extra in EXTRA_NOTES.items():
            if r.asset == a and r.source.startswith(pre) and extra not in notes[i]:
                notes[i] = (notes[i] + "; " + extra).strip("; ")
    seg["notes"] = notes
    # how many daily-era monthly rows of the track file carry each source is irrelevant: the
    # panel recomputes them. Order: universe order, then cash; monthly before daily; by start.
    order = {a: i for i, a in enumerate(list(A.daily.columns) + ["cash"])}
    seg["_o"] = seg["asset"].map(order)
    seg = seg.sort_values(["_o", "start", "freq"]).drop(columns="_o").reset_index(drop=True)
    seg["start"] = seg["start"].dt.date.astype(str)
    seg["end"] = seg["end"].dt.date.astype(str)
    return seg[SEGMENT_COLUMNS], issues


# ============================================================== predictors
# file -> {original series: (new name, unit, scale, description)}
P = 0.01
PRED_SPEC: dict[str, dict[str, tuple[str, str, float, str]]] = {
    "treasury_curve": {
        "yield_2y": ("ust_cmt_yield_2y", "decimal", 1, "H.15 2y constant-maturity par yield"),
        "yield_5y": ("ust_cmt_yield_5y", "decimal", 1, "H.15 5y CMT par yield"),
        "yield_7y": ("ust_cmt_yield_7y", "decimal", 1, "H.15 7y CMT par yield"),
        "yield_10y": ("ust_cmt_yield_10y", "decimal", 1, "H.15 10y CMT par yield"),
        "yield_20y": ("ust_cmt_yield_20y", "decimal", 1, "H.15 20y CMT par yield; gap 1987-01..1993-09"),
        "yield_30y": ("ust_cmt_yield_30y", "decimal", 1, "H.15 30y CMT par yield"),
        "yield_ladder_ust_int": ("ust_ladder_yield_int", "decimal", 1,
                                 "mean GSW par yield of the 7-10y ladder (ust_int construction)"),
        "yield_ladder_ust_long": ("ust_ladder_yield_long", "decimal", 1,
                                  "mean GSW par yield of the 20-30y ladder (ust_long construction)"),
        "duration_ust_int": ("ust_ladder_duration_int", "years", 1,
                             "modified duration of the 7-10y par ladder"),
        "duration_ust_long": ("ust_ladder_duration_long", "years", 1,
                              "modified duration of the 20-30y par ladder; understates TLT's by ~2y after 2020"),
        "duration_ust_int_etf_est": ("ust_etf_duration_est_int", "years", 1,
                                     "IEF duration estimated by rolling 252-day regression (past data only)"),
        "duration_ust_long_etf_est": ("ust_etf_duration_est_long", "years", 1,
                                      "TLT duration estimated by rolling 252-day regression (past data only)"),
    },
    "tips_real_yields": {
        "real_yield_5y": ("tips_real_yield_5y", "decimal", P, "GSW TIPS 5y zero-coupon real yield (cont. comp.)"),
        "real_yield_10y": ("tips_real_yield_10y", "decimal", P, "GSW TIPS 10y zero-coupon real yield (cont. comp.)"),
        "breakeven_5y": ("tips_breakeven_5y", "decimal", P, "GSW 5y zero-coupon breakeven inflation"),
        "breakeven_10y": ("tips_breakeven_10y", "decimal", P, "GSW 10y zero-coupon breakeven inflation"),
    },
    "credit_spreads": {
        "gz_credit_spread_m": ("credit_gz_spread_m", "decimal", P,
                               "Gilchrist-Zakrajsek credit spread (plain average of bond spreads), monthly. "
                               "The companion excess bond premium (gz_excess_bond_premium_m) is EXCLUDED: it is "
                               "the residual of a full-sample regression (look-ahead)"),
        "gz_excess_bond_premium_m": (None, "", 0, "excluded: full-sample regression residual (look-ahead)"),
        "moodys_aaa_yield_m": ("credit_moodys_aaa_yield_m", "decimal", P, "Moody's seasoned Aaa yield, monthly avg; ends 2025-12"),
        "moodys_baa_yield_m": ("credit_moodys_baa_yield_m", "decimal", P, "Moody's seasoned Baa yield, monthly avg; ends 2025-12"),
        "moodys_baa_minus_aaa_m": ("credit_moodys_baa_aaa_spread_m", "decimal", P, "Baa minus Aaa, monthly avg; ends 2025-12"),
        "moodys_baa_minus_aaa_d": ("credit_moodys_baa_aaa_spread_d", "decimal", P, "Baa minus Aaa, daily H.15 1986-2016 (dead)"),
        "moodys_baa_minus_ust10y_m": ("credit_moodys_baa_ust10y_spread_m", "decimal", P, "Baa minus 10y CMT, monthly avg; ends 2025-12"),
        "moodys_baa_minus_ust10y_d": ("credit_moodys_baa_ust10y_spread_d", "decimal", P, "Baa minus 10y CMT, daily 1986-2016 (dead)"),
    },
    "macro": {
        "macro_cpi_level": ("macro_cpi_level", "index level (1982-84=100)", 1, "CPI-U NSA; pre-1913 wholesale-based reconstruction"),
        "macro_infl_vol_36m": ("macro_infl_vol_36m", "decimal (annualized)", P, "trailing 36m std of monthly log CPI change"),
        "macro_infl_yoy": ("macro_infl_yoy", "decimal (log change)", P, "12-month log change of CPI"),
        "macro_ip_yoy": ("macro_ip_yoy", "decimal (log change)", P, "12-month log change of industrial production (latest vintage, revised)"),
        "macro_long_yield": ("macro_long_yield", "decimal", P, "long yield, monthly avg (railroad bonds to 1918, US long bonds to 1953-03, 10y CMT after)"),
        "macro_policy_rate": ("macro_policy_rate", "decimal", P, "NY Fed discount rate to 1954-06, effective fed funds monthly avg after"),
        "macro_policy_rate_chg_12m": ("macro_policy_rate_chg_12m", "decimal (change)", P, "12-month change in policy rate"),
        "macro_policy_rate_chg_3m": ("macro_policy_rate_chg_3m", "decimal (change)", P, "3-month change in policy rate"),
        "macro_real_short_rate": ("macro_real_short_rate", "decimal", P, "short rate minus trailing YoY inflation"),
        "macro_sahm_gap": ("macro_sahm_gap", "decimal (unemployment-rate change)", P, "3m avg unemployment minus prior-12m min"),
        "macro_sahm_recession": ("macro_sahm_recession", "flag 0/1", 1, "1 if Sahm gap >= 0.5pp"),
        "macro_short_rate": ("macro_short_rate", "decimal", P, "short rate, monthly avg (CP-spread to 1919, 3-6m Treasury to 1933, 3m bill after)"),
        "macro_term_spread": ("macro_term_spread", "decimal", P, "long yield minus short rate"),
        "macro_unrate": ("macro_unrate", "decimal", P, "civilian unemployment rate, SA; 2025-10 missing"),
        "macro_unrate_chg_12m": ("macro_unrate_chg_12m", "decimal (change)", P, "12-month change in unemployment rate"),
        "macro_usd_broad_yoy": ("macro_usd_broad_yoy", "decimal (log change)", P, "12-month log change of the broad dollar index"),
    },
}
IMPLIED_UNITS = {"iv_skew_index": ("index level (not a vol)", "Cboe SKEW"),
                 "iv_vix_term_slope": ("ratio - 1", "VIX3M / VIX - 1"),
                 "iv_vix1y_term_slope": ("ratio - 1", "VIX1Y / VIX - 1"),
                 "iv_vvix": ("decimal vol (of VIX)", "Cboe VVIX"),
                 "iv_vxo": ("decimal vol", "old VIX methodology (OEX ATM, trading-day); dead 2021"),
                 "iv_move": ("decimal yield vol (bp vol / 1e4)", "ICE MOVE")}
GROUP_OF_FILE = {"treasury_curve": "treasury", "tips_real_yields": "tips", "credit_spreads": "credit",
                 "macro": "macro", "implied": "implied"}


def _implied_desc(series: str) -> str:
    """Description from the implied track's own spec table (offline import)."""
    try:
        from .sources import implied as imp
    except Exception:
        return ""
    if series in imp.BY_SERIES:
        sp = imp.BY_SERIES[series]
        return f"{sp.name}. {sp.notes}".strip().rstrip(".")
    if series in getattr(imp, "DERIVED_NOTES", {}):
        return imp.DERIVED_NOTES[series][0]
    return ""


def _infer_freq(obs: pd.Series) -> str:
    d = obs.sort_values().diff().dt.days.dropna()
    return "M" if len(d) and d.median() >= 25 else "D"


def build_predictors(ref_end: pd.Timestamp, pred_dir: Path = PREDICTOR_DIR):
    frames, cat, issues, conv = [], [], [], []
    for p in sorted(pred_dir.glob("*.parquet")):
        stem = p.stem
        group = GROUP_OF_FILE.get(stem, stem)
        df = pd.read_parquet(p)
        df = df[PREDICTOR_COLUMNS].copy()
        df["obs_date"] = _dt_index(df["obs_date"])
        df["available_date"] = _dt_index(df["available_date"])
        df["series"] = df["series"].astype(str)
        spec = PRED_SPEC.get(stem, {})
        for s, g in df.groupby("series", sort=True):
            if stem == "implied":
                unit, desc = IMPLIED_UNITS.get(s, ("decimal vol", ""))
                desc = _implied_desc(s) or desc
                new, scale = s, 1.0
            elif s in spec:
                new, unit, scale, desc = spec[s]
                if new is None:
                    conv.append((stem, s, "EXCLUDED", desc))
                    continue
            else:
                new = s if s.startswith(group) else f"{group}_{s}"
                unit, scale, desc = "native (not verified)", 1.0, "series not in the assembler's unit table"
                issues.append(f"{stem}:{s} has no unit entry; passed through in native units as {new}")
            g = g.dropna(subset=["value"]).copy()
            raw_q90 = float(g["value"].abs().quantile(0.9))
            g["value"] = g["value"].astype(float) * scale
            g["series"] = new
            frames.append(g)
            if scale != 1:
                conv.append((stem, s, new, f"x{scale:g} ({unit}); raw |x| q90 = {raw_q90:.3g}"))
            elif s != new:
                conv.append((stem, s, new, f"renamed, units unchanged ({unit})"))
            fq = _infer_freq(g["obs_date"])
            end = g["obs_date"].max()
            live = bool((ref_end - end).days <= LIVE_DAYS[fq])
            note = desc
            if s != new:
                note = f"{note}. Original name `{s}` in predictors/{p.name}" if note else f"Original name `{s}`"
            else:
                note = f"{note}. From predictors/{p.name}" if note else f"From predictors/{p.name}"
            if scale != 1:
                note += f"; converted from percent (x{scale:g})"
            cat.append(dict(series=new, group=group, unit=unit, freq=fq, start=str(g["obs_date"].min().date()),
                            end=str(end.date()), live=live, notes=note))
    pred = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=PREDICTOR_COLUMNS)
    pred = pred.sort_values(["available_date", "series", "obs_date"], kind="stable").reset_index(drop=True)
    catalog = pd.DataFrame(cat, columns=CATALOG_COLUMNS).sort_values(["group", "series"]).reset_index(drop=True)
    if catalog["series"].duplicated().any():
        raise ValueError(f"duplicate series names: {catalog.series[catalog.series.duplicated()].tolist()}")
    return pred, catalog, issues, conv


def predictor_unit_checks(pred: pd.DataFrame) -> dict:
    """Cross-file consistency checks that confirm the unit conversions."""
    w = pred.pivot_table(index="obs_date", columns="series", values="value", aggfunc="first")
    out = {}

    def mm(s):  # month-average
        return s.dropna().groupby(s.dropna().index.to_period("M")).mean()

    if {"macro_long_yield", "ust_cmt_yield_10y"} <= set(w.columns):
        a = mm(w["macro_long_yield"]); b = mm(w["ust_cmt_yield_10y"])
        c = a.index.intersection(b.index)
        c = c[c >= pd.Period("1962-01", "M")]
        out["macro_long_yield - monthly avg ust_cmt_yield_10y (1962+), median abs"] = float((a[c] - b[c]).abs().median())
    if {"tips_breakeven_10y", "ust_cmt_yield_10y", "tips_real_yield_10y"} <= set(w.columns):
        x = (w["ust_cmt_yield_10y"] - w["tips_real_yield_10y"] - w["tips_breakeven_10y"]).dropna()
        out["ust 10y - tips real 10y - tips breakeven 10y, median (par vs zero basis)"] = float(x.median())
    if {"credit_moodys_baa_yield_m", "credit_moodys_aaa_yield_m", "credit_moodys_baa_aaa_spread_m"} <= set(w.columns):
        x = (w["credit_moodys_baa_yield_m"] - w["credit_moodys_aaa_yield_m"] - w["credit_moodys_baa_aaa_spread_m"]).dropna()
        out["Baa - Aaa - (Baa-Aaa spread), max abs"] = float(x.abs().max())
    if {"credit_moodys_baa_yield_m", "credit_moodys_baa_ust10y_spread_m", "macro_long_yield"} <= set(w.columns):
        x = (w["credit_moodys_baa_yield_m"] - w["credit_moodys_baa_ust10y_spread_m"] - w["macro_long_yield"]).dropna()
        x = x[x.index >= "1953-04-30"]
        out["Baa - (Baa-10y spread) - macro_long_yield (1953-04+), median abs"] = float(x.abs().median())
    return out


def cash_rate_check(pred: pd.DataFrame, cash_monthly: pd.Series) -> float:
    """Median |macro_short_rate - 12 x monthly cash return| since 1934 (3m bill vs 1m bill)."""
    s = pred[pred.series == "macro_short_rate"].set_index("obs_date")["value"]
    if s.empty:
        return np.nan
    c = (12 * cash_monthly).reindex(s.index)
    d = (s - c).dropna()
    return float(d[d.index >= "1934-01-01"].abs().median())


# ============================================================== validation
def validate(A: Assembled, seg: pd.DataFrame, pred: pd.DataFrame, catalog: pd.DataFrame,
             cal: pd.DatetimeIndex) -> dict:
    v = {}
    d, m = A.daily, A.monthly
    v["daily_strictly_increasing"] = bool(d.index.is_monotonic_increasing and d.index.is_unique)
    span = cal[(cal >= d.index[0]) & (cal <= d.index[-1])]
    v["daily_index_equals_cash_calendar"] = bool(d.index.equals(span))
    v["monthly_index_month_ends"] = bool((m.index == m.index + pd.offsets.MonthEnd(0)).all())
    v["min_daily_ret"] = float(np.nanmin(d.values))
    v["min_monthly_ret"] = float(np.nanmin(m.values))
    v["n_ret_le_minus1"] = int((d <= -1).sum().sum() + (m <= -1).sum().sum())
    v["last_month"] = str(m.index[-1].date())
    v["no_partial_final_month"] = bool(month_is_complete(d.index[d.index <= m.index[-1]][-1])
                                      and d.index[-1] < m.index[-1] + pd.offsets.MonthEnd(1)
                                      and m.index[-1].to_period("M") <= d.index[-1].to_period("M"))
    v["cash_daily_nan"] = int(A.cash_daily.isna().sum())
    v["cash_monthly_nan"] = int(A.cash_monthly.isna().sum())
    ex_d = np.log1p(d).sub(np.log1p(A.cash_daily), axis=0)
    ex_m = np.log1p(m).sub(np.log1p(A.cash_monthly), axis=0)
    v["excess_nan_from_cash_daily"] = int((d.notna() & ex_d.isna()).sum().sum())
    v["excess_nan_from_cash_monthly"] = int((m.notna() & ex_m.isna()).sum().sum())
    # documented starts
    starts = {}
    for a in d.columns.union(m.columns, sort=False):
        sd = seg[(seg.asset == a) & (seg.freq == "D")]
        sm = seg[(seg.asset == a)]
        fd = d[a].first_valid_index() if a in d else None
        fm = m[a].first_valid_index()
        doc_d = pd.Timestamp(sd.start.min()) if len(sd) else None
        doc_m = pd.Timestamp(sm.start.min()) if len(sm) else None
        starts[a] = dict(first_daily=fd, first_monthly=fm, doc_daily=doc_d, doc_any=doc_m,
                         ok=(fd is None or (doc_d is not None and fd >= doc_d))
                         and (doc_m is not None and fm >= doc_m - pd.offsets.MonthEnd(1)))
    v["starts"] = starts
    # five largest absolute daily returns
    big = {}
    for a in d.columns:
        s = d[a].dropna()
        top = s.abs().sort_values(ascending=False).head(5).index
        big[a] = [(str(t.date()), float(s[t]), A.daily_source.at[t, a]) for t in sorted(top)]
    v["largest_daily"] = big
    # tiers by month
    v["tier_counts"] = {a: A.tiers[a].value_counts().reindex(list(TIER_ORDER)).fillna(0).astype(int).to_dict()
                        for a in A.tiers.columns}
    # calendar folding
    nan_after = {}
    for a in d.columns:
        s = d[a]
        fv, lv = s.first_valid_index(), s.last_valid_index()
        if fv is None:
            continue
        nan_after[a] = dict(n_nan_in_span=int(s.loc[fv:lv].isna().sum()),
                            nan_examples=[str(x.date()) for x in s.loc[fv:lv][s.loc[fv:lv].isna()].index[:6]],
                            last_valid=str(lv.date()))
    v["calendar"] = {a: dict(n_input=f.n_input, n_folded=f.n_off_calendar, n_dropped_after_end=f.n_dropped_after_end,
                             n_folded_across_month=f.n_boundary_moves, examples=f.off_examples, **nan_after.get(a, {}))
                     for a, f in A.folds.items()}
    # predictors
    v["pred_available_ge_obs"] = bool((pred["available_date"] >= pred["obs_date"]).all())
    v["pred_dup_series_obs"] = int(pred.duplicated(["series", "obs_date"]).sum())
    v["pred_n_rows"] = len(pred)
    v["pred_n_series"] = int(pred["series"].nunique())
    v["pred_unit_checks"] = predictor_unit_checks(pred)
    v["pred_unit_checks"]["macro_short_rate - 12 x cash monthly (1934+), median abs"] = cash_rate_check(pred, A.cash_monthly)
    return v


def panel_asof_smoke() -> dict:
    """load_panel() + Panel.asof behaviour on the files just written."""
    from .panel import load_panel
    p = load_panel()
    out = {"loaded": True, "daily_shape": p.daily.shape, "monthly_shape": p.monthly.shape,
           "n_predictor_series": int(p.predictors["series"].nunique())}
    t = pd.Timestamp("2008-10-15")
    v = p.asof(t)
    out["asof_daily_max"] = str(v.daily.index.max().date())
    out["asof_monthly_max"] = str(v.monthly.index.max().date())
    out["asof_pred_avail_max"] = str(v.predictors["available_date"].max().date())
    w = v.predictors_wide()
    out["asof_wide_ok"] = bool(w.index.max() <= t)
    lat = v.predictor_latest(max_age_days=100)
    out["asof_latest_n"] = int(len(lat))
    out["asof_ok"] = bool(v.daily.index.max() <= t and v.monthly.index.max() <= t
                          and (v.predictors["available_date"] <= t).all())
    out["eligible_2008"] = v.eligible()
    out["excess_daily_nan_with_ret"] = int((p.daily.notna() & p.excess_daily.isna()).sum().sum())
    return out


# ============================================================== horizon evidence
def log_excess_daily(A: Assembled) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Log excess returns over each asset's active span, with a NaN day's return set to 0
    (the next return spans that day) while cash still accrues; plus the source label of
    each day (a gap day belongs to the return that spans it)."""
    lc = np.log1p(A.cash_daily)
    X = pd.DataFrame(index=A.daily.index, columns=A.daily.columns, dtype=float)
    S = pd.DataFrame(index=A.daily.index, columns=A.daily.columns, dtype=object)
    for a in A.daily.columns:
        r = A.daily[a]
        fv, lv = r.first_valid_index(), r.last_valid_index()
        if fv is None:
            continue
        x = np.log1p(r.loc[fv:lv]).fillna(0.0) - lc.loc[fv:lv]
        X.loc[fv:lv, a] = x.values
        S.loc[fv:lv, a] = A.daily_source[a].loc[fv:lv].bfill().values
    return X, S


def _ksum(x: np.ndarray, k: int) -> np.ndarray:
    c = np.concatenate([[0.0], np.cumsum(x)])
    return c[k:] - c[:-k]


def vol_k(x: np.ndarray, k: int) -> float:
    if len(x) < k + 20:
        return np.nan
    s = _ksum(x, k)
    return float(np.sqrt(np.mean(s ** 2) * 252 / k))


def corr_k(x: np.ndarray, y: np.ndarray, k: int) -> float:
    if len(x) < k + 20:
        return np.nan
    sx, sy = _ksum(x, k), _ksum(y, k)
    return float(np.sum(sx * sy) / np.sqrt(np.sum(sx ** 2) * np.sum(sy ** 2)))


def _short(src: str, n: int = 38) -> str:
    s = str(src)
    return s if len(s) <= n else s[: n - 1] + "…"


def segment_vol_table(X, S) -> pd.DataFrame:
    rows = []
    for a in X.columns:
        x, s = X[a], S[a]
        for lab, st, en, n in _runs(s):
            v = x.loc[st:en].values
            r = dict(asset=a, source=lab, start=st.date(), end=en.date(), days=n)
            for k in HORIZONS:
                r[f"v{k}"] = vol_k(v, k)
            r["ac1"] = float(pd.Series(v).autocorr(1)) if n > 30 else np.nan
            rows.append(r)
    t = pd.DataFrame(rows)
    for k in HORIZONS:
        t[f"r{k}"] = t[f"v{k}"] / t["v21"]
    return t


def splice_jump_table(X, S, window: int = 504) -> pd.DataFrame:
    """Around each daily splice: J_k = [vol_k(after)/vol_k(before)] / [vol_21(after)/vol_21(before)]
    on +-``window`` days. J_k = 1 means the k-day measure moves exactly like the 21-day measure."""
    rows = []
    for a in X.columns:
        runs = _runs(S[a])
        for (l0, s0, e0, n0), (l1, s1, e1, n1) in zip(runs[:-1], runs[1:]):
            w = min(window, n0, n1)
            if w < 126:
                continue
            xb = X[a].loc[s0:e0].values[-w:]
            xa = X[a].loc[s1:e1].values[:w]
            r = dict(asset=a, splice=s1.date(), before=l0, after=l1, window=w)
            vb21, va21 = vol_k(xb, 21), vol_k(xa, 21)
            for k in HORIZONS:
                r[f"J{k}"] = (vol_k(xa, k) / vol_k(xb, k)) / (va21 / vb21)
            r["vol21_after_over_before"] = va21 / vb21
            rows.append(r)
    return pd.DataFrame(rows)


def pair_corr_table(X, S, pairs=PAIRS, min_days: int = 252) -> pd.DataFrame:
    rows = []
    for a, b in pairs:
        if a not in X.columns or b not in X.columns:
            continue
        both = X[[a, b]].notna().all(axis=1) & S[a].notna() & S[b].notna()
        lab = (S[a].where(both) + SEP + S[b].where(both))
        for l, st, en, n in _runs(lab):
            if n < min_days:
                continue
            sa, sb = l.split(SEP)
            xa, xb = X[a].loc[st:en].values, X[b].loc[st:en].values
            r = dict(pair=f"{a}–{b}", src_a=sa, src_b=sb, start=st.date(), end=en.date(), days=n)
            for k in HORIZONS:
                r[f"c{k}"] = corr_k(xa, xb, k)
            rows.append(r)
    return pd.DataFrame(rows)


def simulate_rv_se(n_days: int = 252, n_sims: int = 20000, rho: float = 0.5, seed: int = 0) -> pd.DataFrame:
    """Relative standard error of a 12-month realized vol (and abs SE of a correlation)
    from overlapping k-day sums fully inside a ``n_days`` window, iid returns."""
    rng = np.random.default_rng(seed)
    out = []
    for dist in ["normal", "t5"]:
        if dist == "normal":
            e1 = rng.standard_normal((n_sims, n_days))
            e2 = rng.standard_normal((n_sims, n_days))
        else:
            e1 = rng.standard_t(5, (n_sims, n_days)) / np.sqrt(5 / 3)
            e2 = rng.standard_t(5, (n_sims, n_days)) / np.sqrt(5 / 3)
        y = rho * e1 + np.sqrt(1 - rho ** 2) * e2
        c1 = np.concatenate([np.zeros((n_sims, 1)), np.cumsum(e1, 1)], 1)
        c2 = np.concatenate([np.zeros((n_sims, 1)), np.cumsum(y, 1)], 1)
        for k in HORIZONS:
            s1 = c1[:, k:] - c1[:, :-k]
            s2 = c2[:, k:] - c2[:, :-k]
            vh = np.sqrt(np.mean(s1 ** 2, 1) / k)
            ch = np.sum(s1 * s2, 1) / np.sqrt(np.sum(s1 ** 2, 1) * np.sum(s2 ** 2, 1))
            N = n_days - k + 1
            formula = 0.5 * np.sqrt(2.0 / N * (2 * k * k + 1) / (3 * k)) if k > 1 else 0.5 * np.sqrt(2.0 / N)
            out.append(dict(dist=dist, k=k, n_sums=N, se_vol_rel=float(vh.std() / np.mean(vh)),
                            bias_vol=float(np.mean(vh) - 1), se_vol_formula=formula,
                            se_corr=float(ch.std()), bias_corr=float(ch.mean() - rho)))
    return pd.DataFrame(out)


NOISE_N = 2520


def simulate_ratio_noise(n_days: int = NOISE_N, n_sims: int = 1500, rho: float = 0.5, seed: int = 1) -> dict:
    """Sampling noise, under iid normal returns, of the statistics in tables (a) and (b) for a
    segment of ``n_days`` (scale by sqrt(n_days / n) for other lengths): SE of v_k / v_21, of
    c_k - c_63 and of c_k - c_21."""
    rng = np.random.default_rng(seed)
    e1 = rng.standard_normal((n_sims, n_days))
    y = rho * e1 + np.sqrt(1 - rho ** 2) * rng.standard_normal((n_sims, n_days))
    c1 = np.concatenate([np.zeros((n_sims, 1)), np.cumsum(e1, 1)], 1)
    c2 = np.concatenate([np.zeros((n_sims, 1)), np.cumsum(y, 1)], 1)
    v, c = {}, {}
    for k in HORIZONS:
        s1, s2 = c1[:, k:] - c1[:, :-k], c2[:, k:] - c2[:, :-k]
        v[k] = np.sqrt(np.mean(s1 ** 2, 1) / k)
        c[k] = np.sum(s1 * s2, 1) / np.sqrt(np.sum(s1 ** 2, 1) * np.sum(s2 ** 2, 1))
    return {"n": n_days,
            "se_r": {k: float(np.std(v[k] / v[21])) for k in HORIZONS},
            "se_d63": {k: float(np.std(c[k] - c[63])) for k in HORIZONS},
            "se_d21": {k: float(np.std(c[k] - c[21])) for k in HORIZONS}}


def add_noise_z(seg_t: pd.DataFrame, pair_t: pd.DataFrame, noise: dict) -> None:
    """z-scores of each statistic against iid sampling noise for its segment length."""
    for k in HORIZONS:
        if k == 21:
            continue
        sc = np.sqrt(noise["n"] / seg_t["days"])
        seg_t[f"z{k}"] = (seg_t[f"r{k}"] - 1) / (noise["se_r"][k] * sc)
    sc = np.sqrt(noise["n"] / pair_t["days"])
    for k in HORIZONS:
        if k == 63:
            continue
        pair_t[f"zd{k}"] = (pair_t[f"c{k}"] - pair_t["c63"]) / (noise["se_d63"][k] * sc)


# ============================================================== reporting helpers
def _md(df: pd.DataFrame, fmt: dict | None = None, default: str = "{:.3f}") -> str:
    fmt = fmt or {}
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for row in df.itertuples(index=False):
        cells = []
        for c, v in zip(cols, row):
            if isinstance(v, (float, np.floating)):
                cells.append("" if np.isnan(v) else fmt.get(c, default).format(v))
            elif isinstance(v, (bool, np.bool_)):
                cells.append("yes" if v else "no")
            else:
                cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def _tiers_by_era(tiers: pd.Series) -> str:
    s = tiers.dropna()
    return ", ".join(f"{t} {st.strftime('%Y-%m')}..{en.strftime('%Y-%m')}" for t, st, en, n in _runs(s))


EVENTS = {
    "1929-10": "1929 crash", "1929-11": "1929 crash", "1931-10": "1931 sterling crisis", "1932": "1932 bear market/rally",
    "1933": "1933 bank holiday/reflation", "1937-10": "1937 crash", "1939-09": "outbreak of WWII", "1940-05": "fall of France", "1962-05": "1962 flash crash",
    "1980-01": "gold spike and collapse", "1980-02": "Volcker / rates spike", "1980-03": "silver crisis / rates spike",
    "1980-04": "Volcker / rates reversal", "1987-10": "Oct 1987 crash", "1989-10": "1989 mini-crash",
    "1997-10": "Asian crisis", "1998-08": "Russia / LTCM", "2000-04": "dot-com", "2001-09": "9/11 reopening",
    "2008-09": "Lehman / GFC", "2008-10": "GFC", "2008-11": "GFC", "2008-12": "GFC", "2009-03": "March 2009 bottom / QE1 / PPIP",
    "2011-08": "US downgrade", "2016-06-24": "Brexit", "2020-03": "COVID crash", "2020-04": "COVID / Fed facilities",
    "2022-03": "2022 war/rates shock",
}


def event_of(date: str) -> str | None:
    for key in (date, date[:7], date[:4]):
        if key in EVENTS:
            return EVENTS[key]
    return None


CAVEATS = """\
Collected from the track reports; each is something a model can get wrong silently.

**Tiers.** A = daily price of an exchange-traded fund (market close); B = daily open-end mutual-fund NAV, index,
or reconstruction; C = monthly index total return; D = monthly model-based reconstruction. Every splice from a
fund NAV to an ETF (VWEHX -> HYG, the IG fund blends -> LQD, VEIEX -> EEM, VGSIX -> VNQ, PRRIX -> TIP) is
therefore a B -> A tier change that models can see.

**Holdout integrity.** No construction parameter that shapes pre-2015 history is fitted on 2015+ data: the
`ig_credit` blend weights, Ibbotson/T-bill weight and tier-D maturity are calibrated on LQD 2002-08..2014-12
(`credit.CALIBRATION_END`). The remaining selection rules were re-run on overlaps ending 2014-12 and give the
same choice (checked for the 2026-11 rebuild): French Hi 30 / Qnt 2+3 still track SPY / IWM best (us_equity.md);
VWEHX still tracks HYG best among the HY funds (monthly corr 0.87, TE 6.6% on 2007-04..2014-12, vs 0.82-0.85 and
7.1-7.6% for FHIIX, PRHYX, SPHIX, AHITX); AQR vs Ken French dev-ex-US keeps the same trade-off vs MSCI EAFE (KF
better monthly, AQR better daily); AQR-EW/BCOM vol ratio 0.89 on 1991-2014 vs 0.91 on 1991-2025; every Cboe/Yahoo
implied-vol agreement check passes or fails identically on 2014-truncated overlaps. The VIX/VXO splice factor is
estimated on 1990-94. Treasury ladders, fund-blend constituents and splice dates are definitional, not fitted.

**Stale or smoothed prices (daily vol understated, positive autocorrelation).**
- `hy_credit` 1985-01..2007-04 is the VWEHX mutual-fund NAV (matrix-priced, tier B): daily vol is about half of
  HYG's, lag-1 autocorrelation 0.12-0.57; monthly-from-daily variance ratio 1.3-2.2. Even at the 21-day target
  horizon its vol is ~0.83x HYG's (0.80x in 2007-09, 0.93x at 63 days; segments.csv notes): do not pool this era's
  vol LEVEL with HYG's. 1980-1984 is one NAV per month sampled on ex-dividend dates (tier C), a few days off
  month-end.
- `ig_credit` 1980-2002 is a daily constant-mix blend of open-end bond funds (tier B): mild staleness (AC1
  0.08-0.15 in the 1980s-90s); daily correlation with LQD only 0.67 (bond 3pm vs ETF 4pm close).
- Daily-accrual bond funds (VWEHX, VFSTX, FBNDX, VWESX from 1985) pay a month of income without an NAV drop; their
  month-end distributions are spread over the accrual month (Yahoo's adjclose would book them on one day).
- `reits` 1986-11..1996-05 is FRESX, an actively managed fund (tier B), and 1996-2004 VGSIX NAV: daily AC1 0.12-0.32
  until 2003, 1990s monthly variance about 2x what daily data imply; the reverse (ratio ~0.6-0.7) after VNQ (2004).
  The pre-1990s REIT market was tiny and illiquid: do not read 12-16% vol as today's normal.
- `ust_int`/`ust_long` daily 1962-c.1990 are reconstructions from quoted yields with positively autocorrelated
  changes (21d/1d variance ratio up to 2.7 in 1970-74); 1960s quotes are in 32nds (36-53% zero changes).
- `tips` 1997-2003 is PRRIX (active PIMCO fund) with daily-accrued distributions re-spread; NAVs move in 0.1% ticks
  (35-40% zero days in 1997-2000).
- `intl_dev` 1986-2001 (AQR index, foreign closes) and `em` 1994-2003 (VEIEX NAV, foreign closes, fair-valued from
  ~2001) react to US news one day late: same-day correlation with US assets badly understated (lag-1 US corr ~0.38).

**US-listed ETFs of foreign/illiquid assets (daily vol overstated, negative autocorrelation).**
- EFA (2001-09+) and EEM (2003-05+): 1-day vol 1.2-1.44x the MSCI index, monthly vols agree within 2%.
- LQD/HYG in 2008 and March 2020 traded at multi-day discounts to NAV; one-day reversals of up to 10-12%.
- TIP in 2008 Q4 traded at sizeable discounts/premia.

**Composition changes at splices.**
- `commodities`: equal-weight AQR futures basket (1926-1991, few agricultural contracts before the 1960s) -> BCOM
  weights (1991+): vol ratio AQR-EW/BCOM 0.75-0.97 on the overlap; treat pre-1970 as a different asset. BCOM
  1991-98 is a backfill. BCOM is not DBC: DBC has ~1.17x BCOM's vol.
- `ig_credit` pre-1980: {ig_wc} x Ibbotson long-term corporate + T-bill (high-grade long-bond risk; gained in
  2008), vol-matched to LQD on 2002-08..2014-12; LQD's own duration drifted from ~7 to ~8.5.
- `ust_long`: a 20-30y par ladder; TLT's effective duration differs (vol ratio 0.86-0.93 vs TLT in 2022-26).
- `intl_dev` 1986-2001 includes Canada (EFA does not); 1975-85 is Ken French's EAFE-weighted index.
- `reits`: Nareit All Equity (1972-86) -> FRESX (active) -> MSCI US REIT (VGSIX) -> MSCI US IMI RE 25/50 (VNQ).
- `gold`: Frankfurt fixing 13:00 CET (1971-98) -> COMEX front-month settle (1999-2004) -> GLD 16:00 ET.

**Month-average (tier D) eras: vol understated ~15%, AC1 ~0.25-0.4, timing blurred.**
- `ust_long` 1919-02..1962-01, `ust_int` 1953-05..1962-01 (month-average yields; WWII peg 1942-51 near-zero vol;
  1919-41 partially tax-exempt bonds; pre-1973 long end includes flower bonds).
- `ig_credit` 1919-02..1925-12 (month-average Moody's yields; monthly corr with the later index only 0.41).
- `us_large` pre-1926 (Shiller/Cowles month-average prices), if present. Cash before 1926-07 is model-based (tier D).

**Series that end, have holes, or have known breaks.**
- `ust_long` daily: 1973-01-04 return dropped (H.15 20y re-basing): a genuine lost day, not a spanning gap.
- `ust_*` daily: Lincoln's birthday/Columbus/Veterans days are NaN (bond market closed, NYSE open); 1968 Wednesdays
  (NYSE closed, bonds traded) are folded into Thursday.
- Confirmed bad prints removed (the price is dropped, the next return spans the gap): `reits` FRESX 1993-07-29;
  `em` VEIEX 2001-07-16; `ig_credit` FBNDX 1982-10-06, 1983-11-21, 1983-11-28, 1984-02-02, 1984-11-14,
  1984-12-06; `hy_credit` VWEHX 1991-08-19; `tips` PRRIX 2000-12-12; `gold` Frankfurt 1996-12-06; `us_large` SPY
  2001-04-30. Spurious/inflated Yahoo distributions corrected: IEF 2003-07-01, 08-01, 09-02, 10-01, 12-16 and LQD
  2003-12-16 (ETF returns are close + validated distributions, not adjclose).
- Cash 2026-10-01.. uses H.15 4-week bill (tier D); the last 3 days carry the last published yield.
- Predictors: Moody's yields end 2025-12 (not live); daily Moody's spreads end 2016-10; `iv_vxo` ended 2021;
  VXHYG has no data 2021-09..2025-05, VXIEF/VXLQD 2022-11..2025-07; `iv_move` not built; the 2025-10 CPI and
  unemployment prints were never published (missing, not imputed); industrial production is latest-vintage
  (revised); unemployment SA factors revised. `gz_excess_bond_premium_m` is excluded (full-sample regression).
- Predictor timing: H.15 CMT yields are available one business day after observation; everything derived from
  the Fed's GSW fitted curves (Treasury ladder yields and durations, ETF duration estimates, TIPS real yields
  and breakevens) ten business days after; monthly averages of daily market rates one business day after
  month-end (H.10 dollar: five). `ust_cmt_yield_20y` starts 1973-01-04 (the series was re-based that day, +74bp).
"""


def write_panel_report(A: Assembled, seg: pd.DataFrame, catalog: pd.DataFrame, v: dict, seg_issues: list,
                       pred_issues: list, conv: list, smoke: dict | None, path: Path) -> None:
    L = ["# Assembled panel",
         "",
         "Built by `python -m assetforecast.data.assemble` (module `src/assetforecast/data/assemble.py`). "
         "Read with `assetforecast.data.panel.load_panel()`. All returns SIMPLE total returns in decimals; "
         "excess/log transforms are done by `Panel`.",
         ""]
    missing = [a for a in ASSET_IDS if a not in A.monthly.columns]
    L.append(f"* Daily panel: {A.daily.index[0].date()} .. {A.daily.index[-1].date()}, {len(A.daily):,} US equity "
             f"trading days (the cash calendar), {A.daily.shape[1]} assets.")
    L.append(f"* Monthly panel: {A.monthly.index[0].date()} .. {A.monthly.index[-1].date()}, {len(A.monthly):,} "
             f"month-ends. Last month = last month whose final trading day is in the daily data "
             f"(data end {A.daily.index[-1].date()}, so the partial current month is excluded).")
    if missing:
        L.append(f"* **Missing assets (no files yet): {', '.join(missing)}.** Re-run the module when they land.")
    L += ["", "## Coverage", ""]
    rows = []
    for a in A.monthly.columns:
        st = v["starts"][a]
        tc = v["tier_counts"][a]
        srcs = seg[(seg.asset == a)]
        rows.append(dict(asset=a, first_monthly=st["first_monthly"].date() if st["first_monthly"] is not None else "",
                         first_daily=st["first_daily"].date() if st["first_daily"] is not None else "",
                         first_daily_month=str(A.daily_months[a][0]) if len(A.daily_months[a]) else "",
                         A=tc["A"], B=tc["B"], C=tc["C"], D=tc["D"],
                         tiers_by_era=_tiers_by_era(A.tiers[a])))
    L.append(_md(pd.DataFrame(rows)))
    L += ["", "`first_daily_month` = first month compounded from daily data (earlier months, including a partial "
          "first daily month, come from the track's monthly file). Tier columns count months.", "",
          "### Sources by segment", "",
          _md(seg.assign(source=seg.source.map(lambda s: _short(s, 70)))[["asset", "freq", "start", "end", "tier", "source"]]),
          "", "Full notes (overlap statistics) are in `data/processed/segments.csv`.", ""]
    if seg_issues:
        L += ["Segment rows published by the tracks that disagree with the files (the files win):", ""]
        L += [f"- {s}" for s in seg_issues] + [""]

    L += ["## Calendar handling", "",
          "Off-calendar returns are compounded into the next calendar day; calendar days without a return are "
          "NaN within the asset's span (the next return spans the gap).", ""]
    rows = []
    for a, c in v["calendar"].items():
        mv = A.checks["monthly_vs_track"].get(a, {})
        rows.append(dict(asset=a, input_days=c["n_input"], folded=c["n_folded"],
                         folded_across_month=c["n_folded_across_month"],
                         dropped=c["n_dropped_after_end"], nan_days_in_span=c.get("n_nan_in_span", 0),
                         examples_folded=", ".join(f"{x}→{y}" for x, y in c["examples"][:2]),
                         examples_nan=", ".join(c.get("nan_examples", [])[:3])))
    L.append(_md(pd.DataFrame(rows)))
    L += ["", "### Monthly = compounded daily", ""]
    rows = []
    for a, c in A.checks["monthly_vs_track"].items():
        rows.append(dict(asset=a, daily_months=c["n_daily_months"], compared=c.get("n_compared", 0),
                         max_abs_vs_track=c.get("max_abs_vs_track", np.nan),
                         worst_month=c.get("worst_month_vs_track", ""),
                         unfolded_max_abs_vs_track=c.get("max_abs_unfolded_vs_track", np.nan),
                         months_moved_by_folding=c["n_months_moved_by_folding"],
                         max_folding_effect=c["max_abs_folding_effect"]))
    L.append(_md(pd.DataFrame(rows), default="{:.2e}"))
    L += ["", "`max_abs_vs_track`: panel month (compounded folded daily) minus the track's monthly file; "
          "`unfolded_max_abs_vs_track`: same with the track's own daily dates (tests the track's internal "
          "consistency); `max_folding_effect`: the part due to folding moving a return across a month boundary "
          "(e.g. a foreign trading day on a US holiday that is the first weekday of a month).",
          f"Cash: compounded daily vs monthly max abs gap {A.checks['cash_compound_gap']:.2e}.", ""]

    L += ["## Validation", ""]
    keys = ["daily_strictly_increasing", "daily_index_equals_cash_calendar", "monthly_index_month_ends",
            "n_ret_le_minus1", "min_daily_ret", "min_monthly_ret", "last_month", "no_partial_final_month",
            "cash_daily_nan", "cash_monthly_nan", "excess_nan_from_cash_daily", "excess_nan_from_cash_monthly",
            "pred_available_ge_obs", "pred_dup_series_obs", "pred_n_rows", "pred_n_series"]
    L.append(_md(pd.DataFrame([dict(check=k, value=v[k]) for k in keys]), default="{:.4f}"))
    bad = [a for a, s in v["starts"].items() if not s["ok"]]
    L += ["", f"Returns before documented start: {'none' if not bad else ', '.join(bad)}.", ""]
    if smoke:
        L += [f"`load_panel()` smoke test: daily {smoke['daily_shape']}, monthly {smoke['monthly_shape']}, "
              f"{smoke['n_predictor_series']} predictor series; `asof(2008-10-15)`: daily max {smoke['asof_daily_max']}, "
              f"monthly max {smoke['asof_monthly_max']}, latest predictor available_date {smoke['asof_pred_avail_max']}, "
              f"no leak: {smoke['asof_ok']}; eligible (36m) then: {', '.join(smoke['eligible_2008'])}.", ""]
    L += ["### Five largest absolute daily returns", ""]
    rows, unknown = [], []
    for a, lst in v["largest_daily"].items():
        cells = []
        for d, r, s in lst:
            ev = event_of(d)
            if ev is None:
                unknown.append(f"{a} {d}")
            cells.append(f"{d} {r:+.1%} ({ev or '?'})")
        rows.append(dict(asset=a, days="; ".join(cells)))
    L.append(_md(pd.DataFrame(rows)))
    L += ["", ("Every date matches a known stress episode. " if not unknown else
               f"Dates without an annotated event (check them): {', '.join(unknown)}. ")
          + "Several ETF days are one-day overshoots that reverse the next day (e.g. LQD 2008-09-29/30, "
            "EEM/EFA 2008-10), documented by the credit and international tracks; the 1980 gold days are the "
            "Frankfurt fixing during the January 1980 spike and collapse (real-assets track keeps them as genuine).", ""]

    L += ["## Predictors", "",
          f"`predictors.parquet`: {v['pred_n_rows']:,} rows, {v['pred_n_series']} series, long format "
          "(obs_date, available_date, series, value). `predictors_catalog.csv` lists every series. "
          "Standard units: rates, yields, spreads, breakevens, inflation/growth rates and implied vols in DECIMALS; "
          "durations in years; index levels and flags unchanged. `live` = an observation within 60 days "
          "(monthly series: 100 days) of the data end.", ""]
    cs = catalog.groupby("group").agg(series=("series", "size"), live=("live", "sum"),
                                      start=("start", "min"), end=("end", "max")).reset_index()
    L.append(_md(cs))
    L += ["", "Conversions applied:", ""]
    L += [f"- `{f}:{s}` → `{n}` {d}" for f, s, n, d in conv]
    dead = catalog[~catalog.live]
    L += ["", "Not live: " + ", ".join(f"`{r.series}` (ends {r.end})" for r in dead.itertuples()) + ".", ""]
    L += ["Unit cross-checks (after conversion; small numbers confirm consistent units across files):", ""]
    L += [f"- {k}: {x:.4f}" for k, x in v["pred_unit_checks"].items()]
    if pred_issues:
        L += ["", "Predictor issues:"] + [f"- {s}" for s in pred_issues]
    m = seg[(seg.asset == "ig_credit")]["notes"].str.extract(r"(\d\.\d\d) x Ibbotson")[0].dropna()
    L += ["", "## Caveats for modelers", "", CAVEATS.replace("{ig_wc}", m.iloc[0] if len(m) else "w")]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L) + "\n")


def write_evidence_report(seg_t: pd.DataFrame, jump_t: pd.DataFrame, pair_t: pd.DataFrame, se_t: pd.DataFrame,
                          noise: dict, A: Assembled, path: Path) -> dict:
    """reports/data/horizon_evidence.md. Every number is computed here; the text interprets them."""
    kk = [k for k in HORIZONS if k != 21]
    big = seg_t[seg_t.days >= 750]
    pt = pair_t[pair_t.days >= 750].copy()
    for k in HORIZONS:
        pt[f"d{k}"] = pt[f"c{k}"] - pt["c63"]
    multi = big.groupby("asset").filter(lambda g: len(g) > 1)
    spread = multi.groupby("asset")[[f"r{k}" for k in HORIZONS]].agg(lambda s: s.max() - s.min())
    se = se_t[se_t.dist == "normal"].set_index("k")
    se5 = se_t[se_t.dist == "t5"].set_index("k")
    yrs = noise["n"] // 252

    vol_rows = []
    for k in HORIZONS:
        r = big[f"r{k}"]
        z = big[f"z{k}"].abs() if k != 21 else pd.Series(0.0, index=big.index)
        vol_rows.append(dict(k=k, min=r.min(), median=r.median(), max=r.max(),
                             spread_median=spread[f"r{k}"].median() if len(spread) else np.nan,
                             spread_max=spread[f"r{k}"].max() if len(spread) else np.nan,
                             share_signif=float((z > 2).mean()),
                             iid_se_10y=noise["se_r"][k]))
    vol_sum = pd.DataFrame(vol_rows)
    corr_rows = []
    for k in [1, 2, 3, 5, 10, 21]:
        d = pt[f"d{k}"]
        corr_rows.append(dict(k=k, median=d.median(), median_abs=d.abs().median(), min=d.min(), max=d.max(),
                              share_signif=float((pt[f"zd{k}"].abs() > 2).mean()), iid_se_10y=noise["se_d63"][k]))
    corr_sum = pd.DataFrame(corr_rows)

    L = ["# Why the target is measured from 21-day returns: evidence from the assembled panel", "",
         "Generated by `python -m assetforecast.data.assemble` from `panel_daily` (re-run it when assets are added). "
         "Log EXCESS returns (asset minus cash); a NaN day contributes 0 to the asset (its move is in the next "
         "return) while cash still accrues. Overlapping k-day sums use only windows that lie entirely inside one "
         "source segment (for pairs: inside one era in which neither asset changes source). "
         "Vol = sqrt(mean(s_k²)·252/k) and correlation = Σ s_a s_b / sqrt(Σ s_a² Σ s_b²), both uncentered as in "
         "the target definition.", ""]
    missing = [a for a in ASSET_IDS if a not in A.daily.columns]
    if missing:
        L += [f"**Not yet in the panel: {', '.join(missing)}.** Pairs involving them (including most of the "
              "requested us_large pairs) are absent; the conclusions below rest on the other assets.", ""]

    L += ["## Summary tables", "",
          f"**Vol.** `r_k` = vol_k / vol_21 for each daily source segment with ≥ 750 days ({len(big)} segments). "
          "`spread` = max − min of r_k across the segments of one asset (how much the k-day vol jumps relative to the "
          f"21-day vol when the source changes; {len(spread)} assets with ≥ 2 such segments). `share_signif` = share "
          f"of segments with |r_k − 1| > 2 iid standard errors for their length; `iid_se_10y` = that standard error "
          f"for a {yrs}-year segment (iid normal returns; real data with volatility clustering are noisier, so "
          "these z-scores overstate significance).", "",
          _md(vol_sum), "",
          f"**Correlation.** c_k − c_63 for each pair-era with ≥ 750 days ({len(pt)} pair-eras); c_63 is the "
          "longest-horizon reference (asynchronous closes and stale prices are absorbed by 63 days), but it is "
          f"itself noisy: `iid_se_10y` is the standard error of c_k − c_63 for a {yrs}-year era with true ρ = 0.5.", "",
          _md(corr_sum), "",
          "**Cost.** Standard error of a 12-month (252-day) realized vol (relative) and correlation (absolute, true "
          "ρ = 0.5) from overlapping k-day sums inside the window; 20,000 simulations of iid returns "
          "(`t5`: Student-t with 5 df).", "",
          _md(pd.DataFrame([dict(k=k, n_sums=int(se.loc[k, "n_sums"]), se_vol=se.loc[k, "se_vol_rel"],
                                 se_vol_formula=se.loc[k, "se_vol_formula"], se_vol_t5=se5.loc[k, "se_vol_rel"],
                                 bias_vol=se.loc[k, "bias_vol"], se_corr=se.loc[k, "se_corr"],
                                 bias_corr=se.loc[k, "bias_corr"]) for k in HORIZONS])), "",
          "Formula: SE(σ̂)/σ ≈ ½·sqrt(2/N · (2k²+1)/(3k)) with N = 252 − k + 1 overlapping sums (k = 1: ½·sqrt(2/N)). "
          "`bias_*` = mean estimate minus truth (small-sample bias of the uncentered estimator).", "",
          "<!-- NARRATIVE -->", ""]

    L += ["## (a) Vol by horizon, per asset and daily source segment", "",
          "`v_k` annualized vol (decimal) of overlapping k-day log excess returns; `r_k` = v_k / v_21; `z63` = "
          "(r_63 − 1) in iid standard errors; `ac1` = lag-1 autocorrelation of daily log excess returns.", ""]
    show = seg_t.copy()
    show["source"] = show["source"].map(lambda s: _short(s, 34))
    cols = (["asset", "source", "start", "end", "days", "v1", "v21", "v63"] + [f"r{k}" for k in kk]
            + ["z1", "z10", "z63", "ac1"])
    L.append(_md(show[cols], fmt={"days": "{:.0f}", "z1": "{:+.1f}", "z10": "{:+.1f}", "z63": "{:+.1f}"},
                 default="{:.2f}"))
    L += ["", "### Around each splice", "",
          "J_k = [vol_k(after)/vol_k(before)] / [vol_21(after)/vol_21(before)] on up to 504 days either side of "
          "the splice; J_k = 1 means the splice does not change how k-day vol relates to 21-day vol. This is "
          "local and noisy (two years each side; the iid standard error of J_1 is about "
          f"{noise['se_r'][1] * np.sqrt(2 * noise['n'] / 504):.2f}), and genuine regime changes enter: the "
          "commodities 2009 splice joins two versions of the SAME index (BCOM ER + T-bill vs BCOMTR) and serves as a "
          "placebo; `vol21` = the 21-day vol change across the splice (2008 sits right after the HYG splice).", ""]
    if len(jump_t):
        js = jump_t.copy()
        js["before"] = js["before"].map(lambda s: _short(s, 26))
        js["after"] = js["after"].map(lambda s: _short(s, 26))
        L.append(_md(js[["asset", "splice", "before", "after", "window"] + [f"J{k}" for k in kk]
                        + ["vol21_after_over_before"]].rename(columns={"vol21_after_over_before": "vol21"}),
                     fmt={"window": "{:.0f}"}, default="{:.2f}"))
    L += ["", "## (b) Correlation by horizon, per pair and era", "",
          "An era is a span in which neither asset changes source (≥ 252 days shown). `c_k` = correlation of "
          "overlapping k-day log excess returns; `zd21` = (c_21 − c_63) in iid standard errors for the era length.", ""]
    ps = pair_t.copy()
    ps["src_a"] = ps["src_a"].map(lambda s: _short(s, 24))
    ps["src_b"] = ps["src_b"].map(lambda s: _short(s, 24))
    L.append(_md(ps[["pair", "src_a", "src_b", "start", "end", "days"] + [f"c{k}" for k in HORIZONS] + ["zd3", "zd21"]],
                 fmt={"days": "{:.0f}", "zd3": "{:+.1f}", "zd21": "{:+.1f}"}, default="{:.2f}"))
    req = [f"{a}–{b}" for a, b in PAIRS[:9]]
    absent = [p for p in req if p not in set(pair_t["pair"])]
    if absent:
        L += ["", "Requested pairs not computable yet (asset missing): " + ", ".join(absent) + "."]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L) + "\n")
    return dict(vol_sum=vol_sum.set_index("k"), corr_sum=corr_sum.set_index("k"), spread=spread, se=se,
                big=big, pt=pt, noise=noise)


# ============================================================== main
def build(write: bool = True, evidence: bool = True, verbose: bool = True) -> dict:
    cash_d, cash_m, assets = load_inputs()
    A = assemble(cash_d, cash_m, assets)
    seg, seg_issues = build_segments(A, assets, cash_d, cash_m)
    pred, catalog, pred_issues, conv = build_predictors(A.daily.index[-1])
    if write:
        PROCESSED.mkdir(parents=True, exist_ok=True)
        A.daily.astype(float).to_parquet(PROCESSED / "panel_daily.parquet")
        A.monthly.astype(float).to_parquet(PROCESSED / "panel_monthly.parquet")
        A.cash_daily.to_frame().to_parquet(PROCESSED / "cash_daily.parquet")
        A.cash_monthly.to_frame().to_parquet(PROCESSED / "cash_monthly.parquet")
        A.tiers.astype(object).where(A.tiers.notna(), None).to_parquet(PROCESSED / "tiers_monthly.parquet")
        seg.to_csv(PROCESSED / "segments.csv", index=False)
        pred.to_parquet(PROCESSED / "predictors.parquet", index=False)
        catalog.to_csv(PROCESSED / "predictors_catalog.csv", index=False)
    v = validate(A, seg, pred, catalog, cash_d.index)
    smoke = panel_asof_smoke() if write else None
    out = dict(A=A, segments=seg, seg_issues=seg_issues, predictors=pred, catalog=catalog,
               pred_issues=pred_issues, conversions=conv, validation=v, smoke=smoke)
    if write:
        write_panel_report(A, seg, catalog, v, seg_issues, pred_issues, conv, smoke, REPORTS / "data" / "panel.md")
    if evidence:
        X, S = log_excess_daily(A)
        seg_t = segment_vol_table(X, S)
        jump_t = splice_jump_table(X, S)
        pair_t = pair_corr_table(X, S)
        se_t = simulate_rv_se()
        noise = simulate_ratio_noise()
        add_noise_z(seg_t, pair_t, noise)
        out.update(seg_vol=seg_t, splice_jumps=jump_t, pair_corr=pair_t, se=se_t, noise=noise)
        if write:
            ev = write_evidence_report(seg_t, jump_t, pair_t, se_t, noise, A,
                                       REPORTS / "data" / "horizon_evidence.md")
            out["evidence"] = ev
            _insert_narrative(REPORTS / "data" / "horizon_evidence.md", ev, seg_t, jump_t, pair_t)
    if verbose:
        _print_summary(out)
    return out


def _insert_narrative(path: Path, ev: dict, seg_t, jump_t, pair_t) -> None:
    """Plain-language reading of the tables; every number is pulled from them."""
    vs, cs, se, big, pt, spread = ev["vol_sum"], ev["corr_sum"], ev["se"], ev["big"], ev["pt"], ev["spread"]
    lo1 = big.sort_values("r1").head(4)
    hi1 = big.sort_values("r1", ascending=False).head(4)
    jumps1 = spread["r1"].sort_values(ascending=False).head(4) if len(spread) else pd.Series(dtype=float)
    f63 = big[big["z63"].abs() > 2].sort_values("z63", ascending=False)
    f10 = big[big["z10"].abs() > 2].sort_values("z10")
    p21 = pt[pt["zd21"].abs() > 2].sort_values("zd21")
    p3 = pt.reindex(pt["d3"].sort_values().index).head(4)

    def seglab(r):
        return f"{r.asset} {_short(r.source, 24)} ({r.start.year}-{r.end.year})"

    t = ["## What the tables show", ""]
    t.append(f"**1. One-day vol depends on the pricing convention.** Across segments vol_1/vol_21 runs from "
             f"{vs.loc[1, 'min']:.2f} to {vs.loc[1, 'max']:.2f}. Low: "
             + "; ".join(f"{seglab(r)} {r.r1:.2f}" for r in lo1.itertuples())
             + " (fund NAVs, quoted-yield reconstructions, foreign-close indices: smoothed or stale, positive "
               "autocorrelation). High: "
             + "; ".join(f"{seglab(r)} {r.r1:.2f}" for r in hi1.itertuples())
             + " (US-listed ETFs: overshoot and reversal, negative autocorrelation). Within one asset the ratio "
               "changes at a splice by up to "
             + ", ".join(f"{a} {x:.2f}" for a, x in jumps1.items())
             + ": a daily-RV history spliced across these sources has artificial level shifts of 20-50%.")
    t.append("")
    t.append("**2. Where the jumps stop.** The share of segments whose k-day vol differs from their 21-day vol by "
             "more than two iid standard errors is "
             + ", ".join(f"{vs.loc[k, 'share_signif']:.0%} at k={k}" for k in (1, 2, 3, 5, 10))
             + f"; the median within-asset spread of r_k falls from {vs.loc[1, 'spread_median']:.2f} (k=1) to "
               f"{vs.loc[3, 'spread_median']:.2f} (k=3), {vs.loc[5, 'spread_median']:.2f} (k=5) and "
               f"{vs.loc[10, 'spread_median']:.2f} (k=10, max {vs.loc[10, 'spread_max']:.2f}). So the convention "
               "effect is large at 1-3 days, still clearly present at 5, and mostly but not entirely gone at 10. "
             + f"The segments still significant at k=10 ({len(f10)}; latest one ends {max([r.end.year for r in f10.itertuples()], default='-')}) are "
             + "; ".join(f"{seglab(r)} r10={r.r10:.2f}, ac1={r.ac1:+.2f}" for r in f10.itertuples())
             + ". Beyond 21 days, r_63 is significantly different from 1 in "
             f"{vs.loc[63, 'share_signif']:.0%} of segments (median {vs.loc[63, 'median']:.2f}, range "
             f"{vs.loc[63, 'min']:.2f}-{vs.loc[63, 'max']:.2f}). The splice table (J_k) tells the same story more "
             "noisily; note the placebo splice in commodities (same index both sides) has J values as far from 1 as "
             "real splices, so single-splice J values should not be over-read.")
    t.append("")
    t.append(f"**3. Correlation bias.** Relative to 63-day correlations, the median gap is "
             f"{cs.loc[1, 'median']:+.2f} at k=1, {cs.loc[3, 'median']:+.2f} at k=3, {cs.loc[10, 'median']:+.2f} "
             f"at k=10 and {cs.loc[21, 'median']:+.2f} at k=21 (median absolute gap {cs.loc[3, 'median_abs']:.2f} / "
             f"{cs.loc[10, 'median_abs']:.2f} / {cs.loc[21, 'median_abs']:.2f}; most negative "
             f"{cs.loc[3, 'min']:+.2f} / {cs.loc[10, 'min']:+.2f} / {cs.loc[21, 'min']:+.2f}). The iid standard error "
             f"of c_k − c_63 over a 10-year era is ≈{cs.loc[21, 'iid_se_10y']:.2f}-{cs.loc[1, 'iid_se_10y']:.2f}, so "
             "typical gaps are within noise, but the stale-price pairs are not: at k=3 the worst are "
             + "; ".join(f"{r.pair} [{_short(r.src_a, 14)}/{_short(r.src_b, 14)} {r.start.year}-{r.end.year}] "
                         f"{r.c3:.2f} vs {r.c63:.2f}" for r in p3.itertuples())
             + ". Three-day sums (the earlier project convention) remove only part of that bias.")
    t.append("")
    t.append(f"**4. Statistical cost.** The relative standard error of a 12-month realized vol is "
             f"{se.loc[1, 'se_vol_rel']:.1%} from 1-day returns, {se.loc[3, 'se_vol_rel']:.1%} at k=3, "
             f"{se.loc[5, 'se_vol_rel']:.1%} at k=5, {se.loc[10, 'se_vol_rel']:.1%} at k=10, "
             f"{se.loc[21, 'se_vol_rel']:.1%} at k=21 and {se.loc[63, 'se_vol_rel']:.1%} at k=63 (iid normal; "
             "fat tails raise the short-k numbers most). A 12-month correlation of 0.5 has a standard error of "
             f"{se.loc[1, 'se_corr']:.2f} at k=1, {se.loc[10, 'se_corr']:.2f} at k=10 and {se.loc[21, 'se_corr']:.2f} "
             f"at k=21. The 21-day target is about {se.loc[21, 'se_vol_rel'] / se.loc[1, 'se_vol_rel']:.1f}x noisier "
             "than a daily-RV target: a real loss of power for ranking forecasts, but unbiased (small-sample bias "
             f"{se.loc[21, 'bias_vol']:+.1%} on vol at k=21).")
    t.append("")
    t.append("## Does anything contradict the choice of 21?")
    t.append("")
    t.append("Nothing argues for a SHORTER horizon: at k ≤ 5 the convention effects are large and pervasive, and "
             "they are what the target is designed to remove. k=10 removes most of them at "
             f"{se.loc[10, 'se_vol_rel']:.0%} instead of {se.loc[21, 'se_vol_rel']:.0%} standard error, which would be "
             "a defensible trade only if the smoothed eras below were handled separately.")
    t.append("")
    t.append("Some evidence says 21 days is NOT ALWAYS LONG ENOUGH:")
    t.append("")
    if len(f63):
        t.append("- 63-day vol differs significantly from 21-day vol in: "
                 + "; ".join(f"{seglab(r)} r63={r.r63:.2f} (z {r.z63:+.1f})" for r in f63.itertuples())
                 + ". Values above 1 mean variance keeps building beyond a month (smoothing that lasts longer than "
                   "a month, as the credit report documents for VWEHX, or persistent trends); values below 1 mean "
                   "mean-reversion over months. The 21-day target "
                   "is still the right quantity for `cov_rv`, but for the 12-month variance the bridge "
                   "(`assetforecast.bridge`) must carry era-specific variance ratios for these segments.")
    if len(p21):
        t.append("- 21-day correlations still differ significantly from 63-day ones in: "
                 + "; ".join(f"{r.pair} [{_short(r.src_a, 14)}/{_short(r.src_b, 14)} {r.start.year}-{r.end.year}] "
                             f"{r.c21:.2f} vs {r.c63:.2f} (z {r.zd21:+.1f})" for r in p21.itertuples())
                 + ". Whether such a gap is a measurement artefact or a genuine horizon effect (stock-bond "
                   "correlation, for instance, is known to differ between monthly and quarterly horizons) has to be "
                   "judged pair by pair.")
    big21 = pt.reindex(pt["d21"].sort_values().index).head(3)
    t.append("- The largest 21-day gaps in absolute terms (not all significant, because the eras are short) are "
             + "; ".join(f"{r.pair} [{_short(r.src_a, 14)}/{_short(r.src_b, 14)} {r.start.year}-{r.end.year}] "
                         f"{r.c21:.2f} vs {r.c63:.2f}" for r in big21.itertuples())
             + ". Where one side is a smoothed NAV (VWEHX for high yield before 2007-04 in particular), "
               "correlations are attenuated even at 21 days; models fitted on those eras will understate "
               "co-movement unless they correct for it.")
    t.append("- The z-scores assume iid returns; with volatility clustering the true noise is larger, so the "
             "lists above are, if anything, too long rather than too short.")
    txt = path.read_text().replace("<!-- NARRATIVE -->", "\n".join(t))
    path.write_text(txt)


def _print_summary(out: dict) -> None:
    A, v = out["A"], out["validation"]
    print(f"daily   {A.daily.index[0].date()} .. {A.daily.index[-1].date()}  {A.daily.shape}")
    print(f"monthly {A.monthly.index[0].date()} .. {A.monthly.index[-1].date()}  {A.monthly.shape}")
    print("assets:", ", ".join(A.monthly.columns))
    missing = [a for a in ASSET_IDS if a not in A.monthly.columns]
    if missing:
        print("MISSING:", ", ".join(missing))
    for a, c in v["calendar"].items():
        mv = A.checks["monthly_vs_track"].get(a, {})
        print(f"  {a:12s} folded {c['n_folded']:4d} (across month {c['n_folded_across_month']}) "
              f"NaN-in-span {c.get('n_nan_in_span', 0):4d}  monthly vs track max {mv.get('max_abs_vs_track', np.nan):.2e}")
    for k in ["daily_index_equals_cash_calendar", "n_ret_le_minus1", "last_month", "no_partial_final_month",
              "excess_nan_from_cash_daily", "excess_nan_from_cash_monthly", "pred_available_ge_obs",
              "pred_dup_series_obs"]:
        print(f"  {k}: {v[k]}")
    if out["seg_issues"]:
        print("segment issues:")
        for s in out["seg_issues"]:
            print("  -", s)
    if out["pred_issues"]:
        print("predictor issues:", *out["pred_issues"], sep="\n  - ")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--no-evidence", action="store_true", help="skip reports/data/horizon_evidence.md")
    a = ap.parse_args(argv)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        build(evidence=not a.no_evidence)


if __name__ == "__main__":
    main()
