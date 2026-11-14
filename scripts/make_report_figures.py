"""Regenerate the figures in reports/figures/ used by reports/REPORT.md.

    python scripts/make_report_figures.py

Reads only results/final/*.csv and forecasts/latest/vols.csv (no model is run).
Writes five PNGs:

    fig1_forecast_vs_realized.png   12-month vol, forecast vs realized, four assets, holdout shaded
    fig2_current_forecast.png       current forecast per asset with the range across model families
    fig3_rank_dev_vs_holdout.png    development rank vs holdout rank of the 70 volatility models
    fig4_stock_bond_corr.png        us_large-ust_int correlation, forecast vs realized
    fig5_6040_risk.png              predicted vs realized vol of the 60/40 portfolio
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "results" / "final"
LATEST = ROOT / "forecasts" / "latest"
OUT = ROOT / "reports" / "figures"

# Colour-blind-safe categorical slots (validated reference palette) plus neutral inks.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
HOLDOUT_START, HOLDOUT_END = pd.Timestamp("2014-01-31"), pd.Timestamp("2025-10-31")

NAMES = {
    "us_large": "US large cap", "us_small": "US small cap", "intl_dev": "Intl developed",
    "em": "Emerging mkts", "ust_int": "Treasuries 7-10y", "ust_long": "Treasuries 20-30y",
    "tips": "TIPS", "ig_credit": "IG credit", "hy_credit": "High yield",
    "reits": "REITs", "gold": "Gold", "commodities": "Commodities",
}

plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "figure.dpi": 100, "savefig.dpi": 150, "savefig.bbox": "tight",
    "legend.frameon": False,
})


def _shade_holdout(ax, label=True):
    ax.axvspan(HOLDOUT_START, HOLDOUT_END, color="#f1efe8", zorder=0, lw=0)
    if label:
        ax.text(HOLDOUT_START + (HOLDOUT_END - HOLDOUT_START) / 2, 0.97, "holdout",
                transform=ax.get_xaxis_transform(), ha="center", va="top", color=INK2, fontsize=9)


def _pct(ax, axis="y", decimals=0):
    fmt = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v * 100:.{decimals}f}%")
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def fig1():
    d = pd.read_csv(FINAL / "dashboard_vol.csv")
    d["date"] = pd.to_datetime(d["origin"]) + pd.offsets.MonthEnd(0)
    assets = ["us_large", "ust_long", "hy_credit", "gold"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=False)
    for ax, a in zip(axes.flat, assets):
        x = d[(d.asset == a) & (d.date >= "1926-01-01")]
        _shade_holdout(ax)
        r = x.dropna(subset=["realized"])
        ax.plot(r.date, r.realized, color=MUTED, lw=1.2, label="realized (next 12 months)")
        ax.plot(x.date, x.primary, color=BLUE, lw=1.6, label="forecast (selected model)")
        live = x[x.date == x.date.max()]
        ax.plot(live.date, live.primary, "o", color=BLUE, ms=7, mec="white", mew=1.5)
        ax.set_title(NAMES[a], loc="left", color=INK)
        ax.set_ylim(0, None)
        _pct(ax)
        ax.xaxis.set_major_locator(mdates.YearLocator(20))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[0, 0].set_ylabel("annualized vol, 21-day returns")
    axes[1, 0].set_ylabel("annualized vol, 21-day returns")
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02))
    fig.text(0.5, -0.01, "x-axis: forecast origin (start of the 12-month window). Dot = current forecast "
             "(Nov 2026 - Oct 2027).", ha="center", color=INK2, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT / "fig1_forecast_vs_realized.png")
    plt.close(fig)


def fig2():
    v = pd.read_csv(LATEST / "vols.csv").set_index("asset")
    v = v.iloc[::-1]  # first asset at the top
    y = np.arange(len(v))
    fig, ax = plt.subplots(figsize=(9, 6.5))
    ax.hlines(y, v.family_p10, v.family_p90, color=BLUE, alpha=0.35, lw=7,
              label="range across model families (10th-90th pct., 21-day basis)")
    ax.plot(v.realized_long_run, y, "|", color=INK, ms=14, mew=2, label="long-run realized (whole history)")
    ax.plot(v.realized_last_12m, y, "o", mfc="white", mec=INK2, mew=1.5, ms=8,
            label="realized, Nov 2025 - Oct 2026")
    ax.plot(v.vol_21d_horizon, y, "o", color=BLUE, ms=8, label="forecast, 21-day-return basis")
    ax.plot(v.vol_12m_return, y, "D", color=ORANGE, ms=7, label="forecast, vol of the 12-month return")
    ax.set_yticks(y, [NAMES[a] for a in v.index])
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, None)
    _pct(ax, "x")
    ax.set_xlabel("annualized volatility of log excess returns")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_current_forecast.png")
    plt.close(fig)


def fig3():
    d = pd.read_csv(FINAL / "vol_leaderboard.csv")
    s = d[(d.era == "all") & (d.subset == "daily")]
    m = (s[s.period == "dev"].set_index("model")[["rank"]]
         .join(s[s.period == "holdout"].set_index("model")[["rank"]], rsuffix="_h"))
    rho = m["rank"].corr(m["rank_h"], method="spearman")
    fig, ax = plt.subplots(figsize=(8, 7.5))
    ax.plot([1, 70], [1, 70], color=MUTED, lw=1, ls="--")
    ax.text(66, 61, "same rank in\nboth periods", color=INK2, fontsize=9, ha="right")
    ax.scatter(m["rank"], m["rank_h"], s=36, color="#c3c2b7", edgecolor="white", lw=0.8, zorder=2)
    marks = {
        "combo_vol_hier_trim": ("Selected model\n(trimmed family average)", BLUE, (12, 2)),
        "statespace_2c_daily": ("Best single model\nin development", ORANGE, (3, 67)),
        "combo_vol_cal_m_hl12": ("Simple runner-up", AQUA, (24, 45)),
        "ref_blend": ("Reference\n(long-run/5-yr blend)", INK, (52, 28)),
        "ref_last12m": ("Reference\n(last 12 months)", INK, (36, 64)),
        "bench_vol_m_hl120": ("10-year average", INK2, (55, 8)),
        "garch_d_gjr": ("GJR-GARCH", INK2, (50, -1)),
    }
    for mdl, (lab, col, xy) in marks.items():
        px, py = m.loc[mdl, "rank"], m.loc[mdl, "rank_h"]
        ax.scatter(px, py, s=90, color=col, edgecolor="white", lw=1.5, zorder=3)
        ax.annotate(f"{lab}: {int(px)} → {int(py)}", (px, py), xytext=xy, fontsize=9, color=INK,
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    ax.set_xlim(0, 72)
    ax.set_ylim(72, -4)
    ax.set_xlabel("rank in development, 1926-2013 (1 = lowest loss)")
    ax.set_ylabel("rank on the holdout, 2014-2025")
    ax.set_title(f"70 volatility models: rank correlation {rho:+.2f}".replace("-", "\u2212"), loc="left", color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_rank_dev_vs_holdout.png")
    plt.close(fig)


def fig4():
    c = pd.read_csv(FINAL / "dashboard_corr.csv")
    c = c[c.pair == "us_large~ust_int"].copy()
    c["date"] = pd.to_datetime(c["origin"]) + pd.offsets.MonthEnd(0)
    c = c[c.date >= "1962-01-01"]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    _shade_holdout(ax)
    ax.axhline(0, color=MUTED, lw=1)
    r = c.dropna(subset=["realized"])
    ax.plot(r.date, r.realized, color=MUTED, lw=1.2, label="realized (next 12 months, 21-day returns)")
    ax.plot(c.date, c.primary, color=BLUE, lw=1.8, label="forecast (selected model)")
    ax.set_ylim(-1, 1)
    ax.set_ylabel("correlation")
    ax.set_title("US large cap vs Treasuries 7-10y", loc="left", color=INK)
    ax.legend(loc="lower left", ncol=2, fontsize=9)
    ax.xaxis.set_major_locator(mdates.YearLocator(10))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    fig.savefig(OUT / "fig4_stock_bond_corr.png")
    plt.close(fig)


def fig5():
    p = pd.read_csv(FINAL / "dashboard_portfolio.csv")
    p = p[p.portfolio == "p6040"].copy()
    p["date"] = pd.to_datetime(p["origin"]) + pd.offsets.MonthEnd(0)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    _shade_holdout(ax)
    ax.plot(p.date, p.realized, color=MUTED, lw=1.2, label="realized (next 12 months)")
    ax.plot(p.date, p.primary, color=BLUE, lw=1.8, label="predicted (selected model)")
    ax.set_ylim(0, None)
    _pct(ax)
    ax.set_ylabel("annualized vol, 21-day returns")
    ax.set_title("60% US large cap / 40% Treasuries 7-10y", loc="left", color=INK)
    ax.legend(loc="upper left", ncol=2, fontsize=9)
    ax.xaxis.set_major_locator(mdates.YearLocator(10))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    fig.savefig(OUT / "fig5_6040_risk.png")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for f in (fig1, fig2, fig3, fig4, fig5):
        f()
        print("wrote", f.__name__)


if __name__ == "__main__":
    main()
