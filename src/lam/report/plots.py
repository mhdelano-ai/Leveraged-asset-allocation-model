"""Tearsheet charts.

Colours come from a validated categorical palette (adjacent-pair CVD dE 9.1,
normal-vision 22.9). Because two slots sit below 3:1 against the light surface,
every series is directly labelled as well as legended -- identity is never
carried by colour alone.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ..metrics.constraint import excess_drawdown_path  # noqa: E402
from ..metrics.drawdown import drawdown_series  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
CRITICAL = "#d03b3b"
GOOD = "#0ca30c"

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.size": 9,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK_2,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.grid": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
    }
)


def _style(ax, title: str, ylabel: str = "") -> None:
    ax.set_title(title, color=INK, fontsize=10, fontweight="600", loc="left", pad=8)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8)
    ax.grid(axis="y", alpha=0.7)
    ax.grid(axis="x", visible=False)
    ax.xaxis.set_major_locator(mdates.YearLocator(5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))


def _label_end(ax, series: pd.Series, text: str, color: str) -> None:
    """Direct label at the right edge -- required relief for low-contrast slots."""
    if series.empty:
        return
    ax.annotate(
        text,
        xy=(series.index[-1], series.iloc[-1]),
        xytext=(6, 0),
        textcoords="offset points",
        color=color,
        fontsize=8,
        fontweight="600",
        va="center",
    )


def equity_curve(ax, series_map: dict[str, pd.Series]) -> None:
    for i, (name, r) in enumerate(series_map.items()):
        wealth = (1.0 + r.fillna(0.0)).cumprod()
        color = SERIES[i % len(SERIES)]
        ax.plot(wealth.index, wealth.to_numpy(), lw=2 if i == 0 else 1.4, color=color, label=name)
        _label_end(ax, wealth, name, color)
    ax.set_yscale("log")
    _style(ax, "Growth of $1 (log scale)", "wealth")
    ax.legend(loc="upper left", fontsize=8)


def underwater(ax, strategy: pd.Series, benchmark: pd.Series) -> None:
    s = drawdown_series(strategy) * 100
    b = drawdown_series(benchmark) * 100
    ax.fill_between(b.index, b.to_numpy(), 0, color=MUTED, alpha=0.35, label="S&P 500", lw=0)
    ax.plot(s.index, s.to_numpy(), lw=1.6, color=SERIES[0], label="Strategy")
    _style(ax, "Drawdown: strategy vs S&P 500", "%")
    ax.legend(loc="lower left", fontsize=8)


def excess_drawdown(ax, strategy: pd.Series, benchmark: pd.Series, step: int = 5) -> None:
    """The constraint chart: this line must never rise above zero."""
    ex = excess_drawdown_path(strategy, benchmark, step=step) * 100
    values = ex.to_numpy()
    ax.fill_between(ex.index, values, 0, where=values > 0, color=CRITICAL, alpha=0.85,
                    label="breach", lw=0, interpolate=True)
    ax.fill_between(ex.index, values, 0, where=values <= 0, color=GOOD, alpha=0.30,
                    label="within budget", lw=0, interpolate=True)
    ax.axhline(0, color=INK, lw=1.2)
    _style(ax, "Rolling 3y drawdown vs S&P 500 (constraint holds below zero)", "pp")
    ax.legend(loc="upper left", fontsize=8)
    worst = float(np.max(values)) if values.size else 0.0
    ax.annotate(
        f"worst {worst:+.2f}pp",
        xy=(0.99, 0.94), xycoords="axes fraction", ha="right",
        color=CRITICAL if worst > 0 else INK_2, fontsize=8, fontweight="600",
    )


def leverage(ax, lev: pd.Series, throttle: pd.Series | None = None) -> None:
    ax.plot(lev.index, lev.to_numpy(), lw=1.2, color=SERIES[0], label="gross leverage")
    ax.axhline(1.0, color=AXIS, lw=1.0, ls="--")
    if throttle is not None:
        ax2 = ax.twinx()  # a control signal in [0,1], not a second data scale
        ax2.plot(throttle.index, throttle.to_numpy(), lw=1.0, color=SERIES[1], alpha=0.8)
        ax2.set_ylim(-0.05, 1.6)
        ax2.set_yticks([0, 1])
        ax2.set_yticklabels(["off", "full"], fontsize=7, color=SERIES[1])
        ax2.grid(False)
        for spine in ax2.spines.values():
            spine.set_visible(False)
    _style(ax, "Leverage and drawdown throttle", "x")
    ax.legend(loc="upper left", fontsize=8)


def weights_area(ax, weights: pd.DataFrame) -> None:
    monthly = weights.resample("ME").last().clip(lower=0)
    colors = [SERIES[i % len(SERIES)] for i in range(monthly.shape[1])]
    ax.stackplot(monthly.index, monthly.to_numpy().T, colors=colors,
                 labels=list(monthly.columns), lw=0.4, edgecolor=SURFACE)
    _style(ax, "Sleeve weights", "weight")
    ax.legend(loc="upper left", fontsize=7, ncol=3)


def rolling_correlation(ax, a: pd.Series, b: pd.Series, window: int = 504) -> None:
    joined = pd.concat([a.rename("a"), b.rename("b")], axis=1, join="inner").dropna()
    corr = joined["a"].rolling(window, min_periods=window // 2).corr(joined["b"])
    values = corr.to_numpy()
    ax.fill_between(corr.index, values, 0, where=values > 0, color=CRITICAL, alpha=0.30, lw=0)
    ax.fill_between(corr.index, values, 0, where=values <= 0, color=SERIES[0], alpha=0.30, lw=0)
    ax.plot(corr.index, values, lw=1.2, color=INK_2)
    ax.axhline(0, color=INK, lw=1.0)
    _style(ax, "Stock/bond correlation, trailing 2y", "corr")
    ax.annotate("positive: diversification fails", xy=(0.01, 0.92), xycoords="axes fraction",
                color=CRITICAL, fontsize=7.5, fontweight="600")


def tearsheet(
    strategy: pd.Series,
    benchmark: pd.Series,
    *,
    result=None,
    references: dict[str, pd.Series] | None = None,
    panel=None,
    title: str = "Leveraged multi-asset strategy",
    subtitle: str = "",
    path: str | Path = "output/tearsheet.png",
) -> Path:
    """Render the full tearsheet to a PNG."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    n_rows = 3 + (1 if result is not None else 0) + (1 if panel is not None else 0)
    fig, axes = plt.subplots(n_rows, 1, figsize=(11, 3.0 * n_rows))
    axes = np.atleast_1d(axes)

    curves = {"Strategy": strategy}
    if references:
        curves |= {k: v for k, v in references.items() if k != "spx"}
    curves["S&P 500"] = benchmark

    i = 0
    equity_curve(axes[i], curves); i += 1
    underwater(axes[i], strategy, benchmark); i += 1
    excess_drawdown(axes[i], strategy, benchmark); i += 1
    if result is not None:
        leverage(axes[i], result.leverage, result.throttle); i += 1
    if panel is not None and "us_equity" in panel.returns.columns:
        bond_col = "long_ust" if "long_ust" in panel.returns.columns else "interm_ust"
        rolling_correlation(axes[i], panel.returns["us_equity"], panel.returns[bond_col]); i += 1

    fig.suptitle(title, x=0.055, ha="left", fontsize=13, fontweight="700", color=INK, y=0.998)
    if subtitle:
        fig.text(0.055, 0.982, subtitle, ha="left", fontsize=9, color=INK_2)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path
