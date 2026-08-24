"""Forward-looking capital market assumptions, and Kelly weights derived from them.

The backtest answers "what was log-optimal", and the honest reading of it was
that 33 years of monthly data cannot identify four risk premia. Historical means
are not merely noisy here -- they are *biased for the purpose*, because the sample
contains a 40-year fall in yields that cannot repeat and a re-rating of US
equities that shows up as return but is a change in price paid, not in cash flow
earned.

So this module estimates expected returns the way an allocator has to: from what
the assets currently yield.

**Bonds.** Starting yield to maturity explains the great majority of the
subsequent 5-10 year return of a constant-maturity bond index, because the pull
to par offsets price moves as the position rolls. Expected return is the index
YTM, less expected credit loss. For currency-hedged foreign bonds it is the
foreign yield *plus the hedge carry* -- rolling a short-dated FX forward earns
roughly the short-rate differential, which in 2026 is large and positive for a
dollar investor.

**Equities.** The Grinold-Kroner decomposition::

    E[r] = dividend yield + net buyback yield + real earnings growth
           + inflation + valuation change

Every term is either observable today (the first, and inflation from the TIPS
breakeven) or an explicit assumption you can move (the rest). The valuation term
defaults to zero: assuming multiples revert is a forecast, not an input, and it
belongs in the sensitivity table rather than the base case.

**The growth term is aggregate, not per share, and the distinction is not
cosmetic.** ``real_growth`` here is the real growth of the *aggregate* earnings of
the index's constituents; the buyback yield is the separate share-count term that
converts it to a per-share figure. So what the holder of an index fund actually
earns in growth is ``real_growth + buyback`` -- 2.80%/yr for US equities in the
base case. Shiller's earnings-per-share series is already *per share*, so its
1.84%/yr long-run growth is the thing to compare 2.80% against, and it must never
be dropped into ``real_growth`` beside a buyback term: that double counts the
share count. :func:`historical_real_eps_growth` measures the comparison directly
so the assumption can be argued with rather than asserted.

**Risk.** Volatility and correlation are estimable from data in a way means are
not, so they come from the sample -- but from the *recent* sample by default. The
stock/bond correlation regime changed in 2022, and a levered book's tolerance for
bonds depends on it entirely.

One conversion is applied and must not be skipped: the build-up produces a
**geometric** expected return, and Kelly is defined on **arithmetic** means. The
gap is sigma^2/2, which for a 16%-volatility equity is 1.3pp -- larger than some
of the risk premia being estimated.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from ..data import fred, rates, yahoo
from .panel import ASSETS, PRETTY

# Foreign bond market weights, approximating the ex-USD global aggregate that
# BNDX tracks: euro area, Japan, UK, and an "other developed" bucket (Canada,
# Australia, Korea, the Nordics) proxied by the euro-area block.
FOREIGN_BLOCKS = {
    "euro": {"weight": 0.58, "long": "IRLTLT01DEM156N", "short": "IR3TIB01DEM156N"},
    "japan": {"weight": 0.22, "long": "IRLTLT01JPM156N", "short": "IR3TIB01JPM156N"},
    "uk": {"weight": 0.08, "long": "IRLTLT01GBM156N", "short": "IR3TIB01GBM156N"},
}
FOREIGN_OTHER_WEIGHT = 0.12  # priced off the euro block

# US aggregate bond index: ~6 years duration, so it is priced off the belly of
# the curve, plus the index option-adjusted spread over Treasuries.
US_AGG_DURATION_TENOR = 6.0
US_AGG_SPREAD = 0.0040
US_AGG_CREDIT_LOSS = 0.0015


@dataclass(frozen=True)
class MarketInputs:
    """Everything observable, fetched live. No judgement in this object."""

    asof: pd.Timestamp
    cash: float
    treasury: dict[str, float]
    breakeven_inflation: float
    distribution_yield: dict[str, float]
    foreign_long: dict[str, float]
    foreign_short: dict[str, float]

    def us_agg_ytm(self) -> float:
        """Yield of the US aggregate index: belly Treasury plus index spread."""
        return self._interpolate(US_AGG_DURATION_TENOR) + US_AGG_SPREAD

    def hedged_foreign_ytm(self) -> float:
        """Foreign yield plus the short-rate differential earned by the hedge."""
        total = 0.0
        for name, spec in FOREIGN_BLOCKS.items():
            carry = self.cash - self.foreign_short[name]
            total += spec["weight"] * (self.foreign_long[name] + carry)
        euro_leg = self.foreign_long["euro"] + (self.cash - self.foreign_short["euro"])
        return total + FOREIGN_OTHER_WEIGHT * euro_leg

    def _interpolate(self, years: float) -> float:
        tenors = sorted((float(k.rstrip("ym")) if k.endswith("y") else 0.25, v)
                        for k, v in self.treasury.items())
        xs = np.array([t for t, _ in tenors])
        ys = np.array([v for _, v in tenors])
        return float(np.interp(years, xs, ys))

    def __str__(self) -> str:
        curve = "  ".join(f"{k} {v:.2%}" for k, v in sorted(self.treasury.items()))
        return (
            f"market inputs as of {self.asof:%Y-%m-%d}\n"
            f"  cash (3m bill, BEY)      {self.cash:.2%}\n"
            f"  Treasury curve           {curve}\n"
            f"  10y breakeven inflation  {self.breakeven_inflation:.2%}\n"
            f"  US agg YTM (6y + 40bp)   {self.us_agg_ytm():.2%}\n"
            f"  hedged foreign bond YTM  {self.hedged_foreign_ytm():.2%}\n"
            + "".join(
                f"  {PRETTY[a]:<22s} distribution yield {self.distribution_yield[a]:.2%}\n"
                for a in ASSETS
            )
        )


@dataclass(frozen=True)
class Assumptions:
    """Judgement, stated as numbers you can move one at a time.

    ``*_real_growth`` is **aggregate** real earnings growth; ``*_buyback`` is the
    net share-count change that turns it into per-share growth. Their sum is what
    to compare against a historical earnings-per-share growth rate --- see
    :func:`historical_real_eps_growth` and ``ForwardCMA.implied_eps_growth``.

    The 1.50% default is deliberately below the ~3.3%/yr real growth the US
    corporate sector delivered in aggregate since 1871, on two grounds: potential
    real GDP growth is now nearer 1.8% than the 3.3% of that era, and the profit
    share of GDP is at a record high, so the margin expansion that lifted earnings
    faster than output cannot be extrapolated. Paired with the buyback term it
    still implies 2.80%/yr of per-share growth, which is *above* Shiller's 1.84%
    long-run median --- the base case is not a pessimistic one.
    """

    us_buyback: float = 0.0130
    us_real_growth: float = 0.0150
    us_valuation: float = 0.0
    intl_buyback: float = 0.0060
    intl_real_growth: float = 0.0150
    intl_valuation: float = 0.0
    intl_currency: float = 0.0
    us_bond_credit_loss: float = US_AGG_CREDIT_LOSS
    intl_bond_credit_loss: float = 0.0005
    # Risk is measured, not assumed; this is the window it is measured over.
    risk_window_years: float = 5.0
    vol_floor: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ForwardCMA:
    """Expected returns and risk, ready for a Kelly solver."""

    asof: pd.Timestamp
    cash: float
    geometric: pd.Series
    arithmetic: pd.Series
    vol: pd.Series
    corr: pd.DataFrame
    build_up: pd.DataFrame

    @property
    def excess(self) -> pd.Series:
        return (self.arithmetic - self.cash).rename("excess")

    @property
    def cov(self) -> pd.DataFrame:
        sigma = np.outer(self.vol.to_numpy(), self.vol.to_numpy()) * self.corr.to_numpy()
        return pd.DataFrame(sigma, index=ASSETS, columns=ASSETS)

    @property
    def sharpe(self) -> pd.Series:
        return (self.excess / self.vol).rename("sharpe")

    @property
    def implied_eps_growth(self) -> pd.Series:
        """Per-share real earnings growth implied by the build-up.

        Aggregate growth plus the buyback yield. This is the number that is
        comparable with a historical earnings-per-share growth rate, and the one
        an equity assumption should be defended on.
        """
        growth = self.build_up["real growth"] + self.build_up["buyback"]
        return growth.rename("implied_real_eps_growth")

    def table(self) -> pd.DataFrame:
        out = self.build_up.copy()
        out["geometric"] = self.geometric
        out["volatility"] = self.vol
        out["arithmetic"] = self.arithmetic
        out["excess"] = self.excess
        out["sharpe"] = self.sharpe
        out.index = [PRETTY[a] for a in out.index]
        return out


def _distribution_yield(symbol: str, years: float = 1.0) -> float:
    """Trailing distribution yield, from the gap between total and price return.

    Yahoo gives both an adjusted (total return) and an unadjusted close; their
    ratio over a year is the income the fund paid. It needs no separate data
    source and it is the fund's actual distribution, not a quoted SEC yield.
    """
    total = yahoo.prices(symbol, field="adjclose")
    price = yahoo.prices(symbol, field="close")
    joined = pd.concat([total.rename("tr"), price.rename("px")], axis=1).dropna()
    window = joined.loc[joined.index[-1] - pd.DateOffset(years=int(years)) :]
    tr = window["tr"].iloc[-1] / window["tr"].iloc[0]
    px = window["px"].iloc[-1] / window["px"].iloc[0]
    return float((tr / px) ** (1.0 / years) - 1.0)


def market_inputs(*, tickers: dict[str, str] | None = None) -> MarketInputs:
    """Fetch every observable the assumptions are built from."""
    tickers = tickers or {
        "us_equity": "VTI",
        "intl_equity": "VXUS",
        "us_bonds": "BND",
        "intl_bonds": "BNDX",
    }
    bills = rates.bill_rate_bey(prefer="fred")
    curve = {
        tenor: float(rates.cmt_yield_pct(tenor, prefer="fred").iloc[-1]) / 100.0
        for tenor in ("5y", "10y", "30y")
    }
    curve["3m"] = float(bills.iloc[-1])

    foreign_long, foreign_short = {}, {}
    for name, spec in FOREIGN_BLOCKS.items():
        foreign_long[name] = float(fred.series(spec["long"]).iloc[-1]) / 100.0
        foreign_short[name] = float(fred.series(spec["short"]).iloc[-1]) / 100.0

    return MarketInputs(
        asof=pd.Timestamp(bills.index[-1]),
        cash=float(bills.iloc[-1]),
        treasury=curve,
        breakeven_inflation=float(fred.series("T10YIE").iloc[-1]) / 100.0,
        distribution_yield={a: _distribution_yield(t) for a, t in tickers.items()},
        foreign_long=foreign_long,
        foreign_short=foreign_short,
    )


def _measured_risk(panel, years: float) -> tuple[pd.Series, pd.DataFrame]:
    """Volatility and correlation from the most recent ``years`` of daily data."""
    if panel.daily is None:
        raise ValueError("panel has no daily data")
    window = panel.daily.loc[panel.daily.index[-1] - pd.DateOffset(years=int(years)) :]
    vol = (window.std(ddof=1) * np.sqrt(252)).rename("vol")
    return vol, window.corr()


def build_cma(
    panel,
    *,
    inputs: MarketInputs | None = None,
    assumptions: Assumptions | None = None,
) -> ForwardCMA:
    """Assemble forward expected returns and risk for the four-asset universe."""
    inputs = inputs or market_inputs()
    a = assumptions or Assumptions()
    infl = inputs.breakeven_inflation

    rows = {
        "us_equity": {
            "income": inputs.distribution_yield["us_equity"],
            "buyback": a.us_buyback,
            "real growth": a.us_real_growth,
            "inflation": infl,
            "valuation": a.us_valuation,
            "currency": 0.0,
        },
        "intl_equity": {
            "income": inputs.distribution_yield["intl_equity"],
            "buyback": a.intl_buyback,
            "real growth": a.intl_real_growth,
            "inflation": infl,
            "valuation": a.intl_valuation,
            "currency": a.intl_currency,
        },
        "us_bonds": {
            "income": inputs.us_agg_ytm(),
            "buyback": 0.0,
            "real growth": 0.0,
            "inflation": 0.0,
            "valuation": 0.0,
            "currency": -a.us_bond_credit_loss,
        },
        "intl_bonds": {
            "income": inputs.hedged_foreign_ytm(),
            "buyback": 0.0,
            "real growth": 0.0,
            "inflation": 0.0,
            "valuation": 0.0,
            "currency": -a.intl_bond_credit_loss,
        },
    }
    build_up = pd.DataFrame(rows).T.loc[ASSETS]
    geometric = build_up.sum(axis=1).rename("geometric")

    vol, corr = _measured_risk(panel, a.risk_window_years)
    for asset, floor in (a.vol_floor or {}).items():
        vol[asset] = max(float(vol[asset]), float(floor))

    # Kelly is defined on arithmetic means; the build-up is a compound rate.
    arithmetic = (geometric + 0.5 * vol**2).rename("arithmetic")

    return ForwardCMA(
        asof=inputs.asof,
        cash=inputs.cash,
        geometric=geometric,
        arithmetic=arithmetic,
        vol=vol,
        corr=corr,
        build_up=build_up,
    )


def kelly(cma: ForwardCMA, *, financing_spread: float = 0.0) -> pd.Series:
    """Unconstrained Kelly weights on forward inputs.

    With no financing charge this is the closed form ``Sigma^-1 mu``. With one it
    is the same objective less ``spread x borrowed``, which stays concave and is
    solved numerically -- the kink at 1x gross is exactly the point the answer
    tends to sit near, so it cannot be approximated away.
    """
    if not financing_spread:
        weights = np.linalg.solve(cma.cov.to_numpy(), cma.excess.to_numpy())
        return pd.Series(weights, index=ASSETS, name="forward_kelly")
    return constrained_kelly(
        cma, mode="unconstrained", financing_spread=financing_spread
    ).rename("forward_kelly")


def constrained_kelly(
    cma: ForwardCMA,
    *,
    mode: str = "long_only_unlevered",
    max_gross: float | None = None,
    financing_spread: float = 0.0,
) -> pd.Series:
    """Kelly on forward inputs under real-account constraints."""
    from scipy.optimize import minimize

    from .solve import _bounds_and_constraints

    mu = cma.excess.to_numpy()
    sigma = cma.cov.to_numpy()
    n = len(mu)
    bounds, cons = _bounds_and_constraints(mode, n, max_gross)

    def negative(w: np.ndarray) -> float:
        borrowed = max(np.abs(w).sum() - 1.0, 0.0)
        return -(w @ mu - 0.5 * w @ sigma @ w - borrowed * financing_spread)

    best, best_val = None, np.inf
    for start in (np.zeros(n), np.full(n, 0.25), np.full(n, 0.6)):
        res = minimize(
            negative, start, method="SLSQP", bounds=bounds, constraints=cons,
            options={"maxiter": 500, "ftol": 1e-12},
        )
        if res.fun < best_val:
            best, best_val = res.x, res.fun
    return pd.Series(best, index=ASSETS, name=f"forward_kelly_{mode}")


def with_equity_premium(cma: ForwardCMA, shift: float) -> ForwardCMA:
    """Shift both equity expected returns by ``shift``, leaving bonds alone.

    The single most important sensitivity: every equity conclusion here is
    downstream of a real-growth assumption nobody can observe.
    """
    geometric = cma.geometric.copy()
    for asset in ("us_equity", "intl_equity"):
        geometric[asset] += shift
    return replace(
        cma,
        geometric=geometric,
        arithmetic=(geometric + 0.5 * cma.vol**2).rename("arithmetic"),
    )


# Global equity market capitalisation split. The starting point any tilt away
# from has to be argued for, rather than an optimiser's corner solution.
GLOBAL_EQUITY_WEIGHTS = (0.63, 0.37)


def bundle_kelly(
    cma: ForwardCMA,
    *,
    equity_split: tuple[float, float] = GLOBAL_EQUITY_WEIGHTS,
    bond_split: tuple[float, float] = (0.65, 0.35),
    financing_spread: float = 0.0,
    max_gross: float | None = None,
    long_only: bool = True,
) -> pd.Series:
    """Kelly over two *bundles* -- global equity and global bonds -- not four sleeves.

    A four-asset optimiser handed two 0.81-correlated equity indices will put
    everything in whichever has the higher expected return, because at that
    correlation an 85bp difference in expected return is an enormous relative
    signal. That corner is an artefact of pretending the estimate is precise.

    Fixing the split *inside* each bundle at market-capitalisation weights removes
    the false precision and leaves the two decisions that the data can actually
    support: how much equity, and how much bonds.
    """
    from scipy.optimize import minimize

    basis = np.array(
        [
            [equity_split[0], equity_split[1], 0.0, 0.0],
            [0.0, 0.0, bond_split[0], bond_split[1]],
        ]
    )
    mu = basis @ cma.excess.to_numpy()
    sigma = basis @ cma.cov.to_numpy() @ basis.T

    def negative(x: np.ndarray) -> float:
        borrowed = max(np.abs(basis.T @ x).sum() - 1.0, 0.0)
        return -(x @ mu - 0.5 * x @ sigma @ x - borrowed * financing_spread)

    bounds = [(0.0, 10.0) if long_only else (-10.0, 10.0)] * 2
    cons = []
    if max_gross is not None:
        cons.append(
            {"type": "ineq", "fun": lambda x: max_gross - np.abs(basis.T @ x).sum()}
        )
    res = minimize(
        negative, np.array([1.0, 0.0]), method="SLSQP", bounds=bounds,
        constraints=cons, options={"maxiter": 500, "ftol": 1e-12},
    )
    return pd.Series(basis.T @ res.x, index=ASSETS, name="bundle_kelly")


def bundle_moments(
    cma: ForwardCMA, *, equity_split: tuple[float, float] = GLOBAL_EQUITY_WEIGHTS
) -> dict:
    """Excess return, volatility and the unlevered Kelly multiple of the equity bundle."""
    w = np.array([equity_split[0], equity_split[1], 0.0, 0.0])
    excess = float(w @ cma.excess.to_numpy())
    vol = float(np.sqrt(w @ cma.cov.to_numpy() @ w))
    return {
        "excess": excess,
        "vol": vol,
        "sharpe": excess / vol,
        "kelly": excess / vol**2,
        "geometric": float(w @ cma.geometric.to_numpy()),
    }


def tilt_sensitivity(
    cma: ForwardCMA,
    *,
    gaps: np.ndarray | None = None,
    financing_spread: float = 0.012,
) -> pd.DataFrame:
    """How the US/international split responds to the assumed return gap.

    ``gaps`` are added to US equity's expected return and subtracted from
    international's, so 0 is the base case and +2% means US out-earns
    international by 2pp more than the yields imply.
    """
    if gaps is None:
        gaps = np.round(np.arange(-0.02, 0.0201, 0.005), 4)
    rows = []
    for gap in gaps:
        geometric = cma.geometric.copy()
        geometric["us_equity"] += gap / 2.0
        geometric["intl_equity"] -= gap / 2.0
        shifted = replace(
            cma,
            geometric=geometric,
            arithmetic=(geometric + 0.5 * cma.vol**2).rename("arithmetic"),
        )
        w = constrained_kelly(
            shifted, mode="long_only", financing_spread=financing_spread
        )
        equity = w["us_equity"] + w["intl_equity"]
        rows.append(
            {
                "us_minus_intl": float(gap),
                "us_equity": float(w["us_equity"]),
                "intl_equity": float(w["intl_equity"]),
                "bonds": float(w["us_bonds"] + w["intl_bonds"]),
                "gross": float(w.abs().sum()),
                "us_share_of_equity": float(w["us_equity"] / equity) if equity > 1e-9 else np.nan,
            }
        )
    return pd.DataFrame(rows).set_index("us_minus_intl")


def bootstrap_paths(
    panel,
    cma: ForwardCMA,
    *,
    years: int = 10,
    paths: int = 400,
    block: int = 21,
    window_years: float = 15.0,
    seed: int = 0,
) -> list:
    """Forward daily return paths: historical *shocks*, forward *drift*.

    Historical returns are standardised, resampled in 21-day blocks -- keeping
    volatility clustering, fat tails and the cross-asset correlation intact -- and
    then rescaled to the forward volatility and re-centred on the forward
    expected return. Nothing about the past level of returns survives; only its
    shape does.
    """
    from .panel import KellyPanel

    if panel.daily is None:
        raise ValueError("panel has no daily data")
    hist = panel.daily.loc[panel.daily.index[-1] - pd.DateOffset(years=int(window_years)) :]
    z = ((hist - hist.mean()) / hist.std(ddof=1)).to_numpy()

    steps = int(years * 252)
    daily_vol = (cma.vol / np.sqrt(252)).to_numpy()
    daily_mu = (cma.arithmetic / 252).to_numpy()
    daily_cash = (1.0 + cma.cash) ** (1.0 / 365.0) - 1.0

    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(steps / block))
    index = pd.bdate_range("2026-09-01", periods=steps)
    out = []
    for _ in range(paths):
        starts = rng.integers(0, len(z) - block, size=n_blocks)
        draw = np.concatenate([z[s : s + block] for s in starts])[:steps]
        returns = daily_mu + draw * daily_vol
        frame = pd.DataFrame(returns, columns=ASSETS, index=index)
        cash = pd.Series(daily_cash * 365 / 252, index=index, name="cash")
        out.append(
            KellyPanel(
                name="forward",
                returns=frame.resample("ME").apply(lambda c: (1 + c).prod() - 1),
                cash=cash.resample("ME").apply(lambda c: (1 + c).prod() - 1),
                tickers=panel.tickers,
                daily=frame,
                daily_cash=cash,
            )
        )
    return out


def forward_ladder(
    panel,
    cma: ForwardCMA,
    weights: pd.Series | np.ndarray,
    *,
    levels: tuple[float, ...] = (1.0, 1.25, 1.5, 1.75, 2.0),
    years: int = 10,
    paths: int = 400,
    financing_spread: float = 0.012,
    seed: int = 0,
    **account_kwargs,
) -> pd.DataFrame:
    """Distribution of forward outcomes by leverage, through a Reg-T account.

    The backtest ladder answers "what would have happened". This answers "what
    is the chance of being sold out, and what does the spread of outcomes look
    like, if the forward assumptions are right".
    """
    from .account import simulate

    sims = bootstrap_paths(panel, cma, years=years, paths=paths, seed=seed)
    rows = []
    for level in levels:
        cagrs, drawdowns, calls, ruins = [], [], [], []
        for sim in sims:
            result = simulate(
                sim,
                weights,
                target_leverage=level,
                financing_spread=financing_spread,
                **account_kwargs,
            )
            summary = result.summary
            cagrs.append(summary["cagr"])
            drawdowns.append(summary["max_drawdown"])
            calls.append(summary["margin_calls"] > 0)
            ruins.append(summary["ruined"])
        cagrs = np.array(cagrs, dtype=float)
        drawdowns = np.array(drawdowns, dtype=float)
        rows.append(
            {
                "leverage": level,
                "median_cagr": float(np.nanmedian(cagrs)),
                "p5_cagr": float(np.nanpercentile(cagrs, 5)),
                "p95_cagr": float(np.nanpercentile(cagrs, 95)),
                "median_max_drawdown": float(np.median(drawdowns)),
                "p5_max_drawdown": float(np.percentile(drawdowns, 5)),
                "p_margin_call": float(np.mean(calls)),
                "p_ruin": float(np.mean(ruins)),
                "p_underperform_cash": float(np.mean(cagrs < cma.cash)),
            }
        )
    return pd.DataFrame(rows).set_index("leverage")


def historical_real_eps_growth(
    *,
    windows: tuple[tuple[str, str], ...] | None = None,
    smooth_years: int = 10,
) -> pd.DataFrame:
    """Realised real earnings-per-share growth of the S&P 500, by period.

    Earnings are violently cyclical, so a growth rate measured between two single
    months is mostly a statement about where those two months sat in the cycle:
    1960-2000 reads 2.25%/yr on raw endpoints and 1.27%/yr on ten-year averaged
    ones. Both are reported, and the smoothed column is the one to believe.
    """
    from ..data import shiller

    series = shiller.real_earnings()
    end = str(series.index[-1].year)
    windows = windows or (
        ("1871", end),
        ("1900", end),
        ("1950", end),
        ("1960", "2000"),
        ("1985", end),
        ("2005", end),
    )

    rows = []
    for start, stop in windows:
        window = series.loc[start:stop].dropna()
        span = (window.index[-1] - window.index[0]).days / 365.25
        raw = (window.iloc[-1] / window.iloc[0]) ** (1.0 / span) - 1.0
        months = smooth_years * 12
        smoothed = (
            window.iloc[-months:].mean() / window.iloc[:months].mean()
        ) ** (1.0 / (span - smooth_years)) - 1.0
        rows.append(
            {
                "period": f"{start}-{stop}",
                "raw_endpoints": float(raw),
                "smoothed_endpoints": float(smoothed),
                "years": round(span, 1),
            }
        )
    return pd.DataFrame(rows).set_index("period")


def rolling_real_eps_growth(
    *, horizon_years: int = 30, smooth_years: int = 10
) -> pd.Series:
    """Every ``horizon_years`` window of real EPS growth in the Shiller record.

    The distribution, not the average, is the honest input to a forward
    assumption: an investor gets one draw from it.
    """
    from ..data import shiller

    smoothed = shiller.real_earnings().rolling(smooth_years * 12).mean().dropna()
    ratio = smoothed / smoothed.shift(horizon_years * 12)
    return (ratio ** (1.0 / horizon_years) - 1.0).dropna().rename(
        f"real_eps_growth_{horizon_years}y"
    )


def growth_needed_for(
    cma: ForwardCMA,
    target_return: float,
    *,
    asset: str = "us_equity",
) -> float:
    """Per-share real growth required for ``asset`` to return ``target_return``.

    Inverts the build-up: income and inflation are observed, so any expected
    return maps to exactly one growth rate, and that rate can be checked against
    the historical distribution instead of argued about in the abstract.
    """
    row = cma.build_up.loc[asset]
    fixed = row["income"] + row["inflation"] + row["valuation"] + row["currency"]
    return float(target_return - fixed)


# Named assumption sets. Each is a coherent position, not a dial to split the
# difference on -- the growth term and the buyback term have to be argued for
# together or the share count gets counted twice.
BASE = Assumptions()

MODERN_REGIME = Assumptions(
    # 1985-2023 delivered 3.86%/yr of real per-share growth. Read through the
    # decomposition, the repeatable part of that is US real GDP growth plus the
    # end of dilution; the +1.75%/yr from a doubling profit share is a level
    # shift. This set grants the repeatable part in full and lifts aggregate
    # growth above US GDP because roughly 40% of index revenue is earned abroad.
    us_real_growth=0.0256,
    us_buyback=0.0130,
    intl_real_growth=0.0200,
    intl_buyback=0.0060,
)

HISTORICAL_MEDIAN = Assumptions(
    # The median 30-year window since 1871, held per share: 1.68%.
    us_real_growth=0.0038,
    us_buyback=0.0130,
    intl_real_growth=0.0038,
    intl_buyback=0.0060,
)

PRESETS = {"base": BASE, "modern": MODERN_REGIME, "historical": HISTORICAL_MEDIAN}


def decompose_growth(
    *, eras: tuple[tuple[str, str], ...] | None = None, smooth_quarters: int = 8
) -> pd.DataFrame:
    """Split realised earnings growth into the parts that can and cannot repeat.

    Three measured series, one identity::

        real per-share EPS growth
            = real GDP growth              (the economy)
            + profit share drift           (margins -- a level shift, bounded)
            - dilution                     (net share issuance -- a policy choice)

    The aggregate side is deflated by the GDP deflator rather than CPI, which
    makes the first two terms exactly additive: profits deflated that way are
    identically ``profit share x real GDP``. Shiller's earnings are CPI-deflated,
    so the dilution residual carries the CPI-minus-deflator wedge (~0.2-0.3pp/yr)
    along with genuine share-count change; it is a good estimate of dilution, not
    an exact one.

    The reason to run this rather than quote an era's growth rate: 1947-1985 and
    1985-2023 have almost the same aggregate profit growth, and wildly different
    per-share growth. What changed was not corporate performance, it was share
    count and margins.
    """
    from ..data import fred, shiller

    eras = eras or (("1947", "1985"), ("1985", "2023"), ("1995", "2023"), ("2010", "2023"))

    profits = fred.series("CPATAX")
    nominal_gdp = fred.series("GDP")
    real_gdp = fred.series("GDPC1")
    eps = shiller.real_earnings()

    quarterly = pd.concat(
        [profits.rename("profits"), nominal_gdp.rename("gdp"), real_gdp.rename("real_gdp")],
        axis=1,
    ).dropna()
    quarterly["share"] = quarterly["profits"] / quarterly["gdp"]
    quarterly["real_profits"] = quarterly["share"] * quarterly["real_gdp"]

    def rate(series: pd.Series, start: str, stop: str, smooth: int) -> float:
        window = series.loc[start:stop].dropna()
        years = (window.index[-1] - window.index[0]).days / 365.25
        first = window.iloc[:smooth].mean()
        last = window.iloc[-smooth:].mean()
        return float((last / first) ** (1.0 / (years - smooth / 4.0)) - 1.0)

    rows = []
    for start, stop in eras:
        window = eps.loc[start:stop].dropna()
        years = (window.index[-1] - window.index[0]).days / 365.25
        per_share = float(
            (window.iloc[-120:].mean() / window.iloc[:120].mean()) ** (1.0 / (years - 10.0))
            - 1.0
        )
        aggregate = rate(quarterly["real_profits"], start, stop, smooth_quarters)
        rows.append(
            {
                "era": f"{start}-{stop}",
                "real_gdp": rate(quarterly["real_gdp"], start, stop, smooth_quarters),
                "profit_share_drift": rate(quarterly["share"], start, stop, smooth_quarters),
                "aggregate_real_profits": aggregate,
                "real_eps_per_share": per_share,
                "dilution": aggregate - per_share,
            }
        )
    return pd.DataFrame(rows).set_index("era")


def profit_share(*, as_of: str | None = None) -> pd.Series:
    """After-tax corporate profits as a share of GDP -- the margin level itself."""
    from ..data import fred

    share = (fred.series("CPATAX") / fred.series("GDP")).dropna().rename("profit_share")
    return share.loc[:as_of] if as_of else share


def geometric_frontier(
    cma: ForwardCMA,
    *,
    fractions: np.ndarray | None = None,
    financing_spread: float = 0.012,
    long_only: bool = True,
    max_gross: float | None = None,
    equity_split: tuple[float, float] | None = None,
) -> pd.DataFrame:
    """The geometric-return frontier, solved by risk aversion rather than by scaling.

    There are two ways to build a fractional-Kelly portfolio, and they are only
    the same thing in a world that does not exist:

    *Scaling* takes the full-Kelly weights and multiplies them by ``f``, parking
    the remainder in cash. That is correct **only** under two-fund separation --
    one risk-free rate for both lending and borrowing, and no position limits.

    *Slope* re-solves at each ``f`` for the portfolio that maximises

        ``w'mu - (1/2f) w'Sigma w - spread x borrowed``

    which is the Kelly objective at ``f = 1`` and a more risk-averse investor
    below it. This is the construction used here.

    The difference is not academic, because two things break separation:

    * **Borrowing costs more than lending.** The financing spread puts a kink at
      1x gross, so the tangency portfolio above 1x is not the one below it.
    * **Cash is not the only low-risk asset.** Below 1x gross nothing is
      borrowed, so bonds stop competing against a 5.00% margin rate and start
      competing against a 3.80% bill. Scaling can never discover that, because it
      only ever dilutes the levered portfolio -- which holds no bonds -- with
      cash. Slope finds it, and the composition changes accordingly.
    """
    from scipy.optimize import minimize

    if fractions is None:
        fractions = np.round(np.arange(0.1, 1.51, 0.05), 3)

    mu = cma.excess.to_numpy()
    sigma = cma.cov.to_numpy()

    # With ``equity_split`` the two equity sleeves move together at market
    # weights, so the frontier answers "how much equity, and how much bonds"
    # rather than re-litigating the knife-edge US/international tilt at every
    # point on the curve.
    if equity_split is None:
        basis = np.eye(len(mu))
    else:
        basis = np.array(
            [[equity_split[0], equity_split[1], 0.0, 0.0],
             [0.0, 0.0, 1.0, 0.0],
             [0.0, 0.0, 0.0, 1.0]]
        )
    k = basis.shape[0]
    bounds = [(0.0, 6.0) if long_only else (-6.0, 6.0)] * k
    cons = []
    if max_gross is not None:
        cons.append({"type": "ineq", "fun": lambda x: max_gross - np.abs(basis.T @ x).sum()})

    rows = []
    for f in fractions:
        aversion = 1.0 / (2.0 * float(f))

        def negative(x: np.ndarray, aversion: float = aversion) -> float:
            w = basis.T @ x
            borrowed = max(np.abs(w).sum() - 1.0, 0.0)
            return -(w @ mu - aversion * w @ sigma @ w - borrowed * financing_spread)

        best, best_val = None, np.inf
        for start in (np.zeros(k), np.full(k, 0.2), np.full(k, 0.5)):
            res = minimize(
                negative, start, method="SLSQP", bounds=bounds,
                constraints=cons, options={"maxiter": 600, "ftol": 1e-13},
            )
            if res.fun < best_val:
                best, best_val = res.x, res.fun
        w = basis.T @ np.asarray(best)

        gross = float(np.abs(w).sum())
        variance = float(w @ sigma @ w)
        borrowed = max(gross - 1.0, 0.0)
        arithmetic = cma.cash + float(w @ mu) - borrowed * financing_spread
        row = {
            "fraction": float(f),
            "gross": gross,
            "vol": float(np.sqrt(variance)),
            "arithmetic": arithmetic,
            "geometric": arithmetic - 0.5 * variance,
        }
        row.update({asset: float(w[i]) for i, asset in enumerate(ASSETS)})
        row["cash"] = 1.0 - gross
        rows.append(row)

    return pd.DataFrame(rows).set_index("fraction")


def scaled_frontier(
    cma: ForwardCMA,
    *,
    fractions: np.ndarray | None = None,
    financing_spread: float = 0.012,
    long_only: bool = True,
    equity_split: tuple[float, float] | None = None,
) -> pd.DataFrame:
    """The same curve built the naive way, by scaling full-Kelly weights.

    Reported only to show what the slope construction buys: below 1x gross this
    one holds cash where the frontier would hold bonds, and gives up growth for
    nothing in return.
    """
    if fractions is None:
        fractions = np.round(np.arange(0.1, 1.51, 0.05), 3)

    if equity_split is None:
        full = constrained_kelly(
            cma,
            mode="long_only" if long_only else "unconstrained",
            financing_spread=financing_spread,
        ).to_numpy()
    else:
        full = bundle_kelly(
            cma, equity_split=equity_split, financing_spread=financing_spread,
            long_only=long_only,
        ).to_numpy()
    sigma = cma.cov.to_numpy()
    mu = cma.excess.to_numpy()

    rows = []
    for f in fractions:
        w = full * float(f)
        gross = float(np.abs(w).sum())
        variance = float(w @ sigma @ w)
        arithmetic = cma.cash + float(w @ mu) - max(gross - 1.0, 0.0) * financing_spread
        row = {
            "fraction": float(f),
            "gross": gross,
            "vol": float(np.sqrt(variance)),
            "arithmetic": arithmetic,
            "geometric": arithmetic - 0.5 * variance,
        }
        row.update({asset: float(w[i]) for i, asset in enumerate(ASSETS)})
        row["cash"] = 1.0 - gross
        rows.append(row)
    return pd.DataFrame(rows).set_index("fraction")
