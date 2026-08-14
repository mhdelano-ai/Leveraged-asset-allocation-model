"""Fetch and cache every raw series the panels need.

Run once; everything afterwards reads from the parquet cache. Requests are paced
(~1/sec) because Yahoo throttles bursts, so a cold run takes a few minutes.
Re-running is cheap -- cached series are skipped.

Yahoo comes first and carries the load, including Treasury yields via the
``^IRX/^FVX/^TNX/^TYX`` indices. FRED is fetched last and treated as optional:
it is a useful cross-check but it throttles hard and can be unreachable for long
stretches, so nothing load-bearing is allowed to depend on it.
"""

from __future__ import annotations

import sys

from lam.data import cache, fred, lbma, shiller, yahoo

# Indices: close only (adjclose is a no-op for indices and carries no dividends).
YAHOO_INDICES = [
    "^GSPC",     # S&P 500 price, 1927+ -- total return rebuilt from Shiller dividends
    "^SP500TR",  # actual S&P 500 total return, 1988+ -- splice target and validator
    "^SPGSCI",   # commodities, 1984+
    "^VIX",      # volatility regime, 1990+
    "^IRX", "^FVX", "^TNX", "^TYX",  # Treasury constant-maturity yields, 1960/1962/1977+
]

# ETFs and mutual funds: adjclose is genuine total return net of fees.
YAHOO_FUNDS = [
    # Modern tradable sleeves
    "SPY", "VTI", "EFA", "EEM", "TLT", "IEF", "LQD", "HYG", "GLD", "DBC", "VNQ",
    # Leveraged ETFs -- calibrate and validate the synthetic LETF model
    "UPRO", "SSO", "TMF", "UGL",
    # Long-history mutual funds extending the diversified sleeves back to the 1980s
    "VTRIX",  # intl value, 1983
    "VUSTX",  # long Treasury, 1986
    "VFITX",  # intermediate Treasury, 1991
    "VWESX",  # IG corporate, 1980
    "VWEHX",  # high yield, 1980
    "FRESX",  # REIT, 1986
    "VFINX",  # S&P 500 index fund -- cross-check on the TR reconstruction
]

# Optional: cross-check only. FRED throttles hard; failures here are not fatal.
FRED_SERIES = ["DGS10", "DGS30", "DTB3", "DFF", "CPIAUCSL"]


def main() -> int:
    required_failures: list[tuple[str, str]] = []
    optional_failures: list[tuple[str, str]] = []

    def attempt(label: str, fn, *, required: bool) -> None:
        try:
            result = fn()
            span = f"{result.index.min().date()} -> {result.index.max().date()}"
            print(f"  ok   {label:12s} n={len(result):6d}  {span}", flush=True)
        except Exception as exc:  # noqa: BLE001 - collect and continue
            msg = f"{type(exc).__name__}: {exc}"
            print(f"  {'FAIL' if required else 'skip'} {label:12s} {msg}", flush=True)
            (required_failures if required else optional_failures).append((label, msg))

    print("Yahoo indices (incl. Treasury yields)", flush=True)
    for sym in YAHOO_INDICES:
        attempt(sym, lambda s=sym: yahoo.prices(s, field="close"), required=True)

    print("\nYahoo funds / ETFs", flush=True)
    for sym in YAHOO_FUNDS:
        attempt(sym, lambda s=sym: yahoo.total_return_prices(s), required=True)

    print("\nLBMA gold", flush=True)
    attempt("gold_pm", lbma.gold_usd, required=True)

    print("\nShiller dividends", flush=True)
    attempt("shiller", shiller.dividend_yield, required=True)

    print("\nFRED (optional cross-check)", flush=True)
    for sid in FRED_SERIES:
        attempt(sid, lambda s=sid: fred.series(s), required=False)

    print(f"\ncache: {cache.CACHE_DIR}", flush=True)
    if optional_failures:
        print(f"{len(optional_failures)} optional series unavailable (fine -- Yahoo covers these)")
    if required_failures:
        print(f"\n{len(required_failures)} REQUIRED failure(s):")
        for label, msg in required_failures:
            print(f"  {label}: {msg}")
        return 1
    print("all required series cached")
    return 0


if __name__ == "__main__":
    sys.exit(main())
