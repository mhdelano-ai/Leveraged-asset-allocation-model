"""Validate every synthetic data construction against a real instrument.

These are the load-bearing assumptions of the whole project: if the synthetic
Treasury returns or the reconstructed S&P total return are wrong, every number
downstream is wrong in a way no amount of careful backtesting would reveal. The
build stops here if any gate fails.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from lam.data import lbma, rates, synth_bonds, synth_equity, yahoo
from lam.metrics.core import ann_vol, cagr
from lam.metrics.drawdown import max_drawdown

PASS, FAIL = "PASS", "FAIL"


def _fmt(ok: bool) -> str:
    return PASS if ok else FAIL


def check_spx_total_return() -> bool:
    print("\n=== S&P 500 total-return reconstruction vs actual ^SP500TR ===")
    stats = synth_equity.validate_reconstruction()
    print(f"  overlap        {stats['start']} -> {stats['end']} ({stats['n_days']} days)")
    print(f"  CAGR           recon {stats['cagr_recon']:.4%}  actual {stats['cagr_actual']:.4%}"
          f"   diff {stats['cagr_diff_bps']:+.1f} bps")
    print(f"  max drawdown   recon {stats['maxdd_recon']:.2%}  actual {stats['maxdd_actual']:.2%}"
          f"   diff {stats['maxdd_diff_pp']:+.3f} pp")
    print(f"  daily corr     {stats['corr']:.5f}   tracking error {stats['te_ann']:.3%}")

    ok = (
        abs(stats["cagr_diff_bps"]) < 15.0
        and abs(stats["maxdd_diff_pp"]) < 0.5
        and stats["corr"] > 0.999
    )
    print(f"  -> {_fmt(ok)}  (gate: |dCAGR| < 15bps, |dMaxDD| < 0.5pp, corr > 0.999)")

    # The price index is not a substitute for total return, and the gap is not
    # academic: it moves the drawdown budget the constraint is measured against.
    price_ret = yahoo.prices("^GSPC", field="close").pct_change().dropna()
    tr = synth_equity.spx_total_return()
    common = price_ret.index.intersection(tr.index)
    print(f"  note: SPX price maxDD {max_drawdown(price_ret.loc[common]):.2%} vs "
          f"total-return {max_drawdown(tr.loc[common]):.2%} "
          f"-- benchmarking on price would loosen the constraint")
    return ok


def _bond_check(label: str, tenor: str, etf: str, grid: np.ndarray) -> tuple[bool, float]:
    yield_pct = rates.cmt_yield_pct(tenor)
    bench = yahoo.total_return_prices(etf).pct_change().dropna()
    best, diag = synth_bonds.calibrate_maturity(yield_pct, bench, grid=grid)
    row = diag.loc[best]
    print(f"\n=== {label}: par-bond roll on {tenor} vs {etf} ===")
    print(f"  calibrated effective maturity  {best:.1f}y  (nominal tenor {tenor})")
    print(f"  daily corr     {row['corr']:.4f}")
    print(f"  CAGR           synth {row['cagr_synth']:.3%}  {etf} {row['cagr_bench']:.3%}"
          f"   diff {(row['cagr_synth'] - row['cagr_bench']) * 1e4:+.0f} bps")
    print(f"  vol            synth {row['vol_synth']:.2%}  {etf} {row['vol_bench']:.2%}")
    print(f"  tracking error {row['te_ann']:.2%}")
    ok = bool(row["corr"] > 0.93 and abs(row["cagr_synth"] - row["cagr_bench"]) < 0.0125)
    print(f"  -> {_fmt(ok)}  (gate: corr > 0.93, |dCAGR| < 125bps)")
    return ok, best


def check_gold() -> bool:
    print("\n=== LBMA gold ===")
    gold = lbma.gold_usd()
    floating = lbma.gold_floating()
    ret = floating.pct_change().dropna()
    print(f"  full series    {gold.index.min().date()} -> {gold.index.max().date()} "
          f"({len(gold)} obs)")
    print(f"  floating era   {floating.index.min().date()} -> {floating.index.max().date()}")
    print(f"  vol {ann_vol(ret):.1%}   maxDD {max_drawdown(ret):.1%}   CAGR {cagr(ret):.2%}")

    # Spot-checks against well-known fixings.
    checks = {"1980-01-21": 850.0, "2011-09-05": 1895.0}
    ok = True
    for date, expected in checks.items():
        if pd.Timestamp(date) in gold.index:
            actual = float(gold.loc[pd.Timestamp(date)])
            hit = abs(actual - expected) / expected < 0.02
            ok &= hit
            print(f"  {date}  fixing {actual:.2f}  (expected ~{expected:.0f})  {_fmt(hit)}")

    # Pre-float gold was administratively pegged; including it would hand the
    # risk sizer a fake riskless real asset.
    pre = gold.loc[:"1971-08-15"].pct_change().dropna()
    print(f"  pre-1971 vol {ann_vol(pre):.1%} vs floating {ann_vol(ret):.1%} "
          f"-- confirms excluding the pegged era")
    print(f"  -> {_fmt(ok)}")
    return ok


def check_rates() -> bool:
    print("\n=== Financing / cash rate ===")
    quoted = rates.cmt_yield_pct("3m")
    bey = rates.bill_rate_bey()
    print(f"  ^IRX (discount) {quoted.index.min().date()} -> {quoted.index.max().date()}"
          f"   latest {quoted.iloc[-1]:.3f}%")
    print(f"  BEY conversion  latest {bey.iloc[-1]:.4%}   1981 peak {bey.loc['1981'].max():.2%}")
    ok = bool((bey > 0).all() and bey.max() < 0.25 and bey.loc["1981"].max() > 0.10)
    print(f"  -> {_fmt(ok)}  (gate: positive, < 25%, 1981 peak > 10%)")

    for tenor in ("5y", "10y", "30y"):
        s = rates.cmt_yield_pct(tenor)
        print(f"  {tenor:4s} {s.index.min().date()} -> {s.index.max().date()}  n={len(s)}")
    return ok


def main() -> int:
    results: dict[str, bool] = {}
    results["spx_total_return"] = check_spx_total_return()
    ok_long, m_long = _bond_check("Long Treasury", "30y", "TLT", np.arange(8.0, 30.5, 0.5))
    ok_int, m_int = _bond_check("Intermediate Treasury", "10y", "IEF", np.arange(2.0, 15.5, 0.5))
    results["long_treasury"] = ok_long
    results["interm_treasury"] = ok_int
    results["gold"] = check_gold()
    results["rates"] = check_rates()

    print("\n" + "=" * 62)
    for name, ok in results.items():
        print(f"  {_fmt(ok):4s}  {name}")
    print(f"\n  calibrated maturities: long={m_long:.1f}y  intermediate={m_int:.1f}y")
    failed = [k for k, v in results.items() if not v]
    if failed:
        print(f"\n{len(failed)} GATE FAILURE(S): {', '.join(failed)}")
        return 1
    print("\nall data gates passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
