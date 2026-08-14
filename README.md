# Leveraged multi-asset allocation under an S&P 500 drawdown constraint

Maximise CAGR subject to a hard constraint: **drawdowns must not exceed the
S&P 500's**, tested two ways —

1. **Full sample** — worst peak-to-trough no deeper than the benchmark's.
2. **Rolling 3-year** — in *every* rolling 3-year window, no deeper than the
   benchmark's drawdown in that same window.

Two leverage vehicles are modelled and compared head to head: daily-reset
leveraged ETFs, and portfolio margin / box spreads.

---

## The headline finding

**The constraint does not bind in crises. It binds in calm markets.**

That is the single most useful result here, and it is not obvious in advance.
The instinct is that a drawdown constraint is about surviving 2008. It isn't.
Over 1971–2026 the S&P's rolling 3-year maximum drawdown distribution is:

| percentile | 3y max drawdown |
|---|---|
| minimum | **5.58%** |
| 5th | 7.89% |
| 10th | 8.47% |
| median | 19.18% |
| maximum | 55.25% |

The 55% crash is an enormous budget and easy to stay inside — trend and a
drawdown throttle handle it. The binding requirement is the **5.58%** at the
other end: during the S&P's smoothest three-year stretch (roughly 1991–94), a
compliant strategy may not draw down more than 5.58% either. A diversified
levered book routinely draws 6–10% in exactly such periods — the 1994 bond
selloff, the 2004–06 rate scare — with no crisis in sight.

So the rolling-3y reading is not "don't crash as hard as the S&P." It is **"be
smoother than the S&P during the S&P's smoothest years,"** which is a far
stronger demand and caps unconditional risk — and therefore return — well below
the benchmark's.

Both readings are reported, because they give very different answers.

---

## Results

Panel L, 1971-09 → 2026-08. Benchmark: S&P 500 total return, **CAGR 11.17%,
max drawdown −55.25%**. From a 512-point Sobol scan over eight parameters.

### The two answers

| | Strict (full-sample **and** rolling 3y) | Full-sample only |
|---|---|---|
| CAGR | **8.05%** (9.98% at the fragile in-sample optimum) | **15.07%** |
| Max drawdown | −13.19% | −40.85% |
| Sharpe | 0.66 | 0.64 |
| Calmar | 0.61 | 0.37 |
| Average leverage | 0.57× | 2.04× |
| vs S&P 500 | **loses** 3.1pp/yr of return; 42pp shallower drawdown | **beats** by 3.9pp/yr; 14pp shallower drawdown |

**Under the strict reading you cannot beat the S&P 500.** The best robust
configuration returns 8.05% against the benchmark's 11.17%. What you get instead
is a far better ride: Calmar 0.61 vs the S&P's 0.20, and a worst loss of 13%
against 55%.

**Under the full-sample-only reading you can, comfortably** — 15.07%/yr with a
−40.85% worst drawdown, both better than the benchmark. The cost is that this
book runs ~2.8× leverage and is *deeper underwater than the S&P in many
individual three-year windows* (worst excess +14.8pp), which is exactly what the
rolling test is designed to forbid.

### The frontier between them

How much return the rolling test costs, as it is relaxed. The full-sample
condition stays strict throughout.

| Rolling-3y tolerance | Feasible | Best CAGR | Max drawdown | Avg leverage |
|---|---|---|---|---|
| 0pp (strict) | 112 / 512 | 9.98% | −16.96% | 0.83× |
| 1pp | 157 | 10.35% | −23.59% | 0.92× |
| 3pp | 259 | 10.95% | −21.48% | 1.07× |
| **5pp** | 347 | **11.43%** | −22.02% | 1.22× |
| 10pp | 479 | 13.80% | −23.46% | 1.78× |
| none (full-sample only) | 512 | 15.07% | −40.85% | 2.04× |

**Roughly 5pp of tolerance is the break-even point** — that is the smallest
relaxation of the rolling test at which the strategy matches the S&P's 11.17%
return, and it does so at −22% drawdown instead of −55%.

The mirror-image question — how much return it costs to demand *safety margin*
rather than bare feasibility — is answered in Robustness below, and is the more
important one, because bare feasibility turns out not to survive contact with
new data.

### The allocation

There is no fixed allocation — the weights are an output of the rules, and they
move a lot. `python scripts/show_allocation.py` answers "what do I hold".

Long-run mix, as a share of invested capital (Panel L, strict):

| Sleeve | Proxy | Share of book | Avg capital | Range |
|---|---|---|---|---|
| US equity | VTI | 45.0% | 25.8% | 0–76% |
| Intermediate Treasuries | IEF | 30.6% | 17.5% | 0–150% |
| Long Treasuries | TLT | 13.6% | 7.8% | 0–44% |
| Gold | GLD | 6.5% | 3.7% | 0–21% |
| Commodities | DBC | 4.4% | 2.5% | 0–19% |

**The strategy is not, in practice, a leveraged one.** Under the strict
constraint gross exposure averages **0.57×**, sits above 1× on only **7.5%** of
days, never exceeds 1.94×, and holds ~43% cash on average. The constraint as
specified very nearly rules leverage out. The full-sample-only configuration is
genuinely levered — 2.04× average, above 2× on 64% of days.

The mix is regime-dependent rather than static (average gross weights by decade):

| | US equity | Long UST | Interm UST | Gold | Commodities | Gross |
|---|---|---|---|---|---|---|
| 1970s | 16.2% | 2.3% | 23.7% | 4.2% | 0.0% | 0.46× |
| 1980s | 28.2% | 7.2% | 11.5% | 2.5% | 2.6% | 0.52× |
| 2000s | 26.1% | 14.3% | 24.1% | 6.3% | 4.7% | 0.75× |
| 2020s | 24.3% | 3.8% | 9.9% | 4.3% | 2.4% | 0.45× |

As of the last bar (2026-08-13) it holds 26.9% US equity, **0% in both Treasury
sleeves**, 1.0% gold, 1.5% commodities — 0.29× gross, **70.6% cash**. The
Treasury sleeves are at zero because the trend gate is off on both, which the
report states rather than leaving to be inferred.

### Out of sample

Anchored walk-forward, 20-year initial training window, re-selected every 2
years, 18 folds. **The stitched out-of-sample curve is the headline result** —
never the in-sample optimum.

| | Out-of-sample strategy | S&P 500, same dates |
|---|---|---|
| Period | 1991-09 → 2026-08 | 1991-09 → 2026-08 |
| CAGR | **8.84%** | 11.04% |
| Volatility | 7.2% | — |
| Sharpe | **0.85** | ~0.37 |
| Max drawdown | **−16.96%** | −55.25% |
| Constraint | **PASS** — 0 of 1,609 windows violating | — |

The constraint holds out of sample, and the Sharpe survives a multiple-testing
haircut (observed 0.85 against an expected-maximum-under-the-null of 0.51). No
fold ever failed to find a feasible configuration, and the worst out-of-sample
fold was 2021-23 at +1.17% — through the 2022 inflation shock that broke most
levered stock/bond books.

**One honest limitation.** Each fold re-selects from a candidate pool that was
itself produced by a scan over the *full* sample, so the candidates embed
full-sample information even though the per-fold choice does not. This measures
parameter **stability** rather than true out-of-sample discovery. A clean
version would re-run the Sobol scan inside every training window — about 18×
the compute — and is the first thing to do with more time. Suggestively, the
selection was identical in all 18 folds (σ 10.9%, max leverage 2.41, drawdown
margin 0.44), which is what parameter stability looks like, but it is not proof.

### Robustness — where this gets uncomfortable

The strict solution is **fragile**, and the project reports that rather than
burying it:

- The in-sample optimum clears the constraint by **8 basis points** and has a
  **plateau radius of 0%** — a 5% move in one parameter breaks feasibility.
- **`sigma_target` is the binding axis at every margin level.** Feasibility dies
  the moment the risk target rises, which means there is no clever structure
  buying extra return here; the constraint is a pure cap on how much volatility
  the book may run.
- Requiring genuine slack costs return: 1pp of margin → 9.12%, 2pp → 8.17%,
  3pp → 7.25%, and **no configuration at all survives a 5pp margin**.
- **Cross-panel, the strict solution fails.** Parameters chosen on Panel L
  breach by +0.80pp on Panel X (1986+, nine sleeves) — a different universe and
  era. It holds on Panel M with 4.2pp to spare.
- Cost sensitivity is benign for the strict solution (8.05% → 7.69% at 3×
  assumed costs) but material for the levered one (15.07% → 10.49%), which at
  triple costs barely beats the benchmark.

The strict solution passes **all 17** named stress episodes. The full-sample one
passes 12 of 17 — it is deeper than the S&P in 1980-82, 1987, 1994, 2013 and
2015-16.

### Where the return actually comes from

Crisis behaviour is the strategy's strongest feature, and it is not subtle. On
Panel L with the strict configuration:

| Episode | Strategy | S&P 500 | Strategy DD | S&P DD |
|---|---|---|---|---|
| 1973-74 oil shock | **+16.8%** | −37.6% | −0.9% | −45.0% |
| 2000-02 dotcom | **+20.1%** | −33.0% | −8.3% | −47.4% |
| 2007-09 GFC | **+14.5%** | −45.8% | −9.1% | −55.3% |
| 2022 inflation | −2.5% | −18.1% | −4.4% | −24.5% |

The mirror image is the bull market: 2010+ returns 6.10%/yr against the S&P's
14.52%. That trade — give up most of the upside, avoid essentially all of the
drawdown — *is* the constraint, made concrete.

---

## Data

Everything is rebuilt from free sources, with each synthetic construction
validated against a real traded instrument before use.

| Series | Source | Span | Validation |
|---|---|---|---|
| S&P 500 total return | `^GSPC` price + Shiller dividends, spliced to `^SP500TR` from 1988 | 1927– | **−4.7 bp/yr** CAGR and 0.03pp max-DD vs actual `^SP500TR` |
| Long Treasury | par-bond roll on the 30y CMT yield, M = 23.5y | 1977– | **+1 bp/yr** vs TLT, corr 0.940 |
| Intermediate Treasury | par-bond roll on the 10y CMT yield, M = 8.0y | 1962– | **−48 bp/yr** vs IEF, corr 0.947 |
| Gold | LBMA daily USD fix | 1968– (used from 1971-09) | exact on spot-checks (1980-01-21 = $850) |
| Cash / financing | 13-week bill, discount → bond-equivalent | 1960– | 1981 peak 18.02% |
| Commodities | `^SPGSCI` + T-bill collateral | 1984– | — |

### Traps found and guarded

Each of these fails **silently** and produces a plausible-looking backtest.

- **pandas 3.0 stores datetimes at microsecond resolution.** The common
  `np.diff(index.asi8) / 86_400e9` idiom then yields day counts 1000× too small.
  This zeroed out every time-scaled accrual at once — dividends, bond coupons
  and the cost of leverage — while leaving price returns untouched. It cost the
  S&P reconstruction 230 bp/yr and made synthetic Treasuries return −0.22%
  instead of 3.57%. All such conversions now go through `lam/timeaxis.py`.
- **Yahoo returns quarterly bars for `range=max&interval=1d`** — ~168 points
  instead of ~8,400, with nothing in the response indicating it. Guarded by a
  bar-spacing assertion; the fetcher only ever sends explicit epochs.
- **Yahoo returns valid-but-truncated payloads.** `^TNX` and `^FVX` both cached
  with **15 rows instead of 16,141** and parsed perfectly. Guarded by per-symbol
  minimum row counts.
- **FRED licence-truncates some series** while returning well-formed CSV:
  `SP500` carries 10 years, the ICE BofA credit series ~3. A status-code check
  passes and the data is useless.
- **`^GSPC`'s `adjclose` equals its `close`** — for indices it carries no
  dividends at all. Using it as total return silently discards 2–4%/yr.
- **Shiller's price column is a monthly *average*.** October 1987 reads ~280
  against an intramonth low of 224.84, so using it for the price path erases
  Black Monday. It is used for dividend yield only.
- **FRED throttles hard** and was unreachable for this entire build, so Treasury
  yields come from Yahoo's `^IRX`/`^FVX`/`^TNX`/`^TYX` indices — the same series
  (`^TNX` opens at 4.06 on 1962-01-02, exactly `DGS10`'s first observation).

### Panels

| Panel | Span | Sleeves | Role |
|---|---|---|---|
| **L** | 1971-09 → | 5 | **Binds the constraint.** Spans 1973-74, 1980-82, 1987, 2000-02, 2008, 2020, 2022 |
| **X** | 1986-11 → | 9 | Extended universe via long-history mutual funds (survivorship-biased; treat as ~100 bp/yr optimistic) |
| **M** | 2006-02 → | 10 | Tradable ETFs. Cost realism and the vehicle comparison **only** — one crisis cannot validate a drawdown constraint |

---

## Method

**Engine.** One daily accounting identity, financing accrued on calendar days at
ACT/360:

```
r_p(t) = Σ wᵢ(t-1)·rᵢ(t) + (1 - L(t-1))·rf(t)·Δ/360 - B(t-1)·spread·Δ/360 - turnover·cost
```

No-lookahead is enforced structurally rather than by convention, and asserted by
a test: corrupting every return after a cutoff leaves all earlier weights and
equity **bit-identical**. Execution lags signals by one day, so a signal formed
on Friday 1987-10-16 takes the full −20.5% of Black Monday at whatever leverage
was on.

**Allocation**, four layers:

1. Inverse-volatility base weights with **cluster risk budgets** — naive
   inverse-vol puts five correlated equity-beta sleeves in the book and calls it
   diversified.
2. **Trend** (time-series momentum, excess of cash). Necessary because
   volatility targeting is *lagging by construction*: realised vol only rises
   after prices fall, so it stays invested through slow bears like 2000-02 and
   2022. Trend and vol targeting fail in different places, which is why both
   exist.
3. **Volatility targeting**, with the risk forecast **floored by a
   crisis-correlation covariance** estimated on the benchmark's worst 10% of
   days — so the sizer never buys a diversification benefit that evaporates in
   the drawdown it is meant to survive.
4. **Drawdown throttle** (CPPI-style, with a dead zone and hysteresis) — the
   only layer that references the constraint itself and the only one that
   guarantees exposure reaches zero before the budget is spent.

Leverage falls freely but rises slowly (an asymmetric ratchet), which stops the
book re-levering into bear-market rallies.

**Search.** Sobol scan over eight parameters, feasibility gated on the hard
maximum but optimised against the CVaR of window excesses — with thousands of
overlapping windows the raw maximum is an extreme order statistic set by a
single path, and optimising it directly just fits that window.

---

## Leverage vehicles

Calibrated by fitting `r = k·r_u − (k−1)(rf + s)·Δ/360 − TER·Δ/365` to each real
fund independently:

| Fund | k | implied spread | simulated vs actual |
|---|---|---|---|
| UPRO | 3× | 68 bp | −56 bp/yr, corr 0.998 |
| SSO | 2× | 98 bp | −15 bp/yr, corr 0.995 |
| TMF | 3× | 53 bp | −6 bp/yr, corr 0.997 |
| UGL | 2× | 164 bp | +78 bp/yr, corr 0.997 |

UPRO and SSO are separate regressions on different multipliers and agree to
30 bp, which is good evidence the functional form is right.

**On "LETF drag".** UPRO's gross gap to 3× the index is ~5.2%/yr, but quoting
that as the fund's inefficiency is wrong: 2.90pp of it is simply the cost of
borrowing two units of cash, which *any* leverage vehicle pays. The
wrapper-specific cost — expense ratio plus swap spread — is **2.27%/yr, about
2.5× the stated 0.91% expense ratio.**

The larger effect is structural, and it is not a cost at all:

> **Leveraged ETFs cannot express this portfolio.** There is no usable 3×
> commodity, REIT, IG-credit or HY-credit fund. Those sleeves must be held
> unlevered, consuming capital one-for-one and crowding out leverage elsewhere.
> The LETF book is therefore *structurally less diversified* than the margin
> book, which matters more than the financing spread.

Margin accounts model maintenance requirements, procyclical house-requirement
hikes, VIX-linked financing spreads, and forced liquidation tested against an
approximated intraday trough rather than the close (close-only checks are a
known source of fake survival).

---

## Outputs

Committed in [`docs/`](docs/):

| File | What it is |
|---|---|
| `findings.html` | Standalone write-up of the result, with interactive charts |
| `tearsheet_strict.png` | Full tearsheet, strict constraint |
| `tearsheet_fullsample.png` | Full tearsheet, full-sample constraint only |
| `walkforward_panelL.png` | Stitched out-of-sample curve |
| `scan_panelL.parquet` | All 512 search trials, for re-deriving any frontier |
| `robust_params.json`, `fullsample_params.json` | The two parameter sets above |

## Reproducing

```bash
pip install -e ".[dev]"
python scripts/fetch_data.py        # caches to data/cache (a few minutes, paced)
python scripts/validate_data.py     # the gates above; stops the build if any fail
pytest -q                           # 48 tests
python scripts/show_allocation.py --panel M          # what to hold, with tickers
python scripts/run_backtest.py --panel L
python scripts/run_optimize.py --panel L --trials 512
python scripts/run_walkforward.py --panel L
python scripts/compare_vehicles.py --panel M --sigma 0.30
```

---

## Caveats

- **Effective sample size is ~7, not ~13,900.** The record holds about seven
  independent major drawdown events (1973-74, 1980-82, 1987, 2000-02, 2007-09,
  2020, 2022). For a constraint defined on drawdowns, that is the sample size,
  and it is why the design budget carries a 20% haircut below the benchmark's
  realised depth.
- **1982–2021 was a 40-year bond bull market.** Any levered stock/bond result
  fitted mainly on that window is fitting a regime that cannot repeat. Stock/bond
  correlation was *positive* for most of 1962–2000 and turned positive again in
  2022; results are reported split by regime for that reason.
- **Gold's 1971–80 run was a one-off monetary regime change.** It is capped a
  priori at 15% rather than left to the optimiser, and its −70% 1980–2001
  collapse is deliberately in-sample.
- Panel X's mutual-fund sleeves are actively managed and survivorship-selected.
- Portfolio margin did not exist for retail investors before 2007; pre-2007
  results under that schedule are counterfactual.
