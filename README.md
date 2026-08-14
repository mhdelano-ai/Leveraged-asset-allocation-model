# Leveraged asset allocation: two objectives, one engine

The same daily backtest engine, the same reconstructed data and the same
discipline about lookahead, pointed at two opposite questions.

| | **Model A — constrained** | **Model B — return-maximising** |
|---|---|---|
| Objective | maximise CAGR *subject to* drawdowns no deeper than the S&P 500's | maximise CAGR, drawdown as a dial |
| Universe | 5–10 diversified sleeves | **one asset** — Nasdaq-100 or S&P 500 |
| Answer | 8.77%/yr at −13% drawdown; **loses to the S&P** | 16.3%/yr out of sample at −50%; **beats QQQ by 6.6pp/yr** |
| Where the risk goes | leverage is nearly ruled out (0.66× average) | 1.8× average, 3.4× peak |

They are worth reading together, because the honest conclusion of each is the
mirror image of the other. Model A says a hard drawdown constraint costs so much
return that the levered book is barely levered. Model B says that if you drop the
constraint and concentrate, leverage pays handsomely — but that almost all of the
*timing* edge on top of plain leverage was earned in two bear markets, and on the
S&P 500 it was not earned at all.

Two leverage vehicles are modelled throughout and compared head to head:
daily-reset leveraged ETFs, and portfolio margin / box spreads.

- [Model A — the drawdown-constrained model](#model-a--drawdown-constrained-multi-asset)
- [Model B — the return-maximising model](#model-b--return-maximising-concentrated-leverage)

---

# Model A — drawdown-constrained multi-asset

Maximise CAGR subject to a hard constraint: **drawdowns must not exceed the
S&P 500's**, tested two ways —

1. **Full sample** — worst peak-to-trough no deeper than the benchmark's.
2. **Rolling 3-year** — in *every* rolling 3-year window, no deeper than the
   benchmark's drawdown in that same window.

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
| CAGR | **8.77%** | **15.10%** |
| Max drawdown | −13.49% | −40.88% |
| Sharpe | 0.74 | 0.64 |
| Calmar | 0.65 | 0.37 |
| Average leverage | 0.66× | 2.04× |
| vs S&P 500 | **loses** 2.4pp/yr of return; 42pp shallower drawdown | **beats** by 3.9pp/yr; 14pp shallower drawdown |

**Under the strict reading you cannot beat the S&P 500.** The best robust
configuration returns 8.77% against the benchmark's 11.17%. What you get instead
is a far better ride: Calmar 0.65 vs the S&P's 0.20, and a worst loss of 13%
against 55%.

**Under the full-sample-only reading you can, comfortably** — 15.10%/yr with a
−40.88% worst drawdown, both better than the benchmark. The cost is that this
book runs ~2.8× leverage and is *deeper underwater than the S&P in many
individual three-year windows* (worst excess +14.8pp), which is exactly what the
rolling test is designed to forbid.

### The frontier between them

How much return the rolling test costs, as it is relaxed. The full-sample
condition stays strict throughout.

| Rolling-3y tolerance | Feasible | Best CAGR | Max drawdown | Avg leverage |
|---|---|---|---|---|
| 0pp (strict) | 86 / 512 | 9.94% | −17.61% | 0.83× |
| 1pp | 123 | 10.49% | −23.81% | 0.93× |
| 3pp | 202 | 10.96% | −21.52% | 1.07× |
| **5pp** | 270 | **11.98%** | −26.54% | 1.25× |
| 10pp | 433 | 14.04% | −40.33% | 1.64× |
| none (full-sample only) | 512 | 15.10% | −40.88% | 2.04× |

**Between 3pp and 5pp of tolerance is the break-even point** — that is the
smallest relaxation of the rolling test at which the strategy matches the S&P's
11.17% return, and at 5pp it does so at −27% drawdown instead of −55%.

The mirror-image question — how much return it costs to demand *safety margin*
rather than bare feasibility — is answered in Robustness below, and is the more
important one, because bare feasibility turns out not to survive contact with
new data.

### The allocation

There is no fixed allocation — the weights are an output of the rules, and they
move a lot. `python scripts/show_allocation.py` answers "what do I hold".

Long-run mix, as a share of invested capital (Panel L, strict):

| Sleeve | Proxy | Share of book |
|---|---|---|
| US equity | VTI | 43.6% |
| Intermediate Treasuries | IEF | 32.4% |
| Long Treasuries | TLT | 13.5% |
| Gold | GLD | 6.5% |
| Commodities | DBC | 4.0% |

**The strategy is not, in practice, a leveraged one.** Under the strict
constraint gross exposure averages **0.66×**, sits above 1× on only **14.3%** of
days, never exceeds 2.54×, and holds ~34% cash on average. The constraint as
specified very nearly rules leverage out. The full-sample-only configuration is
genuinely levered — 2.04× average.

The mix is regime-dependent rather than static (average gross weights by decade):

| | US equity | Long UST | Interm UST | Gold | Commodities | Gross |
|---|---|---|---|---|---|---|
| 1970s | 23.7% | 4.0% | 38.3% | 6.2% | 0.0% | 0.72× |
| 1980s | 31.3% | 8.4% | 13.4% | 2.7% | 2.8% | 0.59× |
| 2000s | 28.6% | 16.2% | 27.5% | 6.9% | 5.1% | 0.84× |
| 2020s | 26.0% | 4.0% | 10.7% | 4.6% | 2.4% | 0.48× |

As of the last bar (2026-08-13) it holds 29.6% US equity, **0% in long
Treasuries**, 3.5% intermediate Treasuries, 1.1% gold, 1.7% commodities —
0.36× gross, **64% cash**. The long Treasury sleeve is at zero because the trend
gate is off on it, which the report states rather than leaving to be inferred.

### Out of sample

Anchored walk-forward, 20-year initial training window, re-selected every 2
years, 18 folds. **The stitched out-of-sample curve is the headline result** —
never the in-sample optimum.

| | Out-of-sample strategy | S&P 500, same dates |
|---|---|---|
| Period | 1991-09 → 2026-08 | 1991-09 → 2026-08 |
| CAGR | **8.78%** | 11.04% |
| Volatility | 7.2% | — |
| Sharpe | **0.84** | ~0.37 |
| Max drawdown | **−16.96%** | −55.25% |
| Constraint | **PASS** — 0 of 1,609 windows violating | — |

The constraint holds out of sample, and the Sharpe survives a multiple-testing
haircut (observed 0.84 against an expected-maximum-under-the-null of 0.51). No
fold ever failed to find a feasible configuration.

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

- The in-sample optimum clears the constraint by **11 basis points** and has a
  **plateau radius of 0%** — a 5% move in one parameter breaks feasibility. The
  committed robust set clears by 0.76pp and survives a 10% move.
- **`sigma_target` is the binding axis.** Feasibility dies the moment the risk
  target rises, for both the robust set and the in-sample optimum, which means
  there is no clever structure buying extra return here; the constraint is a pure
  cap on how much volatility the book may run.
- Requiring genuine slack costs return: 1pp of margin → 9.10%, 2pp → 8.16%,
  3pp → 7.28%, and **no configuration at all survives a 5pp margin**.
- **Cross-panel, the strict solution fails.** Parameters chosen on Panel L
  breach by +0.80pp on Panel X (1986+, nine sleeves) — a different universe and
  era. It holds on Panel M with 4.20pp to spare.
- Cost sensitivity is benign for the strict solution (8.77% → 8.13% at 3×
  assumed costs) but material for the levered one (15.10% → 10.50%), which at
  triple costs barely beats the benchmark.

The strict solution passes **all 17** named stress episodes. The full-sample one
passes 12 of 17 — it is deeper than the S&P in 1980-82, 1987, 1994, 2013 and
2015-16.

### Where the return actually comes from

Crisis behaviour is the strategy's strongest feature, and it is not subtle. On
Panel L with the strict configuration:

| Episode | Strategy | S&P 500 | Strategy DD | S&P DD |
|---|---|---|---|---|
| 1973-74 oil shock | **+15.1%** | −37.6% | −4.0% | −45.0% |
| 2000-02 dotcom | **+17.5%** | −33.0% | −7.0% | −47.4% |
| 2007-09 GFC | **+12.8%** | −45.8% | −7.2% | −55.3% |
| 2022 inflation | −3.4% | −18.1% | −4.0% | −24.5% |

The mirror image is the bull market: 2010+ returns 6.17%/yr against the S&P's
14.52%. That trade — give up most of the upside, avoid essentially all of the
drawdown — *is* the constraint, made concrete.

---

# Model B — return-maximising concentrated leverage

No diversification requirement, no drawdown constraint, one asset plus cash.
**Maximise compound return.** The primary series is the Nasdaq-100 (1985-10+);
the S&P 500 (1960+) is the control, and it is the more sobering of the two.

## The headline finding

**Leverage is the reliable half of the answer. Timing is not — it is a bet on the
next bear market being large and slow.**

Out of sample on the Nasdaq-100, the strategy compounds at 16.26%/yr against the
index's 9.68%, with a −49.6% worst drawdown against −76.3%. That is a real and
large improvement. But decomposing it:

| step | CAGR | contributed |
|---|---|---|
| QQQ buy and hold | 9.68% | — |
| constant 1.21× leverage, same volatility as the strategy | 10.29% | +61 bp |
| the full trend / volatility / throttle stack | **16.26%** | **+596 bp** |

and then decomposing that +596 bp by episode:

| | share of the timing edge |
|---|---|
| 2000-02 dotcom collapse (563 days of 6,504) | **92.9%** |
| 2007-09 GFC | 11.1% |
| every choppy episode (2011, 2015-16, 2018, 2020) | −55.5% |
| all other days | 27.5% |

Excluding 2000-02 the edge is **+41 bp/yr**. Excluding both 2000-02 and 2007-09
it is **−25 bp/yr**. On the S&P 500 the same test comes out *negative even
including* the crises: −94 bp/yr against vol-matched constant leverage, and
−397 bp/yr with the two bears removed.

This is what a trend overlay is: a synthetic long straddle. It bleeds a small
premium in choppy markets and pays enormously in one sustained collapse. The
Nasdaq-100's 2000-02 was −82.5%, the largest sustained move in the modern record,
and it more than covered forty years of premium. The S&P's bears were not big
enough to.

**What survives the test, and what does not:**

| claim | verdict |
|---|---|
| Levering a growth index to a volatility target beats holding it | **holds** — on both indices, in sample and out |
| The book runs a far shallower drawdown than static leverage at the same risk | **holds** — −49.6% vs −83.8% out of sample, and it is structural, not event-driven |
| The trend gate is what makes leverage survivable | **holds** — removing it costs 3.2pp of CAGR and 10pp of drawdown on the Nasdaq, and on the S&P it drops the levered book *below* buy-and-hold |
| The timing rules add compound return in normal markets | **fails** — +41 bp/yr ex-2000-02 on the Nasdaq, negative on the S&P |
| The Sharpe improvement clears a multiple-testing bar | **fails** — 0.58 observed against an expected-maximum-under-the-null of 0.68 over 2,048 trials |

## Why there is an optimum at all

Compound growth in leverage is concave, and past its peak more leverage lowers
return while still raising risk. Constant leverage on the Nasdaq-100, daily
reset, financed at the bill rate + 50 bp:

| leverage | CAGR | max drawdown | $1 becomes |
|---|---|---|---|
| 1.0× | 15.81% | −82.5% | $403 |
| 1.5× | 19.22% | −94.7% | $1,318 |
| **2.0×** | **20.64%** | −98.7% | $2,135 |
| 2.5× | 19.96% | −99.8% | $1,698 |
| 3.0× | 17.20% | −100.0% | $655 |
| 4.0× | 5.87% | −100.0% | $10 |
| 5.0× | **−11.58%** | −100.0% | $0.01 |

**3× the Nasdaq-100 returns barely more than 1× and loses 99.98% on the way.**
The optimum is 2.10×, which agrees closely with the Kelly estimate of 2.13× from
sample moments — and the agreement is less reassuring than it looks, because a
moving-block bootstrap puts the 90% interval for that optimum at **1.18× to
3.10×**. Betting the point estimate is betting the whole interval, which is the
argument for running well below it.

The decomposition says why, in basis points per year:

| leverage | drift | financing | variance drag | trading | = log growth |
|---|---|---|---|---|---|
| 1× | +1,803 | 0 | −334 | −0 | 1,468 |
| 2× | +3,606 | −376 | −1,338 | −11 | 1,876 |
| 3× | +5,408 | −753 | **−3,010** | −34 | 1,587 |

Drift is linear in leverage; variance drag is quadratic. At 3× the drag alone is
30 percentage points a year, and it overwhelms everything else. Reconciliation
against the realised path is within 25 bp at every level.

## Why a dynamic rule can beat any constant

Because the growth-optimal leverage is not constant. Estimated inside states
defined by the *previous* day's data:

| state | share of days | volatility | Kelly optimum |
|---|---|---|---|
| **Nasdaq-100** | | | |
| above 200-day average | 75.7% | 20.8% | **3.20** |
| below 200-day average | 24.3% | 37.5% | **1.10** |
| calmest volatility quintile | 19.9% | 14.0% | 4.43 |
| wildest volatility quintile | 19.9% | 42.8% | 0.82 |
| **S&P 500** | | | |
| above 200-day average | 74.8% | 12.6% | **5.43** |
| below 200-day average | 25.2% | 24.2% | **−0.06** |

The S&P row is the cleanest result in the project: **below its 200-day average,
the growth-optimal leverage on the S&P 500 is zero.** Not reduced — zero. Twenty-five
percent of days, 24% volatility, and no risk premium at all to pay for it. No
volatility target discovers this, because volatility says nothing about the sign
of the drift.

Note also that the wildest volatility quintile has a Kelly of 0.82 against the
fourth quintile's 3.83 — a cut of 79%, where volatility targeting would cut only
41%. That gap is the case for a separate regime layer, and it is the one part of
the design the evidence later refuses to support (see the ablation).

## The frontier

"Maximise return" has no answer until someone says how much drawdown they will
sit through. From a 2,048-point Sobol scan over eight parameters, Panel N:

| drawdown tolerance | best CAGR | realised drawdown | average leverage |
|---|---|---|---|
| 25% | 14.22% | −24.7% | 0.81× |
| 30% | **16.42%** | −29.4% | 1.01× |
| 40% | 20.37% | −38.6% | 1.29× |
| **50%** | **23.54%** | −48.8% | 1.78× |
| 60% | 25.92% | −58.0% | 2.00× |
| no limit | 25.92% | −58.0% | 2.00× |

Two things stand out. At a **30% tolerance the strategy already matches
buy-and-hold's 15.81% return with a third of its drawdown**. And the
*unconstrained* maximum stops at −58%, not −99%: the strategy's own risk
controls bind before any external limit does, which is exactly what the static
leverage curve above cannot do.

The 50% row is used throughout as the headline configuration.

## Results

Panel N, 1985-10 → 2026-08. All figures net of financing at bill + 50 bp and of
transaction costs widened 4× before 1990.

| | strategy | QQQ buy & hold | static 2× | static 1.34× (vol-matched) |
|---|---|---|---|---|
| CAGR | **23.54%** | 15.81% | 20.64% | 18.37% |
| volatility | 34.8% | 25.9% | 51.7% | 34.8% |
| Sharpe | 0.69 | 0.57 | 0.56 | 0.57 |
| max drawdown | **−48.8%** | −82.5% | −98.7% | −92.2% |
| Calmar | 0.48 | 0.19 | 0.21 | 0.20 |

The fourth column is the one that matters. A dynamic strategy running more
leverage than the index will always show a higher return, and reporting *that* as
evidence the timing works is the standard error in this literature. Against
constant leverage carrying **the same 34.8% realised volatility**, the strategy
earns +517 bp/yr and 43 percentage points less drawdown.

### What each layer is worth

| variant | CAGR | max drawdown | Sharpe |
|---|---|---|---|
| full stack | 23.54% | −48.8% | 0.69 |
| no trend gate | 20.31% | −58.5% | 0.61 |
| no volatility regime | **23.54%** | **−48.8%** | **0.69** |
| no drawdown throttle | 24.66% | −58.0% | 0.71 |
| buy and hold | 15.81% | −82.5% | — |

**The volatility-regime layer does nothing.** The optimiser set `stress_cap` to
1.00, which switches it off, and sweeping that parameter across the whole scan
shows no systematic effect on the best achievable return at any fixed drawdown
tolerance. This is reported rather than quietly dropped because the *a priori*
case for it — the Kelly-by-quintile table above — looked strong, and it did not
survive. Volatility targeting plus a trend gate already covers the ground it was
meant to cover.

The drawdown throttle costs 1.1pp of CAGR and buys 9pp of drawdown. That is a
real trade, not a free one, and at a 60% tolerance the optimiser would rather not
pay it.

On the S&P the trend gate is not merely helpful, it is load-bearing:

| S&P 500 variant | CAGR | max drawdown |
|---|---|---|
| full stack | 14.57% | −48.8% |
| no trend gate | 10.34% | −60.2% |
| no trend, no regime | **9.87%** | −63.1% |
| buy and hold | **10.56%** | −55.3% |

**Without the trend gate the levered S&P book loses to simply owning the index.**
Leverage on its own is not a strategy.

### Out of sample

Anchored walk-forward: 15-year initial training window on Panel N, re-selected
every 2 years, 13 folds, candidates drawn from the scan.

| | strategy | QQQ buy & hold | static 1.21× (vol-matched) |
|---|---|---|---|
| Period | 2000-10 → 2026-08 | same | same |
| CAGR | **16.26%** | 9.68% | 10.29% |
| Volatility | 31.5% | 26.1% | 31.5% |
| Sharpe | 0.58 | 0.41 | 0.41 |
| Max drawdown | **−49.6%** | −76.3% | −83.8% |
| $1 becomes | **$49.17** | $10.90 | $12.59 |

Eight of thirteen folds beat the index; the worst was 2000-10 → 2002-10 at
−16.88%, against the index's −50.76%. Parameter selection was stable — the trend
floor was identical in all thirteen folds, and the risk target moved between only
two values.

The same exercise on the S&P (24 folds from 1980) gives 14.06% against the
index's 12.31% — but **−94 bp/yr against vol-matched constant leverage**, which
returned 15.00%. The strategy bought 24.6 percentage points of drawdown
protection and paid for it in return.

**The honest reading of both:** the drawdown reduction is robust and shows up
everywhere. The return advantage over equally-risky constant leverage is real on
the Nasdaq and absent on the S&P, and on the Nasdaq it is one event.

### Robustness

- **Parameters transfer across indices.** S&P-fitted parameters run on the
  Nasdaq return 20.20% against the 23.54% of Nasdaq-fitted ones, and still beat
  vol-matched static leverage by 230 bp. Nasdaq parameters on the S&P return
  14.12% against 14.57%. The rules describe leveraged equity, not one index's
  history.
- **1929-32 survives, as a labelled counterfactual.** Real prices and dividends,
  an *assumed* flat financing rate: worst drawdown −52.9% to −56.3% against the
  index's −83.7%, average leverage 1.0–1.2×, and **no ruin** at either 2% or 5%
  assumed financing. Read the drawdown, not the return.
- **The 1970s work too.** The Nasdaq Composite reaches back to 1971. Over
  1971-1985 alone — a period entirely independent of the 2000-02 event that
  carries the headline result — the strategy returns 17.53% against the index's
  8.55%, and holds 1973-74 to −51.4% against −59.1%. It fails badly in the
  1980-82 whipsaw, at −46.7% against −27.8%.
- **Costs are not the binding issue.** At 3× assumed transaction and financing
  costs the Nasdaq result falls from 23.54% to 20.26%, still well clear of
  buy-and-hold.
- **Sharpe does not clear the multiple-testing bar.** Observed 0.58 against an
  expected maximum under the null of 0.68 across 2,048 trials. Since the strategy
  and the vol-matched benchmark carry identical volatility, this caveat applies
  directly to the +596 bp figure as well.

### Where it wins and where it loses

Against constant leverage at the same volatility, by episode:

| episode | strategy | matched static | verdict |
|---|---|---|---|
| 2000-02 dotcom | −47.8% | −92.2% | **+44.4pp** |
| 2007-09 GFC | −48.3% | −65.6% | **+17.4pp** |
| Black Monday only | −16.6% | −28.3% | **+11.8pp** |
| 1987 crash | −44.9% | −50.5% | +5.6pp |
| 2022 inflation | −38.7% | −45.4% | +6.7pp |
| 1994 bond selloff | −34.5% | −21.1% | −13.4pp |
| 2015-16 China | −33.6% | −21.4% | −12.2pp |
| 2020 COVID | −40.8% | −36.7% | −4.1pp |

The pattern is completely consistent: **it wins in slow, deep bear markets that a
trend signal can see coming, and loses in sharp V-shaped shocks that it cannot.**
The wins are worth 17–44 percentage points; the losses 4–13. The asymmetry is in
the right direction, which is the actual justification for the design.

### Implementation

Can this be traded? Panel N, Nasdaq parameters, through each real vehicle:

| vehicle | CAGR | max drawdown | margin calls |
|---|---|---|---|
| idealised (no vehicle) | 23.54% | −48.8% | — |
| **portfolio margin + box spread** | **24.08%** | −48.4% | 0 |
| Reg-T margin | 21.98% | −48.6% | 0 |
| leveraged ETF (TQQQ) | 19.69% | −50.0% | n/a |

Portfolio margin financed with box spreads *beats* the idealised run, because box
spread financing is cheaper than the 50 bp default the idealised case assumes.
The LETF route costs 385 bp/yr against it. No configuration was ever margin-called,
which is a consequence of the trend gate: the book is already de-levered before
the maintenance requirement would bite.

The Nasdaq LETFs were calibrated the same way as the rest, fitting
`r = k·r_u − (k−1)(rf + s)·Δ/360 − TER·Δ/365` to each fund independently:

| fund | k | implied spread | wrapper cost | vs stated expense ratio | simulated vs actual |
|---|---|---|---|---|---|
| TQQQ | 3× | 1.044% | 2.93%/yr | **3.49×** | −133 bp/yr, corr 0.999 |
| QLD | 2× | 1.257% | 2.21%/yr | 2.32× | −34 bp/yr, corr 0.996 |

Two funds on the same index at different multipliers, fitted separately, agreeing
on the spread to 21 bp. Note what the fit can and cannot see: only the *sum* of
the expense ratio and the swap spread is identified by the data, so the split
between them is an assumption and the validation compares whole return paths
instead.

## So what would you actually hold?

On this evidence, and stated plainly because that is what the question deserves:

**Leveraged Nasdaq-100, sized to a volatility target of roughly 45%, capped near
3×, gated off when the index is below trend, with a hard drawdown throttle.** That
ran at 1.78× average leverage, 16.3%/yr out of sample against QQQ's 9.7%, and a
−50% worst drawdown against −76%.

Three things to be clear-eyed about before doing it:

1. **A −50% drawdown is the design target, not the bad case.** The unconstrained
   optimum sits at −58%. If a 30% drawdown is the real limit, the honest answer
   is 16.4%/yr, not 23%.
2. **Most of the measured timing edge is one event.** The leverage and the
   volatility sizing are the parts that generalise. Budget for the trend overlay
   to cost a little and protect a lot, not to add return every year.
3. **The vehicle matters more than the parameters.** Box-spread financing beat
   TQQQ by 439 bp/yr, which is larger than most of the parameter differences in
   the entire scan.

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
| Nasdaq-100 total return | `^NDX` price + dividends at the QQQ-implied yield | 1985-10– | corr **0.9986** and 0.17pp max-DD vs QQQ, tracking error 1.1%/yr |
| Nasdaq Composite total return | `^IXIC` price + dividends at the ONEQ-implied yield | 1971-02– | calibrated on a shorter, later window than QQQ's |

The Nasdaq total-return construction avoids an obvious-looking trap. The natural
move is to splice QQQ's adjusted close in from its 1999 inception, but QQQ's early
Yahoo history contains bad prints: +12.37% on 2000-01-07 against the index's
+5.65%, +5.91% on 2000-09-22 against −0.46%. Over 1999-2002 the two correlate at
only 0.962, which at that era's ~50% volatility is a 13%/yr tracking error — far
too large to be the expense ratio and dividend timing. The index is the clean
series, so the price path is always `^NDX` and only the *yield* is taken from the
fund, measured over 2010-2026 where QQQ tracks cleanly (1.00% gross, expense
ratio added back).

The one genuine assumption is that this yield also held before 1999, when the
Nasdaq-100 paid less than it does today — so the pre-1999 reconstruction is if
anything slightly generous. `nasdaq.yield_sensitivity` bounds it: moving the
assumption from 0% to 1.5% moves CAGR by 1.7pp and the maximum drawdown by
0.7pp, and leaves the drawdown *path* — which is what the leverage rules react
to — essentially untouched.

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
| **L** | 1971-09 → | 5 | **Model A's binding panel.** Spans 1973-74, 1980-82, 1987, 2000-02, 2008, 2020, 2022 |
| **X** | 1986-11 → | 9 | Extended universe via long-history mutual funds (survivorship-biased; treat as ~100 bp/yr optimistic) |
| **M** | 2006-02 → | 10 | Tradable ETFs. Cost realism and the vehicle comparison **only** — one crisis cannot validate a drawdown constraint |
| **N** | 1985-10 → | 1 | **Model B's primary panel.** Nasdaq-100 alone. Contains 2000-02 (−82.5%), the deepest drawdown of any major index in the modern record |
| **S** | 1960-01 → | 1 | S&P 500 alone. Model B's control, and the more sobering result. Starts where a *daily* bill rate does, not where the price index does — a levered book without a measured financing rate is not a backtest |
| **C** | 1971-02 → | 1 | Nasdaq Composite. Fourteen years longer than N and the only growth-equity series reaching 1973-74 |

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

The engine carried one bug worth recording, found while building Model B. The
leverage ratchet was anchored to the leverage currently *held* rather than to the
previous leverage *target*. Since the book only trades once the gap to its
standing target exceeds `lev_band`, and the ratchet kept every candidate within
`leverage_ratchet_up` of the held book, the two locked each other out whenever
`ratchet < lev_band` — leverage could then only advance on the monthly calendar
trigger, one step per month. A 3× target took two and a half years to reach
instead of thirty days. It is fixed, `tests/test_growth.py` has a regression test
for it, and every number in this README was regenerated afterwards. Model A's
strict result moved from 8.05% to 8.77%; Model B could not have existed without
the fix.

**Allocation (Model A)**, four layers:

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

**Allocation (Model B)**, four layers, in the order they bind:

1. **Volatility target** — `L = sigma_target / sigma_hat`, sets the scale. The
   forecast is a fast EWMA floored against 75% of trailing three-year volatility,
   because a pure fast estimate bottoms out in volatility-suppressed melt-ups and
   hands maximum leverage to the book immediately before the regime breaks.
2. **Trend gate** — the layer that carries the strategy. Deliberately *not* the
   same mapping as Model A's: that one reads 0.5 at neutral trend and averages
   ~0.7 in all weather, which for a compounding objective is a permanent 30% tax
   rather than a risk control. Model B's gate reaches full risk as soon as two of
   three momentum lookbacks are positive and ramps down only as trend genuinely
   turns. With Model A's mapping the Nasdaq book compounds at 8.6% against the
   index's 15.8%.
3. **Volatility regime** — a further non-proportional cut in the top of the
   trailing volatility distribution, ranked against a five-year rolling window
   rather than the full sample. The evidence did not support it; see the ablation.
4. **Drawdown throttle** — the same CPPI-style controller, but with an absolute
   budget rather than one derived from the benchmark, since there is no benchmark
   constraint to derive it from.

**Search.** Model A: Sobol scan over eight parameters, feasibility gated on the
hard maximum but optimised against the CVaR of window excesses — with thousands
of overlapping windows the raw maximum is an extreme order statistic set by a
single path, and optimising it directly just fits that window.

Model B: Sobol scan over eight parameters, ruin excluded, drawdown as a dial
rather than a constraint so one scan yields the whole frontier. Two guards
against the shape of the problem — compound return in leverage is flat at the top
and falls off a cliff just past it, so the in-sample argmax is precisely the point
most likely to be on the wrong side of the cliff. Every trial records the *worse*
of its two half-sample compound returns, and `opt.growth.leverage_headroom`
reports which side of its own growth peak a configuration sits on.

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
| TQQQ | 3× | 104 bp | −133 bp/yr, corr 0.999 |
| QLD | 2× | 126 bp | −34 bp/yr, corr 0.996 |

UPRO and SSO are separate regressions on different multipliers and agree to
30 bp; TQQQ and QLD do the same on a different index and agree to 21 bp. That is
good evidence the functional form is right — though only the *sum* of the
expense ratio and the swap spread is identified by the data, so the split between
them is an assumption and the validation compares whole return paths instead.

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
| `findings.html` | Standalone write-up of Model A, with interactive charts |
| `growth_findings.html` | Standalone write-up of Model B, with interactive charts |
| `tearsheet_strict.png` | Model A tearsheet, strict constraint |
| `tearsheet_fullsample.png` | Model A tearsheet, full-sample constraint only |
| `walkforward_panelL.png` | Model A stitched out-of-sample curve |
| `scan_panelL.parquet` | All 512 Model A trials, for re-deriving any frontier |
| `robust_params.json`, `fullsample_params.json` | Model A's two parameter sets |
| `growth_tearsheet_panelN.png` | Model B tearsheet, Nasdaq-100 |
| `growth_walkforward_panelN.png`, `growth_walkforward_panelS.png` | Model B out-of-sample curves |
| `growth_scan_panelN.parquet`, `growth_scan_panelS.parquet` | All 2,048 Model B trials each |
| `growth_params_N.json`, `growth_params_S.json` | Model B's two parameter sets |

## Reproducing

```bash
pip install -e ".[dev]"
python scripts/fetch_data.py        # caches to data/cache (a few minutes, paced)
python scripts/validate_data.py     # the gates above; stops the build if any fail
pytest -q                           # 83 tests

# Model A - drawdown-constrained
python scripts/show_allocation.py --panel M          # what to hold, with tickers
python scripts/run_backtest.py --panel L --params docs/robust_params.json
python scripts/run_optimize.py --panel L --trials 512
python scripts/run_walkforward.py --panel L
python scripts/run_robustness.py                     # plateau, margin, cross-panel
python scripts/compare_vehicles.py --panel M --sigma 0.30

# Model B - return-maximising
python scripts/leverage_curve.py --panel N           # start here: the reference
python scripts/run_growth_optimize.py --panel N --trials 2048 \
    --save-params docs/growth_params_N.json
python scripts/run_growth.py --panel N --params docs/growth_params_N.json
python scripts/run_growth_walkforward.py --panel N   # the headline number
python scripts/run_growth_robustness.py              # transfer, 1929, vehicles
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
- **Model B's edge rests on n≈2 events, and n≈3 if the Nasdaq Composite's
  1973-74 is counted as independent.** Trend following pays off in sustained
  bear markets, and the historical record contains very few of them. Every
  statistic about that payoff — including the headline +596 bp/yr — is an
  average over a handful of observations, and the confidence interval nobody can
  compute is much wider than the one that can be.
- **The Nasdaq-100 is survivorship-flavoured as an index.** It is a
  rules-based index of the largest non-financial Nasdaq listings, reconstituted
  annually, and the 1985-2026 record is partly the story of the technology sector
  winning. A leveraged bet on it is a leveraged bet that this continues.
- **Model B's headline configuration turns over 15× a year.** Costs are modelled
  and it survives tripling them, but that turnover implies real tax drag in a
  taxable account, which is not modelled at all.
- Panel X's mutual-fund sleeves are actively managed and survivorship-selected.
- Portfolio margin did not exist for retail investors before 2007; pre-2007
  results under that schedule are counterfactual.
