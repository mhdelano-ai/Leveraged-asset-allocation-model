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
- [Appendix — Kelly on the four-asset question](#appendix--kelly-on-the-four-asset-question)
- [Appendix — Kelly inside a Reg-T retail account](#appendix--kelly-inside-a-reg-t-retail-account)
- [Appendix — Kelly at today's yields](#appendix--kelly-at-todays-yields)

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

## Live dashboard

The four-state rotation (see `growth_findings.html`) reduces, day to day, to
two questions: *what state is the market in, and does that differ from what I
hold?* `docs/dashboard.html` answers them without running Python — a
self-contained page, hostable free on GitHub Pages, that fetches what it needs
and stays current whenever it is opened.

**What it does.** `scripts/build_dashboard.py` bakes in the full 1985→present
backtest at generation time via `lam.report.dashboard`, reusing
`backtest_four_state` exactly as the research does — same costs, same one-day
lag, same improved variant (`RuleParams(up_wild=1.0)`: above trend but too
volatile to lever, hold 1x QQQ rather than cash). Only the last day or two of
prices are ever fetched live: on open, the page reads a Twelve Data API key
from `localStorage`, pulls recent QQQ and TQQQ daily bars, and computes the
state, the SMA reading and the 20-day realised volatility **client-side**.
That JavaScript is not a reimplementation taken on faith — `computeStateSeries`
in `docs/dashboard.html` is checked line-for-line against
`lam.alloc.rules.four_state_signal` by `tests/test_dashboard.py` via
Playwright, including the ±1% hysteresis band around the SMA crossing and its
hold-the-previous-state behaviour on choppy days.

**Setup.** Get a free key at [twelvedata.com](https://twelvedata.com/pricing)
(800 calls/day; the page caches each day's fetch in `localStorage`, so opening
it repeatedly costs about two calls a day regardless of how often). Paste the
key into the page's setup card on first open — it is stored only in that
browser and is never committed to this repository or sent anywhere but Twelve
Data. Without a key, or if Twelve Data is unreachable, rate-limited, or rejects
the key, the page falls back to the baked history with an explicit "as of"
banner rather than presenting stale numbers as current.

**Regenerating.** Re-run the generator whenever the model, its parameters, or
the baked history changes — the live segment updates itself on every open and
needs no rebuild:

    python scripts/build_dashboard.py

**On the SMA basis.** `four_state_signal` computed its trend reading on the
*total-return* index by default, but a live page can only ever see *raw*
closes — a price feed does not reinvest dividends for you. The two drift apart
by roughly 0.4% over a 200-day window, which is enough to flip a borderline
day, so `four_state_signal` gained a `price_basis` parameter and the
dashboard's baked backtest is generated on that same raw-close basis, so the
history it shows and the signal it computes live share one definition rather
than two that quietly disagree. A ±1% hysteresis band around the crossing
(`RuleParams.sma_band`, also new) was added at the same time, so the SMA does
not need to move much to keep whipsaw days from re-triggering a state change.
Together these move the improved variant on Panel N from the previously
reported 25.35% CAGR / −67.0% max drawdown to **25.43% / −67.6%, at 7.9
switches/yr** (down from 11.4/yr — the band earns its keep in choppy stretches
like 2011 and 2015-16).

---

## Outputs

Committed in [`docs/`](docs/):

| File | What it is |
|---|---|
| `dashboard.html` | Live operating dashboard for the four-state rotation — see [above](#live-dashboard) |
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
pytest -q                           # 107 tests

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

---

# Appendix — Kelly on the four-asset question

A separate question, asked of the same engine and the same discipline about
lookahead: **what is the Kelly-optimal portfolio of US equities, international
ex-US equities, US bonds, international ex-US bonds and cash, based entirely on
historical data?**

Full write-up: [`docs/kelly_findings.html`](docs/kelly_findings.html). Reproduce
with `python scripts/run_kelly.py --draws 1000`; raw output in
`docs/kelly_report.txt` and `docs/kelly_results.json`.

### The answer

**Full Kelly is not a portfolio.** On 1993–2026 monthly data it asks for
**17.5× gross exposure** — +3.45 US equities, −1.53 international equities,
0.00 US bonds, **+12.48 international bonds**, financed by a −13.4× cash
position — and an account that followed it, re-estimating the weights every year
from data available at the time, **was wiped out in September 2008** on a
−113.9% month.

| Kelly fraction | In-sample log growth | Out-of-sample log growth |
|---|---|---|
| 0.25× | 14.9% | 11.2% |
| **0.50×** | 23.8% | **15.0% — the out-of-sample peak** |
| 0.75× | 29.6% | 10.4% |
| 0.90× and above | 31.3% | **ruin** |

In sample, growth rises monotonically to 1.0× — that is what "optimal" means and
it is evidence of nothing. Out of sample it peaks near **half Kelly** and then
collapses.

### Two structural findings

**Without leverage, Kelly is 100% US equities.** Long-only with no borrowing, the
log-optimal portfolio is all US equity — in both panels, at the bootstrap median,
and after shrinking the means toward an equal-Sharpe prior. At 1× exposure a
15%-vol book gives up ~1.1pp/yr to variance drag, nowhere near enough to buy a
4%-return bond sleeve a place. **Kelly is not an argument for diversification;
it is an argument for leverage.** Bonds earn a weight only when they can be
levered, and levering them is what produces the ruin above.

**The weights are not identified by the data.** A 12-month block bootstrap puts
the US bond weight's 90% interval at −3.10 to +4.54 — the sign is unknown — and
gross exposure between 7.8× and 29.9×. Re-solved decade by decade, the
2×-capped solution is 0.59 US equity / 1.41 international bonds in the 1990s,
0.00 US equity / 1.22 international equity in the 2000s, and 2.00 US equity in
the 2010s: three consecutive decades, three unrelated portfolios.

Run honestly out of sample, the unlevered optimiser earned **9.99%/yr against
11.53%** for simply owning US equities and never solving anything.

### Data, and its one severe limitation

| Sleeve | Panel `modern` (2013-07→) | Panel `long` (1993-01→) |
|---|---|---|
| US equities | VTI | VFINX |
| Intl ex-US equities | VXUS | VTRIX *(active, survivorship-selected)* |
| US bonds | BND | VBMFX |
| Intl ex-US bonds | BNDX | PFORX *(active, survivorship-selected)* |
| Cash | FRED `DTB3`, discount→BEY | same |

**International bonds bind everything.** The asset class as it is actually bought
— currency-hedged global aggregate ex-USD — has a tradable index history starting
2013-06. Thirteen years cannot support a Kelly weight, so the long panel
substitutes the longest-running hedged foreign bond fund, and its 0.85 Sharpe
ratio (an active fund, selected for having survived, measured across a 30-year
global bond bull market) is precisely what the levered solution is betting on.
On the clean modern data those same bonds returned *less than cash*, and Kelly
gives them nothing at all.

---

# Appendix — Kelly inside a Reg-T retail account

The Kelly answer above assumes a frictionless account. A retail margin account —
Robinhood's, and every other Reg-T broker's — breaks that assumption in the one
way that matters: **a maintenance breach converts a drawdown into a realised
loss.** The broker sells at the bottom, and the position is not there for the
recovery.

`src/lam/kelly/account.py` simulates this daily, because a maintenance breach is
a path event inside the month. It models the Reg-T 2× cap, per-asset maintenance
requirements, house requirement hikes as the market falls, intraday testing
rather than close-to-close, slippage on forced sales, and interest accrued daily
on the debit balance. Report:
[`docs/kelly_robinhood.html`](docs/kelly_robinhood.html), raw output in
`docs/kelly_account_report.txt`, run with
`python scripts/run_kelly_account.py --spread 0.012`.

### The maintenance requirement, not Kelly, sets the leverage

100% US equity rebalanced monthly to a fixed gross exposure, 1993–2026,
borrowing at bills + 1.2%:

| Gross | CAGR | Max drawdown | Margin calls | First call |
|---|---|---|---|---|
| 1.00× | 10.86% | −55.3% | 0 | — |
| 1.25× | 12.31% | −64.3% | **0** | — |
| **1.50×** | **13.60%** | −71.9% | **0** | — |
| 1.75× | 14.64% | −80.3% | 1 | Oct 2008 |
| 2.00× | 15.10% | −87.4% | 6 | Jul 2002 |

**1.25× is never called under any assumption tested; 1.5× survives everything
except a 35% maintenance requirement.** At 1.75× and above, a 5pp change in a
number the broker sets unilaterally flips the account from "one bad month" to
"serially liquidated".

### Above 1.5× the extra return is a behavioural bet

Identical simulation, differing only in whether the investor re-levers after
being sold out:

| Gross | Re-levers | Does not re-lever |
|---|---|---|
| 1.50× | 13.60% | 13.60% *(never called)* |
| 1.75× | 14.64% | 10.74% |
| 2.00× | 15.10% | **11.26%** |

At 2×, flinching once leaves the account at 11.26%/yr against 10.86% for never
having borrowed — 40bp/yr, for a −77% drawdown. The whole case for exceeding
1.5× rests on re-borrowing in October 2008.

### Two mechanical facts worth more than the backtest

**The call distance is computable in advance.** At leverage `L` against
maintenance `m`, the call fires once the position has fallen
`1 − (L−1)/(L(1−m))` from the last rebalance — 33% at 2×, 56% at 1.5× on the
textbook rule; 20% and 43% once intraday testing and procyclical hikes are
priced in (`lam.kelly.account.call_threshold`).

**Speed matters more than depth**, because that distance is measured from the
last rebalance and monthly rebalancing sells on the way down. 1.5× came through
2008's slow −55% untouched; 2× was called six times. The crash that breaks a
levered account is a fast one, not necessarily a deep one.

### Leveraged ETFs are not a substitute (the IRA case)

A 2× daily-reset fund held 1993–2026 returned **10.68%/yr at a −91.5%
drawdown** — *less than the unlevered index* (10.86%), at nearly twice the
drawdown, because the daily reset makes volatility drag scale with the square of
the multiplier and the embedded swap financing is wider than a margin loan. Its
only real advantage is that it cannot be margin-called.

---

# Appendix — Kelly at today's yields

Both appendices above run on sample means, and the honest reading of the first
was that 33 years cannot identify four risk premia. Historical means are also
*biased for this purpose*: the sample contains a 40-year fall in yields that
cannot repeat, and a re-rating of US equities that shows up as return but is a
change in the price paid, not in the cash flow earned.

`src/lam/kelly/forward.py` estimates expected returns the way an allocator has
to — from what the assets currently yield. Bonds: index YTM less expected credit
loss, and for hedged foreign bonds the foreign yield **plus the hedge carry**
(rolling FX forwards earns roughly the short-rate differential, worth +1.6pp on
euro bonds and +2.6pp on JGBs today). Equities: the Grinold–Kroner build-up, with
income and inflation observable and growth stated as an explicit assumption.
Report: [`docs/kelly_forward.html`](docs/kelly_forward.html), run with
`python scripts/run_kelly_forward.py --paths 400`.

### The inputs, as of 2026-08-20

| | Income | + growth & inflation | Geometric | Vol | Over cash |
|---|---|---|---|---|---|
| US equities | 1.17% div + 1.30% buyback | 1.50% + 2.34% | 6.31% | 17.5% | +4.05% |
| Intl ex-US equities | 2.90% div + 0.60% buyback | 1.50% + 2.34% | **7.34%** | 16.4% | +4.89% |
| US bonds | 4.85% YTM | — | 4.70% | 6.0% | +1.08% |
| Intl ex-US bonds, hedged | 4.63% YTM | — | 4.58% | 4.9% | +0.91% |
| Cash — 3m bill | 3.80% | — | 3.80% | — | — |

Kelly is defined on **arithmetic** means, so σ²/2 is added — worth 1.5pp on
equities, larger than the entire bond risk premium. Risk is measured over the
last five years, not thirty: the US stock/bond correlation is now **+0.22**
against −0.30 or lower through 2000–2020.

### The answer moves by an order of magnitude

| | Historical inputs | Forward inputs |
|---|---|---|
| Equity premium | 8.92% | 4.36% |
| Full Kelly, borrowing at bills + 1.2% | **17.5× gross** | **1.18× equities** |
| Bond weight | 12.5× intl bonds | **zero** |

**Bonds get nothing, for a structural reason rather than a statistical one:** US
bonds yield 4.85% and hedged foreign bonds 4.63%, and Robinhood lends at 5.00%.
You cannot borrow at 5.00% to buy a 4.70% expected return. At zero financing cost
the optimiser wants 2.4× bonds; at any realistic retail spread it wants none.

### Forward account risk

400 bootstrapped 10-year paths through the same Reg-T account — historical daily
*shocks* rescaled to forward volatility and re-centred on forward expected
returns:

| Gross | Median CAGR | 5th pct | Median max DD | P(margin call) | P(lose to cash) |
|---|---|---|---|---|---|
| 0.50× | 5.62% | 1.16% | −15.9% | 0% | 22.5% |
| 1.00× | 6.64% | −2.42% | −32.8% | 0% | 29.0% |
| **1.25×** | **6.65%** | −4.68% | −40.8% | 0% | 32.0% |
| 1.50× | 6.53% | −7.33% | −48.6% | 0% | 36.8% |
| 1.75× | 6.18% | −9.96% | −55.7% | 17.8% | 40.8% |
| 2.00× | 5.73% | −12.61% | −62.7% | 52.2% | 45.2% |

The Monte Carlo puts the growth-optimal leverage at 1.25×, independently of the
analytic Kelly solution of 1.18×. Between 1× and 1.5× the median curve is flat to
within 12bp — leverage in that range buys no expected growth and 16pp of extra
median drawdown. Over 20 years the ranking is unchanged but the forced-sale risk
compounds: P(margin call) at 2× rises from 52% to **78%**.

### Where the 1.5% growth assumption comes from

It is the only number in the build-up that is neither observed nor derived, and
**it is an *aggregate* rate, not per share** — the buyback yield is the separate
term that converts one to the other. What an index-fund holder receives in growth
is the sum: **1.50% + 1.30% = 2.80%/yr real, per share**. Dropping a historical
*per-share* growth rate into the growth slot beside a buyback term double counts
the share count, which is the classic error in this decomposition.

Measured against the record (Shiller real EPS, 1871–2023, ten-year averaged
endpoints):

| Period | Real EPS growth |
|---|---|
| 1871–2023 — full record | **1.80%** |
| 1900–2023 | 1.68% |
| 1950–2023 | 2.33% |
| 1960–2000 | 1.22% *(2.26% on raw endpoints — same four decades)* |
| 1985–2023 — the buyback era | 3.86% |

Across all 1,199 rolling 30-year windows: 5th pct **−0.51%**, median **1.68%**,
95th pct 3.32%, best ever 3.72%. There are thirty-year stretches in which real
earnings per share went backwards.

**The base case's implied 2.80% sits at the 83rd percentile of that
distribution** — 1.50% looks conservative in isolation and is not, once buybacks
are added. Run it the other way: for US equities to return 8%/yr, per-share real
growth must be **4.49%**, higher than any 30-year stretch since 1871; to repeat
their 1993–2026 realised 10.82% would take **7.31%**, roughly double the best
ever. That is not a claim about growth being weak — it is the starting dividend
yield, which averaged ~4.4% across the record and is 1.17% today.

Why 1.50% rather than the ~3.3% aggregate growth the record implies (1.80% per
share plus ~1.5%/yr of historical dilution, close to real GDP over the same
span): potential real GDP growth is now nearer 1.8%, and much of the 1985–2023
acceleration was margin expansion from what is now a record profit share — a
level shift, not a growth rate. The counter-argument is coherent: 2.5% aggregate
growth puts per-share growth at 3.8% and full Kelly near 1.5×. That is exactly
the belief the 1.5× rung requires.

| If per-share real growth is… | US equities return | Full Kelly |
|---|---|---|
| −0.51% — worst 30y in the record | 3.00% | 0.04× |
| 1.68% — the median 30y | 5.19% | 0.90× |
| **2.80% — this model's base case** | **6.31%** | **1.18×** |
| 3.72% — best 30y in 150 years | 7.22% | 1.52× |

### Is recent history the better guide?

The case that modern companies compound faster is right about something large,
and the decomposition separates it from what cannot repeat. Real per-share EPS
growth = real GDP growth + profit-share drift − dilution, with aggregate profits
deflated by the GDP deflator so the first two terms are exactly additive:

| Era | Real GDP | Profit-share drift | Aggregate real profits | Real EPS/share | Dilution |
|---|---|---|---|---|---|
| 1947–1985 | +3.66% | +0.10% | +3.75% | +1.64% | **+2.12%** |
| 1985–2023 | +2.61% | **+1.75%** | +4.41% | **+3.86%** | +0.55% |
| 1995–2023 | +2.48% | +1.55% | +4.07% | +3.65% | +0.42% |
| 2010–2023 | +2.42% | +0.53% | +2.96% | +3.80% | **−0.84%** |

**What the argument gets right, worth ~3pp/yr.** Dilution ran at +2.12%/yr
against shareholders in 1947–1985 — aggregate profits grew 3.75% and per-share
earnings only 1.64%. By 2010–2023 it is −0.84%: buybacks now *add* most of a
point a year. That is a genuine forty-year regime change, not a phase, and
nothing here assumes it reverses.

**What it gets wrong, worth about the same.** The other half of 1985–2023's
3.86% is +1.75%/yr of the profit share rising from ~5% to 11.4% of GDP. That is a
level shift, not a growth rate — the share doubled and cannot double again.
Holding it permanently at today's record contributes exactly zero to growth.
Falling tax rates (46% → 21%) and falling interest expense are the same kind of
one-off.

| Assumption set | US per-share growth | Percentile of 30y record | Global equity | Over cash | Full Kelly |
|---|---|---|---|---|---|
| Median 30 years | 1.68% | 50th | 5.57% | +3.24% | 0.90× |
| **Base case** | 2.80% | 83rd | 6.69% | +4.36% | **1.18×** |
| **Modern regime** — the argument granted | 3.86% | 100th | 7.54% | +5.21% | **1.50×** |
| Modern regime, margins keep rising | 4.55% | off the scale | 7.98% | +5.65% | 1.66× |

The base case already concedes most of it: 2.80% is modern GDP growth plus the
entire modern buyback yield with zero further margin expansion. Taking the full
3.86% means assuming the next thirty years beat *every* thirty-year window since
1871 — coherent about a changed world, uncomfortable to lever into. Available as
`PRESETS["modern"]`.

**One trap:** adopting 3.86% requires zeroing the buyback term. It is already a
per-share figure; adding 1.30% on top is not a bolder forecast, it is the share
count counted twice.

Under modern-regime assumptions full Kelly is **1.50×** — the same rung the
historical backtest recommended, reached by an entirely different route. What it
does *not* change: bonds still yield less than the margin rate, the
US/international split is still a knife edge, and margin-call probabilities are
driven by volatility rather than expected return, so 1.75×+ stays indefensible
under every growth assumption.

### Two sensitivities that matter more than the base case

**The US/international split is a knife edge.** A ±2pp swing in an unobservable
growth assumption moves it from all-international to all-US, while *total equity
exposure barely moves* (1.22× to 1.56×). The optimiser has a firm view on how
much equity and no real view on which — so hold market weights and spend the
conviction on the leverage decision.

**What 1.5× requires you to believe.** Full Kelly reaches 1.5× only if global
equities compound at 7.7% rather than 6.7% — about 0.9pp/yr more growth than the
yields imply. Defensible, but a view rather than an optimisation.

### The rule, re-runnable

    L* = (expected equity return − borrowing rate) / volatility²

with every input published daily. Re-run it when the dividend yield moves (a
market decline *raises* Kelly leverage), when the broker's rate moves (it
currently costs a third of the leverage), or when volatility moves — it enters
squared, so 16% → 24% more than halves the answer.
