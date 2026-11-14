# Treasuries: `ust_int` (IEF-like) and `ust_long` (TLT-like)

Built by `python -m volmodel.data.sources.treasuries`. Raw downloads are cached in
`data/raw/treasuries/`, where the run also writes `segments_treasuries.csv` (this
track's rows for `segments.csv`) and `diagnostics.json`, the source of every
number below. All returns are **simple total returns** (coupon carry, roll-down
and price change), not excess over cash.

| asset | daily from | monthly from | last daily | last full month |
|---|---|---|---|---|
| `ust_int`  | 1962-01-03 | 1953-05-31 | 2026-11-13 | 2026-10-31 |
| `ust_long` | 1962-01-03 | 1919-02-28 | 2026-11-13 | 2026-10-31 |

## Sources

| source | what | used for |
|---|---|---|
| Yahoo chart API, `close` + dividend events (validated; see below) | IEF, TLT (from 2002-07-30); VFITX, VUSTX (`adjclose`, validation only) | tier A |
| Nasdaq dividend history (`api.nasdaq.com/api/quote/<sym>/dividends`, cached as `nasdaq_div_<sym>.json`) | IEF, TLT distributions (gaps 2005-2010) | second source for the distribution check |
| Fed GSW curve `feds200628.csv` (Gürkaynak, Sack & Wright) | daily Nelson-Siegel-Svensson parameters, 1961-06-14 to 2026-11-06 | tier B |
| Fed H.15 constant-maturity yields, through DBnomics (`FED/H15`) | daily 5y/10y (1962), 7y (1969), 20y (1962, gap 1987-1993), 30y (1977); monthly averages of 10y and 20y (1953-04) | tier B (pre-GSW), tier D, predictors |
| NBER Macrohistory m13033a / m13033b | yields on long-term US government bonds, monthly averages, 1919-01 to 1944-02 and 1941-10 to 1967-12 | tier D |

**Rejected source: Shiller's long rate.** Before 1953 it is annual data
interpolated linearly to months: 92% of its monthly second differences are zero
between 1871 and 1952. Returns computed from it would have almost no monthly
volatility, so it is not used.

## Method

**Definition.** `ust_int` is an equal-weight ladder of constant-maturity par
bonds at 7, 7.5, …, 10 years. `ust_long` is the same at 20, 21, …, 30 years.
These are the maturity buckets of the indices IEF and TLT track. The ladder is
rebalanced daily (monthly in the tier D eras). Duration therefore changes with
the yield level, which is real and is kept. For example, `ust_long`'s modified
duration was about 7 in 1981 and about 20 in 2020.

**GSW repricing (tier B).** On each day t-1, buy a par bond of maturity M with
coupon equal to that day's fitted semiannual par yield. On day t, value its
remaining cash flows (at 0.5k − Δt years) off day t's NSS discount function,
where Δt is calendar days / 365.25. The return is V_t − 1. It contains carry,
roll-down and price change exactly. The NSS parameters reproduce the published
`SVENY10` to within 5e-5 percentage points. The ladder is used only where the
Fed publishes yields at its longest maturity (10y from 1971-08-16, 30y from
1985-11-25). Extrapolating the curve past the published range is unreliable. In
1962-71 the extrapolated 20y daily yield changes correlate only 0.08 with H.15
and are 4× too volatile.

**H.15 CMT repricing (tier B).** Buy a par bond at y_{t-1} and price it at y_t
with maturity M − Δt, using a fractional-period street price. This assumes a
flat local curve, so it has no roll-down. Each ladder rung uses the CMT yield
interpolated linearly between the anchors (`ust_int`: 10y only;
`ust_long`: 20y only before 1977-02-15, then 20y and 30y).

**Month-average yields (tier D).** The same par-bond repricing on monthly yields,
with Δt = 1/12, and one yield used for the whole ladder.

**Splicing.** A higher-quality segment owns its whole span. An earlier segment
fills only dates before the later segment's first return. Monthly files compound
the daily returns over full months. The partial first daily month (Jan 1962) is
taken from the tier D series, and the incomplete current month (Nov 2026) is
left out. The month that contains a splice gets the worse tier (July 2002 = B).

**Known data break.** On 1973-01-04 the H.15 20y CMT jumps +74 bp while the 10y
CMT and the GSW curve fall 2-3 bp. The long-end series was re-based (flower
bonds stopped setting it). That day's `ust_long` return is dropped rather than
filled, so it is a gap.

## Segments

| asset | freq | start | end | tier | source |
|---|---|---|---|---|---|
| ust_int | monthly | 1953-05-31 | 1962-01-31 | D | H.15 10y monthly-average yield, ladder 7-10y |
| ust_int | daily | 1962-01-03 | 1971-08-16 | B | H.15 10y CMT, ladder 7-10y (flat curve) |
| ust_int | daily | 1971-08-17 | 2002-07-30 | B | GSW curve, ladder 7-10y |
| ust_int | daily | 2002-07-31 | 2026-11-13 | A | IEF close + validated distributions (Yahoo; Nasdaq dividend record) |
| ust_long | monthly | 1919-02-28 | 1941-10-31 | D | NBER m13033a long-term US bond yield (monthly avg), ladder 20-30y |
| ust_long | monthly | 1941-11-30 | 1953-04-30 | D | NBER m13033b, ladder 20-30y |
| ust_long | monthly | 1953-05-31 | 1962-01-31 | D | H.15 20y monthly-average yield, ladder 20-30y |
| ust_long | daily | 1962-01-03 | 1977-02-15 | B | H.15 20y CMT, ladder 20-30y (flat beyond 20y); 1973-01-04 dropped |
| ust_long | daily | 1977-02-16 | 1985-11-25 | B | H.15 20y/30y CMT, ladder 20-30y (interpolated) |
| ust_long | daily | 1985-11-26 | 2002-07-30 | B | GSW curve, ladder 20-30y |
| ust_long | daily | 2002-07-31 | 2026-11-13 | A | TLT close + validated distributions (Yahoo; Nasdaq dividend record) |

Each row's notes in `segments_treasuries.csv` carry its overlap statistics.

**Tiers.** Tier A is the daily price of an exchange-traded fund (market close); the
reconstructions are tier B (daily) or D (month-average yields).

**ETF distributions.** The ETF returns are close + distributions, not Yahoo's adjclose.
Each distribution is checked on its ex-date: the ETF's total return is compared with a
regression on daily CMT yield changes fitted on surrounding non-ex days (a paid distribution
leaves a residual near zero, one never paid a residual of its own size), and against
Nasdaq's dividend record where it exists
(`volmodel.data.sources.distributions.validate_distributions`; every check is in
`data/raw/treasuries/distribution_checks.csv`). Five IEF distributions of 2003 were inflated
in Yahoo's record and are corrected: 2003-07-01 0.663 -> 0.304 (price-implied), 2003-08-01
0.717 -> 0.416 (price-implied), 2003-09-02 0.764 -> 0.276 (Nasdaq), 2003-10-01 0.731 ->
0.274 (Nasdaq), 2003-12-16 0.840 -> 0.103 (price-implied; Yahoo's amount was 2.5-3x IEF's
regular payment and the price did not drop by it). `ust_int` 2003 is now +2.4% (was
+5.3%). TLT: all 290 distributions pass (204 agree with Nasdaq).

**Why VFITX and VUSTX are not spliced in as tier A before 2002.** Both have
shorter duration than the target. Our ladder is 17-18% more volatile than either
fund on monthly data (vol ratio 1.175 vs VFITX 1991-2002; 1.176 vs VUSTX
1986-2002). Splicing them in would create a level shift in volatility at 2002
that is an artifact. Their daily correlation with the curve is also only 0.88,
because mutual-fund NAV timing differs from the curve's quote time.

## Overlap diagnostics: reconstruction vs ETFs (2002-07-31 to 2026-11-06)

The GSW ladders are computed through the whole ETF era for this test. Vol ratio
is reconstruction / ETF. Tracking error (TE) is annualized.

| | freq | vol rec | vol ETF | vol ratio | corr | TE |
|---|---|---|---|---|---|---|
| ust_int vs IEF | daily | 6.9% | 6.8% | 1.022 | 0.963 | 1.9% |
| ust_int vs IEF | monthly | 6.9% | 6.6% | 1.045 | 0.987 | 1.1% |
| ust_long vs TLT | daily | 14.0% | 14.3% | 0.978 | 0.959 | 4.1% |
| ust_long vs TLT | monthly | 13.4% | 13.5% | 0.995 | 0.990 | 1.9% |

By sub-period (daily vol ratio / corr):

| | 2002-07 | 2008-14 | 2015-19 | 2020-26 |
|---|---|---|---|---|
| ust_int vs IEF | 1.035 / 0.954 | 1.053 / 0.962 | 1.022 / 0.976 | 0.981 / 0.966 |
| ust_long vs TLT | 1.051 / 0.953 | 0.993 / 0.960 | 1.007 / 0.976 | **0.927** / 0.955 |

Yearly vol ratios for `ust_long` vs TLT: 1.04-1.07 in 2002-2009, about 1.0 in
2015-2021, and **0.86-0.93 in 2022-2026**. The cause is the duration mismatch
between par bonds and what TLT actually holds:

- **2002-2009:** TLT held old high-coupon bonds, and no 30-year bonds were
  issued in 2001-2006, so TLT's duration was shorter than a par 20-30y ladder.
- **2022-2026:** TLT is full of 1.25-2% coupons issued in 2020-21. These now
  trade at deep discounts and have longer duration than par bonds of the same
  maturity.

The predictor file shows this directly. `duration_ust_long_etf_est` is a rolling
252-day regression of TLT's 5-day returns on the ladder's par-yield changes. It
averages 16.3-17.3 in 2023-2025, against 14.4-15.3 for the par ladder (the ratio
is about 1.13, matching the vol ratio of about 0.88). In 2004-2008 it is 12.4-13.5,
against 13.9-15.2 for the par ladder. For IEF the two durations agree within
about 0.7 year. Annual means of the gap (ladder minus estimate) run from −0.3 to +0.65 year.

Daily correlation is about 0.96, not 0.99. That comes from timing, not from the
method. The H.15 CMT ladder gives the same 0.962 vs IEF. Lead/lag correlations
are ±0.03, and correlation of 2-day returns rises to about 0.975. The curve is quoted
around 3-3:30pm and the ETF closes at 4pm, and the ETF premium/discount adds
noise. Monthly correlation is 0.987-0.990.

**Validation of the pre-GSW CMT method on the overlap with the GSW ladders:**

| | freq | vol ratio | corr | TE |
|---|---|---|---|---|
| ust_int CMT 10y-flat vs GSW (1971-2026) | daily | 1.006 | 0.965 | 1.8% |
| | monthly | 0.970 | 0.985 | 1.3% |
| ust_long CMT 20/30 interp vs GSW (1985-2026) | daily | 1.025 | 0.964 | 3.6% |
| | monthly | 1.027 | 0.972 | 3.1% |
| ust_long CMT 20y-flat vs GSW (1985-2026) | daily | 1.049 | 0.965 | 3.6% |
| | monthly | 1.049 | 0.974 | 3.0% |

The 20y-flat variant used before 1977 overstates volatility by about 5%: in this
period the 20y yield was more volatile than the 25-30y yields.

## Month-average yields: damping (tier D)

Compare returns computed from H.15 monthly-average yields with month-end returns
compounded from the daily CMT ladders, both from the same H.15 source.

| | period | vol ratio (avg / month-end) | corr | AC1 avg-based | AC1 month-end |
|---|---|---|---|---|---|
| ust_int | 1962-2026 | **0.846** | 0.730 | **0.32** | 0.10 |
| ust_int | 1962-1986 | 0.858 | 0.743 | 0.35 | 0.11 |
| ust_int | 1994-2026 | 0.832 | 0.718 | 0.27 | 0.09 |
| ust_long | 1962-2026 | **0.849** | 0.728 | **0.27** | 0.07 |

For a random walk the theory predicts a vol ratio of √(2/3) ≈ 0.82 and AC1 of
0.25. The tier D segments therefore understate monthly volatility by about 15%
(variance by about 28%) and carry spurious positive autocorrelation. Their
monthly correlation with the true return is only 0.73.

Cross-checks between the tier D sources:

- NBER m13033b vs H.15 20y average (1953-67): vol ratio 1.036, corr 0.975.
- m13033a vs m13033b (1941-44): vol ratio 1.71, corr 0.63. This overlap falls in
  the WWII rate peg, where both series barely move; TE is 2.4%/yr.

## Daily serial correlation before about 1990 (important for the RV target)

Daily tier B returns before about 1990 are **positively autocorrelated**. The
daily H.15 data from 1962 to 1966 have 36-53% zero changes (quotes in 32nds).
Daily yield changes have AC1 of +0.1 to +0.3 in both GSW and H.15 until the
mid-1980s. The variance ratio of 21-day to 1-day log returns:

| | 60-64 | 65-69 | 70-74 | 75-79 | 80-84 | 85-89 | 90-94 | 95-99 | 00-04 | 05-09 | 10-14 | 15-19 | 20-24 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ust_int | 1.23 | 1.46 | 2.67 | 2.21 | 1.48 | 1.63 | 1.08 | 1.07 | 0.87 | 0.90 | 0.86 | 1.14 | 1.02 |
| ust_long | 1.48 | 1.75 | 2.07 | 1.59 | 1.20 | 1.55 | 1.08 | 0.99 | 0.90 | 1.13 | 0.79 | 1.02 | 0.82 |

The same pattern appears in both the GSW and the CMT reconstructions, so it is
in the underlying quotes and not in the method. (In 1972-76, VR21 is 2.25 for
the GSW ladder and 1.95 for the CMT ladder.) **Consequence:** in the 1960s-80s,
annualized volatility from daily returns is well below volatility from monthly
returns. For example, `ust_int` in the 1970s shows 4.2% from daily returns and
6.4% from monthly returns. The data cannot tell how much of this is stale or
smoothed quotes and how much is genuine momentum in yields. A daily-RV target in
these decades understates 12-month variance. Monthly-based variance is the safer
target before about 1990.

## Annualized volatility by decade

Vol is std × √12 of monthly returns and std × √252 of daily returns.

| decade | ust_int monthly | ust_int daily | ust_int tiers | ust_long monthly | ust_long daily | ust_long tiers |
|---|---|---|---|---|---|---|
| 1920s | | | | 4.5% | | D |
| 1930s | | | | 6.0% | | D |
| 1940s | | | | 2.8% | | D (rate peg) |
| 1950s | 3.2% (from 1953-05) | | D | 4.1% | | D |
| 1960s | 3.8% | 2.8% | B (D Jan 62) | 5.8% | 4.8% | B |
| 1970s | 6.4% | 4.2% | B | 8.4% | 6.3% | B |
| 1980s | 10.6% | 8.7% | B | 14.9% | 13.3% | B |
| 1990s | 6.0% | 6.0% | B | 9.0% | 9.4% | B |
| 2000s | 7.0% | 7.1% | B→A | 12.5% | 12.2% | B→A |
| 2010s | 5.6% | 6.0% | A | 12.7% | 13.7% | A |
| 2020s (to 2026-10) | 7.3% | 7.4% | A | 14.6% | 16.4% | A |

Monthly AC1 is 0.25-0.43 in the tier D decades, against -0.1 to +0.15 later.
This is the damping signature described above.

## Known episodes

Daily vol is annualized. The cumulative return is from monthly data over the
window.

| episode | ust_int cum | ust_int daily vol | ust_long cum | ust_long daily vol | notable days (ust_long) |
|---|---|---|---|---|---|
| Volcker 1979-10 to 1982-12 | +37.6% | 11.8% | +31.3% | 17.3% | 1980-02-19 −3.8%, 1980-04-16 +4.6% |
| Oct 1987 | +4.9% | 18.4% | +7.9% | 24.9% | 1987-10-20 +5.8% (int +4.4%) |
| 1994 | −6.1% | 7.5% | −8.3% | 11.2% | 1994-04-04 −3.3% |
| 2008 Q4 | +12.3% | 11.6% | +27.5% | 23.0% | 2008-11-20 +5.2% |
| Taper, May-Sep 2013 | −5.5% | 7.6% | −12.5% | 15.2% | 2013-07-05 −3.4% |
| March 2020 | +3.7% | 19.8% | +6.4% | **59.9%** | 03-17 −6.7%, 03-20 +7.5% (TLT dislocation) |
| 2022 | −15.2% | 10.2% | −31.2% | 20.3% | |

The Volcker window includes the 1982 rally, so its cumulative return is positive.
Monthly volatility of `ust_long` in 1979-82 is more than twice that of 1962-68
(tested). The WWII peg (1942-50) has monthly vol below 5% (tested).

## Predictors: `data/processed/predictors/treasury_curve.parquet`

The file is in long format (`obs_date, available_date, series, value`). Yields are in
**decimals**; durations are in years. `available_date` follows the publication of the
input: H.15 CMT yields are posted the next business day (`available_date = obs_date + 1
business day`); everything priced off the GSW fitted curve (`yield_ladder_*`, the ladder
durations in the GSW era, and the ETF duration estimates, whose regressor is the GSW ladder
yield) is published by the Fed with a lag of about a week, so it is stamped 10 business days
after observation (the same rule as the TIPS track's GSW series). `yield_20y` starts on
1973-01-04: on that day the H.15 20y series was re-based (+74bp while the 10y moved -2bp;
the level stays shifted), so the earlier observations are a different series and are not
published as the same predictor.

| series | from | content |
|---|---|---|
| `yield_2y`, `yield_5y`, `yield_7y`, `yield_10y`, `yield_20y`, `yield_30y` | 1976-06, 1962-01, 1969-07, 1962-01, 1973-01-04 (gap 1987-01 to 1993-09), 1977-02 | H.15 CMT par yields |
| `duration_ust_int`, `duration_ust_long` | 1962-01 | modified duration of the par-bond ladder, using the yields of the segment in force (CMT before GSW coverage; GSW after; CMT after the last GSW date) |
| `yield_ladder_ust_int`, `yield_ladder_ust_long` | 1971-08, 1985-11 | mean GSW par yield of the ladder |
| `duration_ust_int_etf_est`, `duration_ust_long_etf_est` | 2003-08 | rolling 252-day regression of the ETF's 5-day log return on −Δ(ladder par yield); uses only past data |

In the ETF era the asset returns are the ETFs themselves. For TLT after 2020,
`duration_ust_long_etf_est` is the better guide to its interest-rate exposure;
`duration_ust_long` understates it by about 2 years.

## Caveats

- **ust_int before 1953 is missing.** No credible monthly intermediate-Treasury
  yield exists before 1953 among the sources that respond here. Shiller is
  interpolated, and NBER m13057 covers only 3-5y issues from 1945.
- **Flower bonds before 1973.** The long-end yields (NBER m13033b and H.15 20y)
  include estate-tax-eligible "flower" bonds, which traded rich. Before 1973
  their daily changes were less volatile than the 10y: 1.2 vs 1.8 bp/day in
  1962-66 and 3.9 vs 4.4 in 1967-71, while the two were equal after 1973.
  `ust_long` volatility in the 1960s may therefore be understated.
- **Partially tax-exempt bonds, 1919-1941.** NBER m13033a covers bonds due or
  callable after 12 years (later 15), not a 20-30y ladder. The ladder duration
  is applied to that yield.
- **No volatility rescaling anywhere.** Tier D is reported as-is. The roughly
  15% damping is documented, not corrected.
- **Mismatch with TLT at the splice.** The `ust_long` reconstruction is a par
  ladder. Because TLT's effective duration has drifted around it, the
  reconstruction is a 20-30y par ladder rather than "TLT backfilled" in every
  era. At the 2002 splice the two had about the same volatility (2002-07
  daily vol ratio 1.05).
- **Data lag at the end.** GSW ends 2026-11-06 and H.15 ends 2026-11-09. Both
  asset series end with the ETFs on 2026-11-13.
