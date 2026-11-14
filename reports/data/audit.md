# Independent data audit of the assembled panel

Auditor: an independent agent with no stake in the data. Scope: everything `load_panel()` returns over the
full history (no `ASSETFORECAST_MAX_DATE`): 12 asset return histories, cash, tiers, 62 predictors, and the
`Panel.asof` contract. Regression tests: `tests/data/test_audit.py` (68 pass; 6 strict `xfail`s that encode
the confirmed defects and will flip to passing when they are fixed). Scratch scripts and downloaded
cross-check data are in the scratch directory under `audit/` (`cmp_indep.py`, `div_scan.py`, `badprint_scan.py`,
`splices.py`, `aud.py`).

Severity measures how much a finding can distort a 12-month 21-day-horizon vol or correlation forecast, or the
integrity of the dev/holdout split. Status is **confirmed** (reproduced with independent evidence) or
**suspected**.

---

## Executive summary

| # | finding | severity | status | estimated effect | recommended fix |
|---|---|---|---|---|---|
| 1 | **`ig_credit` before 2002 is calibrated on post-2014 data.** The constant-mix fund-blend weights (0.59/0.70) and the Ibbotson/T-bill weight (0.81) are vol-matched to LQD over 2002-08..2026-10. The tier-D maturity is matched over 2002-08..2016-09. So the holdout leaks into the history every model trains on. | **high** (holdout integrity), medium (size) | confirmed | Vol-matching on 2002-08..2014-12 alone gives weights 0.75 / 0.68 / 0.73. 1926-79 monthly vol would be **0.0367 instead of 0.0407 (as-built is 11% higher)**. 1985-2002 blend vol would be 0.0536 instead of 0.0511 (as-built is 5% lower). | Re-calibrate on data ≤ 2014-12, or better, fix the weights by a rule that uses no LQD vol (e.g. a duration match from fund fact sheets). Rebuild `credit` and re-assemble. |
| 2 | **Spurious Yahoo distributions in IEF and LQD (2003).** `adjclose` credits five inflated or non-existent IEF distributions (2003-07-01, 08-01, 09-02, 10-01, 12-16) and one LQD distribution (2003-12-16). The price does not drop on those days, and the Nasdaq record shows the real amounts (e.g. 0.276 vs Yahoo 0.764). | medium | confirmed | `ust_int` 2003 return **+5.3% vs actual ≈ +2.2%**. Target vol for origins whose window covers 2003: up to **+5.7%** (`ust_int`) and **+5.5%** (`ig_credit`, mean +3.9%). 21-day correlation with `us_large` shifted by up to **0.10**. | Build IEF/TLT/LQD/HYG/VNQ/EFA/EEM returns from close + distributions, as the TIPS and US-equity tracks do. Verify each distribution against the ex-date price drop (bond ETFs) or a second dividend record. |
| 3 | **`hy_credit` 1985-2007 (VWEHX) understates the target-horizon vol relative to the HYG-defined asset.** | medium-high for `hy_credit` | confirmed (known caveat, now measured at the target horizon) | On the 2007-2026 overlap the VWEHX/HYG vol ratio is **0.83 at 21 days** (0.80 in 2007-09), 0.93 at 63 days, 0.49 at 1 day. 21-day corr 0.92. About 22 years of history are therefore ~17% low in vol, with equity correlation understated. | Record the 21-day ratio in `segments.csv`. Give the VWEHX daily era its own tier flag so models never pool its level with HYG's. If a less smoothed HY total-return source becomes reachable, use it. |
| 4 | **Daily-accrual bond funds book a month of income on one day.** VWEHX, VFSTX and FBNDX, and VWESX from 1985, do not drop NAV on the ex-date, so `adjclose` adds ~0.5-1% on each month-end ex-date. The TIPS track fixed this for PRRIX; the credit track did not. | medium for daily-input models; low for the 21-day target | confirmed | `hy_credit` 1985-2007 **1-day vol +25%** (0.041 vs 0.033), AC1 0.15 vs true 0.24. `ig_credit` 1985-2002 1-day vol +8%. 21-day target ≤ +0.8%. `horizon_evidence.md` therefore reports r1 = 0.70 for VWEHX (true ≈ 0.56) and r1 = 1.02 for the 1985-2002 IG blend (true ≈ 0.95). | Spread each distribution over its accrual period, as `tips.py` does for PRRIX, for these four funds. Then re-assemble and regenerate `horizon_evidence.md`. |
| 5 | **`Panel.asof` can widen.** On a view, `view.asof(later_date)` keeps returns truncated, but `predictors_wide()` / `predictor_latest()` read the root cache in `_shared` and expose every predictor published up to `later_date`. `view._shared["root_predictors"]` also hands the full predictor history to any model. | medium (latent) | confirmed | No current model calls `asof` forward. HAR, statespace, ml and bridge only nest backwards, which is safe; a test now pins that. The leak is one careless call away. | In `Panel.asof`: `t = min(t, self.asof_date)` when the view is already cut. Do not let views reach root data via `_shared`; keep only derived caches there. A strict xfail test is in place. |

**Should models trained on the current panel be re-run after a fix?** Yes, but only for reasons 1-2 (and 4 for
models that read daily credit returns). Finding 1 means the history models train on for `ig_credit` is partly
defined by holdout data. That must be fixed before the holdout is scored, and the models refit on the corrected
panel; otherwise the final comparison is not clean. Findings 2 and 4 change `ust_int`/`ig_credit` targets in
2003-04 by up to ~6% in vol and ~0.1 in correlation, and change daily credit inputs by 8-25% in vol. Both are
cheap to fix in the same rebuild. Nothing else found changes a 12-month 21-day-horizon target by more than ~1%.
No predictor was found to embed future information beyond trivial amounts (§G).

---

## A. Returns against independent sources (ETF era)

**Method.** The Alpha Vantage quota (25 requests/day) was already exhausted, so the second source is
**markets.ft.com closes** for all 11 ETFs (since inception) plus **Nasdaq closes for 2016-11 to 2026-11**. For
dividends, Nasdaq's full distribution history is used for the Nasdaq-listed IEF and TLT. For the NYSE-Arca ETFs
no non-Yahoo dividend record responds, so Yahoo's dividend *events* were used (an independent check of
`adjclose` and of prices, but not of the dividend amounts). For bond ETFs and funds, every distribution was then
tested against the price itself. On each ex-date the fund's return was regressed on peers (CMT yield changes,
equities), fitted on non-ex days within ±250 trading days. A real distribution leaves a residual near 0; a
distribution the price never paid out leaves a residual equal to its size.

**Monthly total returns, panel vs independent (months with complete independent daily coverage):**

| ETF | asset | months | corr | vol ratio | mean diff %/yr | max abs diff | months > 0.5% | largest discrepancies |
|---|---|---|---|---|---|---|---|---|
| SPY | us_large | 404 | 0.9997 | 1.001 | +0.03 | 1.50% | 2 | 2001-04 +1.50, 2001-05 −1.39 (one bad close, see below) |
| IWM | us_small | 316 | 1.0000 | 1.000 | 0.00 | 0.15% | 0 | 2007-06 −0.15 |
| EFA | intl_dev | 301 | 1.0000 | 1.000 | 0.00 | 0.10% | 0 | 2007-03 +0.10 |
| EEM | em | 281 | 0.9999 | 0.999 | +0.03 | 0.82% | 2 | 2007-03 −0.82, 2007-02 +0.75 (one close differs, reverses) |
| IEF | ust_int | 291 | 0.9955 | 1.001 | +1.08* | 1.28% | 8 | 2003-12 +1.28, 2003-08 +0.87, 2009-11 +0.85*, 2008-12 +0.74* |
| TLT | ust_long | 291 | 0.9990 | 1.002 | +1.24* | 0.78% | 5 | all from missing Nasdaq dividends* |
| TIP | tips | 273 | 0.9998 | 1.002 | −0.03 | 0.57% | 1 | 2008-04 −0.57 (panel correctly omits Yahoo's duplicated 2008-04-03 distribution) |
| LQD | ig_credit | 291 | 1.0000 | 1.000 | 0.00 | 0.05% | 0 | — (2003-12 not caught: Yahoo dividends on both sides; caught by the price test) |
| HYG | hy_credit | 234 | 1.0000 | 1.000 | 0.00 | 0.03% | 0 | — |
| VNQ | reits | 264 | 1.0000 | 1.000 | 0.00 | 0.16% | 0 | 2006-10 / 2006-09 ±0.16 (dividend dated one day apart) |
| GLD | gold | 262 | 1.0000 | 1.000 | 0.00 | 0.00% | 0 | identical (same vendor; GLD never distributes) |

\* Nasdaq's dividend list for IEF/TLT has gaps (none in 2008-2010, few in 2005/2007). Where both records exist
they agree to 0.0005, except IEF 2003-09-02 and 2003-10-01. FT and Yahoo closes agree to 0.13% (IEF) and 0.33%
(TLT) on every day. The residual IEF/TLT gaps are therefore Nasdaq omissions, except for 2003 (A-1).

**A-1 (medium, confirmed): spurious IEF/LQD distributions in 2003.** Ex-date residual tests:

| ex-date | fund | Yahoo distribution | implied by price | Nasdaq | z | panel return that day |
|---|---|---|---|---|---|---|
| 2003-07-01 | IEF | 0.663 | 0.30 | — | 3.6 | +0.27% |
| 2003-08-01 | IEF | 0.717 | 0.40 | — | 3.3 | +0.41% |
| 2003-09-02 | IEF | 0.764 | 0.22 | 0.276 | — | −0.41% |
| 2003-10-01 | IEF | 0.731 | 0.41 | 0.274 | 3.5 | +0.44% |
| 2003-12-16 | IEF | 0.840 | 0.10 | — | 8.4 | **+1.20%** (TLT +0.33%, IEF price +0.20%) |
| 2003-12-16 | LQD | 1.26% of price | ~0 | n/a | 5.9 | **+1.50%** (LQD price +0.23%) |

Corrected (Nasdaq or price-implied) `ust_int` 2003 return = **+2.2%**, which matches IEF's published 2003 total
return as I remember it (memory). The panel shows +5.3%. Effect on targets: see executive summary. TLT, HYG,
TIP, EFA and EEM show no implausible distributions. One IEF distribution in 2009-11-02 (0.770) looks large, but
the price confirms it (implied 0.81), so it is genuine.

**A-2 (low, suspected): SPY 2001-04-30 close.** Yahoo 126.66 (+0.70%), FT 124.91 (−0.69%); CRSP Hi-30 shows
+0.05%. Both vendors look wrong, and the move reverses the next day. Effect on the 21-day target < 0.1%.

**A-3 (low, unresolved): EEM 2007-02-28.** Yahoo and FT closes differ by 0.8% (36.87 vs 36.58); the gap
reverses the next day. A third source was not available.

## B. Pre-ETF history against published figures

Downloaded: Damodaran `histretSP.xls` (2026-01 update). Memory-based comparisons are marked as such. Annual
returns cannot validate daily vol, but a reconstruction that misses calendar years by more than 1-2pp needs a
reason.

| comparison | source | n years | median abs diff | notable |
|---|---|---|---|---|
| us_large vs S&P 500 TR | Damodaran (download) | 98 | 0.56pp | 1928 −5.8, 1952 −4.1, 1955 −6.3 (CRSP Hi-30 vs the narrow 1950s S&P; composition, not error) |
| ust_int vs 10y T-bond | Damodaran (download) | 72 | 0.84pp | 2003 +4.9pp = finding A-1; other gaps (1981, 86, 87, 91, 2009, 13) are 7-10y ladder vs 10y duration |
| cash vs 3m T-bill | Damodaran (download) | 98 | 0.09pp | max 2.1pp (1930, different source) |
| gold vs year-end price | Damodaran (download) | 54 | 0.71pp | 1974 +7.3, 1979 +7.1 (last-fixing date in fast markets) |
| ust_long vs SBBI LT govt | memory | 19 (1931-76) | ~1.2pp | within ±3pp (1946 +4.0); 1982 46.4 vs 40.4 (20-30y ladder has more duration than SBBI's ~20y bond) |
| reits vs Nareit All Equity | memory | 30 | 0.4pp | **1987-1995 (FRESX era) off by 3-7pp in 8 of 9 years** (1990 −8.7 vs −15.4; 1993 12.5 vs 19.7). The active fund is not the index; monthly vol ratio ~1, so it matters for means and tracking, not vol |
| intl_dev vs MSCI EAFE | memory | 26 | 2.4pp | 1999 38.6 vs 27.0, 1995 4.7 vs 11.2 (AQR global ex-US incl. Canada, cap weights differ) |
| em vs MSCI EM | memory | 8 | 5.3pp | 1993 89 vs 75 (KF EM), 1998 −18 vs −25 (VEIEX) |
| commodities vs BCOM TR | memory | 5 | 0.0pp | — |
| tips vs TIP | memory | 3 | 0.0pp | — |

Verdict: no reconstruction misses calendar years in a way that points to a data error, except 2003 `ust_int`
(A-1). REIT 1987-95, intl 1986-2001 and EM pre-2003 are a different *definition* from the target index. That
is documented as composition, and it is worth knowing for means.

## C. Bad prints and outliers

**Method.** I first ran a robust-MAD z-scan over every asset. In the 1960s Treasury and NAV eras, tick
quantization makes MAD unusable, so a final scan flags days with |r| > 4 × trailing-63-day sd, a next-day
reversal of ≥ 60%, and no confirmation (|z| < 2.5) by two related assets on either day. Each hit was then
checked against peers: other funds in the raw cache, COMEX GC1 (FT, 1978+) for gold, CRSP market for SPY.
Separately, every fund distribution was scanned (A, and C-1 below).

**C-1 (medium for daily inputs, low for the target, confirmed): month-end income lumps in daily-accrual funds.**
The median share of each distribution *not* reflected in the NAV is: VWEHX 0.97-1.02 (every period),
VFSTX 0.99-1.05, FBNDX 0.98, VWESX −0.03 in 1980-84 but 0.88-1.14 from 1985. In other words, the NAV never
drops and `adjclose` books the whole month's income on one day. Effects, measured by re-spreading the income
over the accrual period:

| segment | 1-day vol panel → corrected | 21-day vol | AC1 | r1 = v1/v21 |
|---|---|---|---|---|
| hy_credit 1985-2007 (VWEHX) | 0.0414 → 0.0330 (**+25%**) | +0.8% | 0.15 → 0.24 | 0.70 → 0.56 |
| ig_credit 1985-2002 (blend) | 0.0554 → 0.0513 (+8%) | +0.3% | 0.08 → 0.10 | 1.02 → 0.95 |
| ig_credit 1980-84 (blend) | 0.0731 → 0.0718 (+2%) | +0.1% | — | — |

So the smoothing in these eras is *worse* than `horizon_evidence.md` reports, and the apparently unsmoothed
1985-2002 IG blend (r1 = 1.02) is an artifact.

**C-2 (low, confirmed): reversing bad prints that remain in the panel.**

| asset | date | move | evidence | target effect | 1-day vol effect that year |
|---|---|---|---|---|---|
| em (VEIEX) | 2001-07-16/17 | +5.56% / −5.85% | TEDMX −0.60/−0.80, FOSFX, SCINX flat | < 0.1% | +6.7% |
| ig_credit (FBNDX × 0.41) | 1982-10-06, 1983-08-30, 1983-11-21, **1983-11-28/29** (+3.57/−3.44 raw), **1984-02-02/03** (−2.67/+2.89), 1984-11-14/15, 1984-12-06 | | VWESX and Treasuries flat | ≤ 0.7% | small |
| hy_credit (VWEHX) | 1987-01-29/30 (−1.17/+1.17), 1991-08-19/20 (−1.30/+1.02) | | FAGIX, PRHYX, FHIIX, AHITX ≈ 0 | ≤ 0.7% | +5.5% (1987) |
| tips (PRRIX) | 2000-12-12/13 (+0.84/−0.99) | | ACITX, VIPSX 0.00 | 0.6% | +15% (2000) |
| gold (Frankfurt) | 1996-12-06 (+1.59/−1.48) | | COMEX +0.16/−0.59 | 0.3% | +5.6% |

Bad prints that reverse almost cancel at the 21-day horizon (only the windows that split the pair are hit).
They matter for models that use daily inputs.

**Checked and genuine:** gold 1980-01-23 −16.8% and the other January 1980 days (each matches the COMEX move of
the previous US session; the 13:00 CET fixing comes before the US open), gold 1985-03-20 +8.1% (COMEX +11.8% on
03-19), gold 1979-10-02, VGSIX 1999-12-17 +4.1% (peers +3.9 to +4.4%), REIT 2000-06-30 −2.8% and 2004-04-12
−5.0% (all peers), VGSIX 1998-09-21/22 (peers), VWEHX 2001-09-17 −4.9% (peers −2.9 to −3.5%: real but steeper),
commodities 2018-11-14 (natural-gas spike), us_small 1980-03-27 (Silver Thursday; CRSP, kept). The assembler's
REIT 1993-07-29 removal and the ust_long 1973-01-04 drop are both implemented correctly: the day is NaN, and
for REITs the next return spans the gap.

**Zero runs and duplicates.** VWEHX has 1,181 days in zero-runs of three or more days (stale NAV, documented);
HYG has one three-day run (2026-01-13..15), which Nasdaq confirms. Identical consecutive returns occur only
where expected: CRSP 1bp rounding pre-1950 and flat-yield carry days in 1960s Treasuries. No copied blocks across
assets or eras.

## D. Date alignment

I computed lag −2..+2 cross-correlations of daily log returns for 15 related pairs, separately for every era in
which neither source changes (49 pair-eras; table in `audit/leadlag.csv`). Every same-market pair peaks at
lag 0: us_large/us_small, ust_int/ust_long, ig/ust, tips/ust, us_large/reits and the ETF-era pairs. The lag ±1
peaks found all have a known cause:

- intl_dev AQR 1986-2001 vs US: L+1 0.39 > L0 0.21. Foreign closes; documented.
- gold Frankfurt 1971-98 vs BCOM: L+1 0.30 vs L0 0.03; vs SPY L+1 −0.10. The 13:00 CET fixing reflects the
  previous US session, so the series is effectively one session late. It is correctly dated, and the effect
  on 21-day correlations is about 1/21. Low.
- hy VWEHX vs SPY 1993-2007: L+1 0.16 ≥ L0 0.13. VWEHX vs four peer HY funds peaks at L0 (0.65-0.85), so this is
  matrix-price staleness against equities, not misdating.
- PRRIX 1997-2001 vs CMT: L+1 0.12-0.30. ACITX shows the same and PRRIX vs VIPSX peaks at L0, so it is staleness.

No source segment is shifted by a day.

## E. Splices

`horizon_evidence.md` already shows the 1-day convention jumps. Here the question is what survives at
21 days. Two-to-three-year windows either side of each splice, compared with a continuous reference, are
dominated by genuine regime changes (1993, 2007-08, 1980-82), so I relied instead on the **overlap where both
sources exist**, measured at 21 and 63 days:

| splice | 21d vol ratio old/new | 21d corr | 63d vol ratio | verdict |
|---|---|---|---|---|
| French Hi-30 → SPY (1993-2026 overlap) | 1.005 | 0.997 | 1.013 | clean |
| French Qnt2+3 → IWM | 1.007 | 0.994 | 1.014 | clean |
| VEIEX → EEM | 0.956 | 0.984 | 1.001 | ~4% low at 21 days |
| **VWEHX → HYG** | **0.831** (0.80 in 2007-09) | 0.917 | 0.926 | **level shift persists at 21 days (finding 3)** |
| VGSIX → VNQ | 1.002 | 0.999 | 1.001 | clean |
| PRRIX → TIP | 1.067 | 0.960 | 1.147 | active fund ran 7-15% more risk |
| VWESX/VFSTX blend → LQD, full overlap | 0.981 | 0.897 | 1.009 | by construction (finding 1) |
| same, 2002-2014 only | 0.936 | 0.833 | 0.949 | the calibration leans on post-2014 data |

The remaining splices are monthly (C/D → B/A). On them the monthly series *is* the target proxy, and the
documented damping of month-average (tier D) eras (~15%) is the main issue; I did not find anything beyond it.
`us_small` 1926-2000 (r63 = 1.11) and VWEHX (r63 = 1.16) keep building variance past 21 days. The 21-day target
therefore still understates those eras by about 5-15% relative to a 63-day measure (already listed in
`horizon_evidence.md`; confirmed).

## F. Cash

- Annual cash by decade (%): 1930s 0.55, 1940s 0.41, 1950s 1.87, 1970s 6.33, 1980s 8.92, 2010s 0.52. Peaks:
  1981 14.7, 2024 5.25. All consistent with known T-bill history; Damodaran T-bill median gap 0.09pp. Twelve
  slightly negative months in 1933-41 come from the Ibbotson/French source (bills yielded below zero then).
- The splice to the H.15 4-week bill is continuous: Sep 2026 3.72% → Oct 3.76% → Nov 3.82% (annualized). The
  last three days carry the last published yield.
- Daily cash is the monthly RF spread evenly over trading days: no intra-month information, which is fine.
  Daily compounds to monthly within 3e-15.
- Excess returns are never contaminated: no NaN where a return exists (daily and monthly), no cash value above
  0.06%/day.

## G. Predictors and look-ahead

All 62 series were classified. Derived series were rebuilt from their inputs: all 8 reproduce exactly (≤ 1e-13),
and the ETF durations are unchanged when the inputs are truncated at 2008-10-15, 2014-12-31 and 2020-03-20
(tests in place).

| series / group | classification | evidence and size |
|---|---|---|
| `ust_etf_duration_est_int/_long` | clean (trailing) | Rolling 252-day regression on 5-day changes; reproduced to 1e-13 and truncation-invariant. Inputs (GSW ladder yields) carry the timing issue below |
| `ust_ladder_yield_int/_long`, `ust_ladder_duration_*` (GSW era) | **optimistic timing** (low) | Stamped available on the observation day, but GSW is published with a lag of ≥ 5 business days (file ends 2026-11-06, downloaded 11-14). The TIPS track uses +10 business days for the companion TIPS file. Effect on a 12-month vol forecast: negligible (≈ one week of yield drift) |
| GSW nominal and TIPS curves "refit" | clean in information, **revision risk** unverifiable | Each day is fitted on that day's cross-section only, with no time smoothing; the rules (security filters, NS vs NSS before 1980) were set retrospectively in 2006/2008. Only one vintage is cached, so revisions cannot be measured |
| `ust_cmt_yield_*` (H.15) | optimistic timing (very low) | Stamped same day; Treasury posts CMT yields after the close on the same day, and H.15 republishes them next morning. Under one day |
| `ust_cmt_yield_20y` | **data break** (low) | +74bp jump on 1973-01-04 (re-basing; the 10y moved −2bp) and the matching `ust_ladder_duration_long` drop. A level break in any spread built on it. Gap 1987-01..1993-09 |
| `credit_gz_spread_m` | clean in construction, **revision risk** | A plain average of bond spreads, with no estimated loadings; the EBP is correctly excluded. Lag = end of the following month (conservative: the Oct-2026 value was published before 11-14). The Fed file is the latest vintage, and the sample construction has been revised over time; not verifiable here |
| Moody's (monthly, daily) | clean | Month averages, available next business day |
| `macro_cpi_level`, `_infl_yoy`, `_infl_vol_36m`, `_real_short_rate` | clean from 1947; **not real-time before ~1940** (low) | Lag M+1 (1947+) / M+3. CPI NSA is unrevised. But monthly CPI before 1940 and the pre-1913 Warren-Pearson wholesale index were compiled retrospectively, so M+3 is optimistic for those decades. Pre-1940 CPI is also quantized (26-57% zero months, from 0.1-point rounding), which distorts `infl_vol_36m` |
| `macro_unrate`, `_unrate_chg_12m`, `_sahm_*` | revision risk (low) | Lag M+1, and the Sahm gap uses the latest published input (verified). Seasonal factors are revised (≤ 0.1-0.2pp). The 2025 shutdown override is correct |
| `macro_ip_yoy` | **revision risk** (medium for models that use it) | Latest vintage only; IP revisions to 12-month growth can reach several pp around recessions. ALFRED is not reachable, so the size is unmeasured |
| `macro_short_rate`, `_long_yield`, `_policy_rate*`, `_term_spread`, `_usd_broad_yoy` | clean | Month averages of market rates, available at month-end. The broad dollar's trade weights are revised (tiny) |
| `iv_vix_vxo_spliced` | **definite look-ahead** (trivial) | 1986-89 = VXO × k with k = 1.0074 estimated on 1990-94. Using only the first year of overlap gives k = 1.0033: a 0.4% level effect |
| implied-vol cleaning | definite look-ahead (trivial) | The bad-print filter uses the next day's value, and the stale-run filter uses the full run length, so the backtest sees cleaner prints than a real-time user. Affects a handful of days on thin series |
| `iv_vix` pre-2003, `iv_vix1y` pre-2018, `iv_vix6m` pre-2014, `iv_vxefa` pre-2013 | clean in information; not disseminated in real time | Retrospective calculations from contemporaneous option quotes; no future prices enter |
| other `iv_*`, term slopes | clean | Same-day closes; slopes reproduced exactly |
| BCOM 1991-98 (asset, not predictor) | backfill | Index-construction hindsight (weights designed in 1998); affects returns, not predictor timing. Not testable here |

**Full-sample normalization**: none among the predictors. The one full-sample estimation that does reach the
data is in the *returns*: `ig_credit` (finding 1). Second-order: the French portfolios were chosen (Hi-30,
Qnt 2+3) for tracking SPY/IWM over 1993/2000-2026. That is selection among a few candidates whose vol ratios
differ by ≤ 4%. Low.

## H. Assembly and the `asof` contract

- **Folding:** for every asset, the panel compounds to the track's daily file over the full span (≤ 2e-15)
  and in every calendar year (≤ 1e-16). Week-level differences appear only where a holiday return was folded
  across a weekend (intl 52 weeks, gold 22, Treasuries 8, commodities 6), as designed. No returns are lost.
- **Monthly = compounded daily** in every full daily month (≤ 1.3e-15). No partial final month (last monthly
  2026-10-31, last daily 2026-11-13).
- **Tiers:** present exactly where monthly data are present, letters A-D only, and consistent with
  `segments.csv`. Each splice month takes the worse tier, as intended.
- **`asof`:** the adversarial test covers 25 edge dates (first day, 1933/2001 closures, Saturday era, splice
  eve and day, weekend month-ends, the dev cutoff, shutdown-delayed releases) plus 15 random dates. For each it
  checks that returns, cash and tiers end ≤ t and none are dropped; that predictors have available_date ≤ t and
  obs_date ≤ t; that every wide cell is backed by a published row; and that `predictor_latest` equals the last
  published value. It also checks that a view never mutates the root and that nested `asof` to an earlier date
  equals a direct `asof`. All pass.
- **Finding 5 (medium, latent, confirmed):** `P.asof("2008-10-15").asof("2026-06-30").predictors_wide()`
  returns predictors through 2026-06-30, and `predictor_latest()["iv_vix"]` gives 0.16 instead of 0.69.
  `_shared["root_predictors"]` is also reachable from every view. Fix: clamp the cutoff in `asof`.

## I. Other observations

- `horizon_evidence.md` r1 values for VWEHX and the 1985-2002 IG blend are distorted by C-1. Regenerate after
  the fix.
- The `implied.md` text says `iv_move` is absent, but the catalog has it from 2002-11. The report is stale
  (documentation only).
- The REIT FRESX era is an active fund whose calendar-year returns miss Nareit by up to 7pp. It is tier B and
  documented, but the monthly Nareit index (tier C) is arguably a better 1986-96 *definition* for monthly use.

## Checked and found clean

ETF-era returns for SPY, IWM, EFA, EEM, TLT, TIP, LQD (apart from 2003-12-16), HYG, VNQ and GLD against FT and
Nasdaq prices (monthly corr ≥ 0.9997, typical |diff| < 0.05%/month). The TIP 2008 duplicated distribution is
correctly removed. IEF/TLT Nasdaq dividends match Yahoo's to 0.0005 wherever both exist (except IEF 2003).
Calendar-year us_large, gold, cash, commodities and TIPS returns agree with published or remembered figures.
No source is misdated. The assembler-flagged days are handled correctly and the gold 1980 moves are genuine.
Cash levels and the cash splice are sound. Excess returns are uncontaminated. Every derived predictor is
trailing. Macro publication lags are conservative from 1947. Folding, monthly compounding, tiers and the
final month are correct. `asof` works for every normal (backward) use.

## Not checked, and why

- **Alpha Vantage**: daily quota exhausted before the audit began. FT and Nasdaq were used instead.
- **Distribution amounts for NYSE-Arca ETFs** (SPY, IWM, EFA, EEM, TIP, LQD, HYG, VNQ): Nasdaq serves dividend
  history only for Nasdaq listings and iShares/SSGA pages were not attempted. The ex-date price test only has
  power for bond ETFs; for equity ETFs a dividend error below ~0.3% would go undetected.
- **GSW curve revisions, IP and unemployment vintages, GZ vintages**: only one vintage is available, and
  ALFRED/FRED are unreachable.
- **Pre-1990 CRSP (French), Ibbotson LT corporate, AQR commodities, BCOM 1991-98, Nareit 1972-86, KF EAFE
  1975-85, Shiller pre-1926**: no second source was downloaded. Calendar-year checks against my memory only
  (marked as such above).
- **LBMA gold daily**: blocked. COMEX GC1 from 1978 was used instead.
- The ETF dividend scan for VEIEX/VGSIX/FRESX capital-gain omissions found none, but has little power for
  equity funds below ~1%.
