# `reits` — US equity REITs (VNQ-like)

Built by `python -m volmodel.data.sources.reits` (raw cache in `data/raw/reits/`,
`--refresh` re-downloads). All returns are **simple total returns including
dividends**, in decimals, not excess. Data as of 2026-11-14.

| file | span | rows |
|---|---|---|
| `data/processed/assets/reits_daily.parquet` | 1986-11-17 .. 2026-11-13 | 10,074 |
| `data/processed/assets/reits_monthly.parquet` | 1972-01-31 .. 2026-10-31 | 658 |

The monthly file stops at the last *complete* month (October 2026); November
2026 is only in the daily file.

## Segments

| freq | start | end | source | tier | why |
|---|---|---|---|---|---|
| M | 1972-01 | 1986-11 | FTSE Nareit **All Equity REITs** total return (Nareit monthly index spreadsheet) | C | only REIT index this far back; mortgage REITs excluded |
| D | 1986-11-17 | 1996-05-31 | Fidelity Real Estate Investment Portfolio (FRESX), Yahoo `adjclose` | **B** | oldest daily REIT total return available; actively managed, so tiered B (see below) |
| D | 1996-06-03 | 2004-09-30 | Vanguard REIT Index Inv (VGSIX), Yahoo `adjclose` | B | index fund (MSCI US REIT Index); open-end fund NAV, so tier B |
| D | 2004-10-01 | 2026-11-13 | Vanguard Real Estate ETF (VNQ), Yahoo close + validated distributions | A | the target asset; every Yahoo distribution passed the ex-date price test against IYR/ICF/RWR |

From 1986-12 on, the monthly file is the compounded daily series (the
contract requires monthly == compounded daily where daily exists). November
1986 is a partial daily month, so the Nareit value is used for it.

**Tiers.** Project-wide, tier A is reserved for the daily price of an
exchange-traded fund (market close); open-end mutual-fund NAVs, indices and
reconstructions are tier B. VGSIX (an index fund, but priced at NAV) is
therefore tier B, and the VGSIX -> VNQ splice in 2004-10 is a visible tier
change (B -> A). VNQ's total return is its close plus Yahoo's distributions,
each tested on its ex-date against the total returns of IYR, ICF and RWR
(`volmodel.data.sources.distributions.validate_distributions`); none was
rejected.

**FRESX is tier B** on two counts: it is an open-end NAV, and an
*active* manager's REIT portfolio. In the era where we use it, it tracks the
Nareit index much less closely than the index fund does later: monthly
tracking error is 3.9%/yr, against 1.1% for VGSIX. Its vol ratio is close to 1
(0.985) and its correlation is 0.948. That is good enough for a volatility
proxy, but it is not "the asset".

## Sources tried

| source | result |
|---|---|
| Nareit `https://www.reit.com/sites/default/files/returns/MonthlyHistoricalReturns.xls` | **works with a non-browser User-Agent.** With a browser UA, reit.com returns an F5 JavaScript "Client Challenge" page instead of the file. The loader checks the OLE2 magic bytes and fails loudly if it gets HTML. Contents: Dec 1971 – Oct 2026, with All REITs, Composite, Real Estate 50, All Equity REITs, Equity REITs and Mortgage REITs (total, price and income). |
| Yahoo VNQ (2004-09-29), VGSIX (1996-05-13), FRESX (1986-11-14), CSRSX (1991-07-01), DFREX (1993-01-05), IYR, ICF, RWR | all serve full daily `adjclose` histories. No real-estate fund that Yahoo serves starts before FRESX. |
| Yahoo indices `^RMZ`, `^DJR`, `^RMS` | not found. `^DWRTF` returns only one day. |
| Wilshire REIT / DJ REIT daily index (e.g. FRED WILLREITIND) | FRED times out here. DBnomics has no REIT or Wilshire series. **Not obtained.** A daily REIT index for 1978–1986 would be the one real gain still possible. |
| Ken French 49 industries, `RlEst` | downloaded for diagnostics only. Against Nareit, 1972–2026: corr 0.73, vol ratio **1.55**, TE 18%/yr. It mostly holds developers and operators; French puts REITs (SIC 6798) in `Fin`. **Rejected.** |
| web.archive.org | connection reset by the proxy (not needed in the end). |

## Overlap diagnostics

Vol ratio = vol(a)/vol(b). TE = annualized standard deviation of a − b. The
monthly rows compound fund daily returns to calendar months.

### Splices, in the era each segment is used

| a ~ b | freq | window | n | vol a | vol b | vol ratio | corr | TE | mean diff/yr |
|---|---|---|---|---|---|---|---|---|---|
| FRESX ~ Nareit AER | M | 1986-12..1991-07 | 56 | 12.3% | 13.4% | 0.923 | 0.958 | 3.9% | +0.9% |
| FRESX ~ Nareit AER | M | 1991-08..1996-05 | 58 | 11.4% | 10.6% | 1.078 | 0.941 | 3.9% | −0.9% |
| **FRESX ~ Nareit AER** | M | 1986-12..1996-05 | 114 | 11.8% | 12.0% | **0.985** | **0.948** | **3.9%** | 0.0% |
| **VGSIX ~ Nareit AER** | M | 1996-06..2004-09 | 100 | 13.9% | 13.8% | **1.009** | **0.997** | **1.1%** | −0.2% |
| FRESX ~ VGSIX | D | 1996-06..2004-09 | 2097 | 12.0% | 12.2% | 0.981 | 0.971 | 2.9% | +0.8% |
| **VGSIX ~ VNQ** | D | 2004-09..2026-11 | 5567 | 28.8% | 28.2% | **1.022** | **0.994** | **3.2%** | +0.1% |
| VGSIX ~ VNQ | M | 2004-10..2026-10 | 265 | 21.5% | 21.4% | 1.002 | 0.999 | 0.75% | −0.1% |
| **VNQ ~ Nareit AER** | M | 2004-10..2026-10 | 265 | 21.4% | 21.1% | **1.014** | **0.998** | **1.5%** | −0.3% |

The mean differences are close to zero (0.0% for FRESX, −0.2 to −0.3%/yr for
the index funds, which is about their expense ratio). This confirms that
Yahoo's `adjclose` includes the distributions. A price-only series would lag
by the ~5–7%/yr REIT yield.

### Alternatives considered for the pre-1996 daily era (vs Nareit AER, monthly)

| fund | window | corr | vol ratio | TE |
|---|---|---|---|---|
| FRESX (used) | 1986-12..1996-05 | 0.948 | 0.985 | 3.9% |
| CSRSX (Cohen & Steers) | 1991-08..1996-05 | 0.939 | 1.129 | 4.2% |
| DFREX (DFA) | 1993-02..1996-05 | 0.831 | 0.933 | 6.2% |

FRESX starts earliest and also tracks best. Before 1996, daily correlations
*between* these funds were only 0.68–0.78 (1993–96). In that era, fund NAVs
carry a lot of idiosyncratic, stale-price noise at the daily horizon.

### Other checks (full overlap, vs Nareit AER, monthly)

| series | corr | vol ratio | TE |
|---|---|---|---|
| Nareit Equity REITs (excludes timber and infrastructure) | 0.998 | 1.008 | 1.2% |
| Nareit All REITs (includes mortgage REITs) | 0.945 | 1.030 | 5.8% |
| Nareit Mortgage REITs | 0.590 | 1.264 | 18.0% |
| IYR / ICF / RWR | 0.993–0.995 | 0.98–1.04 | 2.0–2.6% |
| French `RlEst` | 0.734 | 1.546 | 18.1% |

### Daily compounding vs the monthly index (item c)

| era | mean \|diff\| per month | max \|diff\| | corr |
|---|---|---|---|
| FRESX 1986-12..1996-05 | 0.87% | 3.5% (1991-12) | 0.948 |
| VGSIX 1996-06..2004-09 | 0.24% | 0.96% (1997-01) | 0.997 |
| VNQ 2004-10..2026-10 | 0.32% | 2.1% (2026-02) | 0.998 |

The VNQ and VGSIX differences are index-definition differences. MSCI US IMI
Real Estate 25/50 includes non-REIT real estate companies and applies capping;
MSCI US REIT and Nareit AER do not. They are not data errors.

## Annualized volatility by decade (log returns)

| decade | months | vol (monthly ×√12) | days | vol (daily ×√252) | AC1 monthly | AC1 daily | variance ratio monthly/daily |
|---|---|---|---|---|---|---|---|
| 1970s (1972–79, Nareit) | 96 | 16.5% | – | – | 0.07 | – | – |
| 1980s (Nareit to 1986-11, then FRESX) | 120 | 12.2% | 789 | 10.2% | 0.18 | 0.12 | 1.36 |
| 1990s | 120 | 12.5% | 2527 | 8.4% | 0.09 | 0.32 | **2.19** |
| 2000s | 120 | 25.8% | 2515 | 34.0% | 0.14 | −0.19 | **0.58** |
| 2010s | 120 | 15.2% | 2516 | 17.8% | −0.21 | −0.04 | 0.72 |
| 2020s (to 2026-10) | 82 | 20.0% | 1727 | 23.6% | −0.14 | −0.09 | 0.72 |

The variance ratio column uses only months covered by the daily data. Rolling
252-day realized vol ranges from **4.6% (1989-06)** to **91% (2009-07)**.

### What a model should and should not infer from the 1970s–80s

- **Should:** REIT returns in that era were equity-like in direction and
  sensitive to rates and credit. 1973–74 was −34% cumulative, with −14% in
  November 1973. Treat the regime as one of moderate vol.
- **Should not:** read 12–16% vol as the "normal" level of today's sector. The
  pre-1990s REIT market was tiny: a few dozen small, thinly traded equity
  trusts with a total market cap of a few billion dollars. The modern
  REIT era began with the 1992–94 IPO boom and index inclusion in 2001. Low
  measured vol then reflects illiquidity and stale prices (smoothing) as much as
  low risk. That shows up as positive autocorrelation: AC1 of monthly returns is
  0.18 in the 1980s, and AC1 of daily returns is 0.12–0.32 until 2003.
- Since 2004 the sector has traded like a liquid large equity sector: daily vol
  is 18–34% by decade, and daily AC1 is negative.
- **Daily-based RV understates the variance of longer-horizon returns before
  ~2003.** In the 1990s, monthly variance was about 2× what daily data imply,
  because daily NAVs were stale. Since 2004 the reverse holds (ratio ~0.6–0.7):
  daily returns overshoot and then reverse, especially in 2008–09. Any model
  trained on daily RV across the whole span should account for this regime
  shift, for example with era-specific variance ratios in `volmodel.bridge` or
  by down-weighting pre-2004 daily data. A model that takes level from the
  1972–2003 data will forecast too little REIT vol for the modern era.

## Known episodes (item d)

| episode | value | check |
|---|---|---|
| 1973–74 | −33.6% (1973 −15.5%, 1974 −21.4%) | Nareit, plausible |
| Oct 1987 | month −13.5%; 1987-10-19 −9.2%, 10-20 −3.7%, 10-21 +7.8% | REITs fell less than the S&P (−20%) on the crash day: plausible |
| 1998 | −16.3% | Nareit Equity −17.5%: OK |
| 2008 | −37.0%; peak-to-trough 2007-01 → 2009-02: −68.3% | OK |
| Mar 2009 – Dec 2009 | +98.6% | OK |
| worst days | 2008-12-01 −19.5%, 2020-03-16 −17.7%, 2008-12-11 −14.3% | OK |
| Mar 2020 | −19.4% (Feb+Mar −25.1%) | OK |
| 2022 | −26.3% | VNQ −26.2%: OK |

## Caveats and unverified points

- Yahoo `adjclose` for mutual funds in 1986–1996 can only be checked at the
  monthly frequency against Nareit (corr 0.948). Every used segment was scanned
  for one-day spike-reversals (|r| > 2%, reversed the next day to within 25%),
  and each hit was compared with peer funds on the same day. One bad print was
  found and removed: **FRESX 1993-07-29**, close 13.29 between 13.90 and 13.91,
  with no distribution, while CSRSX was +0.4% and DFREX +1.0% that day. That
  price is dropped (`BAD_PRICES`), so 1993-07-30 carries the two-day return.
  Nothing is interpolated. Left in, the print would have added ~85% to 1993's
  realized variance. Every other hit, all of them in 2007–2025, was matched by
  all peers. Before 1991 there are no peer funds, but the scan found no
  candidates then. Remaining daily errors, such as a misdated distribution,
  cannot be ruled out. The largest daily FRESX moves before 1996 fall in
  October 1987 and look genuine. About 12% of FRESX days
  in 1986–1995 have a zero return. That is consistent with a $10 NAV quoted to
  the cent and ~0.5% daily vol, but it adds to the stale-price smoothing.
- The 1992 and 1996 daily data have 254 rows. The extra rows may be fund NAVs
  struck on market holidays; this was not checked against an NYSE calendar.
- Asset definitions shift across the splices. Nareit AER includes timber and
  infrastructure REITs. MSCI US REIT (VGSIX) excludes some specialized REITs.
  VNQ's index (MSCI US IMI RE 25/50 since 2018) also includes non-REIT real
  estate companies. Overlap TEs are 1–1.5%/yr, which is immaterial for vol.
- The Nareit spreadsheet is revised and overwritten in place each month. The
  cached copy is the October 2026 vintage.
