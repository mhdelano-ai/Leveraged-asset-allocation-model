# intl_equity — international developed (`intl_dev`) and emerging (`em`) equity

Built by `python -m assetforecast.data.sources.intl_equity` on 2026-11-14. USD, unhedged, SIMPLE TOTAL returns in decimals (not excess; cash is another track).

## Result

* **intl_dev** — daily 1986-01-02 .. 2026-11-13 (10412 obs); monthly 1975-01-31 .. 2026-10-31 (622 months; daily era compounded; current incomplete month excluded).
* **em** — daily 1994-06-01 .. 2026-11-13 (8169 obs); monthly 1989-07-31 .. 2026-10-31 (448 months; daily era compounded; current incomplete month excluded).

## Sources

Used:
* **Ken French data library** — `F-F_International_Indices.zip` (`Ind_all.Dat`: EAFE-country-weighted, value-weighted within country, USD, with dividends, monthly 1975-01..2025-12; MSCI raw data 1975-2006, Bloomberg after) and `Emerging_5_Factors_CSV.zip` (EM Mkt-RF + RF, monthly 1989-07..).
* **AQR** `Betting-Against-Beta-Equity-Factors-Daily.xlsx` — sheet MKT column `Global Ex USA` (excess return) + sheet RF. Daily from 1984, but until 1985-10 it is Canada only and until 1985-12 Canada + Australia; Europe and Japan enter 1986-01, so it is used from 1986-01. Runs to 2026-08.
* **Yahoo chart API**: EFA and EEM as close + distributions, every distribution validated against the ex-date price move relative to peer funds (`distributions.validate_distributions`: EFA 47 tested, 0 changed; EEM 47 tested, 0 changed); VEIEX adjclose (open-end fund NAV, tier B), bad print(s) dropped: ['2001-07-16'].

Diagnostics only (not in the files):
* **MSCI** public graph endpoint `app2.msci.com/.../getLevelDataForGraph` — works, but refuses start dates before 1997-01-01 and serves gross/net TOTAL return levels only from 2000-12-29 (price index from 1997-01). EAFE 990300 and EM 891800 GRTR daily and month-end, to 2026-11-13. Used as the reference index for every overlap test. MSCI EAFE from 1970 / EM from 1988 is NOT reachable here.
* Ken French `Developed_ex_US_3_Factors(_Daily)` (from 1990-07) — tracks MSCI EAFE slightly better monthly than AQR but worse daily (its daily FX timing makes it more synchronous with the US); AQR also starts 4.5 years earlier. Not used.
* Yahoo: VTMGX, VEA, VGTSX, VWO, TEDMX and long-lived active international funds SCINX (1980), PRITX (1980), VWIGX (1981), VTRIX (1983), FOSFX (1984), TEMFX (1986). See *Rejected candidates*.
* `^GSPC` (Yahoo, price) as the US-close reference for correlation diagnostics.

## Segments

| asset | start | end | freq | tier | source | notes |
|---|---|---|---|---|---|---|
| intl_dev | 1975-01-31 | 1985-12-31 | M | C | french_intl_indices_eafe_weighted | Ken French F-F_International_Indices.zip, Ind_all.Dat, first block (value-weight dollar returns, all firms), Mkt column. Countries at EAFE weights; MSCI raw data 1975-2006, Bloomberg after. Excludes Canada. |
| intl_dev | 1986-01-02 | 2001-08-31 | D | B | aqr_mkt_global_ex_usa_plus_rf | AQR Betting-Against-Beta daily workbook, MKT sheet 'Global Ex USA' (excess) + RF sheet. Cap-weighted developed ex-US incl. Canada (EFA excludes Canada). Index convention: foreign closes. Weekdays with |r|<0.001% (all closed) dropped. Overlap vs KF EAFE-wt monthly 1986-2025: corr=0.991, vol_ratio=1.007, te=0.023. |
| intl_dev | 2001-09-04 | 2026-11-13 | D | A | yahoo_EFA | iShares MSCI EAFE ETF, Yahoo close + distributions, each validated by the ex-date price test against VEA/VTMGX/S&P 500 (all passed). US-close prices. Overlap vs AQR GxUS: monthly corr=0.976, vol_ratio=1.028, te=0.036; daily corr_1d=0.689, corr_3d=0.886, vol_ratio_1d=1.330, vol_ratio_3d=1.141 (async closes). |
| em | 1989-07-31 | 1994-05-31 | M | C | french_emerging_mkt_plus_rf | Ken French Emerging_5_Factors_CSV.zip, Mkt-RF + RF (RF = US 1m T-bill), Bloomberg-based value-weighted EM market. |
| em | 1994-06-01 | 2003-04-30 | D | B | yahoo_VEIEX | Vanguard Emerging Markets Stock Index Inv (tracked MSCI Select EM Free in this era), Yahoo adjclose. Open-end NAV (tier B): foreign closes, with growing fair-value pricing after 2001. Bad NAV print 2001-07-16 (+5.6% then -5.9%, EM peers flat) dropped: the 07-17 return spans it. Overlap vs KF EM monthly 1994-06..2000: corr=0.954, vol_ratio=1.120, te=0.080; full overlap corr=0.973, vol_ratio=1.042, te=0.051. |
| em | 2003-05-01 | 2026-11-13 | D | A | yahoo_EEM | iShares MSCI Emerging Markets ETF, Yahoo close + distributions, each validated by the ex-date price test against VWO/IEMG/S&P 500 (all passed). US-close prices. Overlap vs VEIEX: monthly corr=0.979, vol_ratio=1.044, te=0.043; daily corr_1d=0.937, corr_3d=0.960, vol_ratio_1d=1.293, vol_ratio_3d=1.157. |

Splices are on calendar-month boundaries, so every month in the monthly files is single-source and monthly == compounded daily exactly. Price conventions by segment:

| segment source | price-time convention |
|---|---|
| kf_intl_index | monthly index (foreign closes; month-end timing) |
| aqr_gxus | daily index, foreign local closes converted to USD |
| EFA | US-listed ETF, US close (16:00 ET) |
| kf_em | monthly index (foreign closes) |
| VEIEX | open-end fund NAV: foreign closes, partially fair-valued to US close from ~2001 |
| EEM | US-listed ETF, US close (16:00 ET) |

## Splice / overlap diagnostics

Monthly log returns on the full overlap of the two sources (`vol_ratio` = new / ref, `te` = annualized tracking error, `mean_diff_ann` = new − ref):

| index | start | end | n | corr | vol_ref | vol_new | vol_ratio | te | mean_diff_ann |
|---|---|---|---|---|---|---|---|---|---|
| intl_dev: KF EAFE-wt (C) -> AQR GxUS (B), monthly | 1986-02-28 | 2025-12-31 | 479 | 0.991 | 0.169 | 0.170 | 1.007 | 0.023 | -0.005 |
| intl_dev: KF EAFE-wt -> AQR, monthly, 1986-1995 only | 1986-02-28 | 1995-12-31 | 119 | 0.986 | 0.189 | 0.196 | 1.036 | 0.033 | -0.011 |
| intl_dev: AQR GxUS (B) -> EFA (A), monthly | 2001-09-30 | 2026-07-31 | 299 | 0.976 | 0.164 | 0.168 | 1.028 | 0.036 | -0.008 |
| em: KF EM (C) -> VEIEX (B), monthly | 1994-06-30 | 2026-09-30 | 388 | 0.973 | 0.209 | 0.218 | 1.042 | 0.051 | -0.007 |
| em: KF EM -> VEIEX, monthly, 1994-06..2000 | 1994-06-30 | 2000-12-31 | 79 | 0.954 | 0.233 | 0.261 | 1.120 | 0.080 | 0.024 |
| em: VEIEX (B) -> EEM (A), monthly | 2003-05-31 | 2026-10-31 | 282 | 0.979 | 0.201 | 0.209 | 1.044 | 0.043 | 0.001 |

Daily log returns on the overlap. `corr_3d`/`vol_ratio_3d` use overlapping 3-day sums (the ARCHITECTURE correlation convention):

| index | start | end | n | corr_1d | corr_3d | vol_ref_1d | vol_new_1d | vol_ratio_1d | vol_ratio_3d | te_3d |
|---|---|---|---|---|---|---|---|---|---|---|
| intl_dev: AQR GxUS -> EFA (2001-09..) | 2001-09-04 | 2026-08-31 | 6278 | 0.689 | 0.886 | 0.156 | 0.208 | 1.330 | 1.141 | 0.090 |
| intl_dev: AQR GxUS -> EFA, 2001-09..2003-12 | 2001-09-04 | 2003-12-31 | 583 | 0.772 | 0.901 | 0.150 | 0.209 | 1.397 | 1.271 | 0.093 |
| em: VEIEX -> EEM (2003-05..) | 2003-05-01 | 2026-11-13 | 5924 | 0.937 | 0.960 | 0.209 | 0.271 | 1.293 | 1.157 | 0.073 |
| em: VEIEX -> EEM, 2003-05..2005-12 | 2003-05-01 | 2005-12-30 | 674 | 0.785 | 0.900 | 0.147 | 0.192 | 1.308 | 1.174 | 0.087 |

Reading: the monthly overlaps show no level or vol jump at the splices (vol ratios 0.98-1.04 outside the noisy 1994-2000 EM window). The daily overlaps are dominated by the asynchronous-close effect (next section), not by index differences.

### Every candidate vs the MSCI index (2001+)

Monthly (reference = MSCI EAFE GR or MSCI EM GR, month-ends aligned):

| index | start | end | n | corr | vol_ref | vol_new | vol_ratio | te | mean_diff_ann |
|---|---|---|---|---|---|---|---|---|---|
| intl_dev KF EAFE-wt | 2001-01-31 | 2025-12-31 | 300 | 0.996 | 0.166 | 0.165 | 0.994 | 0.015 | 0.003 |
| intl_dev AQR GxUS | 2001-01-31 | 2026-07-31 | 307 | 0.990 | 0.167 | 0.164 | 0.985 | 0.024 | 0.003 |
| intl_dev KF Dev ex US | 2001-01-31 | 2026-09-30 | 309 | 0.994 | 0.166 | 0.165 | 0.992 | 0.018 | 0.005 |
| intl_dev EFA | 2001-09-30 | 2026-10-31 | 302 | 0.986 | 0.166 | 0.168 | 1.012 | 0.028 | -0.005 |
| intl_dev VTMGX | 2001-01-31 | 2026-10-31 | 310 | 0.990 | 0.166 | 0.170 | 1.025 | 0.024 | -0.001 |
| intl_dev VEA | 2007-08-31 | 2026-10-31 | 231 | 0.984 | 0.174 | 0.179 | 1.031 | 0.032 | 0.000 |
| em KF EM | 2001-01-31 | 2026-09-30 | 309 | 0.995 | 0.210 | 0.202 | 0.963 | 0.021 | 0.005 |
| em VEIEX | 2001-01-31 | 2026-10-31 | 310 | 0.983 | 0.210 | 0.206 | 0.978 | 0.039 | -0.010 |
| em EEM | 2003-05-31 | 2026-10-31 | 282 | 0.974 | 0.206 | 0.209 | 1.015 | 0.047 | -0.009 |
| em VWO | 2005-04-30 | 2026-10-31 | 259 | 0.971 | 0.209 | 0.204 | 0.976 | 0.050 | -0.010 |
| em TEDMX (active) | 2001-01-31 | 2026-10-31 | 310 | 0.971 | 0.210 | 0.210 | 1.000 | 0.051 | -0.006 |

Daily (same reference; MSCI weekday holidays with zero change removed):

| index | start | end | n | corr_1d | corr_3d | vol_ref_1d | vol_new_1d | vol_ratio_1d | vol_ratio_3d | te_3d |
|---|---|---|---|---|---|---|---|---|---|---|
| intl_dev AQR GxUS | 2001-01-02 | 2026-08-31 | 6680 | 0.983 | 0.988 | 0.170 | 0.155 | 0.913 | 0.938 | 0.029 |
| intl_dev KF Dev ex US | 2001-01-02 | 2026-09-30 | 6708 | 0.950 | 0.983 | 0.170 | 0.156 | 0.919 | 0.943 | 0.034 |
| intl_dev EFA | 2001-09-04 | 2026-11-13 | 6337 | 0.687 | 0.891 | 0.171 | 0.207 | 1.215 | 1.070 | 0.088 |
| intl_dev VTMGX | 2001-01-02 | 2026-11-13 | 6506 | 0.749 | 0.918 | 0.170 | 0.194 | 1.139 | 1.040 | 0.075 |
| em VEIEX | 2001-01-02 | 2026-11-13 | 6505 | 0.776 | 0.923 | 0.186 | 0.208 | 1.123 | 1.036 | 0.083 |
| em EEM | 2003-05-01 | 2026-11-13 | 5924 | 0.628 | 0.857 | 0.188 | 0.271 | 1.442 | 1.186 | 0.127 |
| em VWO | 2005-04-01 | 2026-11-13 | 5441 | 0.636 | 0.858 | 0.191 | 0.260 | 1.359 | 1.118 | 0.122 |

### Rejected candidates

Old actively managed international funds would extend DAILY history to 1980, but they are poor index proxies: monthly correlation with the EAFE-weighted index is 0.54-0.93 before 1986 and tracking error 6-10 %/yr (an index fund on the same data shows 1.5-3 %). They were not used; 1975-1985 is monthly tier C instead. TEDMX (Templeton Developing Markets, active, 1991) likewise. VTMGX (EAFE index fund from 1999-08) would add two years of fund-NAV data (tier B) before EFA, at the cost of a third price convention (partially fair-valued NAV); not used.

| index | start | end | n | corr | vol_ref | vol_new | vol_ratio | te | mean_diff_ann |
|---|---|---|---|---|---|---|---|---|---|
| SCINX vs KF EAFE-wt, ..1985 | 1980-02-29 | 1985-12-31 | 71 | 0.929 | 0.158 | 0.156 | 0.986 | 0.059 | -0.016 |
| SCINX vs KF EAFE-wt, full | 1980-02-29 | 2025-12-31 | 551 | 0.919 | 0.167 | 0.171 | 1.024 | 0.069 | -0.008 |
| PRITX vs KF EAFE-wt, ..1985 | 1980-06-30 | 1985-12-31 | 67 | 0.916 | 0.147 | 0.142 | 0.966 | 0.060 | -0.026 |
| PRITX vs KF EAFE-wt, full | 1980-06-30 | 2025-12-31 | 547 | 0.935 | 0.166 | 0.169 | 1.015 | 0.061 | -0.009 |
| VWIGX vs KF EAFE-wt, ..1985 | 1981-11-30 | 1985-12-31 | 50 | 0.827 | 0.148 | 0.170 | 1.145 | 0.096 | -0.022 |
| VWIGX vs KF EAFE-wt, full | 1981-11-30 | 2025-12-31 | 530 | 0.901 | 0.167 | 0.185 | 1.109 | 0.080 | -0.003 |
| VTRIX vs KF EAFE-wt, ..1985 | 1983-06-30 | 1985-12-31 | 31 | 0.747 | 0.136 | 0.133 | 0.976 | 0.096 | -0.105 |
| VTRIX vs KF EAFE-wt, full | 1983-06-30 | 2025-12-31 | 511 | 0.833 | 0.167 | 0.178 | 1.067 | 0.101 | -0.028 |
| FOSFX vs KF EAFE-wt, ..1985 | 1985-01-31 | 1985-12-31 | 12 | 0.542 | 0.081 | 0.084 | 1.035 | 0.079 | 0.149 |
| FOSFX vs KF EAFE-wt, full | 1985-01-31 | 2025-12-31 | 492 | 0.935 | 0.168 | 0.181 | 1.076 | 0.064 | 0.003 |
| TEMFX vs KF EAFE-wt, full | 1986-02-28 | 2025-12-31 | 479 | 0.862 | 0.169 | 0.169 | 1.000 | 0.089 | -0.010 |
| TEDMX vs KF EM, full | 1991-11-30 | 2026-09-30 | 419 | 0.937 | 0.208 | 0.213 | 1.028 | 0.075 | -0.012 |
| KF Dev ex US vs AQR GxUS, monthly, 1990-07..2001-08 | 1990-07-31 | 2001-08-31 | 134 | 0.995 | 0.170 | 0.165 | 0.970 | 0.017 | 0.005 |

| index | start | end | n | corr_1d | corr_3d | vol_ref_1d | vol_new_1d | vol_ratio_1d | vol_ratio_3d | te_3d |
|---|---|---|---|---|---|---|---|---|---|---|
| KF Dev ex US vs AQR GxUS, 1990-07..2001-08 | 1990-07-02 | 2001-08-31 | 2906 | 0.935 | 0.979 | 0.140 | 0.144 | 1.026 | 0.991 | 0.031 |
| VTMGX vs AQR GxUS, 1999-08..2001-08 | 1999-08-18 | 2001-08-31 | 516 | 0.869 | 0.942 | 0.144 | 0.166 | 1.154 | 1.047 | 0.055 |

## Asynchronous closes

Correlation with the US market (^GSPC, US close) on the same day (`corr_us_1d`), with overlapping 3-day sums (`corr_us_3d`) and with the PREVIOUS US day (`corr_us_lag1`); own first-order autocorrelation; annualized vol from 1-day, overlapping 3-day and monthly returns; and the monthly-to-daily variance ratio (vol_monthly / vol_1d)^2:

| index | start | end | corr_us_1d | corr_us_3d | corr_us_lag1 | autocorr_1d | vol_1d | vol_3d | vol_monthly | var_ratio_monthly_to_1d |
|---|---|---|---|---|---|---|---|---|---|---|
| intl_dev AQR GxUS [segment] 1986-01..2001-08 | 1986-01-02 | 2001-08-31 | 0.235 | 0.423 | 0.381 | 0.137 | 0.146 | 0.158 | 0.182 | 1.554 |
| intl_dev AQR GxUS (not used) 2001-09..2026-08 | 2001-09-04 | 2026-08-31 | 0.507 | 0.734 | 0.397 | 0.142 | 0.156 | 0.170 | 0.164 | 1.099 |
| intl_dev MSCI EAFE GR index 2001-09..2026-08 | 2001-09-04 | 2026-08-31 | 0.495 | 0.733 | 0.390 | 0.104 | 0.171 | 0.182 | 0.166 | 0.946 |
| intl_dev EFA [segment] 2001-09..2026-08 | 2001-09-04 | 2026-08-31 | 0.869 | 0.865 | -0.094 | -0.099 | 0.208 | 0.194 | 0.168 | 0.655 |
| intl_dev VTMGX fund NAV 2001-09..2026-08 | 2001-09-04 | 2026-08-31 | 0.840 | 0.854 | -0.032 | -0.053 | 0.194 | 0.188 | 0.170 | 0.768 |
| em VEIEX [segment] 1994-06..2003-04 | 1994-06-01 | 2003-04-30 | 0.455 | 0.572 | 0.337 | 0.233 | 0.189 | 0.217 | 0.256 | 1.836 |
| em VEIEX (not used) 2003-05..2026-10 | 2003-05-01 | 2026-10-30 | 0.782 | 0.770 | -0.031 | 0.017 | 0.209 | 0.214 | 0.202 | 0.929 |
| em MSCI EM GR index 2003-05..2026-10 | 2003-05-01 | 2026-10-30 | 0.430 | 0.652 | 0.378 | 0.161 | 0.188 | 0.208 | 0.207 | 1.224 |
| em EEM [segment] 2003-05..2026-10 | 2003-05-01 | 2026-10-30 | 0.814 | 0.793 | -0.148 | -0.110 | 0.270 | 0.247 | 0.210 | 0.602 |

What this shows:
* **Index / foreign-close data** (AQR, MSCI, VEIEX before ~2001) react to the US day one day late: lag-1 US correlation 0.38 (AQR 1986-2001), 0.39 (MSCI EAFE 2001+), 0.38 (MSCI EM). Same-day correlation with the US is badly understated (EAFE 0.50 vs 0.73 with 3-day sums). Daily returns are positively autocorrelated (staggered closes and stale prices), so the 1-day vol UNDERSTATES monthly vol: variance ratio 1.55 for the AQR segment 1986-2001, 1.84 for the VEIEX segment 1994-2003.
* **US-listed ETFs** (EFA, EEM) price at the US close: same-day US correlation is high (0.87, 0.81) and 3-day correlation barely changes it. But they OVERREACT to the US day and partially revert the next day (lag-1 US correlation -0.09 / -0.15, own autocorrelation -0.10 / -0.11), so 1-day vol OVERSTATES monthly vol: EFA 0.208 vs 0.168 monthly (variance ratio 0.66); EEM 0.270 vs 0.210 (ratio 0.60). On the same dates the MSCI index has 1-day vol 0.171 (EAFE) and 0.188 (EM): EFA's daily vol is 1.22x the index's and EEM's 1.44x, while MONTHLY vols agree within 2 %.
* **Open-end fund NAVs** (VTMGX, VEIEX) sit in between and drift over time: Vanguard's fair-value pricing moved them from foreign-close behaviour (lag-1 US corr ≈ 0.3-0.45 in the 1990s) to near-US-close behaviour by 2007 (see year table).
* **Consequence for the model.** The daily-RV level of `intl_dev` and `em` has a structural break at the ETF splice (2001-09 and 2003-05) that is NOT a change in risk: the monthly-to-daily variance ratio drops from ~1.5-1.8 (index/NAV era) to ~0.6-0.65 (ETF era). Models trained on daily RV across the splice, and the RV→annual bridge, should treat `source`/`tier` (or the splice date) as a regime, or use the 3-day / monthly measures. 3-day overlapping sums remove most of the correlation bias for index data (EAFE–US 0.50 → 0.73) but not all of it, and do not fully remove the ETF vol excess (EFA 3-day vol still 1.07x the index).
* `index_chain()` in the module returns a single-convention foreign-close alternative (AQR → MSCI EAFE GR; MSCI EM GR from 2001) for sensitivity checks; it is not written to disk.

Same-day (`sync`) and lagged (`lagUS`) correlation with ^GSPC by calendar year:

| year  | AQR_GxUS sync | AQR_GxUS lagUS | MSCI_EAFE sync | MSCI_EAFE lagUS | EFA sync | EFA lagUS | VEIEX sync | VEIEX lagUS | MSCI_EM sync | MSCI_EM lagUS | EEM sync | EEM lagUS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1986 | 0.21 | 0.31 |  |  |  |  |  |  |  |  |  |  |
| 1987 | 0.22 | 0.61 |  |  |  |  |  |  |  |  |  |  |
| 1988 | 0.06 | 0.21 |  |  |  |  |  |  |  |  |  |  |
| 1989 | 0.06 | 0.15 |  |  |  |  |  |  |  |  |  |  |
| 1990 | 0.25 | 0.36 |  |  |  |  |  |  |  |  |  |  |
| 1991 | 0.38 | 0.30 |  |  |  |  |  |  |  |  |  |  |
| 1992 | 0.18 | 0.23 |  |  |  |  |  |  |  |  |  |  |
| 1993 | 0.11 | 0.10 |  |  |  |  |  |  |  |  |  |  |
| 1994 | 0.16 | 0.33 |  |  |  |  | 0.19 | 0.43 |  |  |  |  |
| 1995 | -0.03 | 0.22 |  |  |  |  | 0.31 | 0.33 |  |  |  |  |
| 1996 | 0.23 | 0.43 |  |  |  |  | 0.28 | 0.42 |  |  |  |  |
| 1997 | 0.22 | 0.54 |  |  |  |  | 0.44 | 0.38 |  |  |  |  |
| 1998 | 0.41 | 0.30 |  |  |  |  | 0.57 | 0.31 |  |  |  |  |
| 1999 | 0.21 | 0.43 |  |  |  |  | 0.35 | 0.41 |  |  |  |  |
| 2000 | 0.33 | 0.49 |  |  |  |  | 0.47 | 0.48 |  |  |  |  |
| 2001 | 0.50 | 0.40 | 0.48 | 0.37 | 0.84 | 0.21 | 0.61 | 0.25 | 0.46 | 0.44 |  |  |
| 2002 | 0.49 | 0.46 | 0.50 | 0.44 | 0.82 | 0.06 | 0.47 | 0.27 | 0.29 | 0.46 |  |  |
| 2003 | 0.46 | 0.35 | 0.49 | 0.32 | 0.78 | 0.00 | 0.37 | 0.36 | 0.28 | 0.42 | 0.66 | 0.09 |
| 2004 | 0.37 | 0.32 | 0.38 | 0.31 | 0.79 | -0.08 | 0.36 | 0.32 | 0.35 | 0.37 | 0.68 | 0.07 |
| 2005 | 0.20 | 0.43 | 0.20 | 0.40 | 0.72 | -0.11 | 0.56 | 0.14 | 0.23 | 0.45 | 0.74 | -0.08 |
| 2006 | 0.39 | 0.45 | 0.40 | 0.42 | 0.81 | 0.01 | 0.67 | 0.22 | 0.36 | 0.49 | 0.84 | -0.04 |
| 2007 | 0.40 | 0.49 | 0.42 | 0.45 | 0.89 | -0.15 | 0.85 | -0.05 | 0.37 | 0.52 | 0.89 | -0.16 |
| 2008 | 0.45 | 0.55 | 0.43 | 0.57 | 0.94 | -0.15 | 0.89 | -0.10 | 0.48 | 0.46 | 0.91 | -0.26 |
| 2009 | 0.58 | 0.35 | 0.57 | 0.34 | 0.93 | -0.15 | 0.89 | -0.07 | 0.58 | 0.33 | 0.92 | -0.16 |
| 2010 | 0.63 | 0.28 | 0.62 | 0.27 | 0.92 | -0.13 | 0.88 | -0.04 | 0.57 | 0.35 | 0.90 | -0.14 |
| 2011 | 0.68 | 0.34 | 0.69 | 0.30 | 0.94 | -0.15 | 0.90 | -0.01 | 0.52 | 0.48 | 0.92 | -0.14 |
| 2012 | 0.64 | 0.32 | 0.64 | 0.29 | 0.88 | -0.03 | 0.84 | 0.02 | 0.52 | 0.38 | 0.86 | -0.03 |
| 2013 | 0.49 | 0.43 | 0.48 | 0.43 | 0.87 | -0.05 | 0.71 | 0.03 | 0.39 | 0.36 | 0.75 | -0.09 |
| 2014 | 0.41 | 0.52 | 0.41 | 0.46 | 0.83 | -0.04 | 0.67 | 0.00 | 0.33 | 0.40 | 0.66 | -0.10 |
| 2015 | 0.55 | 0.38 | 0.53 | 0.37 | 0.85 | -0.01 | 0.75 | 0.11 | 0.49 | 0.38 | 0.78 | 0.02 |
| 2016 | 0.52 | 0.32 | 0.54 | 0.29 | 0.86 | -0.08 | 0.78 | 0.05 | 0.47 | 0.33 | 0.80 | -0.09 |
| 2017 | 0.35 | 0.23 | 0.34 | 0.21 | 0.69 | -0.08 | 0.57 | 0.04 | 0.26 | 0.34 | 0.62 | -0.06 |
| 2018 | 0.40 | 0.53 | 0.37 | 0.50 | 0.85 | 0.00 | 0.75 | -0.01 | 0.42 | 0.38 | 0.77 | -0.09 |
| 2019 | 0.56 | 0.36 | 0.61 | 0.31 | 0.87 | -0.13 | 0.76 | -0.12 | 0.51 | 0.23 | 0.77 | -0.16 |
| 2020 | 0.67 | 0.20 | 0.66 | 0.21 | 0.92 | -0.27 | 0.87 | -0.25 | 0.60 | 0.15 | 0.88 | -0.35 |
| 2021 | 0.45 | 0.31 | 0.43 | 0.33 | 0.81 | -0.03 | 0.59 | -0.08 | 0.32 | 0.22 | 0.61 | -0.09 |
| 2022 | 0.54 | 0.45 | 0.50 | 0.45 | 0.86 | 0.03 | 0.67 | 0.06 | 0.24 | 0.48 | 0.71 | -0.00 |
| 2023 | 0.47 | 0.35 | 0.42 | 0.35 | 0.77 | -0.01 | 0.63 | 0.05 | 0.30 | 0.41 | 0.67 | -0.02 |
| 2024 | 0.39 | 0.27 | 0.36 | 0.31 | 0.69 | -0.05 | 0.56 | -0.01 | 0.32 | 0.34 | 0.57 | -0.02 |
| 2025 | 0.21 | 0.56 | 0.15 | 0.60 | 0.79 | -0.00 | 0.68 | 0.07 | 0.11 | 0.51 | 0.76 | 0.01 |
| 2026 | 0.49 | 0.33 | 0.44 | 0.34 | 0.80 | -0.11 | 0.81 | -0.07 | 0.27 | 0.44 | 0.79 | -0.20 |

## Annualized volatility by decade

Uncentered, log returns: `monthly` from the monthly file (√12), `daily_1d` from daily returns (√252), `daily_3d` from overlapping 3-day sums. The 2000s daily row mixes conventions (index to 2001-08 / VEIEX to 2003-04, ETF after).

| decade  | intl_dev monthly | intl_dev daily_1d | intl_dev daily_3d | em monthly | em daily_1d | em daily_3d |
|---|---|---|---|---|---|---|
| 1970 | 0.158 |  |  |  |  |  |
| 1980 | 0.180 | 0.155 | 0.164 | 0.228 |  |  |
| 1990 | 0.176 | 0.141 | 0.154 | 0.239 | 0.182 | 0.216 |
| 2000 | 0.187 | 0.234 | 0.215 | 0.257 | 0.325 | 0.295 |
| 2010 | 0.148 | 0.178 | 0.172 | 0.183 | 0.211 | 0.207 |
| 2020 | 0.166 | 0.192 | 0.184 | 0.185 | 0.227 | 0.208 |

## Known episodes (compounded from the monthly files)

| index | em | intl_dev |
|---|---|---|
| 1987-10 |  | -0.125 |
| 1990-01..09 | -0.041 | -0.312 |
| 1997-07..1998-08 | -0.533 | -0.098 |
| 1998-08 | -0.261 | -0.125 |
| 2008-09..2009-02 | -0.462 | -0.447 |
| 2008-10 | -0.256 | -0.208 |
| 2020-03 | -0.158 | -0.141 |
| 2022 | -0.206 | -0.144 |

1987-10 (intl_dev −12.5 %, daily low −13.5 % on 1987-10-20, the day after the US crash: foreign-close convention), 1990 Japan crash (−31 % Jan-Sep 1990), the 1997-98 Asia/Russia crisis for EM (−53 % 1997-07..1998-08, −26 % in 1998-08), 2008 and 2020-03 all look right.

Daily returns beyond ±8 %:

*intl_dev*

| index | ret | source |
|---|---|---|
| 1987-10-20 | -0.1347 | aqr_mkt_global_ex_usa_plus_rf |
| 2008-09-29 | -0.1116 | yahoo_EFA |
| 2008-10-13 | 0.1589 | yahoo_EFA |
| 2008-10-15 | -0.1067 | yahoo_EFA |
| 2008-10-28 | 0.1378 | yahoo_EFA |
| 2008-11-13 | 0.0852 | yahoo_EFA |
| 2008-12-01 | -0.0817 | yahoo_EFA |
| 2016-06-24 | -0.0859 | yahoo_EFA |
| 2020-03-12 | -0.1099 | yahoo_EFA |
| 2020-03-16 | -0.1032 | yahoo_EFA |
| 2020-03-24 | 0.0847 | yahoo_EFA |

*em*

| index | ret | source |
|---|---|---|
| 2001-09-17 | -0.1037 | yahoo_VEIEX |
| 2007-02-27 | -0.0813 | yahoo_EEM |
| 2008-09-18 | 0.0843 | yahoo_EEM |
| 2008-09-19 | 0.1245 | yahoo_EEM |
| 2008-09-29 | -0.1168 | yahoo_EEM |
| 2008-09-30 | 0.0810 | yahoo_EEM |
| 2008-10-02 | -0.0876 | yahoo_EEM |
| 2008-10-13 | 0.2277 | yahoo_EEM |
| 2008-10-15 | -0.1617 | yahoo_EEM |
| 2008-10-22 | -0.1053 | yahoo_EEM |
| 2008-10-24 | -0.1032 | yahoo_EEM |
| 2008-10-28 | 0.2093 | yahoo_EEM |
| 2008-10-30 | 0.1350 | yahoo_EEM |
| 2008-11-04 | 0.0913 | yahoo_EEM |
| 2008-11-05 | -0.1273 | yahoo_EEM |
| 2008-11-07 | 0.0812 | yahoo_EEM |
| 2008-11-13 | 0.1385 | yahoo_EEM |
| 2008-11-14 | -0.0960 | yahoo_EEM |
| 2008-11-19 | -0.0864 | yahoo_EEM |
| 2008-11-21 | 0.1440 | yahoo_EEM |
| 2008-12-01 | -0.0963 | yahoo_EEM |
| 2009-03-10 | 0.0812 | yahoo_EEM |
| 2009-03-23 | 0.0925 | yahoo_EEM |
| 2011-08-08 | -0.0834 | yahoo_EEM |
| 2020-03-12 | -0.1001 | yahoo_EEM |
| 2020-03-16 | -0.1248 | yahoo_EEM |
| 2020-03-18 | -0.0864 | yahoo_EEM |
| 2022-03-16 | 0.0805 | yahoo_EEM |

## Caveats

* `intl_dev` 1986-2001 (AQR) includes Canada (~5-8 % weight; EFA excludes it, VEA includes it). Canada trades in US hours, so this segment is slightly more synchronous with the US than EAFE.
* The 1975-1985 segment (KF, tier C) uses EAFE country weights with Ken French's own value-weighted country portfolios, not MSCI's constituents; its monthly tracking error vs MSCI EAFE is 1.5 %/yr on 2001-2025. MSCI EAFE 1970-1974 was not reachable.
* EM monthly 1989-07..1994-05 (KF) is the only pre-VEIEX EM source found; KF EM vs MSCI EM price index 1997-2000: corr 0.99. MSCI EM 1988 monthly was not reachable. No daily EM source before 1994.
* VEIEX 1994-2000 tracked MSCI EM less tightly than later (monthly corr 0.95 and TE 8 %/yr vs KF EM, 0.97 vs the MSCI EM price index 1997-2000), partly because it tracked 'Select EM Free' (country subset) and partly NAV timing. VEIEX carried purchase/redemption fees in the 1990s that are not in the NAV return.
* After 9/11 the US-close series have no prices 2001-09-11..14; the 2001-09-17 return spans five foreign trading days (VEIEX −10.4 %).
* Yahoo `adjclose` is a vendor series and can be revised; raw JSON is cached in `data/raw/intl_equity/`.
* Daily files keep each source's own calendar: the AQR segment includes non-US trading days (e.g. US Thanksgiving) when foreign markets traded; the assembler should compound or drop such days consistently with cash.
