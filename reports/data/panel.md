# Assembled panel

Built by `python -m volmodel.data.assemble` (module `src/volmodel/data/assemble.py`). Read with `volmodel.data.panel.load_panel()`. All returns SIMPLE total returns in decimals; excess/log transforms are done by `Panel`.

* Daily panel: 1926-07-01 .. 2026-11-13, 26,370 US equity trading days (the cash calendar), 12 assets.
* Monthly panel: 1871-02-28 .. 2026-10-31, 1,869 month-ends. Last month = last month whose final trading day is in the daily data (data end 2026-11-13, so the partial current month is excluded).

## Coverage

| asset | first_monthly | first_daily | first_daily_month | A | B | C | D | tiers_by_era |
|---|---|---|---|---|---|---|---|---|
| us_large | 1871-02-28 | 1926-07-01 | 1926-07 | 405 | 799 | 0 | 665 | D 1871-02..1926-06, B 1926-07..1993-01, A 1993-02..2026-10 |
| us_small | 1926-07-31 | 1926-07-01 | 1926-07 | 317 | 887 | 0 | 0 | B 1926-07..2000-05, A 2000-06..2026-10 |
| intl_dev | 1975-01-31 | 1986-01-02 | 1986-01 | 302 | 188 | 132 | 0 | C 1975-01..1985-12, B 1986-01..2001-08, A 2001-09..2026-10 |
| em | 1989-07-31 | 1994-06-01 | 1994-06 | 389 | 0 | 59 | 0 | C 1989-07..1994-05, A 1994-06..2026-10 |
| ust_int | 1953-05-31 | 1962-01-03 | 1962-02 | 291 | 486 | 0 | 105 | D 1953-05..1962-01, B 1962-02..2002-07, A 2002-08..2026-10 |
| ust_long | 1919-02-28 | 1962-01-03 | 1962-02 | 291 | 486 | 0 | 516 | D 1919-02..1962-01, B 1962-02..2002-07, A 2002-08..2026-10 |
| tips | 1997-02-28 | 1997-01-30 | 1997-02 | 274 | 83 | 0 | 0 | B 1997-02..2003-12, A 2004-01..2026-10 |
| ig_credit | 1919-02-28 | 1980-01-03 | 1980-02 | 561 | 0 | 649 | 83 | D 1919-02..1925-12, C 1926-01..1980-01, A 1980-02..2026-10 |
| hy_credit | 1980-02-29 | 1985-01-02 | 1985-01 | 502 | 0 | 59 | 0 | C 1980-02..1984-12, A 1985-01..2026-10 |
| reits | 1972-01-31 | 1986-11-17 | 1986-12 | 365 | 114 | 179 | 0 | C 1972-01..1986-11, B 1986-12..1996-05, A 1996-06..2026-10 |
| gold | 1971-09-30 | 1971-09-01 | 1971-09 | 263 | 399 | 0 | 0 | B 1971-09..2004-11, A 2004-12..2026-10 |
| commodities | 1926-07-31 | 1991-02-01 | 1991-02 | 0 | 429 | 775 | 0 | C 1926-07..1991-01, B 1991-02..2026-10 |

`first_daily_month` = first month compounded from daily data (earlier months, including a partial first daily month, come from the track's monthly file). Tier columns count months.

### Sources by segment

| asset | freq | start | end | tier | source |
|---|---|---|---|---|---|
| us_large | M | 1871-02-28 | 1926-06-30 | D | shiller_sp_composite_avgprice |
| us_large | D | 1926-07-01 | 1993-01-29 | B | french_me_vw_hi30 |
| us_large | D | 1993-02-01 | 2026-11-13 | A | yahoo_SPY |
| us_small | D | 1926-07-01 | 2000-05-26 | B | french_me_vw_qnt2_qnt3_5050 |
| us_small | D | 2000-05-30 | 2026-11-13 | A | yahoo_IWM |
| intl_dev | M | 1975-01-31 | 1985-12-31 | C | french_intl_indices_eafe_weighted |
| intl_dev | D | 1986-01-02 | 2001-08-31 | B | aqr_mkt_global_ex_usa_plus_rf |
| intl_dev | D | 2001-09-04 | 2026-11-13 | A | yahoo_EFA |
| em | M | 1989-07-31 | 1994-05-31 | C | french_emerging_mkt_plus_rf |
| em | D | 1994-06-01 | 2003-04-30 | A | yahoo_VEIEX |
| em | D | 2003-05-01 | 2026-11-13 | A | yahoo_EEM |
| ust_int | M | 1953-05-31 | 1962-01-31 | D | H.15 10y CMT monthly-average yield, par-bond ladder 7-10y |
| ust_int | D | 1962-01-03 | 1971-08-16 | B | H.15 10y CMT par-bond ladder 7-10y (flat curve) |
| ust_int | D | 1971-08-17 | 2002-07-30 | B | GSW fitted curve, par-bond ladder 7-10y |
| ust_int | D | 2002-07-31 | 2026-11-13 | A | IEF adjclose (Yahoo) |
| ust_long | M | 1919-02-28 | 1941-10-31 | D | NBER m13033a long-term US bond yield (monthly avg), ladder 20-30y |
| ust_long | M | 1941-11-30 | 1953-04-30 | D | NBER m13033b long-term US bond yield (monthly avg), ladder 20-30y |
| ust_long | M | 1953-05-31 | 1962-01-31 | D | H.15 20y CMT monthly-average yield, par-bond ladder 20-30y |
| ust_long | D | 1962-01-03 | 1977-02-15 | B | H.15 20y CMT par-bond ladder 20-30y (flat beyond 20y) |
| ust_long | D | 1977-02-16 | 1985-11-25 | B | H.15 20y/30y CMT par-bond ladder 20-30y (interpolated) |
| ust_long | D | 1985-11-26 | 2002-07-30 | B | GSW fitted curve, par-bond ladder 20-30y |
| ust_long | D | 2002-07-31 | 2026-11-13 | A | TLT adjclose (Yahoo) |
| tips | D | 1997-01-30 | 2003-12-31 | B | yahoo:PRRIX NAV + distributions, month-end distributions spread as da… |
| tips | D | 2004-01-02 | 2026-11-13 | A | yahoo:TIP close + dividends (iShares TIPS Bond ETF) |
| ig_credit | M | 1919-02-28 | 1925-12-31 | D | moodys_aaa_baa_avg_parbond |
| ig_credit | M | 1926-01-31 | 1980-01-31 | C | ibbotson_ltcorp_rf_blend |
| ig_credit | D | 1980-01-03 | 1984-12-31 | A | blend_VWESX_FBNDX |
| ig_credit | D | 1985-01-02 | 2002-07-30 | A | blend_VWESX_VFSTX |
| ig_credit | D | 2002-07-31 | 2026-11-13 | A | yahoo_LQD |
| hy_credit | M | 1980-02-29 | 1984-12-31 | C | yahoo_VWEHX_exdiv_monthly |
| hy_credit | D | 1985-01-02 | 2007-04-11 | A | yahoo_VWEHX |
| hy_credit | D | 2007-04-12 | 2026-11-13 | A | yahoo_HYG |
| reits | M | 1972-01-31 | 1986-11-30 | C | nareit:FTSE Nareit All Equity REITs total return |
| reits | D | 1986-11-17 | 1996-05-31 | B | yahoo:FRESX adjclose (Fidelity Real Estate Investment, active) |
| reits | D | 1996-06-03 | 2004-09-30 | A | yahoo:VGSIX adjclose (Vanguard REIT Index Inv) |
| reits | D | 2004-10-01 | 2026-11-13 | A | yahoo:VNQ adjclose (Vanguard Real Estate ETF) |
| gold | D | 1971-09-01 | 1998-12-30 | B | bundesbank:BBEX3 Frankfurt gold fixing DEM/kg / Frankfurt USD fixing |
| gold | D | 1998-12-31 | 2004-11-30 | B | ft:GC.1:CMX COMEX gold front-month close |
| gold | D | 2004-12-01 | 2026-11-13 | A | ft:GLD close (no distributions; = total return) |
| commodities | M | 1926-07-31 | 1991-01-31 | C | aqr:Commodities for the Long Run EW excess return + KenFrench monthly… |
| commodities | D | 1991-02-01 | 2009-06-30 | B | ft:BCOM (excess return) + KenFrench daily RF |
| commodities | D | 2009-07-01 | 2026-11-13 | B | ft:BCOMTR (Bloomberg Commodity Index Total Return) |
| cash | M | 1871-01-31 | 1919-12-31 | D | nber_m13002_cp_minus_spread |
| cash | M | 1920-01-31 | 1926-06-30 | D | nber_m13029a_treas_3_6m |
| cash | D | 1926-07-01 | 2026-09-30 | C | french_rf |
| cash | D | 2026-10-01 | 2026-11-10 | D | h15_tbill4w |
| cash | D | 2026-11-11 | 2026-11-13 | D | h15_tbill4w_ffill |

Full notes (overlap statistics) are in `data/processed/segments.csv`.

## Calendar handling

Off-calendar returns are compounded into the next calendar day; calendar days without a return are NaN within the asset's span (the next return spans the gap).

| asset | input_days | folded | folded_across_month | dropped | nan_days_in_span | examples_folded | examples_nan |
|---|---|---|---|---|---|---|---|
| us_large | 26370 | 0 | 0 | 0 | 0 |  |  |
| us_small | 26370 | 0 | 0 | 0 | 0 |  |  |
| intl_dev | 10412 | 117 | 3 | 0 | 2 | 1986-02-17→1986-02-18, 1986-03-28→1986-03-31 | 1991-11-01, 1997-12-24 |
| em | 8170 | 0 | 0 | 0 | 0 |  |  |
| ust_int | 16217 | 31 | 1 | 0 | 140 | 1968-06-12→1968-06-13, 1968-06-19→1968-06-20 | 1962-02-12, 1962-10-12, 1962-11-12 |
| ust_long | 16217 | 31 | 1 | 0 | 140 | 1968-06-12→1968-06-13, 1968-06-19→1968-06-20 | 1962-02-12, 1962-10-12, 1962-11-12 |
| tips | 7495 | 0 | 0 | 0 | 0 |  |  |
| ig_credit | 11813 | 0 | 0 | 0 | 0 |  |  |
| hy_credit | 10549 | 0 | 0 | 0 | 0 |  |  |
| reits | 10074 | 0 | 0 | 0 | 1 |  | 1993-07-29 |
| gold | 13830 | 147 | 1 | 0 | 235 | 1971-09-06→1971-09-07, 1971-11-25→1971-11-26 | 1971-11-17, 1971-12-20, 1972-04-03 |
| commodities | 9072 | 62 | 3 | 0 | 1 | 1991-02-18→1991-02-19, 1991-03-29→1991-04-01 | 2017-02-06 |

### Monthly = compounded daily

| asset | daily_months | compared | max_abs_vs_track | worst_month | unfolded_max_abs_vs_track | months_moved_by_folding | max_folding_effect |
|---|---|---|---|---|---|---|---|
| us_large | 1204 | 1204 | 1.03e-15 | 1945-04 | 1.03e-15 | 0 | 2.78e-17 |
| us_small | 1204 | 1204 | 1.33e-15 | 1950-12 | 1.33e-15 | 0 | 2.78e-17 |
| intl_dev | 490 | 490 | 6.28e-03 | 1993-05 | 1.02e-15 | 6 | 6.28e-03 |
| em | 389 | 389 | 7.91e-16 | 2013-09 | 7.91e-16 | 0 | 2.78e-17 |
| ust_int | 777 | 777 | 1.52e-03 | 1968-07 | 1.23e-15 | 2 | 1.52e-03 |
| ust_long | 777 | 777 | 1.57e-03 | 1968-07 | 1.07e-15 | 2 | 1.57e-03 |
| tips | 357 | 357 | 1.25e-16 | 2026-05 | 1.24e-16 | 0 | 6.94e-18 |
| ig_credit | 561 | 561 | 4.16e-17 | 2020-03 | 4.16e-17 | 0 | 6.94e-18 |
| hy_credit | 502 | 502 | 1.39e-17 | 1990-09 | 1.39e-17 | 0 | 1.39e-17 |
| reits | 479 | 479 | 8.74e-16 | 2005-04 | 8.74e-16 | 0 | 1.39e-17 |
| gold | 662 | 662 | 3.07e-03 | 1976-06 | 7.96e-16 | 2 | 3.07e-03 |
| commodities | 429 | 429 | 1.07e-15 | 2003-01 | 1.07e-15 | 0 | 1.39e-17 |

`max_abs_vs_track`: panel month (compounded folded daily) minus the track's monthly file; `unfolded_max_abs_vs_track`: same with the track's own daily dates (tests the track's internal consistency); `max_folding_effect`: the part due to folding moving a return across a month boundary (e.g. a foreign trading day on a US holiday that is the first weekday of a month).
Cash: compounded daily vs monthly max abs gap 2.86e-15.

## Validation

| check | value |
|---|---|
| daily_strictly_increasing | yes |
| daily_index_equals_cash_calendar | yes |
| monthly_index_month_ends | yes |
| n_ret_le_minus1 | 0 |
| min_daily_ret | -0.1951 |
| min_monthly_ret | -0.3275 |
| last_month | 2026-10-31 |
| no_partial_final_month | yes |
| cash_daily_nan | 0 |
| cash_monthly_nan | 0 |
| excess_nan_from_cash_daily | 0 |
| excess_nan_from_cash_monthly | 0 |
| pred_available_ge_obs | yes |
| pred_dup_series_obs | 0 |
| pred_n_rows | 348018 |
| pred_n_series | 62 |

Returns before documented start: none.

`load_panel()` smoke test: daily (26370, 12), monthly (1869, 12), 62 predictor series; `asof(2008-10-15)`: daily max 2008-10-15, monthly max 2008-09-30, latest predictor available_date 2008-10-15, no leak: True; eligible (36m) then: us_large, us_small, intl_dev, em, ust_int, ust_long, tips, ig_credit, hy_credit, reits, gold, commodities.

### Five largest absolute daily returns

| asset | days |
|---|---|
| us_large | 1929-10-30 +11.8% (1929 crash); 1933-03-15 +15.3% (1933 bank holiday/reflation); 1987-10-19 -18.9% (Oct 1987 crash); 2008-10-13 +14.5% (GFC); 2008-10-28 +11.7% (GFC) |
| us_small | 1932-09-21 +13.4% (1932 bear market/rally); 1933-03-15 +21.7% (1933 bank holiday/reflation); 1933-07-21 -14.0% (1933 bank holiday/reflation); 1937-10-20 +16.0% (1937 crash); 1939-09-05 +16.2% (outbreak of WWII) |
| intl_dev | 1987-10-20 -13.5% (Oct 1987 crash); 2008-09-29 -11.2% (Lehman / GFC); 2008-10-13 +15.9% (GFC); 2008-10-28 +13.8% (GFC); 2020-03-12 -11.0% (COVID crash) |
| em | 2008-10-13 +22.8% (GFC); 2008-10-15 -16.2% (GFC); 2008-10-28 +20.9% (GFC); 2008-11-13 +13.8% (GFC); 2008-11-21 +14.4% (GFC) |
| ust_int | 1980-02-19 -3.0% (Volcker / rates spike); 1980-02-28 +3.2% (Volcker / rates spike); 1980-04-16 +3.1% (Volcker / rates reversal); 1987-10-20 +4.4% (Oct 1987 crash); 2009-03-18 +3.4% (March 2009 bottom / QE1 / PPIP) |
| ust_long | 1987-10-20 +5.8% (Oct 1987 crash); 2020-03-16 +6.5% (COVID crash); 2020-03-17 -6.7% (COVID crash); 2020-03-18 -5.6% (COVID crash); 2020-03-20 +7.5% (COVID crash) |
| tips | 2008-10-08 -3.0% (GFC); 2008-12-16 +3.2% (GFC); 2009-03-18 +3.7% (March 2009 bottom / QE1 / PPIP); 2020-03-10 -2.9% (COVID crash); 2020-03-20 +4.5% (COVID crash) |
| ig_credit | 2008-09-16 -6.3% (Lehman / GFC); 2008-09-29 -9.1% (Lehman / GFC); 2008-09-30 +9.8% (Lehman / GFC); 2008-10-13 +5.5% (GFC); 2020-03-23 +7.4% (COVID crash) |
| hy_credit | 2008-09-18 +7.9% (Lehman / GFC); 2008-09-29 -8.1% (Lehman / GFC); 2008-10-10 -6.7% (GFC); 2008-10-13 +12.3% (GFC); 2020-04-09 +6.5% (COVID / Fed facilities) |
| reits | 2008-10-28 +15.6% (GFC); 2008-11-24 +17.0% (GFC); 2008-12-01 -19.5% (GFC); 2009-03-23 +16.2% (March 2009 bottom / QE1 / PPIP); 2020-03-16 -17.7% (COVID crash) |
| gold | 1980-01-03 +12.0% (gold spike and collapse); 1980-01-16 +12.6% (gold spike and collapse); 1980-01-18 +12.9% (gold spike and collapse); 1980-01-23 -16.8% (gold spike and collapse); 1980-03-14 -12.1% (silver crisis / rates spike) |
| commodities | 2008-09-29 -5.2% (Lehman / GFC); 2008-10-06 -5.1% (GFC); 2008-10-10 -6.2% (GFC); 2008-10-29 +5.8% (GFC); 2009-03-19 +5.7% (March 2009 bottom / QE1 / PPIP) |

Every date matches a known stress episode. Several ETF days are one-day overshoots that reverse the next day (e.g. LQD 2008-09-29/30, EEM/EFA 2008-10), documented by the credit and international tracks; the 1980 gold days are the Frankfurt fixing during the January 1980 spike and collapse (real-assets track keeps them as genuine).

## Predictors

`predictors.parquet`: 348,018 rows, 62 series, long format (obs_date, available_date, series, value). `predictors_catalog.csv` lists every series. Standard units: rates, yields, spreads, breakevens, inflation/growth rates and implied vols in DECIMALS; durations in years; index levels and flags unchanged. `live` = an observation within 60 days (monthly series: 100 days) of the data end.

| group | series | live | start | end |
|---|---|---|---|---|
| credit | 7 | 1 | 1919-01-31 | 2026-10-31 |
| implied | 23 | 22 | 1986-01-02 | 2026-11-13 |
| macro | 16 | 16 | 1871-01-31 | 2026-10-31 |
| tips | 4 | 4 | 1999-01-04 | 2026-11-06 |
| treasury | 12 | 12 | 1962-01-02 | 2026-11-09 |

Conversions applied:

- `credit_spreads:gz_credit_spread_m` → `credit_gz_spread_m` x0.01 (decimal); raw |x| q90 = 2.81
- `credit_spreads:gz_excess_bond_premium_m` → `EXCLUDED` excluded: full-sample regression residual (look-ahead)
- `credit_spreads:moodys_aaa_yield_m` → `credit_moodys_aaa_yield_m` x0.01 (decimal); raw |x| q90 = 9.04
- `credit_spreads:moodys_baa_minus_aaa_d` → `credit_moodys_baa_aaa_spread_d` x0.01 (decimal); raw |x| q90 = 1.37
- `credit_spreads:moodys_baa_minus_aaa_m` → `credit_moodys_baa_aaa_spread_m` x0.01 (decimal); raw |x| q90 = 2.05
- `credit_spreads:moodys_baa_minus_ust10y_d` → `credit_moodys_baa_ust10y_spread_d` x0.01 (decimal); raw |x| q90 = 3.19
- `credit_spreads:moodys_baa_minus_ust10y_m` → `credit_moodys_baa_ust10y_spread_m` x0.01 (decimal); raw |x| q90 = 2.92
- `credit_spreads:moodys_baa_yield_m` → `credit_moodys_baa_yield_m` x0.01 (decimal); raw |x| q90 = 10.4
- `macro:macro_infl_vol_36m` → `macro_infl_vol_36m` x0.01 (decimal (annualized)); raw |x| q90 = 5.85
- `macro:macro_infl_yoy` → `macro_infl_yoy` x0.01 (decimal (log change)); raw |x| q90 = 10.3
- `macro:macro_ip_yoy` → `macro_ip_yoy` x0.01 (decimal (log change)); raw |x| q90 = 19.1
- `macro:macro_long_yield` → `macro_long_yield` x0.01 (decimal); raw |x| q90 = 7.58
- `macro:macro_policy_rate` → `macro_policy_rate` x0.01 (decimal); raw |x| q90 = 8.14
- `macro:macro_policy_rate_chg_12m` → `macro_policy_rate_chg_12m` x0.01 (decimal (change)); raw |x| q90 = 2.69
- `macro:macro_policy_rate_chg_3m` → `macro_policy_rate_chg_3m` x0.01 (decimal (change)); raw |x| q90 = 1.02
- `macro:macro_real_short_rate` → `macro_real_short_rate` x0.01 (decimal); raw |x| q90 = 10.7
- `macro:macro_sahm_gap` → `macro_sahm_gap` x0.01 (decimal (unemployment-rate change)); raw |x| q90 = 1.73
- `macro:macro_short_rate` → `macro_short_rate` x0.01 (decimal); raw |x| q90 = 6.44
- `macro:macro_term_spread` → `macro_term_spread` x0.01 (decimal); raw |x| q90 = 2.69
- `macro:macro_unrate` → `macro_unrate` x0.01 (decimal); raw |x| q90 = 7.8
- `macro:macro_unrate_chg_12m` → `macro_unrate_chg_12m` x0.01 (decimal (change)); raw |x| q90 = 2.3
- `macro:macro_usd_broad_yoy` → `macro_usd_broad_yoy` x0.01 (decimal (log change)); raw |x| q90 = 11.7
- `tips_real_yields:breakeven_10y` → `tips_breakeven_10y` x0.01 (decimal); raw |x| q90 = 2.58
- `tips_real_yields:breakeven_5y` → `tips_breakeven_5y` x0.01 (decimal); raw |x| q90 = 2.52
- `tips_real_yields:real_yield_10y` → `tips_real_yield_10y` x0.01 (decimal); raw |x| q90 = 3.37
- `tips_real_yields:real_yield_5y` → `tips_real_yield_5y` x0.01 (decimal); raw |x| q90 = 3.06
- `treasury_curve:duration_ust_int` → `ust_ladder_duration_int` renamed, units unchanged (years)
- `treasury_curve:duration_ust_int_etf_est` → `ust_etf_duration_est_int` renamed, units unchanged (years)
- `treasury_curve:duration_ust_long` → `ust_ladder_duration_long` renamed, units unchanged (years)
- `treasury_curve:duration_ust_long_etf_est` → `ust_etf_duration_est_long` renamed, units unchanged (years)
- `treasury_curve:yield_10y` → `ust_cmt_yield_10y` renamed, units unchanged (decimal)
- `treasury_curve:yield_20y` → `ust_cmt_yield_20y` renamed, units unchanged (decimal)
- `treasury_curve:yield_2y` → `ust_cmt_yield_2y` renamed, units unchanged (decimal)
- `treasury_curve:yield_30y` → `ust_cmt_yield_30y` renamed, units unchanged (decimal)
- `treasury_curve:yield_5y` → `ust_cmt_yield_5y` renamed, units unchanged (decimal)
- `treasury_curve:yield_7y` → `ust_cmt_yield_7y` renamed, units unchanged (decimal)
- `treasury_curve:yield_ladder_ust_int` → `ust_ladder_yield_int` renamed, units unchanged (decimal)
- `treasury_curve:yield_ladder_ust_long` → `ust_ladder_yield_long` renamed, units unchanged (decimal)

Not live: `credit_moodys_aaa_yield_m` (ends 2025-12-31), `credit_moodys_baa_aaa_spread_d` (ends 2016-10-07), `credit_moodys_baa_aaa_spread_m` (ends 2025-12-31), `credit_moodys_baa_ust10y_spread_d` (ends 2016-10-07), `credit_moodys_baa_ust10y_spread_m` (ends 2025-12-31), `credit_moodys_baa_yield_m` (ends 2025-12-31), `iv_vxo` (ends 2021-01-07).

Unit cross-checks (after conversion; small numbers confirm consistent units across files):

- macro_long_yield - monthly avg ust_cmt_yield_10y (1962+), median abs: 0.0000
- ust 10y - tips real 10y - tips breakeven 10y, median (par vs zero basis): -0.0008
- Baa - Aaa - (Baa-Aaa spread), max abs: 0.0000
- Baa - (Baa-10y spread) - macro_long_yield (1953-04+), median abs: 0.0000
- macro_short_rate - 12 x cash monthly (1934+), median abs: 0.0016

## Caveats for modelers

Collected from the track reports; each is something a model can get wrong silently.

**Stale or smoothed prices (daily vol understated, positive autocorrelation).**
- `hy_credit` 1985-01..2007-04 is the VWEHX mutual-fund NAV (matrix-priced): daily vol is about half of HYG's,
  lag-1 autocorrelation 0.12-0.57; monthly-from-daily variance ratio 1.3-2.2. 1980-1984 is one NAV per month
  sampled on ex-dividend dates (tier C), a few days off month-end.
- `ig_credit` 1980-2002 is a daily constant-mix blend of open-end bond funds: mild staleness (AC1 0.08-0.15 in the
  1980s-90s); daily correlation with LQD only 0.67 (bond 3pm vs ETF 4pm close).
- `reits` 1986-11..1996-05 is FRESX, an actively managed fund (tier B), and 1996-2004 VGSIX NAV: daily AC1 0.12-0.32
  until 2003, 1990s monthly variance about 2x what daily data imply; the reverse (ratio ~0.6-0.7) after VNQ (2004).
  The pre-1990s REIT market was tiny and illiquid: do not read 12-16% vol as today's normal.
- `ust_int`/`ust_long` daily 1962-c.1990 are reconstructions from quoted yields with positively autocorrelated
  changes (21d/1d variance ratio up to 2.7 in 1970-74); 1960s quotes are in 32nds (36-53% zero changes).
- `tips` 1997-2003 is PRRIX (active PIMCO fund) with daily-accrued distributions re-spread; NAVs move in 0.1% ticks
  (35-40% zero days in 1997-2000).
- `intl_dev` 1986-2001 (AQR index, foreign closes) and `em` 1994-2003 (VEIEX NAV, foreign closes, fair-valued from
  ~2001) react to US news one day late: same-day correlation with US assets badly understated (lag-1 US corr ~0.38).

**US-listed ETFs of foreign/illiquid assets (daily vol overstated, negative autocorrelation).**
- EFA (2001-09+) and EEM (2003-05+): 1-day vol 1.2-1.44x the MSCI index, monthly vols agree within 2%.
- LQD/HYG in 2008 and March 2020 traded at multi-day discounts to NAV; one-day reversals of up to 10-12%.
- TIP in 2008 Q4 traded at sizeable discounts/premia.

**Composition changes at splices.**
- `commodities`: equal-weight AQR futures basket (1926-1991, few agricultural contracts before the 1960s) -> BCOM
  weights (1991+): vol ratio AQR-EW/BCOM 0.75-0.97 on the overlap; treat pre-1970 as a different asset. BCOM
  1991-98 is a backfill. BCOM is not DBC: DBC has ~1.17x BCOM's vol.
- `ig_credit` pre-1980: 0.81 x Ibbotson long-term corporate + 0.19 x T-bill (high-grade long-bond risk; gained in
  2008), vol-matched to LQD on 2002-2026; LQD's own duration drifted from ~7 to ~8.5.
- `ust_long`: a 20-30y par ladder; TLT's effective duration differs (vol ratio 0.86-0.93 vs TLT in 2022-26).
- `intl_dev` 1986-2001 includes Canada (EFA does not); 1975-85 is Ken French's EAFE-weighted index.
- `reits`: Nareit All Equity (1972-86) -> FRESX (active) -> MSCI US REIT (VGSIX) -> MSCI US IMI RE 25/50 (VNQ).
- `gold`: Frankfurt fixing 13:00 CET (1971-98) -> COMEX front-month settle (1999-2004) -> GLD 16:00 ET.

**Month-average (tier D) eras: vol understated ~15%, AC1 ~0.25-0.4, timing blurred.**
- `ust_long` 1919-02..1962-01, `ust_int` 1953-05..1962-01 (month-average yields; WWII peg 1942-51 near-zero vol;
  1919-41 partially tax-exempt bonds; pre-1973 long end includes flower bonds).
- `ig_credit` 1919-02..1925-12 (month-average Moody's yields; monthly corr with the later index only 0.41).
- `us_large` pre-1926 (Shiller/Cowles month-average prices), if present. Cash before 1926-07 is model-based (tier D).

**Series that end, have holes, or have known breaks.**
- `ust_long` daily: 1973-01-04 return dropped (H.15 20y re-basing): a genuine lost day, not a spanning gap.
- `ust_*` daily: Lincoln's birthday/Columbus/Veterans days are NaN (bond market closed, NYSE open); 1968 Wednesdays
  (NYSE closed, bonds traded) are folded into Thursday.
- `reits` 1993-07-29 FRESX bad print removed (next day spans two days).
- Cash 2026-10-01.. uses H.15 4-week bill (tier D); the last 3 days carry the last published yield.
- Predictors: Moody's yields end 2025-12 (not live); daily Moody's spreads end 2016-10; `iv_vxo` ended 2021;
  VXHYG has no data 2021-09..2025-05, VXIEF/VXLQD 2022-11..2025-07; `iv_move` not built; the 2025-10 CPI and
  unemployment prints were never published (missing, not imputed); industrial production is latest-vintage
  (revised); unemployment SA factors revised. `gz_excess_bond_premium_m` is excluded (full-sample regression).
- Treasury-curve predictors are dated available on the observation day although H.15 is published the next
  business day; GSW TIPS yields carry a 10-business-day publication lag.

