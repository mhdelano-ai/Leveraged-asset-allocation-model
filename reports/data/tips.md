# tips — broad US TIPS total return (TIP-like)

Built by `python -m volmodel.data.sources.tips` on 2026-11-14.

## Result

* daily: 1997-01-30 .. 2026-11-13 (7494 obs)
* monthly: 1997-02-28 .. 2026-10-31 (357 months; compounded daily; current incomplete month excluded)
* returns are SIMPLE TOTAL returns in decimals (not excess).
* TIPS were first issued in Jan 1997; nothing earlier is reported or synthesized.

## Sources

* Yahoo chart API (close + dividend events; Yahoo adjclose used only as a cross-check): TIP, PRRIX, ACITX, VIPSX, VAIPX, SCHP, BPRIX. Total return = (close_t + div_t)/close_{t-1} − 1. Yahoo repeats TIP's 2008-04-01 distribution (0.632) on 2008-04-03 with no second price drop; exact repeats within 5 days are dropped (it is the only one among the symbols used).
* Fed GSW TIPS curve `https://www.federalreserve.gov/data/yield-curve-tables/feds200805.csv` (Gürkaynak-Sack-Wright 2010): real zero curve parameters, zero real yields and breakevens, daily from 1999-01-04.
* CPI-U NSA (BLS CUUR0000SA0) via DBnomics — only for the GSW reconstruction check.

## Segments

| start | end | freq | tier | source | notes |
|---|---|---|---|---|---|
| 1997-01-30 | 2003-12-31 | D | B | yahoo:PRRIX NAV + distributions, month-end distributions spread as daily accrual (PIMCO Real Return Instl, active) | Earliest TIPS fund (inception 1997-01-29). Daily-declared dividends re-spread over their accrual month; December specials kept on ex-date. Active manager: some non-TIPS exposure. Tier B: open-end fund NAV (also an accrual reconstruction and an active mandate). Bad NAV print 2000-12-12 (+0.84%/-0.99%, ACITX and VIPSX flat) dropped: the 12-13 return spans it. |
| 2004-01-02 | 2026-11-13 | D | A | yahoo:TIP close + dividends (iShares TIPS Bond ETF) | Market price (not NAV): includes premium/discount noise, notably 2008 Q4. Every Yahoo distribution validated by the ex-date price test against GSW real-yield changes and VIPSX/SCHP (none rejected; Yahoo's duplicate of 2008-04-01 on 2008-04-03 removed). |

Monthly file = compounded daily; first month 1997-02 (Jan 1997 has only 2 trading days of fund history and is dropped).

## Distribution treatment (why PRRIX is rebuilt from NAV)

PIMCO and American Century declare dividends daily and pay monthly: the NAV does not drop on month-end payment dates, yet Yahoo's adjclose books the month's income as a one-day return on that date. Classification of PRRIX distributions 1997-2003 (NAV return on the payment day should be ≈ −yield for ex-date distributions, ≈ 0 for accruals):

| treated as | n | mean distribution yield | mean NAV return on pay day |
|---|---|---|---|
| daily_accrual | 85 | 0.46% | 0.07% |
| ex_date | 5 | 1.53% | -1.53% |

Effect on the 1997-2003 PRRIX daily vol (annualized): adjclose 4.66% vs rebuilt 4.23%; monthly returns are essentially unchanged (see the adjclose-vs-fix row below).

## Overlap diagnostics

`a vs b`: corr / vol ratio (a/b) / tracking error (annualized) on daily, 3-day-overlapping log (corr only) and monthly returns; `ret diff` = annualized mean monthly return a − b.

| pair | days | corr d | corr 3d | vol ratio d | TE d | months | corr m | vol ratio m | TE m | vol a m | vol b m | ret diff |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| PRRIX_fix vs TIP (splice overlap, 2004-2006) | 755 | 0.940 | 0.975 | 0.989 | 0.018 | 36 | 0.989 | 0.973 | 0.008 | 0.051 | 0.053 | 0.004 |
| PRRIX_fix vs TIP (2004-2026) | 5754 | 0.927 | 0.954 | 0.985 | 0.023 | 274 | 0.970 | 1.082 | 0.015 | 0.062 | 0.057 | 0.004 |
| PRRIX_fix vs ACITX_fix (1997-02..2003) | 1733 | 0.759 | 0.879 | 0.928 | 0.031 | 82 | 0.982 | 1.054 | 0.010 | 0.051 | 0.048 | 0.017 |
| PRRIX_fix vs VIPSX (2000-07..2003; VIPSX divs missing) | 878 | 0.888 | 0.923 | 0.949 | 0.025 | 42 | 0.956 | 1.034 | 0.019 | 0.064 | 0.062 | 0.032 |
| PRRIX ex-date TR (=adjclose) vs PRRIX_fix (1997-2003) | 1739 | 0.922 | 0.935 | 1.101 | 0.018 | 83 | 1.000 | 0.998 | 0.000 | 0.050 | 0.050 | -0.000 |
| ACITX_fix vs TIP (2004-2026) | 5754 | 0.934 | 0.967 | 0.961 | 0.022 | 274 | 0.991 | 0.994 | 0.007 | 0.057 | 0.057 | -0.002 |
| VIPSX vs TIP (2004-2026) | 5754 | 0.945 | 0.972 | 0.952 | 0.020 | 274 | 0.994 | 0.987 | 0.006 | 0.056 | 0.057 | -0.001 |
| VIPSX vs TIP (2004-2007) | 1006 | 0.904 | 0.949 | 1.012 | 0.024 | 48 | 0.996 | 0.965 | 0.005 | 0.050 | 0.052 | -0.000 |
| VIPSX vs TIP (2008-2012) | 1259 | 0.933 | 0.972 | 0.934 | 0.028 | 60 | 0.990 | 0.997 | 0.010 | 0.074 | 0.074 | -0.001 |
| VIPSX vs TIP (2013-2026) | 3489 | 0.967 | 0.979 | 0.947 | 0.014 | 166 | 0.995 | 0.985 | 0.005 | 0.050 | 0.051 | -0.001 |
| VAIPX vs TIP (2005-07..2026) | 5377 | 0.953 | 0.976 | 0.943 | 0.019 | 256 | 0.993 | 0.988 | 0.007 | 0.056 | 0.057 | -0.000 |
| SCHP vs TIP (2010-09..2026) | 4076 | 0.978 | 0.991 | 0.976 | 0.012 | 194 | 0.998 | 0.996 | 0.003 | 0.050 | 0.050 | 0.001 |
| PRRIX_fix vs GSW_10y recon (1999-02..2003) | 1220 | 0.835 | 0.915 | 0.822 | 0.030 | 59 | 0.953 | 0.972 | 0.018 | 0.058 | 0.059 | 0.004 |
| PRRIX_fix vs GSW_7y recon (1999-02..2003) | 1220 | 0.745 | 0.861 | 0.975 | 0.033 | 59 | 0.936 | 1.190 | 0.021 | 0.058 | 0.048 | 0.007 |
| ACITX_fix vs GSW_10y recon (1999-02..2003) | 1220 | 0.834 | 0.915 | 0.803 | 0.030 | 59 | 0.964 | 0.936 | 0.016 | 0.055 | 0.059 | -0.015 |
| VIPSX vs GSW_10y recon (2000-07..2003) | 866 | 0.820 | 0.877 | 0.869 | 0.036 | 42 | 0.929 | 0.955 | 0.025 | 0.063 | 0.066 | -0.035 |
| TIP vs GSW_10y recon (2004-2024) | 5244 | 0.908 | 0.944 | 0.774 | 0.036 | 252 | 0.976 | 0.736 | 0.025 | 0.058 | 0.078 | -0.009 |

Reading: daily correlations between funds are depressed by NAV rounding to the cent (a $10 NAV has a 0.1% tick, comparable to a quiet day's move in 1997-2000) and by the ETF's price noise; 3-day and monthly correlations are the relevant check of co-movement and vol level.

## Duration profile

Empirical real duration = −β of weekly log returns on weekly changes in the GSW 7y zero real yield, per calendar year (noisy; a level check, not a precise duration).

| date | PRRIX_fix | ACITX_fix | VIPSX | TIP | SCHP | spliced |
|---|---|---|---|---|---|---|
| 1999 | 2.8 | 4.3 |  |  |  | 2.8 |
| 2000 | 7.1 | 7.2 | 6.0 |  |  | 7.1 |
| 2001 | 5.9 | 5.5 | 5.6 |  |  | 5.9 |
| 2002 | 6.0 | 5.8 | 6.0 |  |  | 6.0 |
| 2003 | 7.1 | 7.0 | 6.8 |  |  | 7.1 |
| 2004 | 7.2 | 7.4 | 7.3 | 7.4 |  | 7.4 |
| 2005 | 7.2 | 7.1 | 7.2 | 7.5 |  | 7.5 |
| 2006 | 8.0 | 7.4 | 7.5 | 7.1 |  | 7.1 |
| 2007 | 6.6 | 6.5 | 6.3 | 6.4 |  | 6.4 |
| 2008 | 6.6 | 6.0 | 6.0 | 6.5 |  | 6.5 |
| 2009 | 6.9 | 5.8 | 6.2 | 6.3 |  | 6.3 |
| 2010 | 6.7 | 6.5 | 6.4 | 6.4 |  | 6.4 |
| 2011 | 6.4 | 7.4 | 7.4 | 7.5 | 7.3 | 7.5 |
| 2012 | 7.2 | 7.2 | 7.2 | 7.5 | 7.4 | 7.5 |
| 2013 | 7.6 | 6.8 | 6.8 | 6.9 | 6.8 | 6.9 |
| 2014 | 6.4 | 6.1 | 6.7 | 6.7 | 6.7 | 6.7 |
| 2015 | 7.1 | 6.7 | 6.8 | 7.1 | 6.9 | 7.1 |
| 2016 | 6.7 | 6.9 | 6.8 | 7.1 | 7.0 | 7.1 |
| 2017 | 6.6 | 6.9 | 6.7 | 6.8 | 6.7 | 6.8 |
| 2018 | 6.7 | 6.8 | 6.9 | 7.0 | 6.9 | 7.0 |
| 2019 | 6.8 | 7.0 | 7.0 | 7.1 | 6.9 | 7.1 |
| 2020 | 8.2 | 7.5 | 7.2 | 9.0 | 8.6 | 9.0 |
| 2021 | 6.4 | 6.5 | 6.4 | 6.5 | 6.5 | 6.5 |
| 2022 | 5.9 | 6.0 | 6.0 | 6.1 | 6.0 | 6.1 |
| 2023 | 5.4 | 6.0 | 5.5 | 5.6 | 5.6 | 5.6 |
| 2024 | 6.6 | 6.6 | 6.4 | 6.5 | 6.5 | 6.5 |
| 2025 | 6.3 | 5.8 | 6.1 | 5.9 | 5.9 | 5.9 |
| 2026 | 5.8 | 5.2 | 5.3 | 5.2 | 5.3 | 5.2 |

## Annualized volatility by period (daily returns, %)

|  | tips (spliced) | PRRIX_fix | ACITX_fix | VIPSX | VAIPX | SCHP | TIP | GSW_10y recon |
|---|---|---|---|---|---|---|---|---|
| 1997 | 3.24 | 3.24 | 6.04 |  |  |  |  |  |
| 1998 | 3.65 | 3.65 | 3.87 |  |  |  |  |  |
| 1999 | 2.27 | 2.27 | 2.45 |  |  |  |  | 2.57 |
| 2000 | 2.22 | 2.56 | 2.47 | 3.24 |  |  |  | 3.10 |
| 2001 | 4.30 | 4.30 | 4.22 | 4.67 |  |  |  | 4.93 |
| 2002 | 4.56 | 4.56 | 4.15 | 5.25 |  |  |  | 5.56 |
| 2003 | 7.06 | 7.06 | 6.97 | 6.92 |  |  | 4.59 | 8.84 |
| 2004 | 6.10 | 5.85 | 6.05 | 6.62 |  |  | 6.10 | 7.40 |
| 2005 | 5.03 | 4.78 | 4.94 | 4.94 | 4.69 |  | 5.03 | 6.05 |
| 2006 | 4.52 | 4.91 | 4.55 | 4.49 | 4.49 |  | 4.52 | 5.47 |
| 2007 | 5.84 | 5.79 | 5.68 | 5.62 | 5.59 |  | 5.84 | 7.38 |
| 2008 | 11.84 | 11.48 | 10.49 | 10.34 | 10.28 |  | 11.84 | 14.37 |
| 2009 | 7.96 | 8.49 | 7.47 | 7.71 | 7.71 |  | 7.96 | 11.42 |
| 2010 | 5.76 | 5.94 | 5.97 | 5.77 | 5.76 | 6.94 | 5.76 | 7.92 |
| 2011 | 6.99 | 6.25 | 6.91 | 6.84 | 6.81 | 6.75 | 6.99 | 8.51 |
| 2012 | 4.49 | 4.43 | 4.50 | 4.50 | 4.45 | 4.54 | 4.49 | 5.80 |
| 2013 | 5.97 | 6.33 | 5.87 | 5.86 | 5.88 | 5.84 | 5.97 | 8.05 |
| 2014 | 4.51 | 4.38 | 4.21 | 4.41 | 4.37 | 4.45 | 4.51 | 6.11 |
| 2015 | 5.60 | 5.49 | 5.49 | 5.54 | 5.56 | 5.46 | 5.60 | 7.37 |
| 2016 | 4.54 | 4.56 | 4.47 | 4.54 | 4.57 | 4.49 | 4.54 | 5.97 |
| 2017 | 3.53 | 3.59 | 3.73 | 3.56 | 3.46 | 3.47 | 3.53 | 4.76 |
| 2018 | 3.08 | 3.03 | 3.00 | 3.05 | 3.02 | 2.99 | 3.08 | 4.41 |
| 2019 | 3.91 | 3.92 | 3.88 | 3.93 | 3.88 | 3.87 | 3.91 | 5.23 |
| 2020 | 8.79 | 7.97 | 7.37 | 7.09 | 7.08 | 8.27 | 8.79 | 9.03 |
| 2021 | 4.65 | 4.48 | 4.39 | 4.53 | 4.53 | 4.54 | 4.65 | 6.57 |
| 2022 | 9.36 | 8.89 | 9.01 | 8.77 | 8.83 | 9.25 | 9.36 | 12.86 |
| 2023 | 6.99 | 6.83 | 7.26 | 6.81 | 6.78 | 6.83 | 6.99 | 9.94 |
| 2024 | 4.56 | 4.62 | 4.61 | 4.52 | 4.50 | 4.59 | 4.56 | 6.86 |
| 2025 | 4.35 | 4.64 | 5.10 | 4.30 | 4.30 | 4.28 | 4.35 | 6.96 |
| 2026 | 3.75 | 4.24 | 3.60 | 3.74 | 3.72 | 3.68 | 3.75 |  |
| 2008-09..2008-12 (Lehman / TIPS liquidity) | 16.65 | 15.54 | 13.48 | 13.73 | 13.62 |  | 16.65 | 19.71 |
| 2013-05..2013-09 (taper tantrum) | 7.65 | 8.31 | 7.49 | 7.52 | 7.53 | 7.44 | 7.65 | 9.97 |
| 2020-03 (COVID dash for cash) | 26.88 | 24.20 | 21.74 | 21.02 | 21.15 | 25.39 | 26.88 | 26.21 |
| 2022 (inflation / hiking) | 9.36 | 8.89 | 9.01 | 8.77 | 8.83 | 9.25 | 9.36 | 12.86 |

Monthly-return vol of the spliced series by 5-year block (%): 1997-1999: 2.60, 2000-2004: 6.23, 2005-2009: 7.44, 2010-2014: 5.16, 2015-2019: 3.54, 2020-2024: 6.39, 2025-2026: 4.01

## Predictors: `predictors/tips_real_yields.parquet`

Long format `obs_date, available_date, series, value`. Series: `real_yield_5y` (GSW `TIPSY05`), `real_yield_10y` (GSW `TIPSY10`), `breakeven_5y` (GSW `BKEVEN05`), `breakeven_10y` (GSW `BKEVEN10`). Values in PERCENT, zero-coupon, continuously compounded, as published by the Fed. `available_date = obs_date + 10 business days`: the Fed posts this staff research file roughly weekly with about a one-week lag (on 2026-11-14 the file's last observation was 2026-11-06), so two weeks is a conservative bound. Rows: 27828, 1999-01-04 .. 2026-11-06. The file is the current vintage (the Fed may refit history), and 1999-2003 fits rest on few TIPS issues and are noisy (e.g. 1999-01-06 2y breakeven jumps to 2.05%).

## Caveats

* 1997-2003 is an ACTIVE fund (PIMCO Real Return): mostly TIPS, but with some nominal, foreign linker and derivative exposure; its 2004+ monthly vol ratio vs TIP (table above) shows it ran somewhat more risk than the index ETF. ACITX (closer to a plain TIPS fund) has bad one-day NAV prints in 1997-98 and was rejected for the daily series.
* The accrual re-spread assumes income accrues evenly per calendar day within the month. Real accruals vary with the CPI index-ratio path; the error is a few bp per day at most and does not affect monthly returns.
* NAV quantization: 1997-2000 fund NAVs move in 0.1% ticks; ~35-40% of days show zero NAV change. Daily variance in that era is noisy (rounding adds roughly tick²/6 per day).
* TIP is an ETF market price: in 2008 Q4 it traded at sizeable discounts/premia to NAV, so its daily vol then exceeds the Vanguard fund's NAV vol (see episode table). This is investable reality, not an error.
* Calendar-year TR check vs Vanguard IPS (VIPSX, NAV): TIP − VIPSX is within ±0.4pp in most years, but +2.3pp in 2008 and −1.9pp in 2009 (TIP market price ended 2008 rich vs NAV; the 2-year totals agree within ~0.5pp), and +1.4pp in 2016 where SCHP agrees with TIP (VIPSX is missing Yahoo dividends that year). TIP 2013 −8.49%, 2022 −12.24%.
* VIPSX on Yahoo misses most 2000-2002 dividends; its early overlap numbers understate its return (ret diff) but its vol/corr are still informative.
* GSW reconstruction (constant-maturity zero, real return × CPI index ratio) validates the fund era at the monthly level only; daily correlations are low because early curve fits are noisy. It is not spliced in (funds cover 1997+ at a higher tier).
* Zero days: the spliced daily series includes days with exactly 0 return (NAV unchanged); these are real observations, not fills.
