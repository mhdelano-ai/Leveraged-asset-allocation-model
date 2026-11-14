# Implied-volatility predictors (`implied` track)

Built by `python -m volmodel.data.sources.implied` -> `data/processed/predictors/implied.parquet` (long: `obs_date, available_date, series, value`; end-of-day closes, so `available_date = obs_date`). Newest close in the file: **2026-11-13**. A series is *live* if its last close is within 10 calendar days of that.

Units: all vol indices are **annualized vol in decimals** (VIX 20 -> 0.20). Exceptions: `iv_move` is a **yield (bp) vol in decimals** (MOVE 100 -> 0.0100; x modified duration ~ price vol); `iv_skew_index` is the **SKEW level** (not a vol); `*_term_slope` are ratios minus one; `iv_vvix` is the vol of VIX.

## Series

| series | description | unit | start | end | n | live | median | min | max | source | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `iv_gvz` | Cboe gold ETF (GLD) vol (GVZ) | decimal vol | 2009-09-18 | 2026-11-13 | 4314 | live | 0.1713 | 0.0888 | 0.4898 | cboe:GVZ_History.csv |  |
| `iv_ovx` | Cboe crude oil ETF (USO) vol (OVX) | decimal vol | 2009-09-18 | 2026-11-13 | 4313 | live | 0.3501 | 0.145 | 3.252 | cboe:OVX_History.csv | USO options; peaked at 3.25 on 2020-04-21 (negative WTI front month). bad prints dropped: 2011-11-25. |
| `iv_rvx` | Cboe Russell 2000 vol (RVX) | decimal vol | 2009-09-16 | 2026-11-13 | 4313 | live | 0.2206 | 0.1183 | 0.8319 | cboe:RVX_History.csv |  |
| `iv_skew_index` | Cboe S&P 500 SKEW index | index level (NOT a vol) | 1990-01-02 | 2026-11-13 | 9270 | live | 120 | 101.2 | 183.1 | cboe:SKEW_History.csv | 100 - 10 x risk-neutral skewness of 30-day SPX returns. |
| `iv_vix` | Cboe S&P 500 30-day vol (VIX) | decimal vol | 1990-01-02 | 2026-11-13 | 9316 | live | 0.1756 | 0.0914 | 0.8269 | cboe:VIX_History.csv | Sep-2003 redefinition (SPX full strip, model-free) back-calculated to 1990; the pre-2003 'old VIX' is VXO. |
| `iv_vix1y` | Cboe S&P 500 1-year vol (VIX1Y) | decimal vol | 2007-01-03 | 2026-11-13 | 4993 | live | 0.2257 | 0.1331 | 0.5353 | cboe:VIX1Y_History.csv | Close-only (O=H=L=C) until ~2018-04: early history is retrospectively calculated. |
| `iv_vix1y_term_slope` | VIX1Y / VIX - 1 (same-day closes) | ratio - 1 | 2007-01-03 | 2026-11-13 | 4993 | live | 0.2934 | -0.5182 | 0.8763 | derived |  |
| `iv_vix3m` | Cboe S&P 500 3-month vol (VIX3M, formerly VXV) | decimal vol | 2009-09-18 | 2026-11-13 | 4316 | live | 0.1922 | 0.1185 | 0.7298 | cboe:VIX3M_History.csv | Renamed VXV -> VIX3M in 2017; same index. |
| `iv_vix6m` | Cboe S&P 500 6-month vol (VIX6M, formerly VXMT) | decimal vol | 2008-01-02 | 2026-11-13 | 4748 | live | 0.2147 | 0.1375 | 0.6147 | cboe:VIX6M_History.csv | Renamed VXMT -> VIX6M in 2017; close-only (O=H=L=C) until ~2013-12, consistent with retrospective calculation. |
| `iv_vix9d` | Cboe S&P 500 9-day vol (VIX9D) | decimal vol | 2011-01-04 | 2026-11-13 | 3990 | live | 0.1526 | 0.071 | 1.067 | cboe:VIX9D_History.csv |  |
| `iv_vix_term_slope` | VIX3M / VIX - 1 (same-day closes) | ratio - 1 | 2009-09-18 | 2026-11-13 | 4316 | live | 0.1307 | -0.2558 | 0.4077 | derived |  |
| `iv_vix_vxo_spliced` | VIX from 1990-01-02; before that VXO x k, k = exp(mean log(VIX/VXO)) over 1990-1994 | decimal vol | 1986-01-02 | 2026-11-13 | 10326 | live | 0.181 | 0.0914 | 1.513 | derived | k = 1.0074 (1990-94 overlap, n=1261, corr 0.983); spliced VXO part 1986-01-02..1989-12-29. |
| `iv_vvix` | Cboe VIX of VIX (VVIX) | decimal vol of VIX | 2006-06-01 | 2026-11-13 | 5126 | live | 0.9092 | 0.5701 | 2.076 | cboe:VVIX_History.csv | 2006-03..05 dropped: 3-8 prints a month, levels 16-92 vs ~90 after (start-up noise). bad prints dropped: 2006-11-13. 3 stale repeats dropped. |
| `iv_vxd` | Cboe DJIA vol (VXD) | decimal vol | 2009-09-18 | 2026-11-13 | 4300 | live | 0.1568 | 0.0758 | 0.6707 | cboe:VXD_History.csv | bad prints dropped: 2021-07-13, 2022-05-04, 2023-12-28, 2024-01-12, 2024-03-21, 2024-09-17, 2024-09-25. 9 stale repeats dropped. |
| `iv_vxeem` | Cboe EEM ETF vol (VXEEM) | decimal vol | 2011-03-16 | 2026-11-13 | 3933 | live | 0.2151 | 0.1311 | 0.9246 | cboe:VXEEM_History.csv | bad prints dropped: 2021-04-22, 2021-06-23, 2021-08-10, 2022-03-28. |
| `iv_vxefa` | Cboe EFA ETF vol (VXEFA) | decimal vol | 2008-01-02 | 2026-11-13 | 4734 | live | 0.1824 | 0.0762 | 0.8102 | cboe:VXEFA_History.csv | Close-only until ~2013-07 (retrospective). bad prints dropped: 2018-06-13, 2018-09-19, 2019-05-23, 2019-08-14, 2019-10-02, 2019-10-15, 2020-11-09, 2021-10-11, 2021-12-17, 2022-04-13. |
| `iv_vxhyg` | Cboe HY corporate ETF (HYG) vol (VXHYG) | decimal vol | 2015-04-22 | 2026-11-13 | 1931 | live | 0.0775 | 0.04 | 0.5926 | cboe:VXHYG_History.csv | Noisy; one garbage print (-19,738,470 on 2017-10-30) removed; no data 2021-09-24..2025-05-14. bad prints dropped: 2017-08-30, 2017-12-06, 2017-12-11, 2018-02-14, 2018-02-15, 2018-05-03, 2018-05-10, 2018-05-11, 2018-06-13, 2018-10-08, 2018-11-23, 2018-12-07, 2019-02-05, 2025-08-18, 2025-08-20. 12 stale repeats dropped. 1 non-positive dropped. |
| `iv_vxief` | Cboe 7-10yr Treasury ETF (IEF) vol (VXIEF) | decimal vol | 2015-04-22 | 2026-11-13 | 1926 | live | 0.0655 | 0.0364 | 0.2523 | cboe:VXIEF_History.csv | Thin: long stale runs (e.g. one value carried 2020-07..2020-12, dropped); no data 2022-11-08..2025-07-13. bad prints dropped: 2018-02-01, 2018-02-20, 2018-06-11, 2018-11-27, 2018-12-17, 2019-06-19, 2019-07-16, 2019-08-26, 2021-04-26, 2021-05-05. 288 stale repeats dropped. |
| `iv_vxlqd` | Cboe IG corporate ETF (LQD) vol (VXLQD) | decimal vol | 2020-11-27 | 2026-11-13 | 828 | live | 0.08625 | 0.0542 | 0.2361 | cboe:VXLQD_History.csv | Starts 2020-11; no data 2022-11-08..2025-07-13. bad prints dropped: 2021-01-13. |
| `iv_vxn` | Cboe Nasdaq-100 vol (VXN) | decimal vol | 2009-09-14 | 2026-11-13 | 4322 | live | 0.1991 | 0.1031 | 0.8008 | cboe:VXN_History.csv |  |
| `iv_vxo` | Cboe S&P 100 30-day ATM vol (VXO, the pre-2003 'old VIX') | decimal vol | 1986-01-02 | 2021-01-07 | 8819 | **dead** | 0.1822 | 0.0632 | 1.502 | cboe:vxoarchive.xls+vxocurrent.csv | Old methodology: OEX at-the-money, TRADING-day convention (inflates level vs calendar-day VIX). Discontinued by Cboe in 2021. |
| `iv_vxtlt` | Cboe 20+yr Treasury ETF (TLT) vol (VXTLT) | decimal vol | 2004-01-02 | 2026-11-13 | 5742 | live | 0.1413 | 0.0661 | 0.529 | cboe:VXTLT_History.csv |  |

Not built (no data obtainable here): `iv_move`.


## Cboe vs Yahoo on the overlap

Levels in published index points. Yahoo is used to extend a Cboe series only when it agrees (n >= 250, corr > 0.99, median |diff| < 2% of the median level).

| series | Yahoo symbol | Yahoo range | overlap n | corr | mean diff | median abs diff | share abs diff > 0.1 | used to extend |
|---|---|---|---|---|---|---|---|---|
| `iv_vix9d` | ^VIX9D | unavailable |  |  |  |  |  | no |
| `iv_vix` | ^VIX | unavailable |  |  |  |  |  | no |
| `iv_vix3m` | ^VIX3M | unavailable |  |  |  |  |  | no |
| `iv_vix6m` | ^VIX6M | unavailable |  |  |  |  |  | no |
| `iv_vix1y` | ^VIX1Y | unavailable |  |  |  |  |  | no |
| `iv_vxo` | ^VXO | unavailable |  |  |  |  |  | no |
| `iv_vxn` | ^VXN | unavailable |  |  |  |  |  | no |
| `iv_rvx` | ^RVX | unavailable |  |  |  |  |  | no |
| `iv_vxd` | ^VXD | unavailable |  |  |  |  |  | no |
| `iv_vxefa` | ^VXEFA | unavailable |  |  |  |  |  | no |
| `iv_vxeem` | ^VXEEM | unavailable |  |  |  |  |  | no |
| `iv_vxtlt` | ^VXTLT | unavailable |  |  |  |  |  | no |
| `iv_move` | ^MOVE | unavailable |  |  |  |  |  | no |
| `iv_gvz` | ^GVZ | unavailable |  |  |  |  |  | no |
| `iv_ovx` | ^OVX | unavailable |  |  |  |  |  | no |
| `iv_vvix` | ^VVIX | unavailable |  |  |  |  |  | no |
| `iv_skew_index` | ^SKEW | unavailable |  |  |  |  |  | no |

## Mapping to the 12 assets

| asset | natural implied predictor | from | live | comment |
|---|---|---|---|---|
| us_large | `iv_vix1y` | 2007-01-03 | live | 1-yr point matches the horizon; iv_vix / iv_vix_vxo_spliced for long history |
| us_small | `iv_rvx` | 2009-09-16 | live | 30-day only; no small-cap term structure |
| intl_dev | `iv_vxefa` | 2008-01-02 | live | EFA ETF options (dev ex-US/Canada) |
| em | `iv_vxeem` | 2011-03-16 | live | EEM ETF options |
| ust_int | `iv_vxief` | 2015-04-22 | live | yield vol x duration (~6.5) ~ price vol; iv_vxief is thin/gappy |
| ust_long | `iv_vxtlt` | 2004-01-02 | live | TLT ETF options; iv_move x duration (~16) as alternative |
| tips | none | | | no TIPS implied vol exists; proxy with iv_move / iv_vxief |
| ig_credit | `iv_vxlqd` | 2020-11-27 | live | short and gappy; rates gauges (iv_move) are the practical proxy |
| hy_credit | `iv_vxhyg` | 2015-04-22 | live | noisy, long gap 2021-2025; equity vol (iv_vix, iv_rvx) is the practical proxy |
| reits | none | | | no REIT implied vol index; iv_rvx / iv_vix as proxies |
| gold | `iv_gvz` | 2009-09-18 | live | GLD ETF options |
| commodities | `iv_ovx` | 2009-09-18 | live | crude only (USO options); energy-heavy indices like DBC/BCOM partly |

## Methodology breaks and caveats

* **VIX redefinition (2003-09-22).** Cboe switched VIX from the S&P 100
  at-the-money, trading-day formula to the S&P 500 model-free (full strip,
  calendar-day) formula and back-calculated the new VIX to 1990. The old index
  was renamed **VXO**. `iv_vix` is the new definition throughout; `iv_vxo` is
  the old one (1986-2021). VXO's trading-day convention inflates its level
  relative to a calendar-day measure, while VIX's full strip adds the skew
  premium; on 1990-2003 the two largely offset (VXO/VIX ~ 1.04 on average,
  corr ~ 0.98). VXO prints 1.50 on 1987-10-19: kept, it is a real close.
* **`iv_vix_vxo_spliced`** is VIX from 1990, extended to 1986 with VXO x k.
  It exists only for the 1986-1989 pre-history (it includes the 1987 crash);
  it is a level rescale, not a re-computation, and should be used with that
  in mind.
* **Retrospective history.** VIX1Y, VIX6M, VXEFA and VIX9D have long stretches
  where open = high = low = close, i.e. only a daily close was computed
  (from historical option quotes) before live dissemination. They are not
  look-ahead (computed from data available on each day) but the index did not
  exist in real time; VIX1Y in particular is close-only until ~2018-04.
* **Cboe CDN history starts.** The CDN files for VIX3M, VXN, RVX, VXD, GVZ,
  OVX start in Sep 2009 even though the indices were computed earlier
  (VXV/VIX3M from 2007-12, VXN 2001, RVX 2004, VXD 1997, GVZ 2008-06,
  OVX 2007-05). Earlier history is taken from Yahoo only where Yahoo matches
  Cboe on the overlap (see table above). VXN was originally computed with
  the old VXO-style formula and later restated by Cboe with the VIX formula;
  whether Yahoo's pre-2009 VXN is the restated series could not be verified.
* **ETF vol indices (VXIEF, VXLQD, VXHYG).** Thinly traded underlying options:
  noisy, stale runs, and a multi-year hole (Cboe stopped and later resumed
  calculating them: VXHYG has no data 2021-09..2025-05, VXIEF/VXLQD 2022-11..2025-07). Treat as live but
  unreliable; do not use them to define an estimation sample.
* **MOVE** is a normal (yield) vol of 1-month options on 2/5/10/30y
  Treasuries; it is not a price vol and its scale differs from VXTLT.
* **OVX** spiked to 3.25 on 2020-04-21 (negative WTI): genuine, kept.
* **Holidays.** Cboe files contain trading days only (no stale holiday rows);
  missing business days per year match the NYSE holiday count. Yahoo rows with
  null closes are dropped. Nothing is filled or interpolated.
* **Not obtainable here.** EVZ (euro FX vol), VXV/VXMT/VXST legacy files,
  TYVIX/VXTYN (10-yr note vol, discontinued 2020), SRVIX (swaption vol) are
  403 on the Cboe CDN; Yahoo results for these are in the overlap table if any.

