# How should the allocation be set?

*Descriptive allocation study; it replays weighting rules on a history that was all seen before the rules were written, selects nothing and is not out-of-sample evidence.*

Written by `python -m assetforecast.allocation`; returns through 2026-10-31. Four holdings: world stocks and the diversifier study's sleeve (intermediate Treasuries, gold, managed futures). Each weighting rule was fixed before any was run, sees only what was known at each month-end, and is held over the next month, from 1990-01. Dashboard: `dashboard/allocation.html`.

## The reading

**Expected returns: do not forecast them, and do not optimize on them.** A holding's return in one five-year block said nothing about the next (correlation +0.01 over 47 pairs of blocks). Gold's return per unit of risk was -0.18 in the first half of the window and +0.45 in the second; managed futures went from +1.28 to +0.22. Since the live funds began, all three optimizers trailed a third each (best of them -0.09 in Sharpe ratio, none clearly different).

**Weights inside the sleeve: size by long-run risk, or simply a third each.** Risk does carry over: a block's volatility against the next, correlation +0.60. Weighting by 1/volatility on the whole past ran at 5.7% volatility against 6.8% for a third each, with a worst fall of -9% against -14%. Its Sharpe ratio was +0.10 higher over the whole replay (better) and -0.01 since the live funds began (no clear difference). The forecast gave the same weights for 12 times the trading.

**How much risk in total: your choice; use the forecast to know what you chose.** No input picks the stock share: it is the risk you accept. Scaling the 60/40 mix to a 10% risk target with the forecast gave the same Sharpe ratio as leaving it fixed (0.57 against 0.57) with a worst fall of -25% against -32%; scaling with past volatility was worse than not scaling (-0.03, worse). The forecast's 12-month matrix stood +23% above month-to-month volatility, so a target set with it runs below the target.

So the answer is a mix, split by input: nothing from return forecasts, long-run history for the weights inside the sleeve, and the covariance forecast for the size of the total risk. These verdicts are a reading of the tables below, on years that had all been seen.

## Can each input be known in advance?

One block of 60 months against the next, non-overlapping, on each holding's whole history:

| Holding | From | Pairs of blocks | Volatility | Mean return | Return per unit of risk |
|---|---|---|---|---|---|
| World stocks | 1926 | 19 | +0.59 | -0.22 | +0.14 |
| Intermediate Treasuries | 1966 | 11 | +0.09 | -0.16 | +0.05 |
| Gold | 1971 | 10 | +0.68 | +0.13 | +0.15 |
| Managed futures | 1986 | 7 | -0.00 | +0.60 | +0.64 |
| All four, pooled |  | 47 | +0.60 | +0.01 | +0.21 |

The pooled row removes each holding's own average first. Managed futures' return looks persistent because its blocks before 2007 are the paper factor.

Historical means on the common window (1985 on), with the package's 90% interval:

| Holding | Mean a year | 90% interval | Vol | Return per unit of risk | Its interval | To 2005-11 | After |
|---|---|---|---|---|---|---|---|
| World stocks | +7.1% | +3.3% to +10.9% | 15.2% | +0.47 | +0.22 to +0.72 | +0.50 | +0.44 |
| Intermediate Treasuries | +2.9% | +0.6% to +5.2% | 6.7% | +0.43 | +0.09 to +0.77 | +0.64 | +0.22 |
| Gold | +2.7% | -3.4% to +8.8% | 15.4% | +0.17 | -0.22 to +0.57 | -0.18 | +0.45 |
| Managed futures | +5.5% | +3.0% to +8.0% | 7.8% | +0.71 | +0.39 to +1.03 | +1.28 | +0.22 |

## How well each source measured the next 12 months' risk

Volatility: QLIKE loss (lower is better), the forecast's level over the realized level, and the correlation of forecast with realized.

| Source | World stocks: QLIKE | Intermediate Treasuries: QLIKE | Gold: QLIKE | 60/40 mix: QLIKE | 60/40 mix: level | 60/40 mix: correlation |
|---|---|---|---|---|---|---|
| Whole past | 0.307 | 0.173 | 0.260 | 0.220 | 1.02 | -0.19 |
| Last 5 years | 0.510 | 0.212 | 0.255 | 0.335 | 1.02 | -0.09 |
| Forecast, 12-month matrix | 0.305 | 0.205 | 0.282 | 0.223 | 1.23 | +0.32 |
| Forecast, 21-day matrix | 0.276 | 0.162 | 0.239 | 0.186 | 1.11 | +0.32 |

Correlations: mean squared error against the realized 12-month correlation.

| Source | Stocks and Treasuries | Stocks and gold | Treasuries and gold |
|---|---|---|---|
| Whole past | 0.236 | 0.114 | 0.127 |
| Last 5 years | 0.202 | 0.113 | 0.105 |
| Forecast, 12-month matrix | 0.200 | 0.096 | 0.100 |
| Forecast, 21-day matrix | 0.200 | 0.096 | 0.100 |

The forecast does not cover managed futures. The 21-day row was added after the first run showed the 12-month matrix standing above month-to-month volatility; no rule uses it.

## The sleeve: ten ways to weight Treasuries, gold and managed futures

1990-01 to 2026-10 (442 months):

| Rule | Growth over T-bills | Vol | Sharpe | Worst fall | Sharpe against the first row (90% interval) | Turnover a year | Average held: Treasuries / gold / managed futures |
|---|---|---|---|---|---|---|---|
| A third each | +3.9% | 6.8% | 0.57 | -14% |  | 0% | 33% / 33% / 33% |
| 1/volatility, whole past | +3.8% | 5.7% | 0.66 | -9% | +0.10 (+0.01 to +0.18) | 1% | 41% / 20% / 38% |
| 1/volatility, last 5 years | +3.7% | 5.6% | 0.67 | -8% | +0.10 (+0.01 to +0.19) | 6% | 44% / 20% / 37% |
| 1/volatility, forecast | +3.9% | 5.5% | 0.72 | -9% | +0.15 (+0.04 to +0.26) | 11% | 41% / 17% / 41% |
| Least variance, whole past | +3.4% | 5.2% | 0.65 | -8% | +0.08 (-0.06 to +0.22) | 3% | 48% / 14% / 38% |
| Least variance, last 5 years | +2.9% | 5.1% | 0.57 | -8% | +0.00 (-0.19 to +0.20) | 15% | 58% / 8% / 34% |
| Least variance, forecast | +3.6% | 5.2% | 0.70 | -8% | +0.13 (-0.08 to +0.34) | 29% | 47% / 6% / 47% |
| Optimized: past returns, past risk | +4.2% | 5.8% | 0.72 | -10% | +0.15 (-0.09 to +0.40) | 10% | 33% / 2% / 66% |
| Optimized: last 5 years of both | +5.5% | 7.4% | 0.75 | -14% | +0.18 (+0.01 to +0.38) | 59% | 33% / 14% / 53% |
| Optimized: past returns, forecast risk | +4.6% | 6.0% | 0.76 | -10% | +0.19 (-0.06 to +0.45) | 23% | 28% / 0% / 72% |

2007-03 to 2026-10 (236 months):

| Rule | Growth over T-bills | Vol | Sharpe | Worst fall | Sharpe against the first row (90% interval) | Turnover a year | Average held: Treasuries / gold / managed futures |
|---|---|---|---|---|---|---|---|
| A third each | +3.8% | 7.5% | 0.51 | -14% |  | 0% | 33% / 33% / 33% |
| 1/volatility, whole past | +3.0% | 5.9% | 0.50 | -9% | -0.01 (-0.11 to +0.09) | 1% | 43% / 19% / 38% |
| 1/volatility, last 5 years | +3.0% | 5.8% | 0.52 | -8% | +0.01 (-0.10 to +0.12) | 6% | 46% / 18% / 36% |
| 1/volatility, forecast | +2.9% | 5.6% | 0.51 | -9% | +0.00 (-0.11 to +0.12) | 12% | 43% / 17% / 40% |
| Least variance, whole past | +2.3% | 5.2% | 0.45 | -8% | -0.06 (-0.27 to +0.13) | 2% | 52% / 9% / 38% |
| Least variance, last 5 years | +1.8% | 4.9% | 0.36 | -8% | -0.15 (-0.46 to +0.14) | 13% | 63% / 1% / 36% |
| Least variance, forecast | +1.8% | 5.0% | 0.36 | -8% | -0.15 (-0.42 to +0.12) | 27% | 53% / 2% / 45% |
| Optimized: past returns, past risk | +1.6% | 5.5% | 0.28 | -10% | -0.22 (-0.52 to +0.07) | 9% | 38% / 3% / 60% |
| Optimized: last 5 years of both | +3.3% | 8.0% | 0.42 | -14% | -0.09 (-0.25 to +0.06) | 74% | 47% / 23% / 30% |
| Optimized: past returns, forecast risk | +1.5% | 5.5% | 0.27 | -10% | -0.24 (-0.56 to +0.07) | 25% | 40% / 0% / 60% |

## The whole portfolio

Scaled rules aim at 10% volatility, exposure capped at 2 times capital, borrowing at T-bills + 1.38%.

1990-01 to 2026-10 (442 months):

| Rule | Growth over T-bills | Vol | Sharpe | Worst fall | Sharpe against the first row (90% interval) | Turnover a year | Average held: stocks / Treasuries / gold / managed futures | Average exposure |
|---|---|---|---|---|---|---|---|---|
| Fixed 60% stocks / 40% sleeve | +5.4% | 9.5% | 0.57 | -32% |  | 0% | 60% / 13% / 13% / 13% | 100% |
| 60/40 scaled to 10% risk, whole past | +5.6% | 10.3% | 0.55 | -35% | -0.03 (-0.04 to -0.01) | 2% | 65% / 14% / 14% / 14% | 108% |
| 60/40 scaled to 10% risk, last 5 years | +5.4% | 10.7% | 0.51 | -39% | -0.06 (-0.11 to -0.01) | 9% | 67% / 15% / 15% / 15% | 112% |
| 60/40 scaled to 10% risk, forecast | +4.7% | 8.3% | 0.57 | -25% | -0.00 (-0.05 to +0.05) | 14% | 55% / 12% / 12% / 12% | 92% |
| 1/volatility over all four, forecast | +6.5% | 8.8% | 0.73 | -13% | +0.16 (-0.13 to +0.44) | 26% | 27% / 60% / 25% / 60% | 173% |
| Optimized: past returns, past risk | +6.6% | 9.4% | 0.71 | -14% | +0.14 (-0.18 to +0.45) | 20% | 26% / 49% / 2% / 100% | 177% |
| Optimized: last 5 years of both | +8.0% | 10.6% | 0.75 | -21% | +0.18 (-0.11 to +0.45) | 99% | 35% / 52% / 11% / 73% | 171% |
| Optimized: past returns, forecast risk | +7.9% | 10.1% | 0.78 | -14% | +0.21 (-0.13 to +0.55) | 47% | 25% / 46% / 0% / 111% | 182% |

2007-03 to 2026-10 (236 months):

| Rule | Growth over T-bills | Vol | Sharpe | Worst fall | Sharpe against the first row (90% interval) | Turnover a year | Average held: stocks / Treasuries / gold / managed futures | Average exposure |
|---|---|---|---|---|---|---|---|---|
| Fixed 60% stocks / 40% sleeve | +5.9% | 10.2% | 0.57 | -32% |  | 0% | 60% / 13% / 13% / 13% | 100% |
| 60/40 scaled to 10% risk, whole past | +5.9% | 10.9% | 0.54 | -35% | -0.03 (-0.05 to -0.01) | 1% | 64% / 14% / 14% / 14% | 106% |
| 60/40 scaled to 10% risk, last 5 years | +5.1% | 11.2% | 0.46 | -39% | -0.11 (-0.18 to -0.02) | 10% | 64% / 14% / 14% / 14% | 107% |
| 60/40 scaled to 10% risk, forecast | +4.6% | 8.4% | 0.54 | -25% | -0.03 (-0.10 to +0.04) | 16% | 53% / 12% / 12% / 12% | 88% |
| 1/volatility over all four, forecast | +5.0% | 9.1% | 0.54 | -13% | -0.03 (-0.41 to +0.31) | 27% | 27% / 63% / 24% / 59% | 173% |
| Optimized: past returns, past risk | +3.3% | 9.4% | 0.35 | -14% | -0.22 (-0.63 to +0.14) | 16% | 30% / 61% / 3% / 93% | 187% |
| Optimized: last 5 years of both | +4.5% | 10.5% | 0.43 | -21% | -0.14 (-0.45 to +0.13) | 102% | 45% / 75% / 18% / 32% | 169% |
| Optimized: past returns, forecast risk | +3.2% | 9.6% | 0.34 | -14% | -0.24 (-0.71 to +0.21) | 52% | 25% / 71% / 0% / 94% | 191% |

## The menu today

Stocks beside the sleeve at origin 2026-10-31. "Forecast vol" is on the forecast's 12-month matrix; the replay columns are the whole replay and, in brackets, since the live funds began.

Sleeve: 1/volatility, whole past.

| Stocks | Treasuries | Gold | Managed futures | Borrowed | Forecast vol | Replay growth | Replay vol | Replay worst fall |
|---|---|---|---|---|---|---|---|---|
| 40% | 26% | 11% | 22% | 0% | 7.4% | +4.9% (+4.8%) | 6.8% (7.3%) | -20% (-20%) |
| 60% | 17% | 8% | 15% | 0% | 9.9% | +5.4% (+5.5%) | 9.2% (9.9%) | -32% (-32%) |
| 80% | 9% | 4% | 7% | 0% | 12.8% | +5.7% (+6.1%) | 12.1% (12.9%) | -44% (-44%) |
| 100% | 0% | 0% | 0% | 0% | 15.9% | +5.9% (+6.6%) | 15.1% (16.1%) | -54% (-54%) |
| 100% | 22% | 9% | 19% | 50% | 16.3% | +7.1% (+7.5%) | 15.3% (16.4%) | -51% (-51%) |
| 100% | 44% | 19% | 37% | 100% | 17.1% | +8.3% (+8.2%) | 16.0% (17.1%) | -48% (-48%) |

Sleeve: a third each.

| Stocks | Treasuries | Gold | Managed futures | Borrowed | Forecast vol | Replay growth | Replay vol | Replay worst fall |
|---|---|---|---|---|---|---|---|---|
| 40% | 20% | 20% | 20% | 0% | 8.7% | +5.0% (+5.3%) | 7.3% (8.0%) | -20% (-20%) |
| 60% | 13% | 13% | 13% | 0% | 10.5% | +5.4% (+5.9%) | 9.5% (10.2%) | -32% (-32%) |
| 80% | 7% | 7% | 7% | 0% | 13.0% | +5.7% (+6.3%) | 12.2% (13.0%) | -44% (-44%) |
| 100% | 0% | 0% | 0% | 0% | 15.9% | +5.9% (+6.6%) | 15.1% (16.1%) | -54% (-54%) |
| 100% | 17% | 17% | 17% | 50% | 16.9% | +7.2% (+7.9%) | 15.6% (16.7%) | -51% (-51%) |
| 100% | 33% | 33% | 33% | 100% | 19.0% | +8.3% (+9.0%) | 16.7% (18.1%) | -47% (-47%) |

## Caveats

- Every year in the replay had been seen before the rules were written, and the three sleeve assets were chosen on the same years. Nothing here is out-of-sample.
- Managed futures before 2007-03 are a paper factor scaled to the live funds' volatility less a gap measured on 2007-2026. Rules that read managed futures' past return or risk before 2007 read numbers nobody had. The live-funds period is the one to trust, and it is short.
- Differences in Sharpe ratio between rules are small beside their intervals. "No clear difference" is the common result, not a failure of the test.
- No trading costs, fund fees or taxes. Borrowing is at today's spread in every year. Month-end values miss falls inside a month, and a margin loan can be called before a month ends.
- The rules were run once as written. The one addition after seeing results is the 21-day row of the risk table.
