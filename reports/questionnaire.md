# Allocation questionnaire

*Arithmetic under declared rules on the diversifier and allocation studies and the published forecast; the rules are judgements, not advice, and nothing here is out-of-sample evidence.*

Written by `python -m assetforecast.questionnaire`; origin 2026-10-31. Two answers, the account's tax treatment and the goal, give a stock share; the rest goes to the diversifier sleeve (intermediate Treasuries, gold, managed futures). Interactive page: `dashboard/questionnaire.html`.

## The rule

1. Each goal declares the largest fall from a previous high it can sit through. The stock share is the largest, in steps of 5%, whose worst fall in the replay stayed within it, with the rest in the sleeve.
2. The sleeve is weighted by 1/volatility, whole past: today 44% Treasuries, 19% gold, 37% managed futures.
3. At 100% stocks nothing is left to backfill, so the sleeve goes on top, 50% of capital: on a margin loan in a taxable account (T-bills + 1.38%), through stacked funds in a retirement account, where no loan is allowed.
4. Money needed within 3 years is outside the model: T-bills.

| Goal | Fall it can sit through | Stock share |
|---|---|---|
| Spending within 3 years |  | T-bills |
| Spending in 3 to 10 years | 20% | 40% |
| Retirement income, drawn now | 35% | 65% |
| Growth, untouched for 10 years or more | 55% | 100% |

## Every answer

Forecast vol and expected growth are on the four-holding frontier's inputs; growth is over T-bills, with the 90% interval of the historical means. The last column is the frontier portfolio at the same volatility, with the account's financing, less the answer.

| Account | Goal | Answer | Stocks / Treasuries / gold / managed futures | Forecast vol | Expected growth (90%) | Replay growth | Replay worst fall | Frontier at the same risk, less this (range) |
|---|---|---|---|---|---|---|---|---|
| Taxable brokerage | Spending within 3 years | T-bills | 0% / 0% / 0% / 0% | 0.0% |  |  |  |  |
| Taxable brokerage | Spending in 3 to 10 years | 40% stocks, 60% diversifiers | 40% / 26% / 11% / 22% | 7.4% | +4.5% (+1.6% to +7.4%) | +4.9% | -20% | +0.2% (-0.9% to +1.4%) |
| Taxable brokerage | Retirement income, drawn now | 65% stocks, 35% diversifiers | 65% / 15% / 7% / 13% | 10.6% | +5.4% (+2.3% to +8.5%) | +5.5% | -35% | +0.7% (-1.3% to +2.7%) |
| Taxable brokerage | Growth, untouched for 10 years or more | 100% stocks, plus 50% diversifiers on a margin loan | 100% / 22% / 9% / 19% | 16.3% | +7.0% (+2.3% to +11.7%) | +7.1% | -51% | +0.7% (-1.1% to +2.5%) |
| Tax-deferred | Spending within 3 years | T-bills | 0% / 0% / 0% / 0% | 0.0% |  |  |  |  |
| Tax-deferred | Spending in 3 to 10 years | 40% stocks, 60% diversifiers | 40% / 26% / 11% / 22% | 7.4% | +4.5% (+1.6% to +7.4%) | +4.9% | -20% | +1.3% (-1.2% to +3.9%) |
| Tax-deferred | Retirement income, drawn now | 65% stocks, 35% diversifiers | 65% / 15% / 7% / 13% | 10.6% | +5.4% (+2.3% to +8.5%) | +5.5% | -35% | +2.0% (+0.0% to +4.0%) |
| Tax-deferred | Growth, untouched for 10 years or more | 100% stocks, plus 50% diversifiers carried inside stacked funds | 100% / 22% / 9% / 19% | 16.3% | +7.7% (+3.0% to +12.4%) | +7.8% | -50% | +1.4% (-0.4% to +3.2%) |
| Roth | Spending within 3 years | T-bills | 0% / 0% / 0% / 0% | 0.0% |  |  |  |  |
| Roth | Spending in 3 to 10 years | 40% stocks, 60% diversifiers | 40% / 26% / 11% / 22% | 7.4% | +4.5% (+1.6% to +7.4%) | +4.9% | -20% | +1.3% (-1.2% to +3.9%) |
| Roth | Retirement income, drawn now | 65% stocks, 35% diversifiers | 65% / 15% / 7% / 13% | 10.6% | +5.4% (+2.3% to +8.5%) | +5.5% | -35% | +2.0% (+0.0% to +4.0%) |
| Roth | Growth, untouched for 10 years or more | 100% stocks, plus 50% diversifiers carried inside stacked funds | 100% / 22% / 9% / 19% | 16.3% | +7.7% (+3.0% to +12.4%) | +7.8% | -50% | +1.4% (-0.4% to +3.2%) |

## The ladder the stock share is read from

Stocks beside the sleeve, replayed from 1990-01 to 2026-10, rebalanced monthly.

| Stocks | Sleeve | Replay worst fall | Replay growth over T-bills | Replay vol | Forecast vol |
|---|---|---|---|---|---|
| 0% | 100% | -9.1% | +3.8% | 5.7% | 5.9% |
| 5% | 95% | -7.6% | +3.9% | 5.4% | 5.7% |
| 10% | 90% | -7.2% | +4.1% | 5.3% | 5.6% |
| 15% | 85% | -8.2% | +4.2% | 5.3% | 5.6% |
| 20% | 80% | -9.6% | +4.4% | 5.4% | 5.8% |
| 25% | 75% | -10.9% | +4.5% | 5.6% | 6.0% |
| 30% | 70% | -13.7% | +4.7% | 5.9% | 6.4% |
| 35% | 65% | -16.7% | +4.8% | 6.3% | 6.8% |
| 40% | 60% | -19.7% | +4.9% | 6.8% | 7.4% |
| 45% | 55% | -22.6% | +5.1% | 7.4% | 7.9% |
| 50% | 50% | -25.5% | +5.2% | 7.9% | 8.5% |
| 55% | 45% | -28.5% | +5.3% | 8.6% | 9.2% |
| 60% | 40% | -31.8% | +5.4% | 9.2% | 9.9% |
| 65% | 35% | -34.9% | +5.5% | 9.9% | 10.6% |
| 70% | 30% | -37.9% | +5.5% | 10.6% | 11.3% |
| 75% | 25% | -40.8% | +5.6% | 11.3% | 12.1% |
| 80% | 20% | -43.6% | +5.7% | 12.1% | 12.8% |
| 85% | 15% | -46.3% | +5.8% | 12.8% | 13.6% |
| 90% | 10% | -48.8% | +5.8% | 13.6% | 14.4% |
| 95% | 5% | -51.3% | +5.9% | 14.3% | 15.1% |
| 100% | 0% | -53.7% | +5.9% | 15.1% | 15.9% |

## The frontier's inputs

| Holding | Expected return over T-bills | 90% interval | History from | Read from | Forecast vol |
|---|---|---|---|---|---|
| World stocks | +6.3% | +2.9% to +9.8% | 1926-07 | threefund.expected | 15.9% |
| Intermediate Treasuries | +1.5% | -0.3% to +3.4% | 1962-02 | published forecast | 7.7% |
| Gold | +3.8% | -1.8% to +9.4% | 1971-09 | published forecast | 24.5% |
| Managed futures | +1.6% | -0.1% to +3.3% | 2007-03 | live funds only | 9.6% |

Managed futures are not in the covariance forecast: their volatility and correlations are the last 60 months'. Their mean is the live funds' only.

## Funds

One way to hold each answer, from `data/processed/diversifier_vehicles.json` (hand-recorded on 2026-11-16; confirm fees on the issuer's page).

Taxable brokerage, retirement income, drawn now: 65% stocks, 35% diversifiers.

| Fund | Name | Share of capital | Fee a year |
|---|---|---|---|
|  | A world stock fund | 65.0% |  |
| VGIT | Vanguard Intermediate-Term Treasury ETF | 15.3% | 0.03% |
| IAUM | iShares Gold Trust Micro | 6.6% | 0.09% |
| SDMF | Simplify DBi CTA Managed Futures Index ETF | 13.1% | 0.21% |

Recorded fund fees: 0.04% of capital a year.

Taxable brokerage, growth, untouched for 10 years or more: 100% stocks, plus 50% diversifiers on a margin loan.

| Fund | Name | Share of capital | Fee a year |
|---|---|---|---|
|  | A world stock fund | 100.0% |  |
| VGIT | Vanguard Intermediate-Term Treasury ETF | 21.8% | 0.03% |
| IAUM | iShares Gold Trust Micro | 9.4% | 0.09% |
| SDMF | Simplify DBi CTA Managed Futures Index ETF | 18.7% | 0.21% |

Recorded fund fees: 0.05% of capital a year; loan: 0.69%. The same position through stacked funds: 0.29% in fees and no loan.

Roth, growth, untouched for 10 years or more: 100% stocks, plus 50% diversifiers carried inside stacked funds.

| Fund | Name | Share of capital | Fee a year |
|---|---|---|---|
|  | A world stock fund | 49.0% |  |
| RSSB | Return Stacked Global Stocks & Bonds ETF | 21.8% | 0.40% |
| GDE | WisdomTree Efficient Gold Plus Equity Strategy Fund | 10.5% | 0.20% |
| RSST | Return Stacked U.S. Stocks & Managed Futures ETF | 18.7% | 0.99% |

Recorded fund fees: 0.29% of capital a year. Stocks held: 99% of capital.

## Caveats

- The loss limits are judgements, and the stock shares follow from them. A different limit gives a different answer; the ladder shows every one.
- The replay is the allocation study's: every year had been seen, the sleeve's assets were chosen on the same years, and managed futures before 2007-03 are a paper factor. A worst fall in the past is not a bound on the next one.
- Month-end values miss falls inside a month. A margin loan can be called before a month ends.
- Taxes, trading costs and fund fees are not charged. Traditional and Roth accounts get the same numbers.
- The frontier optimizes on historical means. The allocation study found that doing so did not beat fixed weights; the gap is what the optimizer claims.
- The stacked funds hold US or global stocks, not the study's world-stock composite, and none of them is in the study's returns.
