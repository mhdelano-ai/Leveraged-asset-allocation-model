# Allocation questionnaire

*Arithmetic under declared rules on the diversifier and allocation studies and the published forecast; the rules are judgements, not advice, and nothing here is out-of-sample evidence.*

Written by `python -m assetforecast.questionnaire`; origin 2026-10-31. Two answers, the account's tax treatment and the goal, give a stock share; the rest goes to the diversifier sleeve (intermediate Treasuries, gold, managed futures). Interactive page: `dashboard/questionnaire.html`.

Risk decides and returns do not. Every answer is a point on one curve of mixes, and the goal's tolerance for loss picks the point. Nothing is optimized on an expected return; expected growth is shown with its interval and is not an input.

## The rule

1. Each goal declares the largest fall from a previous high it can sit through. The stock share is the largest, in steps of 5%, whose worst fall in the replay stayed within it, with the rest in the sleeve.
2. The sleeve is weighted by 1/volatility, whole past: today 44% Treasuries, 19% gold, 37% managed futures.
3. At 100% stocks nothing is left to backfill, so the sleeve goes on top, the most whose worst fall also stays within the limit: on a margin loan in a taxable account (T-bills + 1.38%, up to 100% of capital), through stacked funds in a retirement account, where no loan is allowed (up to 95%).
4. Money needed within 3 years is outside the model: T-bills.

| Goal | Fall it can sit through | Stock share |
|---|---|---|
| Spending within 3 years |  | T-bills |
| Spending in 3 to 10 years | 20% | 40% |
| Retirement income, drawn now | 35% | 65% |
| Growth, untouched for 10 years or more | 55% | 100% |

## Every answer

Forecast vol is on the package's 12-month covariance forecast. Expected growth is over T-bills, with the 90% interval of the historical means.

| Account | Goal | Answer | Stocks / Treasuries / gold / managed futures | Forecast vol | Replay worst fall | Replay growth | Expected growth (90%) |
|---|---|---|---|---|---|---|---|
| Taxable brokerage | Spending within 3 years | T-bills | 0% / 0% / 0% / 0% | 0.0% |  |  |  |
| Taxable brokerage | Spending in 3 to 10 years | 40% stocks, 60% diversifiers | 40% / 26% / 11% / 22% | 7.4% | -20% | +4.9% | +4.5% (+1.6% to +7.4%) |
| Taxable brokerage | Retirement income, drawn now | 65% stocks, 35% diversifiers | 65% / 15% / 7% / 13% | 10.6% | -35% | +5.5% | +5.4% (+2.3% to +8.5%) |
| Taxable brokerage | Growth, untouched for 10 years or more | 100% stocks, plus 100% diversifiers on a margin loan | 100% / 44% / 19% / 37% | 17.1% | -48% | +8.3% | +7.6% (+1.7% to +13.6%) |
| Tax-deferred | Spending within 3 years | T-bills | 0% / 0% / 0% / 0% | 0.0% |  |  |  |
| Tax-deferred | Spending in 3 to 10 years | 40% stocks, 60% diversifiers | 40% / 26% / 11% / 22% | 7.4% | -20% | +4.9% | +4.5% (+1.6% to +7.4%) |
| Tax-deferred | Retirement income, drawn now | 65% stocks, 35% diversifiers | 65% / 15% / 7% / 13% | 10.6% | -35% | +5.5% | +5.4% (+2.3% to +8.5%) |
| Tax-deferred | Growth, untouched for 10 years or more | 100% stocks, plus 95% diversifiers carried inside stacked funds | 100% / 42% / 18% / 36% | 17.0% | -47% | +9.5% | +8.9% (+3.1% to +14.7%) |
| Roth | Spending within 3 years | T-bills | 0% / 0% / 0% / 0% | 0.0% |  |  |  |
| Roth | Spending in 3 to 10 years | 40% stocks, 60% diversifiers | 40% / 26% / 11% / 22% | 7.4% | -20% | +4.9% | +4.5% (+1.6% to +7.4%) |
| Roth | Retirement income, drawn now | 65% stocks, 35% diversifiers | 65% / 15% / 7% / 13% | 10.6% | -35% | +5.5% | +5.4% (+2.3% to +8.5%) |
| Roth | Growth, untouched for 10 years or more | 100% stocks, plus 95% diversifiers carried inside stacked funds | 100% / 42% / 18% / 36% | 17.0% | -47% | +9.5% | +8.9% (+3.1% to +14.7%) |

## The curve

Stocks beside the sleeve, replayed from 1990-01 to 2026-10, rebalanced monthly. Forecast risk is lowest at 10% stocks.

Nothing borrowed:

| Stocks | Sleeve | Borrowed or carried | Forecast vol | Replay worst fall | Replay growth over T-bills | Replay vol | Expected growth (90%) |
|---|---|---|---|---|---|---|---|
| 0% | 100% | 0% | 5.9% | -9.1% | +3.8% | 5.7% | +2.7% (+0.2% to +5.2%) |
| 5% | 95% | 0% | 5.7% | -7.6% | +3.9% | 5.4% | +2.9% (+0.4% to +5.5%) |
| 10% | 90% | 0% | 5.6% | -7.2% | +4.1% | 5.3% | +3.2% (+0.6% to +5.8%) |
| 15% | 85% | 0% | 5.6% | -8.2% | +4.2% | 5.3% | +3.4% (+0.8% to +6.0%) |
| 20% | 80% | 0% | 5.8% | -9.6% | +4.4% | 5.4% | +3.6% (+1.0% to +6.3%) |
| 25% | 75% | 0% | 6.0% | -10.9% | +4.5% | 5.6% | +3.9% (+1.1% to +6.6%) |
| 30% | 70% | 0% | 6.4% | -13.7% | +4.7% | 5.9% | +4.1% (+1.3% to +6.9%) |
| 35% | 65% | 0% | 6.8% | -16.7% | +4.8% | 6.3% | +4.3% (+1.5% to +7.1%) |
| 40% | 60% | 0% | 7.4% | -19.7% | +4.9% | 6.8% | +4.5% (+1.6% to +7.4%) |
| 45% | 55% | 0% | 7.9% | -22.6% | +5.1% | 7.4% | +4.7% (+1.8% to +7.6%) |
| 50% | 50% | 0% | 8.5% | -25.5% | +5.2% | 7.9% | +4.9% (+1.9% to +7.8%) |
| 55% | 45% | 0% | 9.2% | -28.5% | +5.3% | 8.6% | +5.0% (+2.0% to +8.1%) |
| 60% | 40% | 0% | 9.9% | -31.8% | +5.4% | 9.2% | +5.2% (+2.1% to +8.3%) |
| 65% | 35% | 0% | 10.6% | -34.9% | +5.5% | 9.9% | +5.4% (+2.3% to +8.5%) |
| 70% | 30% | 0% | 11.3% | -37.9% | +5.5% | 10.6% | +5.5% (+2.4% to +8.7%) |
| 75% | 25% | 0% | 12.1% | -40.8% | +5.6% | 11.3% | +5.7% (+2.5% to +8.9%) |
| 80% | 20% | 0% | 12.8% | -43.6% | +5.7% | 12.1% | +5.8% (+2.6% to +9.1%) |
| 85% | 15% | 0% | 13.6% | -46.3% | +5.8% | 12.8% | +6.0% (+2.7% to +9.3%) |
| 90% | 10% | 0% | 14.4% | -48.8% | +5.8% | 13.6% | +6.1% (+2.7% to +9.5%) |
| 95% | 5% | 0% | 15.1% | -51.3% | +5.9% | 14.3% | +6.2% (+2.8% to +9.6%) |
| 100% | 0% | 0% | 15.9% | -53.7% | +5.9% | 15.1% | +6.3% (+2.9% to +9.8%) |

100% stocks with the sleeve on top, on a margin loan at T-bills + 1.38% (up to 100% of capital, 2 times capital in all):

| Stocks | Sleeve | Borrowed or carried | Forecast vol | Replay worst fall | Replay growth over T-bills | Replay vol | Expected growth (90%) |
|---|---|---|---|---|---|---|---|
| 100% | 10% | 10% | 15.9% | -53.1% | +6.2% | 15.1% | +6.5% (+2.8% to +10.2%) |
| 100% | 20% | 20% | 16.0% | -52.4% | +6.4% | 15.1% | +6.6% (+2.7% to +10.6%) |
| 100% | 30% | 30% | 16.1% | -51.8% | +6.7% | 15.2% | +6.8% (+2.6% to +11.0%) |
| 100% | 40% | 40% | 16.1% | -51.2% | +6.9% | 15.2% | +6.9% (+2.4% to +11.3%) |
| 100% | 50% | 50% | 16.3% | -50.6% | +7.1% | 15.3% | +7.0% (+2.3% to +11.7%) |
| 100% | 60% | 60% | 16.4% | -50.0% | +7.4% | 15.4% | +7.1% (+2.2% to +12.1%) |
| 100% | 70% | 70% | 16.5% | -49.4% | +7.6% | 15.5% | +7.3% (+2.1% to +12.5%) |
| 100% | 80% | 80% | 16.7% | -48.8% | +7.9% | 15.6% | +7.4% (+1.9% to +12.8%) |
| 100% | 90% | 90% | 16.9% | -48.1% | +8.1% | 15.8% | +7.5% (+1.8% to +13.2%) |
| 100% | 100% | 100% | 17.1% | -47.5% | +8.3% | 16.0% | +7.6% (+1.7% to +13.6%) |

100% stocks with the sleeve on top, inside stacked funds (up to 95% of capital, where those funds use all the capital; financed inside the funds, counted at the T-bill rate):

| Stocks | Sleeve | Borrowed or carried | Forecast vol | Replay worst fall | Replay growth over T-bills | Replay vol | Expected growth (90%) |
|---|---|---|---|---|---|---|---|
| 100% | 10% | 10% | 15.9% | -53.0% | +6.3% | 15.1% | +6.6% (+2.9% to +10.3%) |
| 100% | 20% | 20% | 16.0% | -52.3% | +6.7% | 15.1% | +6.9% (+2.9% to +10.9%) |
| 100% | 30% | 30% | 16.1% | -51.6% | +7.1% | 15.1% | +7.2% (+3.0% to +11.4%) |
| 100% | 40% | 40% | 16.1% | -50.8% | +7.5% | 15.2% | +7.4% (+3.0% to +11.9%) |
| 100% | 50% | 50% | 16.3% | -50.1% | +7.8% | 15.3% | +7.7% (+3.0% to +12.4%) |
| 100% | 60% | 60% | 16.4% | -49.4% | +8.2% | 15.4% | +8.0% (+3.0% to +12.9%) |
| 100% | 70% | 70% | 16.5% | -48.7% | +8.6% | 15.5% | +8.2% (+3.0% to +13.4%) |
| 100% | 80% | 80% | 16.7% | -48.0% | +8.9% | 15.6% | +8.5% (+3.1% to +13.9%) |
| 100% | 90% | 90% | 16.9% | -47.2% | +9.3% | 15.8% | +8.8% (+3.1% to +14.4%) |

## What risk and expected growth are measured on

| Holding | Forecast vol | Expected return over T-bills | 90% interval | History from | Read from |
|---|---|---|---|---|---|
| World stocks | 15.9% | +6.3% | +2.9% to +9.8% | 1926-07 | threefund.expected |
| Intermediate Treasuries | 7.7% | +1.5% | -0.3% to +3.4% | 1962-02 | published forecast |
| Gold | 24.5% | +3.8% | -1.8% to +9.4% | 1971-09 | published forecast |
| Managed futures | 9.6% | +1.6% | -0.1% to +3.3% | 2007-03 | live funds only |

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

Taxable brokerage, growth, untouched for 10 years or more: 100% stocks, plus 100% diversifiers on a margin loan.

| Fund | Name | Share of capital | Fee a year |
|---|---|---|---|
|  | A world stock fund | 100.0% |  |
| VGIT | Vanguard Intermediate-Term Treasury ETF | 43.7% | 0.03% |
| IAUM | iShares Gold Trust Micro | 18.9% | 0.09% |
| SDMF | Simplify DBi CTA Managed Futures Index ETF | 37.4% | 0.21% |

Recorded fund fees: 0.11% of capital a year; loan: 1.38%.

Roth, growth, untouched for 10 years or more: 100% stocks, plus 95% diversifiers carried inside stacked funds.

| Fund | Name | Share of capital | Fee a year |
|---|---|---|---|
|  | A world stock fund | 3.0% |  |
| RSSB | Return Stacked Global Stocks & Bonds ETF | 41.5% | 0.40% |
| GDE | WisdomTree Efficient Gold Plus Equity Strategy Fund | 19.9% | 0.20% |
| RSST | Return Stacked U.S. Stocks & Managed Futures ETF | 35.6% | 0.99% |

Recorded fund fees: 0.56% of capital a year. Stocks held: 98% of capital.

## Caveats

- The loss limits are judgements, and the stock shares follow from them. A different limit gives a different answer; the curve shows every one.
- A goal that allows 100% stocks gets the end of the curve, the most exposure the account can carry: more sleeve on top lowered the replay's worst fall, so the loss limit never stops it. Forecast volatility rises with it. On margin the end is Regulation T's limit for opening a position (2 times capital); the broker's maintenance requirement, which is not modelled, decides how far it can fall before it is called.
- The replay is the allocation study's: every year had been seen, the sleeve's assets were chosen on the same years, and managed futures before 2007-03 are a paper factor. A worst fall in the past is not a bound on the next one.
- Month-end values miss falls inside a month. A margin loan can be called before a month ends.
- Taxes, trading costs and fund fees are not charged. Traditional and Roth accounts get the same numbers.
- Expected growth rests on historical means, which the allocation study found do not carry over. It is shown, with its interval, and decides nothing.
- The stacked funds hold US or global stocks, not the study's world-stock composite, and none of them is in the study's returns.
