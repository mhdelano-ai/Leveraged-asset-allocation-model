# Allocation questionnaire

*Arithmetic under declared rules on the diversifier and allocation studies and the published forecast; the rules are judgements, not advice, and nothing here is out-of-sample evidence.*

Written by `python -m assetforecast.questionnaire`; origin 2026-10-31. Two dials: how much in world stocks, and how much in the diversifiers (intermediate Treasuries, gold, managed futures). Interactive page: `dashboard/questionnaire.html`.

Risk decides and returns do not. Nothing is chosen on an expected return; expected growth is shown with its interval and is not an input.

## The two dials

1. Stocks cause the large falls, so the goal's drawdown budget sets the stock share: the largest, in steps of 5%, whose worst fall in the replay stayed within the budget. The fall is measured on the whole portfolio, diversifiers included.
2. The diversifiers held up when stocks fell, so they are held beside the stocks as far as the account can carry, always in the same proportions (44% Treasuries, 19% gold, 37% managed futures; 1/volatility, whole past). Stacked funds carry them inside, with no loan (the only way in a retirement account, the default in a taxable one). A margin loan, in a taxable account that chooses it, carries 2 times capital in all at T-bills + 1.38%.
3. Money needed within 3 years has no budget: T-bills.

## Every answer

Forecast vol is on the package's 12-month covariance forecast. Expected growth is over T-bills, with the 90% interval of the historical means. Tax-deferred and Roth accounts get the same answers.

| Account | Above 100% by | Goal | Answer | Stocks / Treasuries / gold / managed futures | Forecast vol | Replay worst fall | Replay growth | Expected growth (90%) | Fund fees | Loan |
|---|---|---|---|---|---|---|---|---|---|---|
| Taxable brokerage | return-stacked funds | Spending within 3 years | T-bills | 0% / 0% / 0% / 0% | 0.0% |  |  |  |  |  |
| Taxable brokerage | return-stacked funds | Spending in 3 to 10 years | 40% stocks and 100% diversifiers, 40% of capital carried inside stacked funds | 40% / 44% / 19% / 37% | 8.8% | -18% | +6.4% | +5.5% (+1.6% to +9.4%) | 0.26% |  |
| Taxable brokerage | return-stacked funds | Retirement income, drawn now | 70% stocks and 100% diversifiers, 70% of capital carried inside stacked funds | 70% / 44% / 19% / 37% | 12.7% | -33% | +8.1% | +7.4% (+2.5% to +12.3%) | 0.48% |  |
| Taxable brokerage | return-stacked funds | Growth, untouched for 10 years or more | 100% stocks and 95% diversifiers, 95% of capital carried inside stacked funds | 100% / 42% / 18% / 36% | 17.0% | -47% | +9.5% | +8.9% (+3.1% to +14.7%) | 0.56% |  |
| Taxable brokerage | margin loan | Spending within 3 years | T-bills | 0% / 0% / 0% / 0% | 0.0% |  |  |  |  |  |
| Taxable brokerage | margin loan | Spending in 3 to 10 years | 35% stocks and 165% diversifiers, on a margin loan of 100% of capital | 35% / 72% / 31% / 62% | 11.3% | -19% | +7.0% | +5.4% (0.0% to +10.7%) | 0.18% | 1.38% |
| Taxable brokerage | margin loan | Retirement income, drawn now | 70% stocks and 130% diversifiers, on a margin loan of 100% of capital | 70% / 57% / 25% / 49% | 13.7% | -33% | +7.8% | +6.7% (+1.1% to +12.4%) | 0.14% | 1.38% |
| Taxable brokerage | margin loan | Growth, untouched for 10 years or more | 100% stocks and 100% diversifiers, on a margin loan of 100% of capital | 100% / 44% / 19% / 37% | 17.1% | -48% | +8.3% | +7.6% (+1.7% to +13.6%) | 0.11% | 1.38% |
| Tax-deferred | return-stacked funds | Spending within 3 years | T-bills | 0% / 0% / 0% / 0% | 0.0% |  |  |  |  |  |
| Tax-deferred | return-stacked funds | Spending in 3 to 10 years | 40% stocks and 100% diversifiers, 40% of capital carried inside stacked funds | 40% / 44% / 19% / 37% | 8.8% | -18% | +6.4% | +5.5% (+1.6% to +9.4%) | 0.26% |  |
| Tax-deferred | return-stacked funds | Retirement income, drawn now | 70% stocks and 100% diversifiers, 70% of capital carried inside stacked funds | 70% / 44% / 19% / 37% | 12.7% | -33% | +8.1% | +7.4% (+2.5% to +12.3%) | 0.48% |  |
| Tax-deferred | return-stacked funds | Growth, untouched for 10 years or more | 100% stocks and 95% diversifiers, 95% of capital carried inside stacked funds | 100% / 42% / 18% / 36% | 17.0% | -47% | +9.5% | +8.9% (+3.1% to +14.7%) | 0.56% |  |

## Return stacking and a margin loan

Both hold more than 100% of capital; the difference is who borrows.

- **Return-stacked funds.** You buy funds that each hold stocks and add a diversifier on top with futures. The borrowing happens inside the fund, at close to the T-bill rate. You owe nothing, so there is no loan on your account for anyone to call, and the most you can lose is what you put in. The price is higher fund fees and less choice: each fund fixes its own blend, and the stocks inside are US or global stocks.
- **Margin loan.** You buy plain funds, then borrow from the broker against them to buy more. The loan costs T-bills + 1.38%. If the account falls far enough, the broker can demand cash or sell your holdings at the bottom. You choose every fund and the loan reaches 2 times capital, but it is not allowed in a retirement account.

The same goal by each route. Cost is recorded fund fees plus, on margin, the loan's interest over T-bills. The replay charges the loan but no fund fees, so the last column takes each route's fees off.

| Goal | Route | Stocks | Diversifiers | Above 100% of capital | Cost a year | Forecast vol | Replay worst fall | Replay growth after fund fees |
|---|---|---|---|---|---|---|---|---|
| Spending in 3 to 10 years | Return-stacked funds | 40% | 100% | 40% | 0.26% | 8.8% | -18% | +6.1% |
| Spending in 3 to 10 years | Margin loan | 35% | 165% | 100% | 1.56% | 11.3% | -19% | +6.8% |
| Retirement income, drawn now | Return-stacked funds | 70% | 100% | 70% | 0.48% | 12.7% | -33% | +7.7% |
| Retirement income, drawn now | Margin loan | 70% | 130% | 100% | 1.52% | 13.7% | -33% | +7.7% |
| Growth, untouched for 10 years or more | Return-stacked funds | 100% | 95% | 95% | 0.56% | 17.0% | -47% | +8.9% |
| Growth, untouched for 10 years or more | Margin loan | 100% | 100% | 100% | 1.49% | 17.1% | -48% | +8.2% |

## Every stock share

Each stock share with the most diversifiers the account can carry beside it, replayed from 1990-01 to 2026-10, rebalanced monthly.

With stacked funds:

| Stocks | Diversifiers | Above 100% of capital | Forecast vol | Replay worst fall | Replay growth over T-bills | Expected growth (90%) |
|---|---|---|---|---|---|---|
| 0% | 100% | 0% | 5.9% | -9.1% | +3.8% | +2.7% (+0.2% to +5.2%) |
| 10% | 100% | 10% | 6.2% | -7.9% | +4.5% | +3.4% (+0.6% to +6.3%) |
| 20% | 100% | 20% | 6.8% | -10.5% | +5.1% | +4.2% (+1.0% to +7.3%) |
| 30% | 100% | 30% | 7.7% | -13.7% | +5.8% | +4.8% (+1.3% to +8.4%) |
| 40% | 100% | 40% | 8.8% | -17.8% | +6.4% | +5.5% (+1.6% to +9.4%) |
| 50% | 100% | 50% | 10.0% | -23.2% | +7.0% | +6.2% (+1.9% to +10.4%) |
| 60% | 100% | 60% | 11.3% | -28.3% | +7.6% | +6.8% (+2.2% to +11.3%) |
| 70% | 100% | 70% | 12.7% | -33.1% | +8.1% | +7.4% (+2.5% to +12.3%) |
| 80% | 100% | 80% | 14.1% | -37.7% | +8.7% | +7.9% (+2.7% to +13.2%) |
| 90% | 95% | 85% | 15.5% | -42.3% | +9.0% | +8.4% (+2.9% to +13.8%) |
| 100% | 95% | 95% | 17.0% | -46.9% | +9.5% | +8.9% (+3.1% to +14.7%) |

On a margin loan at T-bills + 1.38%:

| Stocks | Diversifiers | Above 100% of capital | Forecast vol | Replay worst fall | Replay growth over T-bills | Expected growth (90%) |
|---|---|---|---|---|---|---|
| 0% | 200% | 100% | 11.8% | -21.4% | +5.8% | +3.6% (-1.3% to +8.6%) |
| 10% | 190% | 100% | 11.4% | -18.7% | +6.2% | +4.2% (-0.9% to +9.2%) |
| 20% | 180% | 100% | 11.2% | -16.0% | +6.5% | +4.7% (-0.5% to +9.8%) |
| 30% | 170% | 100% | 11.2% | -17.7% | +6.8% | +5.1% (-0.1% to +10.4%) |
| 40% | 160% | 100% | 11.5% | -20.2% | +7.1% | +5.6% (+0.2% to +10.9%) |
| 50% | 150% | 100% | 12.0% | -22.8% | +7.4% | +6.0% (+0.5% to +11.4%) |
| 60% | 140% | 100% | 12.8% | -27.7% | +7.6% | +6.4% (+0.8% to +11.9%) |
| 70% | 130% | 100% | 13.7% | -32.7% | +7.8% | +6.7% (+1.1% to +12.4%) |
| 80% | 120% | 100% | 14.7% | -37.6% | +8.0% | +7.1% (+1.3% to +12.8%) |
| 90% | 110% | 100% | 15.9% | -42.3% | +8.2% | +7.4% (+1.5% to +13.2%) |
| 100% | 100% | 100% | 17.1% | -47.5% | +8.3% | +7.6% (+1.7% to +13.6%) |

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

Spending in 3 to 10 years, return-stacked funds: 40% stocks and 100% diversifiers, 40% of capital carried inside stacked funds.

| Fund | Name | Share of capital | Fee a year |
|---|---|---|---|
| RSSB | Return Stacked Global Stocks & Bonds ETF | 40.0% | 0.40% |
| VGIT | Vanguard Intermediate-Term Treasury ETF | 3.7% | 0.03% |
| IAUM | iShares Gold Trust Micro | 18.9% | 0.09% |
| SDMF | Simplify DBi CTA Managed Futures Index ETF | 37.4% | 0.21% |

Recorded fund fees: 0.26% of capital a year.

Retirement income, drawn now, return-stacked funds: 70% stocks and 100% diversifiers, 70% of capital carried inside stacked funds.

| Fund | Name | Share of capital | Fee a year |
|---|---|---|---|
| RSSB | Return Stacked Global Stocks & Bonds ETF | 43.7% | 0.40% |
| IAUM | iShares Gold Trust Micro | 18.9% | 0.09% |
| RSST | Return Stacked U.S. Stocks & Managed Futures ETF | 26.3% | 0.99% |
| SDMF | Simplify DBi CTA Managed Futures Index ETF | 11.1% | 0.21% |

Recorded fund fees: 0.48% of capital a year.

Growth, untouched for 10 years or more, return-stacked funds: 100% stocks and 95% diversifiers, 95% of capital carried inside stacked funds.

| Fund | Name | Share of capital | Fee a year |
|---|---|---|---|
|  | A world stock fund | 3.0% |  |
| RSSB | Return Stacked Global Stocks & Bonds ETF | 41.5% | 0.40% |
| GDE | WisdomTree Efficient Gold Plus Equity Strategy Fund | 19.9% | 0.20% |
| RSST | Return Stacked U.S. Stocks & Managed Futures ETF | 35.6% | 0.99% |

Recorded fund fees: 0.56% of capital a year. Stocks held: 98% of capital.

Spending in 3 to 10 years, margin loan: 35% stocks and 165% diversifiers, on a margin loan of 100% of capital.

| Fund | Name | Share of capital | Fee a year |
|---|---|---|---|
|  | A world stock fund | 35.0% |  |
| VGIT | Vanguard Intermediate-Term Treasury ETF | 72.1% | 0.03% |
| IAUM | iShares Gold Trust Micro | 31.1% | 0.09% |
| SDMF | Simplify DBi CTA Managed Futures Index ETF | 61.8% | 0.21% |

Recorded fund fees: 0.18% of capital a year; loan: 1.38%.

Retirement income, drawn now, margin loan: 70% stocks and 130% diversifiers, on a margin loan of 100% of capital.

| Fund | Name | Share of capital | Fee a year |
|---|---|---|---|
|  | A world stock fund | 70.0% |  |
| VGIT | Vanguard Intermediate-Term Treasury ETF | 56.8% | 0.03% |
| IAUM | iShares Gold Trust Micro | 24.5% | 0.09% |
| SDMF | Simplify DBi CTA Managed Futures Index ETF | 48.7% | 0.21% |

Recorded fund fees: 0.14% of capital a year; loan: 1.38%.

Growth, untouched for 10 years or more, margin loan: 100% stocks and 100% diversifiers, on a margin loan of 100% of capital.

| Fund | Name | Share of capital | Fee a year |
|---|---|---|---|
|  | A world stock fund | 100.0% |  |
| VGIT | Vanguard Intermediate-Term Treasury ETF | 43.7% | 0.03% |
| IAUM | iShares Gold Trust Micro | 18.9% | 0.09% |
| SDMF | Simplify DBi CTA Managed Futures Index ETF | 37.4% | 0.21% |

Recorded fund fees: 0.11% of capital a year; loan: 1.38%.

## Caveats

- The drawdown budgets are judgements, and the stock shares follow from them. The tables show what any other budget would give.
- The diversifiers fall too. With no stocks at all, 100% of capital in them fell -9% in the replay, and 200% on margin fell -21%.
- A margin loan is 100% of capital at every goal. That is Regulation T's limit for opening a position; the broker's maintenance requirement, which is not modelled, decides how far it can fall before it is called. Whether the diversifiers earn more than the loan costs is not established: 200% of them on margin has expected growth of -1.3% to +8.6%.
- The replay is the allocation study's: every year had been seen, the diversifiers were chosen on the same years, and managed futures before 2007-03 are a paper factor. A worst fall in the past is not a bound on the next one.
- Month-end values miss falls inside a month.
- Taxes, trading costs and fund fees are not charged.
- Expected growth rests on historical means, which the allocation study found do not carry over. It is shown, with its interval, and decides nothing.
- The stacked funds hold US or global stocks, not the study's world-stock composite, and none of them is in the study's returns.
