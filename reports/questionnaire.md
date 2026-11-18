# Allocation questionnaire

*Arithmetic under declared rules on the diversifier and allocation studies and the published forecast; the rules are judgements, not advice, and nothing here is out-of-sample evidence.*

Written by `python -m assetforecast.questionnaire`; origin 2026-10-31. One choice: how much in world stocks. The rest of the capital is in the diversifiers (intermediate Treasuries, gold, managed futures). Nothing is borrowed: every answer holds at most 100% of capital. Interactive page: `dashboard/questionnaire.html`.

Risk decides and returns do not. Nothing is chosen on an expected return; expected growth is shown with its interval and is not an input.

## How the mix is set

1. Stocks cause the large falls, so the goal's drawdown budget sets the stock share: the largest, in steps of 5%, whose worst fall in the replay stayed within the budget. The fall is measured on the whole portfolio, diversifiers included.
2. The diversifiers held up when stocks fell, so they take the rest of the capital, always in the same proportions (44% Treasuries, 19% gold, 37% managed futures; 1/volatility, whole past).
3. Money needed within 3 years has no budget: T-bills.

## Every answer

Forecast vol is on the package's 12-month covariance forecast. Expected growth is over T-bills, with the 90% interval of the historical means. The account's tax treatment does not change the mix; it changes the notes below.

| Goal | Drawdown budget | Answer | Stocks / Treasuries / gold / managed futures | Forecast vol | Replay worst fall | Replay growth | Expected growth (90%) | Fund fees |
|---|---|---|---|---|---|---|---|---|
| Spending within 3 years |  | T-bills | 0% / 0% / 0% / 0% | 0.0% |  |  |  |  |
| Spending in 3 to 10 years | 20% | 40% stocks and 60% diversifiers | 40% / 26% / 11% / 22% | 7.4% | -20% | +4.9% | +4.5% (+1.6% to +7.4%) | 0.07% |
| Retirement income, drawn now | 35% | 65% stocks and 35% diversifiers | 65% / 15% / 7% / 13% | 10.6% | -35% | +5.5% | +5.4% (+2.3% to +8.5%) | 0.04% |
| Growth, untouched for 10 years or more | 55% | 100% stocks | 100% / 0% / 0% / 0% | 15.9% | -54% | +5.9% | +6.3% (+2.9% to +9.8%) | 0.00% |

## Tax notes

- **Taxable brokerage.** Taxes are not modelled. Stock index funds defer most of their gain until sold. The diversifiers are taxed harder: Treasury interest is ordinary income (free of state tax), funds holding gold bullion are taxed as collectibles, and funds that use futures pay out most of their return as taxable distributions. The replay rebalances every month. In this account a sale can realise a gain.
- **Tax-deferred.** Nothing is taxed until withdrawal, and withdrawals are taxed as ordinary income whatever was held. No holding is tax-favoured over another, and rebalancing costs no tax.
- **Roth.** Qualified withdrawals are untaxed. No holding is tax-favoured over another, and rebalancing costs no tax.

## Every stock share

Each stock share with the diversifiers in the rest of the capital, replayed from 1990-01 to 2026-10, rebalanced monthly.

| Stocks | Diversifiers | Forecast vol | Replay worst fall | Replay growth over T-bills | Expected growth (90%) |
|---|---|---|---|---|---|
| 0% | 100% | 5.9% | -9.1% | +3.8% | +2.7% (+0.2% to +5.2%) |
| 10% | 90% | 5.6% | -7.2% | +4.1% | +3.2% (+0.6% to +5.8%) |
| 20% | 80% | 5.8% | -9.6% | +4.4% | +3.6% (+1.0% to +6.3%) |
| 30% | 70% | 6.4% | -13.7% | +4.7% | +4.1% (+1.3% to +6.9%) |
| 40% | 60% | 7.4% | -19.7% | +4.9% | +4.5% (+1.6% to +7.4%) |
| 50% | 50% | 8.5% | -25.5% | +5.2% | +4.9% (+1.9% to +7.8%) |
| 60% | 40% | 9.9% | -31.8% | +5.4% | +5.2% (+2.1% to +8.3%) |
| 70% | 30% | 11.3% | -37.9% | +5.5% | +5.5% (+2.4% to +8.7%) |
| 80% | 20% | 12.8% | -43.6% | +5.7% | +5.8% (+2.6% to +9.1%) |
| 90% | 10% | 14.4% | -48.8% | +5.8% | +6.1% (+2.7% to +9.5%) |
| 100% | 0% | 15.9% | -53.7% | +5.9% | +6.3% (+2.9% to +9.8%) |

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

Spending in 3 to 10 years: 40% stocks and 60% diversifiers.

| Fund | Name | Share of capital | Fee a year |
|---|---|---|---|
|  | A world stock fund | 40.0% |  |
| VGIT | Vanguard Intermediate-Term Treasury ETF | 26.2% | 0.03% |
| IAUM | iShares Gold Trust Micro | 11.3% | 0.09% |
| SDMF | Simplify DBi CTA Managed Futures Index ETF | 22.5% | 0.21% |

Recorded fund fees: 0.07% of capital a year; no stock fund's fee is recorded.

Retirement income, drawn now: 65% stocks and 35% diversifiers.

| Fund | Name | Share of capital | Fee a year |
|---|---|---|---|
|  | A world stock fund | 65.0% |  |
| VGIT | Vanguard Intermediate-Term Treasury ETF | 15.3% | 0.03% |
| IAUM | iShares Gold Trust Micro | 6.6% | 0.09% |
| SDMF | Simplify DBi CTA Managed Futures Index ETF | 13.1% | 0.21% |

Recorded fund fees: 0.04% of capital a year; no stock fund's fee is recorded.

Growth, untouched for 10 years or more: 100% stocks.

| Fund | Name | Share of capital | Fee a year |
|---|---|---|---|
|  | A world stock fund | 100.0% |  |

Recorded fund fees: 0.00% of capital a year; no stock fund's fee is recorded.

## Caveats

- The drawdown budgets are judgements, and the stock shares follow from them. The tables show what any other budget would give.
- The diversifiers fall too. With no stocks at all, the diversifiers alone fell -9% in the replay.
- The replay is the allocation study's: every year had been seen, the diversifiers were chosen on the same years, and managed futures before 2007-03 are a paper factor. A worst fall in the past is not a bound on the next one.
- Month-end values miss falls inside a month.
- Taxes, trading costs and fund fees are not charged.
- Expected growth rests on historical means, which the allocation study found do not carry over. It is shown, with its interval, and decides nothing.
