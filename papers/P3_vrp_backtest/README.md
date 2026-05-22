# Trading the Volatility Risk Premium on Nifty 50: Backtest with Realistic Frictions

**SSRN:** TBD
**Status:** Draft

## Abstract

Systematic sellers of variance risk on U.S. equity index options have long earned a positive
risk premium, but whether that premium survives realistic transaction costs in an emerging
market with a high Securities Transaction Tax (STT) and wide bid-ask spreads remains an open
question. This paper tests four short-volatility strategies on Nifty 50 index options over
119 monthly expiry cycles from January 2015 to April 2025, applying a cost model that
incorporates STT (0.05% on sell-side premium), brokerage (0.03% per leg), slippage (0.5% of
premium), and, for the delta-hedged variant, daily futures rebalancing costs (0.01% per
trade). The four strategies examined are an ATM short straddle, a crash-neutral spread, a
naked put-write at four moneyness levels (k = 0.94, 0.96, 0.98, 1.00), and a delta-hedged
straddle. Net of all costs, every strategy produces negative annualized returns. The
put-write at k = 0.94 is the least adverse outcome, with a 91.6% monthly win rate and an
annualized return of −0.9% (Sharpe ratio −0.37); the straddle variants lose approximately 44
to 45% per year. Cost attribution confirms that transaction costs are a secondary factor;
the primary driver of losses across all strategies is tail risk. These findings imply that
the volatility risk premium documented in Pillai (2026a) exists in Indian index options but
is not capturable by a retail trader under current market microstructure.

## Methodology summary

Four short-volatility strategies are backtested on Nifty 50 monthly index options over 119
expiry cycles (January 2015 – April 2025): an ATM short straddle, a crash-neutral spread, a
naked put-write at four moneyness levels (k = 0.94, 0.96, 0.98, 1.00), and a delta-hedged
straddle. Each position is held to expiry and marked with a realistic friction model: STT
(0.05% on sell-side premium), brokerage (0.03% per leg), slippage (0.5% of premium), and —
for the delta-hedged variant — daily futures rebalancing at 0.01% per trade. Net P&L,
annualised return, Sharpe ratio, monthly win rate, and worst single-month drawdown are
computed per strategy and moneyness level, and losses are decomposed into a transaction-cost
component and a tail-risk component. The premium that motivates these strategies is the one
estimated in P1 (Pillai, 2026a).

## Data requirements

- **NSE F&O bhavcopy** — daily ZIPs from https://www.nseindia.com/all-reports (free), for
  Nifty 50 option premia and the monthly held-to-maturity panel.
- **Nifty 50 futures** — for the delta-hedged variant's rebalancing leg.

## Reproduction

1. Install dependencies: `pip install -r ../../requirements.txt`
2. Configure environment variables (see `../../docs/data_sources.md`).
3. Run scripts in order:
   - `python 01_strategy_backtest.py` — backtests all four strategies with the full friction
     model and produces the net-of-cost performance table and cost attribution.
4. Final outputs land in `tables/`.

> The friction model (STT, brokerage, slippage, futures rebalancing) is implemented inline
> within `01_strategy_backtest.py`; the cost parameters are defined at the top of the script.

## Reproducibility seed

All scripts set `np.random.seed(20260509)` for reproducibility. Any deviation produces
non-matching numerical results.

## Citation

```bibtex
@unpublished{pillai2026vrp,
  author = {Pillai, Sumin},
  title  = {Trading the Volatility Risk Premium on Nifty 50: Backtest with Realistic Frictions},
  year   = {2026},
  note   = {Working paper, SSRN},
  url    = {https://github.com/SuminPillai/india-derivatives-research}
}
```
