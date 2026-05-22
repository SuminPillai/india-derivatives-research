# Technical-Indicator Strategy Performance in Indian Equities: Joint Test of Market Efficiency 2015–2025

**SSRN:** TBD
**Status:** Draft

## Abstract

The Efficient Market Hypothesis (EMH) in its weak form implies that no trading rule derived
from historical price data can generate persistent risk-adjusted excess returns. Technical
analysis provides a direct empirical test of this proposition, yet prior evidence for Indian
equities is limited and uniformly fails to account for the data-snooping problem inherent in
selecting strategies from a large parameter space. This paper tests 13 technical-indicator
strategy configurations — drawn from six families comprising SMA crossovers, MACD
signal-line crossovers, Bollinger Band squeeze rules, RSI mean-reversion rules, Donchian
channel breakouts, and a multi-indicator confluence strategy — on 78 NSE F&O-eligible stocks
over the period 1 January 2015 through 30 April 2025, applying the Romano and Wolf (2005)
StepM stepdown multiple-testing correction to control the family-wise error rate at 5%
across all 13 tests simultaneously. Eight of the 13 configurations survive this correction.
Seven are trend-following — three SMA crossovers, two MACD variants, and two Donchian
channel breakouts — and the eighth is the most aggressive RSI mean-reversion variant
(RSI_20_80, adjusted p = 0.037). The best-performing configuration, Donchian_20, achieves an
annualised Sharpe ratio of 1.26 with a Romano–Wolf adjusted p-value of 0.0006. These
findings constitute evidence against weak-form efficiency in the trend-following segment of
Indian F&O equities and stand in contrast to the Bajgrowicz and Scaillet (2012) benchmark,
in which no technical rule survives correction on developed-market data after 1987.

## Methodology summary

Thirteen technical-indicator strategy configurations are backtested on 78 NSE F&O-eligible
stocks over January 2015 – April 2025. The configurations span six families: SMA crossovers,
MACD signal-line crossovers, Bollinger Band squeeze rules, RSI mean-reversion rules,
Donchian channel breakouts, and a multi-indicator confluence rule. For each configuration,
per-stock returns are aggregated into a strategy-level performance series and an annualised
Sharpe ratio. To control for data-snooping across the 13 simultaneous tests, the Romano and
Wolf (2005) StepM stepdown procedure is applied with a stationary bootstrap (Politis and
White, 2004), controlling the family-wise error rate at 5%. A strategy is declared a
survivor only if its Romano–Wolf adjusted p-value falls below 0.05.

## Data requirements

- **NSE equity OHLC** for 78 F&O-eligible stocks — DefinEdge API (subscription) staged in
  MongoDB. See `../../docs/data_sources.md`.

## Reproduction

1. Install dependencies: `pip install -r ../../requirements.txt`
2. Configure environment variables (see `../../docs/data_sources.md`).
3. Run scripts in order:
   - `python 01_run_strategies.py` — backtests all 13 configurations, computes per-strategy
     Sharpe ratios, and applies the Romano–Wolf StepM correction to identify survivors.
4. Reference strategy implementations live in `strategies/`.

## Strategy implementations

See `strategies/README.md` for descriptions of the indicator families and the reference
implementation `smamacd1.py`.

## Reproducibility seed

All scripts set `np.random.seed(20260509)` for reproducibility. Any deviation produces
non-matching numerical results.

## Citation

```bibtex
@unpublished{pillai2026technical,
  author = {Pillai, Sumin},
  title  = {Technical-Indicator Strategy Performance in Indian Equities: Joint Test of Market Efficiency 2015--2025},
  year   = {2026},
  note   = {Working paper, SSRN},
  url    = {https://github.com/SuminPillai/india-derivatives-research}
}
```
