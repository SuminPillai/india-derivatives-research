# P2 Strategy Implementations

This directory holds reference implementations of the technical-indicator strategy families
tested in P2. They are backtested and aggregated by `../01_run_strategies.py`.

## Indicator families tested

| Family | Configurations | Description |
|--------|----------------|-------------|
| SMA crossover | fast/slow simple-moving-average crosses | Long when fast SMA crosses above slow SMA |
| MACD | signal-line crossovers | Long/short on MACD vs signal-line crosses |
| Bollinger Band | squeeze / breakout rules | Volatility-band mean-reversion and breakout |
| RSI | mean-reversion bands (e.g. RSI_20_80) | Buy oversold / sell overbought |
| Donchian channel | N-day high/low breakouts | Trend-following channel breakout |
| Confluence | multi-indicator agreement | Trades only when multiple families agree |

## `smamacd1.py`

Reference implementation of the SMA and MACD crossover logic. Per the source project's
conventions this is a **reference implementation and is copied verbatim** — it is not
modified for the public repository.

> Note: the original project split additional strategy variants (Bollinger, RSI, Donchian,
> confluence) across separate working files (`st2`–`st6`) and a confluence aggregator. In
> the public repository these are exercised through `01_run_strategies.py`, which is the
> single entry point that backtests all 13 configurations and produces the Romano–Wolf
> survivor table. The `confluence_backtest_summary.xlsx` referenced in the paper is not
> redistributed here; its column structure is: strategy name, per-stock Sharpe, aggregate
> Sharpe, raw p-value, Romano–Wolf adjusted p-value, survivor flag.
