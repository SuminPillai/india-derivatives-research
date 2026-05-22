# Nifty 50 Index Put Option Mispricing: A BCJ-Style Test 2015–2025

**SSRN:** TBD
**Status:** Draft

## Abstract

Equity index put options in the United States have long earned returns too negative to be
reconciled with standard asset-pricing models, a puzzle first formally quantified by
Broadie, Chernov, and Johannes (2009), hereafter BCJ, using finite-sample Monte Carlo
inference on S&P 500 data. This paper asks whether the same anomaly exists in India. We
apply the BCJ finite-sample simulation framework to Nifty 50 index put options over January
2015 to April 2025, a 119-month sample spanning the 2018 NBFC credit stress, the COVID-19
dislocation of 2020, and the 2022 global inflation and rate-hike cycle. Three structural
models are estimated from Nifty 50 daily returns: Black-Scholes (BS), Heston (1993)
stochastic volatility (SV), and Bates (1996, 2000) stochastic volatility with jumps (SVJ),
with SV parameters obtained via Markov chain Monte Carlo and SVJ parameters via method of
moments (the SVJ MCMC chains did not fully converge; SVJ results are therefore reported as
a robustness check on the SV findings). For each model we generate 25,000 simulated return
paths and compute finite-sample p-values for held-to-maturity put returns across four
moneyness buckets. Delta-neutral strategies — the ATM straddle and crash-neutral spread —
reject the no-additional-risk-premium null at p = 0.000 under all three models, a result
that survives both jump-risk and estimation-risk adjustments to the risk-neutral measure.
Regime analysis shows that the variance risk premium strengthens monotonically across
successive stress episodes and does not revert to pre-2018 levels.

## Methodology summary

The empirical design replicates BCJ's finite-sample simulation test on Indian data. Monthly
expiry option chains are reconstructed from NSE F&O bhavcopy archives and assembled into a
held-to-maturity panel of put returns across four moneyness buckets (strike-to-spot ratios
k = 0.94, 0.96, 0.98, 1.00). Structural parameters for the BS, SV, and SVJ models are
estimated from Nifty 50 daily returns: SV via Metropolis-within-Gibbs MCMC following Eraker,
Johannes, and Polson (2003), and SVJ via method of moments after Lee–Mykland jump detection
(the SVJ MCMC chains did not converge, so MoM serves as the reported estimator). For each
model and moneyness bucket, 25,000 return paths are simulated and the realized
held-to-maturity return is compared against the simulated distribution to obtain a
finite-sample p-value. The p-value structure and the moneyness-return tables follow BCJ
(2009) directly. Regime analysis re-runs the test on four macro sub-periods.

## Data requirements

- **NSE F&O bhavcopy** — daily ZIPs from https://www.nseindia.com/all-reports (free).
- **Nifty 50 spot** — Yahoo Finance ticker `^NSEI` via `yfinance` (free).
- **91-day T-bill yield** — RBI (free, https://rbi.org.in).

## Reproduction

1. Install dependencies: `pip install -r ../../requirements.txt`
2. Configure environment variables (see `../../docs/data_sources.md`).
3. Run scripts in order:
   - `python 00_nifty_spot_and_tbill.py` — pulls Nifty 50 spot and the 91-day T-bill series.
   - `python 01_bhavcopy_downloader.py` — downloads and parses NSE F&O bhavcopy records.
   - `python 02_nifty_options_panel.py` — builds the held-to-maturity monthly put panel.
   - `python 03_mcmc_estimation.py` — estimates BS / SV (MCMC) / SVJ (MoM) parameters.
   - `python 04_option_simulation.py` — generates 25,000 paths and finite-sample p-values.
   - `python 05_regime_split.py` — produces the regime sub-period results.
4. Final outputs land in `tables/` — these are the tables that appear in the paper.

## Reproducibility seed

All scripts set `np.random.seed(20260509)` for reproducibility. Any deviation produces
non-matching numerical results.

## Citation

```bibtex
@unpublished{pillai2026niftyputs,
  author = {Pillai, Sumin},
  title  = {Nifty 50 Index Put Option Mispricing: A BCJ-Style Test 2015--2025},
  year   = {2026},
  note   = {Working paper, SSRN},
  url    = {https://github.com/SuminPillai/india-derivatives-research}
}
```
