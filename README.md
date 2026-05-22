# India Derivatives Research

Reproducibility code and supplementary materials for working papers by Sumin Pillai
(Independent Researcher) on Indian financial markets, distributed via SSRN.

## Author

**Sumin Pillai** — Independent Researcher
- SSRN: https://ssrn.com/author=11504348
- ORCID: https://orcid.org/0009-0001-5302-5899
- Google Scholar: https://scholar.google.com/citations?user=nnhdOvsAAAAJ
- GitHub: https://github.com/SuminPillai

## Papers in this series

| ID | Title | Status | SSRN |
|----|-------|--------|------|
| P0 | Profitability of Delta-Neutral Option Strategies on Nifty 50 (2024 MBA dissertation) | LIVE | https://ssrn.com/abstract=6747280 |
| P1 | Nifty 50 Index Put Option Mispricing: A BCJ-Style Test 2015–2025 | DRAFT | TBD |
| P2 | Technical-Indicator Strategy Performance in Indian Equities: Joint Test of Market Efficiency 2015–2025 | DRAFT | TBD |
| P3 | Trading the Volatility Risk Premium on Nifty 50: Backtest with Realistic Frictions | DRAFT | TBD |
| P4 | LLM and Multi-Agent Systems in Algorithmic Trading: Taxonomy and Research Agenda | DRAFT | TBD |
| P5 | Jump Activity in Indian Index Futures: Affine vs Non-Affine Specifications | PLANNED | TBD |
| P6 | Indian ULIPs Under Stochastic Volatility | PLANNED | TBD |
| P7 | Reproducibility in Indian Quant Strategy Research | PLANNED | TBD |
| P8 | Family Business Group Effects on Stock-Return Comovement in Indian Markets | PLANNED | TBD |

## Reproducibility

Each `papers/<paper_id>/` directory contains:
- `README.md` — paper abstract + reproduction overview
- Numbered scripts (`01_...`, `02_...`) for the empirical pipeline
- `reproduction_notes.md` — environment, data sources, expected outputs

See `docs/reproducibility.md` for end-to-end instructions.

## Data sources

This repository contains code, not data. Data must be re-pulled from the original sources:
- **NSE F&O Bhavcopy archive** — https://www.nseindia.com/all-reports (free, daily ZIPs since 2003)
- **DefinEdge API** — for live and historical equity OHLC (subscription required)
- **Yahoo Finance** — Nifty 50 spot via `^NSEI` (free)
- **RBI** — 91-day T-bill yields (free, https://rbi.org.in)
- **IRDAI** — insurer prospectuses for P6 (free, https://irdai.gov.in)

See `docs/data_sources.md` for the exact URLs and pull procedures.

## Citation

If you use this code or these papers, please cite the relevant paper from the table above.
A `CITATION.cff` file provides machine-readable citation metadata.

For example, citing the repository as a whole:

> Pillai, S. (2026). India Derivatives Research: Reproducibility code for SSRN working papers
> on Indian financial markets [Software]. GitHub.
> https://github.com/SuminPillai/india-derivatives-research

## License

Code in this repository is released under the MIT License (see `LICENSE`).
Working-paper PDFs distributed on SSRN are subject to SSRN's distribution terms;
this repository contains code only, not PDFs.

## Contact

Open an issue on this repository or email suminpillai@gmail.com.
