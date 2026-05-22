# LLM and Multi-Agent Systems in Algorithmic Trading: Taxonomy and Research Agenda

**SSRN:** TBD
**Status:** Draft (survey paper — no empirical code)

## Abstract

The deployment of large language models (LLMs) and multi-agent systems (MAS) in algorithmic
trading has accelerated sharply since 2023, producing a fragmented literature spanning
natural-language processing, reinforcement learning, and market microstructure. This paper
provides a structured taxonomy of this emerging field, organising existing work along five
orthogonal dimensions: the functional role of the LLM within the trading stack (signal
generation, research synthesis, execution, or risk oversight); the information modality
processed (text, tabular, time-series, or multimodal); the learning paradigm (zero-shot
prompting, supervised fine-tuning, or reinforcement learning from market feedback); the
system architecture (single-agent versus coordinated multi-agent); and the evaluation
standard applied (in-sample, out-of-sample backtest, or live market). Applying this taxonomy
to the surveyed literature, the paper identifies open research gaps — including the absence
of a rigorous, cost-adjusted benchmark for LLM-generated signals, the underexplored failure
modes of MAS under adversarial or illiquid conditions, the tension between prompt-engineering
flexibility and reproducibility, and the near-complete absence of rigorous testing in
non-US markets — and closes with a research agenda.

## Contents

This is a survey/taxonomy paper. It contains no empirical pipeline. The reproducibility
artifact for this paper is its bibliography:

- `bibliography.bib` — the verified BibTeX reference list (54 entries) underpinning the
  survey. Entries were verified against publisher pages, arXiv, and SSRN.

## Citation

```bibtex
@unpublished{pillai2026llmagents,
  author = {Pillai, Sumin},
  title  = {LLM and Multi-Agent Systems in Algorithmic Trading: Taxonomy and Research Agenda},
  year   = {2026},
  note   = {Working paper, SSRN},
  url    = {https://github.com/SuminPillai/india-derivatives-research}
}
```
