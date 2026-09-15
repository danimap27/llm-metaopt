# Related work positioning

Compiled on 2026-09-15 from entries verified against the arXiv API and Crossref.
The verification script is `scripts/verify_references.py` and the resulting
entries live in `paper/refs.bib`.

## The space is already contested

Several 2025-2026 works overlap with parts of this paper. They must appear in
Related Work with an explicit statement of the difference.

| Work | What it does | What it does not do |
|---|---|---|
| Zhuang and Guan, `zhuang2025large` (2502.13166) | Uses LLMs to mitigate barren plateaus in quantum neural networks | No continuous supervision loop over a classical optimizer, no regime diagnosis during training, no explainability protocol, no attribution controls |
| Sharma et al., `sharma2026autoqresearch` (2604.24283) | LLM-guided closed-loop policy search for adaptive variational quantum optimization | Searches solver policies for combinatorial optimization, not the energy-landscape optimization of a variational algorithm, and reports no ground-truth regime labels |
| Nguyen et al., `nguyen2026bridg` (2603.23979) | Barren-plateau-resilient initialisation with data-aware LLM-generated circuits | Initialisation only, no slow loop, no intervention space, no latency accounting |
| Jiang et al., `jiang2025qseer` (2607.27262) | Quantum-inspired graph network for parameter initialisation in QAOA | Initialisation only, specific to QAOA, no supervision during training |
| Tyagin and Safro, `tyagin2025qaoa` (2505.06810) | Generation of adaptive QAOA circuits with a language model | Circuit generation for QAOA, not runtime meta-control of a classical optimizer |
| Peng et al., `peng2025breaking` (2508.18514) | Reinforcement-learning initialisations for deep variational circuits | Reinforcement learning rather than language models, and initialisation only |
| Sakka et al., `sakka2026system` (2606.13380) | LLM system for autonomous variational circuit design | Circuit design, not runtime control of the optimization trajectory |
| Syah et al., `syah2026guided` (2504.16350) | LLM-guided initialisation for hybrid quantum-classical medical imaging | Application paper on initialisation, no optimizer supervision, no controls |
| Chen and Zhang, `chen2026driven` (2607.17498) | LLM-driven cross-paradigm design for quantum optimal control | Control-pulse design across paradigms, not variational energy minimization with a supervised fast loop |

## The delta this paper claims

1. A supervisory slow loop that runs during optimization on top of a classical
   stochastic optimizer (SPSA), with an explicit intervention space and without
   modifying the ansatz.
2. An explainability protocol evaluated against simulator ground truth (regime
   labels) instead of post-hoc rationalisation, with a fidelity metric.
3. Attribution controls that separate the controller's judgement from the mere
   effect of perturbing: random actions, a rule-based controller, a classical
   learned policy on identical telemetry and an oracle upper bound.
4. Latency accounting for the slow loop, including failed calls, which is the
   quantitative argument for supervising rather than iterating.

## Reviewer expectations

Expect requests to benchmark against Zhuang and Guan and against AutoQResearch.
Both should be described in Related Work with the differences made explicit. If
their code is available, they are the natural external baselines. The
initialisation works (BRIDG-Q, QSeer, QAOA-GPT) are relevant to the warm-start
claim and should be cited where the initialisation intervention is introduced.
