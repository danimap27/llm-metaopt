# Related work positioning

Compiled on 2026-09-15 from entries verified against the arXiv API and Crossref.
The verification script is `scripts/verify_references.py` and the resulting
entries live in `paper/refs.bib`.

## The space is already contested

Several 2025-2026 works overlap with parts of this paper. They must appear in
Related Work with an explicit statement of the difference.

| Work | What it does | What it does not do |
|---|---|---|
| Zhuang and Guan, `zhuang2025large` (2502.13166), code reviewed in `docs/adainit-analysis.md` | Iterative LLM search for initial parameters of QNN classifiers against barren plateaus, accepted at ACL 2026 Findings | No continuous supervision loop over a classical optimizer, no regime diagnosis during training, no explainability protocol, no attribution controls, no latency accounting |
| Sharma et al., `sharma2026autoqresearch` (2604.24283) | LLM-guided closed-loop policy search for adaptive variational quantum optimization, covering VQE, QAOA with warm-start and multi-angle variants, QRAO and PCE on MIS and CVRP | The model edits solver policy code between runs (about seventeen proposals over hours of simulation); it does not intervene inside a run at window boundaries with the serving latency on the critical path, and it reports no ground-truth regime labels, no explanation-fidelity metric and no sync-versus-async latency analysis |
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

## Candidate unoccupied directions

Evidence from arXiv full-text searches run on 2026-09-15 (queries and hits are
reproducible with `scripts/verify_references.py --search`).

| Direction | Search evidence | Status |
|---|---|---|
| Supervision under non-stationary objectives (concept drift) | `"language model" AND "drift" AND "quantum"` returns only API-drift and secure-coding papers, none about a drifting variational objective | Empty |
| Explanations of optimizer decisions validated by intervening | XAI for QML explains model predictions (`daskin2023explainability`, `power2024feature`, `rybotycki2025explainable`, `gilfuster2024opportunities`), nobody measures whether acting on the explanation produces the predicted effect | Empty |
| Training the supervisor (fine-tuning or reinforcement learning with a physics reward) | Fine-tuning exists for circuit generation (`jern2025agent`) and quantum reasoning, not for supervising an optimizer | Empty in this domain |
| Theory of when supervision pays | No cost-benefit or regret framework found for language-model supervision of numerical optimizers | Empty |
| Classical drift detectors as the fair baseline | Page-Hinkley and ADWIN are standard in streaming learning and unused in this literature | Available and uncontested |

Context for the trained-supervisor direction: `wu2026building` shows that unaided
language models remain below strong classical optimizers in low-budget black-box
optimization and that distilling a frozen text policy from practice recovers a
large part of the gap (48% regret reduction). That is both the motivation and the
template for a trained specialist in this paper.

## Reviewer expectations

Expect requests to benchmark against Zhuang and Guan and against AutoQResearch.
Both should be described in Related Work with the differences made explicit. If
their code is available, they are the natural external baselines. The
initialisation works (BRIDG-Q, QSeer, QAOA-GPT) are relevant to the warm-start
claim and should be cited where the initialisation intervention is introduced.
