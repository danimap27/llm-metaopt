# AdaInit code review (Zhuang and Guan, ACL 2026 Findings)

Reviewed on 2026-09-16 from the public repository
`github.com/junzhuang-code/AdaInit` (MIT License, cloned read-only).
This is the closest prior work to this paper (`zhuang2025large`, arXiv
2502.13166) and the one a connected reviewer is most likely to know.

## What the code actually does

- Task: QNN classification on Iris, Wine, Titanic and MNIST with PennyLane and
  PyTorch. The optimizer is Adam or SGD, not a variational energy minimizer.
- The LLM (hosted GPT or VertexAI Gemini, temperature 0.5, top_p 0.95,
  max 8192 output tokens) proposes the initial VQC parameters as one Python
  dictionary, parsed with `eval()` and up to 10 retries.
- Iterative search (50 searches by default): propose initial parameters, train
  for 30 epochs, measure the variance of the first parameter gradient across
  training batches, compute the expected improvement
  `EI = max(grad_var_new - grad_var_best, 0)`, and accept the candidate only if
  `EI > lb` with `lb = (n_search * n_qubits^6)^{-1}`. On acceptance, the
  previous parameters and the gradient variance are injected into the next
  prompt as feedback.
- Theory: a submartingale argument over the accepted-search sequence with the
  heuristic lower bound above.
- Baselines: eight classical initializations (uniform, normal, beta, Glorot,
  He, orthogonal, each with a data-dependent variant).

## Overlap and difference with this paper

Same claim family: LLMs help train variational quantum models against barren
plateaus. Different in every design dimension that this paper is built on:

- Initialization-only search versus supervision of the optimizer during the
  run, with an explicit intervention space (learning-rate scaling, Gaussian
  perturbation, restart) and no ansatz modification.
- Classification with Adam versus energy minimization with SPSA under noise.
- Hosted APIs versus reproducible open-weights endpoints with per-call latency
  accounting, including failed calls.
- No regime diagnosis, no justification protocol, no comparison against
  simulator ground truth, no attribution controls (random, rule-based,
  learned-classical, oracle), no cost-benefit condition for supervising.

The submartingale analysis governs their search loop. The cost-benefit
condition of `code/theory.py` governs the supervision loop. The two results
are complementary and should be cited side by side in the manuscript.

## What is reusable (MIT License, attribution required)

1. Classical initialization baselines for the warm-start comparison. The eight
   distributions map directly onto Qiskit and are the standard BP-mitigation
   reference points.
2. The feedback-slot prompt pattern (a fixed slot in the prompt that carries
   the previous candidate and its measured metric) for the warm-start phase.
3. The gradient-variance proxy for barren plateaus: variance across batches of
   the first component of the first-layer gradient. Cheap to compute and the
   exact metric the closest work optimizes, so reporting it makes the results
   directly comparable.
4. An external baseline condition. Their iterative search can be adapted to
   the VQE problems of this paper as a warm-start controller
   (`llm_init_iterative`), which gives reviewers the head-to-head comparison
   they will ask for.
5. The retry-then-validate output handling confirms the design choice of this
   repository: validate the structured output and degrade through schema
   variants instead of free-form parsing with `eval()`.

## Consequences for the roadmap

- Add the gradient-variance proxy to the telemetry window so every run reports
  the same barren-plateau metric as the closest work.
- Add the eight classical initializations as warm-start baselines.
- Add the `llm_init_iterative` condition to the sweep as the AdaInit-style
  external baseline, adapted to Qiskit and VQE with attribution.
- Keep the explicit differentiation paragraph in Related Work; ACL 2026
  Findings gives the work visibility in the NLP community, which makes the
  differentiation more urgent, not less.
