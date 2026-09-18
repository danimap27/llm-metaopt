# Review prompt for an external AI code reviewer

Paste the text below into the reviewing AI, together with this repository
(archive or checkout). The prompt is self-contained: it explains the science
so the reviewer can judge the code against the claims, not just style.

---

You are reviewing the research codebase for a paper targeting Quantum Machine
Intelligence (stretch: npj Quantum Information). Review the code against the
scientific claims, not for style. Be adversarial: your job is to find reasons
a reviewer would reject the paper.

## The idea (judge the code against this)

Variational quantum algorithms (VQE family) fail because local optimizers
stall on barren plateaus and noise-induced rough landscapes, while calling a
language model at every iteration is infeasible because of inference latency
(2.6 to 12.2 s per hosted call, measured). The paper proposes a fast-slow
control architecture:

- Fast loop: SPSA runs N_w steps between controller queries, 2 energy
  evaluations per step, milliseconds each.
- Slow loop: every N_w steps an LLM reads one telemetry window (energy series,
  gradient norm, learning rate, theta dispersion), emits a JSON decision with
  a diagnosis from a closed list of 5 regimes, a two-sentence justification, a
  predicted effect, and one intervention from 3 levers: learning-rate scale,
  Gaussian noise on the angles, restart. Failed calls degrade to a no-op.
- Warm start (designed, not yet implemented): the LLM proposes initial angles
  from the symbolic Hamiltonian, with an optional iterative search loop.
- XAI protocol: the declared expected effect is compared against the observed
  effect in the next window, and the diagnosis against simulator ground truth.
- Cost-benefit theory: supervision pays iff the gap recovered per call exceeds
  the gap the fast loop would close during the call latency. Frequency cancels
  out; latency and per-call benefit decide.
- Controls for attribution: random actions, a rule-based heuristic, a classical
  logistic policy trained on identical telemetry, and a simulator oracle upper
  bound, all paired by seed, initial angles, and Rademacher stream.

The five claims: C1 drift of the objective (concept drift) with detection and
recovery, C2 attribution of improvement to controller judgement, C3 fidelity
of explanations validated by intervention, C4 the cost-benefit condition
measured across serving configurations (CPU, single GPU, dual GPU), C5 a
trained specialist that beats both the general model and the classical policy.

Closest prior art that the code must differentiate from: Zhuang and Guan
(AdaInit, arXiv 2502.13166, iterative LLM initialization search), AutoQResearch
(arXiv 2604.24283, LLM closed-loop policy search for combinatorial problems),
BRIDG-Q (initialization only), and wu2026building (arXiv 2609.09468, untrained
LLMs lose to classical optimizers, distilled policies recover 48 percent).

## Repository map

- `code/vqe.py` Hamiltonians (Heisenberg, TFIM, XY), hardware-efficient
  ansatz, energy evaluation. Exact backends only: statevector, density matrix
  (up to 8 qubits), and statevector with exact Kraus averaging of the
  depolarizing noise model (16 qubits). Ground truth by dense diagonalization
  up to 6 qubits and Lanczos above.
- `code/optimizer.py` SPSA with decay schedules and the intervention applier.
- `code/telemetry.py` builds the JSON window the controller sees.
- `code/regimes.py` ground-truth regime labels for the window states.
- `code/labeler.py` counterfactual labeling: for each window, roll out each
  candidate action with the same seed and pick the best (the training signal
  and the oracle's decision).
- `code/llm_client.py` the slow-loop client. Two endpoint dialects (OpenAI
  compatible and native Ollama with thinking disabled), JSON-schema decision
  contract, latency measured per call including failed attempts, raw failure
  snippets recorded.
- `code/llm_cache.py` exact replay cache.
- `code/experiment.py` the closed loop and the six conditions: spsa, spsa_random,
  spsa_heuristic, spsa_policy, spsa_oracle, spsa_llm.
- `code/sweep.py` dataset generation with the same counterfactual labeling.
- `code/drift.py`, `code/detectors.py`, `code/run_drift.py` the drifting-
  objective block with Page-Hinkley and mean-shift detectors on every step.
- `code/xai.py` fidelity of declared vs observed effects.
- `code/theory.py` the cost-benefit condition and its worst-case bound.
- `code/stats.py` bootstrap confidence intervals, paired permutation tests,
  Holm-Bonferroni correction, Cohen's d_z.
- `code/figures.py`, `code/aggregate.py` paper tables and figures from results.
- `tests/` 103 tests. `paper/` the manuscript skeleton with a TikZ figure.
- `docs/` experiment protocol, positioning vs prior art, serving configs,
  dataset schema, AdaInit code review, reproducibility notes.
- `configs/default.yaml` the sweep: 2, 4, 8 and 16 qubits, depolarizing
  p in {0, 0.02, 0.05}, 20 seeds, SPSA 120 steps, N_w = 10.

## How to run

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest tests/ -q          # 103 tests
.venv/bin/python scripts/lint_manuscript.py paper/main.tex
cd paper && latexmk -pdf -f main.tex
```

## Review focus, in priority order

1. Scientific correctness. Is the SPSA update and its schedules right? Are the
   paired-run guarantees real (same seed, same initial theta, same Rademacher
   stream across conditions)? Is the counterfactual labeling unbiased? Is the
   regime ground truth consistent with the telemetry? Are the statistics
   (bootstrap, permutation, Holm, d_z) implemented correctly?
2. Experimental fairness. Do the controls give the LLM no unfair advantage or
   disadvantage? Is anything leaked into the telemetry window that a classical
   controller could not compute? Is the oracle really an upper bound?
3. Reproducibility. Determinism given seeds, cache replay exactness, metadata
   recorded per run, backend names, no wall-clock dependence in results.
4. Scalability claims. Is the Lanczos ground energy correct? Is the Kraus
   averaging backend really exact under the depolarizing model? Is anything
   silently O(4^n)?
5. Honest failure accounting. Are LLM failures and their latency counted in
   the cost? Does a dead endpoint degrade to SPSA without corrupting the run?
6. Q1 readiness. For each claim C1 to C5, does the code as written support it?
   What single missing piece most endangers each claim?

## Constraints

- Propose minimal targeted fixes, not rewrites.
- Do not invent results or assume unmeasured quantities.
- If a claim has no completed run in the code, say "not yet supported", do not
  extrapolate from pilots.
- The paper writes in American English, passive voice, no semicolons, no em
  dashes, no bold or list environments in main prose.

## Output format

1. Findings ordered by severity (blocker, major, minor), each with file and
   line, why it endangers a specific claim, and a minimal suggested fix.
2. A verdict table for C1 to C5: supported, at risk, or not yet supported,
   with the one-sentence reason.
3. The three changes you would make first if you had one day.
