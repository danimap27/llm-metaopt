# LLM-MetaOpt

Experimental code for the paper *Language Models as Heuristic Meta-Optimizers and Intelligent Initializers for Variational Quantum Algorithms with Integrated Explainability*.

Master document and paper status (Spanish notes): `~/Obsidian/quantum-homelab/10-Projects/llm-metaopt/index.md`

## Idea

A fast-slow loop architecture for non-convex optimization of variational quantum algorithms in the NISQ era:

1. **Warm start**: an LLM reads the symbolic description of the Hamiltonian and proposes `theta_0`.
2. **Fast loop**: SPSA runs `N_w` epochs at millisecond speed.
3. **Slow loop**: every `N_w` epochs the LLM receives the telemetry window `T_k`, diagnoses the optimization regime (convergence, barren plateau, local minimum, energy plateau) and decides a macroscopic intervention (scale the learning rate, inject a Gaussian perturbation, restart).

## Layout

```
code/
  vqe.py         Hamiltonians (Heisenberg, TFIM, XY), ansatz and exact energies
  optimizer.py   SPSA fast loop and intervention application
  telemetry.py   Telemetry window T_k and its JSON serialization
  regimes.py     Regime diagnosis with simulator ground truth
  labeler.py     Counterfactual labeling of the best intervention (bandit)
  policy.py      Classical baseline policies trained on the same telemetry
  llm_client.py  Slow-loop client (Ollama or any OpenAI-compatible endpoint)
  experiment.py  Closed-loop runner: SPSA vs SPSA+LLM with baselines
  sweep.py       Dataset generation CLI (HPC array-job friendly)
  stats.py       Statistics for the paper tables (bootstrap CI, paired tests)
  aggregate.py   Result aggregation and LaTeX table generation
configs/
  default.yaml   Default sweep configuration
tests/
  test_smoke.py  Fast end-to-end pipeline tests
docs/
  dataset-schema.md   Dataset field reference
  experiment-protocol.md  Pre-registered experiment protocol
paper/
  main.tex       Paper skeleton (English)
  refs.bib       Bibliography
```

## Usage

```bash
make install            # create the venv and install dependencies
make test               # run the fast test suite
make sweep              # small local dataset generation

# One shard (HPC array jobs): 
.venv/bin/python -m code.sweep --config configs/default.yaml --shard 0 --nshards 1
```

## Project rules

- Single quantum framework: **Qiskit 2.x + qiskit-aer**.
- Every experiment is reproducible: fixed seeds, `temperature=0` for the LLM, and the exact model tag and quantization recorded with the results.
- Code, comments, docs and commit messages are written in English. The Obsidian notes stay in Spanish.
- Raw data goes to `data/` and aggregated results to `results/` (both untracked).
- One atomic commit per logical change, pushed after every commit.
