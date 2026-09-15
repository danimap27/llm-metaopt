# Reproducibility

## Environment

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -r requirements.txt
```

Pinned versions used for the reference runs are recorded in
`requirements-lock.txt` (generate with `uv pip freeze --python .venv/bin/python`).

## Reference commands

```bash
# 1. Counterfactual dataset (one shard per HPC array task)
.venv/bin/python -m code.sweep --config configs/default.yaml --shard 0 --nshards 1 --out data/sweep.jsonl

# 2. Closed-loop comparison with a local LLM endpoint
.venv/bin/python -m code.experiment --config configs/default.yaml --out results/experiment.jsonl \
    --llm-base-url http://localhost:11434/v1 --llm-model qwen3.5:4b

# 3. Offline smoke test of the same pipeline (no endpoint, heuristic controller)
.venv/bin/python -m code.experiment --config configs/default.yaml --mock-llm --limit 6 --out results/ci.jsonl

# 4. Tables
.venv/bin/python -m code.aggregate --results results/experiment.jsonl --out results/tables
```

## Determinism

- Every run derives its angles from `numpy.random.default_rng(seed)` with the
  seed recorded in the run metadata.
- The fast-loop Rademacher stream depends only on the seed, which keeps runs
  paired across conditions.
- LLM calls use `temperature=0` and a fixed `seed` field. Local open-weight
  models are deterministic in practice for greedy decoding, but the backend
  version and quantization must be reported because they can change results.
- `spsa_random`, `spsa_heuristic` and `spsa_policy` are fully deterministic.

## What must be published with the paper

1. This repository at the commit hash used for the final runs.
2. `data/sweep.jsonl` and `results/experiment.jsonl` (or a DOI-backed archive).
3. The prompts and the decision schema (`code/llm_client.py`).
4. The model tag, quantization and host for every LLM condition.
5. `results/tables/summary.tex`, which is the source of the paper table.

## Known sources of nondeterminism

- LLM backends under load can return different token counts and latencies even
  at `temperature=0`. Latency is therefore reported as a distribution, never as
  a single number.
- AerSimulator results are deterministic for the density-matrix method used
  here, so noisy runs reproduce exactly on the same Qiskit version.
