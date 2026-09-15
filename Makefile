PY := .venv/bin/python

.PHONY: help install test sweep experiment aggregate lint clean

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-12s %s\n", $$1, $$2}'

install: ## Create the venv and install dependencies
	uv venv .venv --python 3.12
	uv pip install --python $(PY) -r requirements.txt

test: ## Run the fast test suite
	$(PY) -m pytest tests/ -q

sweep: ## Small local dataset generation (default config)
	$(PY) -m code.sweep --config configs/default.yaml --limit 4 --out data/sweep_smoke.jsonl

experiment: ## Closed-loop SPSA vs SPSA+LLM on the default config
	$(PY) -m code.experiment --config configs/default.yaml --out results/experiment.jsonl

aggregate: ## Build the paper tables from results
	$(PY) -m code.aggregate --results results/experiment.jsonl --out results/tables

lint: ## Byte-compile the package and tests
	$(PY) -m compileall -q code tests

clean: ## Remove caches
	rm -rf .pytest_cache __pycache__ code/__pycache__ tests/__pycache__
