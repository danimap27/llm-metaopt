PY := .venv/bin/python

.PHONY: help install test sweep lint clean

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-10s %s\n", $$1, $$2}'

install: ## Crea el venv e instala dependencias
	uv venv .venv --python 3.12
	uv pip install --python $(PY) -r requirements.txt

test: ## Ejecuta los tests rapidos
	$(PY) -m pytest tests/ -q

sweep: ## Barrido local pequeno (config por defecto)
	$(PY) -m code.sweep --config configs/default.yaml --limit 4 --out data/sweep_smoke.jsonl

lint: ## Comprobacion de sintaxis de todo el paquete
	$(PY) -m compileall -q code tests

clean: ## Limpia caches
	rm -rf .pytest_cache __pycache__ code/__pycache__ tests/__pycache__
