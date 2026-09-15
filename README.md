# LLM-MetaOpt

Codigo experimental del paper *Language Models as Heuristic Meta-Optimizers and Intelligent Initializers for Variational Quantum Algorithms with Integrated Explainability*.

Documento maestro y estado del paper: `~/Obsidian/quantum-homelab/10-Projects/llm-metaopt/index.md`

## Idea

Arquitectura de doble bucle (fast-slow loop) para la optimizacion no convexa de VQAs en era NISQ:

1. **Warm-start**: un LLM lee la descripcion simbolica del Hamiltoniano y propone `theta_0`.
2. **Bucle rapido**: SPSA ejecuta `N_w` epocas a velocidad de milisegundos.
3. **Bucle lento**: cada `N_w` epocas el LLM recibe la telemetria `T_k`, diagnostica el regimen
   (convergencia, barren plateau, minimo local, meseta) y decide una intervencion
   (ajustar `eta`, inyectar ruido, reiniciar).

## Estructura

```
code/
  vqe.py         Hamiltonianos (Heisenberg, TFIM, XY), ansatz y energias exactas
  optimizer.py   Bucle rapido SPSA + aplicacion de intervenciones
  telemetry.py   Construccion de la ventana de telemetria T_k y su serializacion JSON
  regimes.py     Diagnostico de regimen con ground-truth de simulador
  labeler.py     Etiquetado contrafactual de la mejor intervencion (bandit)
  llm_client.py  Cliente del bucle lento (Ollama / OpenAI-compatible) con latencia y JSON schema
  sweep.py       CLI de generacion de datos (compatible con array jobs de Hercules)
configs/
  default.yaml   Configuracion por defecto del barrido
tests/
  test_smoke.py  Tests rapidos del pipeline
```

## Uso

```bash
# Entorno
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -r requirements.txt

# Tests
make test

# Generar datos (barrido local pequeno)
make sweep

# Un shard (para Hercules: array job)
.venv/bin/python -m code.sweep --config configs/default.yaml --shard 0 --nshards 1
```

## Reglas del proyecto

- Framework cuantico unico: **Qiskit 2.x + qiskit-aer** (nada de PennyLane).

- Todos los experimentos son reproducibles: semilla fija, `temperature=0` en el LLM y quant/hash del modelo anotados.
- Los datos crudos van a `data/` (no versionado); los resultados agregados a `results/`.
- El paper se escribe en ingles americano (reglas del vault). Este repo mantiene comentarios y docs en espanol.
- Cada cambio se commitea de forma atomica.
