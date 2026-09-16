# Serving the slow-loop model locally

The headline results of the paper use an open-weights model served on the machine
that runs the experiment, the MSI Prestige (Ryzen AI 9 HX 370, Radeon 890M,
32 GB LPDDR5x, Arch with Omarchy). A hosted model may appear only as an upper
bound. This document records every detail a reviewer would need to reproduce the
endpoint.

## 1. Choose the serving route

| Route | Command | Notes |
|---|---|---|
| llama.cpp with Vulkan | `llama-server -m <model.gguf> -ngl 99 -fa on -c 4096 --host 127.0.0.1 --port 8080` | Preferred on this APU. ROCm wins on prompt processing but loses on decode for gfx1150 |
| Ollama | `ollama serve` then `ollama pull <tag>` | Simpler and reproducible, uses port 11434. Record `ollama show <tag>` for the digest |

Both expose an OpenAI-compatible API, so the client code does not change. Record
which route was used with the results.

## 2. Model candidates

| Role | Candidate | Expected size at Q4_K_M |
|---|---|---|
| Headline (fast, local) | a 4B instruction-tuned open-weights model with structured-output support | about 3 GB |
| Generalist alternative | a 9B instruction-tuned model | about 6 GB |
| Upper bound (optional, slower) | a 26B to 35B mixture-of-experts with 3B to 4B active parameters | about 16 GB |

The mixture-of-experts option decodes as fast as a dense 8B on this APU because
only the active experts are read per token, which makes it a realistic upper
bound rather than a toy. Measure all three and report the Pareto frontier.

## 3. Pin the artifact

```bash
# llama.cpp route
sha256sum <model.gguf> | tee paper/prompts/model_hash.txt

# Ollama route
ollama show <tag> | tee paper/prompts/model_digest.txt
```

The hash or digest goes into the manuscript, together with the llama.cpp build
number or the Ollama version.

## 4. Measure the latency on the machine that runs the experiment

```bash
.venv/bin/python scripts/benchmark_llm.py \
    --base-url http://127.0.0.1:8080/v1 --model <model> --calls 20 \
    --out results/llm_latency.json
```

Rules that make the number honest:

1. Run the benchmark on the MSI, on AC power, with the performance profile
   selected. On battery the decode rate drops by roughly a third.
2. Run it while the experiment is running, not on an idle machine, because the
   inner loop competes for the same cores.
3. Report the distribution (median and p95), never a single call.
4. Report failures. A slow endpoint that sometimes dies is a different claim from
   a fast one that never fails.

## 5. Wire the endpoint into the code

`configs/default.yaml` carries the endpoint under `llm:`. The experiment runner
reads it and every run records the tag, the quantization and whether the response
came from the cache:

```yaml
llm:
  base_url: http://127.0.0.1:11434/v1
  model: qwen3.5:4b
  cache: results/llm_cache.jsonl
  replay: false
```

```bash
# Real endpoint
.venv/bin/python -m code.experiment --config configs/default.yaml --out results/experiment.jsonl

# Replay from the cache, no endpoint needed (this is what the reviewers can run)
.venv/bin/python -m code.experiment --config configs/default.yaml --replay --out results/experiment_replay.jsonl
```

## 6. Fallback if the MSI is unavailable

The homelab (16 GB) can serve the 4B model only. If a run is repeated there, say
so in the manuscript and report the endpoint of each block, because the latency
distribution differs between machines.
