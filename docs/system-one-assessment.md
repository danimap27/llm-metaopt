# System One models (TypeSafe, Jev) as slow-loop backends

Assessment written 2026-09-19, the day TypeSafe AI announced System One models
and their first model Jev in early access. Sources: the announcement post, the
API quickstart, and the open-source LLM adapter repository
(`typesafe-ai/system-one-adapter-python`, MIT).

## What it is

A model class built for structured decisions instead of generated text. The
API is a question-answering contract, not a chat completion: you send a
`state` (our telemetry window) plus a set of typed questions, and receive
typed answers with probabilities. Three primitives:

- `Choice`, pick one option from a criteria map, returns `choice`,
  per-option `probabilities` and a `confidence`.
- `Score`, score the state on a rubric, returns a continuous `score`.
- `Noul`, a truth judgement, returns a probability in [0, 1].

Questions are evaluated in parallel and adding questions barely changes the
response time. The vendor claims around two orders of magnitude lower latency
and cost than chat LLMs on their workflow evaluations, with measured latencies
in the hundred-millisecond range.

## Why it matters for this paper

The decision contract of the slow loop maps onto the primitives almost
verbatim, with one structural upgrade and one loss.

| Current decision JSON | System One equivalent |
|---|---|
| `diagnosis` from a closed set of regimes | `Choice` over the criteria map, with a probability distribution over regimes instead of an argmax |
| `action` from the 7-point grid | `Choice` over the grid, or three atomic `Noul` questions (raise the step size, inject noise, restart) composed in code, which is the pattern the vendor recommends for multi-factor decisions |
| `expected_effect` in energy units | `Noul`: "the gap will improve over the next window", whose calibrated probability replaces the point estimate |
| `justification` (free text) | No equivalent. The audit trail becomes the probability vector, not a sentence |

Consequences for the claims:

1. **C4 becomes sharper.** The break-even condition is linear in call latency,
   so the serving class moves the requirement by two to three orders of
   magnitude, and this is now measured rather than argued. From the smoke cell
   (Heisenberg 4q, p = 0.02, 8 seeds, `results/tables_smoke/break_even_smoke.json`;
   oracle benefit per call 0.034 gap units; baseline gap velocity 3.13 gap/s
   over the full run and 0.62 gap/s in the last third):

   | Serving class | Latency | Required gain per call | Paying |
   |---|---|---|---|
   | System One class (100 ms) | 0.10 s | 0.31 (total phase), 0.06 (late phase) | close to break-even late in the run |
   | Hosted chat, fast (2.6 s) | 2.60 s | 8.1 (total), 1.6 (late) | no |
   | Hosted chat, slow (12.2 s) | 12.20 s | 38.2 (total), 7.6 (late) | no |
   | Local 4B on CPU (110 s) | 110.0 s | 344 (total), 68 (late) | no |

   The honest reading of the smoke cell is that in its current parameter
   regime even a perfect controller does not pay: the runs are still
   descending (late-phase velocity 0.62 gap/s), so interrupting SPSA costs
   more than the interventions recover. Supervision pays only where the
   baseline stalls, in the condition `benefit_per_call > velocity x latency`.
   Two design consequences follow. The cells that carry C4 must contain real
   stalls (higher noise, larger systems, or true plateaus), and supervision
   should be velocity-triggered, which is exactly what the rule-based
   heuristic controller can express.

2. **Type safety removes the dominant observable failure mode.** In our own
   runs the local model failed by emitting JSON without the required field,
   twice in two calls in the worst case, at 60 to 90 seconds per failure. A
   model that cannot emit a type error converts that entire class of
   reliability problems into decision-quality problems, which is what the
   paper actually wants to study.

3. **Calibration strengthens C3.** The explanation-fidelity claim currently
   compares a stated number against the observed effect. With calibrated
   probabilities it becomes a calibration curve: when the model says the gap
   improves with probability 0.8, does it improve in 80 percent of those
   calls. That is a harder, more quantitative fidelity test than text against
   outcome, at the cost of losing the free-text justification.

## Integration plan (ready, not yet built)

When early access exists, the work is small because the contract already
matches:

1. A third `api_style` in `code/llm_client.py` (`"systemone"`) posting to
   `https://api.typesafe.ai/v1/systemone` with the decision schema translated
   to `Choice` (diagnosis), `Choice` (action grid) and `Noul`
   (expected improvement), and the response mapped back to the existing
   `Intervention` plus the probability fields.
2. The run metadata records the model tag, the probability vector and the
   confidence per call, so the calibration analysis has everything it needs.
3. Conditions `spsa_llm` and `spsa_llm_async` run against it unchanged; the
   sync-versus-async and staleness measurements transfer directly.

The open-source adapter repository is useful in the other direction: it
wraps GPT-class LLMs behind the same System One contract with structured
outputs and probability normalization, which makes a three-way comparison
(decision-model API, LLM chat API, local open weights) a configuration
change rather than a rewrite.

## Decision needed

Early access requires an account at `console.typesafe.ai` and an API key.
That is the author's call, not a technical blocker.

## Verified: the pattern runs locally (2026-09-19)

Jev itself is API-only (no weights, no self-hosting option in the docs), but
the System One pattern was executed end to end on the homelab with the
official MIT adapter (`system-one-adapter[openai]`) pointed at the local
Ollama server (`llama3.2:3b`, OpenAI-compatible `chat_completions`,
`structured_outputs=True`). One `Choice` question (regime diagnosis over the
four identifiable labels) and one `Noul` question (will the gap improve) were
answered against a real telemetry window from this benchmark:

- Regime: `CONVERGENCIA_OK` with `probabilities` {1.0, 0.0, 0.0, 0.0} over
  the criteria map, which is a defensible reading of the window (flat energy,
  gradient norm 2e-4, improvement 1e-3).
- Improvement probability (`noul`): 0.0.
- Structured output validated on the first attempt
  (`n_retries_malformed_structure: 0`), with usage (`input_tokens`,
  `output_tokens`, `latency`) and the full attempt history in the response.
- Wall clock: 11.3 s on the homelab CPU. The same call on the 2x3090 node
  with vLLM is expected below one second; a small model on GPU is the only
  way to approach the hundred-millisecond latency class locally.

What the local path gives: the typed-question contract, schema-enforced
answers, probability vectors, retry and usage accounting, and full
reproducibility, all offline. What it does not give: the vendor's calibrated
probabilities (a prompted LLM distribution is miscalibrated by default, which
is itself a measurable claim for the calibration analysis) and the specialized
serving latency. The probe script is `scripts/systemone_local_probe.py`
(run it in a throwaway venv, it needs the optional extra).

## Measured latency budget and its reduction levers (CPU, 2026-09-19)

Raw Ollama timings (`/api/chat`, one question, about 150 input and 60 output
tokens) and adapter-level timings (two questions, 272 to 295 input tokens),
all on the homelab CPU (i5-8500). Decoding dominates, not loading:

| Configuration | Tokens out | Wall, warm | Note |
|---|---|---|---|
| `gemma3:1b`, raw endpoint | 51 | 3.4 s | 19.0 tok/s decode; cold call 4.4 s (prefill 0.9 s) |
| `llama3.2:3b`, raw endpoint | 64 | 7.5 s | 11.3 tok/s; cold call 11.3 s (prefill 3.7 s) |
| `gemma3:4b-it`, raw endpoint | 66 | 8.2 s | 8.9 tok/s |
| `gemma3:1b`, adapter, 2 questions | 118-166 | 9.0-13.1 s | full answer envelope is the cost |
| `gemma3:1b`, adapter, compact criteria | 70-74 | 5.9-6.6 s | short keys and terse instructions: about 40 percent less wall time |
| `llama3.2:3b`, adapter, compact criteria | 67 | 7.5 s | same lever on the 3B model |

The levers measured today, in order of effect: use the smallest capable model
(the 1B decodes twice as fast as the 3B), shrink the answer envelope (short
criteria keys, one question instead of two, fewer requested fields), and keep
the model resident (`keep_alive`; model load itself is 0.1 to 0.3 s, the cold
prefill is the larger 0.9 to 3.7 s cost). The CPU floor is around three to six
seconds per call with the current envelope and no amount of tuning crosses it,
because it is bounded by 8 to 19 tokens per second of decoding.

The reduction that matters is the GPU node: vLLM on one RTX 3090 decodes a 3B
model at more than a thousand tokens per second, so the same 70-token answer
lands in roughly 0.1 to 0.3 s including prefill, which is the hundred-
millisecond serving class. These configurations are exactly the serving axis
of the cost-benefit table, so the measurements feed C4 directly. An incidental
observation for the calibration analysis: the same window received different
diagnoses from different models (`CONVERGENCIA_OK` versus `MINIMO_LOCAL`),
which is the miscalibration the calibration curve is meant to quantify.

## MSI Prestige measurements (Ryzen AI 9 HX 370, CPU, 2026-09-19)

The same probe and breakdown were run against the MSI desktop through the
existing homelab-to-MSI Ollama tunnel, serving `gemma3:4b` from the user-level
Ollama (`~/.ollama`).

| Configuration | Wall, warm | Note |
|---|---|---|
| `gemma3:4b`, raw endpoint | 3.07 s | 22.6 tok/s decode; cold call 8.35 s (load 4.25 s from disk plus prefill 1.26 s) |
| `gemma3:4b`, adapter, compact criteria | 4.8-6.7 s | full answer envelope, same two questions as the homelab runs |
| `qwen3.5:4b`, adapter | not viable | the model emits its thinking into `reasoning` before the answer; through the chat-completions path the adapter burns the budget and stalls |

The MSI decodes about 2.5 times faster than the homelab (22.6 versus 8.9
tokens per second on the same 4B model), so a 4B answer on the MSI costs
roughly what a 1B answer costs on the homelab. Both remain one to two orders
above the hundred-millisecond class, which confirms that the serving node,
not model tuning, is where the latency reduction lives. The Radeon 890M is
unused (the Arch Ollama package ships no ROCm or Vulkan libraries); enabling
a GPU backend there would need a different package and sudo, and is noted as
an option rather than worked around. The user-level `ollama serve` on the MSI
was started for these measurements and left running; stop it with
`pkill -f "ollama serve"` on the MSI if the machine is needed idle.
