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
