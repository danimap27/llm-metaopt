# First real LLM runs on the benchmark (Jev), 2026-09-19

First contact between the closed-loop benchmark and a real slow-loop model.
Jev (TypeSafe System One) ran through the new `systemone` dialect of
`code/llm_client.py`: 176 calls across the two LLM conditions, zero failures,
latency stable in the hundred-millisecond class. Cell: Heisenberg, 4 qubits,
p = 0.02, 8 seeds, 120 SPSA steps, `configs/smoke_cell.yaml`. Tables in
`results/tables_smoke/`.

## Results table (final gap, lower is better)

| Condition | Mean gap | p (Holm) | d_z | LLM latency |
|---|---|---|---|---|
| spsa | 4.107 ± 0.206 | -- | -- | -- |
| spsa_effect (offline bandit) | 3.906 ± 0.128 | 0.305 | -0.65 | -- |
| spsa_heuristic | 4.107 ± 0.206 | 1.0 | 0.00 | -- |
| spsa_llm (Jev, sync) | 4.391 ± 0.326 | 0.0486 | +1.33 | 0.66 s |
| spsa_llm_async (Jev) | 4.379 ± 0.330 | 0.0486 | +1.24 | 0.65 s |
| spsa_oracle | 3.732 ± 0.128 | 0.0486 | -1.71 | -- |
| spsa_random | 4.431 ± 0.557 | 0.892 | +0.34 | -- |

## Finding 1: the first contact is a loss, and the benchmark localizes it

Jev made the result significantly worse than plain SPSA (d_z = +1.33). The
event log explains the mechanism: 88 of 88 windows were diagnosed
`CONVERGENCIA_OK` and 77 of 88 actions were `slow_down`. Halving the step size
for most of the run slows the descent. This mirrors wu2026building: an
off-the-shelf model, deployed as-is, loses to the classical baseline. The
value of the run is that the loss is localized to a specific design surface.

## Finding 2: the root cause is identifiability, and the fix is proven

A contrast probe with three synthetic windows (steep descent 6.2 to 3.3,
flat-above with zero improvement, and a mid-height stall) was diagnosed
`CONVERGENCIA_OK` in all three cases, with probabilities up to 0.96 on a
steep descent. The regimes were defined by distance to the optimum, which the
controller cannot know from the window, so the diagnosis carried no
information. This is review finding M4, now demonstrated with the real model.

The same three windows with an observable taxonomy (DESCENDING,
STALLED_NO_GRADIENT, STALLED_WITH_GRADIENT, OSCILLATING) were diagnosed
DESCENDING with p = 1.00 on the steep descent and STALLED_NO_GRADIENT with
p = 0.66 on the flat window. The model discriminates when the question is
answerable from the state. The fix for the next iteration is therefore not
prompt tuning but a state change: the static block moves to observable
labels, the ground truth is redefined from window-observable quantities, and
the paper reports accuracy against both taxonomies (the unidentifiable one
documents the ceiling the review asked to quantify).

Corroborating evidence that the model itself is not the problem: its
improvement question discriminated correctly across the three probes
(P(improve) = 0.60, 0.18, 0.41) and its expected-gain score varied with the
scene (1.22, 0.05, 0.41).

## Finding 3: the latency class is real, end to end

Through our client, in production conditions (not a warm microbenchmark):
p50 = 0.649 s, min 0.569 s, max 0.823 s over 176 calls, zero failures. For
scale, the same call costs 2.6 to 12.2 s on hosted chat models, 60 to 110 s
on the local CPU 4B, and is projected below 1 s on the 2x3090 node. The
failure count is also the type-safety claim in action: the local model's
dominant failure mode (JSON missing required fields) simply cannot occur.

## Finding 4: asynchronous staleness at this scale is large

`spsa_llm_async` measured a mean staleness of 47.9 fast-loop steps (max 92)
with zero unapplied calls, and performed the same as sync (4.379 vs 4.391).
Two mechanisms explain why async did not pay here. Persistent actions
(a slow-down stays in force until overridden) make late application almost as
harmful as prompt application, and at 4 qubits a fast step costs ~7 ms, so a
0.65 s call is stale by construction. The sync-versus-async differentiation
needs cells where fast steps are expensive (more qubits, noisy backends,
device time), which is exactly the serving-versus-step-cost axis of C4.

## Finding 5: the null models deflated the fidelity headline

The explanation report shows signed accuracy 1.000 over 88 events, but the
majority-sign null is also 1.000 and the permutation p-value is 1.0: every
observed window effect was positive, so constant "it will improve" would
score the same. The honest numbers are Pearson r = 0.512 and MAE = 0.081
(model mean 0.079 against observed 0.141). This is the first time the null
models added in the review response changed the reading of a headline number.

## Next iteration

1. State/regimes v2 with observable labels, ground truth redefined
   accordingly, both taxonomies reported.
2. Re-run the same cell to measure whether diagnosis quality translates into
   performance, and keep the degenerate first contact as the documented
   baseline of what happens without the fix.
3. Push the async comparison to a cell with expensive fast steps.
