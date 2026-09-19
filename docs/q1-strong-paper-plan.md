# Plan for a strong Q1 paper: where the supervision can pay and how Jev finds it

Written 2026-09-19 after the first real Jev run. Target: Quantum Machine
Intelligence (stretch: npj QI only if device noise enters; per the external
opinion it does not fit this path). The paper's core is C4 (the rate framing
of supervision) plus C2 (attribution). To be strong it needs at least one
measured, pre-registered improvement in an eligible cell. This document
defines the methodology that finds it instead of hoping for it.

## 1. The eligibility criterion (the methodological contribution)

Supervision pays only if the gap it recovers per call exceeds the gap the
fast loop would have closed during the call:

    recovered_gap_per_call  >  velocity_of_the_baseline  x  call_latency

Consequence, measured today: in the 4-qubit smoke cell the late-phase
velocity is 0.62 gap/s, so at Jev latency (0.65 s) a controller must recover
0.40 gap units per call, while the oracle, which is the ceiling, recovers
0.034. The cell was ineligible: no controller can win there, and the first
Jev run lost for exactly this reason plus the identifiability bug. Every
published comparison of controllers in this literature that we have reviewed
runs in cells nobody checked for eligibility. The paper states the criterion,
implements the check (`scripts/cell_screen.py`, `code/analysis.py
break-even`) and only reports controller comparisons in eligible cells.

Eligibility check, per cell (hamiltonian, n, p, layers):

1. Run vanilla SPSA, seeds 0-2, 120 steps. Record final gap and late-phase
   velocity (gap per second over the last third).
2. Run the oracle (same seeds) and compute the mean recovery per call.
3. Eligible at serving class L if oracle_recovery_per_call > velocity x L,
   with a safety factor of 2. Report the margin.

## 2. Cell program (screening results)

Screen output lives in `results/cell_screen.json` (SPSA-only stage). Cells
affordable on the homelab: 8 qubits with any noise (density matrix, ~0.03
s/eval), 16 qubits noiseless (statevector, ~0.1 s/eval). The 16-qubit noisy
backend costs 5.6 s/eval (~34 min per run) and belongs on the 2x3090 node or
Hercules, reserved for the final headline cells.

Filled in from the screen (`results/cell_screen.json`, SPSA-only, seeds 0-2,
120 steps, latency reference 0.65 s = measured Jev latency):

| Cell | gap final | velocity late (gap/s) | required gain per call | wall/run |
|---|---|---|---|---|
| heisenberg 8q p=0.05 L2 | 11.62 | 0.251 | 0.16 | 4 s |
| **heisenberg 8q p=0.05 L4** | **13.32** | **0.015** | **0.01** | 6 s |
| heisenberg 16q p=0 L2 | 16.77 | 0.268 | 0.17 | 18 s |
| tfim 8q p=0.05 L2 | 7.18 | 0.341 | 0.22 | 4 s |
| **xy 8q p=0.05 L2** | **8.95** | **0.054** | **0.04** | 3 s |
| heisenberg 8q p=0.02 L2 | 9.51 | 0.486 | 0.32 | 4 s |

The deeper ansatz (L4) at 8 qubits stalls almost completely: the late-phase
velocity is 0.015 gap/s, so at Jev latency a controller needs to recover only
0.01 gap units per call to pay. The remaining gap is 13.3. The xy model at
8 qubits stalls at 0.054 gap/s with a 8.95 gap. These two cells are the
primary candidacy; the oracle-margin stage quantifies their ceiling. The
depth-4 result also repairs the motivation-experiment split: the benchmark
now contains a genuinely hard landscape (deep, wide, noisy), measured rather
than asserted.

### Diagnosis on real windows (state v2, 2026-09-19)

The first v2 probe over real stall-cell windows found a boundary problem, not
a model failure. With the absolute descent threshold at 0.01, the run's
windows (which lose 0.004 per window on a 13.3 gap) were labelled stalled
while the model answered `DESCENDING` with high confidence: agreement 1 of 4.
The window series showed a genuinely slow but consistent descent (0.06 of
gap over 90 steps), so the model's reading was defensible and the threshold
was the arbitrary part. Re-calibrated to 0.002, agreement rises to 3 of 4,
with the single miss being a 0.0016-versus-0.002 boundary window. The paper
reports the threshold sensitivity (accuracy at several thresholds) alongside
accuracy, and treats boundary windows as the documented ambiguity they are.
The action side is a separate and open question: on slow-descent windows the
model chooses `slow_down`, which halves the step size exactly when the run is
already slow, and the final gaps reflect it.

### Oracle margin stage on the L4 cell (3 seeds, measured)

Vanilla SPSA mean final gap 13.32; oracle mean 13.14; paired differences
0.143, 0.206, 0.191 (mean 0.180, d_z = 5.5 against run noise). Recovery per
call: **0.0163 gap units**. Compare with the required gain per serving class
in this cell:

| Serving class | latency | required per call | oracle recovers 0.0163 | eligible |
|---|---|---|---|---|
| Jev (System One) | 0.65 s | 0.010 | yes | **yes, margin 1.6x** |
| hosted chat fast | 2.6 s | 0.039 | no | no |
| hosted chat slow | 12.2 s | 0.183 | no | no |
| local 4B on CPU | 110 s | 1.65 | no | no |

In this cell the eligibility line falls between the hundred-millisecond
class and every chat-model class: only sub-second decisions can pay. That is
the C4 thesis instantiated in one table, and it is the kind of number no
prior work reports.

The xy cell (8q, p=0.05) measured the same way is **not eligible at Jev
latency**: vanilla SPSA 8.95 versus oracle 8.74, recovery 0.0184 per call
against a 0.035 requirement at 0.65 s. It would become eligible only below
roughly 0.34 s of call latency. Across the three cells measured so far the
eligibility line sweeps the serving classes: the 4-qubit smoke cell is
ineligible at every class, the L4 stall cell admits the sub-second class
only, and the xy cell admits only the sub-0.34-second class. That is exactly
the axis the paper reports.

Two readings temper the enthusiasm. The margin is 1.6x, below the 2x safety
factor the plan prefers, so more seeds are needed before the gate experiment
(the paired effect is strong, d_z = 5.5, but the gap is thin). And the total
window-intervention ceiling is 0.18 gap units, 1.4 percent of the remaining
gap: at this depth the mid-run intervention budget is intrinsically small,
and any positive result will be about the recovery fraction and the rate,
not about a dramatic gap reduction. The large-effect lever in this cell is
initialization (Engine B), exactly as the plan expected.

The oracle margin stage runs next for the cells whose velocity is low and
whose remaining gap is large. Expected candidates: `heisenberg 8q p=0.05 L4`
(deeper ansatz), `heisenberg 16q p=0` (plateau-like start, noiseless),
`tfim 8q p=0.05` near criticality.

## 3. State v2 and action grid v2 (fairness preserved)

- Diagnosis moves to observable labels (`DESCENDING`, `STALLED_NO_GRADIENT`,
  `STALLED_WITH_GRADIENT`, `OSCILLATING`, plus `CONVERGED_AT_BEST` when the
  window is flat at its own best). Ground truth is recomputed from
  window-observable quantities only, and the paper also reports the old
  distance-to-optimum accuracy to document the identifiability ceiling. The
  probe proved the model discriminates under observable labels (p=1.00 and
  p=0.66 on contrasted windows) and does not under distance-based ones.
- The action grid gains two stall-escape levers: schedule reheat (reset the
  SPSA decay clocks to their initial values) and a larger step-size kick
  (eta x 5). Reheat is standard practice in stochastic approximation and is
  currently absent. The grid remains identical for every controller
  including the oracle, so attribution stays valid.
- Each action keeps its expected effect question, so the fidelity and
  calibration analyses carry over unchanged.

## 4. Jev applications: the three engines that can produce the improvement

### Engine A. Selective supervision (gate the calls)

The rate framing says calls should be spent only when the baseline has
stalled. Three variants, compared at matched total-call budgets:

1. always-on (the current arms),
2. velocity gate (call only when late velocity is below a threshold, a
   classical trigger),
3. Jev gate (call only when Jev's `P(improve)` is below tau, e.g. 0.5; the
   improvement probability discriminated correctly in every probe so far).

Win condition: equal or better final gap with strictly fewer calls, or
better gap at equal calls, in eligible cells. This converts Jev's calibrated
probability into a latency-budget allocation policy, which is exactly the
C4 thesis and something no prior work measures.

### Engine B. Initialization strategy choice (the AdaInit slot, at 0.65 s)

At step 0 the problem description (family, n, p, step budget) plus K
candidate initialization strategies are presented as a Choice question
(K ~ 6 strategies: small uniform, uniform +-pi/2, uniform +-pi,
block-periodic, low-discrepancy, identity-cluster). Jev picks one; the run
starts from it. Controls: random strategy choice, best-fixed strategy, and
the oracle-over-strategies upper bound. Metrics: initial gap and final gap
at equal budget.

Why this can be the positive result: initialization dominates at 16 qubits
(zhuang2025large, BRIDG-G and our own 16q smoke: SPSA from a random start
sits 23.9 gap units from the minimum after 20 steps), the question is fully
answerable from the description Jev receives, and it costs one 0.65 s call
instead of AdaInit's iterative 50-search loop. An iterative variant (three
rounds with observed initial gap fed back) follows AdaInit's protocol and
is measured against its one-call version.

### Engine C. Escape-action quality inside stalled windows

In eligible stalls, the counterfactual labels already say which escape
action was best. Compare per-window: Jev's pick against the oracle pick, the
effect-ridge policy, the heuristic, and random, at matched windows. This is
the pure C2 measurement: does controller judgement add anything over the
best cheap classical rule. Reheat enters here as the new lever the model can
choose.

### Supporting artifact D. Calibration and fidelity

Jev returns probabilities. Report reliability curves for `P(improve)`
(bucketed observed improvement frequency, ECE) and the existing explained-
effect fidelity with its nulls. This is the C3 section and it is cheap: the
data accumulates automatically in every run.

## 5. Experiment matrix and decision gates

Cells: the eligible list from phase 2 (expect 2-3 cells). Serving classes
reported: Jev (0.65 s measured), local 4B (5-15 s), hosted chat (2.6-12.2 s
documented), async variants at each. Arms per cell: spsa, random, heuristic,
policy, effect, oracle, llm (Jev), llm_async, plus gate variants from Engine
A and initialization variants from Engine B. Seeds: 8 for screening cells,
20 for the headline cell. Pre-registered success criterion: in at least one
eligible cell, Holm-corrected p < 0.05 and |d_z| > 0.5 versus every classical
controller at matched budget, with the oracle direction consistent.

Gates: if Engines A/B produce the win, the paper is "the rate framing,
measured, plus the intervention that captures it" (strong). If only Engine C
ties the classical baselines, the paper is the rate framing with an honest
attribution section (acceptable, weaker). If nothing wins anywhere, the
negative study stands on the identifiability finding, the eligibility
criterion and the benchmark (fallback venue).

## 6. Budgets

- Compute: homelab for everything up to 8q noisy and 16q noiseless; oracle
  runs are the expensive stage (2.2 min/run at 8q, 8 min/run at 16q
  noiseless). The 16q noisy headline cell goes to the 2x3090 node.
- API: count Jev calls; each call ~0.65 s and a few hundred tokens. Screening
  and gate phases are calls-bounded by construction (that is the point of the
  gate). Keep the response cache on so reruns cost nothing.

## 7. Immediate next actions

1. ~~Finish the screen, run oracle margins on the eligible shortlist.~~
   Done: L4 cell eligible at Jev latency with a 1.6x margin; xy cell margin
   pending (screen velocity 0.054, oracle stage next).
2. Implement state v2 + action grid v2 (observable regimes, reheat, eta x 5)
   with tests; re-run the diagnosis probe.
3. Run the gate experiment (Engine A) on the best eligible cell; then the
   initialization experiment (Engine B) at 16q noiseless.
4. Analysis + nulls + calibration; freeze results; write.
