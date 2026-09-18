# Response to the external opinion (2026-09-18)

The opinion reviews commit `ab9eb07`, before the adversarial-review fixes
(`38fb7ac..434eb41`) and before the async condition (`d327fc1`). Disposition
below. Code-level findings are not re-argued here; see
`docs/review-response.md`.

## What was already fixed when this document was written

The measurement-layer defects (midpoint energy, sampled-but-labeled-exact
backend, restarted rollout schedules, discarded failed-attempt latency), the
drift detector (the one non-instrumentation defect), the missing policy
training code, the telemetry and action-space mismatches, and the abstract
claims (warm start, qubit range) are all corrected and tested. The
reporting-machinery-over-evidence asymmetry is being closed by the first
complete cell, run the same day this document was written:
`configs/smoke_cell.yaml` (Heisenberg, 4 qubits, p = 0.02, seeds 0-7) with
results in `results/smoke_cell.jsonl`.

## The synchronous question, which is now an experimental arm

The opinion identifies the load-bearing assumption of C4: the fast loop is
blocked while the slow-loop call is in flight. If the loop could run on, the
latency cost would vanish and the break-even framing would attack a strawman.

Two observations make this a contribution rather than a caveat. First, on real
hardware the assumption is physical: during a hosted call the QPU queue either
idles or keeps spending device time, so the blocked-accounting model is the
honest one for the IBM block. Second, the asynchronous variant is not free:
the action arrives addressed to a state that no longer exists. That
staleness-versus-latency trade-off is measurable, unclaimed in this
literature, and it is now a first-class condition of the benchmark.

Implemented as `spsa_llm_async`: the request fires on a thread at the window
boundary, SPSA keeps running, the action is applied to the state that exists
when the response lands, and every event records `staleness_steps` together
with the call latency. The sync arm (`spsa_llm`) already records the blocked
opportunity cost through `fast_step_wall_s` and per-call latency. The planned
figure is exactly the one the opinion proposes: sync blocked-cost versus
async staleness across serving configurations (CPU, one GPU, dual GPU).

## Positioning corrections accepted

`docs/related-work-positioning.md` now states what AutoQResearch actually
covers (VQE, QAOA warm-start and multi-angle, QRAO, PCE on MIS and CVRP) and
restates the differentiator as the location of the model relative to the run:
policy code edited between runs there, window-boundary interventions with
latency on the critical path here. Its declared future work (matched-budget
comparison against random search, Bayesian optimization, evolutionary search
and bandit controllers) is adopted as the checklist for the strong-classical
ablation; the effect-ridge arm already covers the offline-bandit cell of that
checklist.

## Recommendations recorded for the author, not decided here

1. **Cut the claims to C4 and C2.** The rate framing plus attribution is a
   complete paper. C1 and C3 become supporting sections with whatever the
   drift and fidelity blocks measure; C5 (the trained specialist) is the
   follow-up paper and is stronger once the dataset it trains on exists.
2. **Motivation split.** Either instantiate the pathologies (deeper ansatz
   until plateaus appear, finite-shot noise everywhere) or rewrite the
   motivation around small-scale rough landscapes, which is what the current
   sweep actually contains. The second is cheap and honest; the first is the
   stronger paper if the calendar allows one more experimental block.
3. **Venue.** Quantum Machine Intelligence with three to four months of
   directed work; npj Quantum Information is not on this path because it
   asks for device-coherent noise and a narrower, deeper contribution.

## In one sentence, agreeing with the original

The defects were in the wiring, and the wiring is fixed; what remains between
this repository and a defensible submission is one month of producing
evidence in the order the opinion prescribes: cells first, then claims.
