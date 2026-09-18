# Response to the adversarial code review

Reviewed commit: `ab9eb07b6828376c6825c74a70a113165c79faf0`
Response commits: `38fb7ac (measurement layer), 12a342b (fair controller
measurement), 9604b0f (baselines), fd3130e (analysis layer), 31ac856 (manuscript), plus this
document. Verdicts below describe the repository after the fixes.

## Blockers

**B1 — 16-qubit noisy backend is not exact.** Confirmed independently (spread
3.4e-2 at 10 qubits on identical calls). Fixed: `method="density_matrix"` is
pinned explicitly up to 12 qubits (268 MB), the memory note was corrected by
the factor of 256 the review identified, and above 12 qubits the backend is
renamed `statevector_trajectory`, uses explicit `shots`, and is recorded as
sampled with its shots in the run metadata. Tests: `test_backend_selection`,
`test_trajectory_backend_approaches_density_matrix`.

**B2 — Page-Hinkley false-alarms on convergence.** Rewritten as the textbook
statistic on residuals from a local least-squares slope, two-sided. Verified:
zero alarms on a smooth exponential descent, alarm at the step of a level
shift, and the drift-block test no longer relies on the exempt-safeguard
behavior. Test: `test_page_hinkley_does_not_alarm_on_smooth_convergence`.

**B3 — Safeguard granted to the LLM only.** The safeguard now applies
uniformly to every controller condition in both the main closed loop and the
drift block, and each run records `n_reverted` and the snapshot needed to
audit a revert. Tests: `test_main_block_safeguard_reverts_and_counts`,
`test_safeguard_reverts_a_degrading_intervention`.

**B4 — No policy training code.** `code/train_policy.py` trains both the
logistic policy and the effect-ridge policy on `data/sweep.jsonl`, splitting
by seed and by Hamiltonian family, and reports held-out accuracy. The
`spsa_policy` and `spsa_effect` conditions raise if no trained policy is
supplied instead of silently degrading to SPSA.
Test: `test_policy_condition_raises_without_a_trained_model`.

**B5 — Policy sees less telemetry than the LLM.** `window_features` now
flattens exactly the window JSON the model receives (energy series plus every
scalar block including the new `progress` block). Test:
`test_policy_features_match_the_llm_window_fields`.

**B6 — Failed-attempt latency discarded.** `decide` returns the total wall
latency across all attempts alongside `latency_successful_s` and
`n_attempts`; the experiment records the total. Partial failures (the typical
case quoted in the review) now cost their real 60 to 90 seconds in the
accounting.

**B7 — Counterfactual rollout does not reproduce the advised trajectory.**
Rollouts now continue the deployed SPSA schedules (`k_offset`), the horizon
equals the in-force horizon (`lookahead = n_window = 10`), the Rademacher
stream derives from the run's own controller generator, and each candidate is
averaged over three draws (`label_repeats`) before the argmax, with the
per-candidate standard deviation recorded so the winner's curse is visible.

## Majors

**M1 — Recorded energy was the midpoint proxy.** `spsa_step` now records the
true energy of the new iterate at the price of one counted evaluation per
step. The stale final synthetic history record was removed.
Test: `test_recorded_energy_is_the_true_energy`.

**M2 — Pooled across incommensurable sizes.** `summarize` adds the gap
relative to `|E_min|` with its own bootstrap CI and declares the Holm family
explicitly, and `summarize_by_cell` reports every
(hamiltonian, n_qubits, noise_p) cell separately (written to
`results/tables/summary_by_cell.json`). Test:
`test_summarize_by_cell_keeps_problem_sizes_separate`.

**M3 — XAI measured from the wrong reference.** The experiment captures
`energy_pre` immediately before the intervention and the observed effect is
measured from it. Dropped events are counted and reported
(`n_dropped`, `n_events`). `explanation_report` adds the majority-sign null
and a permutation null for the signed accuracy. Pearson is reported as
undefined (None) for constant predictors, not zero.

**M4 — Two regimes not identifiable from the window.** The window carries a
dimensionless `progress` block (drop from start, gap above best so far), and
static blocks offer the model only the four regimes that can occur in a
static run, so the diagnosis is identifiable from its inputs. The drift block
enables the full enum including `CONCEPT_DRIFT`.

**M5 — Plateau threshold two orders of magnitude off and not comparable.**
`diagnose` normalizes the directional-gradient variance by the squared
coefficient-norm bound of the Hamiltonian and `eps_bp` is now relative
(1e-4) by default, comparable across the 2 to 16 qubit sweep. The
convergence of the ansatz depth question is documented as a framing risk
below.

**M6 — No evaluation counting.** `counting_energy_fn` wraps every energy
function; every run records `n_energy_evals`, so the oracle's roughly
thirteen-fold quantum budget is visible in the tables and matched-budget
reporting is possible.

**M7 — Action spaces unmatched.** The JSON schema restricts `eta_scale` to
{null, 0.5, 1.0, 2.0} and `noise_sigma` to {0.0, 0.05, 0.15}, the same grid
the classical controllers search, with client-side validation in
`parse_decision`.

**M8 — No-op reset the learning rate.** `eta_scale = None` now means keep
the current multiplier; no-ops and failed calls preserve it. Grid actions set
it explicitly.

**M9 — Training prefix excluded converged states.** Labeled windows are
sampled uniformly across the run (deterministic stride) and converged states
are included, so "do nothing" is learnable.

**M10 — Results appended without deduplication.** `experiment.py` resumes by
run_id like the sweep, and `load_results` deduplicates keeping the last
record.

**M11 — Abstract claimed a missing component.** The abstract no longer
mentions the warm start and states the 2, 4, 8 and 16 qubit range with the
three backend families.

## Minors

All accepted: the memory note is corrected; the sweep labeling streams derive
from `SeedSequence` children (no collisions); `expected_effect` is required
in `parse_decision`; Pearson constant handling as above; `code/analysis.py`
provides the break-even grid and the occlusion battery (the window is stored
on every LLM event for replay); the mock-LLM condition rewrite deduplicates
run_ids; and the TFIM two-qubit energy is anchored to the analytic
Jordan-Wigner value minus sqrt(5) in
`test_tfim_two_qubits_matches_jordan_wigner_closed_form`.

## Framing risks accepted as open points, not code defects

1. **Which pathology the noise model instantiates.** With a depth-2 ansatz
   the operative regime is rough-landscape and (above 12 qubits) sampled
   trajectories, not deep-ansatz barren plateaus. The paper now says this in
   the abstract (trajectory-sampled backends) and the protocol documents the
   normalized plateau metric. Deepening the ansatz for a dedicated plateau
   study is a planned ablation, not a defect fix.
2. **Controls answer attribution, not superiority over tuned optimizers.** The
   effect-ridge arm is the natural classical decision-maker over identical
   information. Comparisons against calibrated SPSA variants and Rotosolve
   are planned as the "strong optimizer" ablation before submission; the
   within-run window-boundary intervention (latency on the critical path)
   remains the axis of differentiation from AutoQResearch.
