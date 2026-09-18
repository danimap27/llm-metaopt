# Experiment protocol

Written before running the final experiments, so the analysis plan is fixed in
advance and cannot be tuned to the results.

## Research questions

- **RQ1 (effectiveness).** Does a slow-loop controller improve the final energy
  gap of SPSA on small VQA problems, compared with vanilla SPSA?
- **RQ2 (attribution).** Is the improvement attributable to the controller's
  decisions, and not to the mere act of perturbing the optimization? Three
  controls answer this: random actions, a rule-based heuristic and a classical
  policy trained on the same telemetry.
- **RQ3 (diagnosis quality).** How accurately does the LLM's reported regime
  match the simulator ground-truth label of `code.regimes`?
- **RQ4 (cost).** What is the wall-clock and latency price of the slow loop, and
  how does it scale with the call frequency `N_w` and with the model choice?

## Conditions

| Condition | Controller | Notes |
|---|---|---|
| `spsa` | none | Baseline |
| `spsa_random` | uniform random action from the grid | Controls for "any perturbation helps" |
| `spsa_heuristic` | deterministic rules on the telemetry | Interpretable, zero-cost controller |
| `spsa_policy` | multinomial logistic regression on the telemetry | Classical learning control |
| `spsa_effect` | per-action ridge models of the counterfactual effect | Offline contextual bandit, the natural classical model class for choose-one-of-k from a telemetry vector |
| `spsa_oracle` | best counterfactual action (simulator) | Upper bound, not deployable; spends about 13x the quantum budget, recorded per run |
| `spsa_llm` | LLM with the JSON telemetry window | The proposed method, synchronous: the fast loop is blocked during the call |
| `spsa_llm_async` | same, but the fast loop keeps running | The action is applied to the state that exists when the response lands; each event records `staleness_steps`. Together with `spsa_llm` it measures the sync-versus-async trade-off that the cost-benefit claim rests on |

All classical controllers and the LLM see exactly the same flattened
telemetry window and choose from exactly the same seven-point action grid
(`eta_scale` in {0.5, 1.0, 2.0 or keep}, `noise_sigma` in {0.0, 0.05, 0.15},
optional restart). An intervention whose safeguard window closes with a worse
gap than at the decision point is reverted for every condition alike, and
each run records `n_reverted` and `n_energy_evals`.

Every condition shares the seed, the initial angles and the Rademacher
perturbation stream of the fast loop. Controllers draw from a separate RNG
stream, so the fast-loop noise stays identical across conditions and the runs
remain paired.

## Design

- Problems: Heisenberg, TFIM and XY Hamiltonians with 2, 4, 8 and 16 qubits,
  scaling the barren-plateau regime that the controller claims to address.
- Noise levels: `p = 0`, `0.02` and `0.05` depolarizing per gate.
- Evaluation backends, selected by qubit count and recorded per run:
  | n | p = 0 | p > 0 |
  |---|---|---|
  | 2 to 8 | `statevector_exact` | `density_matrix_exact` |
  | 16 | `statevector_exact` | `statevector_kraus_exact` |
  All three give exact expectation values under their models. The density
  matrix needs 4**n amplitudes (256 MB at 8 qubits, about 68 GB at 16), so
  noisy 16-qubit problems run on the statevector backend, which averages the
  Kraus branches of the noise model exactly. Shot noise is not simulated in
  these blocks, it enters in the hardware block with its finite shot budget.
- Measured cost on the reference machine (2 layers, one energy evaluation):
  16 qubits without noise is 0.1 s, so a 120-step run is about 24 s; 16 qubits
  with noise is about 5.6 s per evaluation, so a noisy run is about 22 min.
  The noiseless block runs on the homelab, the noisy 16-qubit block is
  sharded on Hercules or the dual-GPU node.
- Seeds: at least 20 per cell, giving 180 or more paired runs per condition.
- `N_w` sweep: 5, 10 and 20 fast-loop steps, to separate the effect of the
  controller from the effect of the call frequency.
- Fast loop: SPSA with the schedules of `configs/default.yaml` and 120 steps.
- Slow-loop lookahead for the oracle and for the counterfactual labels: 20 steps.

### Statistical treatment

- Primary endpoint: `final_gap` at a fixed budget of fast-loop steps.
- Secondary endpoints: `steps_to_threshold`, `best_gap`, `n_changes` and the
  LLM latency accounting.
- Reported statistics: mean with a 95% percentile bootstrap confidence interval
  (5000 resamples), paired sign-flip permutation tests against the `spsa`
  baseline (10000 permutations) and Holm-Bonferroni correction across the family
  of comparisons. Effect size reported as Cohen's `d_z` for paired samples.
- Power: to detect a paired standardized effect of `d_z = 0.5` with 80% power at
  alpha 0.05, roughly 34 pairs are needed. Cells with fewer pairs are reported
  as exploratory and labeled as such.
- Sensitivity: results are recomputed excluding the seeds that converge in the
  baseline condition, so the comparison is not driven by runs that were already
  solved.

## Reproducibility requirements

1. Pin the container or the exact dependency versions used for the final runs.
2. Record the LLM tag, quantization and backend host with every run
   (`meta.llm`), together with `temperature=0` and the seed.
3. Publish the prompts verbatim (`code/llm_client.SYSTEM_PROMPT`) and the JSON
   schema of the decision.
4. Publish the dataset, the aggregation script and the LaTeX generator, so every
   number in the paper can be rebuilt from the repository.
5. Report failed LLM calls (`n_llm_failures`): they are part of the cost.

## What would falsify the claims

- `spsa_llm` not better than `spsa_random` means the improvement is not
  attributable to the LLM's judgement.
- `spsa_llm` not better than `spsa_policy` means a classical model over the same
  telemetry captures the same information at lower cost.
- Diagnosis accuracy at chance level means the LLM is not reading the telemetry.
- Gains disappearing under `p = 0.05` mean the method does not survive realistic
  noise, which is the regime the paper motivates.
