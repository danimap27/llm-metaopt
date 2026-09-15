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
| `spsa_oracle` | best counterfactual action (simulator) | Upper bound, not deployable |
| `spsa_llm` | LLM with the JSON telemetry window | The proposed method |

Every condition shares the seed, the initial angles and the Rademacher
perturbation stream of the fast loop. Controllers draw from a separate RNG
stream, so the fast-loop noise stays identical across conditions and the runs
remain paired.

## Design

- Problems: Heisenberg, TFIM and XY Hamiltonians with 2 to 4 qubits.
- Noise levels: `p = 0`, `0.02` and `0.05` depolarizing per gate.
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
