# Dataset schema

Both generators write JSONL with one object per line. Every record carries a
`kind` field.

## Window records (`code.sweep`)

Emitted at each slow-loop boundary of a plain SPSA run.

| Field | Type | Meaning |
|---|---|---|
| `kind` | `"window"` | Record type |
| `run_id` | string | `{hamiltonian}_n{n}_p{noise}_s{seed}` |
| `meta` | object | Run metadata (Hamiltonian, qubits, layers, noise, seed, `e_min`, SPSA config, `n_window`) |
| `step` | int | Fast-loop step at the end of the window |
| `energy` | float | Energy at that step |
| `gap` | float | `energy - e_min` (ground truth gap) |
| `window` | object | Telemetry window `T_k` (see below) |
| `diagnosis` | object | `label`, `reason` and `details` from `code.regimes.diagnose` |
| `label` | object | Present when the regime is not `CONVERGENCIA_OK`: best counterfactual action, gaps, reward and the full candidate list |

### Telemetry window (`window`)

| Field | Meaning |
|---|---|
| `n_window` | `N_w`, number of fast-loop steps in the window |
| `step_end` | Index of the last step |
| `energy_series` | Energies of the window, rounded to significant digits |
| `energy` | `first`, `last`, `min`, `mean`, `std`, `slope_per_step` |
| `improvement` | `first - last` (positive means the energy went down) |
| `grad_norm` | `last`, `mean`, `max` of the SPSA gradient estimate |
| `eta` | `a_k_last`, `c_k_last`, `eta_scale` |
| `theta` | `var`, `mean_abs`, `max_abs`, `dim` of the variational angles |

## Run records (`code.sweep`)

| Field | Meaning |
|---|---|
| `kind` | `"run"` |
| `final_energy`, `final_gap` | Final energy and its distance to `e_min` |
| `converged` | `abs(final_gap) <= regimes.tol_ok` |
| `labels` | Number of counterfactual labels written for this run |
| `wall_s` | Wall-clock duration |

## Closed-loop records (`code.experiment`)

| Field | Meaning |
|---|---|
| `kind` | `"run"` |
| `condition` | `spsa`, `spsa_random`, `spsa_heuristic`, `spsa_oracle`, `spsa_policy` or `spsa_llm` |
| `base_id` | Pairing key shared by all conditions of the same physical run |
| `final_gap`, `best_gap` | Distance to `e_min` at the end and at the best step |
| `steps_to_threshold` | First step reaching `regimes.tol_ok` (null if never) |
| `n_interventions` | Slow-loop calls performed |
| `n_changes` | Slow-loop calls whose action was not the no-op |
| `n_llm_calls`, `n_llm_failures` | LLM calls attempted and failed |
| `llm_latency_total_s`, `llm_latency_mean_s` | Latency accounting for the I/O tax analysis |
| `wall_s` | Wall-clock duration of the whole closed loop |
| `energy_curve` | Energy after every fast-loop step |
| `events` | One entry per slow-loop call with the action, the `is_noop` flag and, for the LLM condition, the diagnosis, the justification and the latency |

## Conventions

- NaN and infinity are written as `null`, so every file is strict JSON.
- Floats inside telemetry windows are rounded to four significant digits with
  `code.telemetry.compact_float`, which is the same representation the prompt
  receives. Full-precision values remain in `code.sweep` run records.
- `e_min` is the exact ground-state energy from dense diagonalization and is the
  reference used by every gap in the dataset.
