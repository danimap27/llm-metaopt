"""Closed-loop runner: SPSA against SPSA plus a slow-loop controller.

Conditions
----------
- ``spsa``: vanilla SPSA, no interventions (baseline).
- ``spsa_random``: one random action from the candidate grid every N_w steps.
- ``spsa_heuristic``: deterministic rule-based controller on the telemetry.
- ``spsa_oracle``: counterfactual oracle action (upper bound, simulator only).
- ``spsa_policy``: classical logistic-regression policy trained on labeled telemetry.
- ``spsa_llm``: the LLM slow loop, queried with the JSON telemetry window.

Every condition shares the seed, the initial parameters and the Rademacher
perturbation stream of the fast loop, so runs are paired and can be compared
with paired tests. The controller draws from a separate RNG stream, which keeps
the fast-loop noise identical across conditions.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import time
from itertools import product
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml

from . import vqe
from .labeler import NO_OP, Intervention, default_candidates, label_state
from .llm_client import LLMConfig, decide as llm_decide
from .optimizer import SPSAConfig, apply_intervention, spsa_step
from .policy import LogisticPolicy
from .regimes import RegimeConfig
from .telemetry import build_window

CONDITIONS: Tuple[str, ...] = (
    "spsa",
    "spsa_random",
    "spsa_heuristic",
    "spsa_oracle",
    "spsa_policy",
    "spsa_llm",
)

_SLOW_LOOP_CONDITIONS = {"spsa_random", "spsa_heuristic", "spsa_oracle", "spsa_policy", "spsa_llm"}


# --------------------------------------------------------------------------- #
# Controllers
# --------------------------------------------------------------------------- #

def heuristic_decision(window: Dict[str, Any], candidates: Sequence[Intervention]) -> Intervention:  # noqa: ARG001
    """Rule-based controller over the telemetry (interpretable baseline).

    Rules (no access to ground truth):
    - no improvement and vanishing gradients -> boost the step size and inject noise;
    - no improvement with a live gradient -> halve the step size;
    - otherwise keep the schedule.
    """
    improvement = float(window["improvement"])
    grad_last = float(window["grad_norm"]["last"])  # type: ignore[index]
    if improvement <= 1e-3 and grad_last < 1e-3:
        return Intervention(eta_scale=2.0, noise_sigma=0.15)
    if improvement <= 1e-3:
        return Intervention(eta_scale=0.5)
    return NO_OP


def random_decision(rng: np.random.Generator, candidates: Sequence[Intervention]) -> Intervention:
    return candidates[int(rng.integers(0, len(candidates)))]


def oracle_decision(
    energy_fn: Callable[[Any], float],
    theta: np.ndarray,
    spsa_cfg: SPSAConfig,
    e_min: float,
    seed: int,
    lookahead: int,
    candidates: Sequence[Intervention],
    init_scale: float,
) -> Tuple[Intervention, Dict[str, Any]]:
    """Best counterfactual action, computed with the simulator (upper bound)."""
    label = label_state(
        energy_fn,
        theta,
        spsa_cfg,
        e_min=e_min,
        seed=seed,
        lookahead=lookahead,
        candidates=candidates,
        init_scale=init_scale,
    )
    action = Intervention(**label["best_action"])  # type: ignore[arg-type]
    return action, {"oracle_improvement": label["improvement"], "oracle_gap": label["best_gap"]}


# --------------------------------------------------------------------------- #
# Experiment configuration and loop
# --------------------------------------------------------------------------- #

def _clean_nan(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _clean_nan(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean_nan(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def run_closed_loop(
    condition: str,
    energy_fn: Callable[[Any], float],
    theta0: np.ndarray,
    spsa_cfg: SPSAConfig,
    e_min: float,
    *,
    steps: int,
    n_window: int,
    threshold: float,
    seed: int,
    init_scale: float = 0.5,
    lookahead: int = 20,
    candidates: Optional[Sequence[Intervention]] = None,
    llm_cfg: Optional[LLMConfig] = None,
    llm_session: Optional[Any] = None,
    policy: Optional[LogisticPolicy] = None,
) -> Dict[str, Any]:
    """Run one closed-loop trajectory and return its metrics."""
    candidates = list(default_candidates() if candidates is None else candidates)
    fast_rng = np.random.default_rng(seed)
    ctrl_rng = np.random.default_rng(seed + 1_000_003)
    theta = np.asarray(theta0, dtype=float).copy()

    history: List[Dict[str, Any]] = []
    events: List[Dict[str, Any]] = []
    eta_scale = 1.0
    n_llm_calls = 0
    n_changes = 0
    llm_latency_total = 0.0
    steps_to_threshold: Optional[int] = None
    llm_failures = 0

    started = time.perf_counter()
    for k in range(steps):
        if condition in _SLOW_LOOP_CONDITIONS and k > 0 and k % n_window == 0 and history:
            window = build_window(history, len(history) - 1, n_window, eta_scale=eta_scale, theta=theta)
            info: Dict[str, Any] = {}
            if condition == "spsa_random":
                action = random_decision(ctrl_rng, candidates)
            elif condition == "spsa_heuristic":
                action = heuristic_decision(window, candidates)
            elif condition == "spsa_oracle":
                action, info = oracle_decision(
                    energy_fn, theta, spsa_cfg, e_min, _event_seed(seed, k), lookahead, candidates, init_scale
                )
            elif condition == "spsa_policy" and policy is not None:
                action = policy.predict(window)
            elif condition == "spsa_llm":
                if llm_cfg is None:
                    action = NO_OP
                else:
                    response = llm_decide(window, llm_cfg, session=llm_session)
                    n_llm_calls += 1
                    if response.get("ok"):
                        llm_latency_total += float(response["latency_s"])
                        payload = response["decision"]
                        action = Intervention(
                            eta_scale=float(payload["action"]["eta_scale"]),
                            noise_sigma=float(payload["action"]["noise_sigma"]),
                            restart=bool(payload["action"]["restart"]),
                        )
                        info = {
                            "diagnosis": payload["diagnosis"],
                            "justification": payload["justification"],
                            "latency_s": response["latency_s"],
                        }
                    else:
                        llm_failures += 1
                        action = NO_OP
            else:
                action = NO_OP

            eta_scale = float(action.eta_scale) if action.eta_scale > 0 else 1.0
            is_noop = action == NO_OP
            if not is_noop:
                n_changes += 1
            theta = apply_intervention(
                theta,
                eta_scale=1.0,
                noise_sigma=float(action.noise_sigma),
                restart=bool(action.restart),
                rng=ctrl_rng,
                init_scale=init_scale,
            )
            events.append(
                {
                    "step": k,
                    "condition": condition,
                    "action": action.to_dict(),
                    "eta_scale": eta_scale,
                    "is_noop": is_noop,
                    **info,
                }
            )

        theta, record = spsa_step(energy_fn, theta, spsa_cfg, k, fast_rng, eta_scale)
        history.append(record)
        if steps_to_threshold is None and abs(float(record["energy"]) - e_min) <= threshold:
            steps_to_threshold = k

    final_energy = float(energy_fn(theta))
    wall_s = time.perf_counter() - started
    return {
        "condition": condition,
        "final_energy": final_energy,
        "final_gap": abs(final_energy - e_min),
        "best_gap": min(abs(float(r["energy"]) - e_min) for r in history),
        "steps_to_threshold": steps_to_threshold,
        "n_interventions": len(events),
        "n_changes": n_changes,
        "n_llm_calls": n_llm_calls,
        "n_llm_failures": llm_failures,
        "llm_latency_total_s": llm_latency_total,
        "llm_latency_mean_s": (llm_latency_total / n_llm_calls) if n_llm_calls else None,
        "wall_s": wall_s,
        "energy_curve": [float(r["energy"]) for r in history],
        "events": events,
    }


def _event_seed(seed: int, k: int) -> int:
    return int(seed) * 1000 + int(k)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def load_config(path: str | pathlib.Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_runs(cfg: Dict[str, Any], conditions: Sequence[str]) -> List[Dict[str, Any]]:
    sweep = cfg["sweep"]
    runs: List[Dict[str, Any]] = []
    for hamiltonian, n_qubits, noise_p, seed in product(
        sweep["hamiltonians"], sweep["n_qubits"], sweep["noise_p"], sweep["seeds"]
    ):
        for condition in conditions:
            runs.append(
                {
                    "run_id": f"{hamiltonian}_n{n_qubits}_p{noise_p:g}_s{seed}_{condition}",
                    "base_id": f"{hamiltonian}_n{n_qubits}_p{noise_p:g}_s{seed}",
                    "hamiltonian": hamiltonian,
                    "n_qubits": int(n_qubits),
                    "noise_p": float(noise_p),
                    "seed": int(seed),
                    "condition": condition,
                    "n_layers": int(cfg["ansatz"]["n_layers"]),
                    "init_scale": float(sweep["init_scale"]),
                }
            )
    return runs


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Closed-loop SPSA vs SPSA+controller experiments")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--out", default="results/experiment.jsonl")
    parser.add_argument("--conditions", default=",".join(CONDITIONS))
    parser.add_argument("--policy", default=None, help="path to a trained policy JSON (needed for spsa_policy)")
    parser.add_argument("--llm-base-url", default="http://localhost:11434/v1")
    parser.add_argument("--llm-model", default="qwen3.5:4b")
    parser.add_argument("--mock-llm", action="store_true", help="use the heuristic controller instead of the LLM (offline smoke test)")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    conditions = [item.strip() for item in args.conditions.split(",") if item.strip()]
    if args.mock_llm and "spsa_llm" in conditions:
        # Offline mode: the heuristic controller stands in for the LLM so the
        # pipeline can be exercised without an endpoint. Results carry
        # mock_llm=true and must never be reported as LLM results.
        conditions = ["spsa_heuristic" if item == "spsa_llm" else item for item in conditions]
        print("[experiment] mock LLM enabled: spsa_llm runs use the heuristic controller")
    runs = build_runs(cfg, conditions)
    if args.limit > 0:
        runs = runs[: args.limit]
    if args.dry_run:
        print(json.dumps(runs[:10], indent=2, ensure_ascii=False))
        print(f"... {len(runs)} runs in total")
        return 0

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    policy = LogisticPolicy.load(args.policy) if args.policy else None
    llm_cfg = LLMConfig(base_url=args.llm_base_url, model=args.llm_model)
    if "spsa_llm" in conditions:
        print(f"[experiment] LLM endpoint: {llm_cfg.base_url} model={llm_cfg.model}")

    spsa_cfg = SPSAConfig(**cfg["spsa"])
    regime_cfg = RegimeConfig(**cfg["regimes"])
    steps = int(cfg["spsa"]["steps"])
    n_window = int(cfg["windows"]["n_window"])
    lookahead = int(cfg["windows"]["lookahead"])

    with out_path.open("a", encoding="utf-8") as handle:
        for index, run in enumerate(runs, start=1):
            hamiltonian = vqe.build_hamiltonian(run["hamiltonian"], run["n_qubits"])
            e_min = vqe.exact_ground_energy(hamiltonian)
            energy_fn = vqe.make_energy_fn(hamiltonian, run["n_qubits"], run["n_layers"], noise_p=run["noise_p"])
            theta0 = vqe.random_initial_theta(
                np.random.default_rng(run["seed"]), run["n_qubits"], run["n_layers"], scale=run["init_scale"]
            )
            result = run_closed_loop(
                run["condition"],
                energy_fn,
                theta0,
                spsa_cfg,
                e_min,
                steps=steps,
                n_window=n_window,
                threshold=regime_cfg.tol_ok,
                seed=run["seed"],
                init_scale=run["init_scale"],
                lookahead=lookahead,
                llm_cfg=llm_cfg,
                policy=policy,
            )
            record = {"kind": "run", **{k: v for k, v in run.items() if k != "run_id"}, **result}
            record["run_id"] = run["run_id"]
            record["meta"] = {
                "e_min": e_min,
                "spsa": spsa_cfg.to_dict(),
                "n_window": n_window,
                "lookahead": lookahead,
                "llm": llm_cfg.to_dict() if run["condition"] == "spsa_llm" else None,
                "mock_llm": bool(args.mock_llm),
            }
            handle.write(json.dumps(_clean_nan(record), ensure_ascii=False) + "\n")
            handle.flush()
            print(
                f"[{index}/{len(runs)}] {run['run_id']}: gap={result['final_gap']:.4f} "
                f"best={result['best_gap']:.4f} interventions={result['n_interventions']} "
                f"({result['wall_s']:.1f}s)"
            )
    print(f"[experiment] output: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
