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

from . import sweep as sweep_module
from . import vqe
from .labeler import NO_OP, Intervention, default_candidates, label_state
from .llm_cache import ResponseCache
from .llm_client import LLMConfig, decide as llm_decide, decide_cached
from .optimizer import SPSAConfig, apply_intervention, counting_energy_fn, spsa_step
from .policy import LogisticPolicy, load_policy
from .regimes import REGIMES, RegimeConfig
from .telemetry import build_window

CONDITIONS: Tuple[str, ...] = (
    "spsa",
    "spsa_random",
    "spsa_heuristic",
    "spsa_oracle",
    "spsa_policy",
    "spsa_llm",
)

_SLOW_LOOP_CONDITIONS = {
    "spsa_random",
    "spsa_heuristic",
    "spsa_oracle",
    "spsa_policy",
    "spsa_effect",
    "spsa_llm",
}


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
    rng: np.random.Generator,
    lookahead: int,
    candidates: Sequence[Intervention],
    init_scale: float,
    k_offset: int = 0,
) -> Tuple[Intervention, Dict[str, Any]]:
    """Best counterfactual action, computed with the simulator (upper bound).

    The rollout continues the deployed run: the SPSA schedules resume at the
    global step ``k_offset`` and the Rademacher stream derives from the run's
    own controller generator.
    """
    label = label_state(
        energy_fn,
        theta,
        spsa_cfg,
        e_min=e_min,
        rng=np.random.default_rng(rng.integers(0, 2**31)),
        lookahead=lookahead,
        candidates=candidates,
        init_scale=init_scale,
        k_offset=k_offset,
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
    cache: Optional[Any] = None,
    policy: Optional[LogisticPolicy] = None,
    safeguard_epsilon: float = 0.05,
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
    # Index into ``events`` of an intervention whose safeguard window is open.
    pending: Optional[int] = None
    for k in range(steps):
        # Safeguard window close: revert an intervention whose window ended with
        # a worse gap than at the decision point. Applies uniformly to every
        # controller condition, like in the drift block (review finding B3).
        if pending is not None and k - int(events[pending]["step"]) >= n_window:
            event = events[pending]
            gap_now = abs(float(history[-1]["energy"]) - e_min)
            if (
                gap_now > abs(float(event["energy_pre"]) - e_min) + safeguard_epsilon
                and event.get("theta_snapshot") is not None
            ):
                theta = np.asarray(event["theta_snapshot"], dtype=float)
                eta_scale = float(event["eta_before"])
                event["reverted"] = True
            pending = None
        if condition in _SLOW_LOOP_CONDITIONS and k > 0 and k % n_window == 0 and history:
            window = build_window(history, len(history) - 1, n_window, eta_scale=eta_scale, theta=theta)
            # Reference for the XAI audit: the energy immediately before the
            # intervention, so the observed effect is measured from the point
            # the decision was made. Costs one evaluation per window for every
            # controller condition, so the budget stays matched.
            energy_pre = float(energy_fn(theta))
            info: Dict[str, Any] = {}
            if condition == "spsa_random":
                action = random_decision(ctrl_rng, candidates)
            elif condition == "spsa_heuristic":
                action = heuristic_decision(window, candidates)
            elif condition == "spsa_oracle":
                action, info = oracle_decision(
                    energy_fn, theta, spsa_cfg, e_min, ctrl_rng, lookahead, candidates, init_scale, k_offset=k
                )
            elif condition in ("spsa_policy", "spsa_effect"):
                if policy is None:
                    raise ValueError(
                        f"condition {condition!r} requires a trained policy; run code/train_policy.py first"
                    )
                action = policy.predict(window)
            elif condition == "spsa_llm":
                if llm_cfg is None:
                    action = NO_OP
                else:
                    response = (
                        decide_cached(window, llm_cfg, cache)
                        if cache is not None
                        else llm_decide(window, llm_cfg, session=llm_session)
                    )
                    n_llm_calls += 1
                    if response.get("ok"):
                        llm_latency_total += float(response["latency_s"])
                        payload = response["decision"]
                        raw_eta = payload["action"]["eta_scale"]
                        action = Intervention(
                            eta_scale=None if raw_eta is None else float(raw_eta),
                            noise_sigma=float(payload["action"]["noise_sigma"]),
                            restart=bool(payload["action"]["restart"]),
                        )
                        info = {
                            "diagnosis": payload["diagnosis"],
                            "justification": payload["justification"],
                            "expected_effect": payload["expected_effect"],
                            "latency_s": response["latency_s"],
                            "n_attempts": response.get("n_attempts"),
                        }
                    else:
                        llm_failures += 1
                        llm_latency_total += sum(
                            float(attempt.get("latency_s") or 0.0)
                            for attempt in response.get("attempts", [])
                        )
                        action = NO_OP
                        info["llm_failed"] = True
                        info["llm_error"] = [
                            {"error": attempt.get("error"), "variant": attempt.get("used_json_schema")}
                            for attempt in response.get("attempts", [])
                        ]
            else:
                action = NO_OP

            # eta_scale None means "keep the current multiplier", so a no-op
            # or a failed call does not reset a previously granted change.
            eta_before = eta_scale
            if action.eta_scale is not None:
                eta_scale = float(action.eta_scale)
            is_noop = action == NO_OP
            if not is_noop:
                n_changes += 1
            theta_snapshot = theta.copy()
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
                    "eta_before": eta_before,
                    "energy_pre": energy_pre,
                    "theta_snapshot": None if is_noop else theta_snapshot.tolist(),
                    # The raw window is stored for LLM runs so the occlusion
                    # battery in code.analysis can replay each call field by field.
                    "window": window if condition == "spsa_llm" else None,
                    "is_noop": is_noop,
                    **info,
                }
            )
            pending = len(events) - 1

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
        "n_reverted": sum(1 for event in events if event.get("reverted")),
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
    parser.add_argument("--llm-base-url", default=None)
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--llm-timeout", type=float, default=None)
    parser.add_argument("--llm-max-tokens", type=int, default=None)
    parser.add_argument("--llm-no-think", action="store_true", help="force the /no_think hint")
    parser.add_argument("--llm-api-style", default=None, choices=["openai", "ollama"])
    parser.add_argument("--cache", default=None, help="response cache path (enables replay)")
    parser.add_argument("--replay", action="store_true", help="serve every call from the cache")
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
        # Mock substitution can collide with an explicit spsa_heuristic entry;
        # keep one run per condition so run_ids stay unique.
        conditions = list(dict.fromkeys(conditions))
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
    policy = load_policy(pathlib.Path(args.policy)) if args.policy else None
    llm_defaults = cfg.get("llm", {}) or {}
    llm_base_url = args.llm_base_url or llm_defaults.get("base_url", "http://127.0.0.1:11434/v1")
    llm_model = args.llm_model or llm_defaults.get("model", "qwen3.5:4b")
    cache_path = args.cache if args.cache is not None else llm_defaults.get("cache")
    replay = bool(args.replay or llm_defaults.get("replay", False))
    cache = ResponseCache(cache_path, replay_only=replay) if cache_path else None
    llm_timeout = args.llm_timeout or float(llm_defaults.get("timeout", 120.0))
    llm_max_tokens = args.llm_max_tokens or int(llm_defaults.get("max_tokens", 400))
    llm_cfg = LLMConfig(
        base_url=llm_base_url,
        model=llm_model,
        timeout=llm_timeout,
        max_tokens=llm_max_tokens,
        api_style=str(args.llm_api_style or llm_defaults.get("api_style", "openai")),
        no_think=bool(args.llm_no_think or llm_defaults.get("no_think", False)),
        extra_body=dict(llm_defaults.get("extra_body", {}) or {}),
        # Static objective: the model is not offered CONCEPT_DRIFT, which it
        # could never identify from a static-run window (review finding M4).
        regimes=tuple(name for name in REGIMES if name != "CONCEPT_DRIFT"),
    )
    if "spsa_llm" in conditions:
        print(
            f"[experiment] LLM endpoint: {llm_cfg.base_url} model={llm_cfg.model} "
            f"style={llm_cfg.api_style} no_think={llm_cfg.no_think} cache={cache_path} replay={replay}"
        )

    spsa_cfg = SPSAConfig(**cfg["spsa"])
    regime_cfg = RegimeConfig(**cfg["regimes"])
    steps = int(cfg["spsa"]["steps"])
    n_window = int(cfg["windows"]["n_window"])
    lookahead = int(cfg["windows"]["lookahead"])

    done_run_ids = sweep_module.read_done_run_ids(out_path)
    with out_path.open("a", encoding="utf-8") as handle:
        for index, run in enumerate(runs, start=1):
            if run["run_id"] in done_run_ids:
                print(f"[{index}/{len(runs)}] {run['run_id']}: already done, skipping")
                continue
            hamiltonian = vqe.build_hamiltonian(run["hamiltonian"], run["n_qubits"])
            e_min = vqe.exact_ground_energy(hamiltonian)
            shots = int(cfg["sweep"].get("shots", vqe.DEFAULT_TRAJECTORY_SHOTS))
            energy_fn, eval_counter = counting_energy_fn(
                vqe.make_energy_fn(hamiltonian, run["n_qubits"], run["n_layers"], noise_p=run["noise_p"], shots=shots)
            )
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
                cache=cache,
                policy=policy,
                safeguard_epsilon=float(cfg.get("safeguard_epsilon", 0.05)),
            )
            record = {
                "kind": "run",
                **{k: v for k, v in run.items() if k != "run_id"},
                "backend": vqe.energy_backend(run["n_qubits"], run["noise_p"]),
                "n_energy_evals": eval_counter["n"],
                **result,
            }
            record["run_id"] = run["run_id"]
            record["meta"] = {
                "e_min": e_min,
                "spsa": spsa_cfg.to_dict(),
                "n_window": n_window,
                "lookahead": lookahead,
                "llm": llm_cfg.to_dict() if run["condition"] == "spsa_llm" else None,
                "cache": cache_path if run["condition"] == "spsa_llm" else None,
                "replay": replay if run["condition"] == "spsa_llm" else None,
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


# --------------------------------------------------------------------------- #
# Drifting-objective block (E2)
# --------------------------------------------------------------------------- #

DRIFT_CONDITIONS: Tuple[str, ...] = ("drift_spsa", "drift_detector", "drift_random", "drift_llm")


def _trapezoid(values: np.ndarray) -> float:
    """Area under a sampled curve, numpy 1.x and 2.x compatible."""
    function = getattr(np, "trapezoid", None)
    if function is None:  # numpy < 2.0
        function = np.trapz  # type: ignore[attr-defined]
    return float(function(values))


def _drift_response_metrics(
    events: Sequence[Dict[str, Any]],
    gaps: Sequence[float],
    boundaries: Sequence[int],
    threshold: float,
) -> Dict[str, Any]:
    """Mean lag, per boundary, to the first real intervention and to recovery.

    Detection latency is counted from the true drift step to the first slow-loop
    call that actually changed the trajectory. Recovery is counted from the drift
    step to the first fast-loop step inside the convergence threshold of the new
    objective. Boundaries never reached by either event are skipped, so the means
    are honest about what happened rather than padded with the horizon.
    """
    detection_lags: List[int] = []
    recovery_lags: List[int] = []
    for boundary in sorted(int(value) for value in boundaries):
        action_step = next(
            (
                int(event["step"])
                for event in events
                if int(event["step"]) >= boundary and bool(event.get("changed", False))
            ),
            None,
        )
        if action_step is not None:
            detection_lags.append(action_step - boundary)
        recovery_step = next(
            (index for index in range(boundary, len(gaps)) if abs(float(gaps[index])) <= threshold),
            None,
        )
        if recovery_step is not None:
            recovery_lags.append(recovery_step - boundary)
    return {
        "n_boundaries": len(list(boundaries)),
        "n_detected": len(detection_lags),
        "n_recovered": len(recovery_lags),
        "detection_latency": float(np.mean(detection_lags)) if detection_lags else None,
        "recovery_steps": float(np.mean(recovery_lags)) if recovery_lags else None,
    }


def run_drift_loop(
    condition: str,
    objective: Any,
    theta0: np.ndarray,
    spsa_cfg: SPSAConfig,
    *,
    n_window: int = 10,
    threshold: float = 0.05,
    seed: int = 0,
    detector_threshold: float = 0.5,
    safeguard_epsilon: float = 0.05,
    llm_cfg: Optional[LLMConfig] = None,
    cache: Optional[Any] = None,
) -> Dict[str, Any]:
    """Track a piecewise-stationary objective and record the drift response.

    The safeguard closes every window by comparing the gap with the gap at the
    moment of the intervention. A degradation larger than ``safeguard_epsilon``
    reverts the angles, which bounds the total damage by the number of
    interventions times epsilon.
    """
    from .detectors import PageHinkley, WindowedMeanShift
    from .llm_client import decide_cached

    if condition not in DRIFT_CONDITIONS:
        raise ValueError(f"Unknown drift condition: {condition!r}. Options: {DRIFT_CONDITIONS}")

    candidates = list(default_candidates())
    fast_rng = np.random.default_rng(seed)
    ctrl_rng = np.random.default_rng(seed + 1_000_003)
    detector = PageHinkley(delta=0.01, threshold=detector_threshold) if condition == "drift_detector" else None
    mean_shift = (
        WindowedMeanShift(window=max(5, n_window), threshold=3.0, min_samples=max(5, n_window))
        if condition == "drift_detector"
        else None
    )
    detector_cooldown = max(5, n_window)
    last_detector_restart = -10**9

    theta = np.asarray(theta0, dtype=float).copy()
    history: List[Dict[str, Any]] = []
    events: List[Dict[str, Any]] = []
    boundaries = set(int(value) for value in objective.spec.boundaries())
    total_steps = int(objective.spec.total_steps)
    eta_scale = 1.0
    llm_calls = 0
    llm_latency = 0.0
    llm_failures = 0
    pending: Optional[int] = None

    def _close_window(index: Optional[int]) -> None:
        """Apply the safeguard to the window that just finished."""
        nonlocal theta
        if index is None:
            return
        event = events[index]
        if history and (float(history[-1]["gap"]) - float(event["gap_before"])) > safeguard_epsilon:
            theta = np.asarray(event["theta_snapshot"], dtype=float)
            event["reverted"] = True
        event.pop("theta_snapshot", None)

    for step in range(total_steps):
        if step > 0 and step % n_window == 0:
            _close_window(pending)
            pending = None
            window = build_window(history, len(history) - 1, n_window, eta_scale=eta_scale, theta=theta)
            window["step"] = step
            info: Dict[str, Any] = {"segment": objective.segment_of(step), "after_boundary": step in boundaries}
            action = NO_OP
            if condition == "drift_detector":
                # The classical controller acts on per-step alarms below, so this
                # window event is bookkeeping only and never enters the safeguard.
                info["bookkeeping"] = True
            elif condition == "drift_random":
                action = random_decision(ctrl_rng, candidates)
            elif condition == "drift_llm" and llm_cfg is not None:
                response = decide_cached(window, llm_cfg, cache) if cache is not None else llm_decide(window, llm_cfg)
                llm_calls += 1
                if response.get("ok"):
                    llm_latency += float(response["latency_s"])
                    payload = response["decision"]
                    raw_eta = payload["action"]["eta_scale"]
                    action = Intervention(
                        eta_scale=None if raw_eta is None else float(raw_eta),
                        noise_sigma=float(payload["action"]["noise_sigma"]),
                        restart=bool(payload["action"]["restart"]),
                    )
                    info.update(
                        {
                            "diagnosis": payload["diagnosis"],
                            "justification": payload["justification"],
                            "expected_effect": payload["expected_effect"],
                            "latency_s": response["latency_s"],
                            "n_attempts": response.get("n_attempts"),
                            "cached": bool(response.get("cached", False)),
                        }
                    )
                else:
                    llm_failures += 1
                    info["llm_failed"] = True
                    info["llm_error"] = [
                        {"error": attempt.get("error"), "variant": attempt.get("used_json_schema")}
                        for attempt in response.get("attempts", [])
                    ]

            changed = action != NO_OP
            theta_snapshot = theta.copy()
            if changed:
                theta = apply_intervention(
                    theta,
                    eta_scale=1.0,
                    noise_sigma=float(action.noise_sigma),
                    restart=bool(action.restart),
                    rng=ctrl_rng,
                    init_scale=objective.spec.init_scale,
                )
            # eta_scale None means "keep the current multiplier" (B7 fairness fix).
            if action.eta_scale is not None:
                eta_scale = float(action.eta_scale)
            events.append(
                {
                    "step": step,
                    "condition": condition,
                    "action": action.to_dict(),
                    "changed": changed,
                    "gap_before": float(history[-1]["gap"]) if history else 0.0,
                    "theta_snapshot": theta_snapshot.tolist(),
                    **info,
                }
            )
            # The safeguard applies uniformly to every controller condition, so
            # no arm is protected from its own harmful interventions while the
            # classical baselines absorb theirs.
            pending = len(events) - 1

        theta, record = spsa_step(
            lambda th, s=step: objective.energy(s, th), theta, spsa_cfg, step, fast_rng, eta_scale
        )
        history.append(
            {
                "step": step,
                # spsa_step already evaluated the true energy of the new iterate.
                "energy": record["energy"],
                "gap": record["energy"] - objective.e_min(step),
                "segment": objective.segment_of(step),
                "grad_norm": record["grad_norm"],
                "a_k": record["a_k"],
                "c_k": record["c_k"],
                "theta": theta.tolist(),
            }
        )

        if condition == "drift_detector":
            energy_now = float(history[-1]["energy"])
            alarm = bool(detector and detector.update(energy_now)) or bool(
                mean_shift and mean_shift.update(energy_now)
            )
            if alarm and step - last_detector_restart >= detector_cooldown:
                theta = apply_intervention(
                    theta,
                    eta_scale=1.0,
                    noise_sigma=0.0,
                    restart=True,
                    rng=ctrl_rng,
                    init_scale=objective.spec.init_scale,
                )
                last_detector_restart = step
                events.append(
                    {
                        "step": step,
                        "condition": condition,
                        "action": Intervention(restart=True).to_dict(),
                        "changed": True,
                        "alarm": True,
                        "gap_before": energy_now - objective.e_min(step),
                        "segment": objective.segment_of(step),
                        "after_boundary": step in boundaries,
                    }
                )

    _close_window(pending)
    gaps = [float(record["gap"]) for record in history]
    metrics = _drift_response_metrics(events, gaps, sorted(boundaries), threshold)
    return {
        "condition": condition,
        "final_gap": abs(gaps[-1]),
        "final_gap_by_segment": [
            abs(float(gaps[end - 1])) for end in _segment_ends(objective)
        ],
        "area_under_gap": _trapezoid(np.abs(np.asarray(gaps))),
        "n_interventions": len(events),
        "n_changes": sum(1 for event in events if event.get("changed")),
        "n_reverted": sum(1 for event in events if event.get("reverted")),
        "n_llm_calls": llm_calls,
        "llm_latency_total_s": llm_latency,
        "llm_failures": llm_failures,
        "energy_curve": [float(record["energy"]) for record in history],
        "gap_curve": gaps,
        "events": events,
        **metrics,
    }


def _segment_ends(objective: Any) -> List[int]:
    """Last step index of each segment."""
    ends: List[int] = []
    running = 0
    for segment in objective.spec.segments:
        running += int(segment["steps"])
        ends.append(running)
    return ends


if __name__ == "__main__":
    raise SystemExit(main())
