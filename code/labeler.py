"""Counterfactual labeling of the best intervention (bandit formulation).

For a frozen state (angles plus run seed) every candidate intervention is rolled
out and the final energy is measured after ``lookahead`` SPSA steps under the
identical Rademacher perturbation stream. The label is the candidate with the
smallest gap to the exact minimum, and the reward is its improvement over the
no-op action (eta_scale=1, no noise, no restart).

This labeling turns the problem into supervised or reinforcement learning with a
physical reward, without relying on the LLM's opinion.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Dict, List, Optional, Sequence, Union

import numpy as np

from .optimizer import SPSAConfig, apply_intervention, run_spsa

ArrayLike = Union[np.ndarray, Sequence[float]]


@dataclass(frozen=True)
class Intervention:
    """One macroscopic slow-loop action."""

    eta_scale: float = 1.0
    noise_sigma: float = 0.0
    restart: bool = False

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

    @property
    def name(self) -> str:
        parts = [f"eta{self.eta_scale:g}"]
        if self.noise_sigma > 0:
            parts.append(f"noise{self.noise_sigma:g}")
        if self.restart:
            parts.append("restart")
        return "_".join(parts)


NO_OP = Intervention()


def default_candidates() -> List[Intervention]:
    """Candidate grid: learning-rate scaling, thermal perturbation and restart."""
    return [
        NO_OP,
        Intervention(eta_scale=0.5),
        Intervention(eta_scale=2.0),
        Intervention(noise_sigma=0.05),
        Intervention(noise_sigma=0.15),
        Intervention(restart=True),
        Intervention(eta_scale=2.0, noise_sigma=0.15),
    ]


def evaluate_candidate(
    energy_fn: Callable[[ArrayLike], float],
    theta: ArrayLike,
    cfg: SPSAConfig,
    action: Intervention,
    e_min: float,
    lookahead: int,
    seed: int,
    init_scale: float = 0.5,
) -> Dict[str, object]:
    """Roll out one counterfactual from the frozen state."""
    rng = np.random.default_rng(seed)
    theta0 = apply_intervention(
        theta,
        eta_scale=action.eta_scale,
        noise_sigma=action.noise_sigma,
        restart=action.restart,
        rng=rng,
        init_scale=init_scale,
    )
    out = run_spsa(
        energy_fn,
        theta0,
        cfg,
        eta_scale=action.eta_scale,
        steps=lookahead,
        seed=seed,
    )
    final_energy = float(out["final_energy"])
    return {
        "action": action.to_dict(),
        "action_name": action.name,
        "final_energy": final_energy,
        "gap": abs(final_energy - e_min),
    }


def label_state(
    energy_fn: Callable[[ArrayLike], float],
    theta: ArrayLike,
    cfg: SPSAConfig,
    e_min: float,
    seed: int,
    lookahead: int = 20,
    candidates: Optional[Sequence[Intervention]] = None,
    init_scale: float = 0.5,
) -> Dict[str, object]:
    """Label the state with the best counterfactual intervention.

    Returns ``best_action``, the gap of every candidate, the no-op gap and the
    improvement (gap_noop - gap_best). An improvement <= 0 means that no
    intervention beats plain SPSA from that state.
    """
    candidates = default_candidates() if candidates is None else list(candidates)
    results = [
        evaluate_candidate(energy_fn, theta, cfg, action, e_min, lookahead, seed, init_scale)
        for action in candidates
    ]
    best = min(results, key=lambda r: float(r["gap"]))
    baseline = next((r for r in results if r["action_name"] == NO_OP.name), results[0])
    improvement = float(baseline["gap"]) - float(best["gap"])
    return {
        "best_action": best["action"],
        "best_action_name": best["action_name"],
        "best_gap": best["gap"],
        "baseline_gap": baseline["gap"],
        "improvement": improvement,
        "beats_noop": bool(improvement > 0.0),
        "candidates": results,
    }
