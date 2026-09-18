"""Counterfactual labeling of the best intervention (bandit formulation).

For a frozen state (angles plus run context) every candidate intervention is
rolled out and the final energy is measured after ``lookahead`` SPSA steps.
The rollout continues the deployed run instead of restarting it: the SPSA
schedules resume at the global step ``k_offset`` and the Rademacher stream is
derived from the run's own random generator, so the counterfactual is measured
under the conditions the action would actually meet. Each candidate is rolled
out ``n_repeats`` times and the mean gap is used, which keeps a single lucky
draw from winning the argmax.

The label is the candidate with the smallest mean gap to the exact minimum,
and the reward is its improvement over the no-op action. This turns the
problem into supervised or reinforcement learning with a physical reward,
without relying on the LLM's opinion.
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

    eta_scale: Optional[float] = None  # None keeps the current multiplier
    noise_sigma: float = 0.0
    restart: bool = False

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "Intervention":
        """Rebuild an action from :meth:`to_dict` output (policy store, records)."""
        eta = payload.get("eta_scale")
        return cls(
            eta_scale=None if eta is None else float(eta),
            noise_sigma=float(payload.get("noise_sigma", 0.0)),
            restart=bool(payload.get("restart", False)),
        )

    @property
    def name(self) -> str:
        base = "noop" if self.eta_scale is None else f"eta{self.eta_scale:g}"
        parts = [base]
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
    rng: np.random.Generator,
    init_scale: float = 0.5,
    k_offset: int = 0,
) -> Dict[str, object]:
    """Roll out one counterfactual from the frozen state.

    ``rng`` is a child of the run's own generator, so the intervention noise
    and the Rademacher stream belong to the run's randomness budget. The SPSA
    schedules continue at ``k_offset`` instead of restarting.
    """
    i_rng, r_rng = rng.spawn(2)
    theta0 = apply_intervention(
        theta,
        eta_scale=1.0 if action.eta_scale is None else action.eta_scale,
        noise_sigma=action.noise_sigma,
        restart=action.restart,
        rng=i_rng,
        init_scale=init_scale,
    )
    out = run_spsa(
        energy_fn,
        theta0,
        cfg,
        eta_scale=1.0 if action.eta_scale is None else action.eta_scale,
        steps=lookahead,
        rng=r_rng,
        k_offset=k_offset,
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
    rng: np.random.Generator,
    lookahead: int,
    candidates: Optional[Sequence[Intervention]] = None,
    init_scale: float = 0.5,
    k_offset: int = 0,
    n_repeats: int = 3,
) -> Dict[str, object]:
    """Label the state with the best counterfactual intervention.

    Every candidate is rolled out ``n_repeats`` times and compared by mean
    gap, so the argmax is not decided by a single noisy draw. Returns
    ``best_action``, the per-repeat gaps of every candidate, the no-op mean
    gap and the improvement (baseline mean minus best mean). The improvement
    is reported together with the no-op standard deviation so the reader can
    judge whether it exceeds the Monte Carlo noise.
    """
    candidates = default_candidates() if candidates is None else list(candidates)
    children = rng.spawn(len(candidates))
    results: List[Dict[str, object]] = []
    for action, child in zip(candidates, children):
        gaps = np.asarray(
            [
                float(
                    evaluate_candidate(
                        energy_fn, theta, cfg, action, e_min, lookahead, child, init_scale, k_offset
                    )["gap"]
                )
                for _ in range(n_repeats)
            ],
            dtype=float,
        )
        results.append(
            {
                "action": action.to_dict(),
                "action_name": action.name,
                "gap": float(gaps.mean()),
                "gap_std": float(gaps.std()),
                "repeats": n_repeats,
            }
        )
    best = min(results, key=lambda r: float(r["gap"]))
    baseline = next((r for r in results if r["action_name"] == NO_OP.name), results[0])
    improvement = float(baseline["gap"]) - float(best["gap"])
    return {
        "best_action": best["action"],
        "best_action_name": best["action_name"],
        "best_gap": best["gap"],
        "best_gap_std": best["gap_std"],
        "baseline_gap": baseline["gap"],
        "baseline_gap_std": baseline["gap_std"],
        "improvement": improvement,
        "beats_noop": bool(improvement > 0.0),
        "candidates": results,
    }
