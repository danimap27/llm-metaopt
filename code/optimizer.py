"""Fast loop: SPSA and application of slow-loop interventions.

The local optimizer is SPSA (Simultaneous Perturbation Stochastic
Approximation), matching the paper's notation:

    theta_{k+1} = theta_k - a_k * g_hat(theta_k)

with the standard decay schedules
    a_k = a / (k + 1 + A)^alpha        c_k = c / (k + 1)^gamma

and the two-evaluation gradient estimate
    g_hat_i = (E(theta + c_k * Delta) - E(theta - c_k * Delta)) / (2 c_k * Delta_i)

``spsa_step`` exposes a single iteration so the closed-loop runner can interleave
slow-loop interventions with fast-loop steps.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

ArrayLike = Union[np.ndarray, Sequence[float]]


@dataclass
class SPSAConfig:
    """Fast-loop hyperparameters."""

    steps: int = 200
    a: float = 0.25
    c: float = 0.1
    alpha: float = 0.602
    gamma: float = 0.101
    A: float = 10.0
    seed: int = 0

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


def schedules(cfg: SPSAConfig, k: int, eta_scale: float = 1.0) -> Tuple[float, float]:
    """Step-size and perturbation schedules at iteration ``k``."""
    a_k = eta_scale * cfg.a / (k + 1.0 + cfg.A) ** cfg.alpha
    c_k = cfg.c / (k + 1.0) ** cfg.gamma
    return a_k, c_k


def spsa_step(
    energy_fn: Callable[[ArrayLike], float],
    theta: ArrayLike,
    cfg: SPSAConfig,
    k: int,
    rng: np.random.Generator,
    eta_scale: float = 1.0,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """One SPSA iteration: gradient estimate plus parameter update."""
    theta = np.asarray(theta, dtype=float)
    a_k, c_k = schedules(cfg, k, eta_scale)
    delta = rng.choice([-1.0, 1.0], size=theta.size)
    e_plus = float(energy_fn(theta + c_k * delta))
    e_minus = float(energy_fn(theta - c_k * delta))
    g_hat = (e_plus - e_minus) / (2.0 * c_k) * delta
    theta_next = theta - a_k * g_hat
    record: Dict[str, object] = {
        "step": k,
        "energy": 0.5 * (e_plus + e_minus),
        "grad_norm": float(np.linalg.norm(g_hat)),
        "a_k": a_k,
        "c_k": c_k,
        "theta": theta_next.tolist(),
    }
    return theta_next, record


def run_spsa(
    energy_fn: Callable[[ArrayLike], float],
    theta0: ArrayLike,
    cfg: SPSAConfig,
    eta_scale: float = 1.0,
    steps: Optional[int] = None,
    seed: Optional[int] = None,
    on_step: Optional[Callable[[int, Dict[str, object], np.ndarray], None]] = None,
) -> Dict[str, object]:
    """Run SPSA from ``theta0``.

    Returns the final ``theta``, the per-step ``history`` (with angle snapshots)
    and the final energy. ``seed`` fixes the Rademacher perturbation stream, so
    two runs with the same seed compare intervention candidates under identical
    measurement noise, which is what the counterfactual labeling needs.
    """
    rng = np.random.default_rng(cfg.seed if seed is None else seed)
    theta = np.asarray(theta0, dtype=float).copy()
    n_steps = cfg.steps if steps is None else int(steps)
    history: List[Dict[str, object]] = []

    for k in range(n_steps):
        theta, record = spsa_step(energy_fn, theta, cfg, k, rng, eta_scale)
        history.append(record)
        if on_step is not None:
            on_step(k, record, theta)

    final_energy = float(energy_fn(theta))
    history.append(
        {
            "step": n_steps,
            "energy": final_energy,
            "grad_norm": float("nan"),
            "a_k": float("nan"),
            "c_k": float("nan"),
            "theta": theta.tolist(),
        }
    )
    return {"theta": theta, "history": history, "final_energy": final_energy}


def apply_intervention(
    theta: ArrayLike,
    eta_scale: float = 1.0,
    noise_sigma: float = 0.0,
    restart: bool = False,
    rng: Optional[np.random.Generator] = None,
    init_scale: float = 0.5,
) -> np.ndarray:
    """Apply one slow-loop action to the variational parameters.

    - ``eta_scale``: multiplicative factor on the SPSA step size.
    - ``noise_sigma``: Gaussian injection N(0, sigma^2) that breaks symmetries.
    - ``restart``: full re-initialization of the angles.
    """
    rng = np.random.default_rng() if rng is None else rng
    theta = np.asarray(theta, dtype=float)
    if restart:
        theta = rng.uniform(-np.pi * init_scale, np.pi * init_scale, size=theta.size)
    else:
        theta = theta.copy()
    if noise_sigma > 0.0:
        theta = theta + rng.normal(0.0, noise_sigma, size=theta.size)
    return theta
