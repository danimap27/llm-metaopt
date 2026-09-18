"""Regime diagnosis with simulator ground truth.

Classes:
- ``CONVERGENCIA_OK``: |E - E_min| <= tol_ok.
- ``BARREN_PLATEAU``: the directional-gradient variance is below threshold, so
  the landscape is exponentially flat in every direction.
- ``MINIMO_LOCAL``: no improvement inside the window with a small gradient while
  the energy stays far from the exact minimum.
- ``MESETA_ENERGIA``: no improvement with a non-negligible gradient (noisy or
  rough landscape).
- ``CONCEPT_DRIFT``: reserved for the time-series case.

Labels are computed in simulation, where the exact minimum is known: they are
ground truth, not the LLM's opinion. The slow loop is trained and evaluated
against these labels.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Dict, Sequence, Tuple, Union

import numpy as np

ArrayLike = Union[np.ndarray, Sequence[float]]

REGIMES: Tuple[str, ...] = (
    "CONVERGENCIA_OK",
    "BARREN_PLATEAU",
    "MINIMO_LOCAL",
    "MESETA_ENERGIA",
    "CONCEPT_DRIFT",
)


@dataclass
class RegimeConfig:
    """Diagnosis thresholds (ablated in the paper).

    ``eps_bp`` is relative: the directional-gradient variance is normalized by
    the squared norm bound of the Hamiltonian, so the same threshold is
    comparable across qubit counts (review finding M5).
    """

    tol_ok: float = 0.05
    eps_bp: float = 1e-4
    min_improvement: float = 1e-3
    eps_grad_local: float = 5e-3
    n_directions: int = 8
    seed: int = 0

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


def directional_gradient_variance(
    energy_fn: Callable[[ArrayLike], float],
    theta: ArrayLike,
    rng: np.random.Generator,
    n_directions: int = 8,
) -> float:
    """Variance of the directional derivative over random unit directions.

    This is the empirical proxy for Var[grad E]: it separates a barren plateau
    (flat along every direction) from a local minimum (flat along some
    directions but with curvature elsewhere).
    """
    theta = np.asarray(theta, dtype=float)
    derivatives = np.empty(n_directions, dtype=float)
    for i in range(n_directions):
        direction = rng.normal(size=theta.size)
        norm = np.linalg.norm(direction)
        if norm == 0.0:
            direction = np.ones_like(theta) / np.sqrt(theta.size)
        else:
            direction = direction / norm
        plus = theta + (np.pi / 2.0) * direction
        minus = theta - (np.pi / 2.0) * direction
        derivatives[i] = 0.5 * (energy_fn(plus) - energy_fn(minus))
    return float(np.var(derivatives))


def diagnose(
    energy: float,
    e_min: float,
    window_improvement: float,
    grad_norm_last: float,
    grad_var: float,
    cfg: RegimeConfig | None = None,
    h_norm: float = 1.0,
) -> Dict[str, object]:
    """Regime label plus the reasons, from telemetry and ground truth.

    ``h_norm`` is an upper bound on the spectral norm of the Hamiltonian
    (the sum of the absolute Pauli coefficients). The directional-gradient
    variance is normalized by ``h_norm**2`` so ``eps_bp`` is dimensionless and
    comparable across problem sizes.
    """
    cfg = RegimeConfig() if cfg is None else cfg
    gap = float(energy - e_min)
    scale = max(float(h_norm) ** 2, 1e-30)
    grad_var_rel = float(grad_var) / scale
    details: Dict[str, object] = {
        "gap": gap,
        "window_improvement": float(window_improvement),
        "grad_norm_last": float(grad_norm_last),
        "grad_var": float(grad_var),
        "grad_var_rel": grad_var_rel,
        "h_norm": float(h_norm),
    }

    if abs(gap) <= cfg.tol_ok:
        label, reason = "CONVERGENCIA_OK", f"|E - E_min| = {abs(gap):.4f} <= tol_ok = {cfg.tol_ok}"
    elif grad_var_rel <= cfg.eps_bp:
        label, reason = "BARREN_PLATEAU", (
            f"relative Var[directional] = {grad_var_rel:.2e} <= eps_bp = {cfg.eps_bp:.1e}"
        )
    elif window_improvement <= cfg.min_improvement and abs(grad_norm_last) <= cfg.eps_grad_local:
        label, reason = "MINIMO_LOCAL", (
            f"window improvement {window_improvement:.2e} <= {cfg.min_improvement:.1e} "
            f"with |grad| = {grad_norm_last:.2e} and gap = {gap:.4f}"
        )
    else:
        label, reason = "MESETA_ENERGIA", (
            f"window improvement {window_improvement:.2e} with non-negligible gradient "
            f"(|grad| = {grad_norm_last:.2e})"
        )

    return {"label": label, "reason": reason, "details": details}
