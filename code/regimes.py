"""Diagnostico de regimen con ground-truth de simulador.

Clases:
- ``CONVERGENCIA_OK``: |E - E_min| <= tol_ok.
- ``BARREN_PLATEAU``: la varianza de la derivada direccional esta por debajo del
  umbral (gradientes exponencialmente planos en todas las direcciones).
- ``MINIMO_LOCAL``: sin mejora en la ventana y gradiente pequeno, pero lejos del
  minimo exacto.
- ``MESETA_ENERGIA``: sin mejora con gradiente no despreciable (meseta por ruido
  o superficie rugosa).
- ``CONCEPT_DRIFT``: reservada para el Caso 2 (series temporales).

Las etiquetas se calculan en simulador, donde se conoce el minimo exacto: son
ground-truth, no la opinion del LLM. El bucle lento debe aprender (o ser evaluado)
contra estas etiquetas.
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
    """Umbrales del diagnostico (ablatables en el paper)."""

    tol_ok: float = 0.05
    eps_bp: float = 1e-3
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
    """Varianza de la derivada direccional sobre direcciones aleatorias unitarias.

    Media experimental de Var[grad E]: es el estadistico que distingue una
    meseta esteril (plana en todas las direcciones) de un minimo local (plano en
    unas direcciones y con curvatura/pendiente en otras).
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
) -> Dict[str, object]:
    """Etiqueta de regimen + motivos, a partir de telemetria y ground-truth."""
    cfg = RegimeConfig() if cfg is None else cfg
    gap = float(energy - e_min)
    details: Dict[str, object] = {
        "gap": gap,
        "window_improvement": float(window_improvement),
        "grad_norm_last": float(grad_norm_last),
        "grad_var": float(grad_var),
    }

    if abs(gap) <= cfg.tol_ok:
        label, reason = "CONVERGENCIA_OK", f"|E - E_min| = {abs(gap):.4f} <= tol_ok = {cfg.tol_ok}"
    elif grad_var <= cfg.eps_bp:
        label, reason = "BARREN_PLATEAU", f"Var[direccional] = {grad_var:.2e} <= eps_bp = {cfg.eps_bp:.1e}"
    elif window_improvement <= cfg.min_improvement and abs(grad_norm_last) <= cfg.eps_grad_local:
        label, reason = "MINIMO_LOCAL", (
            f"mejora de ventana {window_improvement:.2e} <= {cfg.min_improvement:.1e} "
            f"con |grad| = {grad_norm_last:.2e} y gap = {gap:.4f}"
        )
    else:
        label, reason = "MESETA_ENERGIA", (
            f"mejora de ventana {window_improvement:.2e} sin gradiente despreciable "
            f"(|grad| = {grad_norm_last:.2e})"
        )

    return {"label": label, "reason": reason, "details": details}
