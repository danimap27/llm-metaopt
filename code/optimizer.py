"""Bucle rapido: SPSA y aplicacion de intervenciones del bucle lento.

El optimizador local es SPSA (Simultaneous Perturbation Stochastic
Approximation), el mismo que usa el documento maestro:

    theta_{k+1} = theta_k - a_k * g_hat(theta_k)

con esquemas de decaimiento
    a_k = a / (k + 1 + A)^alpha        c_k = c / (k + 1)^gamma

y gradiente de dos evaluaciones
    g_hat_i = (E(theta + c_k * Delta) - E(theta - c_k * Delta)) / (2 c_k * Delta_i)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Dict, List, Optional, Sequence, Union

import numpy as np

ArrayLike = Union[np.ndarray, Sequence[float]]


@dataclass
class SPSAConfig:
    """Hiperparametros del bucle rapido."""

    steps: int = 200
    a: float = 0.25
    c: float = 0.1
    alpha: float = 0.602
    gamma: float = 0.101
    A: float = 10.0
    seed: int = 0

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


def _schedules(cfg: SPSAConfig, k: int, eta_scale: float) -> tuple[float, float]:
    a_k = eta_scale * cfg.a / (k + 1.0 + cfg.A) ** cfg.alpha
    c_k = cfg.c / (k + 1.0) ** cfg.gamma
    return a_k, c_k


def run_spsa(
    energy_fn: Callable[[ArrayLike], float],
    theta0: ArrayLike,
    cfg: SPSAConfig,
    eta_scale: float = 1.0,
    steps: Optional[int] = None,
    seed: Optional[int] = None,
    on_step: Optional[Callable[[int, Dict[str, float], np.ndarray], None]] = None,
) -> Dict[str, object]:
    """Ejecuta SPSA desde ``theta0``.

    Devuelve ``theta`` final, ``history`` (una entrada por paso, con la instantanea
    de los angulos) y la energia final. ``seed`` fija la secuencia de
    perturbaciones Rademacher: dos runs con la misma semilla comparan candidatos
    de intervencion bajo el mismo ruido de medida, que es lo que necesita el
    etiquetado contrafactual.
    """
    rng = np.random.default_rng(cfg.seed if seed is None else seed)
    theta = np.asarray(theta0, dtype=float).copy()
    n_steps = cfg.steps if steps is None else int(steps)
    history: List[Dict[str, object]] = []

    for k in range(n_steps):
        a_k, c_k = _schedules(cfg, k, eta_scale)
        delta = rng.choice([-1.0, 1.0], size=theta.size)
        e_plus = float(energy_fn(theta + c_k * delta))
        e_minus = float(energy_fn(theta - c_k * delta))
        g_hat = (e_plus - e_minus) / (2.0 * c_k) * delta
        theta = theta - a_k * g_hat
        energy = 0.5 * (e_plus + e_minus)
        record: Dict[str, object] = {
            "step": k,
            "energy": energy,
            "grad_norm": float(np.linalg.norm(g_hat)),
            "a_k": a_k,
            "c_k": c_k,
            "theta": theta.tolist(),
        }
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
    """Aplica una accion del bucle lento a los parametros.

    - ``eta_scale``: multiplicador de la tasa de aprendizaje (lo consume SPSA).
    - ``noise_sigma``: inyeccion gaussiana N(0, sigma^2) para romper simetrias.
    - ``restart``: reinicio completo de los angulos.
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
