"""Candidate initialization strategies for the warm-start block (Engine B).

The slow-loop model picks one strategy as a Choice question given the problem
description; the run starts from the angles that strategy generates. The
strategy set spans the natural spectrum: scale (small to full range), structure
(block-periodic, low-discrepancy) and spread (uniform versus near-zero).
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

INIT_STRATEGIES: Sequence[str] = (
    "small_uniform",
    "uniform_pm_pi2",
    "uniform_pm_pi",
    "block_periodic",
    "low_discrepancy",
    "near_zero",
)

INIT_DESCRIPTIONS: Dict[str, str] = {
    "small_uniform": "angles uniform in minus pi/8 to pi/8, close to the identity circuit",
    "uniform_pm_pi2": "angles uniform in minus pi/2 to pi/2, the standard range",
    "uniform_pm_pi": "angles uniform in minus pi to pi, full single-qubit coverage",
    "block_periodic": "one unit cell of angles replicated across every layer, a periodic structure",
    "low_discrepancy": "deterministic low-discrepancy angles on a regular grid over minus pi/2 to pi/2",
    "near_zero": "Gaussian angles with standard deviation 0.1 radians around zero",
}


def theta0_for(
    strategy: str,
    rng: np.random.Generator,
    n_qubits: int,
    n_layers: int,
    scale: float = 0.5,
) -> np.ndarray:
    """Initial angles for one strategy. ``scale`` is unused by design."""
    size = 3 * int(n_qubits) * int(n_layers)
    if strategy == "small_uniform":
        return rng.uniform(-np.pi / 8.0, np.pi / 8.0, size)
    if strategy == "uniform_pm_pi2":
        return rng.uniform(-np.pi / 2.0, np.pi / 2.0, size)
    if strategy == "uniform_pm_pi":
        return rng.uniform(-np.pi, np.pi, size)
    if strategy == "block_periodic":
        cell = rng.uniform(-np.pi / 2.0, np.pi / 2.0, 3 * int(n_qubits))
        return np.tile(cell, int(n_layers))
    if strategy == "low_discrepancy":
        base = np.linspace(-np.pi / 2.0, np.pi / 2.0, size + 2)[1:-1]
        return base + rng.uniform(-0.05, 0.05, size)
    if strategy == "near_zero":
        return rng.normal(0.0, 0.1, size)
    raise ValueError(f"unknown initialization strategy: {strategy!r}")
