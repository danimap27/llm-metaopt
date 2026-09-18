"""Explanation fidelity and occlusion tests for the slow loop.

Fidelity has two halves. The first is diagnostic accuracy against the simulator
ground truth, computed by ``code.regimes``. The second is the claim the
explanations actually make: an intervention with a stated expected effect should
produce that effect. This module measures the second half and provides the
occlusion battery that falsifies the controller's story field by field.
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

FIDELITY_FIELDS: Sequence[str] = (
    "energy",
    "energy_series",
    "grad_norm",
    "eta",
    "theta",
    "improvement",
)


def drop_field(window: Dict[str, object], field: str) -> Dict[str, object]:
    """Return a copy of the telemetry window without one block."""
    return {key: value for key, value in window.items() if key != field}


def occlusions(window: Dict[str, object]) -> Dict[str, Dict[str, object]]:
    """One ablated window per telemetry field present in the window."""
    return {field: drop_field(window, field) for field in FIDELITY_FIELDS if field in window}


def explanation_report(
    predicted: Sequence[float],
    observed: Sequence[float],
    n_perm: int = 2000,
    seed: int = 0,
) -> Dict[str, object]:
    """Agreement between the expected effect stated by the model and the measured effect.

    ``signed_accuracy`` is the fraction of calls whose stated direction was
    right. ``pearson_r`` measures linear agreement and is ``None`` (undefined,
    not zero) when either side is constant. ``mae`` is the magnitude error.
    ``majority_sign_accuracy`` is the base rate a constant sign predictor
    achieves, and ``signed_accuracy_perm_p`` is the permutation p-value of the
    signed accuracy under a random pairing, so the headline number is compared
    against both nulls.
    """
    predicted_array = np.asarray(list(predicted), dtype=float)
    observed_array = np.asarray(list(observed), dtype=float)
    if predicted_array.shape != observed_array.shape:
        raise ValueError("predicted and observed effects must have the same length")
    if predicted_array.size == 0:
        raise ValueError("no intervention events to score")
    signed = np.sign(predicted_array) == np.sign(observed_array)
    accuracy = float(signed.mean())
    if predicted_array.size > 1 and predicted_array.std() > 0 and observed_array.std() > 0:
        correlation: object = float(np.corrcoef(predicted_array, observed_array)[0, 1])
    else:
        correlation = None
    rng = np.random.default_rng(seed)
    perm_hits = 0
    for _ in range(int(n_perm)):
        shuffled = rng.permutation(observed_array)
        if float(np.mean(np.sign(predicted_array) == np.sign(shuffled))) >= accuracy:
            perm_hits += 1
    p_perm = (perm_hits + 1) / (n_perm + 1)
    majority = float(max(np.mean(observed_array > 0), np.mean(observed_array < 0)))
    return {
        "n": int(predicted_array.size),
        "signed_accuracy": accuracy,
        "pearson_r": correlation,
        "mae": float(np.mean(np.abs(predicted_array - observed_array))),
        "mean_predicted": float(predicted_array.mean()),
        "mean_observed": float(observed_array.mean()),
        "majority_sign_accuracy": majority,
        "signed_accuracy_perm_p": float(p_perm),
    }
