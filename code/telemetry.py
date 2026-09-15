"""Telemetry window ``T_k`` and its JSON serialization.

The paper defines

    T_k = [E_{k-N_w..k}, ||grad E||_2, eta_k, Var(theta)]

This module turns the fast-loop history into a compact dictionary with
controlled numerical precision, because float precision inside JSON prompts is
one of the declared limitations of the method.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Union

import numpy as np

ArrayLike = Union[np.ndarray, Sequence[float]]

DEFAULT_SIGNIFICANT_DIGITS = 4


def compact_float(value: float, significant_digits: int = DEFAULT_SIGNIFICANT_DIGITS) -> float:
    """Round to significant digits (avoids long decimal tails inside prompts)."""
    value = float(value)
    if not np.isfinite(value):
        return float("nan")
    if value == 0.0:
        return 0.0
    return float(f"%.{significant_digits}g" % value)


def window_slice(history: Sequence[Dict[str, object]], end_index: int, n_window: int) -> List[Dict[str, object]]:
    """Slice ``[end_index - n_window + 1, end_index]`` out of the history."""
    start = max(0, end_index - n_window + 1)
    return list(history[start : end_index + 1])


def summarize_window(records: Sequence[Dict[str, object]]) -> Dict[str, float]:
    """Window statistics: energy, gradient and progress."""
    energies = np.asarray([float(r["energy"]) for r in records], dtype=float)
    grad_norms = np.asarray(
        [float(r["grad_norm"]) for r in records if np.isfinite(float(r["grad_norm"]))], dtype=float
    )
    steps = np.arange(energies.size, dtype=float)
    slope = float(np.polyfit(steps, energies, 1)[0]) if energies.size > 1 else 0.0
    return {
        "n": int(energies.size),
        "energy_first": compact_float(energies[0]),
        "energy_last": compact_float(energies[-1]),
        "energy_min": compact_float(np.min(energies)),
        "energy_mean": compact_float(np.mean(energies)),
        "energy_std": compact_float(np.std(energies)),
        "energy_slope_per_step": compact_float(slope),
        "improvement": compact_float(energies[0] - energies[-1]),
        "grad_norm_last": compact_float(grad_norms[-1]) if grad_norms.size else float("nan"),
        "grad_norm_mean": compact_float(np.mean(grad_norms)) if grad_norms.size else float("nan"),
        "grad_norm_max": compact_float(np.max(grad_norms)) if grad_norms.size else float("nan"),
    }


def build_window(
    history: Sequence[Dict[str, object]],
    end_index: int,
    n_window: int,
    eta_scale: float = 1.0,
    theta: Optional[ArrayLike] = None,
) -> Dict[str, object]:
    """Telemetry window ready for the prompt and for the dataset."""
    records = window_slice(history, end_index, n_window)
    summary = summarize_window(records)
    window: Dict[str, object] = {
        "n_window": int(n_window),
        "step_end": int(history[end_index]["step"]),  # type: ignore[arg-type]
        "energy_series": [compact_float(float(r["energy"])) for r in records],
        "energy": {
            "first": summary["energy_first"],
            "last": summary["energy_last"],
            "min": summary["energy_min"],
            "mean": summary["energy_mean"],
            "std": summary["energy_std"],
            "slope_per_step": summary["energy_slope_per_step"],
        },
        "improvement": summary["improvement"],
        "grad_norm": {
            "last": summary["grad_norm_last"],
            "mean": summary["grad_norm_mean"],
            "max": summary["grad_norm_max"],
        },
        "eta": {
            "a_k_last": compact_float(float(history[end_index]["a_k"])),  # type: ignore[arg-type]
            "c_k_last": compact_float(float(history[end_index]["c_k"])),  # type: ignore[arg-type]
            "eta_scale": float(eta_scale),
        },
    }
    if theta is not None:
        theta_array = np.asarray(theta, dtype=float)
        window["theta"] = {
            "var": compact_float(np.var(theta_array)),
            "mean_abs": compact_float(np.mean(np.abs(theta_array))),
            "max_abs": compact_float(np.max(np.abs(theta_array))),
            "dim": int(theta_array.size),
        }
    return window
