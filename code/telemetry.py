"""Construccion de la ventana de telemetria T_k y su serializacion JSON.

El documento maestro define

    T_k = [E_{k-N_w..k}, ||grad E||_2, eta_k, Var(theta)]

Esta capa convierte el historial del bucle rapido en un diccionario compacto y
con precision controlada, porque la precision de los floats dentro del prompt
JSON es una de las limitaciones declaradas del paper.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

DEFAULT_SIGNIFICANT_DIGITS = 4


def compact_float(value: float, significant_digits: int = DEFAULT_SIGNIFICANT_DIGITS) -> float:
    """Redondeo a cifras significativas (evita colas de decimales en el prompt)."""
    value = float(value)
    if not np.isfinite(value):
        return float("nan")
    if value == 0.0:
        return 0.0
    return float(f"%.{significant_digits}g" % value)


def window_slice(history: Sequence[Dict[str, float]], end_index: int, n_window: int) -> List[Dict[str, float]]:
    """Ventana [end_index - n_window + 1, end_index] del historial."""
    start = max(0, end_index - n_window + 1)
    return list(history[start : end_index + 1])


def summarize_window(records: Sequence[Dict[str, float]]) -> Dict[str, float]:
    """Estadisticos de la ventana: energia, gradiente y progreso."""
    energies = np.asarray([r["energy"] for r in records], dtype=float)
    grad_norms = np.asarray([r["grad_norm"] for r in records if np.isfinite(r["grad_norm"])], dtype=float)
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
    history: Sequence[Dict[str, float]],
    end_index: int,
    n_window: int,
    eta_scale: float = 1.0,
    theta: Sequence[float] | None = None,
) -> Dict[str, object]:
    """Ventana de telemetria lista para el prompt y para el dataset."""
    records = window_slice(history, end_index, n_window)
    summary = summarize_window(records)
    window: Dict[str, object] = {
        "n_window": int(n_window),
        "step_end": int(history[end_index]["step"]),
        "energy_series": [compact_float(r["energy"]) for r in records],
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
            "a_k_last": compact_float(history[end_index]["a_k"]),
            "c_k_last": compact_float(history[end_index]["c_k"]),
            "eta_scale": float(eta_scale),
        },
    }
    if theta is not None:
        theta = np.asarray(theta, dtype=float)
        window["theta"] = {
            "var": compact_float(np.var(theta)),
            "mean_abs": compact_float(np.mean(np.abs(theta))),
            "max_abs": compact_float(np.max(np.abs(theta))),
            "dim": int(theta.size),
        }
    return window
