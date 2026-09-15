"""Drifting-objective environment for the non-stationary supervision study.

A run is a sequence of segments. Each segment owns a Hamiltonian, a qubit count,
a noise level and a number of fast-loop steps. The objective changes with no
warning at a segment boundary, which is the event the slow loop has to notice.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence

import numpy as np

from . import vqe


@dataclass
class DriftSpec:
    """Description of a drifting run."""

    segments: List[Dict[str, object]]
    n_layers: int = 2
    init_scale: float = 0.5

    @property
    def total_steps(self) -> int:
        return int(sum(int(segment["steps"]) for segment in self.segments))

    def boundaries(self) -> List[int]:
        """Step index at which each drift happens (the start of the run excluded)."""
        edges: List[int] = []
        running = 0
        for segment in self.segments:
            edges.append(running)
            running += int(segment["steps"])
        return edges[1:]


class DriftingObjective:
    """Piecewise-stationary objective with an exact minimum per segment."""

    def __init__(
        self,
        spec: DriftSpec,
        energy_fn_factory: Callable[..., Callable[[Sequence[float]], float]] = vqe.make_energy_fn,
    ) -> None:
        self.spec = spec
        self._ends: List[int] = []
        running = 0
        for segment in spec.segments:
            running += int(segment["steps"])
            self._ends.append(running)
        self._energy_fns: List[Callable[[Sequence[float]], float]] = []
        self._e_mins: List[float] = []
        for segment in spec.segments:
            hamiltonian = vqe.build_hamiltonian(str(segment["hamiltonian"]), int(segment["n_qubits"]))
            self._energy_fns.append(
                energy_fn_factory(
                    hamiltonian,
                    int(segment["n_qubits"]),
                    spec.n_layers,
                    float(segment.get("noise_p", 0.0)),
                )
            )
            self._e_mins.append(vqe.exact_ground_energy(hamiltonian))

    def segment_of(self, step: int) -> int:
        """Segment that owns a given fast-loop step."""
        for index, end in enumerate(self._ends):
            if step < end:
                return index
        return len(self._ends) - 1

    def energy(self, step: int, theta: Sequence[float]) -> float:
        return float(self._energy_fns[self.segment_of(step)](theta))

    def e_min(self, step: int) -> float:
        return float(self._e_mins[self.segment_of(step)])

    def gap(self, step: int, theta: Sequence[float]) -> float:
        return self.energy(step, theta) - self.e_min(step)


def build_drift_spec(
    cfg: Dict[str, object],
    rng: np.random.Generator,
    segments: int | None = None,
) -> DriftSpec:
    """Build a drift specification from the ``drift`` block of a config file.

    Segment Hamiltonians are drawn without immediate repetition so that every
    boundary is a real change of objective.
    """
    drift = cfg["drift"]  # type: ignore[index]
    n_segments = int(drift.get("segments", 3)) if segments is None else int(segments)
    steps_per_segment = int(drift.get("steps_per_segment", 40))
    hamiltonians = [str(name) for name in drift.get("hamiltonians", ["heisenberg"])]
    noise_levels = [float(value) for value in drift.get("noise_p", [0.0])]
    chain: List[Dict[str, object]] = []
    previous: str | None = None
    for _ in range(n_segments):
        choices = [name for name in hamiltonians if name != previous] or hamiltonians
        name = str(rng.choice(choices))
        chain.append(
            {
                "hamiltonian": name,
                "n_qubits": int(cfg["ansatz"].get("drift_n_qubits", 2)),  # type: ignore[index]
                "noise_p": float(rng.choice(noise_levels)),
                "steps": steps_per_segment,
            }
        )
        previous = name
    return DriftSpec(
        segments=chain,
        n_layers=int(cfg["ansatz"]["n_layers"]),  # type: ignore[index]
        init_scale=float(cfg["sweep"]["init_scale"]),  # type: ignore[index]
    )
