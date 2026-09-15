"""Tests for the drifting-objective environment."""

from __future__ import annotations

import numpy as np
import pytest

from code import vqe
from code.drift import DriftSpec, DriftingObjective


def _two_segment_spec() -> DriftSpec:
    return DriftSpec(
        segments=[
            {"hamiltonian": "heisenberg", "n_qubits": 2, "noise_p": 0.0, "steps": 5},
            {"hamiltonian": "tfim", "n_qubits": 2, "noise_p": 0.0, "steps": 5},
        ],
        n_layers=1,
    )


def test_objective_switches_at_segment_boundaries():
    objective = DriftingObjective(_two_segment_spec())
    assert objective.segment_of(0) == 0
    assert objective.segment_of(4) == 0
    assert objective.segment_of(5) == 1
    assert objective.e_min(4) == pytest.approx(vqe.exact_ground_energy(vqe.heisenberg_hamiltonian(2)))
    assert objective.e_min(5) == pytest.approx(vqe.exact_ground_energy(vqe.tfim_hamiltonian(2)))


def test_energy_is_finite_and_segment_dependent():
    objective = DriftingObjective(_two_segment_spec())
    theta = np.zeros(vqe.parameter_count(2, 1))
    first = objective.energy(0, theta)
    second = objective.energy(5, theta)
    assert np.isfinite(first) and np.isfinite(second)
    assert first != pytest.approx(second)


def test_spec_reports_total_steps_and_boundaries():
    spec = _two_segment_spec()
    assert spec.total_steps == 10
    assert spec.boundaries() == [5]
