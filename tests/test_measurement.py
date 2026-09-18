"""Tests for the measurement layer fixes (review findings B1, M1, M6)."""

from __future__ import annotations

import numpy as np
import pytest

from code import vqe
from code.optimizer import SPSAConfig, counting_energy_fn, run_spsa


def test_recorded_energy_is_true_energy() -> None:
    """History energies must equal E(theta_k), not the biased midpoint proxy."""
    hamiltonian = vqe.heisenberg_hamiltonian(2)
    energy_fn = vqe.make_energy_fn(hamiltonian, 2, 1)
    cfg = SPSAConfig(steps=5, seed=0)
    theta0 = vqe.random_initial_theta(np.random.default_rng(0), 2, 1)
    out = run_spsa(energy_fn, theta0, cfg)
    for record in out["history"]:
        theta_k = np.asarray(record["theta"], dtype=float)
        assert record["energy"] == pytest.approx(energy_fn(theta_k), abs=1e-10)


def test_energy_counter_counts_all_evaluations() -> None:
    hamiltonian = vqe.heisenberg_hamiltonian(2)
    fn, counter = counting_energy_fn(vqe.make_energy_fn(hamiltonian, 2, 1))
    cfg = SPSAConfig(steps=7, seed=0)
    theta0 = vqe.random_initial_theta(np.random.default_rng(1), 2, 1)
    run_spsa(fn, theta0, cfg)
    # 2 evaluations for the gradient estimate plus 1 for the recorded
    # energy of the new iterate, per step, plus 1 for the final energy.
    assert counter["n"] == 3 * 7 + 1
