"""Scaling tests: Lanczos ground energies and the qubit-dependent backends."""

from __future__ import annotations

import numpy as np
import pytest

from code import vqe


@pytest.mark.parametrize("name", ["heisenberg", "tfim", "xy"])
@pytest.mark.parametrize("n", [2, 3, 4, 5, 6])
def test_lanczos_matches_dense(name: str, n: int) -> None:
    """The sparse solver must reproduce dense diagonalization up to 6 qubits."""
    hamiltonian = vqe.build_hamiltonian(name, n)
    dense = float(np.linalg.eigvalsh(hamiltonian.to_matrix())[0])
    assert vqe.exact_ground_energy(hamiltonian) == pytest.approx(dense, abs=1e-8)


def test_heisenberg_reference_values() -> None:
    """Anchors: the 2-qubit minimum is exactly -3 and the energy is monotone."""
    e2 = vqe.exact_ground_energy(vqe.heisenberg_hamiltonian(2))
    e4 = vqe.exact_ground_energy(vqe.heisenberg_hamiltonian(4))
    e8 = vqe.exact_ground_energy(vqe.heisenberg_hamiltonian(8))
    e16 = vqe.exact_ground_energy(vqe.heisenberg_hamiltonian(16))
    assert e2 == pytest.approx(-3.0, abs=1e-9)
    assert e4 < e2 and e8 < e4 and e16 < e8
    assert e16 == pytest.approx(-27.6469, abs=1e-3)


def test_backend_selection() -> None:
    assert vqe.energy_backend(2, 0.0) == "statevector_exact"
    assert vqe.energy_backend(16, 0.0) == "statevector_exact"
    assert vqe.energy_backend(8, 0.05) == "density_matrix_exact"
    assert vqe.energy_backend(9, 0.05) == "statevector_kraus_exact"
    assert vqe.energy_backend(16, 0.02) == "statevector_kraus_exact"


@pytest.mark.parametrize("n", [4, 6])
def test_kraus_backend_is_exact(n: int) -> None:
    """The statevector Kraus path must match the density matrix bit for bit."""
    hamiltonian = vqe.heisenberg_hamiltonian(n)
    theta = vqe.random_initial_theta(np.random.default_rng(3), n, 2)
    exact = vqe.make_energy_fn(hamiltonian, n, 2, noise_p=0.05)(theta)
    kraus = vqe.make_energy_fn(hamiltonian, n, 2, noise_p=0.05)(theta)
    assert kraus == pytest.approx(exact, abs=1e-10)


def test_parameter_count_scales() -> None:
    for n, expected in ((2, 12), (4, 24), (8, 48), (16, 96)):
        assert vqe.parameter_count(n, 2) == expected
        theta = vqe.random_initial_theta(np.random.default_rng(0), n, 2)
        assert theta.size == expected
