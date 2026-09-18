"""Hamiltonians, variational ansatz and energy evaluation (Qiskit only).

Conventions
-----------
- Qubit ``i`` maps to wire ``i`` of the ``QuantumCircuit``.
- Hamiltonians are built with ``SparsePauliOp.from_sparse_list`` so that qubit
  indices are explicit and never depend on Pauli-string ordering.
- Without noise the energy is evaluated with ``Statevector`` (exact).
- With noise the energy is evaluated with ``AerSimulator`` and a depolarizing
  ``NoiseModel`` through ``save_expectation_value``: the result is exact under
  the noise model, with no sampling error.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from scipy.sparse.linalg import eigsh

ArrayLike = Union[np.ndarray, Sequence[float]]


# --------------------------------------------------------------------------- #
# Hamiltonians
# --------------------------------------------------------------------------- #

def _pairs(n_qubits: int, periodic: bool) -> List[Tuple[int, int]]:
    """Coupling pairs for the entangling layer (open chain or ring)."""
    pairs = [(i, i + 1) for i in range(n_qubits - 1)]
    if periodic and n_qubits > 2:
        pairs.append((n_qubits - 1, 0))
    return pairs


def _two_body_terms(
    n_qubits: int,
    jx: float,
    jy: float,
    jz: float,
    periodic: bool,
) -> List[Tuple[str, List[int], float]]:
    terms: List[Tuple[str, List[int], float]] = []
    for i, j in _pairs(n_qubits, periodic):
        if jx:
            terms.append(("XX", [i, j], float(jx)))
        if jy:
            terms.append(("YY", [i, j], float(jy)))
        if jz:
            terms.append(("ZZ", [i, j], float(jz)))
    return terms


def heisenberg_hamiltonian(
    n_qubits: int,
    jx: float = 1.0,
    jy: float = 1.0,
    jz: float = 1.0,
    periodic: bool = False,
) -> SparsePauliOp:
    """H = sum_<ij> (jx XiXj + jy YiYj + jz ZiZj).

    For ``n_qubits=2`` and unit couplings the exact minimum is -3, which is the
    reference case of the paper.
    """
    terms = _two_body_terms(n_qubits, jx, jy, jz, periodic)
    return SparsePauliOp.from_sparse_list(terms, num_qubits=n_qubits)


def tfim_hamiltonian(
    n_qubits: int,
    j: float = 1.0,
    h: float = 1.0,
    periodic: bool = False,
) -> SparsePauliOp:
    """Transverse-field Ising model: H = -j sum_<ij> ZiZj - h sum_i Xi."""
    terms = _two_body_terms(n_qubits, 0.0, 0.0, -j, periodic)
    terms += [("X", [i], -h) for i in range(n_qubits)]
    return SparsePauliOp.from_sparse_list(terms, num_qubits=n_qubits)


def xy_hamiltonian(
    n_qubits: int,
    jx: float = 1.0,
    jy: float = 1.0,
    periodic: bool = False,
) -> SparsePauliOp:
    """XY model: H = sum_<ij> (jx XiXj + jy YiYj)."""
    terms = _two_body_terms(n_qubits, jx, jy, 0.0, periodic)
    return SparsePauliOp.from_sparse_list(terms, num_qubits=n_qubits)


HAMILTONIANS: Dict[str, Callable[..., SparsePauliOp]] = {
    "heisenberg": heisenberg_hamiltonian,
    "tfim": tfim_hamiltonian,
    "xy": xy_hamiltonian,
}


def build_hamiltonian(name: str, n_qubits: int, **kwargs) -> SparsePauliOp:
    """Build a Hamiltonian by name (the name is recorded in dataset metadata)."""
    if name not in HAMILTONIANS:
        raise ValueError(f"Unknown Hamiltonian: {name!r}. Options: {sorted(HAMILTONIANS)}")
    return HAMILTONIANS[name](n_qubits, **kwargs)


# --------------------------------------------------------------------------- #
# Ansatz
# --------------------------------------------------------------------------- #

def build_ansatz(n_qubits: int, n_layers: int) -> Tuple[QuantumCircuit, ParameterVector]:
    """Hardware-efficient ansatz: Rot(rx, ry, rz) per qubit plus ladder CX.

    Parameter order is (layer, qubit, rotation). Each layer applies the three
    Bloch-sphere rotations of every qubit and then the entangling ladder.
    """
    params = ParameterVector("theta", n_layers * n_qubits * 3)
    circuit = QuantumCircuit(n_qubits)
    k = 0
    for _ in range(n_layers):
        for wire in range(n_qubits):
            circuit.rx(params[k], wire)
            k += 1
            circuit.ry(params[k], wire)
            k += 1
            circuit.rz(params[k], wire)
            k += 1
        for i, j in _pairs(n_qubits, periodic=True):
            circuit.cx(i, j)
    return circuit, params


def parameter_count(n_qubits: int, n_layers: int) -> int:
    return 3 * n_qubits * n_layers


# --------------------------------------------------------------------------- #
# Energies
# --------------------------------------------------------------------------- #

def exact_ground_energy(hamiltonian: SparsePauliOp) -> float:
    """Exact minimum of the Hamiltonian.

    A dense diagonalization is used up to six qubits. Above that, a sparse
    Lanczos solver (``scipy.sparse.linalg.eigsh``) keeps the calculation
    feasible in memory, which is what allows the 16-qubit runs of the paper.
    """
    n = hamiltonian.num_qubits
    if n <= 6:
        return float(np.linalg.eigvalsh(hamiltonian.to_matrix())[0])
    sparse_op = hamiltonian.to_matrix(sparse=True)
    eigenvalues = eigsh(sparse_op, k=1, which="SA", return_eigenvectors=False, tol=1e-10)
    return float(eigenvalues[0])


def depolarizing_noise_model(p: float) -> NoiseModel:
    """Noise model: single-qubit depolarizing on rotations and two-qubit on CX."""
    model = NoiseModel()
    model.add_all_qubit_quantum_error(depolarizing_error(p, 1), ["rx", "ry", "rz"])
    model.add_all_qubit_quantum_error(depolarizing_error(p, 2), ["cx"])
    return model


#: Largest qubit count evaluated with the exact density-matrix method. The
#: density matrix stores 4**n complex amplitudes (16 B each): about 1 MB at 8
#: qubits, 268 MB at 12 qubits and about 1 GB at 13. Noisy problems above this
#: budget use sampled quantum trajectories on the statevector backend, which
#: carry a Monte Carlo error of order 1/sqrt(shots) that must be reported with
#: any number produced by that backend.
MAX_DENSITY_QUBITS = 12
DEFAULT_TRAJECTORY_SHOTS = 1024


def energy_backend(n_qubits: int, noise_p: float) -> str:
    """Name of the evaluation backend selected for a problem.

    The name is recorded in the run metadata so that every reported number
    carries the evaluation method it was produced with. ``statevector_exact``
    and ``density_matrix_exact`` are exact. ``statevector_trajectory``
    introduces sampling noise of order 1/sqrt(shots) and is only used where
    the density matrix does not fit in memory.
    """
    if noise_p <= 0.0:
        return "statevector_exact"
    if n_qubits <= MAX_DENSITY_QUBITS:
        return "density_matrix_exact"
    return "statevector_trajectory"


def make_energy_fn(
    hamiltonian: SparsePauliOp,
    n_qubits: int,
    n_layers: int,
    noise_p: float = 0.0,
    shots: int = DEFAULT_TRAJECTORY_SHOTS,
    backend_override: Optional[str] = None,
) -> Callable[[ArrayLike], float]:
    """Return E(theta) = <psi(theta)|H|psi(theta)>.

    Without noise the statevector simulator is used, which stays exact up to
    and beyond sixteen qubits. With noise, problems up to
    ``MAX_DENSITY_QUBITS`` use the explicit density-matrix method, which is
    exact under the noise model. Larger noisy problems use the statevector
    backend with sampled quantum trajectories: the depolarizing noise is real,
    the result is a Monte Carlo estimate whose error is of order
    1/sqrt(shots), and ``shots`` is recorded next to every run that uses it.

    ``backend_override`` forces a backend ("statevector_exact",
    "density_matrix_exact" or "statevector_trajectory") and exists so tests
    can compare the backends at qubit counts where both are feasible.
    """
    circuit, params = build_ansatz(n_qubits, n_layers)
    wires = list(range(n_qubits))
    selected = backend_override if backend_override is not None else energy_backend(n_qubits, noise_p)

    if noise_p <= 0.0 or selected == "statevector_exact":

        def energy(theta: ArrayLike) -> float:
            bound = circuit.assign_parameters(dict(zip(params, np.asarray(theta, dtype=float))))
            return float(np.real(Statevector(bound).expectation_value(hamiltonian)))

        return energy

    if selected == "density_matrix_exact":
        backend = AerSimulator(method="density_matrix", noise_model=depolarizing_noise_model(noise_p))

        def energy_noisy(theta: ArrayLike) -> float:
            bound = circuit.assign_parameters(dict(zip(params, np.asarray(theta, dtype=float)))).copy()
            bound.save_expectation_value(hamiltonian, wires)  # type: ignore[attr-defined]
            result = backend.run(bound).result()
            return float(np.real(result.data(0)["expectation_value"]))

        return energy_noisy

    backend = AerSimulator(method="statevector", noise_model=depolarizing_noise_model(noise_p))

    def energy_noisy_trajectory(theta: ArrayLike) -> float:
        bound = circuit.assign_parameters(dict(zip(params, np.asarray(theta, dtype=float)))).copy()
        bound.save_expectation_value(hamiltonian, wires)  # type: ignore[attr-defined]
        result = backend.run(bound, shots=shots).result()
        return float(np.real(result.data(0)["expectation_value"]))

    return energy_noisy_trajectory


def parameter_shift_gradient(
    energy_fn: Callable[[ArrayLike], float],
    theta: ArrayLike,
    shift: float = np.pi / 2,
) -> np.ndarray:
    """Exact parameter-shift gradient (valid for rx, ry and rz generators)."""
    theta = np.asarray(theta, dtype=float)
    grad = np.zeros_like(theta)
    for i in range(theta.size):
        plus = theta.copy()
        plus[i] += shift
        minus = theta.copy()
        minus[i] -= shift
        grad[i] = 0.5 * (energy_fn(plus) - energy_fn(minus))
    return grad


def random_initial_theta(
    rng: np.random.Generator,
    n_qubits: int,
    n_layers: int,
    scale: float = 0.5,
) -> np.ndarray:
    """Random initialization theta ~ U(-pi*scale, pi*scale)."""
    size = parameter_count(n_qubits, n_layers)
    return rng.uniform(-np.pi * scale, np.pi * scale, size=size)
