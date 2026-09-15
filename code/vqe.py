"""Hamiltonianos, ansatz variacional y evaluacion de energia (solo Qiskit).

Convenciones
------------
- El qubit i corresponde al wire i del QuantumCircuit.
- Los Hamiltonianos se construyen con ``SparsePauliOp.from_sparse_list`` para
  evitar la ambiguedad de orden de los strings de Pauli.
- Sin ruido la energia se evalua con ``Statevector`` (exacta).
- Con ruido se evalua con ``AerSimulator`` + ``NoiseModel`` (canal despolarizante
  por puerta) usando ``save_expectation_value``: el resultado es exacto bajo el
  modelo de ruido, sin error de muestreo.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error

ArrayLike = Sequence[float]


# --------------------------------------------------------------------------- #
# Hamiltonianos
# --------------------------------------------------------------------------- #

def _pairs(n_qubits: int, periodic: bool) -> List[Tuple[int, int]]:
    """Pares de qubits del entrelazado (cadena abierta o anillo)."""
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

    Para n_qubits=2 y J=1 el minimo exacto es -3 (Caso 1 del documento maestro).
    """
    terms = _two_body_terms(n_qubits, jx, jy, jz, periodic)
    return SparsePauliOp.from_sparse_list(terms, num_qubits=n_qubits)


def tfim_hamiltonian(
    n_qubits: int,
    j: float = 1.0,
    h: float = 1.0,
    periodic: bool = False,
) -> SparsePauliOp:
    """Transverse-field Ising: H = -j sum_<ij> ZiZj - h sum_i Xi."""
    terms = _two_body_terms(n_qubits, 0.0, 0.0, -j, periodic)
    terms += [("X", [i], -h) for i in range(n_qubits)]
    return SparsePauliOp.from_sparse_list(terms, num_qubits=n_qubits)


def xy_hamiltonian(
    n_qubits: int,
    jx: float = 1.0,
    jy: float = 1.0,
    periodic: bool = False,
) -> SparsePauliOp:
    """Modelo XY: H = sum_<ij> (jx XiXj + jy YiYj)."""
    terms = _two_body_terms(n_qubits, jx, jy, 0.0, periodic)
    return SparsePauliOp.from_sparse_list(terms, num_qubits=n_qubits)


HAMILTONIANS: Dict[str, Callable[..., SparsePauliOp]] = {
    "heisenberg": heisenberg_hamiltonian,
    "tfim": tfim_hamiltonian,
    "xy": xy_hamiltonian,
}


def build_hamiltonian(name: str, n_qubits: int, **kwargs) -> SparsePauliOp:
    """Constructor por nombre (el nombre aparece en los metadatos del dataset)."""
    if name not in HAMILTONIANS:
        raise ValueError(f"Hamiltoniano desconocido: {name!r}. Opciones: {sorted(HAMILTONIANS)}")
    return HAMILTONIANS[name](n_qubits, **kwargs)


# --------------------------------------------------------------------------- #
# Ansatz
# --------------------------------------------------------------------------- #

def build_ansatz(n_qubits: int, n_layers: int) -> Tuple[QuantumCircuit, ParameterVector]:
    """Ansatz hardware-efficient: Rot(rx, ry, rz) por qubit + CX en escalera.

    Orden de parametros: (capa, qubit, rotacion). Cada capa aplica las tres
    rotaciones de la esfera de Bloch seguidas del entrelazado.
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
# Energia
# --------------------------------------------------------------------------- #

def exact_ground_energy(hamiltonian: SparsePauliOp) -> float:
    """Minimo exacto por diagonalizacion (solo valido para pocos qubits)."""
    return float(np.linalg.eigvalsh(hamiltonian.to_matrix())[0])


def depolarizing_noise_model(p: float) -> NoiseModel:
    """Modelo de ruido: despolarizante de 1 qubit en rotaciones y de 2 en CX."""
    model = NoiseModel()
    model.add_all_qubit_quantum_error(depolarizing_error(p, 1), ["rx", "ry", "rz"])
    model.add_all_qubit_quantum_error(depolarizing_error(p, 2), ["cx"])
    return model


def make_energy_fn(
    hamiltonian: SparsePauliOp,
    n_qubits: int,
    n_layers: int,
    noise_p: float = 0.0,
) -> Callable[[ArrayLike], float]:
    """Devuelve E(theta) = <psi(theta)|H|psi(theta)>.

    Con ``noise_p == 0`` usa simulacion de vector de estado. Con ruido usa
    AerSimulator con el modelo despolarizante (matriz densidad).
    """
    circuit, params = build_ansatz(n_qubits, n_layers)
    wires = list(range(n_qubits))

    if noise_p <= 0.0:

        def energy(theta: ArrayLike) -> float:
            bound = circuit.assign_parameters(dict(zip(params, np.asarray(theta, dtype=float))))
            return float(np.real(Statevector(bound).expectation_value(hamiltonian)))

        return energy

    backend = AerSimulator(noise_model=depolarizing_noise_model(noise_p))

    def energy_noisy(theta: ArrayLike) -> float:
        bound = circuit.assign_parameters(dict(zip(params, np.asarray(theta, dtype=float)))).copy()
        bound.save_expectation_value(hamiltonian, wires)
        result = backend.run(bound).result()
        return float(np.real(result.data(0)["expectation_value"]))

    return energy_noisy


def parameter_shift_gradient(
    energy_fn: Callable[[ArrayLike], float],
    theta: ArrayLike,
    shift: float = np.pi / 2,
) -> np.ndarray:
    """Gradiente exacto por parameter-shift (valido para rx, ry, rz)."""
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
    """Inicializacion aleatoria theta ~ U(-pi*scale, pi*scale)."""
    size = parameter_count(n_qubits, n_layers)
    return rng.uniform(-np.pi * scale, np.pi * scale, size=size)
