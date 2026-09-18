"""Tests anchoring the exact ground energies to independent references."""

from __future__ import annotations

import numpy as np
import pytest

from code import vqe
from code.aggregate import summarize_by_cell


def test_tfim_two_qubits_matches_jordan_wigner_closed_form() -> None:
    """TFIM with H = -ZZ - X - X has E0 = -sqrt(5) on two qubits.

    Independent analytic reference: in the even parity sector the Hamiltonian
    reduces to [[-1, -2], [-2, 1]], whose eigenvalues are plus/minus sqrt(5).
    """
    hamiltonian = vqe.build_hamiltonian("tfim", 2)
    assert vqe.exact_ground_energy(hamiltonian) == pytest.approx(-np.sqrt(5), abs=1e-10)


def test_summarize_by_cell_keeps_problem_sizes_separate() -> None:
    """M2: per-cell summaries never pool raw gaps across qubit counts."""
    records = []
    for condition, gap in (("spsa", 0.2), ("spsa_llm", 0.1)):
        records.append(
            {
                "kind": "run",
                "run_id": f"h_n2_p0_s0_{condition}",
                "base_id": "h_n2_p0_s0",
                "condition": condition,
                "hamiltonian": "heisenberg",
                "n_qubits": 2,
                "noise_p": 0.0,
                "seed": 0,
                "final_gap": gap,
                "n_interventions": 3,
                "n_llm_calls": 3 if condition == "spsa_llm" else 0,
                "llm_latency_mean_s": None,
                "meta": {"e_min": -3.0},
            }
        )
        records.append(
            {
                "kind": "run",
                "run_id": f"h_n16_p0_s0_{condition}",
                "base_id": "h_n16_p0_s0",
                "condition": condition,
                "hamiltonian": "heisenberg",
                "n_qubits": 16,
                "noise_p": 0.0,
                "seed": 0,
                "final_gap": 20.0 * gap,
                "n_interventions": 3,
                "n_llm_calls": 3 if condition == "spsa_llm" else 0,
                "llm_latency_mean_s": None,
                "meta": {"e_min": -27.6},
            }
        )
    by_cell = summarize_by_cell(records)
    assert len(by_cell["cells"]) == 2
    gaps = {
        cell["cell"]["n_qubits"]: {row["condition"]: row["gap_mean"] for row in cell["conditions"]}
        for cell in by_cell["cells"]
    }
    # 2-qubit means stay at the 2-qubit scale, 16-qubit at theirs.
    assert gaps[2]["spsa"] == pytest.approx(0.2)
    assert gaps[16]["spsa"] == pytest.approx(4.0)
