"""Fast end-to-end tests for the LLM-MetaOpt pipeline.

They cover the physics (Hamiltonian eigenvalues, parameter-shift against finite
differences), the fast loop (SPSA), telemetry, regime diagnosis, counterfactual
labeling and the (network-free) LLM client layer.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from code import vqe
from code.labeler import NO_OP, Intervention, label_state
from code.llm_client import LLMConfig, build_request_payload, parse_decision
from code.optimizer import SPSAConfig, apply_intervention, run_spsa, schedules, spsa_step
from code.regimes import RegimeConfig, diagnose, directional_gradient_variance
from code.telemetry import build_window

HEISENBERG_2Q = vqe.heisenberg_hamiltonian(2)


# --------------------------------------------------------------------------- #
# Physics
# --------------------------------------------------------------------------- #

def test_heisenberg_2q_ground_energy_is_minus_three():
    assert vqe.exact_ground_energy(HEISENBERG_2Q) == pytest.approx(-3.0, abs=1e-9)


def test_hamiltonian_qubit_indexing():
    """Qubit index i must map to wire i (explicit convention, no ordering magic)."""
    from qiskit.quantum_info import SparsePauliOp

    # A single-site term on qubit 0 is "IZ" in Qiskit's label convention.
    z0 = SparsePauliOp.from_sparse_list([("Z", [0], 1.0)], num_qubits=2)
    assert np.allclose(z0.to_matrix(), SparsePauliOp.from_list([("IZ", 1.0)]).to_matrix())

    # Pair (0, 2) on three qubits is "ZIZ" (the rightmost character is qubit 0).
    zz02 = SparsePauliOp.from_sparse_list([("ZZ", [0, 2], 1.0)], num_qubits=3)
    assert np.allclose(zz02.to_matrix(), SparsePauliOp.from_list([("ZIZ", 1.0)]).to_matrix())

    # The TFIM with j=0 and h=0 is the zero operator.
    ham = vqe.build_hamiltonian("tfim", 2, j=0.0, h=0.0)
    assert np.allclose(ham.to_matrix(), np.zeros((4, 4)))

    # ZZ on the pair (0, 1): eigenvalues +-1 with multiplicity 2.
    zz = vqe.heisenberg_hamiltonian(2, jx=0.0, jy=0.0, jz=1.0)
    assert np.allclose(zz.to_matrix(), SparsePauliOp.from_list([("ZZ", 1.0)]).to_matrix())
    assert np.allclose(np.sort(np.real(np.diag(zz.to_matrix()))), [-1.0, -1.0, 1.0, 1.0])


def test_ansatz_parameter_count():
    circuit, params = vqe.build_ansatz(3, 2)
    assert len(params) == vqe.parameter_count(3, 2) == 18
    assert circuit.num_qubits == 3


def test_energy_fn_is_deterministic_and_bounded():
    energy = vqe.make_energy_fn(HEISENBERG_2Q, 2, 2)
    theta = np.zeros(vqe.parameter_count(2, 2))
    assert energy(theta) == pytest.approx(energy(theta))
    assert -3.0 - 1e-9 <= energy(theta) <= 3.0 + 1e-9


def test_noisy_energy_differs_from_noiseless():
    theta = np.full(vqe.parameter_count(2, 2), 0.3)
    clean = vqe.make_energy_fn(HEISENBERG_2Q, 2, 2, noise_p=0.0)(theta)
    noisy = vqe.make_energy_fn(HEISENBERG_2Q, 2, 2, noise_p=0.02)(theta)
    assert noisy != pytest.approx(clean, abs=1e-9)


def test_parameter_shift_matches_finite_difference():
    energy = vqe.make_energy_fn(vqe.heisenberg_hamiltonian(3), 3, 1)
    theta = np.linspace(-0.4, 0.4, vqe.parameter_count(3, 1))
    grad = vqe.parameter_shift_gradient(energy, theta)
    eps = 1e-6
    finite_difference = np.array(
        [
            (
                energy(np.where(np.arange(theta.size) == i, theta[i] + eps, theta))
                - energy(np.where(np.arange(theta.size) == i, theta[i] - eps, theta))
            )
            / (2 * eps)
            for i in range(theta.size)
        ]
    )
    assert np.allclose(grad, finite_difference, atol=1e-6)


# --------------------------------------------------------------------------- #
# Fast loop
# --------------------------------------------------------------------------- #

def test_schedules_decay_monotonically():
    cfg = SPSAConfig()
    a_values = [schedules(cfg, k)[0] for k in range(5)]
    c_values = [schedules(cfg, k)[1] for k in range(5)]
    assert all(later < earlier for earlier, later in zip(a_values, a_values[1:]))
    assert all(later < earlier for earlier, later in zip(c_values, c_values[1:]))


def test_spsa_step_returns_valid_record():
    energy = vqe.make_energy_fn(HEISENBERG_2Q, 2, 2)
    theta = np.full(vqe.parameter_count(2, 2), 0.2)
    theta_next, record = spsa_step(energy, theta, SPSAConfig(), k=0, rng=np.random.default_rng(0))
    assert theta_next.shape == theta.shape
    assert set(record) == {"step", "energy", "grad_norm", "a_k", "c_k", "theta"}


def test_spsa_reduces_energy():
    energy = vqe.make_energy_fn(HEISENBERG_2Q, 2, 2)
    cfg = SPSAConfig(steps=60, a=0.3, c=0.1, seed=0)
    rng = np.random.default_rng(0)
    theta0 = vqe.random_initial_theta(rng, 2, 2, scale=0.5)
    out = run_spsa(energy, theta0, cfg)
    assert out["final_energy"] < energy(theta0)
    assert len(out["history"]) == cfg.steps
    assert out["theta"].shape == theta0.shape


def test_spsa_is_reproducible_with_same_seed():
    energy = vqe.make_energy_fn(HEISENBERG_2Q, 2, 2)
    cfg = SPSAConfig(steps=25, seed=7)
    theta0 = np.full(vqe.parameter_count(2, 2), 0.1)
    assert run_spsa(energy, theta0, cfg)["final_energy"] == run_spsa(energy, theta0, cfg)["final_energy"]


def test_apply_intervention_variants():
    rng = np.random.default_rng(1)
    theta = np.zeros(6)
    first = apply_intervention(theta, noise_sigma=0.1, rng=rng)
    second = apply_intervention(theta, noise_sigma=0.1, rng=np.random.default_rng(1))
    assert np.allclose(first, second)
    restarted = apply_intervention(theta, restart=True, rng=rng)
    assert not np.allclose(restarted, theta)


# --------------------------------------------------------------------------- #
# Telemetry and diagnosis
# --------------------------------------------------------------------------- #

def test_build_window_has_expected_schema():
    energy = vqe.make_energy_fn(HEISENBERG_2Q, 2, 2)
    cfg = SPSAConfig(steps=20, seed=3)
    theta0 = np.full(vqe.parameter_count(2, 2), 0.2)
    out = run_spsa(energy, theta0, cfg)
    window = build_window(out["history"], end_index=9, n_window=10, eta_scale=1.0, theta=out["theta"])
    assert window["n_window"] == 10
    assert len(window["energy_series"]) == 10
    for key in ("energy", "grad_norm", "eta", "theta"):
        assert key in window
    json.dumps(window)  # must be serializable as-is


def test_diagnose_assigns_expected_labels():
    cfg = RegimeConfig()
    ok = diagnose(energy=-2.99, e_min=-3.0, window_improvement=0.0, grad_norm_last=0.01, grad_var=1e-2, cfg=cfg)
    assert ok["label"] == "CONVERGENCIA_OK"
    plateau = diagnose(energy=-1.0, e_min=-3.0, window_improvement=0.0, grad_norm_last=1e-6, grad_var=1e-6, cfg=cfg)
    assert plateau["label"] == "BARREN_PLATEAU"
    local = diagnose(energy=-1.5, e_min=-3.0, window_improvement=1e-6, grad_norm_last=1e-4, grad_var=1e-2, cfg=cfg)
    assert local["label"] == "MINIMO_LOCAL"
    rough = diagnose(energy=-1.5, e_min=-3.0, window_improvement=1e-6, grad_norm_last=0.5, grad_var=1e-2, cfg=cfg)
    assert rough["label"] == "MESETA_ENERGIA"


def test_directional_gradient_variance_is_non_negative():
    energy = vqe.make_energy_fn(HEISENBERG_2Q, 2, 2)
    rng = np.random.default_rng(0)
    theta = vqe.random_initial_theta(rng, 2, 2)
    assert directional_gradient_variance(energy, theta, rng, n_directions=4) >= 0.0


# --------------------------------------------------------------------------- #
# Counterfactual labeling
# --------------------------------------------------------------------------- #

def test_label_state_returns_valid_action():
    energy = vqe.make_energy_fn(HEISENBERG_2Q, 2, 2)
    cfg = SPSAConfig(steps=20, seed=0)
    rng = np.random.default_rng(0)
    theta = vqe.random_initial_theta(rng, 2, 2)
    label = label_state(
        energy,
        theta,
        cfg,
        e_min=-3.0,
        rng=np.random.default_rng(12345),
        lookahead=10,
        candidates=[NO_OP, Intervention(eta_scale=2.0), Intervention(noise_sigma=0.1)],
    )
    assert set(label["best_action"]) == {"eta_scale", "noise_sigma", "restart", "reheat"}
    assert label["best_gap"] <= label["baseline_gap"]
    assert len(label["candidates"]) == 3


# --------------------------------------------------------------------------- #
# LLM client (no network)
# --------------------------------------------------------------------------- #

def test_llm_payload_and_decision_parsing():
    window = {"n_window": 10, "energy_series": [0.1, 0.2], "improvement": 0.0, "grad_norm": {"last": 1e-5}}
    cfg = LLMConfig(base_url="http://localhost:11434/v1", model="qwen3.5:4b", use_json_schema=True)
    payload = build_request_payload(window, cfg)
    assert payload["model"] == "qwen3.5:4b"
    assert payload["temperature"] == 0.0
    assert payload["response_format"]["type"] == "json_schema"
    assert json.dumps(payload)

    raw = (
        '```json\n{"diagnosis": "BARREN_PLATEAU", "justification": "gradients vanish", '
        '"expected_effect": 0.05, '
        '"action": {"eta_scale": 2.0, "noise_sigma": 0.15, "restart": false}}\n```'
    )
    decision = parse_decision(raw)
    assert decision["diagnosis"] == "BARREN_PLATEAU"
    assert decision["action"]["noise_sigma"] == pytest.approx(0.15)
    assert decision["expected_effect"] == pytest.approx(0.05)


def test_llm_parse_decision_rejects_a_missing_expected_effect():
    # The schema declares expected_effect required and parse enforces it, so a
    # response that omits it must fail loudly instead of being silently dropped
    # downstream (adversarial review, minor finding on llm_client).
    raw = (
        '{"diagnosis": "MINIMO_LOCAL", "justification": "flat", '
        '"action": {"eta_scale": 1.0, "noise_sigma": 0.0, "restart": false}}'
    )
    with pytest.raises(ValueError, match="expected_effect"):
        parse_decision(raw)


def test_llm_payload_supports_the_json_object_fallback():
    window = {"n_window": 10, "improvement": 0.0, "grad_norm": {"last": 1e-5}}
    cfg = LLMConfig(base_url="http://localhost:11434/v1", model="qwen3.5:4b")
    payload = build_request_payload(window, cfg, use_json_schema="json_object")
    assert payload["response_format"] == {"type": "json_object"}


def test_llm_messages_can_disable_thinking():
    from code.llm_client import build_messages

    window = {"n_window": 10, "improvement": 0.0}
    quiet = build_messages(window, LLMConfig(base_url="http://x/v1", model="m", no_think=True))
    loud = build_messages(window, LLMConfig(base_url="http://x/v1", model="m"))
    assert quiet[1]["content"].endswith("/no_think")
    assert not loud[1]["content"].endswith("/no_think")


def test_llm_parse_decision_handles_garbage():
    with pytest.raises(ValueError):
        parse_decision("not json at all")
