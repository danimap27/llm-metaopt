"""Tests for the v2 features: observable regimes, reheat, gates, Engine B init."""

from __future__ import annotations

import numpy as np
import pytest

from code import vqe
from code.experiment import run_closed_loop
from code.init_strategies import INIT_STRATEGIES, theta0_for
from code.llm_client import LLMConfig
from code.optimizer import SPSAConfig, schedules
from code.regimes import RegimeConfig, diagnose_observable

_UNUSED_LLM = LLMConfig(api_style="systemone", base_url="http://unused", model="fake")


# --------------------------------------------------------------------------- #
# Observable taxonomy
# --------------------------------------------------------------------------- #

def test_observable_taxonomy_classifies_the_probe_windows() -> None:
    cfg = RegimeConfig()
    steep = diagnose_observable(2.9, 0.9, 1.8, cfg)
    flat_no_grad = diagnose_observable(0.0001, 0.01, 0.0001, cfg)
    flat_with_grad = diagnose_observable(0.001, 0.01, 0.4, cfg)
    oscillating = diagnose_observable(0.001, 0.8, 0.2, cfg)
    assert steep["label"] == "DESCENDING"
    assert flat_no_grad["label"] == "STALLED_NO_GRADIENT"
    assert flat_with_grad["label"] == "STALLED_WITH_GRADIENT"
    assert oscillating["label"] == "OSCILLATING"


def test_official_prompt_covers_the_observable_labels() -> None:
    from code.llm_client import system_prompt

    prompt = system_prompt(("DESCENDING", "STALLED_NO_GRADIENT", "STALLED_WITH_GRADIENT", "OSCILLATING"))
    for label in ("DESCENDING", "STALLED_NO_GRADIENT", "STALLED_WITH_GRADIENT", "OSCILLATING"):
        assert label in prompt


# --------------------------------------------------------------------------- #
# Reheat
# --------------------------------------------------------------------------- #

_REHEAT_DECISION = {
    "ok": True,
    "latency_s": 0.001,
    "n_attempts": 1,
    "decision": {
        "diagnosis": "STALLED_WITH_GRADIENT",
        "justification": "stalled; reset the schedule",
        "expected_effect": 0.1,
        "action": {"eta_scale": None, "noise_sigma": 0.0, "restart": False, "reheat": True},
    },
}


def test_reheat_resets_the_decay_schedule(monkeypatch) -> None:
    monkeypatch.setattr("code.experiment._llm_call_sync", lambda *a, **k: dict(_REHEAT_DECISION))
    hamiltonian = vqe.build_hamiltonian("heisenberg", 2)
    energy_fn = vqe.make_energy_fn(hamiltonian, 2, 2)
    theta0 = vqe.random_initial_theta(np.random.default_rng(0), 2, 2)
    cfg = SPSAConfig(steps=31, seed=0)
    result = run_closed_loop(
        "spsa_llm",
        energy_fn,
        theta0,
        cfg,
        vqe.exact_ground_energy(hamiltonian),
        steps=31,
        n_window=10,
        threshold=0.05,
        seed=0,
        llm_cfg=_UNUSED_LLM,  # the monkeypatched call needs no real endpoint
    )
    events = result["events"]
    assert any(event["action"].get("reheat") for event in events)
    # The reheat lands at the window boundary k=10, so step 10 already uses the
    # fresh schedule: the step size jumps up from step 9 to step 10 and decays
    # again from there.
    a_k = result["a_k_curve"]
    a_fresh, _ = schedules(cfg, 0, 1.0)
    a_next, _ = schedules(cfg, 1, 1.0)
    assert a_k[10] > a_k[9]
    assert a_k[10] == pytest.approx(a_fresh, rel=1e-9)
    assert a_k[11] < a_k[10]
    assert a_k[11] == pytest.approx(a_next, rel=1e-9)


def test_reheat_event_records_the_shift(monkeypatch) -> None:
    monkeypatch.setattr("code.experiment._llm_call_sync", lambda *a, **k: dict(_REHEAT_DECISION))
    hamiltonian = vqe.build_hamiltonian("heisenberg", 2)
    energy_fn = vqe.make_energy_fn(hamiltonian, 2, 2)
    theta0 = vqe.random_initial_theta(np.random.default_rng(0), 2, 2)
    result = run_closed_loop(
        "spsa_llm",
        energy_fn,
        theta0,
        SPSAConfig(steps=25, seed=0),
        vqe.exact_ground_energy(hamiltonian),
        steps=25,
        n_window=10,
        threshold=0.05,
        seed=0,
        llm_cfg=_UNUSED_LLM,
    )
    first = result["events"][0]
    assert first["shift_before"] == 0
    assert first["action"]["reheat"] is True


# --------------------------------------------------------------------------- #
# Gates
# --------------------------------------------------------------------------- #

def _counting_fake(counter: dict):  # noqa: ANN001, ANN202
    def fake(*args, **kwargs):  # noqa: ANN002, ANN003
        counter["n"] += 1
        return dict(_REHEAT_DECISION)

    return fake


def test_gate_skips_calls_when_improving(monkeypatch) -> None:
    counter = {"n": 0}
    monkeypatch.setattr("code.experiment._llm_call_sync", _counting_fake(counter))
    hamiltonian = vqe.build_hamiltonian("heisenberg", 2)
    energy_fn = vqe.make_energy_fn(hamiltonian, 2, 2)
    theta0 = vqe.random_initial_theta(np.random.default_rng(0), 2, 2)
    result = run_closed_loop(
        "spsa_llm_gate",
        energy_fn,
        theta0,
        SPSAConfig(steps=25, seed=0),
        vqe.exact_ground_energy(hamiltonian),
        steps=25,
        n_window=10,
        threshold=0.05,
        seed=0,
        llm_cfg=_UNUSED_LLM,
        gate_improvement=-1e9,  # every window counts as progressing -> skip all
    )
    assert counter["n"] == 0
    assert result["n_llm_calls"] == 0
    assert all(event.get("gated") for event in result["events"])


def test_gate_calls_when_stalled(monkeypatch) -> None:
    counter = {"n": 0}
    monkeypatch.setattr("code.experiment._llm_call_sync", _counting_fake(counter))
    hamiltonian = vqe.build_hamiltonian("heisenberg", 2)
    energy_fn = vqe.make_energy_fn(hamiltonian, 2, 2)
    theta0 = vqe.random_initial_theta(np.random.default_rng(0), 2, 2)
    result = run_closed_loop(
        "spsa_llm_gate",
        energy_fn,
        theta0,
        SPSAConfig(steps=25, seed=0),
        vqe.exact_ground_energy(hamiltonian),
        steps=25,
        n_window=10,
        threshold=0.05,
        seed=0,
        llm_cfg=_UNUSED_LLM,
        gate_improvement=1e9,  # never counts as progressing -> always call
    )
    assert counter["n"] >= 2
    assert not any(event.get("gated") for event in result["events"])


def test_gate_restart_applies_restart_when_triggered() -> None:
    hamiltonian = vqe.build_hamiltonian("heisenberg", 2)
    energy_fn = vqe.make_energy_fn(hamiltonian, 2, 2)
    theta0 = vqe.random_initial_theta(np.random.default_rng(0), 2, 2)
    result = run_closed_loop(
        "spsa_gate_restart",
        energy_fn,
        theta0,
        SPSAConfig(steps=25, seed=0),
        vqe.exact_ground_energy(hamiltonian),
        steps=25,
        n_window=10,
        threshold=0.05,
        seed=0,
        gate_improvement=1e9,
    )
    assert result["n_changes"] >= 1
    assert all(event["action"]["restart"] for event in result["events"])


# --------------------------------------------------------------------------- #
# Engine B: initialization strategies
# --------------------------------------------------------------------------- #

def test_init_strategies_generate_valid_angles() -> None:
    rng = np.random.default_rng(0)
    size = 3 * 4 * 2
    for strategy in INIT_STRATEGIES:
        theta = theta0_for(strategy, rng, 4, 2)
        assert theta.shape == (size,)
        assert np.all(np.isfinite(theta))
    small = theta0_for("small_uniform", np.random.default_rng(0), 4, 2)
    assert np.max(np.abs(small)) <= np.pi / 8.0 + 1e-9
    block = theta0_for("block_periodic", np.random.default_rng(0), 4, 2)
    assert np.allclose(block[:12], block[12:])
    full = theta0_for("uniform_pm_pi", np.random.default_rng(0), 4, 2)
    assert np.max(np.abs(full)) > np.pi / 2.0


def test_decide_init_maps_the_choice() -> None:
    from code.llm_client import LLMConfig, decide_init

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {
                "model": "jev-test",
                "answers": {
                    "init_strategy": {
                        "type": "choice",
                        "choice": "near_zero",
                        "probabilities": {"near_zero": 0.7, "uniform_pm_pi2": 0.3},
                    },
                    "converges": {"type": "noul", "noul": 0.4},
                },
            }

    class FakeSession:
        def post(self, url, json=None, headers=None, timeout=None):  # noqa: A002, ANN001
            return FakeResponse()

    cfg = LLMConfig(api_style="systemone", api_key="k")
    result = decide_init({"hamiltonian": "heisenberg", "n_qubits": 16}, cfg, session=FakeSession())
    assert result["ok"] is True
    assert result["strategy"] == "near_zero"
    assert result["latency_s"] >= 0.0
