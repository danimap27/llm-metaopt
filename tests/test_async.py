"""Tests for the asynchronous slow-loop condition (spsa_llm_async)."""

from __future__ import annotations

import queue
import threading
import time

import numpy as np

from code import vqe
from code.experiment import run_closed_loop
from code.llm_client import LLMConfig
from code.optimizer import SPSAConfig

_RESPONSE = {
    "ok": True,
    "latency_s": 0.02,
    "n_attempts": 1,
    "decision": {
        "diagnosis": "MINIMO_LOCAL",
        "justification": "flat window",
        "expected_effect": 0.1,
        "action": {"eta_scale": 2.0, "noise_sigma": 0.0, "restart": False},
    },
}


def _hamiltonian_and_energy():
    hamiltonian = vqe.build_hamiltonian("heisenberg", 2)
    return hamiltonian, vqe.make_energy_fn(hamiltonian, 2, 2), vqe.exact_ground_energy(hamiltonian)


def _fake_call(window, llm_cfg, cache, session, out: queue.Queue) -> None:  # noqa: ANN001
    time.sleep(0.02)
    out.put(_RESPONSE)


def _slow_call(window, llm_cfg, cache, session, out: queue.Queue) -> None:  # noqa: ANN001
    time.sleep(30.0)  # lands long after every test timeout
    out.put(_RESPONSE)


def test_async_condition_applies_the_action_with_staleness(monkeypatch) -> None:
    monkeypatch.setattr("code.experiment._async_llm_call", _fake_call)
    hamiltonian, energy_fn, e_min = _hamiltonian_and_energy()
    theta0 = vqe.random_initial_theta(np.random.default_rng(0), 2, 2)
    result = run_closed_loop(
        "spsa_llm_async",
        energy_fn,
        theta0,
        SPSAConfig(steps=30, seed=0),
        e_min,
        steps=30,
        n_window=5,
        threshold=0.05,
        seed=0,
        llm_cfg=LLMConfig(base_url="http://unused", model="fake"),
    )
    applied = [event for event in result["events"] if event.get("applied")]
    assert result["n_llm_calls"] >= 2
    assert applied, "at least one async call must land and be applied"
    assert all("staleness_steps" in event for event in applied)
    assert any(event["staleness_steps"] >= 1 for event in applied)
    assert result["staleness_mean_steps"] is not None
    assert result["fast_step_wall_s"] > 0.0
    assert result["final_gap"] == float(result["final_gap"])


def test_async_condition_counts_unapplied_calls(monkeypatch) -> None:
    monkeypatch.setattr("code.experiment._async_llm_call", _slow_call)
    hamiltonian, energy_fn, e_min = _hamiltonian_and_energy()
    theta0 = vqe.random_initial_theta(np.random.default_rng(0), 2, 2)
    result = run_closed_loop(
        "spsa_llm_async",
        energy_fn,
        theta0,
        SPSAConfig(steps=12, seed=0),
        e_min,
        steps=12,
        n_window=5,
        threshold=0.05,
        seed=0,
        llm_cfg=LLMConfig(base_url="http://unused", model="fake"),
        async_drain_timeout_s=0.1,
    )
    assert result["n_llm_calls"] >= 1
    assert result["n_async_unapplied"] >= 1
    unapplied = [event for event in result["events"] if event.get("unapplied")]
    assert unapplied and all(not event.get("applied") for event in unapplied)


def test_thread_helper_puts_a_response_even_on_error(monkeypatch) -> None:
    from code import experiment as experiment_module

    def _boom(*args, **kwargs):  # noqa: ANN002, ANN003
        raise RuntimeError("endpoint exploded")

    monkeypatch.setattr(experiment_module, "llm_decide", _boom)
    out: queue.Queue = queue.Queue()
    experiment_module._async_llm_call({}, None, None, None, out)  # type: ignore[arg-type]
    response = out.get(timeout=1.0)
    assert response["ok"] is False
    assert "endpoint exploded" in response["attempts"][0]["error"]
