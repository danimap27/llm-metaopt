"""Tests for the drifting-objective closed loop."""

from __future__ import annotations

import numpy as np
import pytest

from code import vqe
from code.drift import DriftSpec, DriftingObjective
from code.experiment import _drift_response_metrics, run_drift_loop
from code.llm_client import LLMConfig
from code.optimizer import SPSAConfig


def _spec(steps: int = 8) -> DriftSpec:
    return DriftSpec(
        segments=[
            {"hamiltonian": "xy", "n_qubits": 2, "noise_p": 0.0, "steps": steps},
            {"hamiltonian": "heisenberg", "n_qubits": 2, "noise_p": 0.0, "steps": steps},
        ],
        n_layers=1,
        init_scale=0.5,
    )


def _theta0(spec: DriftSpec) -> np.ndarray:
    return vqe.random_initial_theta(np.random.default_rng(0), 2, spec.n_layers, scale=spec.init_scale)


def test_baseline_condition_reports_metrics_without_interventions():
    spec = _spec()
    result = run_drift_loop(
        "drift_spsa",
        DriftingObjective(spec),
        _theta0(spec),
        SPSAConfig(seed=0),
        n_window=4,
        threshold=0.05,
        seed=0,
    )
    assert result["condition"] == "drift_spsa"
    assert result["n_changes"] == 0
    assert result["detection_latency"] is None
    assert len(result["gap_curve"]) == spec.total_steps
    assert len(result["final_gap_by_segment"]) == len(spec.segments)
    assert result["area_under_gap"] > 0.0


def test_detector_condition_restarts_after_the_boundary():
    spec = _spec()
    result = run_drift_loop(
        "drift_detector",
        DriftingObjective(spec),
        _theta0(spec),
        SPSAConfig(seed=0),
        n_window=4,
        threshold=0.05,
        seed=0,
        detector_threshold=0.05,
    )
    assert result["n_changes"] >= 1
    assert result["detection_latency"] is not None
    assert any(event["action"]["restart"] for event in result["events"])
    assert any(event.get("alarm") for event in result["events"])
    # The safeguard belongs to the supervised controller, not to the classical baseline.
    assert result["n_reverted"] == 0


def test_detector_restarts_respect_the_cooldown():
    spec = _spec()
    result = run_drift_loop(
        "drift_detector",
        DriftingObjective(spec),
        _theta0(spec),
        SPSAConfig(seed=0),
        n_window=4,
        threshold=0.05,
        seed=0,
        detector_threshold=-1.0,  # alarms on every sample
    )
    alarms = [int(event["step"]) for event in result["events"] if event.get("alarm")]
    assert len(alarms) >= 2
    assert all(later - earlier >= 4 for earlier, later in zip(alarms, alarms[1:]))


def test_llm_condition_survives_an_unreachable_endpoint():
    spec = _spec()
    result = run_drift_loop(
        "drift_llm",
        DriftingObjective(spec),
        _theta0(spec),
        SPSAConfig(seed=0),
        n_window=4,
        threshold=0.05,
        seed=0,
        llm_cfg=LLMConfig(base_url="http://127.0.0.1:9/v1", model="unreachable", timeout=0.5),
    )
    assert result["llm_failures"] >= 1
    assert result["n_changes"] == 0
    assert all(event.get("llm_failed") for event in result["events"])


def test_safeguard_reverts_a_degrading_intervention():
    spec = _spec()
    result = run_drift_loop(
        "drift_random",
        DriftingObjective(spec),
        _theta0(spec),
        SPSAConfig(seed=0),
        n_window=4,
        threshold=0.05,
        seed=1,
        safeguard_epsilon=0.0,
    )
    assert result["n_reverted"] >= 1
    assert all("theta_snapshot" not in event for event in result["events"])


def test_unknown_condition_is_rejected():
    spec = _spec()
    with pytest.raises(ValueError):
        run_drift_loop("drift_nonsense", DriftingObjective(spec), _theta0(spec), SPSAConfig(seed=0))


def test_response_metrics_average_over_boundaries():
    events = [
        {"step": 4, "changed": True},
        {"step": 12, "changed": False},
        {"step": 16, "changed": True},
    ]
    gaps = [10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 0.04, 0.02, 0.01, 0.0, 1.0, 2.0, 3.0, 4.0, 0.03, 0.02]
    metrics = _drift_response_metrics(events, gaps, boundaries=[8, 16], threshold=0.05)
    assert metrics["n_boundaries"] == 2
    assert metrics["n_detected"] == 2
    assert metrics["detection_latency"] == pytest.approx(4.0)
    assert metrics["recovery_steps"] == pytest.approx(0.0)


def test_response_metrics_report_none_when_nothing_happens():
    metrics = _drift_response_metrics([], [5.0, 5.0, 5.0], boundaries=[2], threshold=0.05)
    assert metrics["detection_latency"] is None
    assert metrics["recovery_steps"] is None
    assert metrics["n_detected"] == 0
