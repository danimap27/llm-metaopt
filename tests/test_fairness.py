"""Tests for the batch-3 fairness fixes (review findings B2, B3, B4, B5, M7)."""

from __future__ import annotations

import numpy as np
import pytest

from code import vqe
from code.detectors import PageHinkley
from code.experiment import run_closed_loop
from code.labeler import NO_OP, default_candidates
from code.optimizer import SPSAConfig, run_spsa
from code.policy import EffectRidgePolicy, LogisticPolicy, SCALAR_FIELDS, window_features
from code.telemetry import build_window


def test_page_hinkley_does_not_alarm_on_smooth_convergence() -> None:
    """Adversarial review B2: 98 percent false alarms on drift-free descent."""
    detector = PageHinkley(delta=0.01, threshold=0.05)
    gaps = 2.0 * np.exp(-np.linspace(0.0, 4.0, 120)) + 0.05
    alarms = [i for i, g in enumerate(gaps) if detector.update(float(g))]
    assert alarms == []


def test_page_hinkley_alarms_right_after_a_level_shift() -> None:
    detector = PageHinkley(delta=0.01, threshold=0.05)
    series = list(2.0 * np.exp(-np.linspace(0.0, 2.0, 60)))
    series += [v + 0.8 for v in 2.0 * np.exp(-np.linspace(0.0, 2.0, 60))]
    alarms = [i for i, v in enumerate(series) if detector.update(float(v))]
    assert alarms and alarms[0] <= 65


def test_policy_features_match_the_llm_window_fields() -> None:
    """B5: the classical controllers see the same window as the LLM."""
    hamiltonian = vqe.build_hamiltonian("heisenberg", 2)
    e_min = vqe.exact_ground_energy(hamiltonian)
    energy_fn = vqe.make_energy_fn(hamiltonian, 2, 2)
    theta = vqe.random_initial_theta(np.random.default_rng(0), 2, 2)
    out = run_spsa(energy_fn, theta, SPSAConfig(steps=25, seed=0), seed=0)
    window = build_window(out["history"], len(out["history"]) - 1, 10, theta=theta)
    features = window_features(window)
    assert features.shape == (10 + len(SCALAR_FIELDS),)
    assert np.all(np.isfinite(features))


def test_effect_ridge_policy_picks_the_best_effect() -> None:
    rng = np.random.default_rng(0)
    dim = 10 + len(SCALAR_FIELDS)
    x = rng.normal(size=(60, dim))
    effects = np.zeros((60, len(default_candidates())))
    # Action 2 is best exactly when the third feature is positive. Feature 2
    # is energy_series[2] in the flattened window.
    effects[:, 2] = np.where(x[:, 2] > 0, 1.0, -1.0)
    policy = EffectRidgePolicy().fit(x, effects)
    window = _synthetic_window()
    window["energy_series"][2] = 1.0
    assert policy.predict(window) == default_candidates()[2]


def _synthetic_window() -> dict:
    return {
        "n_window": 10,
        "step_end": 10,
        "energy_series": [0.0] * 10,
        "improvement": 0.0,
        "energy": {"first": 0.0, "last": 0.0, "min": 0.0, "mean": 0.0, "std": 0.0, "slope_per_step": 0.0},
        "grad_norm": {"last": 0.0, "mean": 0.0, "max": 0.0},
        "eta": {"a_k_last": 0.0, "c_k_last": 0.0, "eta_scale": 1.0},
        "theta": {"var": 0.0, "mean_abs": 0.0, "max_abs": 0.0, "dim": 12},
        "progress": {"drop_from_start": 0.0, "gap_above_best_so_far": 0.0},
    }


def test_policy_condition_raises_without_a_trained_model() -> None:
    """B4: spsa_policy must fail loudly instead of silently degrading to SPSA."""
    hamiltonian = vqe.build_hamiltonian("heisenberg", 2)
    energy_fn = vqe.make_energy_fn(hamiltonian, 2, 2)
    theta0 = vqe.random_initial_theta(np.random.default_rng(0), 2, 2)
    for condition in ("spsa_policy", "spsa_effect"):
        with pytest.raises(ValueError, match="trained policy"):
            run_closed_loop(
                condition,
                energy_fn,
                theta0,
                SPSAConfig(steps=11, seed=0),
                e_min=vqe.exact_ground_energy(hamiltonian),
                steps=11,
                n_window=5,
                threshold=0.05,
                seed=0,
            )


def test_main_block_safeguard_reverts_and_counts() -> None:
    """B3: the safeguard exists in the main block and applies uniformly."""
    hamiltonian = vqe.build_hamiltonian("heisenberg", 2)
    energy_fn = vqe.make_energy_fn(hamiltonian, 2, 2)
    theta0 = vqe.random_initial_theta(np.random.default_rng(0), 2, 2)

    class RestartAlways:
        def predict(self, window):  # noqa: ANN001, ANN202
            from code.labeler import Intervention

            return Intervention(restart=True)

    result = run_closed_loop(
        "spsa_policy",
        energy_fn,
        theta0,
        SPSAConfig(steps=30, seed=0),
        e_min=vqe.exact_ground_energy(hamiltonian),
        steps=30,
        n_window=5,
        threshold=0.05,
        seed=0,
        policy=RestartAlways(),
        safeguard_epsilon=0.0,
    )
    assert "n_reverted" in result
    assert result["n_reverted"] >= 1
    assert any(event.get("reverted") for event in result["events"])
