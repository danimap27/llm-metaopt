"""Tests for the analysis layer: statistics, baseline policy and aggregation."""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest

from code.aggregate import summarize, to_latex, to_markdown
from code.labeler import Intervention, default_candidates
from code.policy import FEATURE_NAMES, LogisticPolicy, MajorityPolicy, window_features
from code.stats import bootstrap_ci, cohens_d_paired, holm_bonferroni, paired_permutation_test


def _window(improvement: float = 0.0, grad_last: float = 1e-4) -> dict:
    return {
        "n_window": 10,
        "energy_series": [1.0, 0.9, 0.8],
        "improvement": improvement,
        "energy": {"std": 0.05, "slope_per_step": -0.01},
        "grad_norm": {"last": grad_last, "mean": grad_last, "max": grad_last},
        "eta": {"a_k_last": 0.1, "c_k_last": 0.05, "eta_scale": 1.0},
        "theta": {"var": 0.3, "mean_abs": 0.5, "max_abs": 1.0, "dim": 12},
    }


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #

def test_bootstrap_ci_brackets_the_mean():
    values = np.linspace(0.0, 1.0, 40)
    stats = bootstrap_ci(values, n_boot=500, seed=1)
    assert stats["ci_low"] <= stats["mean"] <= stats["ci_high"]
    assert stats["n"] == 40


def test_permutation_test_detects_a_shift():
    rng = np.random.default_rng(0)
    a = rng.normal(1.0, 0.2, size=30)
    b = rng.normal(0.0, 0.2, size=30)
    assert paired_permutation_test(a, b, n_perm=2000, seed=0)["p_value"] < 0.01
    assert paired_permutation_test(a, a, n_perm=200, seed=0)["p_value"] == 1.0


def test_cohens_dz_and_holm_correction():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    b = np.array([0.0, 0.0, 0.0, 0.0])
    assert cohens_d_paired(a, b) > 1.0
    corrected = holm_bonferroni([0.001, 0.02, 0.2])
    assert corrected[0]["p_adjusted"] <= corrected[1]["p_adjusted"] <= corrected[2]["p_adjusted"]
    assert corrected[0]["reject"] is True


# --------------------------------------------------------------------------- #
# Baseline policy
# --------------------------------------------------------------------------- #

def test_window_features_length():
    features = window_features(_window())
    assert features.shape == (len(FEATURE_NAMES),)
    assert np.all(np.isfinite(features))


def test_logistic_policy_learns_a_separable_pattern():
    candidates = default_candidates()
    rng = np.random.default_rng(0)
    X = np.vstack([np.full((20, len(FEATURE_NAMES)), -1.0), np.full((20, len(FEATURE_NAMES)), 1.0)])
    X += 0.05 * rng.standard_normal(X.shape)
    y = [0] * 20 + [3] * 20
    policy = LogisticPolicy(candidates=candidates).fit(X, y, epochs=400, seed=0)
    assert policy.score(X, y) > 0.9
    assert policy.predict(_window()) in candidates


def test_policy_roundtrip(tmp_path: pathlib.Path):
    candidates = default_candidates()
    policy = MajorityPolicy(candidates=candidates).fit([2, 2, 3])
    assert policy.predict(_window()) == candidates[2]
    trained = LogisticPolicy(candidates=candidates).fit(
        np.vstack([np.zeros((5, len(FEATURE_NAMES))), np.ones((5, len(FEATURE_NAMES)))]), [1] * 5 + [2] * 5
    )
    path = tmp_path / "policy.json"
    trained.save(path)
    reloaded = LogisticPolicy.load(path)
    assert trained.weights is not None and reloaded.weights is not None
    assert reloaded.weights.shape == trained.weights.shape


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #

def _run_record(condition: str, base_id: str, gap: float, steps=None) -> dict:
    return {
        "kind": "run",
        "run_id": f"{base_id}_{condition}",
        "base_id": base_id,
        "condition": condition,
        "final_gap": gap,
        "final_energy": -3.0 + gap,
        "best_gap": gap,
        "steps_to_threshold": steps,
        "n_interventions": 3,
        "n_llm_calls": 3 if condition == "spsa_llm" else 0,
        "llm_latency_mean_s": 1.5 if condition == "spsa_llm" else None,
        "wall_s": 0.5,
    }


def test_aggregate_summarizes_and_compares():
    records = []
    for index in range(5):
        base = f"case{index}"
        records.append(_run_record("spsa", base, 0.20))
        records.append(_run_record("spsa_llm", base, 0.05))
    summary = summarize(records)
    conditions = {row["condition"]: row for row in summary["conditions"]}
    assert conditions["spsa"]["gap_mean"] == pytest.approx(0.20)
    assert conditions["spsa_llm"]["gap_mean"] == pytest.approx(0.05)
    assert conditions["spsa_llm"]["p_adjusted"] is not None
    assert conditions["spsa_llm"]["llm_latency_mean_s"] == pytest.approx(1.5)
    markdown = to_markdown(summary)
    latex = to_latex(summary)
    assert "spsa_llm" in markdown
    assert "\\begin{table}" in latex
    assert json.dumps(summary["conditions"][0])  # JSON serializable
