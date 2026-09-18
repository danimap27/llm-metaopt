"""Tests for the explanation-fidelity layer."""

from __future__ import annotations

import numpy as np
import pytest

from code.xai import FIDELITY_FIELDS, drop_field, explanation_report, occlusions


def _window(with_theta: bool = True) -> dict:
    window = {
        "n_window": 10,
        "energy_series": [1.0, 0.9],
        "improvement": 0.1,
        "energy": {"std": 0.05, "slope_per_step": -0.01},
        "grad_norm": {"last": 0.2, "mean": 0.3, "max": 0.4},
        "eta": {"a_k_last": 0.1, "c_k_last": 0.05, "eta_scale": 1.0},
    }
    if with_theta:
        window["theta"] = {"var": 0.3, "mean_abs": 0.5, "max_abs": 1.0, "dim": 12}
    return window


def test_drop_field_removes_the_requested_block():
    window = drop_field(_window(), "grad_norm")
    assert "grad_norm" not in window
    assert "energy" in window


def test_occlusions_iterate_over_every_field_present():
    variants = occlusions(_window())
    assert set(variants) == set(FIDELITY_FIELDS)
    assert all("n_window" in variant for variant in variants.values())

    without_theta = occlusions(_window(with_theta=False))
    assert "theta" not in without_theta


def test_explanation_report_matches_a_perfect_predictor():
    report = explanation_report([0.10, 0.20, -0.05], [0.10, 0.20, -0.05])
    assert report["n"] == 3
    assert report["signed_accuracy"] == pytest.approx(1.0)
    assert report["pearson_r"] == pytest.approx(1.0, abs=1e-6)
    assert report["mae"] == pytest.approx(0.0)


def test_explanation_report_flags_an_anti_correlated_forecaster():
    report = explanation_report([1.0, 2.0, 3.0], [-1.0, -2.0, -3.0])
    assert report["pearson_r"] < 0.0
    assert report["signed_accuracy"] == pytest.approx(0.0)


def test_explanation_report_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        explanation_report([0.1, 0.2], [0.1])
    with pytest.raises(ValueError):
        explanation_report([], [])


def test_explanation_report_marks_an_undefined_correlation_as_none() -> None:
    report = explanation_report([1.0, 1.0, 1.0], [-1.0, 2.0, 3.0])
    # A constant predictor makes Pearson correlation undefined: report None,
    # not zero, so the table cannot be misread as "no linear relationship".
    assert report["pearson_r"] is None
    assert not np.isnan(report["mae"])
    # The null models exist and are computed.
    assert 0.0 <= report["majority_sign_accuracy"] <= 1.0
    assert 0.0 < report["signed_accuracy_perm_p"] <= 1.0
