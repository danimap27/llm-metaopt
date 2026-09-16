"""Tests for the drift and explanation aggregation tables."""

from __future__ import annotations

import pytest

from code.aggregate import (
    explanation_markdown,
    explanation_table,
    summarize_drift,
    to_latex_drift,
    to_markdown_drift,
)


def _drift_record(condition: str, base: str, area: float, latency, recovery) -> dict:
    return {
        "kind": "drift_run",
        "run_id": f"{condition}_{base}",
        "base_id": base,
        "condition": condition,
        "area_under_gap": area,
        "final_gap": area / 100.0,
        "detection_latency": latency,
        "recovery_steps": recovery,
        "n_changes": 4,
        "n_reverted": 1,
    }


def test_summarize_drift_reports_paired_tests():
    records = []
    for index in range(4):
        base = f"d{index}"
        records.append(_drift_record("drift_spsa", base, area=200.0, latency=None, recovery=10.0))
        records.append(_drift_record("drift_llm", base, area=120.0, latency=2.0, recovery=4.0))
    summary = summarize_drift(records)
    rows = {row["condition"]: row for row in summary["conditions"]}
    assert rows["drift_llm"]["area_mean"] == pytest.approx(120.0)
    assert rows["drift_llm"]["detection_latency_mean"] == pytest.approx(2.0)
    assert rows["drift_llm"]["recovery_steps_mean"] == pytest.approx(4.0)
    assert rows["drift_llm"]["detection_rate"] == pytest.approx(1.0)
    assert rows["drift_spsa"]["p_adjusted"] is None
    assert rows["drift_llm"]["p_adjusted"] is not None
    assert summary["baseline"] == "drift_spsa"


def test_drift_tables_render_from_a_summary():
    records = [
        _drift_record("drift_spsa", "a", 200.0, None, None),
        _drift_record("drift_llm", "a", 120.0, 2.0, 4.0),
    ]
    summary = summarize_drift(records)
    markdown = to_markdown_drift(summary)
    latex = to_latex_drift(summary)
    assert "drift_llm" in markdown
    assert "\\begin{table}" in latex
    assert "drift\\_llm" in latex
    assert "tab:drift" in latex


def test_explanation_table_pairs_predicted_with_observed():
    record = {
        "kind": "run",
        "condition": "spsa_llm",
        "run_id": "r1",
        "base_id": "r1",
        "meta": {"e_min": 0.0, "n_window": 2},
        "energy_curve": [1.0, 0.8, 0.5, 0.4],
        "events": [
            {"step": 0, "expected_effect": 0.5, "action": {"eta_scale": 1.0, "noise_sigma": 0.0, "restart": False}},
            {"step": 2, "expected_effect": 0.1, "action": {"eta_scale": 1.0, "noise_sigma": 0.0, "restart": False}},
        ],
    }
    report = explanation_table([record])
    assert report["n"] == 2
    assert report["mean_predicted"] == pytest.approx(0.3)
    assert report["mean_observed"] == pytest.approx(0.3)
    assert report["signed_accuracy"] == pytest.approx(1.0)


def test_explanation_table_ignores_other_conditions_and_empty_events():
    other = {"kind": "run", "condition": "spsa", "energy_curve": [1.0, 0.5], "events": []}
    assert explanation_table([other])["n"] == 0
    assert "No language-model events" in explanation_markdown({"n": 0})
