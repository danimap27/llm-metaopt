"""Tests for the manuscript figure generation."""

from __future__ import annotations

import pathlib

from code.figures import (
    all_figures,
    convergence_bands,
    drift_timeline,
    latency_pareto,
    regime_confusion,
)


def _closed_loop_record(condition: str, seed: int) -> dict:
    return {
        "kind": "run",
        "condition": condition,
        "run_id": f"{condition}_s{seed}",
        "base_id": f"b{seed}",
        "final_gap": 0.1 * (seed + 1),
        "llm_latency_mean_s": 1.5 if condition == "spsa_llm" else None,
        "energy_curve": [1.0 - 0.01 * index for index in range(40)],
    }


def _drift_record(condition: str, seed: int) -> dict:
    return {
        "kind": "drift_run",
        "condition": condition,
        "run_id": f"{condition}_s{seed}",
        "base_id": f"d{seed}",
        "final_gap": 0.5,
        "gap_curve": [1.0 - 0.005 * index for index in range(40)],
        "meta": {"boundaries": [20]},
        "events": [
            {"step": 10, "diagnosis": "MESETA_ENERGIA", "gap_before": 0.5, "after_boundary": False},
            {"step": 20, "diagnosis": "CONCEPT_DRIFT", "gap_before": 0.9, "after_boundary": True},
        ],
    }


def test_every_figure_is_written(tmp_path: pathlib.Path):
    closed = [_closed_loop_record(condition, seed) for condition in ("spsa", "spsa_llm") for seed in range(3)]
    drift = [_drift_record(condition, seed) for condition in ("drift_spsa", "drift_llm") for seed in range(3)]
    paths = [
        convergence_bands(closed, tmp_path / "convergence.pdf"),
        latency_pareto(closed, tmp_path / "pareto.pdf"),
        drift_timeline(drift, tmp_path / "drift.pdf"),
        regime_confusion(drift, tmp_path / "confusion.pdf"),
    ]
    for path in paths:
        written = pathlib.Path(path)
        assert written.exists()
        assert written.stat().st_size > 1000


def test_all_figures_builds_the_full_set(tmp_path: pathlib.Path):
    records = [_closed_loop_record("spsa", 0), _drift_record("drift_llm", 0)]
    written = all_figures(records, tmp_path, prefix="smoke_")
    assert len(written) == 4
    assert all(pathlib.Path(path).exists() for path in written)


def test_regime_confusion_handles_no_events(tmp_path: pathlib.Path):
    path = regime_confusion([], tmp_path / "empty.pdf")
    assert pathlib.Path(path).exists()
    assert pathlib.Path(path).stat().st_size > 500
