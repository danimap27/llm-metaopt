"""Manuscript figures built from the result records.

Every figure is regenerated from ``results/`` by ``make figures``, so a number in
the paper always traces back to data. The Agg backend is used because the build
runs headless and the output format is PDF, which is what the LaTeX build
includes.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  (backend must be selected first)
import numpy as np  # noqa: E402


def _write(figure: Any, path: Any) -> Any:
    figure.tight_layout()
    figure.savefig(path, format="pdf")
    plt.close(figure)
    return path


def _conditions(records: Sequence[Dict[str, Any]]) -> List[str]:
    return sorted({str(record["condition"]) for record in records})


def _stack(curves: Sequence[np.ndarray], reference: bool = False) -> np.ndarray:
    """Stack curves of equal length, optionally referenced to their first value."""
    length = min(len(curve) for curve in curves)
    stacked = np.asarray([np.asarray(curve, dtype=float)[:length] for curve in curves])
    if reference:
        stacked = stacked - stacked[:, :1]
    return stacked


def convergence_bands(records: Sequence[Dict[str, Any]], path: Any) -> Any:
    """Median energy with an interquartile band over seeds, one line per condition."""
    figure, axes = plt.subplots(figsize=(6.0, 3.6))
    for condition in _conditions(records):
        curves = [
            record["energy_curve"]
            for record in records
            if record.get("condition") == condition and "energy_curve" in record
        ]
        if not curves:
            continue
        stacked = _stack(curves, reference=True)
        steps = np.arange(stacked.shape[1])
        axes.plot(steps, np.median(stacked, axis=0), label=condition)
        axes.fill_between(
            steps,
            np.percentile(stacked, 25, axis=0),
            np.percentile(stacked, 75, axis=0),
            alpha=0.2,
        )
    axes.set_xlabel("fast-loop step")
    axes.set_ylabel("energy relative to the first step")
    axes.legend(frameon=False, fontsize=8)
    return _write(figure, path)


def latency_pareto(records: Sequence[Dict[str, Any]], path: Any) -> Any:
    """Final gap against the mean latency per slow-loop call, one point per condition."""
    figure, axes = plt.subplots(figsize=(4.8, 3.6))
    for condition in _conditions(records):
        group = [record for record in records if record.get("condition") == condition]
        gaps = [float(record["final_gap"]) for record in group]
        latencies = [float(record.get("llm_latency_mean_s") or 0.0) for record in group]
        axes.scatter(float(np.mean(latencies)), float(np.mean(gaps)), label=condition)
    axes.set_xlabel("mean latency per slow-loop call (s)")
    axes.set_ylabel("final gap")
    axes.legend(frameon=False, fontsize=8)
    return _write(figure, path)


def drift_timeline(records: Sequence[Dict[str, Any]], path: Any) -> Any:
    """Absolute gap per step for the drift block, with the true boundaries marked."""
    drift_records = [record for record in records if "gap_curve" in record]
    figure, axes = plt.subplots(figsize=(6.0, 3.6))
    for condition in _conditions(drift_records):
        curves = [record["gap_curve"] for record in drift_records if record.get("condition") == condition]
        if not curves:
            continue
        stacked = np.abs(_stack(curves))
        axes.plot(np.arange(stacked.shape[1]), np.median(stacked, axis=0), label=condition)
    boundaries = sorted(
        {int(value) for record in drift_records for value in record.get("meta", {}).get("boundaries", [])}
    )
    for boundary in boundaries:
        axes.axvline(boundary, color="grey", linestyle=":", linewidth=0.8)
    axes.set_xlabel("fast-loop step")
    axes.set_ylabel("absolute gap")
    axes.legend(frameon=False, fontsize=8)
    return _write(figure, path)


def _ground_truth_label(event: Dict[str, Any], tolerance: float) -> str:
    """Ground-truth regime of an event, from the simulator's own knowledge.

    Inside the first window after a boundary the objective really changed, so the
    correct diagnosis is a concept drift. Otherwise the run is converged when the
    gap is inside the tolerance and on a plateau when it is not.
    """
    if event.get("after_boundary"):
        return "CONCEPT_DRIFT"
    if abs(float(event.get("gap_before", 0.0))) <= tolerance:
        return "CONVERGENCIA_OK"
    return "MESETA_ENERGIA"


def regime_confusion(
    records: Sequence[Dict[str, Any]],
    path: Any,
    tolerance: float = 0.05,
) -> Any:
    """Confusion matrix between the reported diagnosis and the simulator ground truth."""
    pairs = []
    for record in records:
        for event in record.get("events", []):
            diagnosis = event.get("diagnosis")
            if not diagnosis:
                continue
            pairs.append((_ground_truth_label(event, tolerance), str(diagnosis)))
    labels = sorted({truth for truth, _ in pairs} | {prediction for _, prediction in pairs}) or ["(no events)"]
    matrix = np.zeros((len(labels), len(labels)), dtype=int)
    for truth, prediction in pairs:
        matrix[labels.index(truth), labels.index(prediction)] += 1
    figure, axes = plt.subplots(figsize=(4.8, 4.0))
    axes.imshow(matrix, cmap="Blues")
    axes.set_xticks(range(len(labels)), labels, rotation=45, ha="right", fontsize=7)
    axes.set_yticks(range(len(labels)), labels, fontsize=7)
    axes.set_xlabel("reported diagnosis")
    axes.set_ylabel("ground truth")
    return _write(figure, path)


def all_figures(records: Sequence[Dict[str, Any]], out_dir: Any, prefix: str = "") -> List[Any]:
    """Build the full figure set into ``out_dir`` and return the paths."""
    import pathlib

    directory = pathlib.Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    written = [
        convergence_bands(records, directory / f"{prefix}convergence_bands.pdf"),
        latency_pareto(records, directory / f"{prefix}latency_pareto.pdf"),
        drift_timeline(records, directory / f"{prefix}drift_timeline.pdf"),
        regime_confusion(records, directory / f"{prefix}regime_confusion.pdf"),
    ]
    return written


def main(argv: Sequence[str] | None = None) -> int:
    """Build every figure from a results file (``python -m code.figures``)."""
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Build the manuscript figures from a results file")
    parser.add_argument("--results", default="results/experiment.jsonl")
    parser.add_argument("--out", default="results/figures")
    parser.add_argument("--prefix", default="")
    args = parser.parse_args(argv)

    records: List[Dict[str, Any]] = []
    with open(args.results, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if not records:
        print(f"[figures] no records in {args.results}")
        return 1

    written = all_figures(records, args.out, prefix=args.prefix)
    for path in written:
        print(f"[figures] wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
