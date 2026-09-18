"""Build the paper tables from the closed-loop experiment results.

Reads the JSONL produced by ``code.experiment`` and writes, per condition:
mean final gap with a bootstrap confidence interval, success rate, intervention
counts, LLM latency statistics and paired permutation tests against the SPSA
baseline (Holm-Bonferroni corrected across the family of comparisons).

Outputs
-------
- ``<out>/summary.md``: Markdown table.
- ``<out>/summary.tex``: LaTeX (booktabs) table ready to paste into the paper.
- ``<out>/per_run.csv``: one row per run.
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .stats import bootstrap_ci, cohens_d_paired, holm_bonferroni, paired_permutation_test
from .xai import explanation_report

BASELINE_CONDITION = "spsa"
DRIFT_BASELINE_CONDITION = "drift_spsa"


def load_results(path: str | pathlib.Path) -> List[Dict[str, Any]]:
    """Run records deduplicated by run_id, keeping the last occurrence.

    Re-running a partially completed sweep appends new attempts, and the last
    completed record for a run_id is the authoritative one.
    """
    by_id: Dict[str, Dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("kind") == "run":
                by_id[str(record.get("run_id"))] = record
    return list(by_id.values())


def load_drift_records(path: str | pathlib.Path) -> List[Dict[str, Any]]:
    """Drift-block records (``kind == "drift_run"``) from a results file."""
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("kind") == "drift_run":
                records.append(record)
    return records


def group_by_condition(records: Sequence[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["condition"]].append(record)
    return dict(grouped)


def _pair_key(record: Dict[str, Any]) -> str:
    """Pairing key shared by every condition of the same physical run."""
    return str(record.get("base_id") or record.get("run_id"))


def paired_values(
    condition_records: Sequence[Dict[str, Any]],
    baseline_records: Sequence[Dict[str, Any]],
    field: str = "final_gap",
) -> tuple[List[float], List[float], List[str]]:
    """Align two conditions by pairing key so the comparison stays within-run."""
    baseline_map = {_pair_key(record): record for record in baseline_records}
    pairs = [
        (record[field], baseline_map[_pair_key(record)][field], _pair_key(record))
        for record in condition_records
        if _pair_key(record) in baseline_map
    ]
    if not pairs:
        return [], [], []
    values, baseline_values, keys = zip(*pairs)
    return list(values), list(baseline_values), list(keys)


def summarize(records: Sequence[Dict[str, Any]], seed: int = 0) -> Dict[str, Any]:
    grouped = group_by_condition(records)
    baseline = grouped.get(BASELINE_CONDITION, [])
    rows: List[Dict[str, Any]] = []
    pending_p: List[float] = []
    pending_rows: List[Dict[str, Any]] = []

    for condition, condition_records in sorted(grouped.items()):
        gaps = [float(record["final_gap"]) for record in condition_records]
        stats = bootstrap_ci(gaps, seed=seed)
        # Review finding M2: report the gap relative to the exact minimum so the
        # mean is not dominated by the largest instances in the sweep.
        relative = [
            float(record["final_gap"]) / max(abs(float(record.get("meta", {}).get("e_min", 0.0))), 1e-12)
            for record in condition_records
            if record.get("meta", {}).get("e_min") is not None
        ]
        rel_stats = bootstrap_ci(np.asarray(relative), seed=seed) if relative else None
        successes = [1.0 if record.get("steps_to_threshold") is not None else 0.0 for record in condition_records]
        interventions = [float(record["n_interventions"]) for record in condition_records]
        latencies = [
            float(record["llm_latency_mean_s"])
            for record in condition_records
            if record.get("llm_latency_mean_s") is not None
        ]
        row: Dict[str, Any] = {
            "condition": condition,
            "n": int(stats["n"]),
            "gap_mean": float(stats["mean"]),
            "gap_std": float(stats["std"]),
            "gap_ci_low": float(stats["ci_low"]),
            "gap_ci_high": float(stats["ci_high"]),
            "relative_gap_mean": float(rel_stats["mean"]) if rel_stats else None,
            "relative_gap_ci_low": float(rel_stats["ci_low"]) if rel_stats else None,
            "relative_gap_ci_high": float(rel_stats["ci_high"]) if rel_stats else None,
            "success_rate": float(sum(successes) / len(successes)) if successes else 0.0,
            "interventions_mean": float(sum(interventions) / len(interventions)) if interventions else 0.0,
            "llm_latency_mean_s": float(sum(latencies) / len(latencies)) if latencies else None,
            "p_value": None,
            "p_adjusted": None,
            "effect_size_dz": None,
        }
        if condition != BASELINE_CONDITION and baseline:
            values, base_values, _ = paired_values(condition_records, baseline)
            if values:
                test = paired_permutation_test(values, base_values, seed=seed)
                row["p_value"] = float(test["p_value"])
                row["effect_size_dz"] = cohens_d_paired(values, base_values)
                pending_p.append(float(test["p_value"]))
                pending_rows.append(row)
        rows.append(row)

    if pending_p:
        for row, corrected in zip(pending_rows, holm_bonferroni(pending_p)):
            row["p_adjusted"] = float(corrected["p_adjusted"])

    return {
        "conditions": rows,
        "n_runs": len(records),
        "baseline": BASELINE_CONDITION,
        # The declared multiple-comparison family: one test per non-baseline
        # condition against spsa, pooled across the sweep cells.
        "holm_family": [row["condition"] for row in pending_rows],
    }


def summarize_by_cell(records: Sequence[Dict[str, Any]], seed: int = 0) -> Dict[str, Any]:
    """Per-cell summaries over (hamiltonian, n_qubits, noise_p).

    Review finding M2: pooling raw gaps across 2- and 16-qubit problems makes
    the pooled mean uninterpretable, so the paper reports each cell separately
    and the pooled table only with the relative gap.
    """
    cells: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        key = (
            record.get("hamiltonian", "unknown"),
            int(record.get("n_qubits", 0)),
            float(record.get("noise_p", 0.0)),
        )
        cells[key].append(record)
    summaries = []
    for key in sorted(cells, key=lambda item: (str(item[0]), item[1], item[2])):
        cell_records = cells[key]
        summary = summarize(cell_records, seed=seed)
        summary["cell"] = {
            "hamiltonian": key[0],
            "n_qubits": key[1],
            "noise_p": key[2],
        }
        summaries.append(summary)
    return {"cells": summaries, "n_runs": len(records)}


def to_markdown(summary: Dict[str, Any]) -> str:
    header = (
        "| Condition | n | Final gap (mean ± 95% CI) | Success rate | Interventions | LLM latency (s) "
        "| p (Holm) | d_z |\n"
        "|---|---|---|---|---|---|---|---|\n"
    )
    lines = [header]
    for row in summary["conditions"]:
        p_value = "—" if row["p_adjusted"] is None else f'{row["p_adjusted"]:.3g}'
        dz = "—" if row["effect_size_dz"] is None else f'{row["effect_size_dz"]:.2f}'
        latency = "—" if row["llm_latency_mean_s"] is None else f'{row["llm_latency_mean_s"]:.2f}'
        lines.append(
            f'| {row["condition"]} | {row["n"]} '
            f'| {row["gap_mean"]:.4f} ± {0.5 * (row["gap_ci_high"] - row["gap_ci_low"]):.4f} '
            f'| {row["success_rate"]:.2f} | {row["interventions_mean"]:.1f} | {latency} | {p_value} | {dz} |\n'
        )
    return "".join(lines)


def to_latex(summary: Dict[str, Any]) -> str:
    lines = [
        "% Generated by code.aggregate -- do not edit by hand.",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Final energy gap after $N$ fast-loop steps. Success rate is the fraction of runs "
        "reaching the convergence threshold. $p$ values are paired sign-flip permutation tests against "
        "vanilla SPSA, Holm-Bonferroni corrected over the family of comparisons.}",
        "\\label{tab:closed-loop}",
        "\\begin{tabular}{lrrrrr}",
        "\\toprule",
        "Condition & Runs & Final gap & Success & Interventions & $p_{\\mathrm{Holm}}$ \\\\",
        "\\midrule",
    ]
    for row in summary["conditions"]:
        p_value = "--" if row["p_adjusted"] is None else f'{row["p_adjusted"]:.3g}'
        condition_label = str(row["condition"]).replace("_", "\\_")
        lines.append(
            f'{condition_label} & {row["n"]} & '
            f'${row["gap_mean"]:.4f} \\pm {0.5 * (row["gap_ci_high"] - row["gap_ci_low"]):.4f}$ & '
            f'{row["success_rate"]:.2f} & {row["interventions_mean"]:.1f} & {p_value} \\\\'
        )
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(lines)


def summarize_drift(records: Sequence[Dict[str, Any]], seed: int = 0) -> Dict[str, Any]:
    """Per-condition drift metrics with paired tests against plain SPSA."""
    grouped = group_by_condition(records)
    baseline = grouped.get(DRIFT_BASELINE_CONDITION, [])
    rows: List[Dict[str, Any]] = []
    pending_p: List[float] = []
    pending_rows: List[Dict[str, Any]] = []

    def _mean(values: Sequence[Any]) -> Optional[float]:
        clean = [float(value) for value in values if value is not None]
        return float(np.mean(clean)) if clean else None

    for condition, condition_records in sorted(grouped.items()):
        areas = [float(record["area_under_gap"]) for record in condition_records]
        stats = bootstrap_ci(areas, seed=seed)
        row: Dict[str, Any] = {
            "condition": condition,
            "n": int(stats["n"]),
            "area_mean": float(stats["mean"]),
            "area_std": float(stats["std"]),
            "area_ci_low": float(stats["ci_low"]),
            "area_ci_high": float(stats["ci_high"]),
            "final_gap_mean": _mean([record.get("final_gap") for record in condition_records]),
            "detection_latency_mean": _mean([record.get("detection_latency") for record in condition_records]),
            "detection_rate": _mean(
                [1.0 if record.get("detection_latency") is not None else 0.0 for record in condition_records]
            ),
            "recovery_steps_mean": _mean([record.get("recovery_steps") for record in condition_records]),
            "recovery_rate": _mean(
                [1.0 if record.get("recovery_steps") is not None else 0.0 for record in condition_records]
            ),
            "interventions_mean": _mean([record.get("n_changes") for record in condition_records]),
            "reverted_mean": _mean([record.get("n_reverted") for record in condition_records]),
            "p_value": None,
            "p_adjusted": None,
            "effect_size_dz": None,
        }
        if condition != DRIFT_BASELINE_CONDITION and baseline:
            values, base_values, _ = paired_values(condition_records, baseline, field="area_under_gap")
            if values:
                test = paired_permutation_test(values, base_values, seed=seed)
                row["p_value"] = float(test["p_value"])
                row["effect_size_dz"] = cohens_d_paired(values, base_values)
                pending_p.append(float(test["p_value"]))
                pending_rows.append(row)
        rows.append(row)

    if pending_p:
        for row, corrected in zip(pending_rows, holm_bonferroni(pending_p)):
            row["p_adjusted"] = float(corrected["p_adjusted"])

    return {"conditions": rows, "n_runs": len(records), "baseline": DRIFT_BASELINE_CONDITION}


def to_markdown_drift(summary: Dict[str, Any]) -> str:
    """Markdown table of the drift block."""
    lines = [
        "| Condition | Runs | Area under gap (mean plus/minus 95 percent CI) | Final gap | Detection rate | "
        "Detection latency | Recovery rate | Recovery steps | Interventions | Reverted | p (Holm) |\n",
        "|---|---|---|---|---|---|---|---|---|---|---|\n",
    ]
    for row in summary["conditions"]:
        p_value = "not applicable" if row["p_adjusted"] is None else f'{row["p_adjusted"]:.3g}'
        final_gap = "n/a" if row["final_gap_mean"] is None else f'{row["final_gap_mean"]:.4f}'
        latency = "n/a" if row["detection_latency_mean"] is None else f'{row["detection_latency_mean"]:.1f}'
        recovery = "n/a" if row["recovery_steps_mean"] is None else f'{row["recovery_steps_mean"]:.1f}'
        lines.append(
            f'| {row["condition"]} | {row["n"]} | {row["area_mean"]:.2f} '
            f'plus/minus {0.5 * (row["area_ci_high"] - row["area_ci_low"]):.2f} | {final_gap} '
            f'| {row["detection_rate"]:.2f} | {latency} | {row["recovery_rate"]:.2f} | {recovery} '
            f'| {row["interventions_mean"]:.1f} | {row["reverted_mean"]:.1f} | {p_value} |\n'
        )
    return "".join(lines)


def to_latex_drift(summary: Dict[str, Any]) -> str:
    """LaTeX table of the drift block."""
    table_amp = " " + chr(38) + " "
    row_end = chr(92) + chr(92)
    lines = [
        "% Generated by code.aggregate -- do not edit by hand.",
        chr(92) + "begin{table}[t]",
        chr(92) + "centering",
        chr(92) + "caption{Drifting-objective block. The area under the absolute gap curve measures how far the "
        "run stays from the exact minimum of the current objective. Detection rate is the fraction of "
        "boundaries followed by a real intervention. Recovery rate is the fraction of boundaries where the run "
        "returns inside the convergence threshold before the next drift. Values are compared against plain "
        "SPSA with a paired sign-flip permutation test, Holm-Bonferroni corrected.}",
        chr(92) + "label{tab:drift}",
        chr(92) + "begin{tabular}{lrrrrrr}",
        chr(92) + "toprule",
        "Condition" + table_amp + "Runs" + table_amp + "Area under gap" + table_amp + "Final gap"
        + table_amp + "Detect. rate" + table_amp + "Recover. rate" + table_amp + "$p_{\mathrm{Holm}}$ " + row_end,
        chr(92) + "midrule",
    ]
    for row in summary["conditions"]:
        p_value = "--" if row["p_adjusted"] is None else f'{row["p_adjusted"]:.3g}'
        final_gap = "--" if row["final_gap_mean"] is None else f'{row["final_gap_mean"]:.4f}'
        condition_label = str(row["condition"]).replace("_", chr(92) + "_")
        lines.append(
            condition_label + table_amp + f'{row["n"]}' + table_amp
            + f'${row["area_mean"]:.2f} ' + chr(92) + f'pm {0.5 * (row["area_ci_high"] - row["area_ci_low"]):.2f}$'
            + table_amp + final_gap + table_amp + f'{row["detection_rate"]:.2f}' + table_amp
            + f'{row["recovery_rate"]:.2f}' + table_amp + p_value + " " + row_end
        )
    lines += [chr(92) + "bottomrule", chr(92) + "end{tabular}", chr(92) + "end{table}", ""]
    return "\n".join(lines)


def explanation_table(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Pair the expected effect of every language-model call with the observed effect.

    The expected effect is what the model stated it wanted to achieve over the
    next window. The observed effect is the gap reduction over that same window
    measured from the energy at the decision point (``energy_pre``, captured
    immediately before the intervention), not from the post-intervention point,
    so a restart cannot start the measurement from a degraded state (review
    finding M3). Events with a missing expected effect are counted as dropped
    and reported, not silently skipped.
    """
    predicted: List[float] = []
    observed: List[float] = []
    n_dropped = 0
    n_events = 0
    for record in records:
        if record.get("condition") != "spsa_llm":
            continue
        energies = [float(value) for value in record.get("energy_curve", [])]
        if len(energies) < 2:
            continue
        e_min = float(record.get("meta", {}).get("e_min", 0.0))
        n_window = int(record.get("meta", {}).get("n_window", 10))
        for event in record.get("events", []):
            n_events += 1
            expected = event.get("expected_effect")
            if not isinstance(expected, (int, float)):
                n_dropped += 1
                continue
            start = int(event["step"])
            end = min(start + n_window, len(energies) - 1)
            if start >= len(energies) or end <= start:
                n_dropped += 1
                continue
            energy_pre = event.get("energy_pre")
            if isinstance(energy_pre, (int, float)):
                reference = float(energy_pre)
            else:
                reference = energies[start]
            predicted.append(float(expected))
            observed.append((reference - e_min) - (energies[end] - e_min))
    if not predicted:
        return {"n": 0, "n_dropped": n_dropped, "n_events": n_events}
    report = explanation_report(predicted, observed)
    report["n_dropped"] = n_dropped
    report["n_events"] = n_events
    return report


def explanation_markdown(report: Dict[str, Any]) -> str:
    """Single-table report of the explanation-fidelity metrics."""
    if report.get("n", 0) == 0:
        return "No language-model events were scored.\n"
    pearson = report["pearson_r"]
    pearson_text = "undefined" if pearson is None else f"{pearson:.3f}"
    return (
        "| Metric | Value |\n|---|---|\n"
        f'| Calls scored | {report["n"]} |\n'
        f'| Events dropped (missing expected effect or window) | {report.get("n_dropped", 0)} |\n'
        f'| Signed accuracy | {report["signed_accuracy"]:.3f} |\n'
        f'| Majority-sign null accuracy | {report.get("majority_sign_accuracy", float("nan")):.3f} |\n'
        f'| Signed accuracy permutation p | {report.get("signed_accuracy_perm_p", float("nan")):.4f} |\n'
        f'| Pearson r | {pearson_text} |\n'
        f'| Mean absolute error (energy units) | {report["mae"]:.4f} |\n'
        f'| Mean predicted effect | {report["mean_predicted"]:.4f} |\n'
        f'| Mean observed effect | {report["mean_observed"]:.4f} |\n'
    )


def write_csv(path: pathlib.Path, records: Sequence[Dict[str, Any]]) -> None:
    fields = [
        "run_id",
        "base_id",
        "condition",
        "hamiltonian",
        "n_qubits",
        "noise_p",
        "seed",
        "final_energy",
        "final_gap",
        "best_gap",
        "steps_to_threshold",
        "n_interventions",
        "n_changes",
        "n_llm_calls",
        "llm_latency_mean_s",
        "n_energy_evals",
        "n_reverted",
        "wall_s",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            writer.writerow({field: record.get(field) for field in fields})


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Aggregate closed-loop results into paper tables")
    parser.add_argument("--results", default="results/experiment.jsonl")
    parser.add_argument("--out", default="results/tables")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    records = load_results(args.results)
    drift_records = load_drift_records(args.results)
    if not records and not drift_records:
        print(f"[aggregate] no run or drift records found in {args.results}")
        return 1

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if records:
        summary = summarize(records, seed=args.seed)
        (out_dir / "summary.md").write_text(to_markdown(summary), encoding="utf-8")
        (out_dir / "summary.tex").write_text(to_latex(summary), encoding="utf-8")
        write_csv(out_dir / "per_run.csv", records)
        print(to_markdown(summary))

        cell_summary = summarize_by_cell(records, seed=args.seed)
        (out_dir / "summary_by_cell.json").write_text(
            json.dumps(cell_summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"[aggregate] wrote per-cell summaries for {len(cell_summary['cells'])} cells")

        explanation = explanation_table(records)
        (out_dir / "explanation.md").write_text(explanation_markdown(explanation), encoding="utf-8")
        print(
            "[aggregate] explanation fidelity: "
            f"n={explanation.get('n', 0)} signed_accuracy={explanation.get('signed_accuracy', 'n/a')}"
        )

    if drift_records:
        drift_summary = summarize_drift(drift_records, seed=args.seed)
        (out_dir / "drift_summary.md").write_text(to_markdown_drift(drift_summary), encoding="utf-8")
        (out_dir / "drift_summary.tex").write_text(to_latex_drift(drift_summary), encoding="utf-8")
        print(to_markdown_drift(drift_summary))

    print(
        f"[aggregate] wrote tables into {out_dir} "
        f"({len(records)} closed-loop runs, {len(drift_records)} drift runs)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
