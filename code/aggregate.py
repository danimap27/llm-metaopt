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

from .stats import bootstrap_ci, cohens_d_paired, holm_bonferroni, paired_permutation_test

BASELINE_CONDITION = "spsa"


def load_results(path: str | pathlib.Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("kind") == "run":
                records.append(record)
    return records


def group_by_condition(records: Sequence[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["condition"]].append(record)
    return dict(grouped)


def paired_values(
    condition_records: Sequence[Dict[str, Any]],
    baseline_records: Sequence[Dict[str, Any]],
    field: str = "final_gap",
) -> tuple[List[float], List[float], List[str]]:
    """Align two conditions by ``base_id`` so the comparison stays paired."""
    baseline_map = {record["base_id"]: record for record in baseline_records}
    pairs = [
        (record[field], baseline_map[record["base_id"]][field], record["base_id"])
        for record in condition_records
        if record["base_id"] in baseline_map
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

    return {"conditions": rows, "n_runs": len(records), "baseline": BASELINE_CONDITION}


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
    if not records:
        print(f"[aggregate] no run records found in {args.results}")
        return 1

    summary = summarize(records, seed=args.seed)
    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.md").write_text(to_markdown(summary), encoding="utf-8")
    (out_dir / "summary.tex").write_text(to_latex(summary), encoding="utf-8")
    write_csv(out_dir / "per_run.csv", records)

    print(to_markdown(summary))
    print(f"[aggregate] wrote {out_dir}/summary.md, summary.tex and per_run.csv ({summary['n_runs']} runs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
