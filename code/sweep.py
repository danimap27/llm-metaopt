"""Dataset generation: SPSA sweep with counterfactual labeling.

Each run combines one Hamiltonian, qubit count, noise level and seed. SPSA is
executed while recording the telemetry of every window. At each window boundary
the regime is diagnosed against simulator ground truth and, when the run has not
converged, the best counterfactual intervention is labeled.

Output: JSONL with one record per window plus one summary record per run.
Shard-aware (``--shard i --nshards N``) and resumable (runs already present in
the output file are skipped), so it can be launched as an HPC array job.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import time
from itertools import product
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import yaml

from . import vqe
from .labeler import label_state
from .optimizer import SPSAConfig, counting_energy_fn, run_spsa
from .regimes import RegimeConfig, diagnose, directional_gradient_variance
from .telemetry import build_window


def load_config(path: str | pathlib.Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_runs(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Cartesian product Hamiltonian x qubits x noise x seed."""
    sweep = cfg["sweep"]
    runs: List[Dict[str, Any]] = []
    for hamiltonian, n_qubits, noise_p, seed in product(
        sweep["hamiltonians"], sweep["n_qubits"], sweep["noise_p"], sweep["seeds"]
    ):
        runs.append(
            {
                "run_id": f"{hamiltonian}_n{n_qubits}_p{noise_p:g}_s{seed}",
                "hamiltonian": hamiltonian,
                "n_qubits": int(n_qubits),
                "noise_p": float(noise_p),
                "seed": int(seed),
                "n_layers": int(cfg["ansatz"]["n_layers"]),
                "init_scale": float(sweep["init_scale"]),
            }
        )
    return runs


def _clean_nan(value: Any) -> Any:
    """Turn NaN/Inf into None so the JSONL stays strict JSON."""
    if isinstance(value, dict):
        return {k: _clean_nan(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean_nan(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write(handle, record: Dict[str, Any]) -> None:
    handle.write(json.dumps(_clean_nan(record), ensure_ascii=False) + "\n")
    handle.flush()


def read_done_run_ids(path: pathlib.Path) -> set[str]:
    """Resume support: run identifiers already present in the output file."""
    done: set[str] = set()
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("kind") == "run" and record.get("run_id"):
                done.add(record["run_id"])
    return done


def run_one(run: Dict[str, Any], cfg: Dict[str, Any], handle) -> Dict[str, Any]:
    """Execute one run and write its records."""
    n_qubits = run["n_qubits"]
    n_layers = run["n_layers"]
    hamiltonian = vqe.build_hamiltonian(run["hamiltonian"], n_qubits)
    e_min = vqe.exact_ground_energy(hamiltonian)
    shots = int(cfg["sweep"].get("shots", vqe.DEFAULT_TRAJECTORY_SHOTS))
    energy_fn, eval_counter = counting_energy_fn(
        vqe.make_energy_fn(hamiltonian, n_qubits, n_layers, noise_p=run["noise_p"], shots=shots)
    )

    rng = np.random.default_rng(run["seed"])
    theta0 = vqe.random_initial_theta(rng, n_qubits, n_layers, scale=run["init_scale"])
    spsa_cfg = SPSAConfig(seed=run["seed"], **cfg["spsa"])
    regime_cfg = RegimeConfig(**cfg["regimes"])

    started = time.perf_counter()
    out = run_spsa(energy_fn, theta0, spsa_cfg)
    wall_s = time.perf_counter() - started

    history = out["history"]
    n_window = int(cfg["windows"]["n_window"])
    lookahead = int(cfg["windows"]["lookahead"])
    max_labels = int(cfg["windows"]["max_labels_per_run"])
    labels_written = 0

    meta = {
        "hamiltonian": run["hamiltonian"],
        "n_qubits": n_qubits,
        "n_layers": n_layers,
        "noise_p": run["noise_p"],
        "seed": run["seed"],
        "init_scale": run["init_scale"],
        "e_min": e_min,
        "backend": vqe.energy_backend(n_qubits, run["noise_p"]),
        "shots": shots if vqe.energy_backend(n_qubits, run["noise_p"]) == "statevector_trajectory" else None,
        "n_energy_evals": eval_counter["n"],
        "spsa": spsa_cfg.to_dict(),
        "n_window": n_window,
    }

    # Label windows uniformly across the whole run, not just the first
    # ones, and include converged states so that "do nothing" is learnable.
    all_ends = list(range(n_window - 1, len(history) - 1, n_window))
    if len(all_ends) > max_labels:
        picks = np.linspace(0, len(all_ends) - 1, max_labels).round().astype(int)
        label_ends = {all_ends[int(i)] for i in picks}
    else:
        label_ends = set(all_ends)
    label_repeats = int(cfg["windows"].get("label_repeats", 3))

    for end_index in all_ends:
        theta_k = np.asarray(history[end_index]["theta"], dtype=float)
        window = build_window(history, end_index, n_window, eta_scale=1.0, theta=theta_k)
        grad_var = directional_gradient_variance(
            energy_fn,
            theta_k,
            np.random.default_rng(np.random.SeedSequence([run["seed"], 555, end_index])),
            n_directions=regime_cfg.n_directions,
        )
        diagnosis = diagnose(
            energy=float(history[end_index]["energy"]),
            e_min=e_min,
            window_improvement=float(window["improvement"]),
            grad_norm_last=float(window["grad_norm"]["last"]),  # type: ignore[index]
            grad_var=grad_var,
            cfg=regime_cfg,
            h_norm=float(np.sum(np.abs(hamiltonian.coeffs))),
        )
        record: Dict[str, Any] = {
            "kind": "window",
            "run_id": run["run_id"],
            "meta": meta,
            "step": history[end_index]["step"],
            "energy": history[end_index]["energy"],
            "gap": float(history[end_index]["energy"]) - e_min,
            "window": window,
            "diagnosis": diagnosis,
        }
        if end_index in label_ends:
            label_rng = np.random.default_rng(np.random.SeedSequence([run["seed"], 987_654, end_index]))
            record["label"] = label_state(
                energy_fn,
                theta_k,
                spsa_cfg,
                e_min=e_min,
                rng=label_rng,
                lookahead=lookahead,
                init_scale=run["init_scale"],
                k_offset=end_index,
                n_repeats=label_repeats,
            )
            labels_written += 1
        _write(handle, record)

    summary = {
        "kind": "run",
        "run_id": run["run_id"],
        "meta": meta,
        "final_energy": out["final_energy"],
        "final_gap": float(out["final_energy"]) - e_min,
        "converged": bool(abs(float(out["final_energy"]) - e_min) <= regime_cfg.tol_ok),
        "labels": labels_written,
        "wall_s": wall_s,
    }
    _write(handle, summary)
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="SPSA sweep with counterfactual labeling (LLM-MetaOpt)")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--out", default="data/sweep.jsonl")
    parser.add_argument("--limit", type=int, default=0, help="maximum runs in this shard")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--nshards", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true", help="list the runs only")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    runs = build_runs(cfg)
    runs = [run for i, run in enumerate(runs) if i % args.nshards == args.shard]
    if args.limit > 0:
        runs = runs[: args.limit]
    if args.dry_run:
        print(json.dumps(runs, indent=2, ensure_ascii=False))
        return 0

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = read_done_run_ids(out_path)
    pending = [run for run in runs if run["run_id"] not in done]
    print(f"[sweep] shard {args.shard}/{args.nshards}: {len(pending)} pending runs out of {len(runs)}")

    with out_path.open("a", encoding="utf-8") as handle:
        for index, run in enumerate(pending, start=1):
            summary = run_one(run, cfg, handle)
            print(
                f"[{index}/{len(pending)}] {summary['run_id']}: "
                f"E_final={float(summary['final_energy']):.4f} gap={float(summary['final_gap']):.4f} "
                f"converged={summary['converged']} labels={summary['labels']} "
                f"({float(summary['wall_s']):.1f}s)"
            )
    print(f"[sweep] output: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
