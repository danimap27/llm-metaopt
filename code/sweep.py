"""Generador de datos del Bloque 1: barrido SPSA + etiquetado contrafactual.

Cada run combina un Hamiltoniano, un numero de qubits, un nivel de ruido y una
semilla. Se ejecuta SPSA registrando la telemetria de cada ventana; al cerrar
cada ventana se diagnostica el regimen con ground-truth de simulador y, si no ha
convergido, se etiqueta la mejor intervencion contrafactual.

Salida: JSONL con un registro por ventana y un registro final por run.
Compatible con array jobs de Hercules (``--shard i --nshards N``) y reanudable
(los ``run_id`` ya presentes en el fichero de salida se saltan).
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
from .optimizer import SPSAConfig, run_spsa
from .regimes import RegimeConfig, diagnose, directional_gradient_variance
from .telemetry import build_window


def load_config(path: str | pathlib.Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_runs(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Producto cartesiano Hamiltoniano x qubits x ruido x semilla."""
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
    """Convierte NaN/Inf en None para que el JSONL sea JSON estricto."""
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
    """Reanudacion: run_id ya presentes en el fichero de salida."""
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
    """Ejecuta un run completo y escribe sus registros."""
    n_qubits = run["n_qubits"]
    n_layers = run["n_layers"]
    hamiltonian = vqe.build_hamiltonian(run["hamiltonian"], n_qubits)
    e_min = vqe.exact_ground_energy(hamiltonian)
    energy_fn = vqe.make_energy_fn(hamiltonian, n_qubits, n_layers, noise_p=run["noise_p"])

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
        "spsa": spsa_cfg.to_dict(),
        "n_window": n_window,
    }

    for end_index in range(n_window - 1, len(history) - 1, n_window):
        theta_k = np.asarray(history[end_index]["theta"], dtype=float)
        window = build_window(history, end_index, n_window, eta_scale=1.0, theta=theta_k)
        grad_var = directional_gradient_variance(
            energy_fn,
            theta_k,
            np.random.default_rng(run["seed"] + end_index),
            n_directions=regime_cfg.n_directions,
        )
        diagnosis = diagnose(
            energy=history[end_index]["energy"],
            e_min=e_min,
            window_improvement=window["improvement"],
            grad_norm_last=window["grad_norm"]["last"],
            grad_var=grad_var,
            cfg=regime_cfg,
        )
        record: Dict[str, Any] = {
            "kind": "window",
            "run_id": run["run_id"],
            "meta": meta,
            "step": history[end_index]["step"],
            "energy": history[end_index]["energy"],
            "gap": history[end_index]["energy"] - e_min,
            "window": window,
            "diagnosis": diagnosis,
        }
        if diagnosis["label"] != "CONVERGENCIA_OK" and labels_written < max_labels:
            record["label"] = label_state(
                energy_fn,
                theta_k,
                spsa_cfg,
                e_min=e_min,
                seed=run["seed"] * 1000 + end_index,
                lookahead=lookahead,
                init_scale=run["init_scale"],
            )
            labels_written += 1
        _write(handle, record)

    summary = {
        "kind": "run",
        "run_id": run["run_id"],
        "meta": meta,
        "final_energy": out["final_energy"],
        "final_gap": out["final_energy"] - e_min,
        "converged": bool(abs(out["final_energy"] - e_min) <= regime_cfg.tol_ok),
        "labels": labels_written,
        "wall_s": wall_s,
    }
    _write(handle, summary)
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Barrido SPSA + etiquetado contrafactual (LLM-MetaOpt)")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--out", default="data/sweep.jsonl")
    parser.add_argument("--limit", type=int, default=0, help="maximo de runs de este shard")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--nshards", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true", help="solo lista los runs")
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
    print(f"[sweep] shard {args.shard}/{args.nshards}: {len(pending)} runs pendientes de {len(runs)}")

    with out_path.open("a", encoding="utf-8") as handle:
        for index, run in enumerate(pending, start=1):
            summary = run_one(run, cfg, handle)
            print(
                f"[{index}/{len(pending)}] {summary['run_id']}: "
                f"E_final={summary['final_energy']:.4f} gap={summary['final_gap']:.4f} "
                f"converged={summary['converged']} labels={summary['labels']} "
                f"({summary['wall_s']:.1f}s)"
            )
    print(f"[sweep] salida: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
