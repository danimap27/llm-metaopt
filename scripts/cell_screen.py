"""Cell screening: find cells with genuine stalls (low late velocity, large gap).

Runs vanilla SPSA on candidate cells and records the quantities the
eligibility criterion needs: initial gap, final gap, late-phase velocity
(gap/s) and wall time. Cells where the late velocity is near zero while a
large gap remains are the only ones where supervision can pay.
"""

from __future__ import annotations

import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from code import vqe
from code.optimizer import SPSAConfig, run_spsa

CELLS = [
    ("heisenberg", 8, 0.05, 2),
    ("heisenberg", 8, 0.05, 4),   # deeper ansatz
    ("heisenberg", 16, 0.0, 2),   # noiseless 16q (trajectory-free path)
    ("tfim", 8, 0.05, 2),
    ("xy", 8, 0.05, 2),
    ("heisenberg", 8, 0.02, 2),
]

results = []
for hamiltonian, n_qubits, noise_p, n_layers in CELLS:
    H = vqe.build_hamiltonian(hamiltonian, n_qubits)
    e_min = vqe.exact_ground_energy(H)
    energy_fn = vqe.make_energy_fn(H, n_qubits, n_layers, noise_p=noise_p)
    for seed in (0, 1, 2):
        theta0 = vqe.random_initial_theta(np.random.default_rng(seed), n_qubits, n_layers, scale=0.5)
        started = time.perf_counter()
        out = run_spsa(energy_fn, theta0, SPSAConfig(steps=120, seed=seed))
        wall = time.perf_counter() - started
        gaps = [abs(float(rec["energy"]) - e_min) for rec in out["history"]]
        n = len(gaps)
        late = gaps[int(n * 2 / 3):]
        velocity_late = (late[0] - late[-1]) / max(wall * (n - int(n * 2 / 3)) / n, 1e-9)
        entry = {
            "cell": f"{hamiltonian}_n{n_qubits}_p{noise_p}_L{n_layers}",
            "seed": seed,
            "gap0": gaps[0],
            "gap_final": gaps[-1],
            "velocity_late_gap_per_s": velocity_late,
            "wall_s": wall,
            "backend": vqe.energy_backend(n_qubits, noise_p),
        }
        results.append(entry)
        print(json.dumps(entry), flush=True)

by_cell: dict[str, list[dict]] = {}
for entry in results:
    by_cell.setdefault(entry["cell"], []).append(entry)
summary = []
for cell, entries in by_cell.items():
    gap_final = float(np.mean([e["gap_final"] for e in entries]))
    vel = float(np.mean([e["velocity_late_gap_per_s"] for e in entries]))
    wall = float(np.mean([e["wall_s"] for e in entries]))
    required = vel * 0.65  # break-even requirement at Jev latency
    summary.append(
        {
            "cell": cell,
            "gap_final_mean": gap_final,
            "velocity_late_mean": vel,
            "required_gain_at_0.65s": required,
            "wall_s_mean": wall,
            "backend": entries[0]["backend"],
        }
    )
path = pathlib.Path("results/cell_screen.json")
path.write_text(json.dumps({"runs": results, "summary": summary}, indent=2), encoding="utf-8")
print("\n=== SUMMARY ===")
for row in summary:
    print(
        f"{row['cell']:28s} gap_final={row['gap_final_mean']:7.2f} "
        f"velocity_late={row['velocity_late_mean']:7.3f} gap/s "
        f"required@0.65s={row['required_gain_at_0.65s']:6.3f} wall={row['wall_s_mean']:6.1f}s"
    )
print(f"\nwritten to {path}")
