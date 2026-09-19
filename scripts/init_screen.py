"""Engine B offline screen: vanilla SPSA from every initialization strategy.

Runs the 16-qubit noiseless cell from each strategy and seed, and reports the
mean final gap per strategy. The best-fixed strategy becomes the classical
control for the warm-start experiment; the spread across strategies bounds
what the model's choice can win.
"""

from __future__ import annotations

import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from code import vqe
from code.init_strategies import INIT_STRATEGIES, theta0_for
from code.optimizer import SPSAConfig, run_spsa

N_QUBITS = 16
N_LAYERS = 2
SEEDS = list(range(10))
STEPS = 120

H = vqe.build_hamiltonian("heisenberg", N_QUBITS)
E_MIN = vqe.exact_ground_energy(H)
ENERGY_FN = vqe.make_energy_fn(H, N_QUBITS, N_LAYERS)

rows = []
for strategy in INIT_STRATEGIES:
    for seed in SEEDS:
        theta0 = theta0_for(strategy, np.random.default_rng(seed + 777), N_QUBITS, N_LAYERS)
        started = time.perf_counter()
        out = run_spsa(ENERGY_FN, theta0, SPSAConfig(steps=STEPS, seed=seed))
        wall = time.perf_counter() - started
        gap0 = abs(float(ENERGY_FN(theta0)) - E_MIN)
        gap_final = abs(float(out["final_energy"]) - E_MIN)
        row = {
            "strategy": strategy,
            "seed": seed,
            "gap0": gap0,
            "gap_final": gap_final,
            "wall_s": wall,
        }
        rows.append(row)
        print(json.dumps(row), flush=True)

summary = []
for strategy in INIT_STRATEGIES:
    entries = [r for r in rows if r["strategy"] == strategy]
    summary.append(
        {
            "strategy": strategy,
            "gap0_mean": float(np.mean([e["gap0"] for e in entries])),
            "gap_final_mean": float(np.mean([e["gap_final"] for e in entries])),
            "gap_final_min": float(np.min([e["gap_final"] for e in entries])),
            "gap_final_max": float(np.max([e["gap_final"] for e in entries])),
        }
    )

path = pathlib.Path("results/init_screen.json")
path.write_text(json.dumps({"runs": rows, "summary": summary}, indent=2), encoding="utf-8")
print("\n=== INIT SCREEN SUMMARY (16q, noiseless) ===")
for row in sorted(summary, key=lambda r: r["gap_final_mean"]):
    print(
        f"{row['strategy']:18s} gap0={row['gap0_mean']:7.2f} "
        f"final={row['gap_final_mean']:7.2f} (min {row['gap_final_min']:.2f})"
    )
print(f"\nwritten to {path}")
