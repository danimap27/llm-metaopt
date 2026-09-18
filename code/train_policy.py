"""Train the classical controllers on the sweep dataset.

Turns ``data/sweep.jsonl`` into trained policies for the ``spsa_policy``
(logistic) and ``spsa_effect`` (contextual bandit) conditions. The split is by
seed and by Hamiltonian family, so the held-out numbers estimate transfer to
unseen problems, not memorization of trajectories.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from .labeler import Intervention, default_candidates
from .policy import EffectRidgePolicy, LogisticPolicy, window_features


def load_windows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("kind") == "window" and record.get("label"):
                rows.append(record)
    return rows


def build_arrays(
    rows: List[Dict[str, Any]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Features, argmax labels, per-action effects and candidate order."""
    candidates = [Intervention.from_dict(c["action"]) for c in rows[0]["label"]["candidates"]]
    grid = default_candidates()
    order = [grid.index(c) for c in candidates]  # index of each record candidate in the canonical grid
    x: List[np.ndarray] = []
    y: List[int] = []
    effects: List[List[float]] = []
    for record in rows:
        label = record["label"]
        baseline = float(label["baseline_gap"])
        full = [float("nan")] * len(grid)
        for slot, grid_index in enumerate(order):
            full[grid_index] = baseline - float(label["candidates"][slot]["gap"])
        effect = np.asarray(full, dtype=float)
        if np.isnan(effect).any():
            continue
        x.append(window_features(record["window"]))
        y.append(int(np.argmax(effect)))
        effects.append(effect.tolist())
    return np.asarray(x), np.asarray(y, dtype=int), np.asarray(effects), np.asarray(order)


def split_masks(
    rows: List[Dict[str, Any]],
    test_seeds: set,
    test_family: str | None,
) -> Tuple[np.ndarray, np.ndarray]:
    train, test = [], []
    for i, record in enumerate(rows):
        meta = record["meta"]
        is_test = meta["seed"] in test_seeds or (
            test_family is not None and meta["hamiltonian"] == test_family
        )
        (test if is_test else train).append(i)
    return np.asarray(train, dtype=int), np.asarray(test, dtype=int)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/sweep.jsonl"))
    parser.add_argument("--out-dir", type=Path, default=Path("models"))
    parser.add_argument("--test-seeds", type=int, nargs="*", default=[18, 19])
    parser.add_argument("--test-family", type=str, default=None)
    parser.add_argument("--model", choices=["logistic", "effect", "both"], default="both")
    args = parser.parse_args()

    rows = load_windows(args.dataset)
    if not rows:
        raise SystemExit(f"no labeled windows found in {args.dataset}; run the sweep first")
    x, y, effects, _order = build_arrays(rows)
    train_idx, test_idx = split_masks(rows, set(args.test_seeds), args.test_family)
    if len(train_idx) == 0 or len(test_idx) == 0:
        raise SystemExit("split produced an empty side; adjust --test-seeds/--test-family")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    report: Dict[str, Any] = {
        "n_windows": int(len(rows)),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "test_seeds": sorted(args.test_seeds),
        "test_family": args.test_family,
    }

    if args.model in ("logistic", "both"):
        logistic = LogisticPolicy().fit(x[train_idx], y[train_idx])
        acc = float(np.mean([logistic.predict(rows[i]["window"]).name for i in test_idx] ==
                            [Intervention.from_dict(rows[i]["label"]["best_action"]).name for i in test_idx]))
        logistic.save(args.out_dir / "policy.json")
        report["logistic_test_accuracy"] = acc

    if args.model in ("effect", "both"):
        effect = EffectRidgePolicy().fit(x[train_idx], effects[train_idx])
        picks = [effect.predict(rows[i]["window"]) for i in test_idx]
        truth = [Intervention.from_dict(rows[i]["label"]["best_action"]) for i in test_idx]
        acc = float(np.mean([p.name for p in picks] == [t.name for t in truth]))
        effect.save(args.out_dir / "policy_effect.json")
        report["effect_test_accuracy"] = acc

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
