"""Run the drifting-objective block (E2) and write one record per run.

Usage
-----
    .venv/bin/python -m code.run_drift --config configs/default.yaml --seeds 3 --mock-llm
    .venv/bin/python -m code.run_drift --conditions drift_llm --llm-model qwen3.5:4b --cache results/llm_cache.jsonl

The record carries the drift metrics (detection latency, recovery steps, area
under the gap curve), the event trace with the actions and, for the language
model condition, the diagnosis, the justification, the expected effect and the
per-call latency.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import yaml

from . import vqe
from .drift import DriftingObjective, build_drift_spec
from .experiment import DRIFT_CONDITIONS, run_drift_loop
from .llm_cache import ResponseCache
from .llm_client import LLMConfig
from .optimizer import SPSAConfig


def _clean_nan(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _clean_nan(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean_nan(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Drifting-objective block of the LLM-MetaOpt study")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--out", default="results/drift.jsonl")
    parser.add_argument("--conditions", default=",".join(DRIFT_CONDITIONS))
    parser.add_argument("--seeds", type=int, default=3, help="number of seeds starting at zero")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--mock-llm", action="store_true", help="offline smoke: the language model condition falls back to the detector controller")
    parser.add_argument("--llm-base-url", default="http://127.0.0.1:11434/v1")
    parser.add_argument("--llm-model", default="qwen3.5:4b")
    parser.add_argument("--cache", default=None, help="response cache path (enables replay)")
    parser.add_argument("--replay", action="store_true", help="serve every call from the cache")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    with open(args.config, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    conditions: List[str] = [item.strip() for item in args.conditions.split(",") if item.strip()]
    if args.mock_llm:
        conditions = ["drift_detector" if item == "drift_llm" else item for item in conditions]
        print("[run_drift] mock mode: drift_llm runs use the detector controller")
    unknown = [item for item in conditions if item not in DRIFT_CONDITIONS]
    if unknown:
        parser.error(f"unknown conditions: {unknown}. Options: {DRIFT_CONDITIONS}")

    runs = [(condition, seed) for seed in range(args.seeds) for condition in conditions]
    if args.limit > 0:
        runs = runs[: args.limit]
    if args.dry_run:
        print(json.dumps([{"condition": c, "seed": s} for c, s in runs], indent=2))
        return 0

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cache = ResponseCache(args.cache, replay_only=args.replay) if args.cache else None
    llm_cfg = LLMConfig(base_url=args.llm_base_url, model=args.llm_model) if "drift_llm" in conditions else None
    spsa_cfg = SPSAConfig(**cfg["spsa"])
    n_window = int(cfg["windows"]["n_window"])
    tol_ok = float(cfg["regimes"]["tol_ok"])
    detector_threshold = float(cfg["drift"].get("detector_threshold", 0.5))
    safeguard_epsilon = float(cfg["drift"].get("safeguard_epsilon", 0.05))

    with out_path.open("a", encoding="utf-8") as handle:
        for index, (condition, seed) in enumerate(runs, start=1):
            spec = build_drift_spec(cfg, np.random.default_rng(seed))
            objective = DriftingObjective(spec)
            theta0 = vqe.random_initial_theta(
                np.random.default_rng(seed + 7), 2, spec.n_layers, scale=spec.init_scale
            )
            result = run_drift_loop(
                condition,
                objective,
                theta0,
                spsa_cfg,
                n_window=n_window,
                threshold=tol_ok,
                seed=seed,
                detector_threshold=detector_threshold,
                safeguard_epsilon=safeguard_epsilon,
                llm_cfg=llm_cfg,
                cache=cache,
            )
            record = {
                "kind": "drift_run",
                "run_id": f"{condition}_s{seed}",
                "seed": seed,
                "condition": condition,
                "meta": {
                    "segments": spec.segments,
                    "boundaries": spec.boundaries(),
                    "n_window": n_window,
                    "detector_threshold": detector_threshold,
                    "safeguard_epsilon": safeguard_epsilon,
                    "spsa": spsa_cfg.to_dict(),
                    "llm": llm_cfg.to_dict() if llm_cfg is not None else None,
                    "cache": args.cache,
                    "mock_llm": bool(args.mock_llm),
                },
                **result,
            }
            handle.write(json.dumps(_clean_nan(record), ensure_ascii=False) + "\n")
            handle.flush()
            print(
                f"[{index}/{len(runs)}] {record['run_id']}: area={result['area_under_gap']:.2f} "
                f"final_gap={result['final_gap']:.4f} changes={result['n_changes']} "
                f"reverted={result['n_reverted']} latency={result['detection_latency']}"
            )
    print(f"[run_drift] output: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
