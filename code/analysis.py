"""Analysis pipeline for the claims that need a second pass over the results.

Subcommands:

- ``break-even``: sweeps the (latency, per-call benefit) grid with the
  cost-benefit theory and writes the supervision break-even table. No endpoint
  or dataset needed.
- ``occlusions``: replays every recorded LLM call with each telemetry field
  occluded through a live endpoint and measures how often the decision flips.
  Needs ``--endpoint-url`` and a results file produced by the closed loop.

Both outputs land in ``results/tables/`` and are referenced by the manuscript.
"""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any, Dict, List

from .llm_client import LLMConfig, decide
from .theory import CostModel, break_even_gain
from .xai import FIDELITY_FIELDS


def break_even_table(
    latencies_s: List[float],
    gap_rates: List[float],
    fast_step_s: float = 0.05,
) -> List[Dict[str, float]]:
    """Grid of the per-call gap reduction that pays for a call, per configuration."""
    rows = []
    for latency in latencies_s:
        for rate in gap_rates:
            model = CostModel(fast_step_s=fast_step_s, call_latency_s=latency, baseline_gap_rate=rate)
            rows.append(
                {
                    "latency_s": float(latency),
                    "baseline_gap_rate": float(rate),
                    "fast_step_s": float(fast_step_s),
                    "required_gain_per_call": float(break_even_gain(model)),
                }
            )
    return rows


def cmd_break_even(args: argparse.Namespace) -> int:
    latencies = [float(item) for item in args.latencies.split(",")]
    rates = [float(item) for item in args.gap_rates.split(",")]
    rows = break_even_table(latencies, rates, fast_step_s=args.fast_step_s)
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"[analysis] break-even grid ({len(rows)} points) -> {out}")
    return 0


def cmd_occlusions(args: argparse.Namespace) -> int:
    """Replay recorded calls with each telemetry field occluded and count flips."""
    records: List[Dict[str, Any]] = []
    with open(args.results, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("kind") == "run" and record.get("condition") == "spsa_llm":
                records.append(record)

    llm_cfg = LLMConfig(base_url=args.endpoint_url, model=args.model, timeout=args.timeout)
    report: Dict[str, Dict[str, int]] = {
        field: {"n": 0, "flips": 0} for field in FIDELITY_FIELDS
    }
    for record in records:
        for event in record.get("events", []):
            window = event.get("window")
            if not isinstance(window, dict):
                continue
            baseline = decide(window, llm_cfg)
            if not baseline.get("ok"):
                continue
            baseline_action = json.dumps(baseline["decision"]["action"], sort_keys=True)
            for field in FIDELITY_FIELDS:
                if field not in window:
                    continue
                ablated = {key: value for key, value in window.items() if key != field}
                response = decide(ablated, llm_cfg)
                if not response.get("ok"):
                    continue
                report[field]["n"] += 1
                if json.dumps(response["decision"]["action"], sort_keys=True) != baseline_action:
                    report[field]["flips"] += 1

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[analysis] occlusion battery over {len(records)} runs -> {out}")
    return 0


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    be = sub.add_parser("break-even", help="cost-benefit break-even grid")
    be.add_argument("--latencies", default="0.05,0.5,2.0,12.0,60.0,110.0")
    be.add_argument("--gap-rates", default="0.0001,0.001,0.005,0.01")
    be.add_argument("--fast-step-s", type=float, default=0.05)
    be.add_argument("--out", default="results/tables/break_even.json")
    be.set_defaults(func=cmd_break_even)

    occ = sub.add_parser("occlusions", help="occlusion battery over recorded LLM calls")
    occ.add_argument("--results", default="results/experiment.jsonl")
    occ.add_argument("--endpoint-url", required=True)
    occ.add_argument("--model", default="qwen3.5:4b")
    occ.add_argument("--timeout", type=float, default=120.0)
    occ.add_argument("--out", default="results/tables/occlusions.json")
    occ.set_defaults(func=cmd_occlusions)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
