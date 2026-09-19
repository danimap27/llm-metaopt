"""Diagnosis probe (state v2): Jev vs observable ground truth on real windows.

Reads telemetry windows from closed-loop results (events of the LLM
condition store the raw window), computes the observable ground truth with
`diagnose_observable`, queries Jev through the project client and reports the
confusion between the model's diagnosis and the ground truth.
"""

from __future__ import annotations

import json
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from code.llm_client import LLMConfig, decide
from code.regimes import OBSERVABLE_REGIMES, diagnose_observable

RESULTS = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "results/stall_cell.jsonl")
SAMPLES_PER_CLASS = 3

windows = []
for line in RESULTS.open(encoding="utf-8"):
    record = json.loads(line)
    if record.get("kind") != "run" or record.get("condition") != "spsa_llm":
        continue
    for event in record.get("events", []):
        window = event.get("window")
        if isinstance(window, dict):
            truth = diagnose_observable(
                window_improvement=float(window["improvement"]),
                energy_std=float(window["energy"]["std"]),
                grad_norm_last=float(window["grad_norm"]["last"]),
            )
            windows.append((record["run_id"], event["step"], window, truth["label"]))

by_class: dict[str, list] = {}
for entry in windows:
    by_class.setdefault(entry[3], []).append(entry)

sample = []
for label, entries in by_class.items():
    sample.extend(entries[:SAMPLES_PER_CLASS])
print(f"windows: {len(windows)}  classes: { {k: len(v) for k, v in by_class.items()} }  probing {len(sample)}")

if not os.environ.get("TYPESAFE_API_KEY"):
    raise SystemExit("TYPESAFE_API_KEY not set")

cfg = LLMConfig(api_style="systemone", model="jev-latest", regimes=OBSERVABLE_REGIMES)
confusion: dict[str, dict[str, int]] = {}
latencies = []
correct = 0
for run_id, step, window, truth in sample:
    response = decide(window, cfg)
    if not response.get("ok"):
        print(f"FAILED {run_id} step {step}: {response['attempts'][0]['error'][:120]}")
        continue
    latencies.append(response["latency_s"])
    guess = response["decision"]["diagnosis"]
    confusion.setdefault(truth, {}).setdefault(guess, 0)
    confusion[truth][guess] += 1
    correct += int(guess == truth)
    probs = response["decision"]["diagnosis_probabilities"] or {}
    print(
        f"truth={truth:22s} guess={guess:22s} p={max(probs.values()):.2f} "
        f"action={response['decision']['action_name']:12s} latency={response['latency_s']:.2f}s"
    )

n = len(latencies)
if n:
    print(f"\nagreement: {correct}/{n} = {correct / n:.2f}")
    print(f"latency: mean {sum(latencies) / n:.2f}s")
    print("confusion (truth -> guesses):", json.dumps(confusion, indent=1))
else:
    print("no scored calls")

