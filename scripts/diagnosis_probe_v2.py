"""Diagnosis probe (state v2): Jev vs observable ground truth on real windows.

Reads labeled windows from a sweep dataset, queries Jev through the project
client with the observable taxonomy, and reports the confusion between the
model's diagnosis and `diagnosis_observable`.
"""

from __future__ import annotations

import json
import os
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from code.llm_client import LLMConfig, decide
from code.regimes import OBSERVABLE_REGIMES

DATASET = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "data/stall_l4.jsonl")
SAMPLES_PER_CLASS = 3

windows = []
for line in DATASET.open(encoding="utf-8"):
    record = json.loads(line)
    if record.get("kind") != "window" or "diagnosis_observable" not in record:
        continue
    windows.append(record)

by_class: dict[str, list] = {}
for record in windows:
    by_class.setdefault(record["diagnosis_observable"]["label"], []).append(record)

sample = []
for label, records in by_class.items():
    sample.extend(records[:SAMPLES_PER_CLASS])
print(f"windows: {len(windows)}  classes: { {k: len(v) for k, v in by_class.items()} }  probing {len(sample)}")

if not os.environ.get("TYPESAFE_API_KEY"):
    raise SystemExit("TYPESAFE_API_KEY not set")

cfg = LLMConfig(api_style="systemone", model="jev-latest", regimes=OBSERVABLE_REGIMES)
confusion: dict[str, dict[str, int]] = {}
latencies = []
correct = 0
for record in sample:
    truth = record["diagnosis_observable"]["label"]
    response = decide(record["window"], cfg)
    if not response.get("ok"):
        print(f"FAILED {record['run_id']} step {record['step']}: {response['attempts'][0]['error'][:120]}")
        continue
    latencies.append(response["latency_s"])
    guess = response["decision"]["diagnosis"]
    confusion.setdefault(truth, {}).setdefault(guess, 0)
    confusion[truth][guess] += 1
    correct += int(guess == truth)
    print(
        f"truth={truth:22s} guess={guess:22s} "
        f"p={response['decision']['diagnosis_probabilities'] and max(response['decision']['diagnosis_probabilities'].values()):.2f} "
        f"action={response['decision']['action_name']:12s} latency={response['latency_s']:.2f}s"
    )

n = len(latencies)
print(f"\nagreement: {correct}/{n} = {correct / n:.2f}" if n else "no scored calls")
print(f"latency: mean {sum(latencies) / n:.2f}s" if n else "")
print("confusion (truth -> guesses):", json.dumps(confusion, indent=1))
