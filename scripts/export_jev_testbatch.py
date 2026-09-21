"""Export a self-contained Jev test batch from real benchmark windows.

Picks a spread of telemetry windows from closed-loop results, computes the
observable ground truth, queries Jev through the project client, and writes a
JSONL file for an independent rerun: the first line documents the exact
question contract, every following line is one window with the ground truth
and our reference answers.

Usage: .venv/bin/python scripts/export_jev_testbatch.py [out.jsonl]
"""

from __future__ import annotations

import json
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from code.llm_client import (
    _GAIN_ANCHORS,
    _GAIN_CRITERIA,
    _REGIME_DESCRIPTIONS,
    _SYSTEMONE_ACTION_DESCRIPTIONS,
    LLMConfig,
    decide,
)
from code.regimes import OBSERVABLE_REGIMES, diagnose_observable

OUT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "results/jev_testbatch.jsonl")


def collect(path: str, cell: str) -> list[dict]:
    entries = []
    for line in pathlib.Path(path).open(encoding="utf-8"):
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
                entries.append(
                    {
                        "cell": cell,
                        "run_id": record["run_id"],
                        "step": event["step"],
                        "window": window,
                        "observable_truth": truth["label"],
                    }
                )
    return entries


stall = collect("results/stall_cell.jsonl", "L4_8q_p0.05")
smoke = collect("results/smoke_cell.jsonl", "4q_p0.02")

# Spread selection: several steps from the stall cell, two phases from four
# different smoke runs (early and late), keeping every distinct class found.
selection: list[dict] = []
stall_by_step = {e["step"]: e for e in stall}
for step in (10, 30, 50, 70, 90, 110):
    if step in stall_by_step:
        selection.append(stall_by_step[step])
smoke_runs = sorted({e["run_id"] for e in smoke})[:4]
for run_id in smoke_runs:
    run_entries = sorted([e for e in smoke if e["run_id"] == run_id], key=lambda e: e["step"])
    if run_entries:
        selection.append(run_entries[1])
        selection.append(run_entries[-2])

if not os.environ.get("TYPESAFE_API_KEY"):
    raise SystemExit("TYPESAFE_API_KEY not set")

cfg = LLMConfig(api_style="systemone", model="jev-latest", regimes=OBSERVABLE_REGIMES)
questions_spec = {
    "regime": {
        "type": "choice",
        "instructions": "Diagnose the optimization regime of this telemetry window",
        "criteria": {name: _REGIME_DESCRIPTIONS[name] for name in OBSERVABLE_REGIMES},
    },
    "action": {
        "type": "choice",
        "instructions": "Choose one intervention for the fast SPSA optimizer",
        "criteria": dict(_SYSTEMONE_ACTION_DESCRIPTIONS),
    },
    "improve_prob": {
        "type": "noul",
        "instructions": "The energy gap will improve over the next window of optimization steps",
    },
    "expected_gain": {
        "type": "score",
        "instructions": "Expected gap reduction in energy units over the next window",
        "criteria": list(_GAIN_CRITERIA),
        "anchors": list(_GAIN_ANCHORS),
    },
}

lines = [
    json.dumps(
        {
            "kind": "jev_testbatch",
            "created": "2026-09-21",
            "source": "llm-metaopt closed-loop benchmark",
            "how_to_use": (
                "Send each line's 'window' as the System One state with the questions "
                "below; compare the returned diagnosis and action with observable_truth "
                "and with reference_answers (our pipeline, same day, jev-latest)."
            ),
            "questions": questions_spec,
            "scoring": {
                "diagnosis_agreement": "fraction of windows where regime.choice equals observable_truth",
                "action_agreement": "fraction of windows where action.choice equals reference_answers.action_name",
                "style_note": "observable_truth uses the observable taxonomy thresholds: descent_eps 0.002, gradient_eps 0.005, oscillation std 0.05",
            },
        },
        ensure_ascii=False,
    )
]

for index, entry in enumerate(selection, start=1):
    response = decide(entry["window"], cfg)
    if not response.get("ok"):
        print(f"FAILED on {entry['run_id']} step {entry['step']}")
        continue
    decision = response["decision"]
    lines.append(
        json.dumps(
            {
                "kind": "window",
                "id": f"w{index:02d}",
                "cell": entry["cell"],
                "run_id": entry["run_id"],
                "step": entry["step"],
                "window": entry["window"],
                "observable_truth": entry["observable_truth"],
                "reference_answers": {
                    "diagnosis": decision["diagnosis"],
                    "diagnosis_probabilities": decision["diagnosis_probabilities"],
                    "action_name": decision["action_name"],
                    "action": decision["action"],
                    "improvement_probability": decision["improvement_probability"],
                    "gain_score": decision["gain_score"],
                    "latency_s": round(response["latency_s"], 3),
                },
            },
            ensure_ascii=False,
        )
    )
    print(
        f"[{index}/{len(selection)}] {entry['cell']} {entry['run_id']} step {entry['step']}: "
        f"truth={entry['observable_truth']:22s} jev={decision['diagnosis']:22s} "
        f"action={decision['action_name']:12s} p={decision['improvement_probability']} "
        f"{response['latency_s']:.2f}s"
    )

OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"\nwritten {len(lines) - 1} windows + spec -> {OUT}")
