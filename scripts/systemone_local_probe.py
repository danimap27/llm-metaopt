"""Local probe: the System One pattern backed by local open weights.

Runs one Choice question (regime diagnosis) and one Noul question (will the
gap improve) against a real telemetry window, through the official MIT
adapter pointed at an OpenAI-compatible endpoint (Ollama by default).

Needs the optional extra in a throwaway venv, not the project environment:

    uv venv /tmp/s1-venv
    uv pip install --python /tmp/s1-venv/bin/python 'system-one-adapter[openai]'
    /tmp/s1-venv/bin/python scripts/systemone_local_probe.py

Verified 2026-09-19 against llama3.2:3b on the homelab: valid answer on the
first attempt, 11.3 s on CPU. See docs/system-one-assessment.md.
"""

from __future__ import annotations

import argparse
import json
import time

from system_one_adapter import Choice, Noul, SystemOneAdapterClient
from system_one_adapter.providers.openai import OpenAIProvider


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:11434/v1")
    parser.add_argument("--model", default="llama3.2:3b")
    parser.add_argument("--api", default="chat_completions", choices=["chat_completions", "responses"])
    parser.add_argument(
        "--compact",
        action="store_true",
        help="short keys and terse instructions; reduces output tokens (the decode-dominant cost on CPU)",
    )
    args = parser.parse_args()

    state = json.dumps(
        {
            "energy_series": [0.58, 0.55, 0.52, 0.50, 0.49, 0.49, 0.48, 0.48, 0.48, 0.48],
            "improvement": 0.001,
            "grad_norm_last": 0.0002,
            "grad_norm_mean": 0.004,
            "energy_slope_per_step": -0.0005,
            "progress": {"drop_from_start": 0.17, "gap_above_best_so_far": 0.0},
        }
    )

    if args.compact:
        regime_question = Choice(
            instructions="Diagnose the regime",
            criteria={
                "CONV": "at the best value",
                "BP": "gradients vanish",
                "LOCAL": "stalled, small gradient",
                "ROUGH": "stalled, non-small gradient",
            },
        )
        questions = {
            "regime": regime_question,
            "improves": Noul(instructions="Gap improves next window"),
        }
    else:
        regime_question = Choice(
            instructions="Diagnose the optimization regime of this telemetry window",
            criteria={
                "CONVERGENCIA_OK": "energy at the best reachable value",
                "BARREN_PLATEAU": "gradients vanish in every direction far from the optimum",
                "MINIMO_LOCAL": "no improvement with small gradient away from the optimum",
                "MESETA_ENERGIA": "no improvement with a non-negligible gradient",
            },
        )
        questions = {
            "regime": regime_question,
            "gap_improves": Noul(
                instructions="The energy gap will improve over the next window of optimization steps",
            ),
        }

    client = SystemOneAdapterClient(
        structured_outputs=True,
        llm_answer_mode="probabilities",
        normalize_probabilities=True,
        n_retry_malformed_structure=2,
    )
    provider = OpenAIProvider(
        args.model,
        base_url=args.base_url,
        api_key="local",  # required by the OpenAI SDK; local servers ignore it
        api=args.api,
    )

    started = time.perf_counter()
    response = client.system_one(
        state=state,
        questions=questions,
        model=provider,
    )
    wall = time.perf_counter() - started

    usage = response.usage.model_dump() if hasattr(response.usage, "model_dump") else dict(response.usage)
    print(f"wall clock: {wall:.2f} s")
    print(f"usage: {json.dumps(usage)}")
    print(f"regime: {response.answers['regime'].choice}")
    print(f"regime probabilities: {json.dumps(response.answers['regime'].probabilities)}")
    print(f"gap_improves (noul): {response.answers['gap_improves'].noul}")
    print(f"attempts: {len(response.debug.get('llm_attempts', []))}")
    client.close()


if __name__ == "__main__":
    main()
