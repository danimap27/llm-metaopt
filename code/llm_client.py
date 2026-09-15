"""Slow-loop client: the LLM as meta-optimizer.

Works with any OpenAI-compatible endpoint (local Ollama, vLLM, OpenRouter).
It records the per-call latency, which is the central quantity of the paper, and
requests structured output through a JSON schema. The system prompt is kept in
English because it is published verbatim in the paper.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import requests

from .regimes import REGIMES

DECISION_SCHEMA: Dict[str, Any] = {
    "name": "vqa_meta_control",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "diagnosis": {"type": "string", "enum": list(REGIMES)},
            "justification": {"type": "string", "maxLength": 600},
            "expected_effect": {"type": "number", "minimum": -5.0, "maximum": 5.0},
            "action": {
                "type": "object",
                "properties": {
                    "eta_scale": {"type": "number", "minimum": 0.1, "maximum": 10.0},
                    "noise_sigma": {"type": "number", "minimum": 0.0, "maximum": 0.5},
                    "restart": {"type": "boolean"},
                },
                "required": ["eta_scale", "noise_sigma", "restart"],
                "additionalProperties": False,
            },
        },
        "required": ["diagnosis", "justification", "action", "expected_effect"],
        "additionalProperties": False,
    },
}

SYSTEM_PROMPT = """You are the slow-loop meta-optimizer supervising SPSA on a variational quantum algorithm.

You receive a telemetry window of the last optimization steps and must decide one macroscopic intervention.

Telemetry fields:
- energy_series, energy (first, last, min, mean, std, slope_per_step) and improvement: progress of the objective.
- grad_norm (last, mean, max): norm of the SPSA gradient estimate.
- eta (a_k_last, c_k_last, eta_scale): learning-rate and perturbation schedules.
- theta (var, mean_abs, max_abs, dim): dispersion of the variational angles in radians.

Regimes:
- CONVERGENCIA_OK: the energy is at (or within tolerance of) the best reachable value.
- BARREN_PLATEAU: gradients are vanishingly small in every direction, so progress stalls far from the optimum.
- MINIMO_LOCAL: gradients are small and the window shows no improvement while the energy is still far from the optimum.
- MESETA_ENERGIA: no improvement in the window with a non-negligible gradient (noisy or rough landscape).
- CONCEPT_DRIFT: the objective itself changed (time-series case only).

Interventions (choose one action):
- eta_scale: multiplier applied to the SPSA step size (0.1 to 10).
- noise_sigma: standard deviation in radians of a Gaussian perturbation added to all angles (0 to 0.5); use it to break symmetries.
- restart: full re-initialization of the angles.
- expected_effect: the gap reduction you expect from your action over the next window, in energy units (negative if you expect a worsening).

Rules: answer with JSON only, follow the schema, keep justification under two sentences and ground it in the numbers you were given. Prefer the least invasive action that can restore progress."""


@dataclass
class LLMConfig:
    """Endpoint and sampling configuration (reproducibility)."""

    base_url: str = "http://localhost:11434/v1"
    model: str = "qwen3.5:4b"
    api_key: str = ""
    temperature: float = 0.0
    seed: int = 0
    max_tokens: int = 400
    timeout: float = 120.0
    use_json_schema: bool = True
    extra_body: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def build_messages(window: Dict[str, Any]) -> List[Dict[str, str]]:
    """Prompt messages: system prompt plus the serialized telemetry window."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(window, ensure_ascii=False, sort_keys=True)},
    ]


def build_request_payload(
    window: Dict[str, Any],
    cfg: LLMConfig,
    use_json_schema: Optional[bool] = None,
) -> Dict[str, Any]:
    """OpenAI-compatible payload (inspectable offline, used by the tests)."""
    schema = cfg.use_json_schema if use_json_schema is None else use_json_schema
    payload: Dict[str, Any] = {
        "model": cfg.model,
        "messages": build_messages(window),
        "temperature": cfg.temperature,
        "seed": cfg.seed,
        "max_tokens": cfg.max_tokens,
        "stream": False,
    }
    if schema:
        payload["response_format"] = {"type": "json_schema", "json_schema": DECISION_SCHEMA}
    payload.update(cfg.extra_body)
    return payload


def parse_decision(text: str) -> Dict[str, Any]:
    """Extract and validate the JSON decision from the model response."""
    if not text or not text.strip():
        raise ValueError("empty model response")
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?|```$", "", cleaned, flags=re.MULTILINE).strip()
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if match is None:
        raise ValueError(f"no JSON object found in the response: {text[:200]!r}")
    decision = json.loads(match.group(0))
    for key in ("diagnosis", "justification", "action", "expected_effect"):
        if key not in decision:
            raise ValueError(f"missing field {key!r} in the decision")
    action = decision["action"]
    for key in ("eta_scale", "noise_sigma", "restart"):
        if key not in action:
            raise ValueError(f"missing field {key!r} in action")
    if decision["diagnosis"] not in REGIMES:
        raise ValueError(f"diagnosis outside the allowed set: {decision['diagnosis']!r}")
    decision["action"]["eta_scale"] = float(action["eta_scale"])
    decision["action"]["noise_sigma"] = float(action["noise_sigma"])
    decision["action"]["restart"] = bool(action["restart"])
    decision["expected_effect"] = float(decision["expected_effect"])
    return decision


def decide(
    window: Dict[str, Any],
    cfg: LLMConfig,
    session: Optional[requests.Session] = None,
) -> Dict[str, Any]:
    """Query the LLM and return the decision plus its latency.

    If the server rejects ``response_format`` with a JSON schema, the call is
    retried without it. Latency is always measured around the POST with
    ``time.perf_counter``.
    """
    session = requests.Session() if session is None else session
    url = cfg.base_url.rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json"}
    if cfg.api_key:
        headers["Authorization"] = f"Bearer {cfg.api_key}"

    attempts: List[Dict[str, Any]] = []
    variants = [True, False] if cfg.use_json_schema else [False]
    for use_schema in variants:
        payload = build_request_payload(window, cfg, use_json_schema=use_schema)
        started = time.perf_counter()
        try:
            response = session.post(url, json=payload, headers=headers, timeout=cfg.timeout)
            latency = time.perf_counter() - started
            response.raise_for_status()
            data = response.json()
            raw = data["choices"][0]["message"]["content"]
            decision = parse_decision(raw)
            return {
                "ok": True,
                "model": cfg.model,
                "latency_s": latency,
                "decision": decision,
                "raw": raw,
                "usage": data.get("usage"),
                "used_json_schema": use_schema,
            }
        except Exception as exc:  # noqa: BLE001 - recorded and retried without schema
            attempts.append(
                {
                    "error": repr(exc),
                    "used_json_schema": use_schema,
                    "latency_s": time.perf_counter() - started,
                }
            )
    return {"ok": False, "model": cfg.model, "latency_s": None, "decision": None, "attempts": attempts}


def decide_cached(
    window: Dict[str, Any],
    cfg: LLMConfig,
    cache: Any,
    session: Optional[requests.Session] = None,
) -> Dict[str, Any]:
    """Cache-first decision: serve from disk when possible, else call and store.

    ``cache`` is duck typed (any object with ``lookup`` and ``store``), which
    keeps the replay path out of this module's import graph.
    """
    payload = build_request_payload(window, cfg)
    hit = cache.lookup(payload)
    if hit is not None:
        return {**hit, "cached": True}
    response = decide(window, cfg, session=session)
    cache.store(payload, response)
    return {**response, "cached": False}
