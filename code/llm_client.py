"""Slow-loop client: the LLM as meta-optimizer.

Two dialects are supported, because the same weights behave very differently
depending on the server API:

- ``api_style="openai"``: any OpenAI-compatible ``/v1/chat/completions`` endpoint.
- ``api_style="ollama"``: the native Ollama ``/api/chat`` endpoint, which is the
  only one that honours the ``think`` flag. Reasoning models routed through the
  OpenAI-compatible layer put the whole chain of thought into ``reasoning`` and
  return an empty ``content`` that finishes on the token limit, which costs tens
  of seconds and produces no decision at all. Measured on the reference machine:
  2.2 seconds with the native endpoint and thinking disabled against 60 to 90
  seconds without a usable answer through the OpenAI layer.

The client records the per-call latency, which is the central quantity of the
paper, and requests structured output through a JSON schema in both dialects.
The system prompt stays in English because it is published verbatim.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import json
import os
import re
import time

import requests

from .init_strategies import INIT_DESCRIPTIONS
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
                    "eta_scale": {"enum": [None, 0.5, 1.0, 2.0, 5.0]},
                    "noise_sigma": {"enum": [0.0, 0.05, 0.15]},
                    "restart": {"type": "boolean"},
                    "reheat": {"type": "boolean"},
                },
                "required": ["eta_scale", "noise_sigma", "restart", "reheat"],
                "additionalProperties": False,
            },
        },
        "required": ["diagnosis", "justification", "action", "expected_effect"],
        "additionalProperties": False,
    },
}

_REGIME_DESCRIPTIONS: Dict[str, str] = {
    # Observable taxonomy (static block): decidable from the window alone.
    "DESCENDING": "the energy is still decreasing meaningfully across the window",
    "STALLED_NO_GRADIENT": "the energy is flat and the gradient estimate is also near zero",
    "STALLED_WITH_GRADIENT": "the energy is flat but the gradient estimate is clearly non-zero",
    "OSCILLATING": "the energy moves up and down without net progress",
    # Distance-to-optimum taxonomy (kept for the drift block and the ceiling report).
    "CONVERGENCIA_OK": "the energy is at (or within tolerance of) the best reachable value",
    "BARREN_PLATEAU": "gradients are vanishingly small in every direction, so progress stalls far from the optimum",
    "MINIMO_LOCAL": "gradients are small and the window shows no improvement while the energy is still far from the optimum",
    "MESETA_ENERGIA": "no improvement in the window with a non-negligible gradient (noisy or rough landscape)",
    "CONCEPT_DRIFT": "the objective itself changed (time-series case only)",
}

_SYSTEM_TEMPLATE = """You are the slow-loop meta-optimizer supervising SPSA on a variational quantum algorithm.

You receive a telemetry window of the last optimization steps and must decide one macroscopic intervention.

Telemetry fields:
- energy_series, energy (first, last, min, mean, std, slope_per_step) and improvement: progress of the objective.
- grad_norm (last, mean, max): norm of the SPSA gradient estimate.
- eta (a_k_last, c_k_last, eta_scale): learning-rate and perturbation schedules.
- theta (var, mean_abs, max_abs, dim): dispersion of the variational angles in radians.
- progress (drop_from_start, gap_above_best_so_far): dimensionless progress of the run since the beginning.

Regimes (choose exactly one):
{regimes}

Interventions (choose one action):
- eta_scale: one of 0.5, 1.0, 2.0 or 5.0 as a multiplier on the SPSA step size, or null to keep the current multiplier unchanged.
- noise_sigma: one of 0.0, 0.05 or 0.15, the standard deviation in radians of a one-shot Gaussian perturbation added to all angles; use it to break symmetries.
- restart: full re-initialization of the angles.
- reheat: restart the SPSA decay schedules from their initial values while keeping the current angles; use it when the step size has decayed too far to exploit a plateau exit.
- The action space is the same grid given to the classical controllers.
- expected_effect: the gap reduction you expect from your action over the next window, in energy units (negative if you expect a worsening).

Rules: answer with JSON only, follow the schema, keep justification under two sentences and ground it in the numbers you were given. Prefer the least invasive action that can restore progress."""


def system_prompt(regimes: Sequence[str] = REGIMES) -> str:
    """System prompt restricted to the regimes a given block can encounter."""
    lines = "\n".join(f"- {name}: {_REGIME_DESCRIPTIONS[name]}." for name in regimes)
    return _SYSTEM_TEMPLATE.format(regimes=lines)


SYSTEM_PROMPT = system_prompt(REGIMES)


def decision_schema(regimes: Sequence[str] = REGIMES) -> Dict[str, Any]:
    """JSON schema restricted to the regimes a given block can encounter."""
    import copy

    schema = copy.deepcopy(DECISION_SCHEMA)
    schema["schema"]["properties"]["diagnosis"]["enum"] = list(regimes)
    return schema


@dataclass
class LLMConfig:
    """Endpoint and sampling configuration (reproducibility)."""

    base_url: str = "http://127.0.0.1:11434/v1"
    model: str = "qwen3.5:4b"
    api_key: str = ""
    api_style: str = "openai"  # "openai" or "ollama"
    temperature: float = 0.0
    seed: int = 0
    max_tokens: int = 400
    no_think: bool = False
    timeout: float = 120.0
    use_json_schema: bool = True
    extra_body: Dict[str, Any] = field(default_factory=dict)
    # Regimes offered to the model. Static blocks exclude CONCEPT_DRIFT so the
    # diagnosis is identifiable from the window (review finding M4); the drift
    # block enables the full enum.
    regimes: Sequence[str] = REGIMES

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def build_messages(window: Dict[str, Any], cfg: Optional[LLMConfig] = None) -> List[Dict[str, str]]:
    """Prompt messages: system prompt plus the serialized telemetry window.

    With ``cfg.no_think`` the user turn carries the ``/no_think`` hint that
    several open-weights families understand, which turns a classifier into a
    single forward pass instead of a long chain of thought.
    """
    content = json.dumps(window, ensure_ascii=False, sort_keys=True)
    if cfg is not None and cfg.no_think:
        content = content + " /no_think"
    prompt = system_prompt(cfg.regimes) if cfg is not None else SYSTEM_PROMPT
    return [
        {"role": "system", "content": prompt},
        {"role": "user", "content": content},
    ]


def endpoint_url(cfg: LLMConfig) -> str:
    """POST target for the configured dialect."""
    base = cfg.base_url.rstrip("/")
    if cfg.api_style == "ollama":
        if base.endswith("/v1"):
            base = base[: -len("/v1")]
        if base.endswith("/api"):
            return base + "/chat"
        return base + "/api/chat"
    return base + "/chat/completions"


def build_request_payload(
    window: Dict[str, Any],
    cfg: LLMConfig,
    use_json_schema: Optional[bool] = None,
) -> Dict[str, Any]:
    """Request payload for the configured dialect (inspectable offline)."""
    schema = cfg.use_json_schema if use_json_schema is None else use_json_schema

    if cfg.api_style == "ollama":
        payload: Dict[str, Any] = {
            "model": cfg.model,
            "messages": build_messages(window, cfg),
            "stream": False,
            "options": {
                "temperature": cfg.temperature,
                "seed": cfg.seed,
                "num_predict": cfg.max_tokens,
            },
        }
        if cfg.no_think:
            payload["think"] = False
        if schema is True:
            payload["format"] = decision_schema(cfg.regimes)["schema"]
        elif schema == "json_object":
            payload["format"] = "json"
        payload.update(cfg.extra_body)
        return payload

    payload = {
        "model": cfg.model,
        "messages": build_messages(window, cfg),
        "temperature": cfg.temperature,
        "seed": cfg.seed,
        "max_tokens": cfg.max_tokens,
        "stream": False,
    }
    if schema is True:
        payload["response_format"] = {"type": "json_schema", "json_schema": decision_schema(cfg.regimes)}
    elif schema == "json_object":
        payload["response_format"] = {"type": "json_object"}
    payload.update(cfg.extra_body)
    return payload


# --------------------------------------------------------------- systemone
# TypeSafe System One (Jev): typed questions answered with probabilities,
# instead of generated text. The decision contract is identical, expressed as
# a Choice over the regimes, a Choice over the action grid, a Noul for the
# improvement probability and a Score for the expected gain in energy units.

SYSTEMONE_ACTIONS: Dict[str, Dict[str, Any]] = {
    "noop": {"eta_scale": None, "noise_sigma": 0.0, "restart": False, "reheat": False},
    "slow_down": {"eta_scale": 0.5, "noise_sigma": 0.0, "restart": False, "reheat": False},
    "speed_up": {"eta_scale": 2.0, "noise_sigma": 0.0, "restart": False, "reheat": False},
    "big_speed_up": {"eta_scale": 5.0, "noise_sigma": 0.0, "restart": False, "reheat": False},
    "small_noise": {"eta_scale": None, "noise_sigma": 0.05, "restart": False, "reheat": False},
    "medium_noise": {"eta_scale": None, "noise_sigma": 0.15, "restart": False, "reheat": False},
    "speed_up_noise": {"eta_scale": 2.0, "noise_sigma": 0.05, "restart": False, "reheat": False},
    "reheat": {"eta_scale": None, "noise_sigma": 0.0, "restart": False, "reheat": True},
    "restart": {"eta_scale": None, "noise_sigma": 0.0, "restart": True, "reheat": False},
}

_SYSTEMONE_ACTION_DESCRIPTIONS: Dict[str, str] = {
    "noop": "leave everything unchanged",
    "slow_down": "halve the step size",
    "speed_up": "double the step size",
    "big_speed_up": "quintuple the step size",
    "small_noise": "add a small Gaussian perturbation of 0.05 radians to all angles",
    "medium_noise": "add a Gaussian perturbation of 0.15 radians to all angles",
    "speed_up_noise": "double the step size and add a small perturbation",
    "reheat": "restart the step-size schedules from their initial values, keeping the current angles",
    "restart": "reinitialize all angles",
}

# Anchors of the expected-gain Score question, in energy units. The score is
# continuous on the 0..4 index scale and is interpolated between anchors.
_GAIN_ANCHORS = (0.0, 0.1, 0.5, 1.0, 5.0)
_GAIN_CRITERIA = [
    "0 - no improvement",
    "0.1 - small improvement",
    "0.5 - moderate improvement",
    "1.0 - large improvement",
    "5.0 - huge improvement",
]


def build_systemone_payload(window: Dict[str, Any], cfg: LLMConfig) -> Dict[str, Any]:
    """System One request for the slow-loop decision over a telemetry window."""
    return {
        "state": json.dumps(window, ensure_ascii=False, sort_keys=True),
        "model": cfg.model,
        "questions": {
            "regime": {
                "type": "choice",
                "instructions": "Diagnose the optimization regime of this telemetry window",
                "criteria": {name: _REGIME_DESCRIPTIONS[name] for name in cfg.regimes},
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
            },
        },
    }


def _linear_gain(score: float) -> float:
    s = float(score)
    if s <= 0.0:
        return _GAIN_ANCHORS[0]
    if s >= len(_GAIN_ANCHORS) - 1:
        return _GAIN_ANCHORS[-1]
    index = int(s)
    fraction = s - index
    return _GAIN_ANCHORS[index] + fraction * (_GAIN_ANCHORS[index + 1] - _GAIN_ANCHORS[index])


def parse_systemone_answers(data: Dict[str, Any], cfg: LLMConfig) -> Dict[str, Any]:
    """Map a System One response onto the benchmark decision contract."""
    answers = data.get("answers") or {}
    regime = answers.get("regime") or {}
    diagnosis = regime.get("choice")
    if diagnosis not in cfg.regimes:
        raise ValueError(f"diagnosis outside the offered set: {diagnosis!r}")
    action_answer = answers.get("action") or {}
    action_name = action_answer.get("choice")
    if action_name not in SYSTEMONE_ACTIONS:
        raise ValueError(f"action outside the offered grid: {action_name!r}")
    improve = answers.get("improve_prob") or {}
    improve_prob = improve.get("noul")
    gain = answers.get("expected_gain") or {}
    score = gain.get("score")
    if score is None:
        raise ValueError("no score returned for expected_gain")
    regime_probs = regime.get("probabilities")
    action_probs = action_answer.get("probabilities")

    def _prob(probs: Any, key: str) -> Optional[float]:
        return float(probs[key]) if isinstance(probs, dict) and key in probs else None

    justification = (
        f"systemone: P(regime)={_prob(regime_probs, diagnosis)}, "
        f"P(action)={_prob(action_probs, action_name)}, P(improve)={improve_prob}"
    )
    return {
        "diagnosis": diagnosis,
        "justification": justification,
        "expected_effect": float(_linear_gain(score)),
        "action": dict(SYSTEMONE_ACTIONS[action_name]),
        "action_name": action_name,
        "diagnosis_probabilities": regime_probs,
        "action_probabilities": action_probs,
        "improvement_probability": improve_prob,
        "gain_score": float(score),
    }


def build_systemone_init_payload(problem: Dict[str, Any], cfg: LLMConfig) -> Dict[str, Any]:
    """Warm-start request: pick an initialization strategy for the problem."""
    return {
        "state": json.dumps(problem, ensure_ascii=False, sort_keys=True),
        "model": cfg.model,
        "questions": {
            "init_strategy": {
                "type": "choice",
                "instructions": (
                    "Choose the initialization strategy for the variational angles "
                    "that is most likely to reach a low final energy for this problem"
                ),
                "criteria": dict(INIT_DESCRIPTIONS),
            },
            "converges": {
                "type": "noul",
                "instructions": "This run will finish close to the best value this circuit family can reach",
            },
        },
    }


def decide_init(
    problem: Dict[str, Any],
    cfg: LLMConfig,
    session: Optional[requests.Session] = None,
) -> Dict[str, Any]:
    """One System One call for the initialization choice (Engine B).

    Returns ``{"ok": True, "strategy": ..., "probabilities": ..., "latency_s"}``
    or the shared failure shape with ``ok: False``.
    """
    session = requests.Session() if session is None else session
    url = cfg.base_url if cfg.base_url and "typesafe" in cfg.base_url else "https://api.typesafe.ai/v1/systemone"
    key = cfg.api_key or os.environ.get("TYPESAFE_API_KEY", "")
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    payload = build_systemone_init_payload(problem, cfg)
    started = time.perf_counter()
    try:
        response = session.post(url, json=payload, headers=headers, timeout=cfg.timeout)
        latency = time.perf_counter() - started
        response.raise_for_status()
        data = response.json()
        answers = data.get("answers") or {}
        strategy_answer = answers.get("init_strategy") or {}
        strategy = strategy_answer.get("choice")
        if strategy not in INIT_DESCRIPTIONS:
            raise ValueError(f"init strategy outside the offered set: {strategy!r}")
        converges = (answers.get("converges") or {}).get("noul")
        return {
            "ok": True,
            "model": data.get("model", cfg.model),
            "strategy": strategy,
            "probabilities": strategy_answer.get("probabilities"),
            "converges_prob": converges,
            "latency_s": time.perf_counter() - started,
            "latency_successful_s": latency,
        }
    except Exception as exc:  # noqa: BLE001 - shared failure shape
        return {
            "ok": False,
            "attempts": [
                {
                    "error": repr(exc),
                    "latency_s": time.perf_counter() - started,
                    "used_json_schema": True,
                    "raw_snippet": "",
                }
            ],
        }


def _decide_systemone(
    window: Dict[str, Any],
    cfg: LLMConfig,
    session: Optional[requests.Session],
) -> Dict[str, Any]:
    session = requests.Session() if session is None else session
    url = cfg.base_url if cfg.base_url and "typesafe" in cfg.base_url else "https://api.typesafe.ai/v1/systemone"
    key = cfg.api_key or os.environ.get("TYPESAFE_API_KEY", "")
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    payload = build_systemone_payload(window, cfg)
    started = time.perf_counter()
    try:
        response = session.post(url, json=payload, headers=headers, timeout=cfg.timeout)
        latency = time.perf_counter() - started
        response.raise_for_status()
        data = response.json()
        decision = parse_systemone_answers(data, cfg)
        return {
            "ok": True,
            "model": data.get("model", cfg.model),
            "api_style": "systemone",
            "latency_s": time.perf_counter() - started,
            "latency_successful_s": latency,
            "n_attempts": 1,
            "decision": decision,
            "raw": json.dumps(data, ensure_ascii=False)[:2000],
            "usage": data.get("usage"),
            "used_json_schema": True,
        }
    except Exception as exc:  # noqa: BLE001 - failure shape matches the JSON dialects
        return {
            "ok": False,
            "model": cfg.model,
            "api_style": "systemone",
            "attempts": [
                {
                    "error": repr(exc),
                    "latency_s": time.perf_counter() - started,
                    "used_json_schema": True,
                    "raw_snippet": "",
                }
            ],
        }


def extract_content(data: Dict[str, Any], cfg: LLMConfig) -> tuple[str, str, Optional[Dict[str, Any]]]:
    """Return ``(content, reasoning, usage)`` from either response shape."""
    if cfg.api_style == "ollama":
        message = data.get("message", {}) or {}
        content = message.get("content") or ""
        reasoning = message.get("thinking") or ""
        usage = {
            "prompt_tokens": data.get("prompt_eval_count"),
            "completion_tokens": data.get("eval_count"),
        }
    else:
        message = (data.get("choices") or [{}])[0].get("message", {}) or {}
        content = message.get("content") or ""
        reasoning = message.get("reasoning") or message.get("reasoning_content") or ""
        usage = data.get("usage")
    return content, reasoning, usage


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
    for key in ("diagnosis", "justification", "expected_effect", "action"):
        if key not in decision:
            raise ValueError(
                f"missing field {key!r} in the decision. Keys received: {sorted(decision)[:8]}"
            )
    action = decision["action"]
    for key in ("eta_scale", "noise_sigma", "restart"):
        if key not in action:
            raise ValueError(f"missing field {key!r} in action")
    if decision["diagnosis"] not in REGIMES:
        raise ValueError(f"diagnosis outside the allowed set: {decision['diagnosis']!r}")
    if action["eta_scale"] is not None and action["eta_scale"] not in (0.5, 1.0, 2.0, 5.0):
        raise ValueError(f"eta_scale {action['eta_scale']!r} is outside the action grid")
    if action["noise_sigma"] not in (0.0, 0.05, 0.15):
        raise ValueError(f"noise_sigma {action['noise_sigma']!r} is outside the action grid")
    decision["action"]["reheat"] = bool(action.get("reheat", False))
    decision["action"]["eta_scale"] = None if action["eta_scale"] is None else float(action["eta_scale"])
    decision["action"]["noise_sigma"] = float(action["noise_sigma"])
    decision["action"]["restart"] = bool(action["restart"])
    raw_effect = decision.get("expected_effect")
    decision["expected_effect"] = float(raw_effect) if isinstance(raw_effect, (int, float)) else None
    return decision


def decide(
    window: Dict[str, Any],
    cfg: LLMConfig,
    session: Optional[requests.Session] = None,
) -> Dict[str, Any]:
    """Query the endpoint and return the decision plus its latency.

    Three variants are attempted in order: the strict JSON schema, a plain JSON
    mode and no structured output at all, so a weaker server still yields a
    parsable answer. Latency is always measured around the POST. The
    ``systemone`` style is a single typed-question request instead.
    """
    if cfg.api_style == "systemone":
        return _decide_systemone(window, cfg, session)
    session = requests.Session() if session is None else session
    url = endpoint_url(cfg)
    headers = {"Content-Type": "application/json"}
    if cfg.api_key:
        headers["Authorization"] = f"Bearer {cfg.api_key}"

    attempts: List[Dict[str, Any]] = []
    variants = [True, "json_object", False] if cfg.use_json_schema else [False]
    for use_schema in variants:
        payload = build_request_payload(window, cfg, use_json_schema=use_schema)
        started = time.perf_counter()
        raw = ""
        try:
            response = session.post(url, json=payload, headers=headers, timeout=cfg.timeout)
            latency = time.perf_counter() - started
            response.raise_for_status()
            data = response.json()
            raw, reasoning, usage = extract_content(data, cfg)
            if not raw.strip() and reasoning:
                raise ValueError(
                    "empty content: the endpoint returned only reasoning. "
                    "Use api_style=ollama with no_think, or raise max_tokens."
                )
            decision = parse_decision(raw)
            return {
                "ok": True,
                "model": cfg.model,
                "api_style": cfg.api_style,
                "latency_s": time.perf_counter() - started,
                "latency_successful_s": latency,
                "n_attempts": len(attempts) + 1,
                "decision": decision,
                "raw": raw,
                "usage": usage,
                "used_json_schema": use_schema,
            }
        except Exception as exc:  # noqa: BLE001 - recorded and retried without schema
            attempts.append(
                {
                    "error": repr(exc),
                    "used_json_schema": use_schema,
                    "latency_s": time.perf_counter() - started,
                    "raw_snippet": raw[:400],
                }
            )
    return {
        "ok": False,
        "model": cfg.model,
        "api_style": cfg.api_style,
        "latency_s": None,
        "decision": None,
        "attempts": attempts,
    }


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
