"""Tests for the two endpoint dialects the slow-loop client speaks."""

from __future__ import annotations

import json

from code.llm_client import (
    LLMConfig,
    build_request_payload,
    decide,
    endpoint_url,
    extract_content,
)

OLLAMA_RESPONSE = {
    "model": "qwen3.5:4b",
    "message": {
        "role": "assistant",
        "content": json.dumps(
            {
                "diagnosis": "MESETA_ENERGIA",
                "justification": "flat window",
                "expected_effect": 0.02,
                "action": {"eta_scale": 1.0, "noise_sigma": 0.0, "restart": False},
            }
        ),
    },
    "done": True,
    "prompt_eval_count": 120,
    "eval_count": 45,
}


def test_endpoint_url_for_each_dialect():
    openai = LLMConfig(base_url="http://127.0.0.1:8080/v1", model="m")
    ollama = LLMConfig(base_url="http://127.0.0.1:11434", model="m", api_style="ollama")
    ollama_v1 = LLMConfig(base_url="http://127.0.0.1:11434/v1", model="m", api_style="ollama")
    assert endpoint_url(openai) == "http://127.0.0.1:8080/v1/chat/completions"
    assert endpoint_url(ollama) == "http://127.0.0.1:11434/api/chat"
    assert endpoint_url(ollama_v1) == "http://127.0.0.1:11434/api/chat"


def test_ollama_payload_carries_think_and_the_schema():
    cfg = LLMConfig(
        base_url="http://127.0.0.1:11434",
        model="qwen3.5:4b",
        api_style="ollama",
        no_think=True,
        max_tokens=200,
    )
    payload = build_request_payload({"n_window": 10}, cfg)
    assert payload["think"] is False
    assert payload["stream"] is False
    assert payload["options"]["num_predict"] == 200
    assert payload["format"]["type"] == "object"
    assert "response_format" not in payload


def test_openai_payload_has_no_ollama_fields():
    cfg = LLMConfig(base_url="http://127.0.0.1:8080/v1", model="m")
    payload = build_request_payload({"n_window": 10}, cfg)
    assert payload["response_format"]["type"] == "json_schema"
    assert "think" not in payload
    assert "options" not in payload


def test_extract_content_reads_both_shapes():
    ollama_cfg = LLMConfig(base_url="http://x", model="m", api_style="ollama")
    content, _, usage = extract_content(OLLAMA_RESPONSE, ollama_cfg)
    assert "MESETA_ENERGIA" in content
    assert usage is not None and usage["completion_tokens"] == 45

    openai_cfg = LLMConfig(base_url="http://x/v1", model="m")
    content, reasoning, usage = extract_content(
        {
            "choices": [{"message": {"content": "{}", "reasoning": "why"}}],
            "usage": {"completion_tokens": 3},
        },
        openai_cfg,
    )
    assert content == "{}"
    assert reasoning == "why"
    assert usage is not None and usage["completion_tokens"] == 3


class _Response:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


class _Session:
    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.urls: list[str] = []

    def post(self, url, json=None, headers=None, timeout=None):  # noqa: A002 - requests signature
        self.urls.append(url)
        return _Response(self.payload)


def test_decide_speaks_ollama_and_records_usage():
    cfg = LLMConfig(base_url="http://127.0.0.1:11434", model="qwen3.5:4b", api_style="ollama", no_think=True)
    session = _Session(OLLAMA_RESPONSE)
    result = decide({"n_window": 10}, cfg, session=session)
    assert result["ok"] is True
    assert result["decision"]["diagnosis"] == "MESETA_ENERGIA"
    assert result["api_style"] == "ollama"
    assert session.urls[0].endswith("/api/chat")
    assert result["latency_s"] >= 0.0
    assert result["usage"]["completion_tokens"] == 45


def test_decide_reports_reasoning_only_responses():
    cfg = LLMConfig(
        base_url="http://127.0.0.1:11434",
        model="m",
        api_style="ollama",
        use_json_schema=False,
    )
    session = _Session({"message": {"content": "", "thinking": "long thoughts"}, "eval_count": 10})
    result = decide({"n_window": 10}, cfg, session=session)
    assert result["ok"] is False
    assert "only reasoning" in result["attempts"][0]["error"]
