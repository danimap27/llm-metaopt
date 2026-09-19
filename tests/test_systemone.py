"""Tests for the 'systemone' dialect (TypeSafe Jev) against a mocked endpoint."""

from __future__ import annotations

import json

from code.llm_client import (
    SYSTEMONE_ACTIONS,
    LLMConfig,
    _linear_gain,
    build_systemone_payload,
    decide,
    parse_systemone_answers,
)

WINDOW = {
    "n_window": 10,
    "energy_series": [0.5, 0.48, 0.47, 0.46, 0.46, 0.45, 0.45, 0.45, 0.45, 0.45],
    "improvement": 0.05,
    "grad_norm": {"last": 0.02, "mean": 0.1, "max": 0.3},
    "eta": {"a_k_last": 0.04, "c_k_last": 0.08, "eta_scale": 1.0},
    "theta": {"var": 1.0, "mean_abs": 0.9, "max_abs": 1.5, "dim": 12},
    "progress": {"drop_from_start": 0.1, "gap_above_best_so_far": 0.0},
}


def _answers() -> dict:
    return {
        "model": "jev-test",
        "answers": {
            "regime": {
                "type": "choice",
                "choice": "MESETA_ENERGIA",
                "probabilities": {"CONVERGENCIA_OK": 0.1, "BARREN_PLATEAU": 0.05, "MINIMO_LOCAL": 0.1, "MESETA_ENERGIA": 0.75},
                "confidence": 0.6,
            },
            "action": {
                "type": "choice",
                "choice": "slow_down",
                "probabilities": {"noop": 0.2, "slow_down": 0.6, "restart": 0.2},
                "confidence": 0.5,
            },
            "improve_prob": {"type": "noul", "noul": 0.8},
            "expected_gain": {"type": "score", "score": 2.5, "legend": {"0": "0", "1": "0.1", "2": "0.5", "3": "1.0", "4": "5.0"}},
        },
        "usage": {"input_tokens": 900, "output_tokens": 160},
    }


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


class _FakeSession:
    def __init__(self, payload: dict | None = None, error: Exception | None = None) -> None:
        self.payload = payload
        self.error = error
        self.last_request: dict | None = None

    def post(self, url, json=None, headers=None, timeout=None):  # noqa: A002, ANN001
        self.last_request = {"url": url, "json": json, "headers": headers, "timeout": timeout}
        if self.error is not None:
            raise self.error
        return _FakeResponse(self.payload or {})


def test_payload_carries_the_question_contract() -> None:
    cfg = LLMConfig(api_style="systemone", regimes=("CONVERGENCIA_OK", "BARREN_PLATEAU", "MINIMO_LOCAL", "MESETA_ENERGIA"))
    payload = build_systemone_payload(WINDOW, cfg)
    assert json.loads(payload["state"])["improvement"] == 0.05
    questions = payload["questions"]
    assert set(questions["regime"]["criteria"]) == set(cfg.regimes)
    assert set(questions["action"]["criteria"]) == set(SYSTEMONE_ACTIONS)
    assert questions["improve_prob"]["type"] == "noul"
    assert questions["expected_gain"]["type"] == "score"


def test_answers_map_onto_the_decision_contract() -> None:
    cfg = LLMConfig(api_style="systemone")
    decision = parse_systemone_answers(_answers(), cfg)
    assert decision["diagnosis"] == "MESETA_ENERGIA"
    assert decision["action"] == {"eta_scale": 0.5, "noise_sigma": 0.0, "restart": False, "reheat": False}
    assert decision["expected_effect"] == 0.75  # 2.5 interpolates between 0.5 and 1.0
    assert decision["improvement_probability"] == 0.8
    assert decision["action_name"] == "slow_down"
    assert "P(improve)=0.8" in decision["justification"]


def test_decide_systemone_end_to_end_with_a_mocked_session() -> None:
    cfg = LLMConfig(
        api_style="systemone",
        base_url="https://api.typesafe.ai/v1/systemone",
        model="jev-test",
        api_key="test-key",
    )
    session = _FakeSession(payload=_answers())
    result = decide(WINDOW, cfg, session=session)
    assert result["ok"] is True
    assert result["decision"]["action_name"] == "slow_down"
    assert result["latency_s"] >= 0.0
    assert session.last_request is not None
    assert session.last_request["headers"]["Authorization"] == "Bearer test-key"
    assert session.last_request["url"].endswith("/v1/systemone")


def test_decide_systemone_reports_failures_in_the_shared_shape() -> None:
    cfg = LLMConfig(api_style="systemone", api_key="k")
    session = _FakeSession(error=RuntimeError("api down"))
    result = decide(WINDOW, cfg, session=session)
    assert result["ok"] is False
    assert "api down" in result["attempts"][0]["error"]
    assert result["attempts"][0]["latency_s"] >= 0.0


def test_linear_gain_interpolates_and_clamps() -> None:
    assert _linear_gain(0.0) == 0.0
    assert _linear_gain(1.0) == 0.1
    assert _linear_gain(2.5) == 0.75
    assert _linear_gain(10.0) == 5.0
