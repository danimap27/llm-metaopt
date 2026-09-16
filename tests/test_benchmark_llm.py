"""Tests for the endpoint latency benchmark."""

from __future__ import annotations

import json
import pathlib

from scripts.benchmark_llm import measure, percentile, to_markdown

DECISION = {
    "diagnosis": "MESETA_ENERGIA",
    "justification": "no improvement in the window",
    "expected_effect": 0.02,
    "action": {"eta_scale": 1.0, "noise_sigma": 0.0, "restart": False},
}


class _Response:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


class _Session:
    """Fake OpenAI-compatible endpoint that counts calls."""

    def __init__(self, fail_times: int = 0) -> None:
        self.fail_times = fail_times
        self.calls = 0

    def post(self, url, json=None, headers=None, timeout=None):  # noqa: A002 - requests signature
        self.calls += 1
        if self.calls <= self.fail_times:
            raise RuntimeError("endpoint unavailable")
        return _Response(
            {
                "choices": [{"message": {"content": __import__("json").dumps(DECISION)}}],
                "usage": {"completion_tokens": 40},
            }
        )


def test_percentile_interpolates_and_bounds():
    values = [1.0, 2.0, 3.0, 4.0]
    assert percentile(values, 0.0) == 1.0
    assert percentile(values, 1.0) == 4.0
    assert percentile(values, 0.5) == 2.5
    assert percentile([7.0], 0.9) == 7.0


def test_measure_reports_the_distribution():
    session = _Session()
    report = measure("http://localhost:9/v1", "fake-model", calls=5, session=session)
    assert report["ok"] is True
    assert report["calls"] == 5
    assert report["failures"] == 0
    assert report["latency_s"]["min"] <= report["latency_s"]["median"] <= report["latency_s"]["p95"]
    assert report["completion_tokens_mean"] == 40.0
    assert "tokens_per_second" not in report or report["tokens_per_s_mean"] > 0


def test_measure_counts_failures_and_survives_a_dead_endpoint():
    report = measure("http://localhost:9/v1", "dead", calls=3, session=_Session(fail_times=5))
    assert report["ok"] is False
    assert report["failures"] == 3


def test_markdown_renders_both_healthy_and_dead_reports():
    healthy = to_markdown(
        {"ok": True, "model": "m", "base_url": "u", "calls": 2, "failures": 0,
         "latency_s": {"mean": 1.0, "median": 1.0, "p95": 1.2, "min": 0.9, "max": 1.2}}
    )
    dead = to_markdown({"ok": False, "model": "m", "calls": 3})
    assert "Mean latency" in healthy
    assert "every call failed" in dead
    assert json.dumps({"a": 1})  # the module imports stay JSON friendly


def test_main_writes_json_and_markdown(tmp_path: pathlib.Path, monkeypatch):
    from scripts import benchmark_llm

    monkeypatch.setattr(
        benchmark_llm,
        "measure",
        lambda *args, **kwargs: {
            "ok": True, "model": "m", "base_url": "u", "calls": 1, "failures": 0,
            "latency_s": {"mean": 1.0, "median": 1.0, "p95": 1.0, "min": 1.0, "max": 1.0},
        },
    )
    out = tmp_path / "latency.json"
    assert benchmark_llm.main(["--out", str(out)]) == 0
    assert json.loads(out.read_text(encoding="utf-8"))["ok"] is True
    assert out.with_suffix(".md").exists()
