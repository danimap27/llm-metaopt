"""Tests for the language-model response cache and its replay mode."""

from __future__ import annotations

import json

import pytest

from code.llm_cache import ResponseCache, cache_key


def test_cache_key_is_stable_and_payload_sensitive():
    first = {"model": "m", "messages": [{"role": "user", "content": "a"}]}
    second = {"model": "m", "messages": [{"role": "user", "content": "b"}]}
    assert cache_key(first) == cache_key(dict(first))
    assert cache_key(first) != cache_key(second)


def test_cache_roundtrip_and_persistence(tmp_path):
    cache = ResponseCache(tmp_path / "cache.jsonl")
    payload = {"model": "m", "messages": [{"role": "user", "content": "a"}]}
    response = {"ok": True, "decision": {"diagnosis": "BARREN_PLATEAU"}}
    assert cache.lookup(payload) is None
    cache.store(payload, response)
    assert cache.lookup(payload) == response
    assert cache.lookup({"model": "m", "messages": [{"role": "user", "content": "z"}]}) is None

    reloaded = ResponseCache(tmp_path / "cache.jsonl")
    assert reloaded.lookup(payload) == response
    record = json.loads((tmp_path / "cache.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert record["key"] == cache_key(payload)
    assert record["response"] == response


def test_cache_does_not_write_twice(tmp_path):
    cache = ResponseCache(tmp_path / "cache.jsonl")
    payload = {"model": "m", "messages": []}
    cache.store(payload, {"ok": True})
    cache.store(payload, {"ok": False})
    assert cache.lookup(payload) == {"ok": True}
    assert len((tmp_path / "cache.jsonl").read_text(encoding="utf-8").strip().splitlines()) == 1


def test_replay_only_mode_refuses_to_store(tmp_path):
    cache = ResponseCache(tmp_path / "cache.jsonl", replay_only=True)
    with pytest.raises(RuntimeError):
        cache.store({"model": "m"}, {"ok": True})


def test_decide_cached_serves_from_disk_without_calling_the_endpoint(tmp_path):
    from code.llm_client import LLMConfig, build_request_payload, decide_cached

    cfg = LLMConfig(base_url="http://unreachable.invalid/v1", model="m")
    cache = ResponseCache(tmp_path / "cache.jsonl")
    window = {"n_window": 10, "improvement": 0.0, "grad_norm": {"last": 1e-5}}
    cache.store(
        build_request_payload(window, cfg),
        {"ok": True, "latency_s": 1.0, "decision": {"diagnosis": "BARREN_PLATEAU"}},
    )
    response = decide_cached(window, cfg, cache)
    assert response["cached"] is True
    assert response["decision"]["diagnosis"] == "BARREN_PLATEAU"
    assert len(cache) == 1
