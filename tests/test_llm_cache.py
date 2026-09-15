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
