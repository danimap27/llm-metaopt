"""Append-only cache of language-model calls, with a replay mode.

Reviewers ask whether an experiment can be reproduced without the commercial
endpoint. The cache stores the request key and the full response for every call,
so a run can be replayed from disk. Failed calls are cached too, because the
failure rate is part of the reported cost.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
from typing import Any, Dict, Optional


def cache_key(payload: Dict[str, Any]) -> str:
    """Stable hash of a request payload."""
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class ResponseCache:
    """JSONL cache with append-on-write and lookup-on-read."""

    def __init__(self, path: str | pathlib.Path, replay_only: bool = False) -> None:
        self.path = pathlib.Path(path)
        self.replay_only = replay_only
        self._entries: Dict[str, Dict[str, Any]] = {}
        if self.path.exists():
            with self.path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    self._entries[str(record["key"])] = record["response"]

    def __len__(self) -> int:
        return len(self._entries)

    def lookup(self, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Response stored for this payload, or ``None``."""
        return self._entries.get(cache_key(payload))

    def store(self, payload: Dict[str, Any], response: Dict[str, Any]) -> None:
        """Append a response, ignoring a key that is already cached."""
        if self.replay_only:
            raise RuntimeError("cache is in replay-only mode")
        key = cache_key(payload)
        if key in self._entries:
            return
        self._entries[key] = response
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps({"key": key, "payload": payload, "response": response}, ensure_ascii=False) + "\n"
            )
