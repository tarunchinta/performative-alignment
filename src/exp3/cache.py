"""Response cache / replay layer for Experiment 3 judge queries.

Keyed on ``sha256(model_id, rendered_prompt, repeat_index)``. Including the
repeat index preserves independent samples at temperature > 0 while making the
whole run replayable: the full analysis reruns from the logged responses without
re-querying, and the JSONL log ships as supplementary material.

The shipped copy is gzipped -- uncompressed it is ~524 MB across 247,800
records, over GitHub's per-file limit. Loading falls back to ``<path>.gz`` when
the plain JSONL is absent, so a fresh clone replays with no manual decompression
step; new responses are always appended to the uncompressed path.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import threading
from pathlib import Path


def cache_key(model_id: str, prompt: str, repeat_index: int) -> str:
    h = hashlib.sha256()
    h.update(model_id.encode("utf-8"))
    h.update(b"\x00")
    h.update(prompt.encode("utf-8"))
    h.update(b"\x00")
    h.update(str(repeat_index).encode("utf-8"))
    return h.hexdigest()


class ResponseCache:
    """In-memory cache backed by an append-only JSONL log.

    Existing logs are loaded on construction so a rerun replays prior responses.
    """

    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path) if path is not None else None
        self._store: dict[str, str] = {}
        self._lock = threading.Lock()
        if self.path is not None and self._source() is not None:
            self._load()

    def _source(self) -> Path | None:
        """Log to replay from: the plain JSONL, else the shipped ``.gz``."""
        assert self.path is not None
        if self.path.exists():
            return self.path
        packed = self.path.with_name(self.path.name + ".gz")
        return packed if packed.exists() else None

    def _load(self) -> None:
        src = self._source()
        assert src is not None
        opener = gzip.open if src.suffix == ".gz" else open
        with opener(src, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                self._store[rec["key"]] = rec["response"]

    def get(self, key: str) -> str | None:
        return self._store.get(key)

    def put(
        self,
        key: str,
        response: str,
        *,
        model_id: str,
        repeat_index: int,
        prompt: str,
    ) -> None:
        with self._lock:
            if key in self._store:
                return
            self._store[key] = response
            if self.path is not None:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                rec = {
                    "key": key,
                    "model_id": model_id,
                    "repeat_index": repeat_index,
                    "prompt": prompt,
                    "response": response,
                }
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def __len__(self) -> int:
        return len(self._store)
