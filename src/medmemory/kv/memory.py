"""In-process KV store: a sorted dict with TTL. Test double and benchmark baseline.
Not durable: `snapshot()` writes a JSON dump that `MemoryKV.load()` can read back."""

from __future__ import annotations

import base64
import bisect
import json
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path

from medmemory.contracts.protocols import KVOp
from medmemory.errors import KVError


class MemoryKV:
    name = "memory"

    def __init__(self, clock: Callable[[], float] = time.time) -> None:
        self._data: dict[str, tuple[bytes, float | None]] = {}
        self._keys: list[str] = []  # sorted, for prefix scans
        self._lock = threading.RLock()
        self._clock = clock
        self._closed = False

    def _check(self) -> None:
        if self._closed:
            raise KVError("store is closed")

    def _live(self, key: str) -> bytes | None:
        item = self._data.get(key)
        if item is None:
            return None
        value, expires = item
        if expires is not None and expires <= self._clock():
            return None
        return value

    def get(self, key: str) -> bytes | None:
        with self._lock:
            self._check()
            return self._live(key)

    def _put(self, key: str, value: bytes, ttl_s: float | None) -> None:
        if key not in self._data:
            bisect.insort(self._keys, key)
        self._data[key] = (bytes(value), None if ttl_s is None else self._clock() + ttl_s)

    def _delete(self, key: str) -> bool:
        live = self._live(key) is not None
        if key in self._data:
            del self._data[key]
            self._keys.pop(bisect.bisect_left(self._keys, key))
        return live

    def put(self, key: str, value: bytes, ttl_s: float | None = None) -> None:
        with self._lock:
            self._check()
            self._put(key, value, ttl_s)

    def delete(self, key: str) -> bool:
        with self._lock:
            self._check()
            return self._delete(key)

    def scan(self, prefix: str, limit: int | None = None) -> Iterator[tuple[str, bytes]]:
        with self._lock:
            self._check()
            start = bisect.bisect_left(self._keys, prefix)
            out: list[tuple[str, bytes]] = []
            for key in self._keys[start:]:
                if not key.startswith(prefix):
                    break
                value = self._live(key)
                if value is not None:
                    out.append((key, value))
                    if limit is not None and len(out) >= limit:
                        break
        return iter(out)

    def batch(self, ops: Sequence[KVOp]) -> None:
        with self._lock:
            self._check()
            for op in ops:
                if op.op == "put":
                    if op.value is None:
                        raise KVError(f"put without value for {op.key}")
                    self._put(op.key, op.value, op.ttl_s)
                else:
                    self._delete(op.key)

    def count(self, prefix: str = "") -> int:
        return sum(1 for _ in self.scan(prefix))

    def purge_expired(self) -> int:
        with self._lock:
            now = self._clock()
            dead = [k for k, (_, exp) in self._data.items() if exp is not None and exp <= now]
            for k in dead:
                self._delete(k)
            return len(dead)

    def snapshot(self, dest: Path) -> Path:
        with self._lock:
            dump = {k: [base64.b64encode(v).decode(), exp] for k, (v, exp) in self._data.items()}
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(dump), encoding="utf-8")
        return dest

    @classmethod
    def load(cls, path: Path) -> MemoryKV:
        kv = cls()
        for k, (v, exp) in json.loads(path.read_text(encoding="utf-8")).items():
            kv._data[k] = (base64.b64decode(v), exp)
        kv._keys = sorted(kv._data)
        return kv

    def close(self) -> None:
        self._closed = True
