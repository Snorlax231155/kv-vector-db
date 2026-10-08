"""Thread-safe LRU cache with per-entry TTL, tag-based invalidation and metrics.

* O(1) get/set via OrderedDict (move_to_end on hit, popitem(last=False) to evict).
* TTL is checked lazily on read; expired entries count as misses + expirations.
* Tags (e.g. 'patient:P0001') map to keys so a KV write can drop every dependent answer
  in one call. That is the write-through invalidation path.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Generic, TypeVar

from medmemory.contracts.protocols import CacheStats

V = TypeVar("V")


@dataclass(slots=True)
class _Entry(Generic[V]):
    value: V
    expires_at: float | None
    tags: tuple[str, ...]


class LRUCache(Generic[V]):
    def __init__(
        self,
        capacity: int,
        default_ttl_s: float | None = None,
        name: str = "lru",
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.name = name
        self.capacity = capacity
        self.default_ttl_s = default_ttl_s
        self._clock = clock
        self._data: OrderedDict[str, _Entry[V]] = OrderedDict()
        self._tags: dict[str, set[str]] = {}
        self._lock = threading.Lock()
        self._stats = CacheStats(name=name, size=0, capacity=capacity)

    def _drop(self, key: str) -> None:
        entry = self._data.pop(key)
        for tag in entry.tags:
            keys = self._tags.get(tag)
            if keys is not None:
                keys.discard(key)
                if not keys:
                    del self._tags[tag]

    def get(self, key: str) -> V | None:
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                self._stats.misses += 1
                return None
            if entry.expires_at is not None and entry.expires_at <= self._clock():
                self._drop(key)
                self._stats.expirations += 1
                self._stats.misses += 1
                return None
            self._data.move_to_end(key)
            self._stats.hits += 1
            return entry.value

    def peek(self, key: str) -> V | None:
        """Read without touching recency or stats (for the inspector UI)."""
        with self._lock:
            entry = self._data.get(key)
            return None if entry is None else entry.value

    def set(self, key: str, value: V, ttl_s: float | None = None, tags: Iterable[str] = ()) -> None:
        ttl = self.default_ttl_s if ttl_s is None else ttl_s
        expires = None if ttl is None else self._clock() + ttl
        tag_tuple = tuple(dict.fromkeys(tags))
        with self._lock:
            if key in self._data:
                self._drop(key)
            self._data[key] = _Entry(value, expires, tag_tuple)
            for tag in tag_tuple:
                self._tags.setdefault(tag, set()).add(key)
            while len(self._data) > self.capacity:
                oldest = next(iter(self._data))
                self._drop(oldest)
                self._stats.evictions += 1

    def invalidate(self, key: str) -> bool:
        with self._lock:
            if key not in self._data:
                return False
            self._drop(key)
            self._stats.invalidations += 1
            return True

    def invalidate_tag(self, tag: str) -> int:
        with self._lock:
            keys = list(self._tags.get(tag, ()))
            for key in keys:
                self._drop(key)
            self._stats.invalidations += len(keys)
            return len(keys)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()
            self._tags.clear()

    def __len__(self) -> int:
        return len(self._data)

    def __contains__(self, key: object) -> bool:
        return key in self._data

    def stats(self) -> CacheStats:
        with self._lock:
            s = self._stats
            return CacheStats(
                s.name,
                len(self._data),
                self.capacity,
                s.hits,
                s.misses,
                s.evictions,
                s.expirations,
                s.invalidations,
            )
