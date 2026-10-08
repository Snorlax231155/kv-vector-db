"""Module interfaces. Every engine is consumed only through these Protocols.

Rules of the road:
* Real and mock implementations satisfy the same Protocol; `medmemory.container` picks
  one from config. Nothing outside `container.py` imports a concrete class.
* KV, embedding, reranking and routing are synchronous (CPU-bound, sub-millisecond to
  tens of ms). Vector search and generation are async (network I/O for the real adapters).
  The orchestrator moves sync calls off the event loop with `asyncio.to_thread` when it
  needs them to overlap (HYBRID retrieval).
* Errors: implementations raise `medmemory.errors.*` types, never vendor exceptions.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, Literal, Protocol, TypeVar, runtime_checkable

import numpy as np
import numpy.typing as npt

from medmemory.contracts.schemas import (
    Chunk,
    Evidence,
    MetadataFilter,
    Namespace,
    Route,
    RouterDecision,
    ScoredChunk,
)

FloatArray = npt.NDArray[np.float32]
V = TypeVar("V")


# --------------------------------------------------------------------------- KV engine


@dataclass(frozen=True, slots=True)
class KVOp:
    """One operation in an atomic batch."""

    op: Literal["put", "delete"]
    key: str
    value: bytes | None = None
    ttl_s: float | None = None


@runtime_checkable
class KVStore(Protocol):
    """Byte-oriented ordered key-value store with TTL, prefix scan and atomic batches.

    Keys are UTF-8 strings compared bytewise, so `scan("patient:P0001:lab:")` returns
    that patient's labs in key order. Expired entries are invisible to every read.
    """

    name: str

    def get(self, key: str) -> bytes | None: ...

    def put(self, key: str, value: bytes, ttl_s: float | None = None) -> None: ...

    def delete(self, key: str) -> bool:
        """Return True if a live key was removed."""
        ...

    def scan(self, prefix: str, limit: int | None = None) -> Iterator[tuple[str, bytes]]: ...

    def batch(self, ops: Sequence[KVOp]) -> None:
        """Apply all ops atomically: after a crash either all are visible or none are."""
        ...

    def count(self, prefix: str = "") -> int: ...

    def purge_expired(self) -> int:
        """Physically remove expired entries; return how many."""
        ...

    def snapshot(self, dest: Path) -> Path:
        """Write a consistent point-in-time copy that can be reopened as a store."""
        ...

    def close(self) -> None: ...


# --------------------------------------------------------------------------- cache engine


@dataclass(slots=True)
class CacheStats:
    name: str
    size: int
    capacity: int
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    expirations: int = 0
    invalidations: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "size": self.size,
            "capacity": self.capacity,
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "expirations": self.expirations,
            "invalidations": self.invalidations,
            "hit_rate": round(self.hit_rate, 4),
        }


@runtime_checkable
class Cache(Protocol[V]):
    """Exact-key cache. Tags allow bulk invalidation, e.g. every entry tagged 'patient:P0001'."""

    def get(self, key: str) -> V | None: ...

    def set(
        self, key: str, value: V, ttl_s: float | None = None, tags: Iterable[str] = ()
    ) -> None: ...

    def invalidate(self, key: str) -> bool: ...

    def invalidate_tag(self, tag: str) -> int: ...

    def clear(self) -> None: ...

    def stats(self) -> CacheStats: ...


@dataclass(frozen=True, slots=True)
class CacheScope:
    """Everything that must match *exactly* before two queries may share an answer.

    `patient_scope` and `filters_key` prevent cross-patient leakage; `entity_signature`
    stops 'CKD stage 3' reusing a 'CKD stage 4' answer; `data_version` makes writes
    invalidate stale answers without a scan.
    """

    patient_scope: str | None
    filters_key: str
    entity_signature: str
    data_version: int

    def bucket(self) -> str:
        return (
            f"{self.patient_scope}|{self.filters_key}|{self.entity_signature}|v{self.data_version}"
        )


@dataclass(frozen=True, slots=True)
class SemanticHit(Generic[V]):
    value: V
    similarity: float
    matched_query: str


@runtime_checkable
class SemanticCache(Protocol[V]):
    def lookup(
        self, query: str, embedding: FloatArray, scope: CacheScope
    ) -> SemanticHit[V] | None: ...

    def store(
        self,
        query: str,
        embedding: FloatArray,
        scope: CacheScope,
        value: V,
        tags: Iterable[str] = (),
    ) -> None: ...

    def invalidate_tag(self, tag: str) -> int: ...

    def clear(self) -> None: ...

    def stats(self) -> CacheStats: ...


# --------------------------------------------------------------------------- vector engine


@runtime_checkable
class Embedder(Protocol):
    """Returns float32, L2-normalised vectors so cosine == dot product everywhere."""

    name: str
    dim: int

    def embed_documents(self, texts: Sequence[str]) -> FloatArray:
        """Shape (n, dim)."""
        ...

    def embed_query(self, text: str) -> FloatArray:
        """Shape (dim,). May apply a model-specific query instruction prefix."""
        ...


@runtime_checkable
class VectorStore(Protocol):
    name: str
    supports_text_search: bool

    async def upsert(
        self, namespace: Namespace, chunks: Sequence[Chunk], vectors: FloatArray
    ) -> int: ...

    async def query(
        self,
        namespace: Namespace,
        vector: FloatArray,
        top_k: int,
        filter: MetadataFilter | None = None,
    ) -> list[ScoredChunk]: ...

    async def text_query(
        self,
        namespace: Namespace,
        text: str,
        top_k: int,
        filter: MetadataFilter | None = None,
    ) -> list[ScoredChunk]:
        """BM25-style keyword search. Only valid when `supports_text_search` is True."""
        ...

    async def fetch(self, namespace: Namespace, ids: Sequence[str]) -> list[Chunk]: ...

    async def delete(
        self,
        namespace: Namespace,
        ids: Sequence[str] | None = None,
        filter: MetadataFilter | None = None,
        delete_all: bool = False,
    ) -> None: ...

    async def count(self, namespace: Namespace) -> int: ...


@runtime_checkable
class Reranker(Protocol):
    name: str

    def rerank(self, query: str, chunks: Sequence[ScoredChunk], top_k: int) -> list[ScoredChunk]:
        """Return at most top_k chunks, re-ordered, with `rerank_score` set and `rank` renumbered."""
        ...


# --------------------------------------------------------------------------- router


@runtime_checkable
class Router(Protocol):
    name: str

    def route(self, query: str, patient_scope: str | None = None) -> RouterDecision: ...


# --------------------------------------------------------------------------- generation


@runtime_checkable
class AnswerGenerator(Protocol):
    """Produces answer text in which every sentence ends with citation markers like [S1].

    Implementations must only use `evidence`; the post-generation citation check
    enforces that and strips anything it cannot ground.
    """

    name: str

    async def generate(self, query: str, evidence: Sequence[Evidence], route: Route) -> str: ...

    def stream(
        self, query: str, evidence: Sequence[Evidence], route: Route
    ) -> AsyncIterator[str]: ...
