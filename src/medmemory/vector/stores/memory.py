"""Exact (brute-force) in-memory vector store: the mock for Pinecone and the ground truth for
recall@k in benchmarks. Cosine == dot product because embedders return unit vectors.

Persistence: `save(dir)` / `load(dir)` write one .npz + .jsonl per namespace so a restart
doesn't re-embed the corpus.
"""

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from medmemory.contracts.protocols import FloatArray
from medmemory.contracts.schemas import Chunk, MetadataFilter, Namespace, ScoredChunk
from medmemory.vector.bm25 import BM25
from medmemory.vector.filters import matches


def chunk_meta(c: Chunk) -> dict[str, object]:
    """The flat metadata dict filters are evaluated against (same fields Pinecone stores)."""
    meta: dict[str, object] = {
        **c.metadata,
        "doc_id": c.doc_id,
        "section": c.section,
        "title": c.title,
    }
    if c.patient_id:
        meta["patient_id"] = c.patient_id
    return meta


class _NS:
    def __init__(self, dim: int) -> None:
        self.ids: list[str] = []
        self.pos: dict[str, int] = {}
        self.chunks: list[Chunk] = []
        self.meta: list[dict[str, object]] = []
        self.matrix: FloatArray = np.zeros((0, dim), dtype=np.float32)
        self.bm25: BM25 | None = None


class InMemoryVectorStore:
    name = "memory-exact"
    supports_text_search = True

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self._ns: dict[Namespace, _NS] = {}
        self._lock = threading.RLock()

    def _get(self, ns: Namespace) -> _NS:
        if ns not in self._ns:
            self._ns[ns] = _NS(self.dim)
        return self._ns[ns]

    # ------------------------------------------------------------------ writes

    def upsert_sync(
        self, namespace: Namespace, chunks: Sequence[Chunk], vectors: FloatArray
    ) -> int:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors length mismatch")
        with self._lock:
            ns = self._get(namespace)
            new_rows: list[tuple[Chunk, FloatArray]] = []
            for chunk, vec in zip(chunks, vectors, strict=True):
                i = ns.pos.get(chunk.chunk_id)
                if i is not None:  # overwrite in place: upserts are idempotent
                    ns.chunks[i] = chunk
                    ns.meta[i] = chunk_meta(chunk)
                    ns.matrix[i] = vec
                else:
                    ns.pos[chunk.chunk_id] = len(ns.ids) + len(new_rows)
                    new_rows.append((chunk, vec))
            if new_rows:
                for chunk, _ in new_rows:
                    ns.ids.append(chunk.chunk_id)
                    ns.chunks.append(chunk)
                    ns.meta.append(chunk_meta(chunk))
                ns.matrix = np.vstack(
                    [ns.matrix, np.stack([v for _, v in new_rows]).astype(np.float32)]
                )
            ns.bm25 = None
        return len(chunks)

    async def upsert(
        self, namespace: Namespace, chunks: Sequence[Chunk], vectors: FloatArray
    ) -> int:
        return self.upsert_sync(namespace, chunks, vectors)

    async def delete(
        self,
        namespace: Namespace,
        ids: Sequence[str] | None = None,
        filter: MetadataFilter | None = None,
        delete_all: bool = False,
    ) -> None:
        with self._lock:
            ns = self._get(namespace)
            if delete_all:
                self._ns[namespace] = _NS(self.dim)
                return
            drop = set(ids or [])
            if filter:
                drop |= {ns.ids[i] for i, m in enumerate(ns.meta) if matches(m, filter)}
            keep = [i for i, cid in enumerate(ns.ids) if cid not in drop]
            fresh = _NS(self.dim)
            fresh.ids = [ns.ids[i] for i in keep]
            fresh.chunks = [ns.chunks[i] for i in keep]
            fresh.meta = [ns.meta[i] for i in keep]
            fresh.matrix = ns.matrix[keep] if keep else np.zeros((0, self.dim), np.float32)
            fresh.pos = {cid: i for i, cid in enumerate(fresh.ids)}
            self._ns[namespace] = fresh

    # ------------------------------------------------------------------ reads

    def _mask(self, ns: _NS, flt: MetadataFilter | None) -> np.ndarray:
        if not flt:
            return np.ones(len(ns.ids), dtype=bool)
        return np.fromiter((matches(m, flt) for m in ns.meta), dtype=bool, count=len(ns.meta))

    def query_sync(
        self,
        namespace: Namespace,
        vector: FloatArray,
        top_k: int,
        filter: MetadataFilter | None = None,
    ) -> list[ScoredChunk]:
        with self._lock:
            ns = self._get(namespace)
            if not ns.ids:
                return []
            scores = ns.matrix @ np.asarray(vector, dtype=np.float32)
            mask = self._mask(ns, filter)
            scores = np.where(mask, scores, -np.inf)
            k = min(top_k, int(mask.sum()))
            if k <= 0:
                return []
            idx = np.argpartition(-scores, k - 1)[:k]
            idx = idx[np.argsort(-scores[idx])]
            return [
                ScoredChunk(
                    chunk=ns.chunks[i], score=float(scores[i]), rank=r + 1, retriever="dense"
                )
                for r, i in enumerate(idx)
            ]

    async def query(
        self,
        namespace: Namespace,
        vector: FloatArray,
        top_k: int,
        filter: MetadataFilter | None = None,
    ) -> list[ScoredChunk]:
        return self.query_sync(namespace, vector, top_k, filter)

    async def text_query(
        self, namespace: Namespace, text: str, top_k: int, filter: MetadataFilter | None = None
    ) -> list[ScoredChunk]:
        with self._lock:
            ns = self._get(namespace)
            if not ns.ids:
                return []
            if ns.bm25 is None:
                ns.bm25 = BM25([f"{c.title} {c.section} {c.text}" for c in ns.chunks])
            scores = np.asarray(ns.bm25.scores(text), dtype=np.float32)
            mask = self._mask(ns, filter) & (scores > 0)
            scores = np.where(mask, scores, -np.inf)
            k = min(top_k, int(mask.sum()))
            if k <= 0:
                return []
            idx = np.argsort(-scores)[:k]
            return [
                ScoredChunk(
                    chunk=ns.chunks[i], score=float(scores[i]), rank=r + 1, retriever="bm25"
                )
                for r, i in enumerate(idx)
            ]

    async def fetch(self, namespace: Namespace, ids: Sequence[str]) -> list[Chunk]:
        with self._lock:
            ns = self._get(namespace)
            return [ns.chunks[ns.pos[i]] for i in ids if i in ns.pos]

    async def count(self, namespace: Namespace) -> int:
        return len(self._get(namespace).ids)

    def all_chunks(self, namespace: Namespace) -> list[Chunk]:
        return list(self._get(namespace).chunks)

    def matrix(self, namespace: Namespace) -> FloatArray:
        return self._get(namespace).matrix

    # ------------------------------------------------------------------ persistence

    def save(self, directory: Path, fingerprint: str) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        with self._lock:
            for name, ns in self._ns.items():
                np.save(directory / f"{name.value}.npy", ns.matrix)
                with (directory / f"{name.value}.jsonl").open("w", encoding="utf-8") as fh:
                    for c in ns.chunks:
                        fh.write(c.model_dump_json(exclude={"source_id"}) + "\n")
        (directory / "fingerprint.txt").write_text(fingerprint, encoding="utf-8")

    def load(self, directory: Path, fingerprint: str) -> bool:
        fp = directory / "fingerprint.txt"
        if not fp.exists() or fp.read_text(encoding="utf-8") != fingerprint:
            return False
        with self._lock:
            for name in Namespace:
                mat_path = directory / f"{name.value}.npy"
                rows_path = directory / f"{name.value}.jsonl"
                if not mat_path.exists():
                    continue
                chunks = [
                    Chunk.model_validate(json.loads(line))
                    for line in rows_path.read_text(encoding="utf-8").splitlines()
                    if line
                ]
                matrix = np.load(mat_path)
                self._ns[name] = _NS(self.dim)
                self.upsert_sync(name, chunks, matrix)
        return True


async def gather_queries(
    store: InMemoryVectorStore, jobs: list[tuple[Namespace, FloatArray, int, MetadataFilter | None]]
) -> list[list[ScoredChunk]]:
    return await asyncio.gather(*(store.query(ns, v, k, f) for ns, v, k, f in jobs))
