"""Scoped retrieval: embed, search the right namespaces, fuse, enforce scope, rerank.

This is the vector side of the request lifecycle. Patient scoping is enforced twice:
1. the `patient_notes` namespace is only searched with a `patient_id == scope` filter
   (and not at all when there is no scope), and
2. every returned chunk is re-checked here; anything out of scope is dropped and logged as
   a security event. With a correct store, (2) never fires; it exists to catch a
   misconfigured filter or a buggy adapter, and the leakage test proves it works.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from medmemory.contracts.protocols import Embedder, FloatArray, Reranker, VectorStore
from medmemory.contracts.schemas import MetadataFilter, Namespace, ScoredChunk
from medmemory.pipeline.timing import StageTimer
from medmemory.vector.filters import and_filters
from medmemory.vector.rerankers import NoopReranker, rrf_fuse

log = logging.getLogger("medmemory.security")


@dataclass
class RetrievalResult:
    chunks: list[ScoredChunk]
    candidates: int = 0
    blocked_out_of_scope: int = 0
    searched: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)


class ScopedRetriever:
    def __init__(
        self, embedder: Embedder, store: VectorStore, reranker: Reranker, candidate_k: int = 24
    ) -> None:
        self.embedder = embedder
        self.store = store
        self.reranker = reranker
        self.candidate_k = candidate_k

    def embed(self, text: str) -> FloatArray:
        return self.embedder.embed_query(text)

    async def retrieve(
        self,
        query: str,
        namespaces: Sequence[Namespace],
        patient_scope: str | None,
        timer: StageTimer,
        *,
        query_vector: FloatArray | None = None,
        namespace_filters: dict[Namespace, MetadataFilter] | None = None,
        extra_filter: MetadataFilter | None = None,
        top_k: int = 5,
        use_reranker: bool = True,
        hybrid_sparse: bool = False,
    ) -> RetrievalResult:
        result = RetrievalResult(chunks=[])
        if query_vector is None:
            with timer.stage("embed", embedder=self.embedder.name):
                query_vector = await asyncio.to_thread(self.embedder.embed_query, query)

        jobs: list[tuple[Namespace, str, Any]] = []
        for ns in namespaces:
            flt = and_filters(extra_filter, (namespace_filters or {}).get(ns))
            if ns == Namespace.PATIENT_NOTES:
                if patient_scope is None:
                    result.skipped.append(f"{ns.value}: no active patient")
                    continue
                flt = and_filters({"patient_id": {"$eq": patient_scope}}, flt)
            result.searched.append(ns.value)
            jobs.append((ns, "dense", self.store.query(ns, query_vector, self.candidate_k, flt)))
            if hybrid_sparse and self.store.supports_text_search:
                jobs.append((ns, "bm25", self.store.text_query(ns, query, self.candidate_k, flt)))

        with timer.stage(
            "vector_search", store=self.store.name, namespaces=result.searched, hybrid=hybrid_sparse
        ) as st:
            lists = await asyncio.gather(*(j[2] for j in jobs)) if jobs else []
            st["raw_hits"] = sum(len(x) for x in lists)

        per_ns: dict[Namespace, list[list[ScoredChunk]]] = {}
        for (ns, _, _), hits in zip(jobs, lists, strict=True):
            per_ns.setdefault(ns, []).append(hits)
        merged: list[ScoredChunk] = []
        for ns_lists in per_ns.values():
            merged.extend(
                rrf_fuse(ns_lists, top_k=self.candidate_k) if len(ns_lists) > 1 else ns_lists[0]
            )

        # Scope post-filter (defence in depth).
        safe: list[ScoredChunk] = []
        for c in merged:
            pid = c.chunk.patient_id
            if pid is not None and pid != patient_scope:
                result.blocked_out_of_scope += 1
                log.error(
                    "scope_leak_blocked chunk=%s chunk_patient=%s scope=%s",
                    c.chunk.chunk_id,
                    pid,
                    patient_scope,
                )
                continue
            safe.append(c)
        safe.sort(key=lambda c: c.score, reverse=True)
        result.candidates = len(safe)

        reranker = self.reranker if use_reranker else NoopReranker()
        with timer.stage(
            "rerank", reranker=reranker.name, candidates=len(safe), skipped=not use_reranker
        ):
            ranked = await asyncio.to_thread(reranker.rerank, query, safe, max(top_k * 2, top_k))
        result.chunks = diversify(ranked, top_k)
        return result


def diversify(ranked: list[ScoredChunk], top_k: int) -> list[ScoredChunk]:
    """Keep the best chunk of every namespace that produced one, then fill by rank. Without
    this, a HYBRID question tends to get five note chunks and no label chunk."""
    firsts: dict[Namespace, ScoredChunk] = {}
    for c in ranked:
        firsts.setdefault(c.chunk.namespace, c)
    chosen = {c.chunk.chunk_id for c in list(firsts.values())[:top_k]}
    for c in ranked:
        if len(chosen) >= top_k:
            break
        chosen.add(c.chunk.chunk_id)
    out = [c for c in ranked if c.chunk.chunk_id in chosen]
    return [c.model_copy(update={"rank": i + 1}) for i, c in enumerate(out)]
