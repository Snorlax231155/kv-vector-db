"""Composition root: the only module that imports concrete implementations.

`build_container(settings)` turns configuration into a fully wired `Container`. Tests build
their own containers (usually mock mode + memory KV) so nothing here is global state.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from medmemory.cache import LRUCache
from medmemory.config import Settings
from medmemory.contracts.protocols import Embedder, KVStore, Reranker, Router, VectorStore
from medmemory.contracts.schemas import ComponentInfo, QueryResponse
from medmemory.generation import LangChainGenerator, build_generator
from medmemory.kv import RecordRepository, open_store
from medmemory.pipeline.metrics import Metrics
from medmemory.pipeline.orchestrator import Orchestrator, PipelineConfig
from medmemory.router import Lexicon, RulesRouter
from medmemory.vector.retriever import ScopedRetriever

log = logging.getLogger(__name__)


@dataclass
class Container:
    settings: Settings
    kv: KVStore
    records: RecordRepository
    embedder: Embedder
    store: VectorStore
    reranker: Reranker
    retriever: ScopedRetriever
    routers: dict[str, Router]
    generator: LangChainGenerator
    exact_cache: LRUCache[QueryResponse]
    metrics: Metrics
    orchestrator: Orchestrator
    ingest_report: dict[str, Any] = field(default_factory=dict)

    def info(self) -> ComponentInfo:
        s = self.settings
        return ComponentInfo(
            mode=s.mode,
            kv_backend=self.kv.name,
            vector_store=self.store.name,
            embedder=self.embedder.name,
            reranker=self.reranker.name,
            generator=self.generator.name,
            router=self.routers["rules"].name,
            semantic_cache_threshold=None,
            min_evidence_score=s.resolved_min_evidence_score,
        )

    def close(self) -> None:
        self.kv.close()


def build_embedder(s: Settings) -> Embedder:
    if s.resolved_embedder == "hashing":
        from medmemory.vector.embedders import HashingEmbedder

        return HashingEmbedder(dim=s.embedding_dim)
    from medmemory.vector.embedders import SentenceTransformerEmbedder

    return SentenceTransformerEmbedder(s.embedding_model, device=s.device)


def build_store(s: Settings, dim: int) -> VectorStore:
    kind = s.resolved_vector_store
    if kind == "memory":
        from medmemory.vector.stores.memory import InMemoryVectorStore

        return InMemoryVectorStore(dim)
    if kind == "faiss":
        from medmemory.vector.stores.faiss_store import FaissVectorStore

        return FaissVectorStore(dim, kind="flat")
    raise ValueError(f"unknown vector store {kind!r}")


def build_reranker(s: Settings) -> Reranker:
    from medmemory.vector.rerankers import CrossEncoderReranker, LexicalReranker, NoopReranker

    kind = s.resolved_reranker
    if kind == "none":
        return NoopReranker()
    if kind == "lexical":
        return LexicalReranker()
    return CrossEncoderReranker(s.reranker_model, device=s.device)


def build_routers(s: Settings) -> dict[str, Router]:
    lexicon = Lexicon.default(s.seed_dir)
    rules = RulesRouter(lexicon)
    return {
        "rules": rules,
        "auto": rules,
    }


def build_container(s: Settings | None = None, *, kv: KVStore | None = None) -> Container:
    s = s or Settings()
    s.validate_for_runtime()
    s.var_dir.mkdir(parents=True, exist_ok=True)
    kv = kv or open_store(s.kv_backend, s.var_dir, s.kv_durability)
    records = RecordRepository(kv)
    embedder = build_embedder(s)
    store = build_store(s, embedder.dim)
    reranker = build_reranker(s)
    retriever = ScopedRetriever(embedder, store, reranker)
    routers = build_routers(s)
    generator = build_generator(
        s.resolved_generator,
        stream_delay_ms=s.mock_stream_delay_ms,
    )
    exact: LRUCache[QueryResponse] = LRUCache(s.cache_capacity, s.cache_ttl_s, name="exact")
    metrics = Metrics()
    orch = Orchestrator(
        records=records,
        retriever=retriever,
        routers=routers,
        generator=generator,
        exact_cache=exact,
        metrics=metrics,
        config=PipelineConfig(s.resolved_min_evidence_score, s.cache_ttl_s),
    )
    return Container(
        s,
        kv,
        records,
        embedder,
        store,
        reranker,
        retriever,
        routers,
        generator,
        exact,
        metrics,
        orch,
    )
