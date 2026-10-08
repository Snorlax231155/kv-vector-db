"""Rerankers: re-score a candidate list with a more expensive model.

* `CrossEncoderReranker` (real): a cross-encoder reads (query, passage) jointly, which is
  far more precise than comparing two independently computed embeddings. Runs locally on
  GPU/CPU, so there's no Pinecone rerank quota (500 req/month on Starter).
* `LexicalReranker` (mock): blends the retriever score with BM25-style term overlap and an
  exact-entity bonus. Deterministic; good at 'the chunk that names the drug I asked about'.
* `NoopReranker`: identity, for the 'without reranker' ablation.
"""

from __future__ import annotations

import math
import threading
from collections.abc import Sequence

from medmemory.contracts.schemas import ScoredChunk
from medmemory.vector.embedders import tokenize


def _renumber(chunks: list[ScoredChunk]) -> list[ScoredChunk]:
    return [c.model_copy(update={"rank": i + 1}) for i, c in enumerate(chunks)]


class NoopReranker:
    name = "none"

    def rerank(self, query: str, chunks: Sequence[ScoredChunk], top_k: int) -> list[ScoredChunk]:
        return _renumber(list(chunks)[:top_k])


class LexicalReranker:
    name = "lexical"

    def __init__(self, alpha: float = 0.5) -> None:
        self.alpha = alpha

    def rerank(self, query: str, chunks: Sequence[ScoredChunk], top_k: int) -> list[ScoredChunk]:
        q = set(tokenize(query))
        if not q or not chunks:
            return _renumber(list(chunks)[:top_k])
        max_retr = max((c.score for c in chunks), default=1.0) or 1.0
        scored = []
        for c in chunks:
            body = tokenize(f"{c.chunk.section} {c.chunk.text}")
            terms = set(body)
            overlap = len(q & terms) / len(q)
            density = sum(1 for t in body if t in q) / math.sqrt(len(body) + 1)
            section_bonus = 0.15 if any(t in tokenize(c.chunk.section) for t in q) else 0.0
            lexical = 0.6 * overlap + 0.25 * min(density, 1.0) + section_bonus
            s = self.alpha * (c.score / max_retr) + (1 - self.alpha) * lexical
            scored.append(c.model_copy(update={"rerank_score": round(s, 6)}))
        scored.sort(key=lambda c: c.rerank_score or 0.0, reverse=True)
        return _renumber(scored[:top_k])


class CrossEncoderReranker:
    def __init__(
        self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", device: str = "auto"
    ) -> None:
        from sentence_transformers import CrossEncoder

        if device == "auto":
            try:
                import torch

                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"
        self._model = CrossEncoder(model_name, device=device)
        self.name = f"{model_name}@{device}"
        self._lock = threading.Lock()

    def rerank(self, query: str, chunks: Sequence[ScoredChunk], top_k: int) -> list[ScoredChunk]:
        if not chunks:
            return []
        pairs = [(query, f"{c.chunk.section}: {c.chunk.text}") for c in chunks]
        with self._lock:
            logits = self._model.predict(pairs, convert_to_numpy=True, show_progress_bar=False)
        scored = [
            c.model_copy(update={"rerank_score": float(1 / (1 + math.exp(-float(s))))})
            for c, s in zip(chunks, logits, strict=True)
        ]
        scored.sort(key=lambda c: c.rerank_score or 0.0, reverse=True)
        return _renumber(scored[:top_k])


def rrf_fuse(
    lists: Sequence[Sequence[ScoredChunk]], k: int = 60, top_k: int | None = None
) -> list[ScoredChunk]:
    """Reciprocal Rank Fusion: score = sum over lists of 1 / (k + rank). Scale-free, so it can
    fuse cosine and BM25 lists whose raw scores aren't comparable."""
    fused: dict[str, float] = {}
    best: dict[str, ScoredChunk] = {}
    dense_score: dict[str, float] = {}
    for lst in lists:
        for item in lst:
            cid = item.chunk.chunk_id
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (k + item.rank)
            best.setdefault(cid, item)
            if item.retriever.startswith("dense"):
                dense_score[cid] = item.score
    order = sorted(fused, key=lambda c: fused[c], reverse=True)
    if top_k is not None:
        order = order[:top_k]
    # Keep the dense cosine as `score` when available so the sufficiency threshold still
    # means something; the fused value goes in rerank_score-free metadata via retriever tag.
    return [
        best[cid].model_copy(
            update={
                "score": dense_score.get(cid, best[cid].score),
                "rank": i + 1,
                "retriever": f"hybrid-rrf({fused[cid]:.4f})",
            }
        )
        for i, cid in enumerate(order)
    ]
