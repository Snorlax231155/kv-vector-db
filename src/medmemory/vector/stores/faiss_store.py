"""Local FAISS store, used as the benchmark baseline against Pinecone. GPU is optional.

Index kinds: "flat" (exact inner product), "hnsw" (graph ANN, CPU only), "ivf" (inverted
lists; trained on the first batch). On a CUDA build of FAISS, flat/ivf move to the GPU with
`index_cpu_to_gpu`. PyPI's faiss-gpu wheels are Linux-only, so on Windows this runs on CPU
(see ADR-0001 and the README's GPU section).

Metadata filtering: FAISS indexes vectors, not metadata, so we over-fetch and post-filter.
If the filter is very selective (e.g. one patient's notes), we fall back to an exact scan of
the matching rows, which is fast because the subset is small.
"""

from __future__ import annotations

import threading
from collections.abc import Sequence
from typing import Any, Literal

import numpy as np

from medmemory.contracts.protocols import FloatArray
from medmemory.contracts.schemas import Chunk, MetadataFilter, Namespace, ScoredChunk
from medmemory.vector.filters import matches
from medmemory.vector.stores.memory import InMemoryVectorStore

IndexKind = Literal["flat", "hnsw", "ivf"]


def faiss_gpu_available() -> bool:
    try:
        import faiss

        return hasattr(faiss, "StandardGpuResources") and faiss.get_num_gpus() > 0
    except Exception:
        return False


class FaissVectorStore(InMemoryVectorStore):
    supports_text_search = True

    def __init__(
        self, dim: int, kind: IndexKind = "flat", use_gpu: bool | None = None, overfetch: int = 8
    ) -> None:
        super().__init__(dim)
        import faiss

        self._faiss: Any = faiss
        self.kind = kind
        self.use_gpu = (
            faiss_gpu_available() if use_gpu is None else (use_gpu and faiss_gpu_available())
        )
        self.overfetch = overfetch
        self._indexes: dict[Namespace, Any] = {}
        self._dirty: set[Namespace] = set()
        self._ilock = threading.Lock()
        self._gpu_res: Any = faiss.StandardGpuResources() if self.use_gpu else None
        self.name = f"faiss-{kind}-{'gpu' if self.use_gpu else 'cpu'}"

    def upsert_sync(
        self, namespace: Namespace, chunks: Sequence[Chunk], vectors: FloatArray
    ) -> int:
        n = super().upsert_sync(namespace, chunks, vectors)
        self._dirty.add(namespace)
        return n

    def _build(self, namespace: Namespace) -> Any:
        faiss = self._faiss
        mat = np.ascontiguousarray(self.matrix(namespace), dtype=np.float32)
        if self.kind == "hnsw":
            index = faiss.IndexHNSWFlat(self.dim, 32, faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efSearch = 64
        elif self.kind == "ivf":
            nlist = max(1, min(1024, int(np.sqrt(len(mat))) * 4))
            quantizer = faiss.IndexFlatIP(self.dim)
            index = faiss.IndexIVFFlat(quantizer, self.dim, nlist, faiss.METRIC_INNER_PRODUCT)
            index.train(mat)
            index.nprobe = max(1, nlist // 16)
        else:
            index = faiss.IndexFlatIP(self.dim)
        if self.use_gpu and self.kind != "hnsw":
            index = faiss.index_cpu_to_gpu(self._gpu_res, 0, index)
        if len(mat):
            index.add(mat)
        return index

    def _index(self, namespace: Namespace) -> Any:
        with self._ilock:
            if namespace in self._dirty or namespace not in self._indexes:
                self._indexes[namespace] = self._build(namespace)
                self._dirty.discard(namespace)
            return self._indexes[namespace]

    def search_batch(
        self, namespace: Namespace, queries: FloatArray, top_k: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Unfiltered batched search; used by the benchmark to measure raw index throughput."""
        return self._index(namespace).search(np.ascontiguousarray(queries, dtype=np.float32), top_k)

    def query_sync(
        self,
        namespace: Namespace,
        vector: FloatArray,
        top_k: int,
        filter: MetadataFilter | None = None,
    ) -> list[ScoredChunk]:
        ns = self._get(namespace)
        if not ns.ids:
            return []
        q = np.asarray(vector, dtype=np.float32).reshape(1, -1)
        fetch = min(len(ns.ids), top_k * (self.overfetch if filter else 1))
        scores, idx = self._index(namespace).search(q, fetch)
        hits = [
            (float(s), int(i))
            for s, i in zip(scores[0], idx[0], strict=True)
            if i >= 0 and matches(ns.meta[i], filter)
        ]
        if filter and len(hits) < top_k:
            return super().query_sync(
                namespace, vector, top_k, filter
            )  # selective filter: exact subset scan
        return [
            ScoredChunk(chunk=ns.chunks[i], score=s, rank=r + 1, retriever=f"dense-{self.kind}")
            for r, (s, i) in enumerate(hits[:top_k])
        ]
