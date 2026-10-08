# Module 2: Vector engine (`medmemory.vector`)

**Owner:** Person 2 (Vector Storage & Similarity Search) · **Interfaces:** `Embedder`, `VectorStore`, `Reranker` · **Tests:** `tests/vector/`

The vector engine handles document chunking, lexical and dense vector representation, structured metadata filtering, similarity indexing, and multi-stage ranking.

| Subsystem | File | Description |
|---|---|---|
| Chunking | `chunking.py` | **Section-aware chunking:** Clinical note headers (HPI, CURRENT MEDICATIONS) and FDA label sections serve as hard boundaries. Sentences are packed to ~120–160 words with 1-sentence overlap. Each chunk prepends contextual headers (`title \| section \| text`). Deterministic chunk IDs ensure idempotent upserts. |
| Embedders | `embedders.py` | `HashingEmbedder` provides fast, reproducible lexical vector generation without external model downloads. `SentenceTransformerEmbedder` (`BAAI/bge-small-en-v1.5`, 384-d) provides dense semantic embeddings. |
| Stores | `stores/` | `InMemoryVectorStore` (exact cosine similarity with disk snapshot persistence) and `FaissVectorStore` (Flat/HNSW index support). |
| Filters | `filters.py` | Full MongoDB-style query filter evaluator (`$eq`, `$in`, `$gte`, `$and`, `$or`), enabling localized metadata-constrained vector queries. |
| Rerankers | `rerankers.py` | `LexicalReranker`, `CrossEncoderReranker`, and `rrf_fuse` for Reciprocal Rank Fusion combining dense semantic scores with BM25 sparse keyword scores. |
| Retriever | `retriever.py` | Coordinates parallel multi-namespace search, fusions, post-filtering, reranking, and result diversification. |

## Scoping & Isolation
- The `patient_notes` namespace is searched strictly with `patient_id == scope`, and never executed without an active patient scope.
- Post-search validation drops any out-of-scope chunks as defense-in-depth against leakage.
