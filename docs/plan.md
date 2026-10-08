# 5-week plan

Owners are placeholders (O1–O5) until the team assigns names. The shared work goes to the
owners whose modules are smallest: the KV and Cache engines.

| Owner | Module (primary) | Shared duties |
|---|---|---|
| O1 | KV engine (`kv/`) | Frontend (patient view, dashboard), data ingestion (KV side) |
| O2 | Cache engine (`cache/`) | Frontend (chat, inspector), evaluation harness |
| O3 | Vector engine (`vector/`) | Data ingestion (vector side), benchmarks |
| O4 | Memory router (`router/`) | LoRA training pipeline |
| O5 | Systems/API (`api/`, `pipeline/`) | CI/Docker, docs editor, integration lead |

**Gold-set rule:** O1, O2 and O5 write the evaluation queries. O4 owns the router and the
training templates, so O4 must not see the gold set before it is frozen. This is the
leakage guard from ADR-0004.

## Week 1: contracts and a walking skeleton
- Day 1–2: freeze `contracts/` (Protocols and schemas). Any later change needs a PR
  reviewed by O5 and by the frontend owner.
- O5: config, mock mode, container, `/v1/query` returning a stubbed response with the full
  timing shape. CI green.
- O1/O3: seed generator and ingestion (`make seed`), plus the data manifest with licenses.
- O3: embedding bake-off (bge-small vs PubMedBERT, recall@5 on seed). Freeze the winner on
  Friday; changing it later means re-indexing.
- O1/O2/O5: gold set v1 (≥60 queries) written and frozen.
- **M1 (Fri):** `make demo` answers a query end-to-end in mock mode with stub modules.

## Week 2: real modules
- O1: SQLite + RocksDB backends, contract tests, crash-recovery test.
- O2: LRU+TTL with tags, single-flight, metrics.
- O3: section-aware chunking, Pinecone adapter, in-memory/FAISS stores, reranker.
- O4: normalizer, entity extractors, rules router; router F1 on the gold set.
- O5: orchestrator with every stage timed; red-flag pre-check; scope enforcement.
- **M2 (Fri):** the real path (Pinecone + Claude) answers seed queries. The eval harness
  prints baseline numbers.

## Week 3: safety, caching, UI shell. This is the scope cut line.
- O2: semantic cache with the strict gating from ADR-0005, and adversarial tests.
- O5: citation post-check, sufficiency and conflict abstention, SSE streaming.
- O4: synthetic training data and the first LoRA run.
- O1/O2: Next.js shell: chat with streaming, citation chips, Inspector panel.
- **M3 (Fri):** every "Must" is done. Anything unfinished from the "Should" list is
  re-planned. "Could" items start only if M3 is green.

## Week 4: views, ablations, stretch
- O1: patient view, vector search view, dashboard (latency, hit rate, confusion matrix).
- O2: ablation runner (rules vs LoRA, reranker on/off, semantic cache on/off).
- O3: benchmarks (Pinecone vs FAISS CPU/GPU at seed scale and at 100k vectors).
- O4: LoRA vs rules comparison and calibration.
- O5: stretch goals: consistent-hash ring, sharded KV over 3 processes, Cluster view.
- **M4 (Fri):** feature freeze.

## Week 5: hardening and the viva
- Bug bash, a11y pass, final eval run with real components, results committed.
- README, a "built vs. integrated" table, limitations, viva script and rehearsal ×2.
- **M5:** tag `v1.0`, then a recorded 3-minute demo as a fallback in case the live demo fails.

## Priorities (from Stage 0)
- **Must:** mock mode, SQLite KV, exact cache, Pinecone dense, rules router, safety
  pre-check, abstention, citation check, API with stage timings, chat + inspector, eval
  harness with frozen gold set.
- **Should:** semantic cache, reranker, LoRA router, dashboard, patient/vector views.
- **Could:** hybrid dense+BM25, FAISS-GPU benchmark, Cluster view, multi-process sharding.
