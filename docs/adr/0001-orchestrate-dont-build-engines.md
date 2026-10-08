# ADR-0001: Orchestrate existing engines instead of building storage engines

- Status: accepted
- Date: 2026-10-07

## Context
The project has five people and five weeks. A storage engine, an ANN index or an LLM
would each take more time than that to build well, and grading rewards a working system
whose claims can be defended.

## Decision
Use SQLite (WAL) and RocksDB (via `rocksdict`) for KV, Pinecone serverless and FAISS for
vectors, sentence-transformers for embeddings, and LangChain together with Claude for
generation. Our own code is the layer around them: routing, scope enforcement, caching
(exact, semantic and single-flight), result merging, evidence sufficiency, citation
checking, instrumentation, the evaluation harness, and the UI.

## Consequences
- The README has a "built vs. integrated" table, and claims are scoped to what we built.
- Every vendor sits behind a Protocol, so a vendor can be swapped without touching the
  orchestrator and every module can be tested with fakes.
