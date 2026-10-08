# MedMemory Architecture (Phase 1 Milestone)

> **Educational prototype. Not medical advice. Synthetic and public data only.**

MedMemory answers clinical questions by orchestrating **exact key-value lookups** (patient records, lab results, ICD-10-CM codes, drug identifiers) with **semantic vector search** (clinical notes, drug labels, consumer health summaries).

Our core contribution is the hybrid database and orchestration layer: query routing, patient scoping, exact caching with tag invalidation, multi-engine retrieval, grounded synthesis, and end-to-end stage timing.

---

## High-Level Architecture

```mermaid
flowchart LR
  Client[REST / SSE Client] --> API[FastAPI Service]
  subgraph Service[API & Middleware Layer]
    API --> MW[Auth Stub · Rate Limiter · Trace ID · Audit Logger]
    MW --> ORCH[Pipeline Orchestrator]
  end
  ORCH --> SAFE[Safety Pre-Check<br/>Red flags · Scope violation · Out of scope]
  ORCH --> ROUTER[Rules Router<br/>Normalization · Deterministic Entity Extraction]
  ORCH --> CACHE[Exact Cache Engine<br/>LRU + TTL + SingleFlight + Tag Invalidation]
  ORCH --> KV[(KV Engine<br/>SQLite WAL / Memory)]
  ORCH --> VEC[Vector Engine<br/>Hashing / BGE Embedder · In-Memory / FAISS Index]
  ORCH --> GEN[Grounded Generator<br/>Extractive Synthesis]
  ORCH --> CHECK[Sufficiency & Citation Checkers]
```

---

## Request Lifecycle

```mermaid
sequenceDiagram
  autonumber
  participant C as Client
  participant A as API / Orchestrator
  participant S as Safety
  participant R as Router
  participant X as Cache
  participant K as KV Engine
  participant V as Vector Engine
  participant G as Generator
  participant Q as Checks

  C->>A: POST /v1/query {query, patient_scope}
  A->>S: pre-check(query)
  alt red flag emergency
    S-->>A: Emergency guidance
    A-->>C: Immediate advisory (no retrieval, no generation)
  end
  A->>R: route(query, scope)
  R-->>A: Route (KV | VECTOR | HYBRID), Entities, Confidence
  A->>A: Scope check (named patient == active patient?)
  A->>X: exact lookup(key = query + scope + filters + data_version)
  alt exact hit
    X-->>A: Cached response
  else miss (SingleFlight per key)
    par KV route or HYBRID
      A->>K: get / scan(prefix)
    and VECTOR route or HYBRID
      A->>V: embed → search(namespace, filter) → rerank
    end
    A->>Q: merge + evidence sufficiency (+ conflict check)
    alt insufficient or conflicting
      Q-->>A: Abstain with clinical rationale
    else sufficient
      A->>G: generate(query, evidence S1..Sn)
      G-->>A: Grounded answer with [S#] citations
      A->>Q: citation check (markers valid, numbers grounded)
    end
    A->>X: write exact entry (tagged by patient)
  end
  A-->>C: answer, route, entities, cache_hit, timings, records, chunks, citations, trace_id
```

Every stage records `{stage, start_ms, duration_ms, status, meta}` for observability and auditability.

---

## Patient Scoping (Defense in Depth)

1. **Request boundary:** Active patient scope is validated before retrieval. If a query references a different patient ID or names multiple patients, execution immediately halts with `scope_violation`.
2. **Key-Value prefix containment:** All patient records live strictly under `patient:{id}:`. The KV repository scans only the in-scope prefix.
3. **Vector namespace isolation:** The `patient_notes` namespace is searched exclusively with an exact metadata filter `{"patient_id": {"$eq": scope}}`. Post-filtering verifies that no out-of-scope chunks survive.
4. **Cache scoping:** Cache keys incorporate patient scope and data versions (`meta:version:{patient}`). Cached entries are tagged with `patient:{id}`, enabling atomic write-through invalidation when new labs or records are added.
5. **Privacy-preserving audit logging:** Audit logs record salted hashes of patient IDs rather than raw patient identifiers.

---

## Module Boundaries & Protocols

| Module | Package | Interface (`contracts/protocols.py`) | Implementation |
|---|---|---|---|
| **KV Engine** | `medmemory.kv` | `KVStore` | `SQLiteKV` (WAL), `MemoryKV` |
| **Vector Engine** | `medmemory.vector` | `Embedder`, `VectorStore`, `Reranker` | `HashingEmbedder`, `InMemoryVectorStore`, `FaissVectorStore`, `LexicalReranker` |
| **Router** | `medmemory.router` | `Router` | `RulesRouter` + `Lexicon` + `extract` |
| **Cache Engine** | `medmemory.cache` | `Cache` | `LRUCache`, `SingleFlight` |
| **Pipeline & API** | `medmemory.pipeline`, `medmemory.api` | Pipeline Orchestrator & FastAPI routes | `Orchestrator`, `create_app` |
| **Safety & Checks** | `medmemory.safety`, `medmemory.generation` | Pre-checks & Grounding | `precheck`, `check_citations`, `ExtractiveChatModel` |

All concrete dependencies are injected via `container.py` (`Container`), enabling isolated unit testing and modular team development.
