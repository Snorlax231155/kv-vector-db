# Module 5: API & Orchestration (`medmemory.api`, `medmemory.pipeline`)

**Subsystem:** Pipeline & API Layer · **Contract:** OpenAPI spec at `/docs` or `python tasks.py openapi` · **Tests:** `tests/api/`, `tests/pipeline/`

## Pipeline Orchestrator (`medmemory.pipeline`)
- `orchestrator.py`: Implements the end-to-end request lifecycle: safety pre-check → routing → patient scoping → exact caching → parallel KV/Vector retrieval → evidence merging → sufficiency evaluation → grounded extractive generation → citation validation → cache write.
- `merger.py`: Deduplicates and ranks retrieved evidence from both engines.
- `timing.py`: Records per-stage latency (`{stage, start_ms, duration_ms, status, meta}`).
- `metrics.py`: Tracks in-process latency histograms, request counters, and cache hit metrics.

## API Service (`medmemory.api`)
- `app.py`: FastAPI application lifespan, CORS handling, trace ID propagation, request logging, and route registration.
- `security.py`: Token-based bearer authentication stub and per-client token-bucket rate limiter.
- `observability.py`: Structured JSON logging and privacy-preserving audit logs with salted patient hashes.

## Endpoints
- `POST /v1/query`: Synchronous JSON query endpoint.
- `POST /v1/query/stream`: Server-Sent Events (SSE) streaming endpoint emitting stage events (`meta`, `retrieval`, `token`, `final`).
- `GET /v1/scenarios`: Built-in clinical demo scenarios.
- `GET /v1/patients[/{id}]`: Patient overview and record view.
- `POST /v1/patients/{id}/labs`: Lab record ingestion with write-through cache invalidation.
- `GET /v1/kv[/{key}]`: Key-value browser with internal prefix protections.
- `POST /v1/search`: Direct vector similarity search endpoint.
- `GET /v1/sources/{source_id}`: Evidence provenance inspection.
- `GET /v1/metrics`: System metrics and cache efficiency statistics.
- `GET /healthz`, `GET /readyz`: Standard health and readiness probes.
