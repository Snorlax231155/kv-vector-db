"""FastAPI application: lifecycle, middleware, and HTTP endpoints.

Run: `python -m medmemory serve` (or `python tasks.py serve`). OpenAPI at /docs.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from medmemory.api.observability import AuditLog, configure_logging, trace_id_var
from medmemory.api.scenarios import SCENARIOS
from medmemory.api.security import TokenBucket, User, rate_limited
from medmemory.config import Settings
from medmemory.container import Container, build_container
from medmemory.contracts.schemas import (
    DISCLAIMER,
    ComponentInfo,
    KVRecord,
    LabWrite,
    Namespace,
    Origin,
    PatientRecord,
    PatientSummary,
    QueryOptions,
    QueryRequest,
    QueryResponse,
    ReadyStatus,
    Scenario,
    SearchRequest,
    SearchResponse,
    SourceView,
    WriteResult,
)
from medmemory.errors import ConfigError, MedMemoryError, ScopeViolationError
from medmemory.ingest.build import ingest
from medmemory.pipeline.timing import StageTimer
from medmemory.vector.filters import and_filters

log = logging.getLogger("medmemory.api")
API_VERSION = "1.0.0"
ROOT = Path.cwd()


def get_container(request: Request) -> Container:
    c: Container | None = getattr(request.app.state, "container", None)
    if c is None:
        raise HTTPException(503, "service is starting")
    return c


def create_app(settings: Settings | None = None, container: Container | None = None) -> FastAPI:
    settings = settings or Settings()
    configure_logging(settings.log_level, settings.log_json)
    cfg: Settings = settings

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        c: Container = (
            container if container is not None else await asyncio.to_thread(build_container, cfg)
        )
        report = await ingest(c)
        log.info("ingest complete", extra={"fields": {"ingest": report}})
        # warm-up: compile router regexes and touch every stage once so the first real
        # request isn't paying one-off costs in the latency charts
        await c.orchestrator.answer(
            QueryRequest(
                query="warm-up: latest hemoglobin a1c",
                patient_scope="P0001",
                options=QueryOptions(use_cache=False),
            )
        )
        c.metrics.samples.clear()
        c.metrics.counters.clear()
        app.state.container = c
        yield
        c.close()

    app = FastAPI(
        title="MedMemory API",
        version=API_VERSION,
        description=f"Hybrid KV + vector retrieval for a clinical knowledge assistant. **{DISCLAIMER}**",
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.rate_limiter = TokenBucket(settings.rate_limit_per_minute)
    app.state.audit = AuditLog(settings.var_dir / "audit.jsonl")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
        expose_headers=["X-Trace-Id"],
    )

    @app.middleware("http")
    async def trace_and_log(request: Request, call_next: Any) -> Any:
        trace = request.headers.get("x-trace-id") or uuid.uuid4().hex[:16]
        token = trace_id_var.set(trace)
        request.state.trace_id = trace
        t0 = time.perf_counter()
        try:
            response = await call_next(request)
        finally:
            trace_id_var.reset(token)
        response.headers["X-Trace-Id"] = trace
        if request.url.path not in ("/healthz", "/readyz"):
            log.info(
                "request",
                extra={
                    "fields": {
                        "trace_id": trace,
                        "method": request.method,
                        "path": request.url.path,
                        "status": response.status_code,
                        "ms": round((time.perf_counter() - t0) * 1000, 2),
                    }
                },
            )
        return response

    @app.exception_handler(MedMemoryError)
    async def medmemory_error(request: Request, exc: MedMemoryError) -> JSONResponse:
        code = 400 if isinstance(exc, (ScopeViolationError, ConfigError)) else 502
        log.warning("handled error: %s", exc)
        return JSONResponse(
            {"detail": str(exc), "trace_id": getattr(request.state, "trace_id", None)},
            status_code=code,
        )

    GUI_HTML_PATH = Path(__file__).resolve().parent / "static" / "gui.html"

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    @app.get("/ui", response_class=HTMLResponse, include_in_schema=False)
    async def web_gui() -> HTMLResponse:
        if GUI_HTML_PATH.exists():
            return HTMLResponse(GUI_HTML_PATH.read_text(encoding="utf-8"))
        return HTMLResponse("<h1>MedMemory API Running</h1><p><a href='/docs'>Swagger UI</a></p>")

    # ------------------------------------------------------------------ health

    @app.get("/healthz", tags=["ops"])
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz", response_model=ReadyStatus, tags=["ops"])
    async def readyz(request: Request) -> ReadyStatus:
        c: Container | None = getattr(request.app.state, "container", None)
        if c is None:
            return ReadyStatus(ready=False, checks={"container": "starting"})
        checks: dict[str, str] = {}
        try:
            checks["kv"] = "ok" if c.records.patient_exists("P0001") else "empty"
        except Exception as exc:
            checks["kv"] = f"error: {exc}"
        try:
            n = await c.store.count(Namespace.DRUG_LABELS)
            checks["vector"] = "ok" if n > 0 else "empty"
        except Exception as exc:
            checks["vector"] = f"error: {exc}"
        checks["generator"] = c.generator.name
        return ReadyStatus(
            ready=all(v in ("ok",) for k, v in checks.items() if k != "generator"), checks=checks
        )

    @app.get("/v1/config", response_model=ComponentInfo, tags=["ops"])
    async def config(c: Container = Depends(get_container)) -> ComponentInfo:
        return c.info()

    @app.get("/v1/ingest", tags=["ops"])
    async def ingest_report(c: Container = Depends(get_container)) -> dict[str, Any]:
        manifest = c.settings.data_dir / "manifest.json"
        return {
            "report": c.ingest_report,
            "manifest": json.loads(manifest.read_text(encoding="utf-8"))
            if manifest.exists()
            else None,
        }

    # ------------------------------------------------------------------ query

    def audit(request: Request, user: User, resp: QueryResponse) -> None:
        a: AuditLog = request.app.state.audit
        a.write(
            trace_id=resp.trace_id,
            user=user.id,
            role=user.role,
            patient=a.h(resp.patient_scope),
            route=resp.route.value if resp.route else None,
            status=resp.status.value,
            cache=resp.cache_hit.value,
            sources=len(resp.citations),
            total_ms=resp.total_ms,
        )

    @app.post("/v1/query", response_model=QueryResponse, tags=["query"])
    async def query(
        req: QueryRequest,
        request: Request,
        user: User = Depends(rate_limited),
        c: Container = Depends(get_container),
    ) -> QueryResponse:
        resp = await c.orchestrator.run(req, request.state.trace_id)
        audit(request, user, resp)
        return resp

    @app.post(
        "/v1/query/stream",
        tags=["query"],
        response_class=StreamingResponse,
        responses={
            200: {
                "content": {"text/event-stream": {}},
                "description": "SSE: meta, retrieval, token…, final",
            }
        },
    )
    async def query_stream(
        req: QueryRequest,
        request: Request,
        user: User = Depends(rate_limited),
        c: Container = Depends(get_container),
    ) -> StreamingResponse:
        queue: asyncio.Queue[tuple[str, dict[str, Any]] | None] = asyncio.Queue()
        trace = request.state.trace_id

        async def emit(event: str, data: dict[str, Any]) -> None:
            await queue.put((event, data))

        async def runner() -> None:
            try:
                resp = await c.orchestrator.run(req, trace, emit)
                audit(request, user, resp)
                await queue.put(("final", resp.model_dump(mode="json")))
            except Exception as exc:  # surface errors as an SSE event, not a dropped stream
                log.exception("stream failed")
                await queue.put(("error", {"detail": str(exc), "trace_id": trace}))
            finally:
                await queue.put(None)

        async def events() -> AsyncIterator[bytes]:
            task = asyncio.create_task(runner())
            try:
                while True:
                    item = await queue.get()
                    if item is None:
                        break
                    if await request.is_disconnected():
                        task.cancel()
                        break
                    name, data = item
                    yield f"event: {name}\ndata: {json.dumps(data, default=str)}\n\n".encode()
            finally:
                if not task.done():
                    task.cancel()

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "X-Trace-Id": trace},
        )

    @app.get("/v1/scenarios", response_model=list[Scenario], tags=["query"])
    async def scenarios() -> list[Scenario]:
        return SCENARIOS

    # ------------------------------------------------------------------ KV views

    @app.get("/v1/patients", response_model=list[PatientSummary], tags=["records"])
    async def patients(
        q: str = "", limit: int = Query(100, ge=1, le=500), c: Container = Depends(get_container)
    ) -> list[PatientSummary]:
        return c.records.list_patients(q, limit)

    @app.get("/v1/patients/{pid}", response_model=PatientRecord, tags=["records"])
    async def patient(
        pid: str, c: Container = Depends(get_container), user: User = Depends(rate_limited)
    ) -> PatientRecord:
        if not c.records.patient_exists(pid):
            raise HTTPException(404, f"patient {pid} not found")
        summary = next(s for s in c.records.list_patients(pid, 500) if s.patient_id == pid)
        recs = c.records.patient_records(pid)
        notes = [
            {"doc_id": d["doc_id"], "title": d["title"], "date": d["date"]}
            for d in (
                c.records.document(k.removeprefix("doc:")) for k, _ in c.kv.scan(f"doc:note:{pid}:")
            )
            if d
        ]
        request_audit: AuditLog = app.state.audit
        request_audit.write(
            trace_id=trace_id_var.get(),
            user=user.id,
            action="view_patient",
            patient=request_audit.h(pid),
        )
        return PatientRecord(
            summary=summary,
            records=recs,
            notes=sorted(notes, key=lambda n: n["date"], reverse=True),
        )

    @app.post("/v1/patients/{pid}/labs", response_model=WriteResult, tags=["records"])
    async def add_lab(
        pid: str,
        body: LabWrite,
        c: Container = Depends(get_container),
        user: User = Depends(rate_limited),
    ) -> WriteResult:
        """Write path: KV put, then write-through invalidation of every cached answer tagged
        with this patient (and the data version bump orphans any stragglers)."""
        try:
            key, version = c.records.add_lab(pid, body.lab, body.value, body.unit, body.date)
        except KeyError:
            raise HTTPException(404, f"patient {pid} not found") from None
        dropped = c.exact_cache.invalidate_tag(f"patient:{pid}")
        app.state.audit.write(
            trace_id=trace_id_var.get(),
            user=user.id,
            action="write_lab",
            patient=app.state.audit.h(pid),
        )
        return WriteResult(key=key, invalidated_cache_entries=dropped, data_version=version)

    @app.get("/v1/kv", response_model=list[KVRecord], tags=["records"])
    async def kv_scan(
        prefix: str = Query(..., min_length=3),
        limit: int = Query(100, ge=1, le=500),
        c: Container = Depends(get_container),
    ) -> list[KVRecord]:
        from medmemory.kv.records import to_record

        if prefix.startswith(("meta:", "doc:")):
            raise HTTPException(400, "internal prefixes are not browsable")
        return [to_record(k, v) for k, v in c.kv.scan(prefix, limit)]

    @app.get("/v1/kv/{key:path}", response_model=KVRecord, tags=["records"])
    async def kv_get(key: str, c: Container = Depends(get_container)) -> KVRecord:
        rec = c.records.get(key)
        if rec is None:
            raise HTTPException(404, f"no live key {key}")
        return rec

    # ------------------------------------------------------------------ vector view

    @app.post("/v1/search", response_model=SearchResponse, tags=["vector"])
    async def search(
        req: SearchRequest, c: Container = Depends(get_container), _: User = Depends(rate_limited)
    ) -> SearchResponse:
        if req.namespace == Namespace.PATIENT_NOTES and not req.patient_scope:
            raise HTTPException(400, "searching patient_notes requires patient_scope")
        timer = StageTimer()
        res = await c.retriever.retrieve(
            req.query,
            [req.namespace],
            req.patient_scope,
            timer,
            top_k=req.top_k,
            use_reranker=req.rerank,
            hybrid_sparse=req.hybrid_sparse,
            extra_filter=and_filters(dict(req.filters) if req.filters else None),
        )
        return SearchResponse(
            chunks=res.chunks,
            timings=timer.stages,
            embedder=c.embedder.name,
            store=c.store.name,
            reranker=c.reranker.name if req.rerank else None,
        )

    @app.get("/v1/sources/{source_id:path}", response_model=SourceView, tags=["vector"])
    async def source(source_id: str, c: Container = Depends(get_container)) -> SourceView:
        if source_id.startswith("kv:"):
            rec = c.records.get(source_id.removeprefix("kv:"))
            if rec is None:
                raise HTTPException(404, "source not found")
            return SourceView(
                source_id=source_id,
                origin=Origin.KV,
                title=rec.title,
                text=str(rec.data.get("text", "")),
                metadata={k: v for k, v in rec.data.items() if k != "text"},
            )
        if source_id.startswith("vec:"):
            chunk_id = source_id.removeprefix("vec:")
            doc_id = chunk_id.split("#")[0]
            ns = {
                "note": Namespace.PATIENT_NOTES,
                "label": Namespace.DRUG_LABELS,
                "topic": Namespace.GUIDELINES,
            }.get(doc_id.split(":")[0])
            doc = c.records.document(doc_id)
            if ns is None or doc is None:
                raise HTTPException(404, "source not found")
            chunks = await c.store.fetch(ns, [chunk_id])
            meta = {k: v for k, v in doc.items() if k not in ("text", "title")}
            if chunks:
                meta.update(section=chunks[0].section, chunk_text=chunks[0].text)
            return SourceView(
                source_id=source_id,
                origin=Origin.VECTOR,
                title=doc["title"],
                text=doc["text"],
                metadata=meta,
            )
        raise HTTPException(400, "source ids start with kv: or vec:")

    # ------------------------------------------------------------------ dashboards

    @app.get("/v1/metrics", tags=["ops"])
    async def metrics(c: Container = Depends(get_container)) -> dict[str, Any]:
        snap = c.metrics.snapshot()
        snap["caches"] = {
            "exact": c.exact_cache.stats().as_dict(),
            "singleflight": c.orchestrator.singleflight.stats(),
        }
        snap["kv"] = {
            "backend": c.kv.name,
            "keys": c.kv.count(""),
            "patients": c.kv.count("idx:patient:"),
        }
        snap["vector"] = {ns.value: await c.store.count(ns) for ns in Namespace}
        return snap

    def _read_result(rel: str) -> dict[str, Any]:
        path = ROOT / rel
        if not path.exists():
            raise HTTPException(
                404, f"{rel} not found. Run the corresponding `python tasks.py` target first."
            )
        data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        return data

    @app.get("/v1/eval/latest", tags=["ops"])
    async def eval_latest() -> dict[str, Any]:
        return _read_result("eval/results/latest.json")

    @app.get("/v1/bench/latest", tags=["ops"])
    async def bench_latest() -> dict[str, Any]:
        return _read_result("bench/results/latest.json")

    @app.get("/v1/audit", tags=["ops"])
    async def audit_tail(n: int = Query(50, ge=1, le=500)) -> list[dict[str, Any]]:
        a: AuditLog = app.state.audit
        return a.tail(n)

    @app.get("/", include_in_schema=False)
    async def root() -> dict[str, str]:
        return {"name": "MedMemory API", "docs": "/docs", "disclaimer": DISCLAIMER}

    return app
