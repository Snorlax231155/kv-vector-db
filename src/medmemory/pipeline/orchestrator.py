"""The request lifecycle (docs/architecture.md, sequence diagram).

safety → route → scope → cache (exact, then scoped semantic) → retrieval (KV ∥ vector)
→ merge → sufficiency → generate → citation check → cache write → response

`run()` is the single implementation. The streaming endpoint passes an `emit` callback
that receives `meta`, `retrieval` and `token` events; the non-streaming endpoint passes none
and gets single-flight coalescing of identical concurrent misses.
"""

from __future__ import annotations

import asyncio
import logging
import re
import uuid
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any

from medmemory.cache import LRUCache, SingleFlight, exact_key
from medmemory.contracts.protocols import FloatArray, Router
from medmemory.contracts.schemas import (
    AnswerStatus,
    CacheResult,
    Evidence,
    EvidenceSufficiency,
    KVRecord,
    Namespace,
    Origin,
    QueryRequest,
    QueryResponse,
    Route,
    RouterDecision,
    SafetyResult,
    ScoredChunk,
)
from medmemory.generation import INSUFFICIENT, LangChainGenerator
from medmemory.kv.records import RecordRepository
from medmemory.pipeline.metrics import Metrics
from medmemory.pipeline.planning import namespace_filters, plan_namespaces
from medmemory.pipeline.timing import StageTimer
from medmemory.safety import assess_sufficiency, check_citations, precheck, resolve_scope
from medmemory.safety.redflags import OUT_OF_SCOPE_MESSAGE, PATIENT_REF, looks_clinical
from medmemory.vector.retriever import RetrievalResult, ScopedRetriever

log = logging.getLogger(__name__)

Emit = Callable[[str, dict[str, Any]], Awaitable[None]]
MIN_GROUNDED_FRACTION = 0.6
MAX_EVIDENCE = 10


async def _no_emit(event: str, data: dict[str, Any]) -> None:
    return None


@dataclass
class PipelineConfig:
    min_evidence_score: float
    cache_ttl_s: float


class Orchestrator:
    def __init__(
        self,
        *,
        records: RecordRepository,
        retriever: ScopedRetriever,
        routers: dict[str, Router],
        generator: LangChainGenerator,
        exact_cache: LRUCache[QueryResponse],
        metrics: Metrics,
        config: PipelineConfig,
    ) -> None:
        self.records = records
        self.retriever = retriever
        self.routers = routers
        self.generator = generator
        self.exact_cache = exact_cache
        self.singleflight: SingleFlight[QueryResponse] = SingleFlight()
        self.metrics = metrics
        self.config = config
        self.vocabulary: frozenset[str] | None = None  # set by ingestion

    # ------------------------------------------------------------------ public

    async def answer(self, req: QueryRequest, trace_id: str | None = None) -> QueryResponse:
        return await self.run(req, trace_id or uuid.uuid4().hex[:16])

    async def run(
        self, req: QueryRequest, trace_id: str, emit: Emit | None = None
    ) -> QueryResponse:
        timer = StageTimer()
        streaming = emit is not None
        emit = emit or _no_emit
        opts = req.options

        # 1. safety pre-check: red flags only. Out-of-scope is decided after routing, when we
        # know whether the query mentions any recognised entity (brand names, typos, codes).
        with timer.stage("safety") as m:
            safety = precheck(req.query, clinical_context=True)
            m["verdict"] = safety.verdict
        if safety.verdict != "ok":
            status = (
                AnswerStatus.RED_FLAG if safety.verdict == "red_flag" else AnswerStatus.OUT_OF_SCOPE
            )
            if safety.verdict == "red_flag":
                self.metrics.incr(f"red_flag:{safety.category}")
            await emit(
                "meta",
                {
                    "trace_id": trace_id,
                    "route": None,
                    "confidence": None,
                    "router": None,
                    "patient_scope": req.patient_scope,
                    "safety": safety.model_dump(mode="json"),
                },
            )
            return self._finish(
                trace_id,
                timer,
                status,
                safety.message or "",
                safety=safety,
                cache=CacheResult.BYPASS,
            )

        # 2. route
        router = self.routers.get(opts.router) or self.routers["auto"]
        with timer.stage("route", router=router.name) as m:
            decision = await asyncio.to_thread(router.route, req.query, req.patient_scope)
            m.update(
                route=decision.route.value,
                confidence=decision.confidence,
                decided_by=decision.router,
            )
        e = decision.entities
        recognised = any(
            (
                e.patient_ids,
                e.icd10_codes,
                e.rxnorm_codes,
                e.drugs,
                e.labs,
                e.conditions,
                e.record_types,
            )
        )
        about_patient = req.patient_scope is not None and PATIENT_REF.search(req.query) is not None
        if not recognised and not about_patient and not looks_clinical(req.query):
            oos = SafetyResult(
                verdict="out_of_scope", category="non_clinical", message=OUT_OF_SCOPE_MESSAGE
            )
            await emit(
                "meta",
                {
                    "trace_id": trace_id,
                    "route": decision.route.value,
                    "confidence": decision.confidence,
                    "router": decision.model_dump(mode="json"),
                    "patient_scope": req.patient_scope,
                    "safety": oos.model_dump(mode="json"),
                },
            )
            return self._finish(
                trace_id,
                timer,
                AnswerStatus.OUT_OF_SCOPE,
                OUT_OF_SCOPE_MESSAGE,
                safety=oos,
                decision=decision,
                cache=CacheResult.BYPASS,
                scope=req.patient_scope,
            )

        # 3. scope
        with timer.stage("scope") as m:
            sd = resolve_scope(
                req.patient_scope,
                [*decision.entities.patient_ids, *self.records.patients_named(req.query)],
                self.records.patient_exists,
            )
            m.update(scope=sd.scope, violation=bool(sd.violation))
        if sd.violation:
            self.metrics.incr("scope_violation_refused")
            await emit(
                "meta",
                {
                    "trace_id": trace_id,
                    "route": decision.route.value,
                    "confidence": decision.confidence,
                    "router": decision.model_dump(mode="json"),
                    "patient_scope": req.patient_scope,
                    "safety": safety.model_dump(mode="json"),
                },
            )
            return self._finish(
                trace_id,
                timer,
                AnswerStatus.SCOPE_VIOLATION,
                sd.violation,
                safety=safety,
                decision=decision,
                cache=CacheResult.BYPASS,
                scope=req.patient_scope,
            )
        scope = sd.scope
        await emit(
            "meta",
            {
                "trace_id": trace_id,
                "route": decision.route.value,
                "confidence": decision.confidence,
                "router": decision.model_dump(mode="json"),
                "patient_scope": scope,
                "safety": safety.model_dump(mode="json"),
            },
        )

        # 4. cache
        namespaces = plan_namespaces(decision, scope, req)
        version = self.records.data_version(scope)
        answer_opts = {
            "rerank": opts.use_reranker,
            "hybrid": opts.hybrid_sparse,
            "k": opts.top_k,
            "router": opts.router,
        }
        key = exact_key(
            decision.normalized_query,
            scope,
            req.filters,
            [n.value for n in namespaces],
            answer_opts,
            version,
        )
        tags = [f"patient:{scope}"] if scope else ["global"]
        if opts.use_cache:
            with timer.stage("cache_exact", key=key[:28]) as m:
                cached = self.exact_cache.get(key)
                m["hit"] = cached is not None
            if cached is not None:
                return await self._replay(
                    cached, trace_id, timer, CacheResult.EXACT, emit, streaming
                )
        else:
            timer.skip("cache_exact", "disabled by request")

        qvec: FloatArray | None = None
        if decision.route != Route.KV:
            with timer.stage("embed", embedder=self.retriever.embedder.name):
                qvec = await asyncio.to_thread(self.retriever.embed, decision.normalized_query)

        async def compute() -> QueryResponse:
            return await self._compute(
                req,
                trace_id,
                timer,
                decision,
                scope,
                namespaces,
                qvec,
                safety,
                emit,
                streaming,
                key,
                tags,
            )

        if streaming or not opts.use_cache:
            return await compute()
        resp, shared = await self.singleflight.do(key, compute)
        if shared:
            self.metrics.incr("singleflight_coalesced")
            return resp.model_copy(update={"trace_id": trace_id, "cache_hit": CacheResult.EXACT})
        return resp

    # ------------------------------------------------------------------ internals

    async def _compute(
        self,
        req: QueryRequest,
        trace_id: str,
        timer: StageTimer,
        decision: RouterDecision,
        scope: str | None,
        namespaces: list[Any],
        qvec: FloatArray | None,
        safety: SafetyResult,
        emit: Emit,
        streaming: bool,
        key: str,
        tags: list[str],
    ) -> QueryResponse:
        opts = req.options
        route = decision.route
        miss = CacheResult.MISS if opts.use_cache else CacheResult.BYPASS

        async def kv_side() -> list[KVRecord]:
            with timer.stage("kv") as m:
                recs = await asyncio.to_thread(self.records.lookup, decision.entities, scope)
                m["records"] = len(recs)
            return recs

        async def vector_side() -> RetrievalResult:
            return await self.retriever.retrieve(
                decision.normalized_query,
                namespaces,
                scope,
                timer,
                query_vector=qvec,
                namespace_filters=namespace_filters(decision),
                extra_filter=dict(req.filters) if req.filters else None,
                top_k=opts.top_k,
                use_reranker=opts.use_reranker,
                hybrid_sparse=opts.hybrid_sparse,
            )

        kv_records: list[KVRecord] = []
        retrieval = RetrievalResult(chunks=[])
        if route == Route.HYBRID:
            kv_records, retrieval = await asyncio.gather(kv_side(), vector_side())
        elif route == Route.KV:
            kv_records = await kv_side()
            if not kv_records:  # exact miss: widen to semantic search rather than give up
                decision.reasons.append("no KV match; widened to vector search")
                retrieval = await vector_side()
            elif (
                scope
                and decision.entities.dosage_intent
                and any(r.kind == "medication" for r in kv_records)
            ):
                # Dose answers are cross-checked against the latest documented medication list,
                # so a structured record that disagrees with the chart abstains instead of answering.
                decision.reasons.append("dose cross-checked against documented medication lists")
                retrieval = await self.retriever.retrieve(
                    decision.normalized_query,
                    [Namespace.PATIENT_NOTES],
                    scope,
                    timer,
                    query_vector=qvec,
                    extra_filter={"section": "Current Medications"},
                    top_k=3,
                    use_reranker=False,
                )
        else:
            retrieval = await vector_side()
        if retrieval.blocked_out_of_scope:
            self.metrics.incr("scope_leak_blocked", retrieval.blocked_out_of_scope)
        chunks = retrieval.chunks
        await emit(
            "retrieval",
            {
                "kv_records": [r.model_dump(mode="json") for r in kv_records],
                "vector_chunks": [c.model_dump(mode="json") for c in chunks],
                "namespaces": retrieval.searched,
            },
        )

        with timer.stage("merge") as m:
            evidence = build_evidence(kv_records, chunks)
            m.update(
                kv=sum(e.origin == Origin.KV for e in evidence),
                vector=sum(e.origin == Origin.VECTOR for e in evidence),
            )

        with timer.stage("sufficiency") as m:
            suff = assess_sufficiency(
                route,
                decision.entities,
                kv_records,
                chunks,
                self.config.min_evidence_score,
                query=decision.normalized_query,
                scope=scope,
                vocabulary=self.vocabulary,
            )
            m.update(sufficient=suff.sufficient, reason=suff.reason, max_score=suff.max_score)

        common: dict[str, Any] = {
            "safety": safety,
            "decision": decision,
            "scope": scope,
            "kv_records": kv_records,
            "chunks": chunks,
            "evidence": evidence,
            "sufficiency": suff,
        }
        if not suff.sufficient:
            timer.skip("generate", "insufficient evidence")
            resp = self._finish(
                trace_id,
                timer,
                AnswerStatus.ABSTAINED,
                abstain_text(suff),
                cache=miss,
                **common,
            )
            self._store(key, tags, resp, timer, opts.use_cache)
            return resp

        with timer.stage("generate", generator=self.generator.name, evidence=len(evidence)) as m:
            if streaming:
                parts: list[str] = []
                async for piece in self.generator.stream(req.query, evidence, route):
                    parts.append(piece)
                    await emit("token", {"text": piece})
                raw = "".join(parts)
            else:
                raw = await self.generator.generate(req.query, evidence, route)
            m["chars"] = len(raw)

        if raw.strip().startswith(INSUFFICIENT):
            suff = EvidenceSufficiency(
                sufficient=False,
                reason="The model judged the retrieved sources insufficient.",
                max_score=suff.max_score,
            )
            common["sufficiency"] = suff
            resp = self._finish(
                trace_id,
                timer,
                AnswerStatus.ABSTAINED,
                abstain_text(suff),
                cache=miss,
                **common,
            )
            self._store(key, tags, resp, timer, opts.use_cache)
            return resp

        with timer.stage("citation_check") as m:
            checked = check_citations(raw, evidence)
            m.update(
                grounding_rate=checked.check.grounding_rate,
                removed=len(checked.check.removed_sentences),
            )
        grounded_ok = (
            checked.check.supported_sentences > 0
            and checked.check.grounding_rate >= MIN_GROUNDED_FRACTION
        )
        if not grounded_ok:
            suff = EvidenceSufficiency(
                sufficient=False,
                reason="The draft answer could not be grounded in the retrieved sources.",
                max_score=suff.max_score,
            )
            common["sufficiency"] = suff
            resp = self._finish(
                trace_id,
                timer,
                AnswerStatus.ABSTAINED,
                abstain_text(suff),
                cache=miss,
                citation_check=checked.check,
                **common,
            )
            self._store(key, tags, resp, timer, opts.use_cache)
            return resp

        resp = self._finish(
            trace_id,
            timer,
            AnswerStatus.ANSWERED,
            checked.text,
            cache=miss,
            citation_check=checked.check,
            citations=checked.citations,
            **common,
        )
        self._store(key, tags, resp, timer, opts.use_cache)
        return resp

    def _store(
        self,
        key: str,
        tags: list[str],
        resp: QueryResponse,
        timer: StageTimer,
        use_cache: bool,
    ) -> None:
        if not use_cache:
            return
        with timer.stage("cache_write") as m:
            self.exact_cache.set(key, resp, ttl_s=self.config.cache_ttl_s, tags=tags)
            m.update(exact=True)
        # the stored response's timing list was captured before cache_write; refresh ours
        resp.timings = list(timer.stages)
        resp.latency_ms = timer.latency_map()
        resp.total_ms = timer.total_ms()

    async def _replay(
        self,
        cached: QueryResponse,
        trace_id: str,
        timer: StageTimer,
        how: CacheResult,
        emit: Emit,
        streaming: bool,
        extra: dict[str, Any] | None = None,
    ) -> QueryResponse:
        await emit(
            "meta",
            {
                "trace_id": trace_id,
                "route": cached.route.value if cached.route else None,
                "confidence": cached.confidence,
                "router": cached.router.model_dump(mode="json") if cached.router else None,
                "patient_scope": cached.patient_scope,
                "safety": cached.safety.model_dump(mode="json"),
                "cache_hit": how.value,
                **(extra or {}),
            },
        )
        await emit(
            "retrieval",
            {
                "kv_records": [r.model_dump(mode="json") for r in cached.kv_records],
                "vector_chunks": [c.model_dump(mode="json") for c in cached.vector_chunks],
                "namespaces": [],
            },
        )
        if streaming:
            for piece in re.findall(r"\S+\s*", cached.answer):
                await emit("token", {"text": piece})
        resp = cached.model_copy(
            update={
                "trace_id": trace_id,
                "cache_hit": how,
                "timings": list(timer.stages),
                "latency_ms": timer.latency_map(),
                "total_ms": timer.total_ms(),
            }
        )
        self.metrics.record(resp)
        return resp

    def _finish(
        self,
        trace_id: str,
        timer: StageTimer,
        status: AnswerStatus,
        answer: str,
        *,
        safety: SafetyResult,
        cache: CacheResult,
        decision: RouterDecision | None = None,
        scope: str | None = None,
        kv_records: Sequence[KVRecord] = (),
        chunks: Sequence[ScoredChunk] = (),
        evidence: Sequence[Evidence] = (),
        sufficiency: EvidenceSufficiency | None = None,
        citation_check: Any = None,
        citations: Sequence[Any] = (),
    ) -> QueryResponse:
        origins: list[Origin] = []
        for c in citations:
            if c.origin not in origins:
                origins.append(c.origin)
        resp = QueryResponse(
            trace_id=trace_id,
            status=status,
            answer=answer,
            route=decision.route if decision else None,
            confidence=decision.confidence if decision else None,
            entities=decision.entities if decision else None,
            router=decision,
            cache_hit=cache,
            latency_ms=timer.latency_map(),
            total_ms=timer.total_ms(),
            timings=list(timer.stages),
            kv_records=list(kv_records),
            vector_chunks=list(chunks),
            evidence=list(evidence),
            citations=list(citations),
            citation_check=citation_check,
            sufficiency=sufficiency,
            evidence_origin=origins,
            safety=safety,
            patient_scope=scope,
            generator=self.generator.name
            if status in (AnswerStatus.ANSWERED, AnswerStatus.ABSTAINED)
            else "",
        )
        self.metrics.record(resp)
        return resp


def build_evidence(kv_records: Sequence[KVRecord], chunks: Sequence[ScoredChunk]) -> list[Evidence]:
    """KV facts first (exact, highest trust), then passages by rank; numbered S1..Sn."""
    out: list[Evidence] = []
    for r in kv_records:
        out.append(
            Evidence(
                key="",
                source_id=r.source_id,
                origin=Origin.KV,
                title=r.title,
                text=str(r.data.get("text", "")),
                score=None,
                patient_id=r.patient_id,
            )
        )
    for c in chunks:
        out.append(
            Evidence(
                key="",
                source_id=c.chunk.source_id,
                origin=Origin.VECTOR,
                title=f"{c.chunk.title} · {c.chunk.section}",
                text=c.chunk.text,
                score=round(c.rerank_score if c.rerank_score is not None else c.score, 4),
                patient_id=c.chunk.patient_id,
                namespace=c.chunk.namespace,
            )
        )
    out = out[:MAX_EVIDENCE]
    return [e.model_copy(update={"key": f"S{i + 1}"}) for i, e in enumerate(out)]


def abstain_text(s: EvidenceSufficiency) -> str:
    text = f"I don't have enough information to answer that reliably. {s.reason}"
    if s.conflicts:
        text += (
            " Conflicting sources: "
            + "; ".join(s.conflicts)
            + ". Please verify against the original record."
        )
    return text
