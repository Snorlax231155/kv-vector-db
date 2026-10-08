"""End-to-end lifecycle tests in mock mode."""

from __future__ import annotations

import asyncio
from datetime import date

from medmemory.container import Container
from medmemory.contracts.schemas import (
    AnswerStatus,
    CacheResult,
    Origin,
    QueryOptions,
    QueryRequest,
    Route,
)

EXPECTED_MISS_STAGES = {
    "safety",
    "route",
    "scope",
    "cache_exact",
    "kv",
    "merge",
    "sufficiency",
    "generate",
    "citation_check",
    "cache_write",
}


async def ask(c: Container, q: str, scope: str | None = None, **opts: object) -> object:
    return await c.orchestrator.answer(
        QueryRequest(query=q, patient_scope=scope, options=QueryOptions(**opts))
    )  # type: ignore[arg-type]


async def test_kv_route_answer_is_cited_and_timed(container: Container) -> None:
    r = await container.orchestrator.answer(
        QueryRequest(query="What is the latest HbA1c for P0001?")
    )
    assert r.status == AnswerStatus.ANSWERED and r.route == Route.KV
    assert r.kv_records and r.kv_records[0].key.startswith("patient:P0001:lab:hemoglobin_a1c:")
    assert r.citations and r.evidence_origin == [Origin.KV]
    assert r.citation_check and r.citation_check.passed
    assert {t.stage for t in r.timings} >= EXPECTED_MISS_STAGES
    assert r.total_ms >= max(t.start_ms for t in r.timings)
    assert r.disclaimer == "Educational prototype. Not medical advice."


async def test_hybrid_runs_kv_and_vector_and_cites_both(container: Container) -> None:
    r = await container.orchestrator.answer(
        QueryRequest(
            query="Given her latest eGFR, what does the metformin label say about kidney function?",
            patient_scope="P0001",
        )
    )
    assert r.route == Route.HYBRID and r.status == AnswerStatus.ANSWERED
    assert r.kv_records and r.vector_chunks
    assert set(r.evidence_origin) == {Origin.KV, Origin.VECTOR}
    stages = {t.stage: t for t in r.timings}
    assert "kv" in stages and "vector_search" in stages


async def test_vector_route_uses_drug_filter(container: Container) -> None:
    r = await container.orchestrator.answer(
        QueryRequest(query="What are the warnings for apixaban?")
    )
    assert r.route == Route.VECTOR
    labels = [c for c in r.vector_chunks if c.chunk.metadata.get("doc_type") == "drug_label"]
    assert labels and all(c.chunk.metadata["drug"] == "apixaban" for c in labels)


async def test_exact_cache_hit_and_miss(container: Container) -> None:
    first = await container.orchestrator.answer(
        QueryRequest(query="What is the latest HbA1c for P0001?")
    )
    second = await container.orchestrator.answer(
        QueryRequest(query="What is the latest HbA1c for P0001?")
    )
    assert (first.cache_hit, second.cache_hit) == (
        CacheResult.MISS,
        CacheResult.EXACT,
    )
    assert second.answer == first.answer and second.trace_id != first.trace_id
    assert "generate" not in {t.stage for t in second.timings}


async def test_cache_disabled_option(container: Container) -> None:
    await container.orchestrator.answer(QueryRequest(query="What is the latest HbA1c for P0001?"))
    r = await container.orchestrator.answer(
        QueryRequest(
            query="What is the latest HbA1c for P0001?", options=QueryOptions(use_cache=False)
        )
    )
    assert r.cache_hit == CacheResult.BYPASS


async def test_single_flight_coalesces_identical_concurrent_misses(container: Container) -> None:
    calls = 0
    real = container.generator.generate

    async def slow_generate(*a: object, **k: object) -> str:
        nonlocal calls
        calls += 1
        await asyncio.sleep(0.05)
        return await real(*a, **k)  # type: ignore[arg-type]

    container.generator.generate = slow_generate  # type: ignore[method-assign]
    try:
        rs = await asyncio.gather(
            *(
                container.orchestrator.answer(
                    QueryRequest(query="What is the latest LDL for P0005?")
                )
                for _ in range(10)
            )
        )
    finally:
        container.generator.generate = real  # type: ignore[method-assign]
    assert calls == 1
    assert len({r.answer for r in rs}) == 1 and len({r.trace_id for r in rs}) == 10


async def test_conflicting_sources_abstain_with_both_named(container: Container) -> None:
    r = await container.orchestrator.answer(
        QueryRequest(
            query="Is her levothyroxine dose in the notes consistent with her medication record?",
            patient_scope="P0004",
        )
    )
    assert r.status == AnswerStatus.ABSTAINED
    assert (
        r.sufficiency and r.sufficiency.conflicts and "75 mcg" in r.answer and "50 mcg" in r.answer
    )
    assert any(t.stage == "generate" and t.status == "skipped" for t in r.timings)


async def test_unknown_drug_abstains(container: Container) -> None:
    r = await container.orchestrator.answer(
        QueryRequest(query="What is the recommended dose of semaglutide?")
    )
    assert r.status == AnswerStatus.ABSTAINED and "enough information" in r.answer


async def test_red_flag_skips_everything(container: Container) -> None:
    r = await container.orchestrator.answer(
        QueryRequest(query="I have crushing chest pain spreading to my left arm")
    )
    assert r.status == AnswerStatus.RED_FLAG and r.cache_hit == CacheResult.BYPASS
    assert [t.stage for t in r.timings] == ["safety"] and r.route is None


async def test_write_invalidates_cached_answer(fresh_container: Container) -> None:
    c = fresh_container
    q = QueryRequest(query="What is the latest HbA1c for P0001?")
    before = await c.orchestrator.answer(q)
    assert (await c.orchestrator.answer(q)).cache_hit == CacheResult.EXACT
    c.records.add_lab("P0001", "hemoglobin a1c", 6.4, "%", date(2026, 10, 5))
    dropped = c.exact_cache.invalidate_tag("patient:P0001")
    after = await c.orchestrator.answer(q)
    assert dropped >= 1 and after.cache_hit == CacheResult.MISS
    assert "6.4" in after.answer and after.answer != before.answer


async def test_stream_emits_events_and_final_matches(container: Container) -> None:
    events: list[tuple[str, dict]] = []

    async def emit(name: str, data: dict) -> None:
        events.append((name, data))

    req = QueryRequest(query="What did the notes say about ankle swelling?", patient_scope="P0002")
    final = await container.orchestrator.run(req, "trace-test", emit)
    names = [n for n, _ in events]
    assert names[0] == "meta" and names[1] == "retrieval" and names.count("token") > 3
    streamed = "".join(d["text"] for n, d in events if n == "token")
    assert final.status == AnswerStatus.ANSWERED and set(final.answer.split()) <= set(
        streamed.split()
    )


async def test_stream_meta_always_first(container: Container) -> None:
    events: dict[str, list[str]] = {}

    def recorder(q: str):  # type: ignore[no-untyped-def]
        async def emit(name: str, data: dict) -> None:
            events.setdefault(q, []).append(name)

        return emit

    for q in (
        "I have crushing chest pain spreading to my left arm",
        "What medications is P0002 taking?",
    ):
        await container.orchestrator.run(
            QueryRequest(query=q, patient_scope="P0001"), "t", recorder(q)
        )
        assert events[q] and events[q][0] == "meta"
