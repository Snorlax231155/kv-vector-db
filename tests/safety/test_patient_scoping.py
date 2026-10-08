"""Patient scoping: a query about patient A must never surface patient B's data, in
retrieval, cache, or logs. Each defence layer is tested on its own, plus end to end."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from medmemory.api.observability import AuditLog
from medmemory.container import Container
from medmemory.contracts.schemas import (
    AnswerStatus,
    CacheResult,
    Chunk,
    Entities,
    Namespace,
    QueryRequest,
)
from medmemory.errors import ScopeViolationError
from medmemory.pipeline.timing import StageTimer
from medmemory.vector.retriever import ScopedRetriever
from medmemory.vector.stores.memory import InMemoryVectorStore

ALL = [f"P{i:04d}" for i in range(1, 51)]


# ---- layer 1: request scope


async def test_query_naming_other_patient_is_refused_before_retrieval(container: Container) -> None:
    r = await container.orchestrator.answer(
        QueryRequest(query="What medications is P0002 taking?", patient_scope="P0001")
    )
    assert r.status == AnswerStatus.SCOPE_VIOLATION
    assert not r.kv_records and not r.vector_chunks
    assert [t.stage for t in r.timings] == ["safety", "route", "scope"]


async def test_query_naming_two_patients_is_refused(container: Container) -> None:
    r = await container.orchestrator.answer(QueryRequest(query="Compare P0001 and P0002 HbA1c"))
    assert r.status == AnswerStatus.SCOPE_VIOLATION


async def test_naming_another_patient_by_name_is_refused(container: Container) -> None:
    other = container.records.list_patients("P0002", 1)[0].name
    r = await container.orchestrator.answer(
        QueryRequest(query=f"What is {other} taking these days?", patient_scope="P0001")
    )
    assert r.status == AnswerStatus.SCOPE_VIOLATION and "P0002" in r.answer
    own = container.records.list_patients("P0001", 1)[0].name
    ok = await container.orchestrator.answer(
        QueryRequest(query=f"What is {own} taking?", patient_scope="P0001")
    )
    assert ok.status != AnswerStatus.SCOPE_VIOLATION


# ---- layer 2: KV


def test_kv_lookup_only_reads_scope_prefix(container: Container) -> None:
    recs = container.records.lookup(
        Entities(record_types=["medications", "labs", "allergies"]), "P0001"
    )
    assert recs and all(r.patient_id == "P0001" for r in recs)
    assert all(r.key.startswith("patient:P0001:") or r.key.startswith("drug:") for r in recs)


def test_kv_rejects_malformed_scope(container: Container) -> None:
    with pytest.raises(ScopeViolationError):
        container.records.patient_records("P0001:lab:*")


# ---- layer 3: vector


async def test_patient_notes_always_filtered_to_scope(container: Container) -> None:
    for pid in ("P0001", "P0004", "P0017"):
        res = await container.retriever.retrieve(
            "swelling cough pain medications",
            [Namespace.PATIENT_NOTES],
            pid,
            StageTimer(),
            top_k=10,
        )
        assert res.chunks and all(c.chunk.patient_id == pid for c in res.chunks)


async def test_patient_notes_not_searched_without_scope(container: Container) -> None:
    res = await container.retriever.retrieve(
        "swelling", [Namespace.PATIENT_NOTES], None, StageTimer()
    )
    assert res.chunks == [] and res.skipped


class LeakyStore(InMemoryVectorStore):
    """Simulates a misconfigured adapter that ignores the metadata filter."""

    async def query(self, namespace, vector, top_k, filter=None):  # type: ignore[no-untyped-def,override]
        return self.query_sync(namespace, vector, top_k, None)


async def test_post_filter_blocks_leaks_from_a_buggy_store(
    container: Container, caplog: pytest.LogCaptureFixture
) -> None:
    emb = container.embedder
    store = LeakyStore(emb.dim)
    chunks = [
        Chunk(
            chunk_id=f"n{p}#0",
            doc_id=f"n{p}",
            namespace=Namespace.PATIENT_NOTES,
            title="t",
            section="s",
            text="reports ankle swelling",
            patient_id=p,
        )
        for p in ("P0001", "P0002", "P0003")
    ]
    store.upsert_sync(
        Namespace.PATIENT_NOTES, chunks, emb.embed_documents([c.text for c in chunks])
    )
    retriever = ScopedRetriever(emb, store, container.reranker)
    with caplog.at_level(logging.ERROR, logger="medmemory.security"):
        res = await retriever.retrieve(
            "ankle swelling", [Namespace.PATIENT_NOTES], "P0001", StageTimer(), top_k=5
        )
    assert [c.chunk.patient_id for c in res.chunks] == ["P0001"]
    assert res.blocked_out_of_scope == 2
    assert "scope_leak_blocked" in caplog.text


# ---- layer 4: cache


async def test_cached_answer_for_one_patient_never_served_to_another(container: Container) -> None:
    q = "What is her latest HbA1c?"
    a = await container.orchestrator.answer(QueryRequest(query=q, patient_scope="P0001"))
    b = await container.orchestrator.answer(QueryRequest(query=q, patient_scope="P0006"))
    assert a.cache_hit == CacheResult.MISS and b.cache_hit == CacheResult.MISS
    assert all(r.patient_id == "P0006" for r in b.kv_records)
    assert "P0001" not in b.answer
    again = await container.orchestrator.answer(QueryRequest(query=q, patient_scope="P0001"))
    assert again.cache_hit == CacheResult.EXACT and again.answer == a.answer


# ---- end to end: sweep every patient


@pytest.mark.slow
async def test_no_cross_patient_evidence_for_any_patient(container: Container) -> None:
    for pid in ALL:
        r = await container.orchestrator.answer(
            QueryRequest(query="Summarize recent notes, medications and labs", patient_scope=pid)
        )
        foreign = {e.patient_id for e in r.evidence if e.patient_id and e.patient_id != pid}
        assert not foreign, f"{pid} saw evidence from {foreign}"
        assert all(f"P{n:04d}" not in r.answer for n in range(1, 51) if f"P{n:04d}" != pid)


# ---- logs


def test_audit_log_hashes_patient_ids(tmp_path: Path) -> None:
    audit = AuditLog(tmp_path / "audit.jsonl", salt="s")
    audit.write(user="dr.demo", patient=audit.h("P0001"), status="answered")
    text = (tmp_path / "audit.jsonl").read_text(encoding="utf-8")
    assert "P0001" not in text and audit.h("P0001") in text
    assert audit.h("P0001") != audit.h("P0002")
