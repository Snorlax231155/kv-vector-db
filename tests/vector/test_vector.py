from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from medmemory.contracts.protocols import Embedder, VectorStore
from medmemory.contracts.schemas import Chunk, Namespace, ScoredChunk
from medmemory.vector.chunking import (
    chunk_label,
    chunk_note,
    chunk_topic,
    pack,
    split_note_sections,
)
from medmemory.vector.embedders import HashingEmbedder
from medmemory.vector.filters import and_filters, matches
from medmemory.vector.rerankers import LexicalReranker, NoopReranker, rrf_fuse
from medmemory.vector.stores.memory import InMemoryVectorStore

ROOT = Path(__file__).resolve().parents[2]
SEED = ROOT / "data" / "seed"


def _jsonl(name: str) -> list[dict]:
    return [json.loads(line) for line in (SEED / name).read_text(encoding="utf-8").splitlines()]


# --------------------------------------------------------------------------- chunking


def test_note_chunks_never_cross_sections() -> None:
    note = _jsonl("notes.jsonl")[0]
    sections = [s for s, _ in split_note_sections(note["text"])]
    assert "Current Medications" in sections and "Assessment And Plan" in sections
    for cd in chunk_note(note):
        assert cd.chunk.section in sections
        assert cd.chunk.patient_id == note["patient_id"]
        assert cd.embed_text.startswith(note["title"])
        assert cd.chunk.metadata["date_num"] == int(note["date"].replace("-", ""))


def test_chunk_ids_are_deterministic_and_unique() -> None:
    notes = _jsonl("notes.jsonl")[:20]
    a = [c.chunk.chunk_id for n in notes for c in chunk_note(n)]
    b = [c.chunk.chunk_id for n in notes for c in chunk_note(n)]
    assert a == b and len(set(a)) == len(a)
    assert all(cid.isascii() and len(cid) <= 512 for cid in a)  # Pinecone ID constraints


def test_label_chunks_strip_section_numbering_and_carry_drug_metadata() -> None:
    label = next(lbl for lbl in _jsonl("drug_labels.jsonl") if lbl["drug"] == "metformin")
    chunks = chunk_label(label)
    contra = [c for c in chunks if c.chunk.section == "Contraindications"]
    assert contra and not contra[0].chunk.text.startswith("4 CONTRAINDICATIONS")
    assert all(
        c.chunk.metadata["drug"] == "metformin" and c.chunk.patient_id is None for c in chunks
    )


def test_topic_chunks_use_question_headings() -> None:
    topic = _jsonl("guidelines.jsonl")[0]
    chunks = chunk_topic(topic)
    assert chunks and all(c.chunk.namespace == Namespace.GUIDELINES for c in chunks)


def test_pack_respects_budget_with_overlap() -> None:
    sents = [f"Sentence number {i} has five words." for i in range(10)]
    out = pack(sents, max_words=12, overlap_sentences=1)
    assert len(out) > 1
    assert all(len(c.split()) <= 18 for c in out)
    assert out[0].endswith(sents[1]) and out[1].startswith(
        sents[1]
    )  # overlap carries the last sentence forward


# --------------------------------------------------------------------------- embedding + filters


def test_hashing_embedder_is_deterministic_unit_norm_and_lexically_sensible() -> None:
    e = HashingEmbedder()
    assert isinstance(e, Embedder)
    v1, v2 = e.embed_query("latest HbA1c"), e.embed_query("latest HbA1c")
    assert np.allclose(v1, v2) and abs(np.linalg.norm(v1) - 1) < 1e-5
    near = float(e.embed_query("hemoglobin a1c result") @ e.embed_query("HbA1c value"))
    far = float(e.embed_query("hemoglobin a1c result") @ e.embed_query("knee osteoarthritis pain"))
    assert near > far


@pytest.mark.parametrize(
    ("flt", "expected"),
    [
        ({"patient_id": "P0001"}, True),
        ({"patient_id": {"$eq": "P0002"}}, False),
        ({"date_num": {"$gte": 20250101, "$lte": 20251231}}, True),
        ({"tags": {"$in": ["b"]}}, True),
        ({"tags": {"$nin": ["a"]}}, False),
        ({"$or": [{"patient_id": "P9"}, {"section": "plan"}]}, True),
        ({"$and": [{"patient_id": "P0001"}, {"section": "x"}]}, False),
        ({"missing": {"$exists": False}}, True),
    ],
)
def test_filter_semantics_match_pinecone(flt: dict, expected: bool) -> None:
    meta = {"patient_id": "P0001", "date_num": 20250315, "tags": ["a", "b"], "section": "plan"}
    assert matches(meta, flt) is expected


def test_and_filters_combines() -> None:
    assert and_filters(None, {"a": 1}) == {"a": 1}
    assert and_filters({"a": 1}, {"b": 2}) == {"$and": [{"a": 1}, {"b": 2}]}
    assert and_filters(None, None) is None


# --------------------------------------------------------------------------- stores


def _chunks(n: int) -> list[Chunk]:
    return [
        Chunk(
            chunk_id=f"d{i}#s#0",
            doc_id=f"d{i}",
            namespace=Namespace.PATIENT_NOTES,
            title=f"doc {i}",
            section="s",
            text=f"note {i} about {'cough' if i % 2 else 'swelling'}",
            patient_id=f"P000{i % 3}",
        )
        for i in range(n)
    ]


def _stores() -> list[VectorStore]:
    out: list[VectorStore] = [InMemoryVectorStore(384)]
    try:
        from medmemory.vector.stores.faiss_store import FaissVectorStore

        out += [FaissVectorStore(384, "flat"), FaissVectorStore(384, "hnsw")]
    except ImportError:  # pragma: no cover
        pass
    return out


@pytest.mark.parametrize("store", _stores(), ids=lambda s: s.name)
async def test_store_upsert_query_filter_delete(store: VectorStore) -> None:
    emb = HashingEmbedder()
    chunks = _chunks(12)
    vecs = emb.embed_documents([c.text for c in chunks])
    assert await store.upsert(Namespace.PATIENT_NOTES, chunks, vecs) == 12
    await store.upsert(Namespace.PATIENT_NOTES, chunks, vecs)  # idempotent
    assert await store.count(Namespace.PATIENT_NOTES) == 12
    q = emb.embed_query("cough")
    hits = await store.query(Namespace.PATIENT_NOTES, q, 3)
    assert len(hits) == 3 and hits[0].score >= hits[-1].score and "cough" in hits[0].chunk.text
    scoped = await store.query(Namespace.PATIENT_NOTES, q, 10, {"patient_id": {"$eq": "P0001"}})
    assert scoped and all(h.chunk.patient_id == "P0001" for h in scoped)
    text_hits = await store.text_query(Namespace.PATIENT_NOTES, "swelling", 5)
    assert text_hits and all("swelling" in h.chunk.text for h in text_hits)
    await store.delete(Namespace.PATIENT_NOTES, filter={"patient_id": "P0000"})
    assert await store.count(Namespace.PATIENT_NOTES) == 8
    assert await store.fetch(Namespace.PATIENT_NOTES, ["d1#s#0"])
    await store.delete(Namespace.PATIENT_NOTES, delete_all=True)
    assert await store.count(Namespace.PATIENT_NOTES) == 0


def test_memory_store_persistence_roundtrip(tmp_path: Path) -> None:
    emb = HashingEmbedder()
    s = InMemoryVectorStore(384)
    chunks = _chunks(5)
    s.upsert_sync(Namespace.PATIENT_NOTES, chunks, emb.embed_documents([c.text for c in chunks]))
    s.save(tmp_path, "fp1")
    s2 = InMemoryVectorStore(384)
    assert not s2.load(tmp_path, "other-fingerprint")
    assert s2.load(tmp_path, "fp1")
    assert len(s2.all_chunks(Namespace.PATIENT_NOTES)) == 5


# --------------------------------------------------------------------------- rerank + fusion


def _scored(ids: list[str], retriever: str = "dense") -> list[ScoredChunk]:
    return [
        ScoredChunk(
            chunk=Chunk(
                chunk_id=i,
                doc_id=i,
                namespace=Namespace.DRUG_LABELS,
                title=i,
                section="s",
                text=f"text {i}",
            ),
            score=1.0 - k * 0.1,
            rank=k + 1,
            retriever=retriever,
        )
        for k, i in enumerate(ids)
    ]


def test_rrf_rewards_agreement_between_lists() -> None:
    fused = rrf_fuse([_scored(["a", "b", "c"]), _scored(["c", "b", "d"], "bm25")])
    order = [f.chunk.chunk_id for f in fused]
    assert order[0] in ("b", "c") and set(order) == {"a", "b", "c", "d"}
    assert [f.rank for f in fused] == [1, 2, 3, 4]


def test_lexical_reranker_prefers_on_topic_chunk() -> None:
    chunks = [
        ScoredChunk(
            chunk=Chunk(
                chunk_id="x",
                doc_id="x",
                namespace=Namespace.DRUG_LABELS,
                title="t",
                section="Adverse reactions",
                text="Nausea and headache were reported.",
            ),
            score=0.6,
            rank=1,
        ),
        ScoredChunk(
            chunk=Chunk(
                chunk_id="y",
                doc_id="y",
                namespace=Namespace.DRUG_LABELS,
                title="t",
                section="Contraindications",
                text="Contraindicated in severe renal impairment (eGFR below 30).",
            ),
            score=0.55,
            rank=2,
        ),
    ]
    out = LexicalReranker().rerank("contraindications renal impairment", chunks, 2)
    assert out[0].chunk.chunk_id == "y" and out[0].rerank_score is not None and out[0].rank == 1
    assert [c.chunk.chunk_id for c in NoopReranker().rerank("q", chunks, 1)] == ["x"]
