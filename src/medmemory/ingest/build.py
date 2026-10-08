"""Load the seed dataset into the configured KV and vector stores. Idempotent.

A fingerprint (seed file hashes + embedder + chunker version) is stored in the KV store and
next to local vector files. If nothing changed, startup skips work; if anything changed,
both sides are rebuilt so KV and vectors never drift apart.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from medmemory.contracts.schemas import Namespace
from medmemory.text import words
from medmemory.vector.chunking import ChunkedDoc, chunk_label, chunk_note, chunk_topic
from medmemory.vector.stores.memory import InMemoryVectorStore

if TYPE_CHECKING:
    from medmemory.container import Container

log = logging.getLogger(__name__)
CHUNKER_VERSION = "chunker-v4"


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def fingerprint(seed_dir: Path, embedder_name: str) -> str:
    manifest = seed_dir.parent / "manifest.json"
    files = json.loads(manifest.read_text(encoding="utf-8"))["files"] if manifest.exists() else {}
    payload = json.dumps(
        {"files": files, "embedder": embedder_name, "chunker": CHUNKER_VERSION}, sort_keys=True
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:24]


def chunk_all(seed_dir: Path) -> dict[Namespace, list[ChunkedDoc]]:
    out: dict[Namespace, list[ChunkedDoc]] = {ns: [] for ns in Namespace}
    for note in _jsonl(seed_dir / "notes.jsonl"):
        out[Namespace.PATIENT_NOTES].extend(chunk_note(note))
    for label in _jsonl(seed_dir / "drug_labels.jsonl"):
        out[Namespace.DRUG_LABELS].extend(chunk_label(label))
    for topic in _jsonl(seed_dir / "guidelines.jsonl"):
        out[Namespace.GUIDELINES].extend(chunk_topic(topic))
    return out


def load_kv(c: Container, seed_dir: Path) -> dict[str, int]:
    repo = c.records
    notes = _jsonl(seed_dir / "notes.jsonl")
    notes_per_patient: dict[str, int] = {}
    for n in notes:
        notes_per_patient[n["patient_id"]] = notes_per_patient.get(n["patient_id"], 0) + 1
        repo.load_document(
            n["doc_id"],
            {
                "doc_id": n["doc_id"],
                "title": n["title"],
                "text": n["text"],
                "namespace": Namespace.PATIENT_NOTES.value,
                "patient_id": n["patient_id"],
                "date": n["date"],
            },
        )
    ops = 0
    patients = _jsonl(seed_dir / "patients.jsonl")
    for p in patients:
        p["note_count"] = notes_per_patient.get(p["patient_id"], 0)
        ops += repo.load_patient(p)
    ops += repo.load_icd10(_jsonl(seed_dir / "icd10cm.jsonl"))
    labels = _jsonl(seed_dir / "drug_labels.jsonl")
    for label in labels:
        repo.load_drug(label)
        doc_id = f"label:{label['drug'].replace(' ', '-')}"
        text = "\n\n".join(f"{k.upper()}:\n{v}" for k, v in label["sections"].items())
        repo.load_document(
            doc_id,
            {
                "doc_id": doc_id,
                "title": f"{label['drug'].title()} — FDA label",
                "text": text,
                "namespace": Namespace.DRUG_LABELS.value,
                "source_url": label["source_url"],
                "effective_time": label["effective_time"],
            },
        )
    for t in _jsonl(seed_dir / "guidelines.jsonl"):
        doc_id = f"topic:{t['topic_id']}"
        repo.load_document(
            doc_id,
            {
                "doc_id": doc_id,
                "title": f"{t['title']} (MedlinePlus)",
                "text": t["summary"],
                "namespace": Namespace.GUIDELINES.value,
                "source_url": t["url"],
                "attribution": t["attribution"],
            },
        )
    repo.bump_version(None)
    return {"patients": len(patients), "notes": len(notes), "labels": len(labels), "kv_ops": ops}


async def _upsert_all(
    c: Container, chunked: dict[Namespace, list[ChunkedDoc]], batch: int = 256
) -> dict[str, Any]:
    stats: dict[str, Any] = {}
    for ns, docs in chunked.items():
        t0 = time.perf_counter()
        for i in range(0, len(docs), batch):
            part = docs[i : i + batch]
            vecs = await asyncio.to_thread(c.embedder.embed_documents, [d.embed_text for d in part])
            await c.store.upsert(ns, [d.chunk for d in part], vecs)
        stats[ns.value] = {
            "chunks": len(docs),
            "embed_upsert_s": round(time.perf_counter() - t0, 2),
        }
    return stats


async def ingest(c: Container, force: bool = False) -> dict[str, Any]:
    seed_dir = c.settings.seed_dir
    fp = fingerprint(seed_dir, c.embedder.name)
    report: dict[str, Any] = {"fingerprint": fp}
    t0 = time.perf_counter()

    kv_fp = c.kv.get("meta:ingest_fingerprint")
    if force or kv_fp is None or kv_fp.decode() != fp:
        report["kv"] = load_kv(c, seed_dir)
        c.kv.put("meta:ingest_fingerprint", fp.encode())
        c.exact_cache.clear()
    else:
        report["kv"] = "up to date"

    chunked = chunk_all(seed_dir)
    c.orchestrator.vocabulary = build_vocabulary(seed_dir, chunked)
    store = c.store
    vec_dir = c.settings.var_dir / "vectors" / fp
    if isinstance(store, InMemoryVectorStore):  # includes FAISS (subclass)
        if not force and store.load(vec_dir, fp):
            report["vectors"] = "loaded from disk"
        else:
            report["vectors"] = await _upsert_all(c, chunked)
            store.save(vec_dir, fp)
    else:
        counts = {ns: await store.count(ns) for ns in chunked}
        if force or any(counts[ns] < len(docs) for ns, docs in chunked.items()):
            report["vectors"] = await _upsert_all(c, chunked, batch=96)
            wait = getattr(store, "wait_for_count", None)
            if wait is not None:
                for ns, docs in chunked.items():
                    await wait(ns, len(docs))
        else:
            report["vectors"] = {
                ns.value: {"chunks": n, "status": "present"} for ns, n in counts.items()
            }
    report["seconds"] = round(time.perf_counter() - t0, 2)
    c.ingest_report = report
    return report


def build_vocabulary(seed_dir: Path, chunked: dict[Namespace, list[ChunkedDoc]]) -> frozenset[str]:
    """Every word the knowledge base contains (documents, codes, drug names). Used to tell an
    unknown subject ('tirzepatide') from an ordinary word ('watch')."""
    vocab: set[str] = set()
    for docs in chunked.values():
        for d in docs:
            vocab |= words(d.embed_text)
    for row in _jsonl(seed_dir / "icd10cm.jsonl"):
        vocab |= words(row["name"])
    for row in _jsonl(seed_dir / "drug_labels.jsonl"):
        vocab |= words(" ".join([row["drug"], row["generic_name"], *row.get("brand_names", [])]))
    for row in _jsonl(seed_dir / "guidelines.jsonl"):
        vocab |= words(" ".join(row.get("also_called", [])))
    return frozenset(vocab)


def iter_chunk_texts(seed_dir: Path) -> Iterable[tuple[Namespace, str]]:
    for ns, docs in chunk_all(seed_dir).items():
        for d in docs:
            yield ns, d.embed_text
