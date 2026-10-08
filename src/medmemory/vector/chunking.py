"""Section-aware chunking for clinical text.

Clinical documents have strong structure: notes have SOAP-style headers (HISTORY OF PRESENT
ILLNESS, ASSESSMENT AND PLAN, ...), and drug labels have regulated sections (Dosage and
administration, Contraindications, ...). We never let a chunk straddle two sections, because
'metformin 1000 mg' in CURRENT MEDICATIONS means something different from the same string in
HISTORY OF PRESENT ILLNESS ('tried 1000 mg, went back to 500 mg').

Within a section we pack whole sentences up to `max_words`, with a sentence-level overlap,
so a fact split across a boundary still appears intact in one chunk.

Each chunk's `embed_text` is prefixed with "title | section |", a contextual header that
lets a short chunk like 'Denies chest pain.' still be found for the right patient/document.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from medmemory.contracts.schemas import Chunk, Namespace

_HEADER = re.compile(r"^([A-Z][A-Z0-9 /&()-]{2,60}):\s*$", re.MULTILINE)
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\[(-])|\n+")


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT.split(text) if s and s.strip()]


def pack(sentences: list[str], max_words: int, overlap_sentences: int = 1) -> list[str]:
    chunks: list[str] = []
    cur: list[str] = []
    words = 0
    for s in sentences:
        n = len(s.split())
        if cur and words + n > max_words:
            chunks.append(" ".join(cur))
            cur = cur[-overlap_sentences:] if overlap_sentences else []
            words = sum(len(x.split()) for x in cur)
        cur.append(s)
        words += n
    if cur:
        tail = " ".join(cur)
        if not chunks or tail != chunks[-1]:
            chunks.append(tail)
    return chunks


@dataclass(frozen=True)
class ChunkedDoc:
    chunk: Chunk
    embed_text: str


def split_note_sections(text: str) -> list[tuple[str, str]]:
    matches = list(_HEADER.finditer(text))
    if not matches:
        return [("note", text.strip())]
    out = []
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[m.end() : end].strip()
        if body:
            out.append((m.group(1).strip().title(), body))
    return out


def chunk_note(note: dict[str, Any], max_words: int = 120) -> list[ChunkedDoc]:
    out = []
    for section, body in split_note_sections(note["text"]):
        for i, piece in enumerate(pack(split_sentences(body), max_words)):
            meta: dict[str, Any] = {
                "doc_type": "note",
                "note_type": note["type"],
                "date": note["date"],
                "date_num": int(note["date"].replace("-", "")),
                "encounter_id": note["encounter_id"],
            }
            chunk = Chunk(
                chunk_id=f"{note['doc_id']}#{_slug(section)}#{i}",
                doc_id=note["doc_id"],
                namespace=Namespace.PATIENT_NOTES,
                title=note["title"],
                section=section,
                text=piece,
                patient_id=note["patient_id"],
                metadata=meta,
            )
            out.append(ChunkedDoc(chunk, f"{note['title']} | {section} | {piece}"))
    return out


_LABEL_HEADER = re.compile(r"^\s*(?:\d+(?:\.\d+)*\s+)?[A-Z][A-Z ,&/()-]{3,}(?=\s+[A-Z][a-z])\s*")


def chunk_label(label: dict[str, Any], max_words: int = 160) -> list[ChunkedDoc]:
    doc_id = f"label:{label['drug'].replace(' ', '-')}"
    title = f"{label['drug'].title()} — FDA label"
    out = []
    for section, raw_body in label["sections"].items():
        body = _LABEL_HEADER.sub("", raw_body, count=1)  # drop "4 CONTRAINDICATIONS" lead-in
        for i, piece in enumerate(pack(split_sentences(body), max_words, overlap_sentences=1)):
            chunk = Chunk(
                chunk_id=f"{doc_id}#{_slug(section)}#{i}",
                doc_id=doc_id,
                namespace=Namespace.DRUG_LABELS,
                title=title,
                section=section,
                text=piece,
                metadata={
                    "doc_type": "drug_label",
                    "drug": label["drug"],
                    "rxcui": label.get("ingredient_rxcui", ""),
                    "set_id": label["set_id"],
                    "effective_time": label["effective_time"],
                    "source_url": label["source_url"],
                },
            )
            out.append(ChunkedDoc(chunk, f"{label['drug']} label | {section} | {piece}"))
    return out


def chunk_topic(topic: dict[str, Any], max_words: int = 140) -> list[ChunkedDoc]:
    doc_id = f"topic:{topic['topic_id']}"
    out = []
    # MedlinePlus summaries use question headings ("What is ...?"): treat each as a section.
    paragraphs = [p.strip() for p in topic["summary"].split("\n") if p.strip()]
    sections: list[tuple[str, list[str]]] = []
    for p in paragraphs:
        if p.endswith("?") and len(p.split()) <= 14:
            sections.append((p, []))
        else:
            if not sections:
                sections.append(("Summary", []))
            sections[-1][1].append(p)
    for section, paras in sections:
        sentences = [s for p in paras for s in split_sentences(p)]
        for i, piece in enumerate(pack(sentences, max_words)):
            chunk = Chunk(
                chunk_id=f"{doc_id}#{_slug(section)[:60]}#{i}",
                doc_id=doc_id,
                namespace=Namespace.GUIDELINES,
                title=f"{topic['title']} (MedlinePlus)",
                section=section,
                text=piece,
                metadata={
                    "doc_type": "health_topic",
                    "topic": topic["topic_id"],
                    "source_url": topic["url"],
                    "attribution": topic["attribution"],
                },
            )
            out.append(ChunkedDoc(chunk, f"{topic['title']} | {section} | {piece}"))
    return out
