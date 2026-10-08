"""Pre-generation evidence sufficiency and conflict detection.

Abstaining is a correct, tested output. We abstain (and skip the LLM) when:
* nothing relevant was retrieved, or the best semantic match is below the calibrated
  threshold and there are no exact KV records;
* a KV-route question found no matching record;
* a dosing question has no drug-label evidence (doses must come from labels, never from
  model memory) and no patient medication record;
* sources conflict, e.g. the structured medication record says 500 mg but the medication
  list in a clinical note says 1000 mg. We surface both rather than pick one.
"""

from __future__ import annotations

import difflib
import re
from collections.abc import Sequence

from medmemory.contracts.schemas import (
    Entities,
    EvidenceSufficiency,
    KVRecord,
    Namespace,
    Route,
    ScoredChunk,
)

_DOSE = r"(\d+(?:\.\d+)?)\s*(mg|mcg|units?)\b"
_WORD = re.compile(r"[a-z][a-z0-9-]{4,}")
_COMMON = frozenset(
    [
        "about",
        "above",
        "after",
        "again",
        "against",
        "among",
        "answer",
        "being",
        "below",
        "between",
        "brand",
        "cause",
        "causes",
        "could",
        "daily",
        "doctor",
        "doses",
        "during",
        "effect",
        "effects",
        "every",
        "example",
        "explain",
        "first",
        "given",
        "guidance",
        "having",
        "health",
        "known",
        "label",
        "labels",
        "level",
        "levels",
        "likely",
        "medicine",
        "medicines",
        "might",
        "other",
        "people",
        "please",
        "question",
        "recommended",
        "should",
        "since",
        "someone",
        "something",
        "taking",
        "their",
        "there",
        "these",
        "thing",
        "those",
        "through",
        "treat",
        "treated",
        "treatment",
        "typical",
        "usual",
        "using",
        "usually",
        "versus",
        "what",
        "when",
        "where",
        "which",
        "while",
        "whose",
        "would",
        "years",
    ]
)


def unknown_subject(
    query: str,
    entities: Entities,
    scope: str | None,
    evidence_text: str,
    vocabulary: frozenset[str] | None = None,
) -> list[str]:
    """Words that look like the question's subject but that the knowledge base doesn't know.

    With a `vocabulary` (every word in every ingested document), a long word that appears
    nowhere in the corpus, and isn't a near-typo of a word that does, marks a subject we have
    no source for, e.g. an unlisted drug ('tirzepatide'). Then quoting a neighbouring topic
    would be wrong. Skipped when a known drug was recognised. Without a vocabulary we fall
    back to the retrieved evidence, but only for unscoped questions with no entities.
    """
    if entities.drugs or entities.icd10_codes or entities.rxnorm_codes:
        return []
    candidates = [w for w in _WORD.findall(query.lower()) if w not in _COMMON]
    if vocabulary is not None:
        vocab_list = sorted(vocabulary)
        return [
            w
            for w in candidates
            if len(w) >= 6
            and w not in vocabulary
            and not difflib.get_close_matches(w, vocab_list, n=1, cutoff=0.85)
        ]
    if scope or entities.labs or entities.conditions:
        return []
    text = evidence_text.lower()
    return [w for w in candidates if w not in text]


def detect_conflicts(kv_records: Sequence[KVRecord], chunks: Sequence[ScoredChunk]) -> list[str]:
    conflicts = []
    meds = [r for r in kv_records if r.kind == "medication" and r.data.get("status") == "active"]
    for med in meds:
        drug = str(med.data["drug"])
        kv_dose = str(med.data["dose"]).replace(" ", "").lower()
        for c in chunks:
            if (
                c.chunk.patient_id != med.patient_id
                or c.chunk.section.lower() != "current medications"
            ):
                continue
            for m in re.finditer(rf"\b{re.escape(drug)}\s+{_DOSE}", c.chunk.text, re.IGNORECASE):
                note_dose = f"{m.group(1)}{m.group(2)}".lower()
                if note_dose != kv_dose:
                    conflicts.append(
                        f"{drug}: structured record says {med.data['dose']} but the medication list in "
                        f"'{c.chunk.title}' says {m.group(1)} {m.group(2)}"
                    )
    return sorted(set(conflicts))


def assess(
    route: Route,
    entities: Entities,
    kv_records: Sequence[KVRecord],
    chunks: Sequence[ScoredChunk],
    min_score: float,
    query: str = "",
    scope: str | None = None,
    vocabulary: frozenset[str] | None = None,
) -> EvidenceSufficiency:
    best = max((c.score for c in chunks), default=None)
    if query:
        corpus = " ".join(
            [
                *(str(r.data.get("text", "")) for r in kv_records),
                *(f"{c.chunk.title} {c.chunk.text}" for c in chunks),
            ]
        )
        missing = unknown_subject(query, entities, scope, corpus, vocabulary)
        if missing:
            return EvidenceSufficiency(
                sufficient=False,
                reason=f"The question's subject ({', '.join(missing[:3])}) does not appear in any retrieved source.",
                max_score=best,
            )
    strong_chunks = [c for c in chunks if c.score >= min_score]
    if not kv_records and not chunks:
        return EvidenceSufficiency(
            sufficient=False, reason="No records or passages were retrieved.", max_score=best
        )
    if route == Route.KV and not kv_records and not strong_chunks:
        return EvidenceSufficiency(
            sufficient=False, reason="No matching structured record was found.", max_score=best
        )
    if not kv_records and not strong_chunks:
        return EvidenceSufficiency(
            sufficient=False,
            reason=f"The closest passage scored {best:.2f}, below the relevance threshold {min_score:.2f}.",
            max_score=best,
        )
    if (
        entities.dosage_intent
        and not entities.drugs
        and not any(r.kind == "medication" for r in kv_records)
    ):
        return EvidenceSufficiency(
            sufficient=False,
            reason="Dosing questions are only answered for drugs in the FDA label set, and no such drug was identified in the question.",
            max_score=best,
        )
    if entities.dosage_intent and entities.drugs:
        has_label = any(c.chunk.namespace == Namespace.DRUG_LABELS for c in strong_chunks)
        has_med = any(r.kind == "medication" for r in kv_records)
        if not has_label and not has_med:
            return EvidenceSufficiency(
                sufficient=False,
                reason="Dosing questions are only answered from a retrieved FDA drug label or the patient's medication record, and neither was found.",
                max_score=best,
            )
    conflicts = detect_conflicts(kv_records, chunks)
    if conflicts:
        return EvidenceSufficiency(
            sufficient=False, reason="The sources disagree.", max_score=best, conflicts=conflicts
        )
    return EvidenceSufficiency(sufficient=True, reason="ok", max_score=best)
