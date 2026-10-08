from __future__ import annotations

import pytest

from medmemory.contracts.schemas import (
    Chunk,
    Entities,
    Evidence,
    KVRecord,
    Namespace,
    Origin,
    Route,
    ScoredChunk,
)
from medmemory.safety import assess_sufficiency, check_citations, precheck, resolve_scope

# --------------------------------------------------------------------------- red flags


@pytest.mark.parametrize(
    ("query", "category"),
    [
        ("I have crushing chest pain spreading to my left arm", "cardiac"),
        ("chest pain radiating to my jaw and I'm sweating", "cardiac"),
        ("My dad suddenly has slurred speech and his face is drooping", "stroke"),
        ("worst headache of my life right now", "stroke"),
        ("I want to kill myself", "self_harm"),
        ("I've been thinking about suicide", "self_harm"),
        ("I took too many pills an hour ago", "overdose"),
        ("my throat is closing after eating shrimp", "anaphylaxis"),
        ("I can't breathe", "breathing"),
        ("he is unconscious and won't wake up", "unresponsive"),
    ],
)
def test_red_flags_short_circuit(query: str, category: str) -> None:
    r = precheck(query)
    assert r.verdict == "red_flag" and r.category == category and r.message


def test_self_harm_message_has_crisis_lines() -> None:
    msg = precheck("I want to end my life").message or ""
    assert "14416" in msg and "988" in msg


@pytest.mark.parametrize(
    "query",
    [
        "Has P0005 reported chest pain radiating to the arm?",
        "Does P0004 have a history of suicidal ideation?",
        "Is there any mention of overdose in P0012's notes?",
    ],
)
def test_record_questions_on_red_flag_topics_are_not_emergencies(query: str) -> None:
    r = precheck(query)
    assert r.verdict == "ok" and r.category == "record_query_on_red_flag_topic"


def test_out_of_scope_and_clinical_context() -> None:
    assert precheck("tell me a joke").verdict == "out_of_scope"
    assert precheck("tell me a joke", clinical_context=True).verdict == "ok"
    assert precheck("Why was lisinopril stopped?").verdict == "ok"  # drug vocabulary counts


# --------------------------------------------------------------------------- scope


def exists(pid: str) -> bool:
    return pid in {"P0001", "P0002"}


def test_scope_resolution() -> None:
    assert resolve_scope("P0001", [], exists).scope == "P0001"
    assert resolve_scope(None, ["P0002"], exists).scope == "P0002"
    assert resolve_scope("P0001", ["P0001"], exists).scope == "P0001"
    v = resolve_scope("P0001", ["P0002"], exists)
    assert v.scope is None and v.violation and "Switch" in v.violation
    assert resolve_scope(None, ["P0001", "P0002"], exists).violation
    assert resolve_scope(None, ["P0999"], exists).violation


# --------------------------------------------------------------------------- citations


def ev(key: str, text: str, origin: Origin = Origin.VECTOR) -> Evidence:
    return Evidence(key=key, source_id=f"vec:{key}", origin=origin, title=f"title {key}", text=text)


EVIDENCE = [
    ev("S1", "Patient P0001 hemoglobin a1c on 2026-08-02: 7.7% (flag: high).", Origin.KV),
    ev("S2", "The maximum recommended daily dose of metformin is 2550 mg in adults."),
]


def test_fully_cited_grounded_answer_passes() -> None:
    out = check_citations(
        "Her latest HbA1c was 7.7% on 2026-08-02 [S1]. The label lists a maximum of 2550 mg per day [S2].",
        EVIDENCE,
    )
    assert out.check.passed and out.check.grounding_rate == 1.0
    assert [c.key for c in out.citations] == ["S1", "S2"]


def test_trailing_marker_after_period_is_attached() -> None:
    out = check_citations("Her latest HbA1c was 7.7%. [S1]", EVIDENCE)
    assert out.check.passed


def test_invented_number_is_removed() -> None:
    out = check_citations(
        "The maximum dose is 3000 mg per day [S2]. Her HbA1c was 7.7% [S1].", EVIDENCE
    )
    assert not out.check.passed and out.check.unsupported_numbers == ["3000"]
    assert "3000" not in out.text and "7.7%" in out.text
    assert out.check.grounding_rate == 0.5


def test_uncited_claim_and_bad_key_are_removed() -> None:
    out = check_citations("Metformin causes weight loss. Her HbA1c was 7.7% [S9].", EVIDENCE)
    assert out.check.supported_sentences == 0 and out.check.invalid_citation_keys == ["S9"]
    assert out.text == ""


def test_hedge_sentences_are_exempt() -> None:
    out = check_citations(
        "Based on the retrieved sources, here is information consistent with your question. Her HbA1c was 7.7% [S1].",
        EVIDENCE,
    )
    assert out.check.passed and out.check.total_sentences == 1
    assert out.text.startswith("Based on")


def test_multi_key_marker() -> None:
    out = check_citations("Her HbA1c was 7.7% and the maximum dose is 2550 mg [S1, S2].", EVIDENCE)
    assert out.check.passed


# --------------------------------------------------------------------------- sufficiency


def med(dose: str) -> KVRecord:
    return KVRecord(
        key="patient:P0004:med:levothyroxine",
        kind="medication",
        patient_id="P0004",
        title="m",
        data={"drug": "levothyroxine", "dose": dose, "status": "active", "text": "x"},
    )


def note_chunk(text: str, section: str = "Current Medications", score: float = 0.5) -> ScoredChunk:
    return ScoredChunk(
        chunk=Chunk(
            chunk_id="n#0",
            doc_id="n",
            namespace=Namespace.PATIENT_NOTES,
            title="note",
            section=section,
            text=text,
            patient_id="P0004",
        ),
        score=score,
        rank=1,
    )


def test_conflicting_doses_abstain() -> None:
    s = assess_sufficiency(
        Route.HYBRID,
        Entities(),
        [med("75 mcg")],
        [note_chunk("- levothyroxine 50 mcg once daily")],
        0.1,
    )
    assert not s.sufficient and s.conflicts and "75 mcg" in s.conflicts[0]


def test_dose_mentioned_in_history_is_not_a_conflict() -> None:
    s = assess_sufficiency(
        Route.HYBRID,
        Entities(),
        [med("75 mcg")],
        [note_chunk("Previously on levothyroxine 50 mcg.", section="History Of Present Illness")],
        0.1,
    )
    assert s.sufficient


def test_low_similarity_without_records_abstains() -> None:
    s = assess_sufficiency(
        Route.VECTOR, Entities(), [], [note_chunk("unrelated", score=0.05)], 0.12
    )
    assert not s.sufficient and "below" in s.reason


def test_dosing_question_without_known_drug_abstains() -> None:
    s = assess_sufficiency(
        Route.VECTOR, Entities(dosage_intent=True), [], [note_chunk("dose text", score=0.9)], 0.1
    )
    assert not s.sufficient and "label" in s.reason
