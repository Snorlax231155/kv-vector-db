from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from medmemory.contracts.schemas import Route
from medmemory.router import Lexicon, RulesRouter, extract, normalize

ROOT = Path(__file__).resolve().parents[2]
LEX = Lexicon.default(ROOT / "data" / "seed")
ROUTER = RulesRouter(LEX)
TODAY = date(2026, 10, 1)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Latest HbA1c?", "latest hemoglobin a1c?"),
        ("history of MI and HTN", "history of myocardial infarction and hypertension"),
        ("mi casa", "mi casa"),  # ambiguous abbreviations only expand in upper case
        ("T2DM with CKD", "type 2 diabetes with chronic kidney disease"),
        ("glycated haemoglobin", "hemoglobin a1c"),
        ("on an ACE inhibitor", "on an ace inhibitor"),
        ("eGFR trend", "egfr trend"),
        ("LDL cholesterol", "ldl cholesterol"),  # no double expansion
    ],
)
def test_normalize(text: str, expected: str) -> None:
    assert normalize(text)[0] == expected


def test_normalize_reports_expansions() -> None:
    _, applied = normalize("HbA1c and BP")
    assert any("hemoglobin a1c" in a for a in applied) and any(
        "blood pressure" in a for a in applied
    )


def test_extract_ids_codes_and_gazetteers() -> None:
    e, _, _ = extract("Is P0003 on Eliquis? Check E11.9 and RxCUI 6809, latest A1c", LEX, TODAY)
    assert e.patient_ids == ["P0003"]
    assert e.icd10_codes == ["E11.9"]
    assert e.rxnorm_codes == ["6809"]
    assert e.drugs == ["apixaban"]  # brand name mapped to generic from the openFDA label
    assert e.labs == ["hemoglobin a1c"]
    assert e.time_range is not None and e.time_range.latest_only


@pytest.mark.parametrize("text", ["P0007's labs", "p0007s labs", "labs for P0007", "P0007s"])
def test_patient_id_with_possessive(text: str) -> None:
    e, _, _ = extract(text, LEX, TODAY)
    assert e.patient_ids == ["P0007"]


def test_vitamin_b12_is_not_an_icd_code() -> None:
    e, _, _ = extract("Should I take vitamin B12?", LEX, TODAY)
    assert e.icd10_codes == []


def test_time_ranges() -> None:
    e, _, _ = extract("A1c over the last 6 months", LEX, TODAY)
    assert e.time_range and e.time_range.start == date(2026, 4, 4) and e.time_range.end == TODAY
    e, _, _ = extract("labs in 2025", LEX, TODAY)
    assert (
        e.time_range
        and e.time_range.start == date(2025, 1, 1)
        and e.time_range.end == date(2025, 12, 31)
    )


def test_negation_and_dosage_intent() -> None:
    e, _, _ = extract("Is she not on warfarin?", LEX, TODAY)
    assert e.negated == ["warfarin"]
    e, _, _ = extract("What is the maximum daily dose of metformin?", LEX, TODAY)
    assert e.dosage_intent and e.drugs == ["metformin"]


@pytest.mark.parametrize(
    ("query", "scope", "route"),
    [
        ("What is the latest HbA1c for P0001?", None, Route.KV),
        ("What does ICD-10-CM code N18.30 mean?", None, Route.KV),
        ("What is the RxCUI for apixaban?", None, Route.KV),
        ("List her current medications", "P0002", Route.KV),
        ("What are the common side effects of metformin?", None, Route.VECTOR),
        ("What did the notes say about ankle swelling?", "P0002", Route.VECTOR),
        ("How is high blood pressure treated?", None, Route.VECTOR),
        ("Given her latest eGFR, is metformin still appropriate?", "P0001", Route.HYBRID),
        ("Why was lisinopril stopped?", "P0001", Route.HYBRID),
        ("What are the side effects of the statin she takes?", "P0005", Route.HYBRID),
    ],
)
def test_rules_routes(query: str, scope: str | None, route: Route) -> None:
    d = ROUTER.route(query, scope, TODAY)
    assert d.route == route, d.reasons
    assert 0.0 <= d.confidence <= 1.0 and d.router == "rules"


def test_rules_fallback() -> None:
    scoped = ROUTER.route("what about this one", "P0005", TODAY)
    assert scoped.route == Route.HYBRID and "defaulting to HYBRID" in " ".join(scoped.reasons)
    open_q = ROUTER.route("what about this one", None, TODAY)
    assert open_q.route == Route.VECTOR and "defaulting to semantic search" in " ".join(
        open_q.reasons
    )


def test_entity_signature_is_order_independent() -> None:
    a, _, _ = extract("warfarin and apixaban for P0001", LEX, TODAY)
    b, _, _ = extract("apixaban and warfarin for P0001", LEX, TODAY)
    assert a.signature() == b.signature()
    c, _, _ = extract("most recent HbA1c", LEX, TODAY)
    d, _, _ = extract("latest HbA1c", LEX, TODAY)
    assert c.signature() == d.signature()
