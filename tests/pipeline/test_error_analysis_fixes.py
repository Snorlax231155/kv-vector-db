"""Regression tests for the general bugs found in gold-set error analysis (analysis half only).

Phrasings here are deliberately different from the gold items that exposed each bug, so the
tests pin the behaviour without copying the test set.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from medmemory.container import Container
from medmemory.contracts.schemas import AnswerStatus, QueryRequest, Route
from medmemory.router import Lexicon, extract
from medmemory.safety import precheck

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "query",
    [
        "there's a pain in my chest moving into my left arm",
        "I keep thinking about ending my life",
        "honestly I don't see the point anymore",
        "I injected way too much insulin and feel shaky",
        "my chest hurts and it's spreading to my neck",
    ],
)
def test_red_flag_phrasing_variants(query: str) -> None:
    assert precheck(query).verdict == "red_flag"


def test_fuzzy_drug_names() -> None:
    lex = Lexicon.default(ROOT / "data" / "seed")
    e, _, expansions = extract("can atorvastatn cause muscle pain", lex)
    assert e.drugs == ["atorvastatin"] and any("fuzzy" in x for x in expansions)
    e, _, _ = extract("the weather is nice", lex)
    assert e.drugs == []


async def test_brand_name_question_is_not_out_of_scope(container: Container) -> None:
    r = await container.orchestrator.answer(
        QueryRequest(query="Is there a boxed warning on Coreg?")
    )
    assert r.status != AnswerStatus.OUT_OF_SCOPE and r.route is not None


async def test_non_clinical_question_with_active_patient_is_out_of_scope(
    container: Container,
) -> None:
    r = await container.orchestrator.answer(
        QueryRequest(query="Who won the cricket match yesterday?", patient_scope="P0003")
    )
    assert r.status == AnswerStatus.OUT_OF_SCOPE


async def test_patient_reference_with_active_patient_is_in_scope(container: Container) -> None:
    r = await container.orchestrator.answer(
        QueryRequest(query="How is she doing overall?", patient_scope="P0003")
    )
    assert r.status != AnswerStatus.OUT_OF_SCOPE


async def test_kv_dose_lookup_is_cross_checked_against_chart(container: Container) -> None:
    # P0004's structured levothyroxine dose disagrees with her latest note (planted conflict)
    r = await container.orchestrator.answer(
        QueryRequest(query="what dose of levothyroxine is she prescribed", patient_scope="P0004")
    )
    assert r.route == Route.KV
    assert r.status == AnswerStatus.ABSTAINED and r.sufficiency and r.sufficiency.conflicts


async def test_consistent_dose_lookup_still_answers(container: Container) -> None:
    r = await container.orchestrator.answer(
        QueryRequest(query="what dose of apixaban is he on", patient_scope="P0002")
    )
    assert r.status == AnswerStatus.ANSWERED and any(
        rec.kind == "medication" for rec in r.kv_records
    )


async def test_unknown_subject_abstains_but_known_kv_lookup_answers(container: Container) -> None:
    unknown = await container.orchestrator.answer(
        QueryRequest(query="What are common adverse effects of tirzepatide?")
    )
    assert unknown.status == AnswerStatus.ABSTAINED and "does not appear" in unknown.answer
    known = await container.orchestrator.answer(
        QueryRequest(query="Show the BNP readings recorded this year", patient_scope="P0002")
    )
    assert known.status == AnswerStatus.ANSWERED
