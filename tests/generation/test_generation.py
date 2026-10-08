from __future__ import annotations

from medmemory.contracts.schemas import Evidence, Origin, Route
from medmemory.generation import INSUFFICIENT, build_generator
from medmemory.generation.extractive import compose, parse_prompt
from medmemory.generation.prompts import PROMPT, prompt_inputs
from medmemory.safety import check_citations

EVIDENCE = [
    Evidence(
        key="S1",
        source_id="kv:x",
        origin=Origin.KV,
        title="P0001 · lab · egfr · 2026-08-02",
        text="Patient P0001 egfr on 2026-08-02: 47 mL/min/1.73m2 (flag: low).",
    ),
    Evidence(
        key="S2",
        source_id="vec:y",
        origin=Origin.VECTOR,
        title="Metformin — FDA label (2026) · Contraindications",
        text="Metformin is contraindicated in patients with an eGFR below 30 mL/min/1.73 m2. Obtain an eGFR before initiating.",
    ),
]


def test_prompt_contains_rules_and_numbered_sources() -> None:
    msgs = PROMPT.format_messages(**prompt_inputs("Is metformin OK?", EVIDENCE, Route.HYBRID))
    system, human = msgs[0].content, msgs[1].content
    assert "never from your own knowledge" in system and INSUFFICIENT in system
    assert "[S1] (kv · P0001 · lab · egfr · 2026-08-02)" in human and "[S2] (vector · " in human


def test_parse_prompt_roundtrip_handles_parentheses_in_titles() -> None:
    human = PROMPT.format_messages(**prompt_inputs("q?", EVIDENCE, Route.HYBRID))[1].content
    question, sources = parse_prompt(str(human))
    assert question == "q?" and [s[0] for s in sources] == ["S1", "S2"]
    assert sources[1][2] == "Metformin — FDA label (2026) · Contraindications"


def test_extractive_answer_is_verbatim_and_grounded() -> None:
    text = compose(
        "Is metformin contraindicated at her eGFR?",
        [(e.key, e.origin.value, e.title, e.text) for e in EVIDENCE],
    )
    assert "[S1]" in text and "[S2]" in text
    assert check_citations(text, EVIDENCE).check.passed


def test_extractive_abstains_when_subject_missing() -> None:
    assert (
        compose(
            "side effects of ozempic", [(e.key, e.origin.value, e.title, e.text) for e in EVIDENCE]
        )
        == INSUFFICIENT
    )


async def test_chain_generate_and_stream_agree() -> None:
    gen = build_generator("extractive")
    full = await gen.generate("Is metformin contraindicated at her eGFR?", EVIDENCE, Route.HYBRID)
    streamed = "".join(
        [
            p
            async for p in gen.stream(
                "Is metformin contraindicated at her eGFR?", EVIDENCE, Route.HYBRID
            )
        ]
    )
    assert full == streamed and full.startswith("Based on the retrieved sources")
