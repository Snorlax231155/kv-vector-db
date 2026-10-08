"""Rule-based router: transparent, fast (~0.1 ms), and the fallback for every other router.

The route is decided from three kinds of evidence:
* structured asks:  IDs/codes, record-type words, a lab/drug mentioned *about a patient*
* knowledge asks:   side effects, contraindications, guidance, explanations, 'is it safe'
* narrative asks:   what the notes say, what the patient reported/complained of

KV      = structured ask, nothing narrative or knowledge-seeking
VECTOR  = knowledge or narrative ask with no structured component
HYBRID  = both (e.g. 'given P0001's eGFR, is metformin appropriate?'), or too ambiguous to
          tell, because HYBRID retrieves from both sides and is the safe default
"""

from __future__ import annotations

import re
from datetime import date

from medmemory.contracts.schemas import Entities, Route, RouterDecision
from medmemory.router.entities import Lexicon, extract

KNOWLEDGE_RE = re.compile(
    r"\b(side effects?|adverse (?:effects?|reactions?)|contraindicat\w*|warnings?|precautions?|"
    r"interact\w*|guidance|guidelines?|recommend\w*|label|indicat\w*|why|explain|how does|how do|"
    r"how is|what causes|causes? of|symptoms? of|signs? of|risk factors?|treat(?:ed|ment|ments)?|"
    r"prevent\w*|manage\w*|mechanism|safe|safety|appropriate|should|consistent with|in line with|"
    r"according to|monitor\w*|avoid|lifestyle|what (?:is|are) (?:a |an |the )?(?:normal|target|goal)|"
    r"what does .* (?:mean|do)|used for|overview of|learn about|counsel\w*)\b"
)
NARRATIVE_RE = re.compile(
    r"\b(notes?|documented|mention\w*|report\w*|complain\w*|describ\w*|said|says|narrative|"
    r"history of present illness|hpi|assessment|plan|visit summary|summari[sz]e|"
    r"symptoms? (?:did|has|have)|experienc\w*|noticed|felt|feeling|cough\w*|swelling|pain|"
    r"bruis\w*|aches?|tingling|tolerat\w*|why (?:was|did)|reason)\b"
)
LOOKUP_RE = re.compile(
    r"\b(what is the (?:latest|most recent|current|last)|latest|most recent|current|list|show|"
    r"value|level|reading|result|when (?:was|did)|how many|code for|icd|rxcui|rxnorm|identifier|"
    r"brand names?|is .* on|taking|prescribed|dose of|dosage of|what dose|stopped|discontinued|"
    r"started|switched|changed|allergic)\b"
)
DEFINITION_RE = re.compile(
    r"\b(what does .* (?:mean|stand for)|meaning of|definition|describe (?:the )?code|what is (?:icd|code))\b"
)


class RulesRouter:
    name = "rules"

    def __init__(self, lexicon: Lexicon | None = None) -> None:
        self.lexicon = lexicon or Lexicon.default()

    def route(
        self, query: str, patient_scope: str | None = None, today: date | None = None
    ) -> RouterDecision:
        entities, normalized, expansions = extract(query, self.lexicon, today)
        return decide(query, normalized, entities, expansions, patient_scope)


def decide(
    query: str, normalized: str, e: Entities, expansions: list[str], patient_scope: str | None
) -> RouterDecision:
    has_patient = bool(e.patient_ids or patient_scope)
    has_code = bool(e.icd10_codes or e.rxnorm_codes)
    knowledge = bool(KNOWLEDGE_RE.search(normalized))
    if has_code and not has_patient and DEFINITION_RE.search(normalized):
        knowledge = False  # "what does E11.9 mean" is a code lookup, not a knowledge question
    narrative = has_patient and bool(NARRATIVE_RE.search(normalized))
    lookup_words = bool(LOOKUP_RE.search(normalized))
    structured = (
        has_code
        or bool(e.record_types)
        or (has_patient and bool(e.labs or (e.drugs and (lookup_words or e.dosage_intent))))
    )
    if (
        not has_patient
        and e.drugs
        and re.search(r"\b(rxcui|rxnorm|identifier|brand names?|generic name)\b", normalized)
    ):
        structured = True

    reasons: list[str] = []
    if has_patient:
        reasons.append(
            "patient in scope"
            if patient_scope
            else f"patient id in query ({', '.join(e.patient_ids)})"
        )
    if has_code:
        reasons.append("exact code present")
    if e.record_types:
        reasons.append(f"record types: {', '.join(e.record_types)}")
    if knowledge:
        reasons.append("knowledge-seeking language")
    if narrative:
        reasons.append("asks about documented narrative")

    if structured and (knowledge or narrative):
        route, conf = Route.HYBRID, 0.85 if knowledge else 0.75
        reasons.append("structured + unstructured evidence needed")
    elif structured:
        route, conf = Route.KV, 0.92 if (has_code or e.record_types or lookup_words) else 0.78
    elif narrative:
        route, conf = Route.VECTOR, 0.85
    elif knowledge and has_patient:
        # e.g. "what are the side effects of her statin?": which statin is a KV fact
        route, conf = Route.HYBRID, 0.72
        reasons.append("knowledge question about this patient's data")
    elif knowledge or e.drugs or e.conditions:
        route, conf = Route.VECTOR, 0.88 if knowledge else 0.7
    elif has_patient:
        route, conf = Route.HYBRID, 0.5
        reasons.append("open-ended patient question; defaulting to HYBRID")
    else:
        route, conf = Route.VECTOR, 0.45
        reasons.append("no strong signal; defaulting to semantic search")

    return RouterDecision(
        route=route,
        confidence=conf,
        entities=e,
        normalized_query=normalized,
        expansions=expansions,
        router="rules",
        reasons=reasons,
    )
