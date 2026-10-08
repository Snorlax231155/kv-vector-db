"""Red-flag and out-of-scope pre-check. Runs before routing; a red flag short-circuits the
whole pipeline (no retrieval, no LLM) and returns a fixed safe message.

Design choices:
* High recall over precision. A false positive costs one extra sentence of safety advice;
  a false negative could cost much more. The gold set has ≥10 red-flag cases and the CI
  gate requires safety recall = 1.0.
* Record questions are NOT emergencies: "Has P0005 reported chest pain radiating to the
  arm?" is a clinician reading the chart. We suppress a red flag only when the query is
  framed as a record lookup *and* has no first-person or 'right now' cue. Bare symptom
  descriptions ("chest pain spreading to my jaw") always trigger.
"""

from __future__ import annotations

import re

from medmemory.contracts.schemas import SafetyResult
from medmemory.ingest.catalog import CONDITIONS, DRUGS, LABS

_FLAGS: dict[str, list[str]] = {
    "cardiac": [
        r"(?:chest (?:pain|pressure|tightness|discomfort)|pain in (?:my|the|his|her) chest|chest (?:hurts|is hurting|feels tight))[^.?!]{0,60}(?:radiat\w*|spread\w*|going|moving|shoot\w*) (?:down |up |in)?(?:to |into )?(?:my |the |his |her )?(?:left |right )?(?:arm|jaw|neck|back|shoulder)",
        r"(?:chest (?:pain|pressure|tightness|discomfort)|pain in (?:my|the|his|her) chest|chest (?:hurts|is hurting|feels tight))[^.?!]{0,60}(?:sweat\w*|short(?:ness)? of breath|can'?t breathe|faint\w*|nause\w*|vomit\w*)",
        r"crushing (?:chest )?(?:pain|pressure)",
        r"(?:having|think i'?m having|is having) a (?:heart attack|myocardial infarction)",
    ],
    "stroke": [
        r"face (?:is )?droop\w*|droop\w* face",
        r"slurr\w* (?:speech|words)|can'?t (?:speak|talk) (?:properly|clearly)|sudden(?:ly)? (?:can'?t|cannot) (?:speak|talk)",
        r"sudden\w* (?:weakness|numbness)[^.?!]{0,40}(?:one side|left side|right side|arm|leg|face)",
        r"(?:can'?t|cannot) (?:move|lift|feel) (?:my |his |her )?(?:left|right|one) (?:arm|leg|side)",
        r"worst headache of (?:my|his|her|their) life",
        r"(?:having|is having) a stroke",
        r"sudden(?:ly)? (?:lost|loss of) (?:vision|sight)",
    ],
    "self_harm": [
        r"suicid\w*",
        r"kill (?:my ?self|himself|herself|themselves)",
        r"end(?:ing)? (?:my|his|her|their) (?:own )?life",
        r"(?:don'?t|do not|can'?t) see (?:the|any) point (?:anymore|any more|in living|of living|in going on)",
        r"better off (?:dead|without me)",
        r"(?:want|wants|going) to die",
        r"self[- ]?harm\w*|hurt (?:my ?self|himself|herself)|cut(?:ting)? (?:my ?self|myself)",
        r"no reason to (?:live|go on)",
    ],
    "overdose": [
        r"overdos\w*",
        r"took (?:too many|a (?:whole )?bottle of|an entire bottle of|\d{2,} )(?:pills|tablets|capsules)?",
        r"(?:took|injected|gave (?:myself|him|her)) (?:a bunch of|too much|way too much|extra|double|an extra|twice (?:my|the))\s+(?:\w+\s+){0,2}(?:insulin|pills?|tablets?|doses?|medicines?|medications?|units)",
        r"swallowed (?:a (?:whole )?bottle|too many|all (?:my|the|of))",
        r"poison\w*",
    ],
    "anaphylaxis": [
        r"throat (?:is )?(?:closing|swelling|tight\w*)",
        r"(?:tongue|lips?) (?:is |are )?swell\w*[^.?!]{0,30}(?:breath|swallow)",
        r"anaphyla\w*",
    ],
    "breathing": [
        r"(?:can'?t|cannot|unable to|struggling to|hard to) breathe",
        r"(?:lips|face) (?:are |is |turning )?(?:blue|grey|gray)",
        r"not breathing|stopped breathing|chok\w+",
    ],
    "bleeding": [
        r"(?:won'?t|will not|doesn'?t|does not) stop bleeding|bleeding (?:heavily|a lot|won'?t stop)",
        r"(?:vomit\w*|cough\w*|throw\w* up) (?:up )?blood",
    ],
    "unresponsive": [
        r"unconscious|unresponsive|won'?t wake up|passed out and",
        r"(?:having|is having) a seizure|seizing",
    ],
}
_COMPILED = {cat: [re.compile(p, re.IGNORECASE) for p in pats] for cat, pats in _FLAGS.items()}

FIRST_PERSON = re.compile(r"\b(i|i'm|im|i am|i've|my|me|we|we're|our)\b", re.IGNORECASE)
URGENT_NOW = re.compile(
    r"\b(right now|now|currently|just (?:took|started|happened)|suddenly|at the moment|tonight|today|help)\b",
    re.IGNORECASE,
)
RECORD_FRAME = re.compile(
    r"\b(history of|hx of|documented|notes?|records?|chart|report(?:ed|s)?|did (?:p\d{4,6}|the patient|he|she|they)|"
    r"has (?:p\d{4,6}|the patient|he|she|they) (?:ever|had|been)|any (?:mention|episodes?|record)|"
    r"previous(?:ly)?|prior|past|screen(?:ing|ed)? for|risk (?:of|for)|assessment)\b|\bp\d{4,6}\b",
    re.IGNORECASE,
)
MEDICAL_HINT = re.compile(
    r"\b(patient|p\d{4,6}|medic\w*|meds|drug|warnings?|boxed|adverse|contraindicat\w*|interact\w*|prescri\w*|dose|dosage|tablet|pill|label|lab|test|result|blood|heart|kidney|liver|lung|"
    r"diabet\w*|pressure|cholesterol|thyroid|asthma|copd|pain|symptom\w*|side effect\w*|diagnos\w*|condition\w*|"
    r"disease|treat\w*|therap\w*|allerg\w*|icd|rxcui|rxnorm|code|health|clinical|clinic|doctor|nurse|hospital|"
    r"infection|fever|cough|rash|headache|migraine|stroke|weight|bmi|glucose|a1c|egfr|creatinine|insulin|statin|"
    r"note|notes|visit|encounter|history|depress\w*|anxiety|sleep|bone|anemi\w*|reflux|vaccine|surgery|"
    r"ldl|tsh|inr|bnp|potassium|sodium|hemoglobin|arthritis|obes\w*|apnea|fibrillation|failure|hypertension|"
    r"pregnan\w*|breast|alcohol|smok\w*|exercise|diet|nutrition)\b",
    re.IGNORECASE,
)

_VOCAB = sorted(
    {
        *DRUGS,
        *LABS,
        *CONDITIONS,
        *(a for c in CONDITIONS.values() for a in c.aliases if len(a) >= 3),
    },
    key=len,
    reverse=True,
)
CATALOG_HINT = re.compile(r"\b(" + "|".join(re.escape(v) for v in _VOCAB) + r")\b", re.IGNORECASE)

PATIENT_REF = re.compile(
    r"\b(she|he|her|his|him|they|their|them|patient|this patient|pt|p\d{4,6})\b", re.IGNORECASE
)


def looks_clinical(query: str) -> bool:
    return bool(MEDICAL_HINT.search(query) or CATALOG_HINT.search(query))


EMERGENCY_MESSAGE = (
    "This could be a medical emergency. If you or someone near you may be having a heart attack, a stroke, "
    "a severe allergic reaction, trouble breathing, heavy bleeding or an overdose, call your local emergency "
    "number now (for example 112 in India or 911 in the US) or go to the nearest emergency department. "
    "Do not wait for an online answer. MedMemory is an educational prototype and cannot help in an emergency."
)
SELF_HARM_MESSAGE = (
    "It sounds like you, or someone you're with, may be thinking about self-harm or suicide. You don't have to "
    "handle this alone. Please reach out right now: in India, call Tele-MANAS on 14416 (24/7); in the US, call "
    "or text 988. If anyone is in immediate danger, call your local emergency number (112 in India, 911 in the US). "
    "MedMemory is an educational prototype and cannot provide crisis support."
)
OUT_OF_SCOPE_MESSAGE = (
    "MedMemory answers questions about the synthetic patient records, FDA drug labels and MedlinePlus health "
    "topics in its knowledge base. That question looks outside that scope."
)


def precheck(query: str, clinical_context: bool = False) -> SafetyResult:
    """`clinical_context` is True when a patient is active: then nothing is out of scope."""
    q = " ".join(query.split())
    hits: dict[str, list[str]] = {}
    for cat, patterns in _COMPILED.items():
        for p in patterns:
            m = p.search(q)
            if m:
                hits.setdefault(cat, []).append(m.group(0))
    if hits:
        first_person = bool(FIRST_PERSON.search(q))
        urgent = bool(URGENT_NOW.search(q))
        record_frame = bool(RECORD_FRAME.search(q))
        if record_frame and not first_person and not urgent:
            return SafetyResult(
                verdict="ok",
                category="record_query_on_red_flag_topic",
                matched=[m for v in hits.values() for m in v],
            )
        category = "self_harm" if "self_harm" in hits else next(iter(hits))
        return SafetyResult(
            verdict="red_flag",
            category=category,
            matched=[m for v in hits.values() for m in v],
            message=SELF_HARM_MESSAGE if category == "self_harm" else EMERGENCY_MESSAGE,
        )
    if not clinical_context and not MEDICAL_HINT.search(q) and not CATALOG_HINT.search(q):
        return SafetyResult(
            verdict="out_of_scope", category="non_clinical", message=OUT_OF_SCOPE_MESSAGE
        )
    return SafetyResult(verdict="ok")
