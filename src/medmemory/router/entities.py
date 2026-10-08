"""Deterministic entity extraction (used by every router, including the LoRA one).

Order of operations: raw query → IDs/codes (regex on the *raw* text, case matters) →
normalize abbreviations → gazetteer matching for drugs, labs and conditions on the
normalized text → time range, negation, dosage intent, record types.
"""

from __future__ import annotations

import difflib
import json
import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

from medmemory.contracts.schemas import Entities, TimeRange
from medmemory.ingest.catalog import CONDITIONS, DRUGS, LABS
from medmemory.router.normalize import normalize

# IDs may carry a possessive with or without the apostrophe: "P0003's", "P0003s"
PATIENT_RE = re.compile(r"\b[Pp](\d{4,6})(?:'?s)?\b")
ICD_RE = re.compile(r"\b([A-TV-Z]\d[0-9A-Z](?:\.[0-9A-Z]{1,4})?)\b")
RXCUI_RE = re.compile(
    r"\b(?:rx\s*cui|rxnorm|rx\s*norm)\s*(?:id|code|identifier)?\s*[:#]?\s*(\d{2,8})\b",
    re.IGNORECASE,
)
NEGATION_RE = re.compile(
    r"\b(no|not|without|denies|denied|negative for|never|isn't|wasn't|isnt|wasnt|no longer|free of|absence of)\b"
)

DOSAGE_RE = re.compile(
    r"\b(dose|doses|dosed|dosage|dosing|how much|how many (?:mg|milligrams|units)|mg|mcg|milligram|units?|maximum|max(?:imum)? daily|titrat\w*|strength|tablets?)\b"
)
RECORD_TYPE_PATTERNS: dict[str, re.Pattern[str]] = {
    "medications": re.compile(
        r"\b(medications?|medicines?|prescriptions?|prescribed|taking|on any|what (?:is|are) (?:she|he|they) on|drug list|med list)\b"
    ),
    "allergies": re.compile(r"\b(allerg\w*|reactions? to)\b"),
    "labs": re.compile(r"\b(labs?|lab results?|bloodwork|blood work|test results?)\b"),
    "conditions": re.compile(
        r"\b(conditions?|diagnos[ie]s|problem list|problems|comorbidit\w+|medical history|chronic illness\w*)\b"
    ),
    "encounters": re.compile(r"\b(visits?|encounters?|appointments?|seen (?:last|in))\b"),
    "demographics": re.compile(
        r"\b(age|how old|date of birth|dob|born|sex|gender|demographics?)\b"
    ),
}

_NUM_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "twelve": 12,
    "eighteen": 18,
    "twenty-four": 24,
}
_UNIT_DAYS = {"day": 1, "week": 7, "month": 30, "year": 365}


@dataclass
class Lexicon:
    """surface form → canonical key, per entity type. Longest match wins."""

    drugs: dict[str, str] = field(default_factory=dict)
    labs: dict[str, str] = field(default_factory=dict)
    conditions: dict[str, str] = field(default_factory=dict)
    icd_codes: set[str] = field(default_factory=set)
    _compiled: dict[str, re.Pattern[str]] = field(default_factory=dict, repr=False)

    @classmethod
    def default(cls, seed_dir: Path | None = None) -> Lexicon:
        lex = cls()
        for key in DRUGS:
            lex.drugs[key] = key
        for spec in LABS.values():
            lex.labs[spec.key] = spec.key
            for a in spec.aliases:
                if len(a) >= 3:
                    lex.labs[a] = spec.key
        lex.labs["blood glucose"] = "fasting glucose"
        lex.labs["glucose"] = "fasting glucose"
        lex.labs["blood pressure"] = "systolic blood pressure"  # router adds diastolic below
        for cspec in CONDITIONS.values():
            lex.conditions[cspec.key] = cspec.key
            for a in cspec.aliases:
                if len(a) >= 3:
                    lex.conditions[a] = cspec.key
        # canonical expansions produced by normalize()
        lex.conditions.update(
            {
                "myocardial infarction": "old myocardial infarction",
                "diabetes": "type 2 diabetes",
                "chronic kidney disease": "chronic kidney disease",
                "atrial fibrillation": "atrial fibrillation",
                "stroke": "history of stroke",
                "osteoarthritis": "knee osteoarthritis",
            }
        )
        if seed_dir is not None:
            lex.add_seed(seed_dir)
        return lex

    def add_seed(self, seed_dir: Path) -> None:
        labels = seed_dir / "drug_labels.jsonl"
        if labels.exists():
            for line in labels.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                for brand in row.get("brand_names", []):
                    b = brand.lower()
                    if len(b) >= 4 and not any(ch.isdigit() for ch in b) and row["drug"] not in b:
                        self.drugs[b] = row["drug"]
        icd = seed_dir / "icd10cm.jsonl"
        if icd.exists():
            self.icd_codes = {
                json.loads(line)["code"] for line in icd.read_text(encoding="utf-8").splitlines()
            }
        self._compiled.clear()

    def _pattern(self, kind: str) -> re.Pattern[str]:
        if kind not in self._compiled:
            surfaces = sorted(getattr(self, kind), key=len, reverse=True)
            self._compiled[kind] = re.compile(
                r"\b(" + "|".join(re.escape(s) for s in surfaces) + r")\b"
            )
        return self._compiled[kind]

    def find(self, kind: str, text: str) -> list[tuple[str, int, int]]:
        table: dict[str, str] = getattr(self, kind)
        return [(table[m.group(1)], m.start(), m.end()) for m in self._pattern(kind).finditer(text)]


def _time_range(text: str, today: date) -> TimeRange | None:
    m = re.search(
        r"\b(?:last|past|previous|within the last)\s+(\d+|one|two|three|four|five|six|twelve|eighteen|twenty-four)?\s*(day|week|month|year)s?\b",
        text,
    )
    if m:
        n = (
            int(m.group(1))
            if m.group(1) and m.group(1).isdigit()
            else _NUM_WORDS.get(m.group(1) or "one", 1)
        )
        return TimeRange(
            text=m.group(0), start=today - timedelta(days=n * _UNIT_DAYS[m.group(2)]), end=today
        )
    m = re.search(r"\bsince\s+(20\d\d)\b", text)
    if m:
        return TimeRange(text=m.group(0), start=date(int(m.group(1)), 1, 1), end=today)
    m = re.search(r"\b(?:in|during)\s+(20\d\d)\b", text)
    if m:
        y = int(m.group(1))
        return TimeRange(text=m.group(0), start=date(y, 1, 1), end=date(y, 12, 31))
    m = re.search(r"\b(this year)\b", text)
    if m:
        return TimeRange(text=m.group(0), start=date(today.year, 1, 1), end=today)
    m = re.search(r"\b(latest|most recent|last|current|recent|newest)\b", text)
    if m:
        return TimeRange(text=m.group(0), latest_only=True)
    return None


def _negated(text: str, spans: list[tuple[str, int, int]]) -> list[str]:
    out = []
    for key, start, _ in spans:
        window = text[max(0, start - 40) : start]
        trig = list(NEGATION_RE.finditer(window))
        if trig and len(window[trig[-1].end() :].split()) <= 4:
            out.append(key)
    return sorted(set(out))


def _fuzzy_drugs(
    text: str, lexicon: Lexicon, found: list[tuple[str, int, int]], expansions: list[str]
) -> list[tuple[str, int, int]]:
    """Typo-tolerant drug matching ('levothyroxin', 'metfromin') for words the exact gazetteer
    missed. Conservative: one-word names of 6+ letters, similarity >= 0.88."""
    single = [s for s in lexicon.drugs if " " not in s and len(s) >= 6]
    out: list[tuple[str, int, int]] = []
    for m in re.finditer(r"[a-z]{6,}", text):
        if any(start <= m.start() < end for _, start, end in found):
            continue
        close = difflib.get_close_matches(m.group(0), single, n=1, cutoff=0.88)
        if close and close[0] != m.group(0):
            out.append((lexicon.drugs[close[0]], m.start(), m.end()))
            expansions.append(f"{m.group(0)} → {lexicon.drugs[close[0]]} (fuzzy)")
    return out


def extract(
    query: str, lexicon: Lexicon, today: date | None = None
) -> tuple[Entities, str, list[str]]:
    today = today or date.today()
    patient_ids = sorted({f"P{m.group(1)}" for m in PATIENT_RE.finditer(query)})
    rx = sorted({m.group(1) for m in RXCUI_RE.finditer(query)})
    icd = []
    has_icd_context = bool(
        re.search(r"\b(icd|code|diagnosis code|billing)\b", query, re.IGNORECASE)
    )
    for m in ICD_RE.finditer(query):
        code = m.group(1).upper()
        if code in lexicon.icd_codes or "." in code or has_icd_context:
            icd.append(code)
    icd = sorted(set(icd))

    normalized, expansions = normalize(query)
    drug_spans = lexicon.find("drugs", normalized)
    lab_spans = lexicon.find("labs", normalized)
    cond_spans = lexicon.find("conditions", normalized)
    labs = sorted({k for k, _, _ in lab_spans})
    if (
        "systolic blood pressure" in labs
        and "blood pressure" in normalized
        and "systolic" not in normalized
    ):
        labs = sorted({*labs, "diastolic blood pressure"})
    # "blood pressure" as a condition word ("high blood pressure") was normalized to hypertension
    record_types = sorted(rt for rt, pat in RECORD_TYPE_PATTERNS.items() if pat.search(normalized))
    drug_spans += _fuzzy_drugs(normalized, lexicon, drug_spans, expansions)
    entities = Entities(
        patient_ids=patient_ids,
        icd10_codes=icd,
        rxnorm_codes=rx,
        drugs=sorted({k for k, _, _ in drug_spans}),
        labs=labs,
        conditions=sorted({k for k, _, _ in cond_spans}),
        time_range=_time_range(normalized, today),
        negated=_negated(normalized, drug_spans + lab_spans + cond_spans),
        dosage_intent=bool(DOSAGE_RE.search(normalized)),
        record_types=record_types,
    )
    return entities, normalized, expansions
