"""Deterministic post-generation citation check (ADR-0006).

A sentence is *supported* when:
  1. it carries at least one citation marker [S#] (or [S1, S3]),
  2. every marker refers to evidence that was actually provided, and
  3. every number in it (doses, lab values, dates, percentages) appears in the text of at
     least one cited source.
Hedging/disclaimer sentences without numbers are exempt (not counted either way).
Unsupported sentences are removed from the answer that reaches the user.

Limitation (documented): a claim with no numbers that paraphrases a source *wrongly* passes
rule 3. The eval's human spot-check estimates how often that happens.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from medmemory.contracts.schemas import Citation, CitationCheck, Evidence
from medmemory.text import SENTENCE_SPLIT

MARKER = re.compile(r"\[(S\d+(?:\s*,\s*S\d+)*)\]")
NUMBER = re.compile(r"(?<![A-Za-z])(\d{4}-\d{2}-\d{2}|\d+(?:[.,]\d+)?)")
EXEMPT = re.compile(
    r"(not medical advice|educational|consult|clinician|healthcare (?:provider|professional)|pharmacist|"
    r"doctor|information consistent with|based on the retrieved|the retrieved (?:sources|records|evidence)|"
    r"cannot (?:provide|determine|confirm)|does not replace|i don't have enough information|"
    r"not (?:stated|mentioned|documented) in the (?:sources|records|evidence))",
    re.IGNORECASE,
)
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\-•*])|\n+")


@dataclass
class CheckedAnswer:
    text: str
    check: CitationCheck
    citations: list[Citation]


def _attach_trailing_markers(text: str) -> str:
    # "...twice daily. [S1]" -> "...twice daily [S1]."  so markers stay with their sentence
    return re.sub(
        r"([.!?])\s*((?:\[S\d+(?:\s*,\s*S\d+)*\]\s*)+)",
        lambda m: " " + m.group(2).strip() + m.group(1),
        text,
    )


def split_sentences(text: str) -> list[str]:
    return [
        s.strip()
        for s in SENTENCE_SPLIT.split(_attach_trailing_markers(text.strip()))
        if s and s.strip()
    ]


def _norm_num(n: str) -> str:
    n = n.replace(",", "")
    if "." in n and "-" not in n:
        n = n.rstrip("0").rstrip(".") if n.rstrip("0").rstrip(".") else "0"
    return n


def _numbers_in(text: str) -> set[str]:
    return {_norm_num(m) for m in NUMBER.findall(text)}


def check(answer: str, evidence: Sequence[Evidence]) -> CheckedAnswer:
    by_key = {e.key: e for e in evidence}
    ev_numbers = {e.key: _numbers_in(e.text) | _numbers_in(e.title) for e in evidence}
    kept: list[str] = []
    removed: list[str] = []
    bad_keys: set[str] = set()
    bad_numbers: list[str] = []
    counted = supported = 0
    used: list[str] = []

    for sent in split_sentences(answer):
        keys = [k.strip() for grp in MARKER.findall(sent) for k in grp.split(",")]
        body = MARKER.sub("", sent).strip()
        if not body or re.fullmatch(r"[\W_]+", body):
            continue
        nums = _numbers_in(body)
        if not keys:
            if EXEMPT.search(body) and not nums:
                kept.append(sent)
                continue
            counted += 1
            removed.append(sent)
            continue
        counted += 1
        invalid = [k for k in keys if k not in by_key]
        if invalid:
            bad_keys.update(invalid)
            removed.append(sent)
            continue
        available = set().union(*(ev_numbers[k] for k in keys))
        missing = sorted(n for n in nums if n not in available)
        if missing:
            bad_numbers.extend(missing)
            removed.append(sent)
            continue
        supported += 1
        kept.append(sent)
        for k in keys:
            if k not in used:
                used.append(k)

    rate = supported / counted if counted else 0.0
    citations = [
        Citation(
            key=k,
            source_id=by_key[k].source_id,
            origin=by_key[k].origin,
            title=by_key[k].title,
            snippet=by_key[k].text[:280],
        )
        for k in used
    ]
    return CheckedAnswer(
        text=" ".join(kept),
        check=CitationCheck(
            passed=counted > 0 and supported == counted,
            grounding_rate=round(rate, 4),
            total_sentences=counted,
            supported_sentences=supported,
            removed_sentences=removed,
            unsupported_numbers=sorted(set(bad_numbers)),
            invalid_citation_keys=sorted(bad_keys),
        ),
        citations=citations,
    )
