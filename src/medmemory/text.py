"""Shared text utilities. The generator and the citation checker MUST split sentences the same
way; otherwise a quoted sentence can be cut in two and lose its citation marker."""

from __future__ import annotations

import re

SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\-•*])|\n+")
WORD = re.compile(r"[a-z][a-z0-9-]+")


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in SENTENCE_SPLIT.split(text) if s and s.strip()]


def words(text: str) -> set[str]:
    return set(WORD.findall(text.lower()))
