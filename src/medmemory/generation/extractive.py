"""`ExtractiveChatModel`: the mock LLM. A real LangChain `BaseChatModel`, so mock mode runs the
exact same prompt → model → parser chain (and streaming path) as production.

It reads the SOURCES block out of the prompt, scores each source sentence by term overlap
with the QUESTION, and answers by quoting the best sentences verbatim with their [S#]
marker. Quoting verbatim means it can never invent a number. That makes it a faithful
baseline, not a smart one: no synthesis, no reasoning across sources.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import AsyncIterator, Iterator
from typing import Any

from langchain_core.callbacks import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

from medmemory.generation.prompts import INSUFFICIENT
from medmemory.text import split_sentences
from medmemory.vector.embedders import tokenize

_BLOCK = re.compile(
    r"^\[(S\d+)\] \((\w+) · (.*?)\)\n(.*?)(?=\n\n\[S\d+\] \(|\Z)", re.MULTILINE | re.DOTALL
)
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])|\n+")
_QUESTION_NOISE = {
    "what",
    "which",
    "latest",
    "recent",
    "current",
    "present",
    "list",
    "show",
    "tell",
    "know",
    "any",
    "there",
    "all",
    "active",
}

_RECORD_KIND_TERMS = {
    "condition": {"condition", "diagnosi", "problem", "illness", "hx", "histor"},
    "medication": {"medication", "drug", "med", "meds", "prescript", "prescrib", "take", "taking", "regimen"},
    "lab": {"lab", "test", "result", "reading", "value", "level", "bloodwork"},
    "encounter": {"encounter", "visit", "appointment", "seen"},
    "allergy": {"allergi", "allergy", "allergic", "reaction"},
    "demographics": {"demograph", "age", "sex", "dob", "birth"},
}


def parse_prompt(text: str) -> tuple[str, list[tuple[str, str, str, str]]]:
    q = re.search(r"QUESTION:\s*(.*)", text)
    question = q.group(1).strip() if q else ""
    sources = [
        (m.group(1), m.group(2), m.group(3), m.group(4).strip()) for m in _BLOCK.finditer(text)
    ]
    return question, sources


def compose(question: str, sources: list[tuple[str, str, str, str]], max_sentences: int = 6) -> str:
    q_terms = {t for t in tokenize(question) if t not in _QUESTION_NOISE}
    # Off-topic evidence is caught before generation (safety.sufficiency.unknown_subject);
    # here we only refuse when no source sentence shares a single term with the question.
    candidates: list[tuple[float, int, str, str]] = []
    for order, (key, origin, title, body) in enumerate(sources):
        title_terms = set(tokenize(title))
        kind_match = False
        if origin == "kv":
            for kind, kterms in _RECORD_KIND_TERMS.items():
                if kind in title_terms and any(qt in kterms for qt in q_terms):
                    kind_match = True
                    break

        for sent in split_sentences(body):
            sent = sent.strip(" -•")
            if len(sent.split()) < 3:
                continue
            terms = set(tokenize(sent))
            overlap = len(q_terms & (terms | title_terms)) / (len(q_terms) or 1)
            if overlap == 0 and not kind_match:
                continue
            if overlap == 0 and kind_match:
                overlap = 0.5
            # prefer exact structured facts, then earlier (better-ranked) sources, then shorter
            score = (
                overlap
                + (0.25 if origin == "kv" else 0.0)
                - 0.01 * order
                - 0.001 * len(sent.split())
            )
            candidates.append((score, order, key, sent))
    if not candidates:
        return INSUFFICIENT
    candidates.sort(key=lambda c: c[0], reverse=True)
    picked: list[tuple[int, str, str]] = []
    per_source: dict[str, int] = {}
    for _, order, key, sent in candidates:
        if per_source.get(key, 0) >= 2 or any(sent == p[2] for p in picked):
            continue
        picked.append((order, key, sent))
        per_source[key] = per_source.get(key, 0) + 1
        if len(picked) >= max_sentences:
            break
    picked.sort(key=lambda p: p[0])
    lines = ["Based on the retrieved sources, here is information consistent with your question."]
    for _, key, sent in picked:
        sent = sent.rstrip()
        end = sent[-1] if sent[-1] in ".!?" else "."
        body = sent[:-1] if sent[-1] in ".!?" else sent
        lines.append(f"{body} [{key}]{end}")
    return " ".join(lines)


class ExtractiveChatModel(BaseChatModel):
    """Deterministic, offline stand-in for a chat LLM."""

    stream_delay_s: float = 0.0
    max_sentences: int = 6

    @property
    def _llm_type(self) -> str:
        return "medmemory-extractive"

    def _answer(self, messages: list[BaseMessage]) -> str:
        human = next((m for m in reversed(messages) if m.type == "human"), None)
        text = human.content if human and isinstance(human.content, str) else ""
        question, sources = parse_prompt(text)
        return compose(question, sources, self.max_sentences)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=self._answer(messages)))]
        )

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        for piece in re.findall(r"\S+\s*", self._answer(messages)):
            yield ChatGenerationChunk(message=AIMessageChunk(content=piece))

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        for piece in re.findall(r"\S+\s*", self._answer(messages)):
            if self.stream_delay_s:
                await asyncio.sleep(self.stream_delay_s)
            chunk = ChatGenerationChunk(message=AIMessageChunk(content=piece))
            if run_manager:
                await run_manager.on_llm_new_token(piece, chunk=chunk)
            yield chunk
