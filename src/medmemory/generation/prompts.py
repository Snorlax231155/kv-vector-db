"""Prompt for grounded, cited, non-definitive answers. Shared by the real and mock models."""

from __future__ import annotations

from collections.abc import Sequence

from langchain_core.prompts import ChatPromptTemplate

from medmemory.contracts.schemas import Evidence, Route

INSUFFICIENT = "INSUFFICIENT_EVIDENCE"

SYSTEM = """You are MedMemory, an educational clinical information assistant. You are not a \
clinician and you do not give medical advice.

Answer ONLY from the numbered SOURCES. Rules:
1. End every sentence with one or more citation markers naming the sources it relies on, e.g. [S1] or [S2, S4].
2. Never state a fact, number, dose, date or lab value that does not appear in a cited source. Dosages and \
drug facts must come from FDA label sources or the patient's medication record, never from your own knowledge.
3. Do not diagnose or recommend treatment. Use wording such as "the records show", "the label states", \
"information consistent with".
4. If the sources do not answer the question, reply with exactly: {insufficient}
5. Be concise: at most 5 sentences. Plain text, no headings, no bullet lists."""

HUMAN = """QUESTION: {question}
ROUTE: {route}

SOURCES:
{sources}"""

PROMPT = ChatPromptTemplate.from_messages([("system", SYSTEM), ("human", HUMAN)])


def format_sources(evidence: Sequence[Evidence]) -> str:
    blocks = []
    for e in evidence:
        header = f"[{e.key}] ({e.origin.value} · {e.title})"
        blocks.append(f"{header}\n{e.text.strip()}")
    return "\n\n".join(blocks)


def prompt_inputs(question: str, evidence: Sequence[Evidence], route: Route) -> dict[str, str]:
    return {
        "question": question,
        "route": route.value,
        "sources": format_sources(evidence),
        "insufficient": INSUFFICIENT,
    }
