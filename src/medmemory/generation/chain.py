"""The answer chain: ChatPromptTemplate | chat model | StrOutputParser.

The chat model is the only thing that differs between mock and real mode. The parsed text is
then checked by `safety.citations.check`, which turns it into a structured, grounded answer.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser

from medmemory.contracts.schemas import Evidence, Route
from medmemory.errors import GenerationError
from medmemory.generation.prompts import PROMPT, prompt_inputs


class LangChainGenerator:
    def __init__(self, model: BaseChatModel, name: str) -> None:
        self.model = model
        self.name = name
        self.chain: Any = PROMPT | model | StrOutputParser()

    async def generate(self, query: str, evidence: Sequence[Evidence], route: Route) -> str:
        try:
            return str(await self.chain.ainvoke(prompt_inputs(query, evidence, route)))
        except Exception as exc:
            raise GenerationError(f"{self.name} failed: {exc}") from exc

    async def stream(
        self, query: str, evidence: Sequence[Evidence], route: Route
    ) -> AsyncIterator[str]:
        try:
            async for piece in self.chain.astream(prompt_inputs(query, evidence, route)):
                if piece:
                    yield str(piece)
        except Exception as exc:
            raise GenerationError(f"{self.name} failed: {exc}") from exc


def build_generator(
    kind: str = "extractive",
    *,
    stream_delay_ms: int = 0,
) -> LangChainGenerator:
    if kind == "extractive":
        from medmemory.generation.extractive import ExtractiveChatModel

        return LangChainGenerator(
            ExtractiveChatModel(stream_delay_s=stream_delay_ms / 1000.0), "extractive-mock"
        )
    raise ValueError(f"unknown generator {kind!r}")
