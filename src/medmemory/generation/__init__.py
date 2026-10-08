"""Grounded answer generation (LangChain chain; Claude in real mode, extractive mock offline)."""

from medmemory.generation.chain import LangChainGenerator, build_generator
from medmemory.generation.prompts import INSUFFICIENT

__all__ = ["INSUFFICIENT", "LangChainGenerator", "build_generator"]
