"""Module 3: query router. Normalization, clinical entities, and rule-based query routing."""

from medmemory.router.entities import Lexicon, extract
from medmemory.router.normalize import normalize
from medmemory.router.rules import RulesRouter

__all__ = [
    "Lexicon",
    "RulesRouter",
    "extract",
    "normalize",
]
