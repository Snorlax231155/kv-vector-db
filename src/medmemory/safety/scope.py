"""Patient-scope resolution: the first scoping layer (see docs/architecture.md).

Clinician persona: the UI sets an active patient. A query may name that patient, or name a
patient when none is active (that patient becomes the scope). It may never name a different
patient, or more than one. Violations are refused before any retrieval runs.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class ScopeDecision:
    scope: str | None
    violation: str | None = None


def resolve_scope(
    request_scope: str | None, named: list[str], exists: Callable[[str], bool]
) -> ScopeDecision:
    named = sorted(set(named))
    if len(named) > 1:
        return ScopeDecision(
            None,
            f"This question mentions several patients ({', '.join(named)}). "
            "MedMemory answers about one patient at a time.",
        )
    if request_scope and named and named[0] != request_scope:
        return ScopeDecision(
            None,
            f"This question is about {named[0]}, but the active patient is {request_scope}. "
            f"Switch the active patient to {named[0]} to continue.",
        )
    scope = request_scope or (named[0] if named else None)
    if scope is not None and not exists(scope):
        return ScopeDecision(None, f"No patient with ID {scope} exists in this dataset.")
    return ScopeDecision(scope)
