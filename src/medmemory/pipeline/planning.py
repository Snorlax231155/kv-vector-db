"""Which vector namespaces to search, with which metadata filters, for a routed query."""

from __future__ import annotations

import re

from medmemory.contracts.schemas import MetadataFilter, Namespace, QueryRequest, RouterDecision
from medmemory.router.rules import KNOWLEDGE_RE

_NOTES_ONLY = re.compile(
    r"\b(notes?|documented|report\w*|complain\w*|said|mention\w*|visit|hpi|history of present illness)\b"
)
_DRUGISH = re.compile(
    r"\b(side effects?|adverse|dose|dosage|dosing|label|interact\w*|contraindicat\w*|warnings?|medication|drug|tablet|pill|overdos\w*)\b"
)


def plan_namespaces(
    decision: RouterDecision, scope: str | None, req: QueryRequest
) -> list[Namespace]:
    if req.namespaces:
        return list(dict.fromkeys(req.namespaces))
    q = decision.normalized_query
    e = decision.entities
    knowledge = bool(KNOWLEDGE_RE.search(q)) or bool(_DRUGISH.search(q))
    if scope and _NOTES_ONLY.search(q) and not knowledge:
        return [Namespace.PATIENT_NOTES]
    out: list[Namespace] = []
    if scope:
        out.append(Namespace.PATIENT_NOTES)
    if e.drugs or _DRUGISH.search(q):
        out.append(Namespace.DRUG_LABELS)
    if e.conditions or not e.drugs or not scope:
        out.append(Namespace.GUIDELINES)
    if not scope and Namespace.DRUG_LABELS not in out:
        out.insert(0, Namespace.DRUG_LABELS)
    return list(dict.fromkeys(out))


def namespace_filters(decision: RouterDecision) -> dict[Namespace, MetadataFilter]:
    e = decision.entities
    out: dict[Namespace, MetadataFilter] = {}
    if e.drugs:
        out[Namespace.DRUG_LABELS] = {"drug": {"$in": list(e.drugs)}}
    tr = e.time_range
    if tr and (tr.start or tr.end):
        rng: dict[str, int] = {}
        if tr.start:
            rng["$gte"] = int(tr.start.strftime("%Y%m%d"))
        if tr.end:
            rng["$lte"] = int(tr.end.strftime("%Y%m%d"))
        out[Namespace.PATIENT_NOTES] = {"date_num": rng}
    return out
