"""Evaluate Pinecone-style metadata filters locally, so the in-memory and FAISS stores
behave exactly like Pinecone for the operators we use.

Supported: implicit equality {"k": "v"}, $eq, $ne, $in, $nin, $gt, $gte, $lt, $lte,
$exists, and top-level $and / $or. List-valued metadata matches $eq/$in if any element does
(Pinecone semantics).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def _cmp(value: Any, op: str, arg: Any) -> bool:
    if op == "$exists":
        return (value is not None) == bool(arg)
    if value is None:
        return op in ("$ne", "$nin")
    values = value if isinstance(value, list) else [value]
    if op == "$eq":
        return any(v == arg for v in values)
    if op == "$ne":
        return all(v != arg for v in values)
    if op == "$in":
        return any(v in arg for v in values)
    if op == "$nin":
        return all(v not in arg for v in values)
    try:
        if op == "$gt":
            return any(v > arg for v in values)
        if op == "$gte":
            return any(v >= arg for v in values)
        if op == "$lt":
            return any(v < arg for v in values)
        if op == "$lte":
            return any(v <= arg for v in values)
    except TypeError:
        return False
    raise ValueError(f"unsupported filter operator {op}")


def matches(metadata: Mapping[str, Any], flt: Mapping[str, Any] | None) -> bool:
    if not flt:
        return True
    for key, cond in flt.items():
        if key == "$and":
            if not all(matches(metadata, sub) for sub in cond):
                return False
        elif key == "$or":
            if not any(matches(metadata, sub) for sub in cond):
                return False
        elif isinstance(cond, Mapping):
            if not all(_cmp(metadata.get(key), op, arg) for op, arg in cond.items()):
                return False
        elif not _cmp(metadata.get(key), "$eq", cond):
            return False
    return True


def and_filters(*filters: Mapping[str, Any] | None) -> dict[str, Any] | None:
    parts = [dict(f) for f in filters if f]
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    return {"$and": parts}
