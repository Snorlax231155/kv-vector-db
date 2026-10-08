"""Module 2: cache engine. Exact LRU+TTL (tagged) and single-flight request coalescing."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from medmemory.cache.lru import LRUCache
from medmemory.cache.singleflight import SingleFlight


def exact_key(
    normalized_query: str,
    patient_scope: str | None,
    filters: dict[str, Any] | None,
    namespaces: list[str] | None,
    options: dict[str, Any],
    data_version: int,
) -> str:
    """Stable key for the exact cache. Includes everything that can change the answer:
    scope, filters, namespaces, answer-affecting options and the data version."""
    payload = {
        "q": " ".join(normalized_query.lower().split()),
        "scope": patient_scope,
        "filters": sorted((filters or {}).items()),
        "ns": sorted(namespaces or []),
        "opt": sorted(options.items()),
        "v": data_version,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[
        :32
    ]
    return f"ans:{patient_scope or '-'}:{digest}"


def filters_key(filters: dict[str, Any] | None, namespaces: list[str] | None) -> str:
    return json.dumps(
        {"f": sorted((filters or {}).items()), "ns": sorted(namespaces or [])}, default=str
    )


__all__ = ["LRUCache", "SingleFlight", "exact_key", "filters_key"]
