"""RxNorm ingredient identifiers via the NLM RxNav REST API (no UMLS license needed for
ingredient-level RxCUIs). Used so `drug:rx:{rxcui}` keys are canonical identifiers rather
than label-specific product codes."""

from __future__ import annotations

from urllib.parse import quote

from medmemory.ingest.sources.http import CachedFetcher

API = "https://rxnav.nlm.nih.gov/REST"


def ingredient_rxcui(fetcher: CachedFetcher, name: str) -> str | None:
    data = fetcher.get_json(f"{API}/rxcui.json?name={quote(name)}&search=0")
    ids = (data or {}).get("idGroup", {}).get("rxnormId") or []
    return ids[0] if ids else None
