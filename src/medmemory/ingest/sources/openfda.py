"""openFDA drug labels (public domain, US FDA): the *only* source of drug facts and doses.

For each catalog drug we pick one current prescription label whose generic name matches
exactly (ignoring salt forms), so combination products like 'sitagliptin and metformin'
are not confused with metformin.
"""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import quote

from medmemory.ingest.sources.http import CachedFetcher

API = "https://api.fda.gov/drug/label.json"

# Label fields we keep, in display order, with readable section titles.
SECTIONS: dict[str, str] = {
    "boxed_warning": "Boxed warning",
    "indications_and_usage": "Indications and usage",
    "dosage_and_administration": "Dosage and administration",
    "dosage_forms_and_strengths": "Dosage forms and strengths",
    "contraindications": "Contraindications",
    "warnings_and_cautions": "Warnings and precautions",
    "warnings": "Warnings",
    "adverse_reactions": "Adverse reactions",
    "drug_interactions": "Drug interactions",
    "use_in_specific_populations": "Use in specific populations",
}
MAX_SECTION_CHARS = 3500

_SALTS = {
    "hydrochloride",
    "hcl",
    "sodium",
    "potassium",
    "calcium",
    "besylate",
    "succinate",
    "tartrate",
    "sulfate",
    "maleate",
    "mesylate",
    "bromide",
    "propionate",
    "magnesium",
    "monohydrate",
    "dihydrate",
    "anhydrous",
    "hyclate",
    "citrate",
    "fumarate",
}


def _core_name(name: str) -> str:
    words = [w for w in re.split(r"[\s,]+", name.lower()) if w and w not in _SALTS]
    return " ".join(words)


def _clip(text: str, limit: int = MAX_SECTION_CHARS) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    cut = text[:limit]
    end = max(cut.rfind(". "), cut.rfind("; "))
    return (cut[: end + 1] if end > limit * 0.6 else cut) + " [truncated]"


def fetch_label(fetcher: CachedFetcher, generic: str, route: str = "oral") -> dict[str, Any] | None:
    query = f'openfda.generic_name:"{generic}"+AND+_exists_:indications_and_usage'
    url = f"{API}?search={quote(query, safe=':+_')}&limit=25"
    data = fetcher.get_json(url)
    if not data or "results" not in data:
        return None
    wanted = _core_name(generic)
    candidates = []
    for r in data["results"]:
        o = r.get("openfda", {})
        names = [_core_name(n) for n in o.get("generic_name", [])]
        if wanted not in names:
            continue
        rx = "HUMAN PRESCRIPTION DRUG" in o.get("product_type", [])
        has_dose = "dosage_and_administration" in r
        route_ok = route.upper() in o.get("route", [])
        candidates.append((route_ok, rx, has_dose, r.get("effective_time", ""), r))
    if not candidates:
        return None
    candidates.sort(key=lambda c: (c[0], c[1], c[2], c[3]), reverse=True)
    r = candidates[0][4]
    o = r.get("openfda", {})
    sections = {}
    for field, title in SECTIONS.items():
        if r.get(field):
            sections[title] = _clip(" ".join(r[field]))
    set_id = r.get("set_id", "")
    return {
        "drug": wanted,
        "generic_name": (o.get("generic_name") or [generic])[0].lower(),
        "brand_names": sorted({b.title() for b in o.get("brand_name", [])}),
        "rxcuis": sorted(set(o.get("rxcui", []))),
        "product_type": (o.get("product_type") or [""])[0],
        "route": [x.lower() for x in o.get("route", [])],
        "pharm_class": sorted(o.get("pharm_class_epc", [])),
        "set_id": set_id,
        "label_version": r.get("version", ""),
        "effective_time": r.get("effective_time", ""),
        "source_url": f"https://dailymed.nlm.nih.gov/dailymed/lookup.cfm?setid={set_id}",
        "sections": sections,
    }
