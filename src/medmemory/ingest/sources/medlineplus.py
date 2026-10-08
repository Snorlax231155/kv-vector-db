"""MedlinePlus health-topic summaries via the NLM web service.

Licensing: only NLM-authored content is public domain. Health-topic summaries qualify.
A.D.A.M. encyclopedia articles and ASHP drug monographs are copyrighted and must NOT be
ingested; we only request `db=healthTopics` and keep the NLM-written FullSummary field.
Attribution: "Courtesy of MedlinePlus from the National Library of Medicine".

Note on naming: these populate the `guidelines` namespace, but they are consumer health
summaries, not clinical practice guidelines. The README says so.
"""

from __future__ import annotations

import html
import re
import xml.etree.ElementTree as ET
from typing import Any
from urllib.parse import quote_plus

from medmemory.ingest.sources.http import CachedFetcher

API = "https://wsearch.nlm.nih.gov/ws/query"
ATTRIBUTION = "Courtesy of MedlinePlus from the National Library of Medicine"

# (search term, expected page slug)
TOPICS: list[tuple[str, str]] = [
    ("diabetes type 2", "diabetestype2.html"),
    ("high blood pressure", "highbloodpressure.html"),
    ("cholesterol", "cholesterol.html"),
    ("chronic kidney disease", "chronickidneydisease.html"),
    ("asthma", "asthma.html"),
    ("copd", "copd.html"),
    ("atrial fibrillation", "atrialfibrillation.html"),
    ("heart failure", "heartfailure.html"),
    ("coronary artery disease", "coronaryarterydisease.html"),
    ("heart attack", "heartattack.html"),
    ("hypothyroidism", "hypothyroidism.html"),
    ("depression", "depression.html"),
    ("anxiety", "anxiety.html"),
    ("gerd", "gerd.html"),
    ("osteoarthritis", "osteoarthritis.html"),
    ("obesity", "obesity.html"),
    ("sleep apnea", "sleepapnea.html"),
    ("stroke", "stroke.html"),
    ("osteoporosis", "osteoporosis.html"),
    ("anemia", "anemia.html"),
    ("migraine", "migraine.html"),
    ("blood thinners", "bloodthinners.html"),
    ("diabetes medicines", "diabetesmedicines.html"),
    ("statins", "statins.html"),
]


def _strip(fragment: str) -> str:
    text = html.unescape(fragment)
    text = re.sub(r"</(p|li|h\d)>", "\n", text)
    text = re.sub(r"<li>", "- ", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n\s*\n+", "\n", text).strip()


def fetch_topic(fetcher: CachedFetcher, term: str, slug: str) -> dict[str, Any] | None:
    url = f"{API}?db=healthTopics&term={quote_plus(term)}&retmax=10"
    raw = fetcher.get_text(url)
    if not raw:
        return None
    root = ET.fromstring(raw)
    for doc in root.iter("document"):
        page = doc.get("url", "")
        if not page.endswith("/" + slug):
            continue
        fields: dict[str, list[str]] = {}
        for content in doc.findall("content"):
            fields.setdefault(content.get("name", ""), []).append(content.text or "")
        org = " ".join(fields.get("organizationName", []))
        if "National Library of Medicine" not in org:
            return None  # only NLM-authored content is public domain
        summary = _strip(" ".join(fields.get("FullSummary", [])))
        if not summary:
            return None
        return {
            "topic_id": slug.removesuffix(".html"),
            "title": _strip(" ".join(fields.get("title", [])[:1])),
            "url": page,
            "also_called": sorted({_strip(a) for a in fields.get("altTitle", [])}),
            "summary": summary,
            "attribution": ATTRIBUTION,
        }
    return None
