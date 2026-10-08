"""ICD-10-CM codes (US CDC/CMS, public domain) via the NLM Clinical Tables API.

The seed contains every code under the 3-character categories our synthetic patients use
(a few hundred codes). For the full ~74k-code list, `load_cms_file` parses the CMS
"Code Descriptions in Tabular Order" text file you download yourself
(https://www.cms.gov/medicare/coding-billing/icd-10-codes). We use ICD-10-CM, not WHO
ICD-10, which is under a different license.
"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import quote

from medmemory.ingest.sources.http import CachedFetcher

API = "https://clinicaltables.nlm.nih.gov/api/icd10cm/v3/search"


def fetch_category(
    fetcher: CachedFetcher, category: str, max_list: int = 200
) -> list[dict[str, str]]:
    url = f"{API}?sf=code&terms={quote(category)}&maxList={max_list}&df=code,name"
    data = fetcher.get_json(url)
    if not data:
        return []
    out = []
    for code, name in data[3]:
        if code.startswith(category):
            out.append({"code": code, "name": name})
    return out


def fetch_code(fetcher: CachedFetcher, code: str) -> dict[str, str] | None:
    url = f"{API}?sf=code&terms={quote(code)}&maxList=5&df=code,name"
    data = fetcher.get_json(url)
    if not data:
        return None
    for c, name in data[3]:
        if c == code:
            return {"code": c, "name": name}
    return None


def load_cms_file(path: Path) -> list[dict[str, str]]:
    """Parse CMS `icd10cm-codes-YYYY.txt`: 'A000    Cholera due to ...' (no dots)."""
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw, name = line[:8].strip(), line[8:].strip()
        code = raw if len(raw) <= 3 else f"{raw[:3]}.{raw[3:]}"
        out.append({"code": code, "name": name})
    return out
