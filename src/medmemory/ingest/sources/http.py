"""Polite, cached HTTP GET for public data sources.

Responses are cached under `data/raw/http/` keyed by URL hash, so re-running ingestion is
idempotent and works offline after the first fetch. The cache is gitignored; the derived
seed files under `data/seed/` are what we commit.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any

import httpx

log = logging.getLogger(__name__)

USER_AGENT = "MedMemory-edu-prototype/0.1 (university capstone; synthetic+public data only)"


class CachedFetcher:
    def __init__(
        self, cache_dir: Path, min_interval_s: float = 0.35, offline: bool = False
    ) -> None:
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.min_interval_s = min_interval_s
        self.offline = offline
        self._last = 0.0
        self._client = httpx.Client(
            headers={"User-Agent": USER_AGENT}, timeout=30.0, follow_redirects=True
        )

    def _path(self, url: str) -> Path:
        return self.cache_dir / (hashlib.sha256(url.encode()).hexdigest()[:24] + ".cache")

    def get_text(self, url: str) -> str:
        path = self._path(url)
        if path.exists():
            return path.read_text(encoding="utf-8")
        if self.offline:
            raise FileNotFoundError(f"offline and not cached: {url}")
        wait = self.min_interval_s - (time.monotonic() - self._last)
        if wait > 0:
            time.sleep(wait)
        for attempt in range(4):
            try:
                resp = self._client.get(url)
                self._last = time.monotonic()
                if resp.status_code == 404:
                    text = ""
                    break
                if resp.status_code in (429, 500, 502, 503):
                    time.sleep(1.5 * (attempt + 1))
                    continue
                resp.raise_for_status()
                text = resp.text
                break
            except httpx.TransportError as exc:
                log.warning("fetch failed (%s), retrying: %s", exc, url)
                time.sleep(1.5 * (attempt + 1))
        else:
            raise RuntimeError(f"giving up on {url}")
        path.write_text(text, encoding="utf-8")
        return text

    def get_json(self, url: str) -> Any:
        text = self.get_text(url)
        return json.loads(text) if text else None

    def close(self) -> None:
        self._client.close()
