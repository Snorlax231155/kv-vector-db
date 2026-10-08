"""Structured JSON logging with trace IDs, and the audit log.

Every request gets a trace ID (from `X-Trace-Id` or freshly generated) stored in a
contextvar, so every log line emitted while serving it carries the same ID, and the
response echoes it back. The audit log records who asked about which patient. Patient
IDs and query text are salted-hashed, never written in clear (see the scoping section of
docs/architecture.md).
"""

from __future__ import annotations

import contextvars
import hashlib
import json
import logging
import os
import secrets
import threading
import time
from pathlib import Path
from typing import Any

trace_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("trace_id", default="-")


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": round(record.created, 3),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "trace_id": trace_id_var.get(),
        }
        extra = getattr(record, "fields", None)
        if isinstance(extra, dict):
            payload.update(extra)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO", json_logs: bool = True) -> None:
    root = logging.getLogger()
    root.handlers.clear()
    handler = logging.StreamHandler()
    handler.setFormatter(
        JsonFormatter() if json_logs else logging.Formatter("%(levelname)s %(name)s %(message)s")
    )
    root.addHandler(handler)
    root.setLevel(level)
    for noisy in ("httpx", "httpcore", "sentence_transformers", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


class AuditLog:
    """Append-only JSONL. One line per answered request; no PHI-like content in clear."""

    def __init__(self, path: Path, salt: str | None = None) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        salt_file = path.with_suffix(".salt")
        if salt is None:
            if not salt_file.exists():
                salt_file.write_text(secrets.token_hex(16), encoding="utf-8")
            salt = salt_file.read_text(encoding="utf-8").strip()
        self._salt = salt
        self._lock = threading.Lock()

    def h(self, value: str | None) -> str | None:
        if value is None:
            return None
        return hashlib.sha256((self._salt + value).encode()).hexdigest()[:16]

    def write(self, **fields: Any) -> None:
        fields.setdefault("ts", round(time.time(), 3))
        line = json.dumps(fields, default=str)
        with self._lock, self.path.open("a", encoding="utf-8") as fh:
            fh.write(line + os.linesep)

    def tail(self, n: int = 50) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        lines = self.path.read_text(encoding="utf-8").splitlines()[-n:]
        return [json.loads(x) for x in lines if x.strip()]
