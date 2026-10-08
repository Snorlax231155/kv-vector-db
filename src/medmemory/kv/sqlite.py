"""SQLite KV backend (default). Durable via write-ahead logging.

Durability: `journal_mode=WAL` + `synchronous=FULL` means a committed `put` survives a
process crash *and* an OS crash/power loss (the WAL is fsynced on every commit).
`durability="normal"` (`synchronous=NORMAL`) survives process crashes but may lose the last
transactions on power loss, in exchange for much faster writes. The crash-recovery test
hard-kills a writer process and checks that every acknowledged write is readable.

Prefix scan uses a half-open range [prefix, prefix + U+10FFFF) on the primary key, so it is
an index range scan, not LIKE.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Literal

from medmemory.contracts.protocols import KVOp
from medmemory.errors import KVError

_SCHEMA = """
CREATE TABLE IF NOT EXISTS kv (
    key        TEXT PRIMARY KEY,
    value      BLOB NOT NULL,
    expires_at REAL
) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS kv_expires ON kv(expires_at) WHERE expires_at IS NOT NULL;
"""

_PREFIX_END = "\U0010ffff"


class SQLiteKV:
    name = "sqlite"

    def __init__(
        self,
        path: Path | str,
        durability: Literal["full", "normal"] = "full",
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.path = Path(path)
        if str(path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._clock = clock
        self._lock = threading.RLock()
        try:
            # isolation_level=None -> autocommit; we open explicit transactions for batches.
            self._conn = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute(f"PRAGMA synchronous={'FULL' if durability == 'full' else 'NORMAL'}")
            self._conn.execute("PRAGMA busy_timeout=5000")
            self._conn.executescript(_SCHEMA)
        except sqlite3.Error as exc:
            raise KVError(f"cannot open sqlite store at {path}: {exc}") from exc
        self._closed = False

    # ------------------------------------------------------------------ helpers

    def _now(self) -> float:
        return float(self._clock())

    def _exec(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        if self._closed:
            raise KVError("store is closed")
        try:
            return self._conn.execute(sql, params)
        except sqlite3.Error as exc:
            raise KVError(str(exc)) from exc

    # ------------------------------------------------------------------ protocol

    def get(self, key: str) -> bytes | None:
        with self._lock:
            row = self._exec(
                "SELECT value FROM kv WHERE key = ? AND (expires_at IS NULL OR expires_at > ?)",
                (key, self._now()),
            ).fetchone()
        return bytes(row[0]) if row else None

    def put(self, key: str, value: bytes, ttl_s: float | None = None) -> None:
        expires = None if ttl_s is None else self._now() + ttl_s
        with self._lock:
            self._exec(
                "INSERT INTO kv(key, value, expires_at) VALUES (?, ?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value, expires_at = excluded.expires_at",
                (key, sqlite3.Binary(value), expires),
            )

    def delete(self, key: str) -> bool:
        with self._lock:
            live = self.get(key) is not None
            self._exec("DELETE FROM kv WHERE key = ?", (key,))
        return live

    def scan(self, prefix: str, limit: int | None = None) -> Iterator[tuple[str, bytes]]:
        sql = (
            "SELECT key, value FROM kv WHERE key >= ? AND key < ? "
            "AND (expires_at IS NULL OR expires_at > ?) ORDER BY key"
        )
        params: tuple = (prefix, prefix + _PREFIX_END, self._now())
        if limit is not None:
            sql += " LIMIT ?"
            params = (*params, limit)
        with self._lock:
            rows = self._exec(sql, params).fetchall()
        return iter([(k, bytes(v)) for k, v in rows])

    def batch(self, ops: Sequence[KVOp]) -> None:
        now = self._now()
        with self._lock:
            if self._closed:
                raise KVError("store is closed")
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                for op in ops:
                    if op.op == "put":
                        if op.value is None:
                            raise KVError(f"put without value for {op.key}")
                        expires = None if op.ttl_s is None else now + op.ttl_s
                        self._conn.execute(
                            "INSERT INTO kv(key, value, expires_at) VALUES (?, ?, ?) "
                            "ON CONFLICT(key) DO UPDATE SET value = excluded.value, expires_at = excluded.expires_at",
                            (op.key, sqlite3.Binary(op.value), expires),
                        )
                    else:
                        self._conn.execute("DELETE FROM kv WHERE key = ?", (op.key,))
                self._conn.execute("COMMIT")
            except Exception as exc:
                self._conn.execute("ROLLBACK")
                if isinstance(exc, KVError):
                    raise
                raise KVError(f"batch failed and was rolled back: {exc}") from exc

    def count(self, prefix: str = "") -> int:
        with self._lock:
            row = self._exec(
                "SELECT COUNT(*) FROM kv WHERE key >= ? AND key < ? AND (expires_at IS NULL OR expires_at > ?)",
                (prefix, prefix + _PREFIX_END, self._now()),
            ).fetchone()
        return int(row[0])

    def purge_expired(self) -> int:
        with self._lock:
            cur = self._exec(
                "DELETE FROM kv WHERE expires_at IS NOT NULL AND expires_at <= ?", (self._now(),)
            )
        return cur.rowcount

    def snapshot(self, dest: Path) -> Path:
        """Online, consistent copy via the SQLite backup API (readers/writers keep going)."""
        dest.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            target = sqlite3.connect(str(dest))
            try:
                self._conn.backup(target)
            finally:
                target.close()
        return dest

    def checkpoint(self) -> None:
        """Fold the WAL into the main DB file (optional housekeeping)."""
        with self._lock:
            self._exec("PRAGMA wal_checkpoint(TRUNCATE)")

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._conn.close()
                self._closed = True
