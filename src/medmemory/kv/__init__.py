"""Module 1: KV engine. Byte-level stores behind `KVStore` plus the record repository."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from medmemory.contracts.protocols import KVStore
from medmemory.kv.memory import MemoryKV
from medmemory.kv.records import RecordRepository
from medmemory.kv.sqlite import SQLiteKV


def open_store(
    backend: Literal["sqlite", "memory"],
    var_dir: Path,
    durability: Literal["full", "normal"] = "full",
) -> KVStore:
    if backend == "memory":
        return MemoryKV()
    if backend == "sqlite":
        return SQLiteKV(var_dir / "kv.sqlite3", durability=durability)
    raise ValueError(f"unknown KV backend {backend!r}")


__all__ = ["MemoryKV", "RecordRepository", "SQLiteKV", "open_store"]
