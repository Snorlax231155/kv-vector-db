# ADR-0002: SQLite (WAL) is the default KV backend; RocksDB is optional

- Status: accepted
- Date: 2026-10-07

## Context
The team develops on Windows and macOS laptops. RocksDB Python bindings are hard to build
on Windows, while `rocksdict` ships wheels but adds a native dependency. The brief requires
durability and crash recovery.

## Decision
`SQLiteKV` is the default, using `journal_mode=WAL` and `synchronous=FULL`, a
`(key PRIMARY KEY, value, expires_at)` table, and range scans for prefixes. `RocksKV`
implements the same `KVStore` Protocol using RocksDB's own WAL and synced writes. Both
backends must pass one shared contract test suite (`tests/kv/test_contract.py`), which
includes a hard-kill crash-recovery test.

## Consequences
- Developers and CI environments get a working KV store with zero native dependencies.
- RocksDB numbers in the benchmark are real, but they are an optional path.
- TTL is lazy: expired keys are hidden on read and removed by `purge_expired()`. That is
  simpler than RocksDB's compaction-filter TTL, and both backends behave the same.
