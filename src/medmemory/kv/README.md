# Module 1: KV engine (`medmemory.kv`)

**Subsystem:** Core Key-Value Storage Engine · **Interface:** `KVStore` in `contracts/protocols.py` · **Tests:** `tests/kv/`

The KV engine is a byte-oriented, ordered key-value store with TTL, prefix scan, atomic batches, and snapshots. It provides two backends that pass the contract suite:

| Backend | File | Durability | When to use |
|---|---|---|---|
| `MemoryKV` | `memory.py` | None (JSON snapshot only) | Fast tests, benchmarks baseline |
| `SQLiteKV` | `sqlite.py` | WAL mode + `synchronous=FULL` (or NORMAL) | **Default** durable engine |

`records.py` provides the clinical domain layer on top:
- Key schemas: `patient:{id}:lab:{lab}:{date}`, `icd10:{code}`, `drug:rx:{rxcui}`
- JSON serialization/deserialization with schema validation
- Clinical sentence rendering for evidence grounding
- Patient data version tracking (`meta:version:{patient}`) for cache invalidation
- Prefix-bounded patient lookups (`patient:{scope}:`) preventing cross-patient leakage

## Guarantees (tested)
- **Ordered prefix scanning:** scans are ordered lexicographically; sibling prefixes (`hemoglobin` vs `hemoglobin_a1c`) never leak.
- **TTL lifecycle:** expired keys are invisible to `get`, `scan`, and `count`; explicit `purge_expired()` reclaims storage.
- **Atomic batch execution:** `batch()` guarantees all-or-nothing execution. Tested via hard process kill (`SIGKILL`) recovery tests (`test_hard_kill_preserves_acked_batches_atomically`).
- **Consistent snapshotting:** point-in-time snapshots created via SQLite backup API or JSON memory export.

## Performance
- Read lookups (`get`): ~1–4 µs.
- Durable fsynced writes: ~100–130 µs (amortized significantly via atomic batching).
