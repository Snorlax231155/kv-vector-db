# Module 4: Cache Engine (`medmemory.cache`)

**Subsystem:** Caching & Concurrency Control · **Interface:** `Cache` · **Tests:** `tests/cache/`

The caching layer accelerates clinical query evaluation safely while guaranteeing consistency across patient data updates.

| Component | File | Description |
|---|---|---|
| `LRUCache` | `lru.py` | O(1) Least-Recently-Used in-memory cache supporting TTL expiry, tag-based group invalidation, and detailed operational statistics (hits, misses, evictions, expirations, invalidations). |
| `SingleFlight` | `singleflight.py` | Concurrency coalescer ensuring multiple identical simultaneous requests share a single execution rather than redundantly triggering duplicate backend computations (stampede protection). |
| `exact_key()` | `__init__.py` | Deterministic composite hashing function generating cache keys based on normalized query, patient scope, filter parameters, namespaces, query options, and underlying data version. |

## Invalidation Architecture
1. **Version-based isolation:** Every KV write increments the patient data version (`meta:version:{patient}`). New queries automatically generate new cache keys, preventing stale reads by construction.
2. **Write-through tag invalidation:** Clinical update endpoints (`POST /v1/patients/{id}/labs`) trigger immediate tag eviction (`patient:{id}`) across all cached entries.
3. **TTL expiration:** Configurable duration safeguard (`MEDMEMORY_CACHE_TTL_S`, default 900s).
