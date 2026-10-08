# ADR-0005: Semantic cache gated by exact scope, entity signature and data version

- Status: accepted
- Date: 2026-10-07

## Context
Embedding similarity cannot see the differences that matter clinically. "Is P0003 on
warfarin" and "is P0003 not on warfarin", or CKD stage 3 and CKD stage 4, can score above
0.95. Reusing an answer across patients would be a privacy breach.

## Decision
A semantic hit requires **all** of the following:
1. The same `patient_scope` and filter set.
2. An identical normalised entity signature: IDs, codes, drugs, labs, conditions, negated
   terms, time range and dosage intent.
3. The same `data_version`, which every KV write to that patient bumps.
4. Cosine similarity of at least τ (0.95 for bge, 0.90 for the hashing mock), tuned on the
   adversarial near-miss pairs in `tests/cache/test_semantic_cache.py`.

The semantic cache is never consulted for red-flag queries, dosage-intent queries,
abstentions or scope violations. Entries are tagged `patient:{id}` and invalidated on write.

## Consequences
The semantic hit rate will be low by design. That is the correct trade-off, and it gives
the viva a concrete example of a performance feature being constrained by safety.
