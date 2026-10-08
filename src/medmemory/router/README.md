# Module 3: Memory Router (`medmemory.router`)

**Subsystem:** Hybrid Query Routing & NLP Analysis · **Interface:** `Router` · **Tests:** `tests/router/`

The memory router classifies clinical incoming queries into exact structured lookup (`KV`), unstructured semantic search (`VECTOR`), or combined multi-engine execution (`HYBRID`).

```
Query → Normalize (synonyms & abbreviations) → Extract Entities (regex + gazetteers) → RulesRouter
```

| Component | File | Responsibilities |
|---|---|---|
| Query Normalization | `normalize.py` | Clinical abbreviation expansion and synonym resolution (e.g. `HbA1c` → `hemoglobin a1c`, `HTN` → `hypertension`). Ambiguous abbreviations (`MI`, `AF`, `Cr`) expand strictly when uppercase to prevent false matches. |
| Entity Extraction | `entities.py` | Deterministic extraction of patient IDs (`P0001`), ICD-10-CM codes (validated against authoritative code lists), RxCUI codes, brand/generic drug names, lab tests, time ranges (e.g. "last 6 months"), and negation windows. |
| Decision Engine | `rules.py` | Transparent multi-criteria decision maker evaluating structured vs. narrative evidence needs. Returns route (`KV`, `VECTOR`, `HYBRID`), confidence score (0.0–1.0), and human-interpretable rationale for auditability. |

## Rationale
Entity extraction is kept strictly deterministic via compiled regexes and validated clinical gazetteers, preventing hallucinated entity IDs or improper patient scopes.
