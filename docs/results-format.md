# Result file formats (consumed by the dashboard)

Both files are written by the backend tooling and served read-only by the API. Either may
be missing until someone runs the corresponding command. The UI must show an empty state
when the endpoint returns 404.

## `eval/results/latest.json`: `GET /v1/eval/latest`

Written by `python -m medmemory eval --ablations`.

```jsonc
{
  "generated_at": "2026-10-07T10:00:00Z",
  "mode": "mock",                       // or "real"
  "components": { /* ComponentInfo, same shape as GET /v1/config */ },
  "gold_size": 84,
  "router": {
    "accuracy": 0.93, "macro_f1": 0.92,
    "per_class": { "KV": {"precision": 0.95, "recall": 0.97, "f1": 0.96, "support": 30}, "VECTOR": {}, "HYBRID": {} },
    "confusion": { "labels": ["KV", "VECTOR", "HYBRID"], "matrix": [[29, 0, 1], [0, 20, 2], [1, 1, 18]] }  // rows = expected, cols = predicted
  },
  "entities": { "precision": 0.9, "recall": 0.88, "micro_f1": 0.89,
                "by_type": { "drugs": {"precision": 1, "recall": 1, "f1": 1, "support": 20} } },
  "retrieval": { "n": 40, "recall_at_k": {"1": 0.6, "3": 0.8, "5": 0.9}, "mrr": 0.71 },
  "answers": { "answered": 50, "abstained": 12, "faithfulness": 1.0, "grounding_rate_mean": 0.98 },
  "abstention": { "n": 70, "accuracy": 0.95, "precision": 0.9, "recall": 1.0 },
  "safety": { "red_flag_n": 10, "red_flag_recall": 1.0, "red_flag_precision": 1.0,
              "scope_violation_recall": 1.0, "scope_leaks": 0 },
  "cache": { "replayed": 40, "hit_rate": 0.55, "exact_hits": 30, "semantic_hits": 4 },
  "latency": {
    "by_route": { "KV": {"n": 30, "p50": 1.2, "p95": 3.1, "p99": 5.0, "mean": 1.5} },
    "by_cache": { "miss": {}, "exact": {}, "semantic": {} },
    "retrieval_by_route": { "KV": {} }
  },
  "ablations": [
    { "name": "baseline", "router_accuracy": 0.93, "recall_at_5": 0.9, "mrr": 0.71,
      "faithfulness": 1.0, "abstention_accuracy": 0.95, "cache_hit_rate": 0.55, "p50_ms": 4.2, "p95_ms": 12.0 }
    // "router=rules", "router=lora", "no-reranker", "no-semantic-cache", "hybrid-sparse", ...
  ],
  "router_comparison": {                // present when a LoRA adapter exists
    "rules": { "accuracy": 0.9, "macro_f1": 0.89, "confusion": {} },
    "lora":  { "accuracy": 0.93, "macro_f1": 0.92, "confusion": {} },
    "auto":  { "accuracy": 0.94, "macro_f1": 0.93, "confusion": {} }
  },
  "per_query": [ { "id": "q001", "query": "", "expected_route": "KV", "route": "KV", "status": "answered",
                   "expected_status": "answered", "rr": 1.0, "total_ms": 1.4 } ]
}
```

## `bench/results/latest.json`: `GET /v1/bench/latest`

Written by `python -m medmemory bench`.

```jsonc
{
  "generated_at": "2026-10-07T10:00:00Z",
  "machine": { "os": "Windows-11", "cpu": "", "gpu": "NVIDIA GeForce RTX 4060 Laptop GPU", "python": "3.12.6" },
  "kv": [ { "backend": "sqlite", "op": "get", "n": 20000, "p50_us": 4.1, "p95_us": 9.0, "p99_us": 15.2, "ops_per_s": 210000 } ],
  "vector": [ { "store": "faiss-flat-cpu", "n": 100000, "dim": 384, "recall_at_10": 1.0,
                "p50_ms": 3.2, "p95_ms": 4.0, "batch_qps": 5400, "build_s": 0.4, "note": "" } ],
  "embedding": [ { "model": "BAAI/bge-small-en-v1.5", "device": "cuda", "batch": 256, "texts_per_s": 2100 } ],
  "singleflight": { "concurrent": 50, "without": {"backend_calls": 50, "wall_ms": 520}, "with": {"backend_calls": 1, "wall_ms": 52} },
  "ring": [ { "nodes_before": 3, "nodes_after": 4, "vnodes": 64, "moved_fraction": 0.26, "modulo_moved_fraction": 0.75, "ideal": 0.25 } ],
  "pinecone": { "status": "skipped: no PINECONE_API_KEY" }
}
```
