"""In-process metrics for the dashboard: latency samples per route / cache result / stage,
counters, and a recent-requests feed. Bounded ring buffers; no external dependencies.
(A production system would export these to Prometheus; the shape here maps 1:1.)"""

from __future__ import annotations

import threading
import time
from collections import Counter, deque
from typing import Any

import numpy as np

from medmemory.contracts.schemas import QueryResponse

BUCKETS_MS = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]


def _pct(values: list[float]) -> dict[str, float]:
    if not values:
        return {"n": 0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "mean": 0.0}
    a = np.asarray(values)
    return {
        "n": len(values),
        "p50": round(float(np.percentile(a, 50)), 3),
        "p95": round(float(np.percentile(a, 95)), 3),
        "p99": round(float(np.percentile(a, 99)), 3),
        "mean": round(float(a.mean()), 3),
    }


def _hist(values: list[float]) -> list[dict[str, Any]]:
    counts = [0] * (len(BUCKETS_MS) + 1)
    for v in values:
        i = next((i for i, b in enumerate(BUCKETS_MS) if v <= b), len(BUCKETS_MS))
        counts[i] += 1
    labels = [f"≤{b}" for b in BUCKETS_MS] + [f">{BUCKETS_MS[-1]}"]
    return [{"bucket": lbl, "count": c} for lbl, c in zip(labels, counts, strict=True)]


class Metrics:
    def __init__(self, maxlen: int = 5000) -> None:
        self._lock = threading.Lock()
        self.samples: deque[dict[str, Any]] = deque(maxlen=maxlen)
        self.counters: Counter[str] = Counter()
        self.started = time.time()

    def record(self, resp: QueryResponse) -> None:
        sample = {
            "t": time.time(),
            "trace_id": resp.trace_id,
            "route": resp.route.value if resp.route else "NONE",
            "status": resp.status.value,
            "cache": resp.cache_hit.value,
            "total_ms": resp.total_ms,
            "router": resp.router.router if resp.router else "",
            "stages": dict(resp.latency_ms),
            "grounding": resp.citation_check.grounding_rate if resp.citation_check else None,
        }
        with self._lock:
            self.samples.append(sample)
            self.counters["requests"] += 1
            self.counters[f"status:{sample['status']}"] += 1
            self.counters[f"route:{sample['route']}"] += 1
            self.counters[f"cache:{sample['cache']}"] += 1

    def incr(self, name: str, n: int = 1) -> None:
        with self._lock:
            self.counters[name] += n

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            samples = list(self.samples)
            counters = dict(self.counters)
        by_route: dict[str, list[float]] = {}
        by_cache: dict[str, list[float]] = {}
        by_stage: dict[str, list[float]] = {}
        for s in samples:
            by_route.setdefault(s["route"], []).append(s["total_ms"])
            by_cache.setdefault(s["cache"], []).append(s["total_ms"])
            for stage, ms in s["stages"].items():
                by_stage.setdefault(stage, []).append(ms)
        retrieval_only = {
            r: [
                sum(
                    v
                    for k, v in s["stages"].items()
                    if k in ("kv", "embed", "vector_search", "rerank")
                )
                for s in samples
                if s["route"] == r
            ]
            for r in by_route
        }
        cached = sum(1 for s in samples if s["cache"] in ("exact", "semantic"))
        eligible = sum(1 for s in samples if s["cache"] != "bypass")
        return {
            "uptime_s": round(time.time() - self.started, 1),
            "counters": counters,
            "latency_by_route": {k: _pct(v) for k, v in by_route.items()},
            "retrieval_latency_by_route": {k: _pct(v) for k, v in retrieval_only.items()},
            "latency_by_cache": {k: _pct(v) for k, v in by_cache.items()},
            "latency_by_stage": {k: _pct(v) for k, v in by_stage.items()},
            "histogram_by_cache": {k: _hist(v) for k, v in by_cache.items()},
            "histogram_by_route": {k: _hist(v) for k, v in by_route.items()},
            "cache_hit_rate": round(cached / eligible, 4) if eligible else 0.0,
            "recent": samples[-50:][::-1],
        }
