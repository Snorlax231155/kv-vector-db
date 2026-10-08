"""Benchmarks: KV engines, vector stores (CPU/GPU, exact/ANN, Pinecone if keyed), embedding
throughput (CPU vs GPU), single-flight, and consistent-hash key movement.

    python -m medmemory bench            # full run, writes bench/results/latest.json + charts
    python -m medmemory bench --quick    # smaller sizes, for CI smoke

Honesty notes printed into the results:
* PyPI faiss-gpu wheels are Linux-only. On Windows the GPU vector baseline is exact search
  with a torch CUDA matmul, which is the same maths as faiss IndexFlatIP on GPU. On Linux
  with faiss-gpu installed, the faiss GPU index is benchmarked too.
* Vector benchmarks use synthetic clustered unit vectors (384-d), because the seed corpus
  (~2k chunks) is far too small to show index behaviour. Recall is measured against exact
  brute force on the same vectors.
* Pinecone latency includes the network round trip to AWS us-east-1, and is reported
  separately from server-side compute.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import platform
import random
import shutil
import statistics
import tempfile
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from medmemory.contracts.protocols import KVOp, KVStore

log = logging.getLogger(__name__)


def _pct_us(samples_ns: list[int]) -> dict[str, float]:
    a = np.asarray(samples_ns, dtype=float) / 1000.0
    return {
        "p50_us": round(float(np.percentile(a, 50)), 2),
        "p95_us": round(float(np.percentile(a, 95)), 2),
        "p99_us": round(float(np.percentile(a, 99)), 2),
    }


def _time_ops(fn: Callable[[int], object], n: int) -> tuple[list[int], float]:
    samples = []
    t0 = time.perf_counter()
    for i in range(n):
        s = time.perf_counter_ns()
        fn(i)
        samples.append(time.perf_counter_ns() - s)
    return samples, time.perf_counter() - t0


# --------------------------------------------------------------------------- KV


def bench_kv(n: int) -> list[dict[str, Any]]:
    from medmemory.kv.memory import MemoryKV
    from medmemory.kv.sqlite import SQLiteKV

    makers: list[tuple[str, Callable[[Path], KVStore]]] = [
        ("memory", lambda d: MemoryKV()),
        ("sqlite (sync=FULL)", lambda d: SQLiteKV(d / "a.sqlite3", "full")),
        ("sqlite (sync=NORMAL)", lambda d: SQLiteKV(d / "b.sqlite3", "normal")),
    ]
    try:
        from medmemory.kv.rocks import RocksKV

        makers += [
            ("rocksdb (sync WAL)", lambda d: RocksKV(d / "a.rocks", "full")),
            ("rocksdb (async WAL)", lambda d: RocksKV(d / "b.rocks", "normal")),
        ]
    except Exception:  # pragma: no cover
        pass
    rows = []
    value = json.dumps(
        {"lab": "hemoglobin a1c", "value": 7.4, "unit": "%", "date": "2026-01-01", "flag": "high"}
    ).encode()
    for name, make in makers:
        tmp = Path(tempfile.mkdtemp(prefix="mmkv"))
        kv = make(tmp)
        rows.extend(_kv_rows(name, kv, n, value))
        kv.close()
        shutil.rmtree(tmp, ignore_errors=True)
        log.info("kv %s done", name)
    return rows


def _kv_rows(name: str, kv: KVStore, n: int, value: bytes) -> list[dict[str, Any]]:
    keys = [f"patient:P{i % 500:04d}:lab:hemoglobin_a1c:{i:08d}" for i in range(n)]
    rng = random.Random(1)
    order = [rng.randrange(n) for _ in range(n * 2)]
    batch = [KVOp("put", f"batch:{j:05d}", value) for j in range(100)]

    def put(i: int) -> None:
        kv.put(keys[i], value)

    def get(i: int) -> None:
        kv.get(keys[order[i]])

    def scan(i: int) -> None:
        list(kv.scan(f"patient:P{i % 500:04d}:lab:"))

    def write_batch(i: int) -> None:
        kv.batch(batch)

    rows = []
    for op, fn, count, per in (
        ("put", put, n, 1),
        ("get", get, n * 2, 1),
        ("scan(prefix, ~n/500 keys)", scan, 1000, 1),
        ("batch(100 puts)", write_batch, 50, 100),
    ):
        samples, wall = _time_ops(fn, count)
        rows.append(
            {
                "backend": name,
                "op": op,
                "n": count,
                **_pct_us(samples),
                "ops_per_s": round(count * per / wall),
            }
        )
    return rows


# --------------------------------------------------------------------------- vectors


def synthetic_vectors(
    n: int, dim: int = 384, clusters: int = 200, seed: int = 7
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(clusters, dim)).astype(np.float32)
    assign = rng.integers(0, clusters, n)
    x = centers[assign] + 0.6 * rng.normal(size=(n, dim)).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    q_assign = rng.integers(0, clusters, 500)
    q = centers[q_assign] + 0.6 * rng.normal(size=(500, dim)).astype(np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    return x.astype(np.float32), q.astype(np.float32)


def _recall(found: np.ndarray, truth: np.ndarray, k: int) -> float:
    hits = sum(len(set(f[:k]) & set(t[:k])) for f, t in zip(found, truth, strict=True))
    return round(hits / (len(truth) * k), 4)


def bench_vectors(n: int, k: int = 10) -> list[dict[str, Any]]:
    import faiss

    x, q = synthetic_vectors(n)
    dim = x.shape[1]
    rows = []
    t0 = time.perf_counter()
    truth = np.argsort(-(q @ x.T), axis=1)[:, :k]
    exact_batch_s = time.perf_counter() - t0

    def single_latency(search: Callable[[np.ndarray], Any], m: int = 200) -> list[float]:
        out = []
        for i in range(m):
            s = time.perf_counter()
            search(q[i : i + 1])
            out.append((time.perf_counter() - s) * 1000)
        return out

    def row(
        store: str, build_s: float, recall: float, lat: list[float], batch_s: float, note: str = ""
    ) -> dict[str, Any]:
        return {
            "store": store,
            "n": n,
            "dim": dim,
            "recall_at_10": recall,
            "build_s": round(build_s, 3),
            "p50_ms": round(statistics.median(lat), 3),
            "p95_ms": round(float(np.percentile(lat, 95)), 3),
            "batch_qps": round(len(q) / batch_s),
            "note": note,
        }

    lat = single_latency(lambda v: np.argpartition(-(v @ x.T)[0], k)[:k])
    rows.append(row("numpy exact (CPU)", 0.0, 1.0, lat, exact_batch_s, "ground truth"))

    flat = faiss.IndexFlatIP(dim)
    t0 = time.perf_counter()
    flat.add(x)
    b = time.perf_counter() - t0
    t0 = time.perf_counter()
    _, idx = flat.search(q, k)
    rows.append(
        row(
            "faiss Flat (CPU, exact)",
            b,
            _recall(idx, truth, k),
            single_latency(lambda v: flat.search(v, k)),
            time.perf_counter() - t0,
        )
    )

    hnsw = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
    hnsw.hnsw.efConstruction = 80
    hnsw.hnsw.efSearch = 64
    t0 = time.perf_counter()
    hnsw.add(x)
    b = time.perf_counter() - t0
    t0 = time.perf_counter()
    _, idx = hnsw.search(q, k)
    rows.append(
        row(
            "faiss HNSW (CPU, ANN)",
            b,
            _recall(idx, truth, k),
            single_latency(lambda v: hnsw.search(v, k)),
            time.perf_counter() - t0,
            "M=32 efSearch=64",
        )
    )

    nlist = int(4 * np.sqrt(n))
    quant = faiss.IndexFlatIP(dim)
    ivf = faiss.IndexIVFFlat(quant, dim, nlist, faiss.METRIC_INNER_PRODUCT)
    t0 = time.perf_counter()
    ivf.train(x[: min(n, 50 * nlist)])
    ivf.add(x)
    b = time.perf_counter() - t0
    ivf.nprobe = 16
    t0 = time.perf_counter()
    _, idx = ivf.search(q, k)
    rows.append(
        row(
            "faiss IVF (CPU, ANN)",
            b,
            _recall(idx, truth, k),
            single_latency(lambda v: ivf.search(v, k)),
            time.perf_counter() - t0,
            f"nlist={nlist} nprobe=16",
        )
    )

    if (
        hasattr(faiss, "StandardGpuResources") and faiss.get_num_gpus() > 0
    ):  # Linux faiss-gpu builds
        res = faiss.StandardGpuResources()
        gflat = faiss.index_cpu_to_gpu(res, 0, faiss.IndexFlatIP(dim))
        t0 = time.perf_counter()
        gflat.add(x)
        b = time.perf_counter() - t0
        t0 = time.perf_counter()
        _, idx = gflat.search(q, k)
        rows.append(
            row(
                "faiss Flat (GPU, exact)",
                b,
                _recall(idx, truth, k),
                single_latency(lambda v: gflat.search(v, k)),
                time.perf_counter() - t0,
            )
        )
    try:
        import torch

        if torch.cuda.is_available():
            dev = torch.device("cuda")
            t0 = time.perf_counter()
            xt = torch.from_numpy(x).to(dev).half()
            torch.cuda.synchronize()
            b = time.perf_counter() - t0
            qt = torch.from_numpy(q).to(dev).half()

            def gsearch(v: np.ndarray) -> Any:
                vt = torch.from_numpy(v).to(dev).half()
                out = torch.topk(vt @ xt.T, k, dim=1).indices
                torch.cuda.synchronize()
                return out

            gsearch(q[:1])  # warm-up (CUDA context, kernels)
            t0 = time.perf_counter()
            idx = torch.topk(qt @ xt.T, k, dim=1).indices.cpu().numpy()
            torch.cuda.synchronize()
            bs = time.perf_counter() - t0
            rows.append(
                row(
                    f"torch exact (GPU fp16, {torch.cuda.get_device_name(0)})",
                    b,
                    _recall(idx, truth, k),
                    single_latency(gsearch),
                    bs,
                    "faiss-gpu has no Windows wheel; same maths as IndexFlatIP on GPU",
                )
            )
    except ImportError:  # pragma: no cover
        pass
    return rows


async def bench_pinecone(n: int) -> dict[str, Any]:
    key = os.environ.get("PINECONE_API_KEY")
    if not key:
        return {"status": "skipped: no PINECONE_API_KEY"}
    from medmemory.contracts.schemas import Chunk, Namespace
    from medmemory.vector.stores.pinecone_store import PineconeVectorStore

    store = PineconeVectorStore(
        key,
        os.environ.get("MEDMEMORY_PINECONE_INDEX", "medmemory"),
        384,
        namespace_prefix="bench__",
    )
    x, q = synthetic_vectors(n)
    chunks = [
        Chunk(
            chunk_id=f"b{i}",
            doc_id=f"b{i}",
            namespace=Namespace.GUIDELINES,
            title="bench",
            section="s",
            text="x",
        )
        for i in range(n)
    ]
    t0 = time.perf_counter()
    await store.upsert(Namespace.GUIDELINES, chunks, x)
    await store.wait_for_count(Namespace.GUIDELINES, n, 300)
    build = time.perf_counter() - t0
    lat = []
    for i in range(100):
        s = time.perf_counter()
        await store.query(Namespace.GUIDELINES, q[i], 10)
        lat.append((time.perf_counter() - s) * 1000)
    await store.delete(Namespace.GUIDELINES, delete_all=True)
    return {
        "status": "ok",
        "n": n,
        "upsert_and_index_s": round(build, 1),
        "p50_ms": round(statistics.median(lat), 1),
        "p95_ms": round(float(np.percentile(lat, 95)), 1),
        "note": "includes network RTT to us-east-1",
        "read_units": store.read_units,
    }


# --------------------------------------------------------------------------- embeddings


def bench_embeddings(quick: bool) -> list[dict[str, Any]]:
    from medmemory.ingest.build import iter_chunk_texts
    from medmemory.vector.embedders import HashingEmbedder

    texts = [t for _, t in iter_chunk_texts(Path("data/seed"))]
    if quick:
        texts = texts[:300]
    rows = []
    he = HashingEmbedder()
    t0 = time.perf_counter()
    he.embed_documents(texts)
    rows.append(
        {
            "model": "hashing-384 (mock)",
            "device": "cpu",
            "batch": len(texts),
            "texts_per_s": round(len(texts) / (time.perf_counter() - t0)),
        }
    )
    try:
        import torch

        from medmemory.vector.embedders import SentenceTransformerEmbedder

        devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
        for dev in devices:
            emb = SentenceTransformerEmbedder("BAAI/bge-small-en-v1.5", device=dev, batch_size=128)
            emb.embed_documents(texts[:64])  # warm-up
            t0 = time.perf_counter()
            emb.embed_documents(texts)
            dt = time.perf_counter() - t0
            name = torch.cuda.get_device_name(0) if dev == "cuda" else platform.processor() or "cpu"
            rows.append(
                {
                    "model": "BAAI/bge-small-en-v1.5",
                    "device": f"{dev} ({name})",
                    "batch": 128,
                    "texts_per_s": round(len(texts) / dt),
                    "corpus_seconds": round(dt, 2),
                }
            )
            del emb
    except Exception as exc:  # model not downloadable, no torch, ...
        rows.append({"model": "BAAI/bge-small-en-v1.5", "device": "n/a", "error": str(exc)[:160]})
    return rows


# --------------------------------------------------------------------------- single-flight + ring


async def bench_singleflight(concurrent: int = 50, backend_ms: float = 40.0) -> dict[str, Any]:
    from medmemory.cache import SingleFlight

    calls = 0

    async def backend() -> str:
        nonlocal calls
        calls += 1
        await asyncio.sleep(backend_ms / 1000)
        return "answer"

    t0 = time.perf_counter()
    await asyncio.gather(*(backend() for _ in range(concurrent)))
    without = {"backend_calls": calls, "wall_ms": round((time.perf_counter() - t0) * 1000, 1)}
    calls = 0
    sf: SingleFlight[str] = SingleFlight()
    t0 = time.perf_counter()
    await asyncio.gather(*(sf.do("same-key", backend) for _ in range(concurrent)))
    with_sf = {"backend_calls": calls, "wall_ms": round((time.perf_counter() - t0) * 1000, 1)}
    return {
        "concurrent": concurrent,
        "backend_ms": backend_ms,
        "without": without,
        "with": with_sf,
        "note": "Wall time is similar because the backend is async; what single-flight saves is backend work (LLM tokens, Pinecone read units).",
    }


def bench_ring() -> list[dict[str, Any]]:
    from medmemory.cluster.hashring import HashRing, modulo_movement, movement

    keys = [f"patient:P{i:05d}" for i in range(20000)] + [f"icd10:K{i:05d}" for i in range(5000)]
    rows = []
    for v in (1, 8, 32, 64, 128, 256):
        before = HashRing([f"node-{i}" for i in range(3)], v)
        after = HashRing([f"node-{i}" for i in range(4)], v)
        m = movement(keys, before, after)
        dist = after.distribution(keys)
        imbalance = max(dist.values()) / (len(keys) / len(dist))
        rows.append(
            {
                "nodes_before": 3,
                "nodes_after": 4,
                "vnodes": v,
                "moved_fraction": m["moved_fraction"],
                "modulo_moved_fraction": modulo_movement(keys, 3, 4),
                "ideal": 0.25,
                "max_load_over_mean": round(imbalance, 3),
            }
        )
    return rows


# --------------------------------------------------------------------------- main


def _machine() -> dict[str, Any]:
    info: dict[str, Any] = {
        "os": platform.platform(),
        "python": platform.python_version(),
        "cpu": platform.processor(),
        "cpu_count": os.cpu_count(),
    }
    try:
        import torch

        info["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        info["torch"] = torch.__version__
    except ImportError:
        info["gpu"] = None
    try:
        import faiss

        info["faiss"] = faiss.__version__
        info["faiss_gpus"] = faiss.get_num_gpus() if hasattr(faiss, "get_num_gpus") else 0
    except ImportError:
        pass
    return info


def main(args: argparse.Namespace) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    n_vec = 20_000 if args.quick else args.vectors
    result: dict[str, Any] = {
        "generated_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "machine": _machine(),
    }
    result["kv"] = bench_kv(2000 if args.quick else 10_000)
    result["vector"] = bench_vectors(n_vec)
    result["vector_seed_scale"] = bench_vectors(
        2_131
    )  # the real corpus size: GPU should NOT win here
    result["embedding"] = bench_embeddings(args.quick)
    result["singleflight"] = asyncio.run(bench_singleflight())
    result["ring"] = bench_ring()
    result["pinecone"] = asyncio.run(bench_pinecone(2_000 if args.quick else 10_000))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "latest.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    try:
        from medmemory.bench.charts import render

        render(result, out / "charts")
    except ImportError:
        pass
    print(json.dumps({k: v for k, v in result.items() if k not in ("machine",)}, indent=1)[:6000])
    return 0
