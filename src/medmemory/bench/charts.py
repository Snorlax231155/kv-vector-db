"""PNG charts for benchmark results (report/viva). The dashboard shows the same data live."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from medmemory.evaluation.charts import INK, MUTED, SERIES, _save, _style


def render(result: dict[str, Any], out: Path) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    files = []
    gets = [r for r in result["kv"] if r["op"] == "get"]
    if gets:
        fig, ax = plt.subplots(figsize=(6.5, 3), dpi=160)
        ax.barh([r["backend"] for r in gets], [r["p50_us"] for r in gets], color=SERIES[0])
        ax.set_xlabel("p50 get latency (µs)", color=MUTED)
        _style(ax, "KV get latency by backend")
        files.append(_save(fig, out / "kv_get.png"))
    vec = result["vector"]
    if vec:
        fig, ax = plt.subplots(figsize=(8, 3.4), dpi=160)
        labels = [
            r["store"].split(", NVIDIA")[0].replace("GPU fp16", "GPU fp16)").rstrip(")") + ")"
            if "NVIDIA" in r["store"]
            else r["store"]
            for r in vec
        ]
        ax.barh(labels, [r["p50_ms"] for r in vec], color=SERIES[1])
        for i, r in enumerate(vec):
            ax.text(
                r["p50_ms"],
                i,
                f"  recall@10 {r['recall_at_10']:.2f}",
                va="center",
                fontsize=8,
                color=INK,
            )
        ax.set_xlabel(f"p50 single-query latency (ms), n={vec[0]['n']:,}", color=MUTED)
        _style(ax, "Vector search p50: exact vs ANN, CPU vs GPU")
        files.append(_save(fig, out / "vector_latency.png"))
    emb = [r for r in result["embedding"] if "texts_per_s" in r]
    if emb:
        fig, ax = plt.subplots(figsize=(6.5, 2.8), dpi=160)
        ax.barh(
            [f"{r['model'].split('/')[-1]} · {r['device'].split(' (')[0]}" for r in emb],
            [r["texts_per_s"] for r in emb],
            color=SERIES[2],
        )
        ax.set_xscale("log")
        ax.set_xlabel("texts / second (log)", color=MUTED)
        _style(ax, "Embedding throughput")
        files.append(_save(fig, out / "embedding.png"))
    ring = result["ring"]
    fig, ax = plt.subplots(figsize=(5.5, 3), dpi=160)
    ax.plot(
        [r["vnodes"] for r in ring],
        [r["moved_fraction"] for r in ring],
        marker="o",
        color=SERIES[3],
        label="consistent hashing",
    )
    ax.axhline(ring[0]["modulo_moved_fraction"], color=SERIES[4], linestyle="--", label="hash % n")
    ax.axhline(0.25, color=MUTED, linestyle=":", label="ideal 1/(n+1)")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("virtual nodes per node", color=MUTED)
    ax.set_ylabel("keys moved (3→4 nodes)", color=MUTED)
    ax.legend(frameon=False, fontsize=8)
    _style(ax, "Key movement when adding a node")
    files.append(_save(fig, out / "ring_movement.png"))
    return files
