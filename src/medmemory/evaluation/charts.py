"""Static charts for the report and viva slides (PNG). The live dashboard renders the same
data interactively; these exist so results survive outside the app."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK = "#1f2933"
MUTED = "#7b8794"
SERIES = ["#2563eb", "#0f766e", "#b45309", "#7c3aed", "#be123c", "#0891b2", "#4d7c0f", "#475569"]


def _style(ax: Any, title: str) -> None:
    ax.set_title(title, loc="left", fontsize=11, color=INK, fontweight="bold")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(MUTED)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=INK, labelsize=9)


def render(result: dict[str, Any], out: Path) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    files: list[Path] = []

    # 1. confusion matrix
    conf = result["router"]["confusion"]
    fig, ax = plt.subplots(figsize=(4.2, 3.6), dpi=160)
    m = conf["matrix"]
    ax.imshow(m, cmap="Blues")
    ax.set_xticks(range(len(conf["labels"])), conf["labels"])
    ax.set_yticks(range(len(conf["labels"])), conf["labels"])
    ax.set_xlabel("predicted", color=MUTED)
    ax.set_ylabel("expected", color=MUTED)
    vmax = max(max(row) for row in m) or 1
    for i, row in enumerate(m):
        for j, v in enumerate(row):
            ax.text(
                j,
                i,
                str(v),
                ha="center",
                va="center",
                color="white" if v > vmax / 2 else INK,
                fontsize=11,
            )
    _style(ax, f"Router confusion (acc {result['router']['accuracy']:.2f})")
    files.append(_save(fig, out / "router_confusion.png"))

    # 2. ablations
    rows = [a for a in result.get("ablations", []) if "error" not in a]
    if rows:
        metrics = [
            ("router_accuracy", "router acc"),
            ("recall_at_5", "recall@5"),
            ("mrr", "MRR"),
            ("abstention_accuracy", "abstention acc"),
        ]
        fig, ax = plt.subplots(figsize=(10, 3.8), dpi=160)
        width = 0.8 / len(rows)
        for i, row in enumerate(rows):
            xs = [k + i * width for k in range(len(metrics))]
            ax.bar(
                xs,
                [row[m] for m, _ in metrics],
                width=width,
                label=row["name"],
                color=SERIES[i % len(SERIES)],
            )
        ax.set_xticks(
            [k + 0.4 - width / 2 for k in range(len(metrics))], [lbl for _, lbl in metrics]
        )
        ax.set_ylim(0, 1.05)
        ax.legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5))
        _style(ax, "Ablations (gold set, n=113)")
        files.append(_save(fig, out / "ablations.png"))

    # 3. latency by route (miss) vs cache
    lat = result["latency"]
    labels, p50, p95 = [], [], []
    for k, v in {
        **{f"{r} (miss)": lat["by_route"][r] for r in lat["by_route"]},
        **{f"cache {c}": lat["by_cache"][c] for c in ("exact", "semantic") if c in lat["by_cache"]},
    }.items():
        labels.append(k)
        p50.append(v["p50"])
        p95.append(v["p95"])
    if labels:
        fig, ax = plt.subplots(figsize=(7, 3.4), dpi=160)
        y = range(len(labels))
        ax.barh(y, p95, color="#cbd5e1", label="p95")
        ax.barh(y, p50, color=SERIES[0], label="p50")
        ax.set_yticks(list(y), labels)
        ax.set_xlabel("ms (mock mode)", color=MUTED)
        ax.legend(frameon=False, fontsize=8)
        _style(ax, "End-to-end latency")
        files.append(_save(fig, out / "latency.png"))

    # 4. recall@k
    rk = result["retrieval"]["recall_at_k"]
    fig, ax = plt.subplots(figsize=(4.2, 3.2), dpi=160)
    ax.plot([int(k) for k in rk], list(rk.values()), marker="o", color=SERIES[1])
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("k", color=MUTED)
    _style(ax, f"Retrieval recall@k (MRR {result['retrieval']['mrr']:.2f})")
    files.append(_save(fig, out / "recall_at_k.png"))
    return files


def _save(fig: Any, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path
