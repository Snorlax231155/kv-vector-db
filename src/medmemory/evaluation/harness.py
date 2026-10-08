"""Evaluation harness: run the gold set through the real pipeline, compute every metric,
run ablations, write JSON + charts.

    python -m medmemory eval                 # baseline only
    python -m medmemory eval --ablations     # + rules/LoRA, reranker, semantic cache, hybrid, embedder
    python -m medmemory eval --quick --fail-under router_accuracy=0.85,safety_recall=1.0,scope_leaks=0

Protocol per configuration:
  pass 1  caches cleared; every gold item in file order (paraphrases come after the item
          they paraphrase, so semantic-cache hits are measured honestly)
  pass 2  replay every answerable item: exact-cache hit rate and cached latency
Routing, entity and retrieval metrics come from pass 1 only.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import platform
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from medmemory.config import Settings
from medmemory.container import Container, build_container
from medmemory.contracts.schemas import (
    AnswerStatus,
    CacheResult,
    QueryOptions,
    QueryRequest,
    QueryResponse,
)
from medmemory.evaluation.metrics import (
    ROUTES,
    binary,
    classification_report,
    entity_prf,
    percentiles,
    retrieval_scores,
)
from medmemory.ingest.build import ingest

log = logging.getLogger(__name__)
RETRIEVAL_STAGES = ("kv", "embed", "vector_search", "rerank")


@dataclass
class GoldItem:
    id: str
    query: str
    patient_scope: str | None
    category: str
    expected_route: str | None
    expected_status: str
    expected_entities: dict[str, list[str]]
    expected_sources: list[str]
    paraphrase_of: str | None = None
    notes: str = ""


@dataclass
class Ablation:
    name: str
    options: dict[str, Any] = field(default_factory=dict)
    settings: dict[str, Any] = field(default_factory=dict)


def load_gold(path: Path) -> list[GoldItem]:
    items = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            d = json.loads(line)
            items.append(GoldItem(**{k: d.get(k) for k in GoldItem.__dataclass_fields__ if k in d}))
    return items


def _eval_settings(base: Settings, tag: str, **overrides: Any) -> Settings:
    data = base.model_dump()
    data.update(
        var_dir=base.var_dir / "eval" / tag,
        kv_backend="memory",
        mock_stream_delay_ms=0,
        log_level="WARNING",
    )
    data.update(overrides)
    return Settings(**data)


async def _run_items(
    c: Container, gold: list[GoldItem], opts: dict[str, Any]
) -> list[QueryResponse]:
    out = []
    for g in gold:
        req = QueryRequest(
            query=g.query, patient_scope=g.patient_scope, options=QueryOptions(**opts)
        )
        out.append(await c.orchestrator.answer(req, trace_id=f"eval-{g.id}"))
    return out


def _sources(r: QueryResponse) -> list[str]:
    return [e.source_id for e in r.evidence]


def _retrieval_ms(r: QueryResponse) -> float:
    return round(sum(v for k, v in r.latency_ms.items() if k in RETRIEVAL_STAGES), 3)


def score(
    gold: list[GoldItem], first: list[QueryResponse], replay: list[QueryResponse], c: Container
) -> dict[str, Any]:
    routed = [(g, r) for g, r in zip(gold, first, strict=True) if g.expected_route]
    router = classification_report(
        [g.expected_route or "" for g, _ in routed],
        [r.route.value if r.route else None for _, r in routed],
    )

    # entities: extracted directly so red-flag/scope items (which stop before routing) count too
    rules = c.routers["rules"]
    predicted_entities = []
    for g in gold:
        e = rules.route(g.query, g.patient_scope).entities
        predicted_entities.append(
            {
                k: list(getattr(e, k))
                for k in (
                    "patient_ids",
                    "icd10_codes",
                    "rxnorm_codes",
                    "drugs",
                    "labs",
                    "conditions",
                )
            }
        )
    entities = entity_prf([g.expected_entities for g in gold], predicted_entities)

    retrieval_rows = []
    per_query = []
    for g, r in zip(gold, first, strict=True):
        row: dict[str, Any] = {
            "id": g.id,
            "query": g.query,
            "category": g.category,
            "patient_scope": g.patient_scope,
            "expected_route": g.expected_route,
            "route": r.route.value if r.route else None,
            "expected_status": g.expected_status,
            "status": r.status.value,
            "cache": r.cache_hit.value,
            "total_ms": r.total_ms,
            "retrieval_ms": _retrieval_ms(r),
            "grounding": r.citation_check.grounding_rate if r.citation_check else None,
        }
        if g.expected_sources and g.expected_status == "answered":
            rs = retrieval_scores(g.expected_sources, _sources(r))
            retrieval_rows.append(rs)
            row["rr"] = round(rs["rr"], 4)
            row["recall_at_5"] = rs["recall"]["5"]
        per_query.append(row)
    n_ret = len(retrieval_rows)
    retrieval = {
        "n": n_ret,
        "recall_at_k": {
            k: round(sum(x["recall"][k] for x in retrieval_rows) / n_ret, 4) if n_ret else 0.0
            for k in ("1", "3", "5", "10")
        },
        "mrr": round(sum(x["rr"] for x in retrieval_rows) / n_ret, 4) if n_ret else 0.0,
    }

    answered = [r for r in first if r.status == AnswerStatus.ANSWERED and r.citation_check]
    answers = {
        "answered": sum(r.status == AnswerStatus.ANSWERED for r in first),
        "abstained": sum(r.status == AnswerStatus.ABSTAINED for r in first),
        "faithfulness": round(
            sum(r.citation_check.passed for r in answered if r.citation_check) / len(answered), 4
        )
        if answered
        else 0.0,
        "grounding_rate_mean": round(
            sum(r.citation_check.grounding_rate for r in answered if r.citation_check)
            / len(answered),
            4,
        )
        if answered
        else 0.0,
    }
    ab_pairs = [
        (g, r)
        for g, r in zip(gold, first, strict=True)
        if g.expected_status in ("answered", "abstained")
    ]
    abstention = binary(
        [g.expected_status == "abstained" for g, _ in ab_pairs],
        [r.status == AnswerStatus.ABSTAINED for _, r in ab_pairs],
    )

    red = binary(
        [g.expected_status == "red_flag" for g in gold],
        [r.status == AnswerStatus.RED_FLAG for r in first],
    )
    scope_items = [
        (g, r) for g, r in zip(gold, first, strict=True) if g.expected_status == "scope_violation"
    ]
    leaks = 0
    for r in first:
        scope = r.patient_scope
        leaks += sum(1 for e in r.evidence if e.patient_id and e.patient_id != scope)
    safety = {
        "red_flag_n": sum(g.expected_status == "red_flag" for g in gold),
        "red_flag_recall": red["recall"],
        "red_flag_precision": red["precision"],
        "scope_violation_recall": round(
            sum(r.status == AnswerStatus.SCOPE_VIOLATION for _, r in scope_items)
            / len(scope_items),
            4,
        )
        if scope_items
        else 1.0,
        "out_of_scope_recall": _recall(gold, first, "out_of_scope", AnswerStatus.OUT_OF_SCOPE),
        "scope_leaks": leaks,
    }
    status_acc = sum(
        g.expected_status == r.status.value for g, r in zip(gold, first, strict=True)
    ) / len(gold)

    para = [(g, r) for g, r in zip(gold, first, strict=True) if g.paraphrase_of]
    cache = {
        "replayed": len(replay),
        "hit_rate": round(
            sum(r.cache_hit in (CacheResult.EXACT, CacheResult.SEMANTIC) for r in replay)
            / len(replay),
            4,
        )
        if replay
        else 0.0,
        "exact_hits": sum(r.cache_hit == CacheResult.EXACT for r in replay),
        "semantic_hits": sum(r.cache_hit == CacheResult.SEMANTIC for r in first),
        "paraphrase_items": len(para),
        "paraphrase_semantic_hits": sum(r.cache_hit == CacheResult.SEMANTIC for _, r in para),
    }
    by_route: dict[str, list[float]] = {}
    ret_by_route: dict[str, list[float]] = {}
    for r in first:
        if r.cache_hit == CacheResult.MISS and r.route:
            by_route.setdefault(r.route.value, []).append(r.total_ms)
            ret_by_route.setdefault(r.route.value, []).append(_retrieval_ms(r))
    by_cache: dict[str, list[float]] = {}
    for r in [*first, *replay]:
        by_cache.setdefault(r.cache_hit.value, []).append(r.total_ms)
    latency = {
        "by_route": {k: percentiles(v) for k, v in by_route.items()},
        "retrieval_by_route": {k: percentiles(v) for k, v in ret_by_route.items()},
        "by_cache": {k: percentiles(v) for k, v in by_cache.items()},
    }
    return {
        "router": router,
        "entities": entities,
        "retrieval": retrieval,
        "answers": answers,
        "abstention": abstention,
        "safety": safety,
        "status_accuracy": round(status_acc, 4),
        "cache": cache,
        "latency": latency,
        "per_query": per_query,
    }


def _recall(
    gold: list[GoldItem], resp: list[QueryResponse], expected: str, status: AnswerStatus
) -> float:
    items = [(g, r) for g, r in zip(gold, resp, strict=True) if g.expected_status == expected]
    return round(sum(r.status == status for _, r in items) / len(items), 4) if items else 1.0


def gold_half(item_id: str) -> str:
    """Deterministic split for honest error analysis: failures are only ever inspected on the
    'analysis' half; the 'heldout' half is reported but never looked at item by item."""
    return (
        "analysis" if int(hashlib.sha256(item_id.encode()).hexdigest(), 16) % 2 == 0 else "heldout"
    )


def split_summary(per_query: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for half in ("analysis", "heldout"):
        rows = [r for r in per_query if gold_half(r["id"]) == half]
        routed = [r for r in rows if r["expected_route"]]
        red = [r for r in rows if r["expected_status"] == "red_flag"]
        ab = [r for r in rows if r["expected_status"] in ("answered", "abstained")]
        ret = [r for r in rows if "recall_at_5" in r]
        out[half] = {
            "n": len(rows),
            "router_accuracy": round(
                sum(r["route"] == r["expected_route"] for r in routed) / len(routed), 4
            )
            if routed
            else None,
            "status_accuracy": round(
                sum(r["status"] == r["expected_status"] for r in rows) / len(rows), 4
            )
            if rows
            else None,
            "red_flag_recall": round(sum(r["status"] == "red_flag" for r in red) / len(red), 4)
            if red
            else None,
            "abstention_accuracy": round(
                sum(
                    (r["status"] == "abstained") == (r["expected_status"] == "abstained")
                    for r in ab
                )
                / len(ab),
                4,
            )
            if ab
            else None,
            "recall_at_5": round(sum(r["recall_at_5"] for r in ret) / len(ret), 4) if ret else None,
        }
    return out


def router_comparison(c: Container, gold: list[GoldItem]) -> dict[str, Any]:
    routed = [g for g in gold if g.expected_route]
    out: dict[str, Any] = {}
    for name, router in c.routers.items():
        t0 = time.perf_counter()
        preds = [router.route(g.query, g.patient_scope).route.value for g in routed]
        ms = (time.perf_counter() - t0) * 1000 / max(1, len(routed))
        rep = classification_report([g.expected_route or "" for g in routed], preds)
        out[name] = {**rep, "router": router.name, "ms_per_query": round(ms, 3)}
    return out


async def evaluate(
    base: Settings, gold: list[GoldItem], ablation: Ablation
) -> tuple[dict[str, Any], Container]:
    c = build_container(_eval_settings(base, ablation.name.replace("=", "-"), **ablation.settings))
    await ingest(c)
    c.exact_cache.clear()
    opts = {
        "use_cache": True,
        "use_reranker": True,
        "hybrid_sparse": False,
        "router": "rules",
        **ablation.options,
    }
    first = await _run_items(c, gold, opts)
    answerable = [g for g in gold if g.expected_status in ("answered", "abstained")]
    replay = await _run_items(c, answerable, opts)
    return score(gold, first, replay, c), c


def _summary_row(name: str, s: dict[str, Any]) -> dict[str, Any]:
    miss = [r["total_ms"] for r in s["per_query"] if r["cache"] == "miss"]
    p = percentiles(miss)
    return {
        "name": name,
        "router_accuracy": s["router"]["accuracy"],
        "router_macro_f1": s["router"]["macro_f1"],
        "recall_at_5": s["retrieval"]["recall_at_k"]["5"],
        "mrr": s["retrieval"]["mrr"],
        "faithfulness": s["answers"]["faithfulness"],
        "abstention_accuracy": s["abstention"]["accuracy"],
        "status_accuracy": s["status_accuracy"],
        "cache_hit_rate": s["cache"]["hit_rate"],
        "semantic_hits": s["cache"]["semantic_hits"],
        "p50_ms": p["p50"],
        "p95_ms": p["p95"],
    }


def default_ablations(c: Container, real_embeddings: bool) -> list[Ablation]:
    abl = [
        Ablation("no-reranker", {"use_reranker": False}),
        Ablation("hybrid-sparse", {"hybrid_sparse": True}),
    ]
    if real_embeddings:
        abl.append(Ablation("embedder=bge-small", settings={"embedder": "sentence-transformers"}))
        abl.append(
            Ablation(
                "embedder=bge-small+cross-encoder",
                settings={"embedder": "sentence-transformers", "reranker": "cross-encoder"},
            )
        )
    return abl


async def run(args: argparse.Namespace) -> dict[str, Any]:
    base = Settings()
    gold = load_gold(Path(args.gold))
    t0 = time.perf_counter()
    baseline, c = await evaluate(base, gold, Ablation("baseline"))
    result: dict[str, Any] = {
        "generated_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "mode": base.mode,
        "components": c.info().model_dump(),
        "gold_size": len(gold),
        "gold_categories": _counts(g.category for g in gold),
        "machine": {"os": platform.platform(), "python": platform.python_version()},
        **baseline,
        "router_comparison": router_comparison(c, gold),
        "splits": split_summary(baseline["per_query"]),
        "ablations": [_summary_row("baseline", baseline)],
    }
    if args.ablations and not args.quick:
        for ab in default_ablations(c, getattr(args, "real_embeddings", False)):
            log.info("ablation %s", ab.name)
            try:
                s, _ = await evaluate(base, gold, ab)
                result["ablations"].append(_summary_row(ab.name, s))
            except Exception as exc:  # e.g. model not downloadable offline
                result["ablations"].append({"name": ab.name, "error": str(exc)[:200]})
    result["seconds"] = round(time.perf_counter() - t0, 1)
    return result


def _counts(items: Any) -> dict[str, int]:
    out: dict[str, int] = {}
    for i in items:
        out[i] = out.get(i, 0) + 1
    return dict(sorted(out.items()))


def check_gates(result: dict[str, Any], spec: str) -> list[str]:
    lookup = {
        "router_accuracy": lambda r: r["router"]["accuracy"],
        "safety_recall": lambda r: r["safety"]["red_flag_recall"],
        "scope_violation_recall": lambda r: r["safety"]["scope_violation_recall"],
        "faithfulness": lambda r: r["answers"]["faithfulness"],
        "recall_at_5": lambda r: r["retrieval"]["recall_at_k"]["5"],
        "abstention_accuracy": lambda r: r["abstention"]["accuracy"],
    }
    failures = []
    for part in filter(None, spec.split(",")):
        name, _, raw = part.partition("=")
        want = float(raw)
        if name == "scope_leaks":
            if result["safety"]["scope_leaks"] > want:
                failures.append(f"scope_leaks {result['safety']['scope_leaks']} > {want}")
            continue
        got = lookup[name](result)
        if got < want:
            failures.append(f"{name} {got} < {want}")
    return failures


def main(args: argparse.Namespace) -> int:
    logging.getLogger("medmemory").setLevel(logging.WARNING)
    result = asyncio.run(run(args))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    text = json.dumps(result, indent=2, default=str)
    (out / "latest.json").write_text(text, encoding="utf-8")
    if not args.quick:
        runs = Path("eval/runs")
        runs.mkdir(parents=True, exist_ok=True)
        (runs / f"{stamp}.json").write_text(text, encoding="utf-8")
        try:
            from medmemory.evaluation.charts import render

            render(result, out / "charts")
        except ImportError:
            log.warning("matplotlib not installed; skipping charts (pip install .[eval])")
    print(summary(result))
    failures = check_gates(result, args.fail_under) if args.fail_under else []
    for f in failures:
        print("GATE FAILED:", f)
    return 1 if failures else 0


def summary(r: dict[str, Any]) -> str:
    lines = [
        f"gold={r['gold_size']}  mode={r['mode']}  ({r.get('seconds', 0)}s)",
        f"router   acc={r['router']['accuracy']:.3f}  macroF1={r['router']['macro_f1']:.3f}",
        f"entities microF1={r['entities']['micro_f1']:.3f}",
        f"retrieval recall@5={r['retrieval']['recall_at_k']['5']:.3f}  MRR={r['retrieval']['mrr']:.3f}  (n={r['retrieval']['n']})",
        f"answers  faithfulness={r['answers']['faithfulness']:.3f}  answered={r['answers']['answered']}  abstained={r['answers']['abstained']}",
        f"abstain  acc={r['abstention']['accuracy']:.3f}  P={r['abstention']['precision']:.3f}  R={r['abstention']['recall']:.3f}",
        f"safety   red-flag R={r['safety']['red_flag_recall']:.3f} P={r['safety']['red_flag_precision']:.3f}  scope-violation R={r['safety']['scope_violation_recall']:.3f}  leaks={r['safety']['scope_leaks']}",
        f"status   acc={r['status_accuracy']:.3f}",
        f"cache    replay hit rate={r['cache']['hit_rate']:.3f}  paraphrase semantic hits={r['cache']['paraphrase_semantic_hits']}/{r['cache']['paraphrase_items']}",
    ]
    for route in ROUTES:
        p = r["latency"]["by_route"].get(route)
        if p:
            lines.append(
                f"latency  {route:6} p50={p['p50']:.2f}ms p95={p['p95']:.2f}ms (miss, n={p['n']})"
            )
    for half, m in r.get("splits", {}).items():
        lines.append(
            f"split    {half:8} n={m['n']:3} router={m['router_accuracy']} status={m['status_accuracy']} red-flag R={m['red_flag_recall']} abstain={m['abstention_accuracy']} R@5={m['recall_at_5']}"
        )
    if len(r["ablations"]) > 1:
        lines.append("ablations:")
        for a in r["ablations"]:
            if "error" in a:
                lines.append(f"  {a['name']:<34} ERROR {a['error']}")
            else:
                lines.append(
                    f"  {a['name']:<34} routerAcc={a['router_accuracy']:.3f} R@5={a['recall_at_5']:.3f} MRR={a['mrr']:.3f} faith={a['faithfulness']:.3f} abst={a['abstention_accuracy']:.3f} hit={a['cache_hit_rate']:.3f} p50={a['p50_ms']:.1f}ms"
                )
    return "\n".join(lines)
