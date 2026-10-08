"""Pure metric functions (no I/O), unit-tested in tests/evaluation."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np

ROUTES = ["KV", "VECTOR", "HYBRID"]
ENTITY_TYPES = ["patient_ids", "icd10_codes", "rxnorm_codes", "drugs", "labs", "conditions"]


def classification_report(
    expected: Sequence[str], predicted: Sequence[str | None], labels: Sequence[str] = ROUTES
) -> dict[str, Any]:
    pairs = [(e, p or "NONE") for e, p in zip(expected, predicted, strict=True)]
    n = len(pairs)
    per_class: dict[str, dict[str, float]] = {}
    for label in labels:
        tp = sum(1 for e, p in pairs if e == label and p == label)
        fp = sum(1 for e, p in pairs if e != label and p == label)
        fn = sum(1 for e, p in pairs if e == label and p != label)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per_class[label] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "support": tp + fn,
        }
    matrix = [[sum(1 for e, p in pairs if e == r and p == c) for c in labels] for r in labels]
    acc = sum(1 for e, p in pairs if e == p) / n if n else 0.0
    macro = float(np.mean([v["f1"] for v in per_class.values()])) if per_class else 0.0
    return {
        "n": n,
        "accuracy": round(acc, 4),
        "macro_f1": round(macro, 4),
        "per_class": per_class,
        "confusion": {"labels": list(labels), "matrix": matrix},
    }


def entity_prf(
    expected: Iterable[dict[str, list[str]]], predicted: Iterable[dict[str, list[str]]]
) -> dict[str, Any]:
    tot = {"tp": 0, "fp": 0, "fn": 0}
    by_type = {t: {"tp": 0, "fp": 0, "fn": 0} for t in ENTITY_TYPES}
    for exp, pred in zip(expected, predicted, strict=True):
        for t in ENTITY_TYPES:
            e = {str(x).lower() for x in exp.get(t, [])}
            p = {str(x).lower() for x in pred.get(t, [])}
            tp, fp, fn = len(e & p), len(p - e), len(e - p)
            for d in (tot, by_type[t]):
                d["tp"] += tp
                d["fp"] += fp
                d["fn"] += fn

    def prf(d: dict[str, int]) -> dict[str, float]:
        prec = d["tp"] / (d["tp"] + d["fp"]) if d["tp"] + d["fp"] else 1.0
        rec = d["tp"] / (d["tp"] + d["fn"]) if d["tp"] + d["fn"] else 1.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        return {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "support": d["tp"] + d["fn"],
        }

    overall = prf(tot)
    return {
        **{k: overall[k] for k in ("precision", "recall")},
        "micro_f1": overall["f1"],
        "by_type": {t: prf(d) for t, d in by_type.items()},
    }


def source_matches(expected: str, source_id: str) -> bool:
    """`kv:` expectations match by key prefix; `doc:` expectations match the chunk's doc_id."""
    if expected.startswith("kv:"):
        return source_id.startswith(expected)
    if expected.startswith("doc:"):
        if not source_id.startswith("vec:"):
            return False
        return source_id[4:].split("#")[0] == expected[4:]
    return source_id == expected


def retrieval_scores(
    expected: Sequence[str], ranked_sources: Sequence[str], ks: Sequence[int] = (1, 3, 5, 10)
) -> dict[str, Any]:
    first = next(
        (
            i + 1
            for i, s in enumerate(ranked_sources)
            if any(source_matches(e, s) for e in expected)
        ),
        None,
    )
    recall = {}
    for k in ks:
        top = ranked_sources[:k]
        found = sum(1 for e in expected if any(source_matches(e, s) for s in top))
        recall[str(k)] = found / len(expected) if expected else 0.0
    return {"rr": 1.0 / first if first else 0.0, "recall": recall}


def percentiles(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {"n": 0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "mean": 0.0}
    a = np.asarray(values, dtype=float)
    return {
        "n": len(values),
        "p50": round(float(np.percentile(a, 50)), 3),
        "p95": round(float(np.percentile(a, 95)), 3),
        "p99": round(float(np.percentile(a, 99)), 3),
        "mean": round(float(a.mean()), 3),
    }


def binary(expected: Sequence[bool], predicted: Sequence[bool]) -> dict[str, float]:
    tp = sum(1 for e, p in zip(expected, predicted, strict=True) if e and p)
    fp = sum(1 for e, p in zip(expected, predicted, strict=True) if not e and p)
    fn = sum(1 for e, p in zip(expected, predicted, strict=True) if e and not p)
    tn = sum(1 for e, p in zip(expected, predicted, strict=True) if not e and not p)
    n = tp + fp + fn + tn
    return {
        "n": n,
        "accuracy": round((tp + tn) / n, 4) if n else 0.0,
        "precision": round(tp / (tp + fp), 4) if tp + fp else 1.0,
        "recall": round(tp / (tp + fn), 4) if tp + fn else 1.0,
    }
