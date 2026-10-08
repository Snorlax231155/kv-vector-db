from __future__ import annotations

from pathlib import Path

from medmemory.evaluation.harness import check_gates
from medmemory.evaluation.metrics import (
    binary,
    classification_report,
    entity_prf,
    retrieval_scores,
    source_matches,
)

ROOT = Path(__file__).resolve().parents[2]


def test_classification_report_confusion_and_f1() -> None:
    rep = classification_report(["KV", "KV", "VECTOR", "HYBRID"], ["KV", "HYBRID", "VECTOR", None])
    assert rep["accuracy"] == 0.5
    assert rep["confusion"]["matrix"] == [[1, 0, 1], [0, 1, 0], [0, 0, 0]]
    assert rep["per_class"]["VECTOR"]["f1"] == 1.0 and rep["per_class"]["HYBRID"]["f1"] == 0.0


def test_entity_prf_micro() -> None:
    out = entity_prf(
        [{"drugs": ["metformin"], "labs": ["egfr"]}],
        [{"drugs": ["metformin", "aspirin"], "labs": []}],
    )
    assert out["precision"] == 0.5 and out["recall"] == 0.5
    assert out["by_type"]["drugs"]["recall"] == 1.0


def test_source_matching_rules() -> None:
    assert source_matches("kv:patient:P0001:lab:egfr:", "kv:patient:P0001:lab:egfr:2026-08-02")
    assert not source_matches("kv:patient:P0001:lab:egfr:", "kv:patient:P0010:lab:egfr:2026-08-02")
    assert source_matches("doc:label:metformin", "vec:label:metformin#contraindications#0")
    assert not source_matches("doc:label:metformin", "vec:label:metformin-er#x#0")


def test_retrieval_scores_rr_and_recall() -> None:
    ranked = ["kv:icd10:I10", "vec:label:lisinopril#warnings#0", "vec:topic:highbloodpressure#x#0"]
    s = retrieval_scores(["doc:label:lisinopril", "doc:topic:highbloodpressure"], ranked)
    assert s["rr"] == 0.5 and s["recall"]["1"] == 0.0 and s["recall"]["3"] == 1.0


def test_binary_metrics() -> None:
    m = binary([True, False, True, False], [True, True, False, False])
    assert m == {"n": 4, "accuracy": 0.5, "precision": 0.5, "recall": 0.5}


def test_gates() -> None:
    result = {
        "router": {"accuracy": 0.8},
        "safety": {"red_flag_recall": 1.0, "scope_leaks": 0, "scope_violation_recall": 1.0},
        "answers": {"faithfulness": 1.0},
        "retrieval": {"recall_at_k": {"5": 0.9}},
        "abstention": {"accuracy": 0.9},
    }
    assert check_gates(result, "router_accuracy=0.85,safety_recall=1.0,scope_leaks=0") == [
        "router_accuracy 0.8 < 0.85"
    ]
    result["safety"]["scope_leaks"] = 2
    assert any("scope_leaks" in f for f in check_gates(result, "scope_leaks=0"))
