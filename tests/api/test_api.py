from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from urllib.parse import quote

import pytest
from fastapi.testclient import TestClient

from medmemory.api.app import create_app
from medmemory.container import Container
from tests.conftest import make_container, mock_settings


@pytest.fixture(scope="module")
def client(tmp_path_factory: pytest.TempPathFactory) -> Iterator[TestClient]:
    tmp = tmp_path_factory.mktemp("api")
    settings = mock_settings(tmp, rate_limit_per_minute=10_000)
    c: Container = make_container(tmp)
    with TestClient(create_app(settings, container=c)) as cl:
        yield cl


def test_health_ready_config(client: TestClient) -> None:
    assert client.get("/healthz").json() == {"status": "ok"}
    ready = client.get("/readyz").json()
    assert ready["ready"] is True
    cfg = client.get("/v1/config").json()
    assert cfg["mode"] == "mock" and cfg["generator"] == "extractive-mock"


def test_trace_id_roundtrip(client: TestClient) -> None:
    r = client.post(
        "/v1/query",
        json={"query": "What does ICD-10-CM code I10 mean?"},
        headers={"X-Trace-Id": "abc123"},
    )
    assert r.headers["x-trace-id"] == "abc123" and r.json()["trace_id"] == "abc123"


def test_query_validation(client: TestClient) -> None:
    assert client.post("/v1/query", json={"query": ""}).status_code == 422
    assert (
        client.post("/v1/query", json={"query": "x", "patient_scope": "DROP TABLE"}).status_code
        == 422
    )
    assert client.post("/v1/query", json={"query": "x", "unknown_field": 1}).status_code == 422


def test_sse_stream(client: TestClient) -> None:
    body = {"query": "What does the metformin label say about kidney function?"}
    events: list[str] = []
    final = None
    with client.stream("POST", "/v1/query/stream", json=body) as resp:
        assert resp.headers["content-type"].startswith("text/event-stream")
        current = None
        for line in resp.iter_lines():
            if line.startswith("event: "):
                current = line[7:]
                events.append(current)
            elif line.startswith("data: ") and current == "final":
                final = json.loads(line[6:])
    assert events[0] == "meta" and events[-1] == "final" and "token" in events
    assert final is not None and final["status"] == "answered"


def test_patient_views_and_sources(client: TestClient) -> None:
    pts = client.get("/v1/patients", params={"q": "P0001"}).json()
    assert pts[0]["patient_id"] == "P0001"
    rec = client.get("/v1/patients/P0001").json()
    kinds = {r["kind"] for r in rec["records"]}
    assert {"demographics", "condition", "medication", "lab"} <= kinds and rec["notes"]
    assert client.get("/v1/patients/P9999").status_code == 404
    note_src = client.get(
        "/v1/sources/" + quote("vec:label:metformin#contraindications#0", safe="")
    ).json()
    assert note_src["origin"] == "vector" and note_src["metadata"]["section"] == "Contraindications"
    kv_src = client.get("/v1/sources/" + quote("kv:icd10:E11.9", safe="")).json()
    assert "Type 2 diabetes" in kv_src["text"]


def test_kv_browse_guards_internal_prefixes(client: TestClient) -> None:
    assert client.get("/v1/kv", params={"prefix": "icd10:E11"}).json()
    assert client.get("/v1/kv", params={"prefix": "meta:version"}).status_code == 400
    assert client.get("/v1/kv/icd10:I10").json()["data"]["code"] == "I10"


def test_search_requires_scope_for_notes(client: TestClient) -> None:
    assert (
        client.post("/v1/search", json={"query": "cough", "namespace": "patient_notes"}).status_code
        == 400
    )
    r = client.post(
        "/v1/search",
        json={"query": "cough", "namespace": "patient_notes", "patient_scope": "P0003"},
    ).json()
    assert r["chunks"] and all(c["chunk"]["patient_id"] == "P0003" for c in r["chunks"])


def test_write_endpoint_invalidates(client: TestClient) -> None:
    q = {"query": "What is the latest TSH for P0004?"}
    client.post("/v1/query", json=q)
    assert client.post("/v1/query", json=q).json()["cache_hit"] == "exact"
    w = client.post(
        "/v1/patients/P0004/labs",
        json={"lab": "tsh", "value": 2.5, "unit": "mIU/L", "date": "2026-10-06"},
    ).json()
    assert w["invalidated_cache_entries"] >= 1
    after = client.post("/v1/query", json=q).json()
    assert after["cache_hit"] == "miss" and "2.5" in after["answer"]


def test_metrics_audit_scenarios(client: TestClient) -> None:
    m = client.get("/v1/metrics").json()
    assert m["counters"]["requests"] >= 1 and "exact" in m["caches"]
    assert client.get("/v1/audit", params={"n": 5}).json()
    assert len(client.get("/v1/scenarios").json()) >= 8


def test_auth_stub_and_rate_limit(tmp_path: Path) -> None:
    settings = mock_settings(tmp_path, auth_required=True, rate_limit_per_minute=12)
    c = make_container(tmp_path)
    with TestClient(create_app(settings, container=c)) as cl:
        body = {"query": "What does ICD-10-CM code I10 mean?"}
        assert cl.post("/v1/query", json=body).status_code == 401
        assert (
            cl.post("/v1/query", json=body, headers={"Authorization": "Bearer nope"}).status_code
            == 401
        )
        ok = {"Authorization": "Bearer dev-clinician-token"}
        codes = [cl.post("/v1/query", json=body, headers=ok).status_code for _ in range(10)]
        assert codes[0] == 200 and 429 in codes
