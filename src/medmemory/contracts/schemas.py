"""Shared Pydantic v2 schemas: the wire contract between modules, the API, and the UI.

Anything that crosses a module boundary or the HTTP boundary is defined here, so module
owners can work in parallel against a stable shape. Changing a field here is a contract
change: update the OpenAPI snapshot (`make openapi`) and tell the frontend owner.
"""

from __future__ import annotations

from datetime import date
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field

DISCLAIMER = "Educational prototype. Not medical advice."

# Seed patients are P0001..P0050; imported Synthea patients get P10001+.
PATIENT_ID_PATTERN = r"^P\d{4,6}$"

MetadataValue = str | int | float | bool | list[str]
# Pinecone-style filter expression, e.g. {"patient_id": {"$eq": "P0001"}}. Evaluated
# natively by Pinecone and by `medmemory.vector.filters.matches` for local stores.
MetadataFilter = dict[str, Any]


class _Strict(BaseModel):
    """Inbound models: reject unknown fields so client typos fail loudly."""

    model_config = ConfigDict(extra="forbid")


class _Out(BaseModel):
    """Outbound/internal models: tolerate extra fields so saved JSON (eval runs, cache
    snapshots) can be re-read after additive schema changes."""

    model_config = ConfigDict(extra="ignore")


class Route(StrEnum):
    KV = "KV"
    VECTOR = "VECTOR"
    HYBRID = "HYBRID"


class CacheResult(StrEnum):
    EXACT = "exact"
    SEMANTIC = "semantic"
    MISS = "miss"
    BYPASS = "bypass"  # cache disabled for this request, or not applicable (e.g. red flag)


class Namespace(StrEnum):
    PATIENT_NOTES = "patient_notes"
    DRUG_LABELS = "drug_labels"
    GUIDELINES = "guidelines"


class AnswerStatus(StrEnum):
    ANSWERED = "answered"
    ABSTAINED = "abstained"
    RED_FLAG = "red_flag"
    OUT_OF_SCOPE = "out_of_scope"
    SCOPE_VIOLATION = "scope_violation"


class Origin(StrEnum):
    KV = "kv"
    VECTOR = "vector"


# --------------------------------------------------------------------------- router


class TimeRange(_Out):
    text: str = Field(description="The phrase that produced this range, e.g. 'last 6 months'.")
    start: date | None = None
    end: date | None = None
    latest_only: bool = Field(False, description="True for 'latest' / 'most recent'.")


def _time_key(tr: TimeRange | None) -> str:
    """'latest' and 'most recent' are the same constraint; ranges compare by their dates."""
    if tr is None:
        return ""
    if tr.latest_only:
        return "latest"
    return f"{tr.start}..{tr.end}"


class Entities(_Out):
    patient_ids: list[str] = Field(default_factory=list)
    icd10_codes: list[str] = Field(default_factory=list)
    rxnorm_codes: list[str] = Field(default_factory=list)
    drugs: list[str] = Field(default_factory=list, description="Normalized generic names.")
    labs: list[str] = Field(default_factory=list, description="Normalized lab names.")
    conditions: list[str] = Field(default_factory=list, description="Normalized condition names.")
    time_range: TimeRange | None = None
    negated: list[str] = Field(
        default_factory=list, description="Entities mentioned under negation ('not on warfarin')."
    )
    dosage_intent: bool = Field(False, description="Query asks about doses/amounts.")
    record_types: list[str] = Field(
        default_factory=list,
        description="Structured record kinds asked for: medications, allergies, labs, conditions, encounters, demographics.",
    )

    def signature(self) -> str:
        """Canonical, order-independent string. Two queries may share a semantic-cache entry
        only if their signatures are identical."""
        parts = [
            "p=" + ",".join(sorted(self.patient_ids)),
            "icd=" + ",".join(sorted(self.icd10_codes)),
            "rx=" + ",".join(sorted(self.rxnorm_codes)),
            "d=" + ",".join(sorted(self.drugs)),
            "l=" + ",".join(sorted(self.labs)),
            "c=" + ",".join(sorted(self.conditions)),
            "neg=" + ",".join(sorted(self.negated)),
            "t=" + _time_key(self.time_range),
            "dose=" + str(int(self.dosage_intent)),
            "rt=" + ",".join(sorted(self.record_types)),
        ]
        return "|".join(parts)


class RouterDecision(_Out):
    route: Route
    confidence: float = Field(ge=0.0, le=1.0)
    entities: Entities
    normalized_query: str
    expansions: list[str] = Field(
        default_factory=list, description="Applied normalizations, e.g. 'HbA1c → hemoglobin a1c'."
    )
    router: str = Field(description="Which router decided: rules | lora | lora→rules | …")
    probabilities: dict[str, float] | None = None
    reasons: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- safety


class SafetyResult(_Out):
    verdict: Literal["ok", "red_flag", "out_of_scope"]
    category: str | None = None
    matched: list[str] = Field(default_factory=list)
    message: str | None = None


# --------------------------------------------------------------------------- retrieval


class KVRecord(_Out):
    key: str
    kind: str = Field(
        description="demographics | condition | medication | allergy | lab | encounter | icd10 | drug"
    )
    patient_id: str | None = None
    title: str
    data: dict[str, Any]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def source_id(self) -> str:
        return f"kv:{self.key}"


class Chunk(_Out):
    chunk_id: str
    doc_id: str
    namespace: Namespace
    title: str
    section: str
    text: str
    patient_id: str | None = None
    metadata: dict[str, MetadataValue] = Field(default_factory=dict)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def source_id(self) -> str:
        return f"vec:{self.chunk_id}"


class ScoredChunk(_Out):
    chunk: Chunk
    score: float = Field(description="Retriever similarity (cosine for dense, RRF for hybrid).")
    rerank_score: float | None = None
    rank: int
    retriever: str = "dense"


class Evidence(_Out):
    key: str = Field(description="Citation key shown to the model and user, e.g. 'S1'.")
    source_id: str
    origin: Origin
    title: str
    text: str
    score: float | None = None
    patient_id: str | None = None
    namespace: Namespace | None = None


class Citation(_Out):
    key: str
    source_id: str
    origin: Origin
    title: str
    snippet: str


class CitationCheck(_Out):
    passed: bool
    grounding_rate: float = Field(ge=0.0, le=1.0)
    total_sentences: int
    supported_sentences: int
    removed_sentences: list[str] = Field(default_factory=list)
    unsupported_numbers: list[str] = Field(default_factory=list)
    invalid_citation_keys: list[str] = Field(default_factory=list)


class EvidenceSufficiency(_Out):
    sufficient: bool
    reason: str
    max_score: float | None = None
    conflicts: list[str] = Field(default_factory=list)


class StageTiming(_Out):
    stage: str
    start_ms: float = Field(description="Offset from request start.")
    duration_ms: float
    status: Literal["ok", "skipped", "error"] = "ok"
    meta: dict[str, Any] = Field(default_factory=dict)


# --------------------------------------------------------------------------- query API


class QueryOptions(_Strict):
    use_cache: bool = True
    use_reranker: bool = True
    hybrid_sparse: bool = Field(False, description="Fuse dense + BM25 results with RRF.")
    router: Literal["rules", "auto"] = "rules"
    top_k: int = Field(5, ge=1, le=20)


class QueryRequest(_Strict):
    query: str = Field(min_length=1, max_length=2000)
    patient_scope: str | None = Field(
        None,
        pattern=PATIENT_ID_PATTERN,
        description="Active patient. Queries naming a different patient are refused.",
    )
    namespaces: list[Namespace] | None = Field(
        None, description="Restrict vector search; default depends on route and scope."
    )
    filters: dict[str, str | int | float | bool] | None = Field(
        None, description="Extra equality filters on chunk metadata (e.g. {'section': 'warnings'})."
    )
    options: QueryOptions = Field(default_factory=QueryOptions)


class QueryResponse(_Out):
    trace_id: str
    status: AnswerStatus
    answer: str
    route: Route | None = None
    confidence: float | None = None
    entities: Entities | None = None
    router: RouterDecision | None = None
    cache_hit: CacheResult
    latency_ms: dict[str, float] = Field(description="Per-stage duration in ms, keyed by stage.")
    total_ms: float
    timings: list[StageTiming]
    kv_records: list[KVRecord] = Field(default_factory=list)
    vector_chunks: list[ScoredChunk] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    citation_check: CitationCheck | None = None
    sufficiency: EvidenceSufficiency | None = None
    evidence_origin: list[Origin] = Field(
        default_factory=list, description="Stages the cited evidence came from."
    )
    safety: SafetyResult
    patient_scope: str | None = None
    generator: str = ""
    disclaimer: str = DISCLAIMER


class StreamEvent(_Out):
    """One server-sent event on /v1/query/stream. `final` carries a full QueryResponse."""

    event: Literal["meta", "retrieval", "token", "final", "error"]
    data: dict[str, Any]


# --------------------------------------------------------------------------- other API


class PatientSummary(_Out):
    patient_id: str
    name: str
    sex: str
    birth_date: date
    age: int
    conditions: list[str]
    medication_count: int
    lab_count: int
    note_count: int
    last_encounter: date | None = None


class PatientRecord(_Out):
    summary: PatientSummary
    records: list[KVRecord]
    notes: list[dict[str, Any]] = Field(description="Note metadata: doc_id, title, date.")


class LabWrite(_Strict):
    lab: str = Field(min_length=1, max_length=80)
    value: float
    unit: str = Field(max_length=20)
    date: date


class WriteResult(_Out):
    key: str
    invalidated_cache_entries: int
    data_version: int


class SearchRequest(_Strict):
    query: str = Field(min_length=1, max_length=2000)
    namespace: Namespace
    patient_scope: str | None = Field(None, pattern=PATIENT_ID_PATTERN)
    filters: dict[str, str | int | float | bool] | None = None
    top_k: int = Field(8, ge=1, le=50)
    rerank: bool = True
    hybrid_sparse: bool = False


class SearchResponse(_Out):
    chunks: list[ScoredChunk]
    timings: list[StageTiming]
    embedder: str
    store: str
    reranker: str | None = None


class SourceView(_Out):
    source_id: str
    origin: Origin
    title: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class Scenario(_Out):
    id: str
    title: str
    description: str
    query: str
    patient_scope: str | None = None
    expect: str
    run_twice: bool = False


class ComponentInfo(_Out):
    mode: Literal["mock", "real"]
    kv_backend: str
    vector_store: str
    embedder: str
    reranker: str
    generator: str
    router: str
    semantic_cache_threshold: float | None = None
    min_evidence_score: float


class ReadyStatus(_Out):
    ready: bool
    checks: dict[str, str]
