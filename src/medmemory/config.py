"""Environment-based configuration.

Every setting can be overridden with an env var prefixed `MEDMEMORY_` (see .env.example).
`MEDMEMORY_MODE=mock` (the default) swaps Pinecone, the LLM and the embedding model for
deterministic local fakes, so the whole system runs offline with no keys and no GPU.
Individual components can still be overridden in either mode, e.g. mock mode plus the
real sentence-transformers embedder.
"""

from __future__ import annotations

from functools import cached_property
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from medmemory.errors import ConfigError

KVBackend = Literal["sqlite", "memory"]
VectorBackend = Literal["memory", "faiss"]
EmbedderKind = Literal["hashing", "sentence-transformers"]
RerankerKind = Literal["none", "lexical", "cross-encoder"]
GeneratorKind = Literal["extractive"]
RouterKind = Literal["rules"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="MEDMEMORY_", env_file=".env", extra="ignore", frozen=False
    )

    mode: Literal["mock", "real"] = "mock"
    data_dir: Path = Path("data")
    var_dir: Path = Field(
        Path("var"), description="Runtime state: KV files, local indexes, audit log."
    )

    # ---- KV engine
    kv_backend: KVBackend = "sqlite"
    kv_durability: Literal["full", "normal"] = "full"

    # ---- cache engine
    cache_capacity: int = 2048
    cache_ttl_s: float = 900.0

    # ---- vector engine
    vector_store: VectorBackend | None = None
    embedder: EmbedderKind | None = None
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384
    device: Literal["auto", "cpu", "cuda"] = "auto"
    reranker: RerankerKind | None = None
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    min_evidence_score: float | None = None

    # ---- generation
    generator: GeneratorKind | None = None
    mock_stream_delay_ms: int = Field(8, description="Pacing for mock token streaming in the API.")

    # ---- router
    router: RouterKind = "rules"

    # ---- API
    auth_required: bool = False
    api_tokens: str = Field(
        "dev-clinician-token:dr.demo:clinician",
        description="Comma-separated token:user:role triples for the auth stub.",
    )
    rate_limit_per_minute: int = 120
    cors_origins: list[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]
    log_level: str = "INFO"
    log_json: bool = True

    # ------------------------------------------------------------------ resolution

    @property
    def resolved_vector_store(self) -> VectorBackend:
        return self.vector_store or "memory"

    @property
    def resolved_embedder(self) -> EmbedderKind:
        return self.embedder or ("sentence-transformers" if self.mode == "real" else "hashing")

    @property
    def resolved_reranker(self) -> RerankerKind:
        return self.reranker or ("cross-encoder" if self.mode == "real" else "lexical")

    @property
    def resolved_generator(self) -> GeneratorKind:
        return self.generator or "extractive"

    @property
    def resolved_min_evidence_score(self) -> float:
        """Below this best-chunk similarity the evidence is 'insufficient'. Calibrated on the
        seed set per embedder (scores from a hashing embedder are not comparable to bge)."""
        if self.min_evidence_score is not None:
            return self.min_evidence_score
        return 0.55 if self.resolved_embedder == "sentence-transformers" else 0.12

    @cached_property
    def tokens(self) -> dict[str, tuple[str, str]]:
        out: dict[str, tuple[str, str]] = {}
        for triple in filter(None, (t.strip() for t in self.api_tokens.split(","))):
            parts = triple.split(":")
            if len(parts) != 3:
                raise ConfigError(f"bad MEDMEMORY_API_TOKENS entry: {triple!r}")
            out[parts[0]] = (parts[1], parts[2])
        return out

    def validate_for_runtime(self) -> None:
        """Fail fast with a readable message instead of a vendor stack trace at query time."""
        if self.resolved_vector_store == "pinecone" and self.pinecone_api_key is None:
            raise ConfigError(
                "vector_store=pinecone needs PINECONE_API_KEY. Use MEDMEMORY_MODE=mock "
                "or MEDMEMORY_VECTOR_STORE=faiss to run without it."
            )
        if self.resolved_generator == "anthropic" and self.anthropic_api_key is None:
            raise ConfigError(
                "generator=anthropic needs ANTHROPIC_API_KEY. Use MEDMEMORY_MODE=mock "
                "or MEDMEMORY_GENERATOR=extractive to run without it."
            )

    @property
    def seed_dir(self) -> Path:
        return self.data_dir / "seed"
