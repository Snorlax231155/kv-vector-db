"""Project exception hierarchy. Adapters translate vendor exceptions into these."""

from __future__ import annotations


class MedMemoryError(Exception):
    """Base class for all MedMemory errors."""


class ConfigError(MedMemoryError):
    """Invalid or incomplete configuration (e.g. real mode without an API key)."""


class KVError(MedMemoryError):
    """KV engine failure (I/O, corruption, closed store)."""


class VectorStoreError(MedMemoryError):
    """Vector store failure (network, quota, schema mismatch)."""


class GenerationError(MedMemoryError):
    """LLM call failed or returned unusable output."""


class ScopeViolationError(MedMemoryError):
    """A request tried to read data outside its patient scope."""


class RateLimitedError(MedMemoryError):
    def __init__(self, retry_after_s: float) -> None:
        super().__init__(f"rate limited; retry after {retry_after_s:.1f}s")
        self.retry_after_s = retry_after_s
