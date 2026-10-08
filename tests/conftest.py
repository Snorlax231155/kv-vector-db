"""Shared fixtures. Everything runs in mock mode: no keys, no network, no GPU."""

from __future__ import annotations

import asyncio
import os
from collections.abc import Iterator
from pathlib import Path

import pytest

os.environ.setdefault("MEDMEMORY_MODE", "mock")
os.environ.setdefault("MEDMEMORY_LOG_JSON", "false")

from medmemory.config import Settings
from medmemory.container import Container, build_container
from medmemory.ingest.build import ingest

ROOT = Path(__file__).resolve().parents[1]


def mock_settings(tmp: Path, **overrides: object) -> Settings:
    base: dict[str, object] = {
        "mode": "mock",
        "kv_backend": "memory",
        "var_dir": tmp,
        "data_dir": ROOT / "data",
        "router": "rules",
        "mock_stream_delay_ms": 0,
        "log_json": False,
        "log_level": "WARNING",
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


def make_container(tmp: Path, **overrides: object) -> Container:
    c = build_container(mock_settings(tmp, **overrides))
    asyncio.run(ingest(c))
    return c


@pytest.fixture(scope="session")
def shared_container(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Container]:
    """Read-mostly container shared across tests. Tests that write KV must build their own."""
    c = make_container(tmp_path_factory.mktemp("shared"))
    yield c
    c.close()


@pytest.fixture
def container(shared_container: Container) -> Container:
    shared_container.exact_cache.clear()
    return shared_container


@pytest.fixture
def fresh_container(tmp_path: Path) -> Iterator[Container]:
    c = make_container(tmp_path)
    yield c
    c.close()
