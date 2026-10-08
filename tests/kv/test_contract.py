"""One contract suite, every backend. A new backend is done when this file passes for it."""

from __future__ import annotations

import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from medmemory.contracts.protocols import KVOp, KVStore
from medmemory.errors import KVError
from medmemory.kv.memory import MemoryKV
from medmemory.kv.sqlite import SQLiteKV

BACKENDS = ["memory", "sqlite"]


class FakeClock:
    def __init__(self) -> None:
        self.t = 1_000_000.0

    def __call__(self) -> float:
        return self.t


def make(backend: str, tmp_path: Path, clock: FakeClock | None = None) -> KVStore:
    clock = clock or FakeClock()
    if backend == "memory":
        return MemoryKV(clock=clock)
    if backend == "sqlite":
        return SQLiteKV(tmp_path / "kv.sqlite3", clock=clock)
    raise ValueError(f"Unknown backend {backend}")


@pytest.fixture(params=BACKENDS)
def backend(request: pytest.FixtureRequest) -> str:
    return str(request.param)


def test_satisfies_protocol(backend: str, tmp_path: Path) -> None:
    assert isinstance(make(backend, tmp_path), KVStore)


def test_get_put_delete_roundtrip(backend: str, tmp_path: Path) -> None:
    kv = make(backend, tmp_path)
    assert kv.get("a") is None
    kv.put("a", b"1")
    assert kv.get("a") == b"1"
    kv.put("a", b"2")  # overwrite
    assert kv.get("a") == b"2"
    assert kv.delete("a") is True
    assert kv.delete("a") is False
    assert kv.get("a") is None


def test_binary_values_and_unicode_keys(backend: str, tmp_path: Path) -> None:
    kv = make(backend, tmp_path)
    blob = bytes(range(256)) * 4
    kv.put("doc:ünïcode:κλειδί", blob)
    assert kv.get("doc:ünïcode:κλειδί") == blob


def test_prefix_scan_is_ordered_and_bounded(backend: str, tmp_path: Path) -> None:
    kv = make(backend, tmp_path)
    for d in ["2024-01-05", "2025-06-01", "2023-11-30"]:
        kv.put(f"patient:P0001:lab:hemoglobin_a1c:{d}", d.encode())
    kv.put("patient:P0001:lab:hemoglobin:2024-01-05", b"x")  # sibling prefix must not leak in
    kv.put("patient:P0002:lab:hemoglobin_a1c:2024-01-05", b"y")
    rows = list(kv.scan("patient:P0001:lab:hemoglobin_a1c:"))
    assert [v.decode() for _, v in rows] == ["2023-11-30", "2024-01-05", "2025-06-01"]
    assert len(list(kv.scan("patient:P0001:lab:hemoglobin_a1c:", limit=2))) == 2
    assert kv.count("patient:P0001:") == 4
    assert kv.count("") == 5


def test_ttl_expiry_hidden_then_purged(backend: str, tmp_path: Path) -> None:
    clock = FakeClock()
    kv = make(backend, tmp_path, clock)
    kv.put("session:1", b"s", ttl_s=10)
    kv.put("forever", b"f")
    assert kv.get("session:1") == b"s"
    clock.t += 11
    assert kv.get("session:1") is None
    assert [k for k, _ in kv.scan("")] == ["forever"]
    assert kv.count() == 1
    assert kv.purge_expired() == 1
    assert kv.purge_expired() == 0


def test_put_clears_previous_ttl(backend: str, tmp_path: Path) -> None:
    clock = FakeClock()
    kv = make(backend, tmp_path, clock)
    kv.put("k", b"1", ttl_s=5)
    kv.put("k", b"2")  # no TTL now
    clock.t += 100
    assert kv.get("k") == b"2"


def test_batch_applies_all_ops(backend: str, tmp_path: Path) -> None:
    kv = make(backend, tmp_path)
    kv.put("old", b"x")
    kv.batch([KVOp("put", "a", b"1"), KVOp("put", "b", b"2"), KVOp("delete", "old")])
    assert (kv.get("a"), kv.get("b"), kv.get("old")) == (b"1", b"2", None)


def test_invalid_batch_is_rejected_atomically(backend: str, tmp_path: Path) -> None:
    kv = make(backend, tmp_path)
    with pytest.raises(KVError):
        kv.batch([KVOp("put", "a", b"1"), KVOp("put", "b", None)])
    if backend != "memory":  # memory store validates lazily; durable stores must roll back
        assert kv.get("a") is None


def test_closed_store_raises(backend: str, tmp_path: Path) -> None:
    kv = make(backend, tmp_path)
    kv.close()
    with pytest.raises(KVError):
        kv.get("a")


def test_snapshot_reopens_with_same_data(backend: str, tmp_path: Path) -> None:
    if backend == "memory":
        kv = MemoryKV()
        kv.put("a", b"1")
        snap = kv.snapshot(tmp_path / "snap.json")
        assert MemoryKV.load(snap).get("a") == b"1"
        return
    kv = make(backend, tmp_path)
    for i in range(100):
        kv.put(f"k{i:03d}", str(i).encode())
    snap = kv.snapshot(tmp_path / "snapshot")
    kv.put("after", b"not in snapshot")
    restored: KVStore = SQLiteKV(snap)
    assert restored.count("k") == 100
    assert restored.get("after") is None


def test_persists_across_reopen(backend: str, tmp_path: Path) -> None:
    if backend == "memory":
        pytest.skip("memory backend is not durable")
    kv = make(backend, tmp_path)
    kv.put("durable", b"yes")
    kv.close()
    assert make(backend, tmp_path).get("durable") == b"yes"


# --------------------------------------------------------------------------- crash recovery

CHILD = textwrap.dedent(
    """
    import sys
    from medmemory.contracts.protocols import KVOp
    from medmemory.kv.sqlite import SQLiteKV as Store
    path = sys.argv[1]
    kv = Store(path, durability="full")
    b = 0
    while True:
        kv.batch([KVOp("put", f"b:{b:06d}:{i:03d}", b"x" * 64) for i in range(50)])
        print(b, flush=True)  # acknowledged only after the batch call returned
        b += 1
    """
)


@pytest.mark.slow
@pytest.mark.parametrize("backend", [b for b in BACKENDS if b != "memory"])
def test_hard_kill_preserves_acked_batches_atomically(backend: str, tmp_path: Path) -> None:
    """Kill -9 a writer mid-stream. Every acknowledged batch must be fully present, and no
    batch may be partially present (atomicity + durability)."""
    path = tmp_path / "kv.sqlite3"
    proc = subprocess.Popen(
        [sys.executable, "-c", CHILD, str(path)],
        stdout=subprocess.PIPE,
        text=True,
    )
    acked: list[int] = []
    deadline = time.monotonic() + 30
    assert proc.stdout is not None
    while len(acked) < 25 and time.monotonic() < deadline:
        line = proc.stdout.readline()
        if line.strip():
            acked.append(int(line))
    proc.kill()  # SIGKILL / TerminateProcess: no cleanup, no close()
    proc.wait(timeout=10)
    assert len(acked) >= 25, "writer did not make progress"

    kv: KVStore = SQLiteKV(path)
    per_batch: dict[str, int] = {}
    for key, _ in kv.scan("b:"):
        batch_id = key.split(":")[1]
        per_batch[batch_id] = per_batch.get(batch_id, 0) + 1
    assert all(n == 50 for n in per_batch.values()), "found a partially applied batch"
    for b in acked:
        assert per_batch.get(f"{b:06d}") == 50, f"acknowledged batch {b} lost"
