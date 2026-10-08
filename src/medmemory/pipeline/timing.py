"""Per-stage timing. Every lifecycle stage runs inside `timer.stage(name, **meta)`, which
records start offset, duration, status and metadata for the UI's latency waterfall."""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from medmemory.contracts.schemas import StageTiming


class StageTimer:
    def __init__(self) -> None:
        self._t0 = time.perf_counter()
        self.stages: list[StageTiming] = []

    def now_ms(self) -> float:
        return (time.perf_counter() - self._t0) * 1000.0

    @contextmanager
    def stage(self, name: str, **meta: Any) -> Iterator[dict[str, Any]]:
        start = self.now_ms()
        status = "ok"
        try:
            yield meta  # callers may add metadata while the stage runs
        except BaseException:
            status = "error"
            raise
        finally:
            if meta.pop("skipped", False) is True:
                status = "skipped"
            self.stages.append(
                StageTiming(
                    stage=name,
                    start_ms=round(start, 3),
                    duration_ms=round(self.now_ms() - start, 3),
                    status=status,
                    meta=meta,
                )
            )

    def skip(self, name: str, reason: str) -> None:
        now = self.now_ms()
        self.stages.append(
            StageTiming(
                stage=name,
                start_ms=round(now, 3),
                duration_ms=0.0,
                status="skipped",
                meta={"reason": reason},
            )
        )

    def latency_map(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for s in self.stages:
            out[s.stage] = round(out.get(s.stage, 0.0) + s.duration_ms, 3)
        return out

    def total_ms(self) -> float:
        return round(self.now_ms(), 3)
