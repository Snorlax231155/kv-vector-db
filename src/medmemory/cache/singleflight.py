"""Single-flight: coalesce concurrent identical cache misses into one computation.

Without it, N simultaneous requests for an uncached query trigger N retrievals and N LLM
calls (a "thundering herd"). With it, the first caller computes and the others await the
same future. Errors propagate to every waiter and are not cached.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Generic, TypeVar

T = TypeVar("T")


class SingleFlight(Generic[T]):
    def __init__(self) -> None:
        self._inflight: dict[str, asyncio.Future[T]] = {}
        self.leaders = 0
        self.coalesced = 0

    def in_flight(self, key: str) -> bool:
        return key in self._inflight

    async def do(self, key: str, fn: Callable[[], Awaitable[T]]) -> tuple[T, bool]:
        """Run `fn` once per key at a time. Returns (result, shared) where shared=True means
        this caller piggy-backed on another caller's computation."""
        existing = self._inflight.get(key)
        if existing is not None:
            self.coalesced += 1
            return await asyncio.shield(existing), True
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[T] = loop.create_future()
        self._inflight[key] = fut
        self.leaders += 1
        try:
            result = await fn()
        except BaseException as exc:
            fut.set_exception(exc)
            fut.exception()  # mark retrieved so asyncio doesn't warn when nobody waited
            raise
        else:
            fut.set_result(result)
            return result, False
        finally:
            self._inflight.pop(key, None)

    def stats(self) -> dict[str, int]:
        return {
            "leaders": self.leaders,
            "coalesced": self.coalesced,
            "in_flight": len(self._inflight),
        }
