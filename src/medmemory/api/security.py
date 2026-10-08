"""Auth stub and rate limiting.

AUTH IS A STUB. Tokens are static strings from config, mapped to a user and role. This
exists so the request lifecycle has a 'who' for the audit log and rate limiter, and so the
seam for real auth (OIDC/JWT) is in the right place. It is not security.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

from fastapi import Depends, HTTPException, Request, status

from medmemory.config import Settings


@dataclass(frozen=True)
class User:
    id: str
    role: str
    authenticated: bool


ANONYMOUS = User("anonymous", "clinician", False)


def current_user(request: Request) -> User:
    settings: Settings = request.app.state.settings
    header = request.headers.get("authorization", "")
    token = (
        header.removeprefix("Bearer ").strip()
        if header.startswith("Bearer ")
        else request.headers.get("x-api-key", "")
    )
    if token:
        found = settings.tokens.get(token)
        if found is None:
            raise HTTPException(
                status.HTTP_401_UNAUTHORIZED,
                "invalid token",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return User(found[0], found[1], True)
    if settings.auth_required:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED, "missing token", headers={"WWW-Authenticate": "Bearer"}
        )
    return ANONYMOUS


class TokenBucket:
    """Per-key token bucket: `rate` tokens per minute, burst = rate/4 (min 5)."""

    def __init__(self, per_minute: int) -> None:
        self.rate = per_minute / 60.0
        self.capacity = max(5.0, per_minute / 4)
        self._state: dict[str, tuple[float, float]] = {}
        self._lock = threading.Lock()

    def take(self, key: str, now: float | None = None) -> float:
        """Consume one token. Returns 0 if allowed, else seconds until a token is available."""
        now = time.monotonic() if now is None else now
        with self._lock:
            tokens, last = self._state.get(key, (self.capacity, now))
            tokens = min(self.capacity, tokens + (now - last) * self.rate)
            if tokens >= 1.0:
                self._state[key] = (tokens - 1.0, now)
                return 0.0
            self._state[key] = (tokens, now)
            return (1.0 - tokens) / self.rate if self.rate > 0 else 60.0


def rate_limited(request: Request, user: User = Depends(current_user)) -> User:
    bucket: TokenBucket = request.app.state.rate_limiter
    key = (
        user.id
        if user.authenticated
        else f"ip:{request.client.host if request.client else 'unknown'}"
    )
    wait = bucket.take(key)
    if wait > 0:
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS,
            "rate limit exceeded",
            headers={"Retry-After": f"{wait:.0f}"},
        )
    return user
