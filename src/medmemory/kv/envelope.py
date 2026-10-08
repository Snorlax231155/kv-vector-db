"""Value envelope shared by backends that have no native TTL column (RocksDB, remote).

Layout: 1 byte version | 8 bytes big-endian float64 expires_at (0.0 = never) | payload
"""

from __future__ import annotations

import struct
import time

_HEADER = struct.Struct(">Bd")
VERSION = 1


def wrap(value: bytes, ttl_s: float | None, now: float | None = None) -> bytes:
    expires = 0.0 if ttl_s is None else (now if now is not None else time.time()) + ttl_s
    return _HEADER.pack(VERSION, expires) + value


def unwrap(raw: bytes, now: float | None = None) -> bytes | None:
    """Return the payload, or None if expired."""
    version, expires = _HEADER.unpack_from(raw)
    if version != VERSION:
        raise ValueError(f"unknown envelope version {version}")
    if expires and expires <= (now if now is not None else time.time()):
        return None
    return raw[_HEADER.size :]


def is_expired(raw: bytes, now: float) -> bool:
    _, expires = _HEADER.unpack_from(raw)
    return bool(expires) and expires <= now
