"""Domain model for the cache simulation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CacheObject:
    """A synthetic object stored in the simulated remote object store."""

    object_id: int
    size_bytes: int
    remote_fetch_ms: float
    partition_id: int


@dataclass(frozen=True, slots=True)
class QueryRequest:
    """A single query that needs exactly one object. Timestamp is logical (query index)."""

    query_id: int
    required_object_id: int
    timestamp: int
