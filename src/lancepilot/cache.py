"""Single RAM cache tier with a strict byte capacity."""

from __future__ import annotations


class CapacityError(RuntimeError):
    """Raised when an insert would violate used_bytes <= capacity_bytes."""


class RamCache:
    """Byte-accounted residency set. Policies decide what to insert and evict;
    this class only enforces the capacity invariant."""

    def __init__(self, capacity_bytes: int) -> None:
        if capacity_bytes <= 0:
            raise ValueError("capacity_bytes must be positive")
        self.capacity_bytes = capacity_bytes
        self.used_bytes = 0
        self.evictions = 0
        self._sizes: dict[int, int] = {}

    def __contains__(self, object_id: object) -> bool:
        return object_id in self._sizes

    def __len__(self) -> int:
        return len(self._sizes)

    @property
    def free_bytes(self) -> int:
        return self.capacity_bytes - self.used_bytes

    def can_ever_fit(self, size_bytes: int) -> bool:
        return size_bytes <= self.capacity_bytes

    def size_of(self, object_id: int) -> int:
        return self._sizes[object_id]

    def insert(self, object_id: int, size_bytes: int) -> None:
        if object_id in self._sizes:
            raise KeyError(f"object {object_id} already resident")
        if self.used_bytes + size_bytes > self.capacity_bytes:
            raise CapacityError(
                f"insert of {size_bytes}B exceeds capacity "
                f"({self.used_bytes}/{self.capacity_bytes})"
            )
        self._sizes[object_id] = size_bytes
        self.used_bytes += size_bytes

    def evict(self, object_id: int) -> int:
        size = self._sizes.pop(object_id)
        self.used_bytes -= size
        self.evictions += 1
        return size
