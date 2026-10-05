from lancepilot.models import CacheObject


def obj(oid: int, size: int, cost: float = 10.0, partition: int = 0) -> CacheObject:
    return CacheObject(object_id=oid, size_bytes=size, remote_fetch_ms=cost, partition_id=partition)
