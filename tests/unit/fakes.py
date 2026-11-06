import math
import random
import threading
import time
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import replace
from typing import Any, ClassVar

from to_weaviate.plan import CollectionPlan
from to_weaviate.sources.base import (
    FieldStats,
    Page,
    Partition,
    Record,
    SourceCollection,
)
from to_weaviate.target.writer import Reject, TargetObject
from to_weaviate.types import SPARSE_VECTOR, value_type


def _distance(metric: str | None, a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    if metric == "dot":
        return -dot
    if metric == "l2-squared":
        return sum((x - y) ** 2 for x, y in zip(a, b, strict=True))
    norms = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return 1.0 - dot / norms if norms else 1.0


class FakeSource:
    """Source over an in-memory list with offset partitions, like ChromaSource.

    Cursors are decimal offsets into `records`. Reads are counted and timed so
    tests can check paging and concurrency.
    """

    name: ClassVar[str] = "fake"
    capabilities: ClassVar[frozenset[str]] = frozenset(
        {"vectors", "vector_search", "filter"}
    )

    REF = "fake-collection"
    NAMESPACE = "default"
    COLLECTION = "docs"

    def __init__(
        self, records: list[Record], page_size: int = 100, read_delay: float = 0.0
    ) -> None:
        self.records = list(records)
        self.page_size = page_size
        self.read_delay = read_delay
        self.read_calls = 0
        self.peak_concurrent_reads = 0
        self._active_reads = 0
        self._growth: list[Record] = []
        self._lock = threading.Lock()

    def grow_after_partitions(self, records: list[Record]) -> None:
        """Append `records` at the end of the next `partitions()` call."""
        self._growth.extend(records)

    def inspect(
        self,
        namespaces: list[str] | None,
        collections: list[str] | None,
        sample_pages: int | None,
    ) -> list[SourceCollection]:
        if namespaces is not None and self.NAMESPACE not in namespaces:
            return []
        if collections is not None and self.COLLECTION not in collections:
            return []
        types: dict[str, Counter[str]] = {}
        for record in self.records:
            for key, value in record.metadata.items():
                types.setdefault(key, Counter())[value_type(value)] += 1
        sparse = sorted(k for k, seen in types.items() if SPARSE_VECTOR in seen)
        vectors = [r.vector for r in self.records if r.vector is not None]
        documents = [r.document for r in self.records if r.document]
        return [
            SourceCollection(
                ref=self.REF,
                namespace=self.NAMESPACE,
                name=self.COLLECTION,
                count=len(self.records),
                dimension=len(vectors[0]) if vectors else None,
                distance="cosine",
                embedding=None,
                fields=[
                    FieldStats(key, dict(seen))
                    for key, seen in sorted(types.items())
                    if key not in sparse
                ],
                sparse_keys=sparse,
                document_bytes_estimate=sum(len(d.encode()) for d in documents),
            )
        ]

    def partitions(
        self, collection: SourceCollection, parallelism: int
    ) -> list[Partition]:
        self.records.extend(self._growth)
        self._growth = []
        count = collection.count
        if count == 0:
            return []
        parts = min(max(parallelism, 1), count)
        bounds = [i * count // parts for i in range(parts + 1)]
        return [
            Partition(f"{start}-{stop}", str(start), str(stop))
            for start, stop in zip(bounds, bounds[1:], strict=False)
        ]

    def read(
        self,
        collection: SourceCollection,
        partition: Partition,
        cursor: str | None,
        with_vectors: bool,
    ) -> Page:
        with self._lock:
            self.read_calls += 1
            self._active_reads += 1
            self.peak_concurrent_reads = max(
                self.peak_concurrent_reads, self._active_reads
            )
        try:
            if self.read_delay:
                time.sleep(self.read_delay)
            return self._page(partition, cursor, with_vectors)
        finally:
            with self._lock:
                self._active_reads -= 1

    def _page(
        self, partition: Partition, cursor: str | None, with_vectors: bool
    ) -> Page:
        offset = int(cursor if cursor is not None else partition.start or 0)
        limit = self.page_size
        stop = None if partition.stop is None else int(partition.stop)
        if stop is not None:
            limit = min(limit, stop - offset)
        if limit <= 0:
            return Page([], None)
        records = self.records[offset : offset + limit]
        if not with_vectors:
            records = [replace(r, vector=None) for r in records]
        next_offset = offset + len(records)
        done = len(records) < limit or (stop is not None and next_offset >= stop)
        return Page(records, None if done else str(next_offset))

    def ids(self, collection: SourceCollection) -> Iterator[str]:
        yield from [r.id for r in self.records]

    def count(self, collection: SourceCollection) -> int:
        return len(self.records)

    def sample(
        self,
        collection: SourceCollection,
        size: int,
        rng: random.Random,
        with_vectors: bool,
    ) -> list[Record]:
        picked = rng.sample(self.records, min(size, len(self.records)))
        if with_vectors:
            return picked
        return [replace(r, vector=None) for r in picked]

    def vector_search(
        self, collection: SourceCollection, vector: list[float], limit: int
    ) -> list[str]:
        scored = [
            (_distance(collection.distance, vector, r.vector), r.id)
            for r in self.records
            if r.vector is not None
        ]
        return [record_id for _, record_id in sorted(scored)[:limit]]

    def filter_equal(
        self, collection: SourceCollection, key: str, value: Any, limit: int
    ) -> list[str]:
        matches = [
            r.id for r in self.records if key in r.metadata and r.metadata[key] == value
        ]
        return matches[:limit]


class FakeWriter:
    """Writer that keeps the objects in memory. Safe to call from several threads.

    The call numbered `fail_on_call` (1-based) stores its objects and then
    raises RuntimeError, like a write whose response was lost.
    """

    def __init__(
        self, fail_on_call: int | None = None, reject_ids: Iterable[str] = ()
    ) -> None:
        self.fail_on_call = fail_on_call
        self.reject_ids = set(reject_ids)
        self.objects: dict[str, TargetObject] = {}
        self.writes_per_uuid: Counter[str] = Counter()
        self.calls = 0
        self._lock = threading.Lock()

    def write(self, entry: CollectionPlan, objects: list[TargetObject]) -> list[Reject]:
        with self._lock:
            self.calls += 1
            rejects: list[Reject] = []
            for obj in objects:
                source_id = str(obj.properties.get("source_id", obj.uuid))
                if source_id in self.reject_ids:
                    rejects.append(Reject(source_id, "rejected by fake"))
                    continue
                self.objects[obj.uuid] = obj
                self.writes_per_uuid[obj.uuid] += 1
            if self.calls == self.fail_on_call:
                raise RuntimeError(f"fake write failure on call {self.calls}")
            return rejects
