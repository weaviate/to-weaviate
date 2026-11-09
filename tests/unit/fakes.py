import math
import random
import threading
import time
from collections import Counter
from collections.abc import Callable, Iterable, Iterator
from dataclasses import replace
from types import SimpleNamespace
from typing import Any, ClassVar, Literal, NoReturn

import typer
from typer.testing import CliRunner, Result

from to_weaviate.cli import build_app
from to_weaviate.plan import CollectionPlan
from to_weaviate.settings import API_KEY_MAP, ChromaSettings, WeaviateSettings
from to_weaviate.sources.base import (
    FieldStats,
    Page,
    Partition,
    Record,
    SourceCollection,
)
from to_weaviate.target.schema import PlanMismatch
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
        self,
        records: list[Record],
        page_size: int = 100,
        read_delay: float = 0.0,
        embedding: dict[str, Any] | None = None,
    ) -> None:
        self.records = list(records)
        self.page_size = page_size
        self.read_delay = read_delay
        self.embedding = embedding
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
                embedding=self.embedding,
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


class FakeRpcError(Exception):
    """Stands in for grpc.RpcError: `code()` returns an object with the status name."""

    def __init__(self, status: str) -> None:
        super().__init__(status)
        self.status = status

    def code(self) -> SimpleNamespace:
        return SimpleNamespace(name=self.status)


def raise_grpc_failure(status: str, error: Callable[[str], Exception]) -> NoReturn:
    """Raise `error` inside the except block of a FakeRpcError, as the client does.

    The FakeRpcError becomes the `__context__` of the raised error.
    """
    try:
        raise FakeRpcError(status)
    except FakeRpcError:
        raise error(f"fake gRPC failure {status}")  # noqa: B904


def _source_id(obj: TargetObject) -> str:
    return str(obj.properties.get("source_id", obj.uuid))


class FakeWriter:
    """Writer that keeps the objects in memory. Safe to call from several threads.

    `stored` holds the objects by (target name, tenant), then by uuid. The
    call numbered `fail_on_call` (1-based) stores its objects and then raises
    RuntimeError, like a write whose response was lost.
    """

    def __init__(
        self, fail_on_call: int | None = None, reject_ids: Iterable[str] = ()
    ) -> None:
        self.fail_on_call = fail_on_call
        self.reject_ids = set(reject_ids)
        self.stored: dict[tuple[str, str | None], dict[str, TargetObject]] = {}
        self.writes_per_uuid: Counter[str] = Counter()
        self.calls = 0
        self._lock = threading.Lock()

    @property
    def objects(self) -> dict[str, TargetObject]:
        """Every stored object by uuid, over all collections and tenants."""
        return {u: o for place in self.stored.values() for u, o in place.items()}

    def write(self, entry: CollectionPlan, objects: list[TargetObject]) -> list[Reject]:
        with self._lock:
            self.calls += 1
            place = self.stored.setdefault((entry.target.name, entry.target.tenant), {})
            rejects: list[Reject] = []
            for obj in objects:
                source_id = _source_id(obj)
                if source_id in self.reject_ids:
                    rejects.append(Reject(source_id, "rejected by fake"))
                    continue
                place[obj.uuid] = obj
                self.writes_per_uuid[obj.uuid] += 1
            if self.calls == self.fail_on_call:
                raise RuntimeError(f"fake write failure on call {self.calls}")
            return rejects


class FakeReader:
    """TargetReader over a FakeWriter's objects of one collection and tenant."""

    def __init__(self, writer: FakeWriter) -> None:
        self.writer = writer

    def _objects(self, entry: CollectionPlan) -> dict[str, TargetObject]:
        return self.writer.stored.get((entry.target.name, entry.target.tenant), {})

    def exists(self, entry: CollectionPlan) -> bool:
        """A collection or a tenant exists once a write named it."""
        name, tenant = entry.target.name, entry.target.tenant
        if tenant:
            return (name, tenant) in self.writer.stored
        return any(stored_name == name for stored_name, _ in self.writer.stored)

    def count(self, entry: CollectionPlan) -> int:
        return len(self._objects(entry))

    def fetch(self, entry: CollectionPlan, uuids: list[str]) -> dict[str, TargetObject]:
        objects = self._objects(entry)
        return {u: objects[u] for u in uuids if u in objects}

    def near_vector(
        self, entry: CollectionPlan, vector: list[float], limit: int
    ) -> list[str]:
        kept = next(v for v in entry.target.vectors if v.mode == "keep")
        scored = [
            (_distance(kept.distance, vector, obj.vectors[kept.name]), _source_id(obj))
            for obj in self._objects(entry).values()
        ]
        return [source_id for _, source_id in sorted(scored)[:limit]]

    def filter_equal(
        self, entry: CollectionPlan, prop: str, value: Any, limit: int
    ) -> list[str]:
        matches = [
            _source_id(obj)
            for obj in self._objects(entry).values()
            if obj.properties.get(prop) == value
        ]
        return matches[:limit]

    def source_ids(self, entry: CollectionPlan) -> Iterator[tuple[str, str]]:
        for obj in list(self._objects(entry).values()):
            yield obj.uuid, _source_id(obj)


class FakeSchema:
    """Schema over a dict of existing target collections.

    `existing` maps a target name to the differences its collection has from
    the plan. `ensure` raises `ensure_error` when it is set.
    """

    def __init__(
        self,
        modules: Iterable[str] = ("text2vec-openai", "text2vec-weaviate"),
        existing: dict[str, list[str]] | None = None,
        ensure_error: Exception | None = None,
    ) -> None:
        self.enabled = set(modules)
        self.existing = dict(existing or {})
        self.ensure_error = ensure_error
        self.ensured: list[str] = []

    def modules(self) -> set[str]:
        return set(self.enabled)

    def differences(self, group: list[CollectionPlan]) -> list[str] | None:
        return self.existing.get(group[0].target.name)

    def ensure(self, group: list[CollectionPlan]) -> Literal["created", "exists"]:
        if self.ensure_error is not None:
            raise self.ensure_error
        name = group[0].target.name
        found = self.existing.get(name)
        if found:
            raise PlanMismatch("; ".join(found))
        self.ensured.append(name)
        if found is None:
            self.existing[name] = []
            return "created"
        return "exists"


class FakeFactories:
    """The four `build_app` factories over fakes. Counts the factory calls.

    `source_error` is raised by the source factory, `target_error` by the
    writer, reader and schema factories, like a failed connection. The reader
    reads what the writer stored.
    """

    def __init__(
        self,
        source: FakeSource,
        writer: FakeWriter | None = None,
        schema: FakeSchema | None = None,
        source_error: Exception | None = None,
        target_error: Exception | None = None,
    ) -> None:
        self.source = source
        self.writer = writer or FakeWriter()
        self.schema = schema or FakeSchema()
        self.source_error = source_error
        self.target_error = target_error
        self.source_calls = 0
        self.schema_calls = 0
        self.consistency: list[str | None] = []

    def app(self) -> typer.Typer:
        return build_app(self._source, self._writer, self._reader, self._schema)

    def _source(self, settings: ChromaSettings) -> FakeSource:
        self.source_calls += 1
        if self.source_error is not None:
            raise self.source_error
        return self.source

    def _writer(
        self, settings: WeaviateSettings, consistency: str | None
    ) -> FakeWriter:
        if self.target_error is not None:
            raise self.target_error
        self.consistency.append(consistency)
        return self.writer

    def _reader(self, settings: WeaviateSettings) -> FakeReader:
        if self.target_error is not None:
            raise self.target_error
        return FakeReader(self.writer)

    def _schema(self, settings: WeaviateSettings) -> FakeSchema:
        self.schema_calls += 1
        if self.target_error is not None:
            raise self.target_error
        return self.schema


def cli_env(**overrides: str | None) -> dict[str, str | None]:
    """Runner environment with the keys the commands need; None unsets a variable."""
    env: dict[str, str | None] = {var: None for var in API_KEY_MAP}
    env.update(
        {
            "CHROMA_URL": None,
            "CHROMA_TENANT": None,
            "CHROMA_DATABASE": None,
            "CHROMA_TOKEN_HEADER": None,
            "CHROMA_API_KEY": "chroma-key",
            "WEAVIATE_URL": "http://localhost:8099",
            "WEAVIATE_GRPC_URL": None,
            "WEAVIATE_API_KEY": "weaviate-key",
            "TO_WEAVIATE_DEBUG": None,
        }
    )
    env.update(overrides)
    return env


def cli_records(n: int, year_text_at: int | None = None) -> list[Record]:
    """Records with an int `year`; the record at `year_text_at` has a text year."""
    return [
        Record(
            f"id-{i}",
            [float(i), 1.0],
            f"doc {i}",
            None,
            {"year": "unknown" if i == year_text_at else 2000 + i},
        )
        for i in range(n)
    ]


def invoke(fakes: FakeFactories, *args: str, **env: str | None) -> Result:
    """Run the app of `fakes` in `cli_env` with `env` applied over it."""
    return CliRunner().invoke(fakes.app(), list(args), env=cli_env(**env))
