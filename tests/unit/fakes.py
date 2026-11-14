import math
import random
import sqlite3
import threading
import time
from collections import Counter
from collections.abc import Callable, Iterable, Iterator
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar, Literal, NoReturn

import typer
from typer.testing import CliRunner, Result
from weaviate.exceptions import WeaviateInsertManyAllFailedError, WeaviateRetryError

from to_weaviate.cli import build_app
from to_weaviate.plan import CollectionPlan
from to_weaviate.settings import API_KEY_MAP, ChromaSettings, WeaviateSettings
from to_weaviate.sources.base import (
    FieldStats,
    Page,
    Partition,
    Record,
    Sample,
    SourceCollection,
)
from to_weaviate.target.schema import Created, PlanMismatch, auto_factor
from to_weaviate.target.writer import Reject, TargetObject, WeaviateWriter
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
    tests can check paging and concurrency. `id_reads` counts the `ids_at`
    calls apart from `read_calls`. `calls` names each call of a method that
    reads the collection, in order.

    With `compaction` None, the source has no compaction, as single-node
    Chroma. Otherwise `wait_compacted` and `compaction_state` return its first
    item, removed unless it is the last. `wait_compacted` raises `wait_error`
    when it is set and reports each pair of `waits` first.
    """

    name: ClassVar[str] = "fake"
    capabilities: ClassVar[frozenset[str]] = frozenset(
        {"vectors", "vector_search", "filter"}
    )

    REF = "fake-collection"
    NAMESPACE = "default"
    COLLECTION = "docs"
    TENANT = "default_tenant"
    tenant = TENANT
    namespaces_note: str | None = None

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
        self.inspected_namespaces: list[list[str] | None] = []
        self.searched_namespaces: list[str] = []
        self.compaction: list[dict[str, int]] | None = None
        self.waits: list[tuple[int, int]] = []
        self.wait_error: Exception | None = None
        self.calls: list[str] = []
        self.read_calls = 0
        self.id_reads = 0
        self.peak_concurrent_reads = 0
        self._active_reads = 0
        self._growth: list[Record] = []
        self._lock = threading.Lock()

    def grow_after_partitions(self, records: list[Record]) -> None:
        """Append `records` at the end of the next `partitions()` call."""
        self._growth.extend(records)

    def namespaces(self) -> list[str]:
        return [self.NAMESPACE]

    def inspect(
        self,
        namespaces: list[str] | None,
        collections: list[str] | None,
        sample_pages: int | None,
    ) -> list[SourceCollection]:
        self.inspected_namespaces.append(namespaces)
        self.searched_namespaces = (
            [self.NAMESPACE] if namespaces is None else namespaces
        )
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
            self.calls.append("read")
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
        self.calls.append("ids")
        return iter([r.id for r in self.records])

    def get(
        self, collection: SourceCollection, ids: list[str], with_vectors: bool
    ) -> list[Record]:
        self.calls.append("get")
        wanted = set(ids)
        found = [r for r in self.records if r.id in wanted]
        return found if with_vectors else [replace(r, vector=None) for r in found]

    def ids_at(
        self, collection: SourceCollection, offset: int, limit: int
    ) -> list[str]:
        with self._lock:
            self.calls.append("ids_at")
            self.id_reads += 1
        return [r.id for r in self.records[offset : offset + limit]]

    def count(self, collection: SourceCollection) -> int:
        self.calls.append("count")
        return len(self.records)

    def wait_compacted(
        self, collection: SourceCollection, waiting: Callable[[int, int], None]
    ) -> dict[str, int] | None:
        self.calls.append("wait_compacted")
        if self.wait_error is not None:
            raise self.wait_error
        for unindexed, total in self.waits:
            waiting(unindexed, total)
        return None if self.compaction is None else self._compaction()

    def compaction_state(self, collection: SourceCollection) -> dict[str, int]:
        self.calls.append("compaction_state")
        return self._compaction()

    def _compaction(self) -> dict[str, int]:
        assert self.compaction is not None, "compaction_state needs compaction"
        if len(self.compaction) > 1:
            return self.compaction.pop(0)
        return self.compaction[0]

    def sample(
        self,
        collection: SourceCollection,
        size: int,
        rng: random.Random,
        with_vectors: bool,
    ) -> Sample:
        self.calls.append("sample")
        picked = rng.sample(self.records, min(size, len(self.records)))
        if not with_vectors:
            picked = [replace(r, vector=None) for r in picked]
        return Sample(picked, len(self.records))

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


def raise_grpc_failure(
    status: str, error: Callable[[str], Exception], batch: bool = False
) -> NoReturn:
    """Raise what client 4.24.1 raises when a gRPC call fails with `status`.

    For UNAVAILABLE the client retries, then raises WeaviateRetryError from the
    FakeRpcError. A batch call lets that error pass. A search or aggregate call
    raises `error` inside its except block, so the WeaviateRetryError becomes
    the `__context__`. For any other status, `error` is raised inside the
    except block of the FakeRpcError, which becomes the `__context__`.
    """
    try:
        try:
            raise FakeRpcError(status)
        except FakeRpcError as rpc:
            if status != "UNAVAILABLE":
                raise error(f"fake gRPC failure {status}")  # noqa: B904
            raise WeaviateRetryError(str(rpc), 3) from rpc
    except WeaviateRetryError as retry:
        if batch:
            raise
        raise error(str(retry))  # noqa: B904


def all_failed(reason: str) -> WeaviateInsertManyAllFailedError:
    """What client 4.24.1 raises when Weaviate refuses every object of a batch."""
    return WeaviateInsertManyAllFailedError(f"Here is the set of all errors: {reason}")


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


class FakeInsertClient:
    """The part of a WeaviateClient that WeaviateWriter uses. Thread-safe.

    Call `n` (1-based) of `insert_many` raises `fail(n)` when that returns an
    error. `refuse(objects)` is called next and may raise. An object for
    which `reject(object)` returns a message is not stored and is reported
    with it, as the server reports a refused object; when every object of a
    call is refused, the call raises what the client raises. The other
    objects are stored. `stored` has the shape of `FakeWriter.stored`, so a
    FakeReader can read it.
    """

    def __init__(
        self,
        fail: Callable[[int], Exception | None] = lambda n: None,
        reject: Callable[[Any], str | None] = lambda obj: None,
        refuse: Callable[[list[Any]], None] = lambda objects: None,
    ) -> None:
        self.fail = fail
        self.reject = reject
        self.refuse = refuse
        self.calls = 0
        self.stored: dict[tuple[str, str | None], dict[str, TargetObject]] = {}
        self.writes_per_uuid: Counter[str] = Counter()
        self.collections = SimpleNamespace(
            get=lambda name: _InsertCollection(self, name)
        )
        self._lock = threading.Lock()

    def insert_many(
        self, place: tuple[str, str | None], objects: list[Any]
    ) -> SimpleNamespace:
        with self._lock:
            self.calls += 1
            stored = self.stored.setdefault(place, {})
            error = self.fail(self.calls)
            if error is not None:
                raise error
            self.refuse(objects)
            refused = {i: m for i, o in enumerate(objects) if (m := self.reject(o))}
            if objects and len(refused) == len(objects):
                raise all_failed("\n".join(set(refused.values())))
            for index, obj in enumerate(objects):
                if index in refused:
                    continue
                uuid = str(obj.uuid)
                vectors = dict(obj.vector or {})
                stored[uuid] = TargetObject(uuid, dict(obj.properties), vectors)
                self.writes_per_uuid[uuid] += 1
        errors = {i: SimpleNamespace(message=m) for i, m in refused.items()}
        return SimpleNamespace(errors=errors)


class _InsertCollection:
    def __init__(
        self, client: FakeInsertClient, name: str, tenant: str | None = None
    ) -> None:
        self._client = client
        self._name = name
        self._tenant = tenant
        self.data = SimpleNamespace(insert_many=self._insert_many)

    def _insert_many(self, objects: list[Any]) -> SimpleNamespace:
        return self._client.insert_many((self._name, self._tenant), objects)

    def with_tenant(self, tenant: str) -> "_InsertCollection":
        return _InsertCollection(self._client, self._name, tenant)

    def with_consistency_level(self, level: Any) -> "_InsertCollection":
        return self


class FakeReader:
    """TargetReader over the stored objects of a FakeWriter or FakeInsertClient."""

    def __init__(self, writer: FakeWriter | FakeInsertClient) -> None:
        self.writer = writer

    def _objects(self, entry: CollectionPlan) -> dict[str, TargetObject]:
        return self.writer.stored.get((entry.target.name, entry.target.tenant), {})

    def active(self, entry: CollectionPlan) -> bool:
        """False for a (name, tenant) in the store's `inactive` set."""
        inactive = getattr(self.writer, "inactive", set())
        return (entry.target.name, entry.target.tenant) not in inactive

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

    def source_ids(self, entry: CollectionPlan) -> Iterator[tuple[str, str | None]]:
        for obj in list(self._objects(entry).values()):
            value = obj.properties.get("source_id")
            yield obj.uuid, None if value is None else str(value)


class FakeDeleter:
    """Deleter over the stored objects of a FakeWriter or FakeInsertClient.

    `deleted` holds (target name, tenant, uuid) in deletion order. `error` is
    raised by every call.
    """

    def __init__(
        self, store: FakeWriter | FakeInsertClient, error: Exception | None = None
    ) -> None:
        self.store = store
        self.error = error
        self.deleted: list[tuple[str, str | None, str]] = []
        self.calls = 0

    def delete(self, entry: CollectionPlan, uuids: list[str]) -> int:
        self.calls += 1
        if self.error is not None:
            raise self.error
        place = self.store.stored.get((entry.target.name, entry.target.tenant), {})
        removed = 0
        for uuid in uuids:
            if place.pop(uuid, None) is not None:
                self.deleted.append((entry.target.name, entry.target.tenant, uuid))
                removed += 1
        return removed


class FakeSchema:
    """Schema over a dict of existing target collections.

    `existing` maps a target name to the differences its collection has from
    the plan. `ensure` raises `ensure_error` when it is set. `properties`
    holds the property names of each collection that `ensure` created;
    `ensure` adds the planned properties that such a collection lacks and
    records them in `added`. `enable_trained_quantizers` records (target
    name, tenant) and raises `update_error` when it is set. `version` is
    the version text that the server reports.

    `node_count` is what `nodes` returns; None is a key that may not read
    the nodes. `factors` holds the replication factor of each collection; a
    collection without one has `server_factor`, which is also what a
    collection gets when `ensure` creates it without a node count. With
    `refuse_factor`, the server refuses the factor of an `auto` plan with
    that message.
    """

    def __init__(
        self,
        modules: Iterable[str] = ("text2vec-openai", "text2vec-weaviate"),
        existing: dict[str, list[str]] | None = None,
        ensure_error: Exception | None = None,
        update_error: Exception | None = None,
        version: str | None = "1.40.4",
        node_count: int | None = 1,
        factors: dict[str, int] | None = None,
        server_factor: int = 1,
        refuse_factor: str | None = None,
    ) -> None:
        self.refuse_factor = refuse_factor
        self.node_count = node_count
        self.factors = dict(factors or {})
        self.server_factor = server_factor
        self.enabled = set(modules)
        self.server_version = version
        self.existing = dict(existing or {})
        self.ensure_error = ensure_error
        self.update_error = update_error
        self.ensured: list[str] = []
        self.updated: list[tuple[str, str | None]] = []
        self.properties: dict[str, set[str]] = {}
        self.added: list[tuple[str, str]] = []

    def modules(self) -> set[str]:
        return set(self.enabled)

    def version(self) -> str | None:
        return self.server_version

    def differences(self, group: list[CollectionPlan]) -> list[str] | None:
        return self.existing.get(group[0].target.name)

    def missing_properties(self, group: list[CollectionPlan]) -> list[str]:
        stored = self.properties.get(group[0].target.name)
        if stored is None:
            return []
        planned = dict.fromkeys(p.name for e in group for p in e.target.properties)
        return [name for name in planned if name not in stored]

    def nodes(self) -> int | None:
        return self.node_count

    def replication_factor(self, group: list[CollectionPlan]) -> int | None:
        name = group[0].target.name
        if name not in self.existing:
            return None
        return self.factors.get(name, self.server_factor)

    def _create(self, group: list[CollectionPlan]) -> Created:
        name = group[0].target.name
        planned = group[0].target.replication_factor
        if planned != "auto":
            return Created(name, planned, "plan")
        if self.node_count is None:
            return Created(name, self.server_factor, "server")
        if self.refuse_factor is not None:
            return Created(
                name, self.server_factor, "server", self.node_count, self.refuse_factor
            )
        return Created(name, auto_factor(self.node_count), "nodes", self.node_count)

    def ensure(
        self,
        group: list[CollectionPlan],
        created: Callable[[Created], None] | None = None,
    ) -> Literal["created", "exists"]:
        if self.ensure_error is not None:
            raise self.ensure_error
        name = group[0].target.name
        found = self.existing.get(name)
        if found:
            raise PlanMismatch("; ".join(found))
        self.ensured.append(name)
        missing = self.missing_properties(group)
        self.added.extend((name, prop) for prop in missing)
        self.properties.setdefault(name, set()).update(
            p.name for e in group for p in e.target.properties
        )
        if found is None:
            self.existing[name] = []
            made = self._create(group)
            self.factors[name] = made.factor
            if created is not None:
                created(made)
            return "created"
        return "exists"

    def enable_trained_quantizers(self, entry: CollectionPlan) -> None:
        if self.update_error is not None:
            raise self.update_error
        self.updated.append((entry.target.name, entry.target.tenant))


class FakeFactories:
    """The five `build_app` factories over fakes. Counts the factory calls.

    `source_error` is raised by the source factory, `target_error` by the
    writer, reader, schema and deleter factories, like a failed connection.
    The reader and the deleter work on what the writer stored. A
    WeaviateWriter over a FakeInsertClient needs that client as `store`.
    """

    def __init__(
        self,
        source: FakeSource,
        writer: FakeWriter | WeaviateWriter | None = None,
        schema: FakeSchema | None = None,
        source_error: Exception | None = None,
        target_error: Exception | None = None,
        store: FakeInsertClient | None = None,
    ) -> None:
        self.source = source
        self.writer = writer or FakeWriter()
        self.store = store
        self.schema = schema or FakeSchema()
        self.source_error = source_error
        self.target_error = target_error
        self.source_calls = 0
        self.schema_calls = 0
        self.consistency: list[str | None] = []
        self.delete_consistency: list[str | None] = []
        self.deleter = FakeDeleter(self.store or self.writer)  # type: ignore[arg-type]

    def app(self) -> typer.Typer:
        return build_app(
            self._source, self._writer, self._reader, self._schema, self._deleter
        )

    def _source(self, settings: ChromaSettings) -> FakeSource:
        self.source_calls += 1
        if self.source_error is not None:
            raise self.source_error
        return self.source

    def _writer(
        self, settings: WeaviateSettings, consistency: str | None
    ) -> FakeWriter | WeaviateWriter:
        if self.target_error is not None:
            raise self.target_error
        self.consistency.append(consistency)
        return self.writer

    def _reader(self, settings: WeaviateSettings) -> FakeReader:
        if self.target_error is not None:
            raise self.target_error
        return FakeReader(self.store or self.writer)  # type: ignore[arg-type]

    def _deleter(
        self, settings: WeaviateSettings, consistency: str | None
    ) -> FakeDeleter:
        if self.target_error is not None:
            raise self.target_error
        self.delete_consistency.append(consistency)
        return self.deleter

    def _schema(self, settings: WeaviateSettings) -> FakeSchema:
        self.schema_calls += 1
        if self.target_error is not None:
            raise self.target_error
        return self.schema


# The schema of state.sqlite before the state stored the target of a copy.
OLD_STATE_SCHEMA = """
CREATE TABLE IF NOT EXISTS collections (
    ref TEXT PRIMARY KEY,
    total INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS partitions (
    ref TEXT NOT NULL,
    key TEXT NOT NULL,
    start TEXT,
    stop TEXT,
    cursor TEXT,
    finished INTEGER NOT NULL DEFAULT 0,
    copied INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (ref, key)
);
CREATE TABLE IF NOT EXISTS rejects (
    ref TEXT NOT NULL,
    source_id TEXT NOT NULL,
    reason TEXT NOT NULL,
    PRIMARY KEY (ref, source_id)
);
CREATE TABLE IF NOT EXISTS hashes (
    ref TEXT NOT NULL,
    source_id TEXT NOT NULL,
    hash BLOB NOT NULL,
    PRIMARY KEY (ref, source_id)
);
"""


def old_state_file(
    path: Path, ref: str, total: int, partition: Partition, cursor: str, copied: int
) -> None:
    """Write a state file in OLD_STATE_SCHEMA: `ref` with one unfinished partition."""
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path)
    try:
        with db:
            db.executescript(OLD_STATE_SCHEMA)
            db.execute(
                "INSERT INTO collections (ref, total) VALUES (?, ?)", (ref, total)
            )
            db.execute(
                "INSERT INTO partitions (ref, key, start, stop, cursor, copied) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (ref, partition.key, partition.start, partition.stop, cursor, copied),
            )
    finally:
        db.close()


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
            "WEAVIATE_GRPC_URL": "http://localhost:50099",
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


def invoke(
    fakes: FakeFactories, *args: str, input: str | None = None, **env: str | None
) -> Result:
    """Run the app of `fakes` in `cli_env` with `env` applied over it.

    `input` is what the command reads on stdin.
    """
    return CliRunner().invoke(fakes.app(), list(args), input=input, env=cli_env(**env))
