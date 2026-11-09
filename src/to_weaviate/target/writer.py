import json
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any, Protocol

import weaviate
from weaviate.classes.config import ConsistencyLevel
from weaviate.classes.data import DataObject
from weaviate.exceptions import WeaviateBaseError, WeaviateInsertManyAllFailedError

from to_weaviate.ids import to_uuid
from to_weaviate.plan import CollectionPlan
from to_weaviate.retry import ATTEMPTS, backoff_delay, with_retry
from to_weaviate.sources.base import Record
from to_weaviate.target.schema import refused_for_size, transient
from to_weaviate.types import coerce

MAX_INSERT_BYTES = 4 * 1024 * 1024
OBJECT_OVERHEAD_BYTES = 64
BYTES_PER_FLOAT = 4
# Client 4.24.1 puts this text before the server's errors.
ALL_FAILED_PREFIX = (
    "Every object failed during insertion. Here is the set of all errors: "
)


@dataclass(frozen=True)
class Reject:
    source_id: str
    reason: str


# Parts of the per-object errors that Weaviate 1.40.4 gives while a node or a
# shard is unavailable, seen during a rolling restart of a 3-node cluster. The
# coordinator could not reach the replica, or the shard was not loaded yet.
OUTAGE_TEXTS = (
    "connection refused",
    "connection reset",
    "broken pipe",
    "deadline exceeded",
    "context canceled",
    "resolve node name",
    "no route to host",
    "i/o timeout",
    "unavailable",
    "not ready",
)


def node_outage(message: str) -> bool:
    """Whether a per-object error says that a node or a shard was unavailable."""
    return any(text in message for text in OUTAGE_TEXTS)


class BatchRefused(Exception):
    """Weaviate refused objects of a chunk on every attempt.

    `refused` is the number of objects refused on the last attempt, or None
    when Weaviate refused the chunk as a whole.
    """

    def __init__(
        self, size: int, collection: str, reason: str, refused: int | None = None
    ) -> None:
        what = (
            "every object"
            if refused is None
            else f"{refused} object{'' if refused == 1 else 's'}"
        )
        super().__init__(
            f"Weaviate refused {what} of a batch of {size} in {collection}: {reason}"
        )


@dataclass(frozen=True)
class TargetObject:
    uuid: str
    properties: dict[str, Any]
    vectors: dict[str, list[float]]  # kept vectors only, keyed by vector name


def unmapped_key_text(key: str) -> str:
    """Why metadata key `key`, which the plan neither maps nor drops, is refused."""
    return (
        f"metadata key {key} is not in plan.yaml; add a property for it or list "
        "it under dropped"
    )


def _source_value(record: Record, key: str) -> Any:
    if key == "#document":
        return record.document
    if key == "#id":
        return record.id
    if key == "#uri":
        return record.uri
    return record.metadata.get(key)


def to_object(record: Record, entry: CollectionPlan) -> TargetObject | Reject:
    target = entry.target
    known = {p.source_key for p in target.properties}
    known.update(d.source_key for d in target.dropped)
    for key in record.metadata:
        if key not in known:
            return Reject(record.id, unmapped_key_text(key))
    properties: dict[str, Any] = {}
    for prop in target.properties:
        value = _source_value(record, prop.source_key)
        if value is None:
            continue
        if not prop.source_key.startswith("#"):
            try:
                value = coerce(value, prop.type)
            except ValueError as exc:
                return Reject(record.id, f"{prop.source_key}: {exc}")
        properties[prop.name] = value
    vectors: dict[str, list[float]] = {}
    # With a kept vector, a record without a document gets no embedded vector.
    keeps = any(v.mode == "keep" for v in target.vectors)
    for vector in target.vectors:
        if vector.mode == "keep":
            if record.vector is None:
                return Reject(record.id, "no vector")
            vectors[vector.name] = record.vector
        elif not record.document and not keeps:
            return Reject(record.id, "no document to embed")
    return TargetObject(to_uuid(record.id), properties, vectors)


def _estimate(obj: TargetObject) -> int:
    size = len(json.dumps(obj.properties, ensure_ascii=False).encode("utf-8"))
    size += BYTES_PER_FLOAT * sum(len(v) for v in obj.vectors.values())
    return size + OBJECT_OVERHEAD_BYTES


def chunks(
    objects: list[TargetObject], max_bytes: int | None = None
) -> Iterator[list[TargetObject]]:
    limit = MAX_INSERT_BYTES if max_bytes is None else max_bytes
    chunk: list[TargetObject] = []
    total = 0
    for obj in objects:
        size = _estimate(obj)
        if chunk and total + size > limit:
            yield chunk
            chunk, total = [], 0
        chunk.append(obj)
        total += size
    if chunk:
        yield chunk


def _refused_or_transient(exc: Exception) -> float | None:
    """`transient`, which also accepts a refusal of every object of a batch."""
    if isinstance(exc, WeaviateInsertManyAllFailedError):
        return 0.0
    return transient(exc)


def _transient_unless_too_large(exc: Exception) -> float | None:
    """`transient`, except for a message that gRPC refuses for its size.

    A refusal of the one object of a chunk is retried only for a node outage.
    """
    if isinstance(exc, WeaviateInsertManyAllFailedError):
        return 0.0 if node_outage(str(exc)) else None
    return None if refused_for_size(exc) else transient(exc)


def _source_id(obj: TargetObject) -> str:
    return str(obj.properties.get("source_id", obj.uuid))


class Writer(Protocol):
    def write(
        self, entry: CollectionPlan, objects: list[TargetObject]
    ) -> list[Reject]: ...


class WeaviateWriter:
    def __init__(
        self,
        client: weaviate.WeaviateClient,
        consistency: str | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._client = client
        self._consistency = consistency
        self._sleep = sleep

    def write(self, entry: CollectionPlan, objects: list[TargetObject]) -> list[Reject]:
        """Insert `objects` and return the objects that Weaviate refused.

        A chunk of more than one object that Weaviate refuses as a whole is
        retried like a transient error. An object that Weaviate refuses with
        an error of a node outage is sent again, with the others of its chunk
        that it refused so, after a delay. When the retries are used up, this
        raises BatchRefused. An object refused for anything else is a reject,
        and so is a chunk of one object that gRPC refuses for its size.
        """
        collection = self._client.collections.get(entry.target.name)
        if entry.target.tenant:
            collection = collection.with_tenant(entry.target.tenant)
        if self._consistency:
            collection = collection.with_consistency_level(
                ConsistencyLevel(self._consistency)
            )
        rejects: list[Reject] = []
        for chunk in chunks(objects):
            rejects.extend(self._write_chunk(collection, entry.target.name, chunk))
        return rejects

    def _write_chunk(
        self, collection: Any, name: str, chunk: list[TargetObject]
    ) -> list[Reject]:
        rejects: list[Reject] = []
        pending = chunk
        for attempt in range(ATTEMPTS):
            batch = [
                DataObject(
                    properties=o.properties, vector=o.vectors or None, uuid=o.uuid
                )
                for o in pending
            ]
            single = len(pending) == 1
            try:
                result = with_retry(
                    lambda b=batch: collection.data.insert_many(b),  # type: ignore[misc]
                    _transient_unless_too_large if single else _refused_or_transient,
                    sleep=self._sleep,
                )
            except WeaviateInsertManyAllFailedError as exc:
                reason = str(exc).removeprefix(ALL_FAILED_PREFIX)
                if single and not node_outage(reason):
                    rejects.append(Reject(_source_id(pending[0]), str(exc)))
                    return rejects
                refused = None if len(pending) == len(chunk) else len(pending)
                raise BatchRefused(len(chunk), name, reason, refused) from exc
            except WeaviateBaseError as exc:
                if not (single and refused_for_size(exc)):
                    raise
                rejects.append(Reject(_source_id(pending[0]), exc.message))
                return rejects
            again: list[TargetObject] = []
            reason = ""
            for index, error in result.errors.items():
                if node_outage(error.message):
                    again.append(pending[index])
                    reason = reason or error.message
                else:
                    rejects.append(Reject(_source_id(pending[index]), error.message))
            if not again:
                return rejects
            if attempt == ATTEMPTS - 1:
                raise BatchRefused(len(chunk), name, reason, len(again))
            self._sleep(backoff_delay(attempt))
            pending = again
        raise AssertionError("ATTEMPTS must be at least 1")
