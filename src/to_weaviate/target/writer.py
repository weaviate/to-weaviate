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
from to_weaviate.retry import with_retry
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


class BatchRefused(Exception):
    """Weaviate refused every object of a chunk of more than one object."""

    def __init__(self, size: int, collection: str, reason: str) -> None:
        super().__init__(
            f"Weaviate refused every object of a batch of {size} in "
            f"{collection}: {reason}"
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
    """`transient`, except for a message that gRPC refuses for its size."""
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
        retried like a transient error. When the retries are used up, this
        raises BatchRefused. A refused chunk of one object becomes one reject,
        and so does a chunk of one object that gRPC refuses for its size.
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
            batch = [
                DataObject(
                    properties=o.properties, vector=o.vectors or None, uuid=o.uuid
                )
                for o in chunk
            ]
            single = len(chunk) == 1
            try:
                result = with_retry(
                    lambda b=batch: collection.data.insert_many(b),  # type: ignore[misc]
                    _transient_unless_too_large if single else _refused_or_transient,
                    sleep=self._sleep,
                )
            except WeaviateInsertManyAllFailedError as exc:
                if not single:
                    reason = str(exc).removeprefix(ALL_FAILED_PREFIX)
                    raise BatchRefused(len(chunk), entry.target.name, reason) from exc
                rejects.append(Reject(_source_id(chunk[0]), str(exc)))
                continue
            except WeaviateBaseError as exc:
                if not (single and refused_for_size(exc)):
                    raise
                rejects.append(Reject(_source_id(chunk[0]), exc.message))
                continue
            for index, error in result.errors.items():
                rejects.append(Reject(_source_id(chunk[index]), error.message))
        return rejects
