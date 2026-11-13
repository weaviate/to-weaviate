import json
import re
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.parse import urlsplit

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


# How a Weaviate vectorizer module reports the answer of a model provider,
# as in "connection to: OpenAI API failed with status: 401 ...", "connection
# to Cohere API failed with status 403: ..." or "connection to: <url> failed
# with status: 429 ...".
PROVIDER_STATUS = re.compile(
    r"connection to:? (?P<provider>.+?) (?:request )?failed with status:? "
    r"(?P<status>\d{3})\b"
)
# The per-object error of Weaviate 1.40 when the provider's request limit
# resets later than the time that Weaviate gives a batch.
PROVIDER_LIMIT_TEXT = "request rate limit exceeded and will not refresh in time"


class ProviderRefused(Exception):
    """A model provider refused the key that Weaviate sent it for a batch.

    The message does not repeat the provider's text, which can echo the key.
    """

    def __init__(self, provider: str, status: int) -> None:
        super().__init__(
            f"{provider} refused the key that Weaviate sent: HTTP {status}"
        )
        self.provider = provider
        self.status = status


def provider_refusal(message: str) -> ProviderRefused | None:
    """The refusal in `message`, or None when it says something else.

    A provider named by its URL is named by its host.
    """
    for match in PROVIDER_STATUS.finditer(message):
        if match["status"] not in ("401", "403"):
            continue
        provider = match["provider"]
        if "://" in provider:
            provider = urlsplit(provider).hostname or provider
        return ProviderRefused(provider, int(match["status"]))
    return None


def provider_busy(message: str) -> bool:
    """Whether a per-object error says that a model provider answered 429 or 5xx.

    Weaviate does not send the request to the provider again. The status is
    the first one in `message`: a status that the provider's own text repeats
    after it does not count.
    """
    if PROVIDER_LIMIT_TEXT in message:
        return True
    match = PROVIDER_STATUS.search(message)
    if match is None:
        return False
    status = int(match["status"])
    return status == 429 or 500 <= status <= 599


def provider_declined(message: str) -> bool:
    """Whether `message` holds an answer of a model provider that a retry
    does not change: any status other than 401, 403, 429 and 5xx.

    `message` can hold the errors of several objects.
    """
    for match in PROVIDER_STATUS.finditer(message):
        status = int(match["status"])
        if status not in (401, 403, 429) and not 500 <= status <= 599:
            return True
    return False


def _sent_again(message: str) -> bool:
    return node_outage(message) or provider_busy(message)


class BatchRefused(Exception):
    """Weaviate refused objects of a chunk on every attempt.

    `refused` is the number of objects refused on the last attempt, or None
    when Weaviate refused the chunk as a whole. `provider_busy` tells whether
    the reason is a model provider that answered 429 or 5xx.
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
        self.provider_busy = provider_busy(reason)


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
    """`transient`, which also accepts a refusal of every object of a batch.

    A provider that refused the key does not accept it on a retry, and
    neither does a provider that answered an object with another status
    than 429 or 5xx.
    """
    if isinstance(exc, WeaviateInsertManyAllFailedError):
        message = str(exc)
        final = provider_refusal(message) or provider_declined(message)
        return None if final else 0.0
    return transient(exc)


def _transient_unless_too_large(exc: Exception) -> float | None:
    """`transient`, except for a message that gRPC refuses for its size.

    A refusal of the one object of a chunk is retried only for a node outage
    or a model provider that answered 429 or 5xx.
    """
    if isinstance(exc, WeaviateInsertManyAllFailedError):
        return 0.0 if _sent_again(str(exc)) else None
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
        retried like a transient error, unless a model provider answered an
        object of it with a status other than 401, 403, 429 and 5xx: its
        objects are then sent one by one. An object that Weaviate refuses with
        an error of a node outage, or because a model provider answered 429 or
        5xx, is sent again, with the others of its chunk that it refused so,
        after a delay. When the retries are used up, this
        raises BatchRefused. A model provider that refuses the key, for the
        chunk or for one object, raises ProviderRefused. An object refused for
        anything else is a reject, and so is a chunk of one object that gRPC
        refuses for its size.
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
                refused_key = provider_refusal(reason)
                if refused_key is not None:
                    # Not chained: the debug traceback would print the
                    # provider's text, which can echo the key.
                    raise refused_key from None
                if single and not _sent_again(reason):
                    rejects.append(Reject(_source_id(pending[0]), str(exc)))
                    return rejects
                if not single and provider_declined(reason):
                    # The error does not tell which object got which answer.
                    for obj in pending:
                        rejects.extend(self._write_chunk(collection, name, [obj]))
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
                refused_key = provider_refusal(error.message)
                if refused_key is not None:
                    raise refused_key
                if _sent_again(error.message):
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
