import time
from collections.abc import Callable, Iterator
from typing import Any, Protocol, TypeVar, cast

import weaviate
from weaviate.classes.query import Filter
from weaviate.collections import Collection

from to_weaviate.plan import CollectionPlan
from to_weaviate.retry import with_retry
from to_weaviate.target.schema import transient
from to_weaviate.target.writer import TargetObject

T = TypeVar("T")

FETCH_BATCH = 100
SOURCE_ID = "source_id"


class TargetReader(Protocol):
    def count(self, entry: CollectionPlan) -> int: ...

    def fetch(
        self, entry: CollectionPlan, uuids: list[str]
    ) -> dict[str, TargetObject]: ...

    def near_vector(
        self, entry: CollectionPlan, vector: list[float], limit: int
    ) -> list[str]: ...  # source ids

    def filter_equal(
        self, entry: CollectionPlan, prop: str, value: Any, limit: int
    ) -> list[str]: ...  # source ids

    def source_ids(
        self, entry: CollectionPlan
    ) -> Iterator[tuple[str, str]]: ...  # (uuid, source_id)


def _source_id(obj: Any) -> str:
    """The object's source_id, or its uuid when the property is not set."""
    value = obj.properties.get(SOURCE_ID)
    return str(obj.uuid) if value is None else str(value)


class WeaviateReader:
    def __init__(
        self,
        client: weaviate.WeaviateClient,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._client = client
        self._sleep = sleep

    def _collection(self, entry: CollectionPlan) -> Collection[Any, Any]:
        collection = self._client.collections.get(entry.target.name)
        if entry.target.tenant:
            collection = collection.with_tenant(entry.target.tenant)
        return collection

    def _call(self, call: Callable[[], T]) -> T:
        return with_retry(call, transient, sleep=self._sleep)

    def count(self, entry: CollectionPlan) -> int:
        collection = self._collection(entry)
        result = self._call(lambda: collection.aggregate.over_all(total_count=True))
        return result.total_count or 0

    def fetch(self, entry: CollectionPlan, uuids: list[str]) -> dict[str, TargetObject]:
        """Stored objects by uuid, with set properties and kept vectors only."""
        collection = self._collection(entry)
        kept = {v.name for v in entry.target.vectors if v.mode == "keep"}
        found: dict[str, TargetObject] = {}
        for start in range(0, len(uuids), FETCH_BATCH):
            batch = uuids[start : start + FETCH_BATCH]
            response = self._call(
                lambda b=batch: collection.query.fetch_objects(  # type: ignore[misc]
                    filters=Filter.by_id().contains_any(b),
                    include_vector=True,
                    limit=len(b),
                )
            )
            for obj in response.objects:
                properties = {k: v for k, v in obj.properties.items() if v is not None}
                vectors = {
                    name: cast(list[float], vector)
                    for name, vector in obj.vector.items()
                    if name in kept
                }
                found[str(obj.uuid)] = TargetObject(str(obj.uuid), properties, vectors)
        return found

    def near_vector(
        self, entry: CollectionPlan, vector: list[float], limit: int
    ) -> list[str]:
        kept = [v.name for v in entry.target.vectors if v.mode == "keep"]
        if not kept:
            raise ValueError(f"{entry.target.name} has no kept vector to search")
        collection = self._collection(entry)
        response = self._call(
            lambda: collection.query.near_vector(
                near_vector=vector,
                target_vector=kept[0],
                limit=limit,
                return_properties=[SOURCE_ID],
            )
        )
        return [_source_id(obj) for obj in response.objects]

    def filter_equal(
        self, entry: CollectionPlan, prop: str, value: Any, limit: int
    ) -> list[str]:
        collection = self._collection(entry)
        response = self._call(
            lambda: collection.query.fetch_objects(
                filters=Filter.by_property(prop).equal(value),
                limit=limit,
                return_properties=[SOURCE_ID],
            )
        )
        return [_source_id(obj) for obj in response.objects]

    def source_ids(self, entry: CollectionPlan) -> Iterator[tuple[str, str]]:
        # The client's iterator keeps its cursor when a page request raises,
        # so a retried next() asks for the same page again.
        objects = iter(self._collection(entry).iterator(return_properties=[SOURCE_ID]))
        while (obj := self._call(lambda: next(objects, None))) is not None:
            yield str(obj.uuid), _source_id(obj)
