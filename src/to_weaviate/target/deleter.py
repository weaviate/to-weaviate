import time
from collections.abc import Callable
from typing import Any, Protocol

import weaviate
from weaviate.classes.config import ConsistencyLevel
from weaviate.classes.query import Filter

from to_weaviate.plan import CollectionPlan
from to_weaviate.retry import with_retry
from to_weaviate.target.schema import transient

DELETE_BATCH = 100


class DeleteFailed(Exception):
    """Weaviate reported `failed` of the `matched` objects of a batch as not deleted."""

    def __init__(self, collection: str, failed: int, matched: int) -> None:
        super().__init__(
            f"Weaviate failed to delete {failed} of {matched} objects of {collection}"
        )
        self.collection = collection
        self.failed = failed
        self.matched = matched


class Deleter(Protocol):
    def delete(self, entry: CollectionPlan, uuids: list[str]) -> int:
        """Delete the objects with `uuids`; returns how many were deleted.

        Raises DeleteFailed when the target deleted only some of a batch.
        """
        ...


class WeaviateDeleter:
    def __init__(
        self,
        client: weaviate.WeaviateClient,
        consistency: str | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._client = client
        self._consistency = consistency
        self._sleep = sleep

    def delete(self, entry: CollectionPlan, uuids: list[str]) -> int:
        collection: Any = self._client.collections.get(entry.target.name)
        if entry.target.tenant:
            collection = collection.with_tenant(entry.target.tenant)
        if self._consistency:
            collection = collection.with_consistency_level(
                ConsistencyLevel(self._consistency)
            )
        deleted = 0
        for start in range(0, len(uuids), DELETE_BATCH):
            batch = uuids[start : start + DELETE_BATCH]
            result = with_retry(
                lambda b=batch: collection.data.delete_many(  # type: ignore[misc]
                    where=Filter.by_id().contains_any(b)
                ),
                transient,
                sleep=self._sleep,
            )
            if result.failed:
                raise DeleteFailed(
                    entry.target.name, int(result.failed), int(result.matches)
                )
            deleted += int(result.successful)
        return deleted
