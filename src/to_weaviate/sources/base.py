import random
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any, ClassVar, Protocol


@dataclass(frozen=True)
class FieldStats:
    key: str
    types: dict[str, int]


@dataclass(frozen=True)
class SourceCollection:
    ref: str
    namespace: str
    name: str
    count: int
    dimension: int | None
    distance: str | None
    embedding: dict[str, Any] | None
    fields: list[FieldStats]
    sparse_keys: list[str]
    document_bytes_estimate: int
    # "listing" or "sample": where `dimension` came from; None without one.
    dimension_from: str | None = None


@dataclass(frozen=True)
class Record:
    id: str
    vector: list[float] | None
    document: str | None
    uri: str | None
    metadata: dict[str, Any]


@dataclass(frozen=True)
class Partition:
    key: str
    start: str | None
    stop: str | None


@dataclass(frozen=True)
class Page:
    records: list[Record]
    next_cursor: str | None  # None means the partition is finished


class SourceChanged(Exception):
    """The source compacted `collection` during a read: `before` became `after`."""

    def __init__(
        self, collection: str, before: dict[str, int], after: dict[str, int]
    ) -> None:
        super().__init__(f"{collection} changed from {before} to {after}")
        self.collection = collection
        self.before = before
        self.after = after


class SourceNotCompacted(Exception):
    """After `seconds`, `unindexed` of `total` operations were not compacted."""

    def __init__(
        self, collection: str, unindexed: int, total: int, seconds: float
    ) -> None:
        super().__init__(
            f"{collection}: {unindexed} of {total} operations not compacted "
            f"after {seconds} s"
        )
        self.collection = collection
        self.unindexed = unindexed
        self.total = total
        self.seconds = seconds


class Source(Protocol):
    name: ClassVar[str]
    capabilities: ClassVar[frozenset[str]]  # subset of vectors, vector_search, filter

    @property
    def tenant(self) -> str: ...

    @property
    def page_size(self) -> int:
        """The most records that one read returns."""
        ...

    def namespaces(self) -> list[str]:
        """Every namespace of the tenant, in the order the source lists them."""
        ...

    def inspect(
        self,
        namespaces: list[str] | None,
        collections: list[str] | None,
        sample_pages: int | None,
    ) -> list[SourceCollection]: ...

    def partitions(
        self, collection: SourceCollection, parallelism: int
    ) -> list[Partition]: ...

    def read(
        self,
        collection: SourceCollection,
        partition: Partition,
        cursor: str | None,
        with_vectors: bool,
    ) -> Page: ...

    def ids(self, collection: SourceCollection) -> Iterator[str]: ...

    def ids_at(
        self, collection: SourceCollection, offset: int, limit: int
    ) -> list[str]:
        """The ids of up to `limit` records from `offset`, in the order `read` uses."""
        ...

    def count(self, collection: SourceCollection) -> int: ...

    def wait_compacted(
        self, collection: SourceCollection, waiting: Callable[[int, int], None]
    ) -> dict[str, int] | None:
        """Wait until the source has no write of `collection` left to compact.

        Returns the compaction state for `compaction_state` to compare with,
        or None when the source has no compaction; the caller then compares
        nothing. `waiting(unindexed, total)` reports the wait. Raises
        SourceNotCompacted when the wait ends first. The engine knows no
        compaction: it only compares the states.
        """
        ...

    def compaction_state(self, collection: SourceCollection) -> dict[str, int]:
        """A state of `collection` that changes when the source compacts it."""
        ...

    def sample(
        self,
        collection: SourceCollection,
        size: int,
        rng: random.Random,
        with_vectors: bool,
    ) -> list[Record]: ...

    def vector_search(
        self, collection: SourceCollection, vector: list[float], limit: int
    ) -> list[str]: ...

    def filter_equal(
        self, collection: SourceCollection, key: str, value: Any, limit: int
    ) -> list[str]: ...
