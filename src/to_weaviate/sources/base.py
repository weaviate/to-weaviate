import random
from collections.abc import Iterator
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


class Source(Protocol):
    name: ClassVar[str]
    capabilities: ClassVar[frozenset[str]]  # subset of vectors, vector_search, filter

    @property
    def tenant(self) -> str: ...

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

    def count(self, collection: SourceCollection) -> int: ...

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
