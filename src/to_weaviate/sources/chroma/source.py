import math
import random
from collections import Counter
from collections.abc import Iterator
from typing import Any, ClassVar

from to_weaviate.sources.base import (
    FieldStats,
    Page,
    Partition,
    Record,
    SourceCollection,
)
from to_weaviate.sources.chroma.client import CHROMA_MAX_PAGE, ChromaClient
from to_weaviate.sources.chroma.config import (
    distance_of,
    embedding_of,
    sparse_keys_of,
)
from to_weaviate.types import SPARSE_VECTOR, value_type

DOCUMENT_SAMPLE_PAGES = 10
SAMPLE_PLACES = 10
READ_INCLUDE = ["documents", "metadatas", "uris"]


def _page_count(count: int) -> int:
    return math.ceil(count / CHROMA_MAX_PAGE)


def _spaced_pages(total: int, wanted: int) -> list[int]:
    """Up to `wanted` page indexes spread evenly over `total` pages."""
    if wanted >= total:
        return list(range(total))
    return sorted({i * total // wanted for i in range(wanted)})


def _sample_pages(count: int, size: int, rng: random.Random) -> list[tuple[int, int]]:
    """(offset, limit) of each page that a sample of `size` of `count` records reads.

    When `count <= size`, the pages cover every record. Otherwise there are
    `places = min(size, max(SAMPLE_PLACES, ceil(size / CHROMA_MAX_PAGE)))`
    pages. Their limits differ by at most one and add up to `size`. Page i
    starts after the records of the pages before it plus a gap drawn from
    `[i * slack // places, (i + 1) * slack // places]`, where `slack = count -
    size`. The gaps do not decrease, so the pages do not overlap, and the last
    page ends at or before `count`. Page i ends inside the i-th of `places`
    equal parts of `[0, count)` and starts at most one record before it.
    """
    if count <= size:
        return [
            (offset, min(CHROMA_MAX_PAGE, count - offset))
            for offset in range(0, count, CHROMA_MAX_PAGE)
        ]
    places = min(size, max(SAMPLE_PLACES, math.ceil(size / CHROMA_MAX_PAGE)))
    slack = count - size
    pages = []
    for i in range(places):
        before, through = i * size // places, (i + 1) * size // places
        gap = rng.randint(i * slack // places, (i + 1) * slack // places)
        pages.append((before + gap, through - before))
    return pages


def _number(value: Any) -> float:
    """`value` as a float. A value that is not a number, such as None, gives NaN."""
    return float(value) if isinstance(value, int | float) else math.nan


def _records(response: dict[str, Any]) -> list[Record]:
    ids = response["ids"]

    def column(name: str) -> list[Any]:
        return response.get(name) or [None] * len(ids)

    vectors = column("embeddings")
    documents = column("documents")
    uris = column("uris")
    metadatas = column("metadatas")
    return [
        Record(
            id=ids[i],
            vector=None if vectors[i] is None else [_number(x) for x in vectors[i]],
            document=documents[i],
            uri=uris[i],
            metadata=metadatas[i] or {},
        )
        for i in range(len(ids))
    ]


class ChromaSource:
    name: ClassVar[str] = "chroma"
    capabilities: ClassVar[frozenset[str]] = frozenset(
        {"vectors", "vector_search", "filter"}
    )

    def __init__(self, client: ChromaClient, tenant: str) -> None:
        self._client = client
        self._tenant = tenant

    def _get(
        self,
        collection: SourceCollection,
        *,
        limit: int,
        offset: int,
        include: list[str],
        where: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self._client.get(
            self._tenant,
            collection.namespace,
            collection.ref,
            limit=limit,
            offset=offset,
            include=include,
            where=where,
        )

    @property
    def tenant(self) -> str:
        return self._tenant

    def namespaces(self) -> list[str]:
        return self._client.databases(self._tenant)

    def inspect(
        self,
        namespaces: list[str] | None,
        collections: list[str] | None,
        sample_pages: int | None,
    ) -> list[SourceCollection]:
        databases = self.namespaces() if namespaces is None else namespaces
        found: list[SourceCollection] = []
        for database in databases:
            for raw in self._client.collections(self._tenant, database):
                if collections is None or raw["name"] in collections:
                    found.append(self._inspect_one(database, raw, sample_pages))
        return found

    def _inspect_one(
        self, database: str, raw: dict[str, Any], sample_pages: int | None
    ) -> SourceCollection:
        ref = raw["id"]
        count = self._client.count(self._tenant, database, ref)
        shell = SourceCollection(
            ref=ref,
            namespace=database,
            name=raw["name"],
            count=count,
            dimension=raw.get("dimension"),
            distance=distance_of(raw),
            embedding=embedding_of(raw),
            fields=[],
            sparse_keys=[],
            document_bytes_estimate=0,
        )
        types: dict[str, Counter[str]] = {}
        total_pages = _page_count(count)
        wanted = total_pages if sample_pages is None else sample_pages
        for page in _spaced_pages(total_pages, wanted):
            response = self._get(
                shell,
                limit=CHROMA_MAX_PAGE,
                offset=page * CHROMA_MAX_PAGE,
                include=["metadatas"],
            )
            for record in _records(response):
                for key, value in record.metadata.items():
                    types.setdefault(key, Counter())[value_type(value)] += 1
        sparse = set(sparse_keys_of(raw))
        sparse.update(key for key, seen in types.items() if SPARSE_VECTOR in seen)
        fields = [
            FieldStats(key, dict(seen))
            for key, seen in sorted(types.items())
            if key not in sparse
        ]
        return SourceCollection(
            ref=shell.ref,
            namespace=shell.namespace,
            name=shell.name,
            count=count,
            dimension=shell.dimension,
            distance=shell.distance,
            embedding=shell.embedding,
            fields=fields,
            sparse_keys=sorted(sparse),
            document_bytes_estimate=self._document_bytes(shell),
        )

    def _document_bytes(self, collection: SourceCollection) -> int:
        total_pages = _page_count(collection.count)
        if total_pages == 0:
            return 0
        pages = {0, *_spaced_pages(total_pages, DOCUMENT_SAMPLE_PAGES)}
        sampled = 0
        size = 0
        for page in sorted(pages):
            response = self._get(
                collection,
                limit=CHROMA_MAX_PAGE,
                offset=page * CHROMA_MAX_PAGE,
                include=["documents"],
            )
            for record in _records(response):
                sampled += 1
                if record.document is not None:
                    size += len(record.document.encode("utf-8"))
        if sampled == 0:
            return 0
        return int(size / sampled * collection.count)

    def partitions(
        self, collection: SourceCollection, parallelism: int
    ) -> list[Partition]:
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
        offset = int(cursor if cursor is not None else partition.start or 0)
        limit = CHROMA_MAX_PAGE
        stop = None if partition.stop is None else int(partition.stop)
        if stop is not None:
            limit = min(limit, stop - offset)
        if limit <= 0:
            return Page([], None)
        include = [*READ_INCLUDE, "embeddings"] if with_vectors else READ_INCLUDE
        records = _records(
            self._get(collection, limit=limit, offset=offset, include=include)
        )
        next_offset = offset + len(records)
        done = len(records) < limit or (stop is not None and next_offset >= stop)
        return Page(records, None if done else str(next_offset))

    def ids(self, collection: SourceCollection) -> Iterator[str]:
        offset = 0
        while True:
            response = self._get(
                collection, limit=CHROMA_MAX_PAGE, offset=offset, include=[]
            )
            ids: list[str] = response["ids"]
            yield from ids
            if len(ids) < CHROMA_MAX_PAGE:
                return
            offset += CHROMA_MAX_PAGE

    def count(self, collection: SourceCollection) -> int:
        return self._client.count(self._tenant, collection.namespace, collection.ref)

    def sample(
        self,
        collection: SourceCollection,
        size: int,
        rng: random.Random,
        with_vectors: bool,
    ) -> list[Record]:
        """`min(size, count)` different records, read at the `_sample_pages` pages.

        `count` is the source count read at the start of the call, not
        `collection.count`. A source that changes during the call can give
        fewer records.
        """
        include = [*READ_INCLUDE, "embeddings"] if with_vectors else READ_INCLUDE
        seen: dict[str, Record] = {}
        for offset, limit in _sample_pages(self.count(collection), size, rng):
            response = self._get(
                collection, limit=limit, offset=offset, include=include
            )
            for record in _records(response):
                seen.setdefault(record.id, record)
        return list(seen.values())

    def vector_search(
        self, collection: SourceCollection, vector: list[float], limit: int
    ) -> list[str]:
        return self._client.query(
            self._tenant,
            collection.namespace,
            collection.ref,
            embedding=vector,
            n_results=limit,
        )

    def filter_equal(
        self, collection: SourceCollection, key: str, value: Any, limit: int
    ) -> list[str]:
        response = self._get(
            collection,
            limit=limit,
            offset=0,
            include=[],
            where={key: {"$eq": value}},
        )
        ids: list[str] = response["ids"]
        return ids
