import math
import random
import threading
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
from to_weaviate.sources.chroma.client import CHROMA_MAX_PAGE, ChromaClient, ChromaError
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
    page ends at or before `count`. Part i of `[0, count)` is
    `[i * count // places, (i + 1) * count // places)`. Page i starts at most
    one record before part i and ends at or before the end of part i. A page
    of few records can lie wholly before its part.
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


def _no_advance(collection: SourceCollection, offset: int) -> ChromaError:
    return ChromaError(
        200, f"the records of {collection.name} did not advance at offset {offset}"
    )


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
        # The offset and the ids of the last page read, by (ref, partition key).
        # Workers read other partitions at the same time.
        self._last_pages: dict[tuple[str, str], tuple[int, set[str]]] = {}
        self._pages_lock = threading.Lock()

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

    @property
    def page_size(self) -> int:
        return CHROMA_MAX_PAGE

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
        """The page at `cursor`. The next page starts after the rows received.

        A partition ends at its stop or at an empty page: a server can return
        fewer rows than the limit before its last page.
        """
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
        self._check_advance(collection, partition.key, offset, records)
        next_offset = offset + len(records)
        done = not records or (stop is not None and next_offset >= stop)
        return Page(records, None if done else str(next_offset))

    def _check_advance(
        self,
        collection: SourceCollection,
        key: str,
        offset: int,
        records: list[Record],
    ) -> None:
        """Raise ChromaError when the page at `offset` holds only rows of the last page.

        The last page is the page of partition `key` read before at a lower
        offset. A server that ignores `offset` would otherwise give a tail
        without end.
        """
        ids = {r.id for r in records}
        place = (collection.ref, key)
        with self._pages_lock:
            last_offset, last_ids = self._last_pages.get(place, (-1, set()))
            self._last_pages[place] = (offset, ids)
        if ids and offset > last_offset and ids <= last_ids:
            raise _no_advance(collection, offset)

    def ids(self, collection: SourceCollection) -> Iterator[str]:
        """Every id. Pages start after the rows received and end at an empty page."""
        offset = 0
        last: set[str] = set()
        while True:
            response = self._get(
                collection, limit=CHROMA_MAX_PAGE, offset=offset, include=[]
            )
            ids: list[str] = response["ids"]
            if not ids:
                return
            if last.issuperset(ids):
                raise _no_advance(collection, offset)
            yield from ids
            last = set(ids)
            offset += len(ids)

    def ids_at(
        self, collection: SourceCollection, offset: int, limit: int
    ) -> list[str]:
        response = self._get(collection, limit=limit, offset=offset, include=[])
        ids: list[str] = response["ids"]
        return ids

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
