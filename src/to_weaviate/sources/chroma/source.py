import math
import random
import threading
import time
from collections import Counter
from collections.abc import Callable, Iterator
from typing import Any, ClassVar

from to_weaviate.sources.base import (
    FieldStats,
    Page,
    Partition,
    Record,
    SourceCollection,
    SourceNotCompacted,
)
from to_weaviate.sources.chroma.client import (
    CHROMA_MAX_PAGE,
    COLLECTION_GONE_HINT,
    ChromaClient,
    ChromaError,
)
from to_weaviate.sources.chroma.config import (
    distance_of,
    embedding_of,
    sparse_keys_of,
)
from to_weaviate.types import SPARSE_VECTOR, value_type

DOCUMENT_SAMPLE_PAGES = 10
SAMPLE_PLACES = 10
READ_INCLUDE = ["documents", "metadatas", "uris"]
# The INDEXING_TIMEOUT of the acceptance seed: on CI, with the compactor at
# 200m CPU, 50,000 records needed more than 300 s to compact.
COMPACTION_WAIT_SECONDS = 600.0
# indexing_status moves in the 60 s steps of the frontend's collection cache.
COMPACTION_POLL_SECONDS = 5.0


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

    def __init__(
        self,
        client: ChromaClient,
        tenant: str,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._client = client
        self._tenant = tenant
        self._sleep = sleep
        self._clock = clock
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
        # A listing can give the dimension as null. The sampled pages then
        # read embeddings until one record has an embedding.
        dimension = shell.dimension
        dimension_from = None if dimension is None else "listing"
        types: dict[str, Counter[str]] = {}
        total_pages = _page_count(count)
        wanted = total_pages if sample_pages is None else sample_pages
        for page in _spaced_pages(total_pages, wanted):
            include = (
                ["metadatas"] if dimension is not None else ["metadatas", "embeddings"]
            )
            response = self._get(
                shell,
                limit=CHROMA_MAX_PAGE,
                offset=page * CHROMA_MAX_PAGE,
                include=include,
            )
            for record in _records(response):
                if dimension is None and record.vector:
                    dimension, dimension_from = len(record.vector), "sample"
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
            dimension=dimension,
            distance=shell.distance,
            embedding=shell.embedding,
            fields=fields,
            sparse_keys=sorted(sparse),
            document_bytes_estimate=self._document_bytes(shell),
            dimension_from=dimension_from,
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

    def _indexing_status(self, collection: SourceCollection) -> dict[str, Any] | None:
        return self._client.indexing_status(
            self._tenant, collection.namespace, collection.ref
        )

    def wait_compacted(
        self, collection: SourceCollection, waiting: Callable[[int, int], None]
    ) -> dict[str, int] | None:
        """Wait until indexing_status counts no write left to compact.

        Distributed Chroma serves new writes from a log until a compaction run
        moves them into the collection. A run of more than 5,000 records gives
        them new offsets in another order, so a read by offset across it skips
        records. Single-node Chroma has no compaction: one request, then None.

        Each poll reads the state first and the status second, and a 0 returns
        the state read just before it. The 0 comes from the frontend's cache,
        which then holds every write, so a compaction that commits after the
        state read changes the state that the caller reads after its last
        page. A state read after the 0 could take a version that the cache
        serves only later, in the middle of the pages.

        `waiting(unindexed, total)` is called each time the count changes.
        Raises SourceNotCompacted after COMPACTION_WAIT_SECONDS.
        """
        started = self._clock()
        if self._indexing_status(collection) is None:
            return None
        reported: int | None = None
        while True:
            state = self.compaction_state(collection)
            status = self._indexing_status(collection)
            if status is None:
                return None
            unindexed, total = status["num_unindexed_ops"], status["total_ops"]
            if unindexed == 0:
                return state
            if unindexed != reported:
                waiting(unindexed, total)
                reported = unindexed
            if self._clock() - started >= COMPACTION_WAIT_SECONDS:
                raise SourceNotCompacted(
                    collection.name, unindexed, total, COMPACTION_WAIT_SECONDS
                )
            self._sleep(COMPACTION_POLL_SECONDS)

    def compaction_state(self, collection: SourceCollection) -> dict[str, int]:
        """The `version` and `log_position` of the collection document.

        Each compaction run raises them. A distributed answer without them
        raises KeyError, an internal error, instead of switching the guard off.
        """
        raw = self._client.collection_by_name(
            self._tenant, collection.namespace, collection.name
        )
        if raw["id"] != collection.ref:
            raise ChromaError(
                200,
                f"the collection {collection.name} has another id than the "
                "inspected one",
                hint=COLLECTION_GONE_HINT,
            )
        return {"version": raw["version"], "log_position": raw["log_position"]}

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
