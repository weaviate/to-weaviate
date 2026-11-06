import json
import math
import random
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

from to_weaviate.sources.base import Partition
from to_weaviate.sources.chroma.client import CHROMA_MAX_PAGE
from to_weaviate.sources.chroma.source import ChromaSource

FIXTURES = Path(__file__).parent / "fixtures"


@dataclass
class Call:
    limit: int
    offset: int
    include: list[str]
    where: dict | None


class FakeClient:
    def __init__(self, collection, records):
        self.collection = collection
        self.records = records
        self.calls: list[Call] = []
        self.queries: list[tuple[list[float], int]] = []
        self.count_calls = 0

    def databases(self, tenant):
        return ["db"]

    def collections(self, tenant, database):
        return [self.collection] if database == "db" else []

    def count(self, tenant, database, collection_id):
        self.count_calls += 1
        return len(self.records)

    def get(
        self, tenant, database, collection_id, *, limit, offset, include, where=None
    ):
        self.calls.append(Call(limit, offset, list(include), where))
        rows = self.records
        if where is not None:
            ((key, cond),) = where.items()
            rows = [r for r in rows if (r["metadata"] or {}).get(key) == cond["$eq"]]
        rows = rows[offset : offset + limit]

        def column(name, field):
            return [r[field] for r in rows] if name in include else None

        return {
            "ids": [r["id"] for r in rows],
            "embeddings": column("embeddings", "vector"),
            "documents": column("documents", "document"),
            "uris": column("uris", "uri"),
            "metadatas": column("metadatas", "metadata"),
        }

    def query(self, tenant, database, collection_id, *, embedding, n_results):
        self.queries.append((embedding, n_results))
        return ["r1", "r2"]

    def calls_including(self, include):
        return sum(all(name in c.include for name in include) for c in self.calls)


class LazyRecords(Sequence):
    """`n` records like make_records gives, built only when read."""

    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(self.n))]
        return {
            "id": f"id{index}",
            "vector": [float(index), 1.0],
            "document": None,
            "uri": None,
            "metadata": {},
        }


def make_records(n, years_text_from=None):
    records = []
    for i in range(n):
        year = (
            str(2000 + i)
            if years_text_from is not None and i >= years_text_from
            else 2000 + i
        )
        metadata = {"year": year}
        if i % 2 == 0:
            metadata["tags"] = ["a", "b"]
        records.append(
            {
                "id": f"id{i}",
                "vector": [float(i), 1.0],
                "document": f"doc {i}" if i % 5 else None,
                "uri": None,
                "metadata": metadata,
            }
        )
    return records


@pytest.fixture
def collection():
    return json.loads((FIXTURES / "collection_1x_hnsw.json").read_text())


@pytest.fixture
def fake_client(collection):
    return FakeClient(collection, make_records(650, years_text_from=600))


@pytest.fixture
def source(fake_client):
    return ChromaSource(fake_client, tenant="t")


@pytest.fixture
def inspected(source):
    return source.inspect(None, None, None)[0]


def test_inspect_scans_all_metadata(fake_client):
    c = ChromaSource(fake_client, tenant="t").inspect(None, None, None)[0]
    assert c.count == 650
    assert {f.key: f.types for f in c.fields}["year"] == {"int": 600, "text": 50}
    assert fake_client.calls_including(["metadatas"]) == 3
    assert all("embeddings" not in call.include for call in fake_client.calls)


def test_inspect_collection_fields(inspected):
    assert inspected.ref == "5b2f1c1e-0f0a-4f5e-9a39-2f6f6d5d0a11"
    assert inspected.namespace == "db"
    assert inspected.name == "docs"
    assert inspected.dimension == 1536
    assert inspected.distance == "cosine"
    assert inspected.embedding["name"] == "openai"
    assert [f.key for f in inspected.fields] == ["tags", "year"]
    assert {f.key: f.types for f in inspected.fields}["tags"] == {"text[]": 325}
    assert inspected.sparse_keys == []


def test_inspect_sample_pages_limits_metadata_pages(fake_client):
    ChromaSource(fake_client, tenant="t").inspect(None, None, 2)
    assert fake_client.calls_including(["metadatas"]) == 2


def test_inspect_filters_by_name_and_namespace(source):
    assert source.inspect(None, ["nope"], None) == []
    assert source.inspect(["other"], None, None) == []
    assert len(source.inspect(["db"], ["docs"], None)) == 1


def test_inspect_document_bytes_estimate(fake_client):
    fake_client.records = [
        {"id": "a", "vector": None, "document": "abcd", "uri": None, "metadata": {}},
        {"id": "b", "vector": None, "document": None, "uri": None, "metadata": {}},
    ]
    c = ChromaSource(fake_client, tenant="t").inspect(None, None, None)[0]
    assert c.document_bytes_estimate == 4
    assert fake_client.calls_including(["documents"]) >= 1


def test_inspect_empty_collection(fake_client):
    fake_client.records = []
    c = ChromaSource(fake_client, tenant="t").inspect(None, None, None)[0]
    assert c.count == 0 and c.fields == [] and c.document_bytes_estimate == 0


def test_inspect_sparse_keys_from_values(fake_client):
    sparse = {"#type": "sparse_vector", "indices": [1], "values": [1.0]}
    fake_client.records = [
        {
            "id": "a",
            "vector": None,
            "document": None,
            "uri": None,
            "metadata": {"sp": sparse, "k": 1},
        }
    ]
    c = ChromaSource(fake_client, tenant="t").inspect(None, None, None)[0]
    assert c.sparse_keys == ["sp"]
    assert [f.key for f in c.fields] == ["k"]


def test_partitions_cover_range_without_overlap(source, inspected):
    parts = source.partitions(inspected, 4)
    spans = sorted((int(p.start), int(p.stop)) for p in parts)
    assert spans[0][0] == 0 and spans[-1][1] == 650
    assert all(a[1] == b[0] for a, b in zip(spans, spans[1:], strict=False))
    assert all(p.key == f"{p.start}-{p.stop}" for p in parts)


def test_partitions_small_and_empty(source, inspected):
    from dataclasses import replace

    parts = source.partitions(replace(inspected, count=2), 4)
    assert sorted((int(p.start), int(p.stop)) for p in parts) == [(0, 1), (1, 2)]
    assert source.partitions(replace(inspected, count=0), 4) == []


def test_read_ends_partition_at_stop(source, fake_client, inspected):
    page = source.read(inspected, Partition("0-300", "0", "300"), None, False)
    assert len(page.records) == 300
    assert page.next_cursor is None
    assert fake_client.calls[-1].include == ["documents", "metadatas", "uris"]


def test_read_cursor_continues_and_clamps(source, fake_client, inspected):
    part = Partition("0-400", "0", "400")
    first = source.read(inspected, part, None, True)
    assert len(first.records) == 300 and first.next_cursor == "300"
    assert "embeddings" in fake_client.calls[-1].include
    second = source.read(inspected, part, first.next_cursor, False)
    assert len(second.records) == 100 and second.next_cursor is None
    assert fake_client.calls[-1].limit == 100


def test_read_open_ended_partition_stops_on_short_page(source, inspected):
    part = Partition("600-", "600", None)
    page = source.read(inspected, part, None, False)
    assert len(page.records) == 50 and page.next_cursor is None


def test_read_record_mapping(source, inspected):
    page = source.read(inspected, Partition("0-5", "0", "5"), None, True)
    first = page.records[0]
    assert first.id == "id0" and first.vector == [0.0, 1.0]
    assert first.document is None and first.metadata["year"] == 2000
    assert page.records[1].document == "doc 1"


def test_read_null_metadata_becomes_empty(source, fake_client, inspected):
    fake_client.records[3]["metadata"] = None
    page = source.read(inspected, Partition("0-5", "0", "5"), None, False)
    assert page.records[3].metadata == {}


def test_read_decodes_a_null_in_an_embedding_as_nan(source, fake_client, inspected):
    fake_client.records[1]["vector"] = [None, 1.0]
    page = source.read(inspected, Partition("0-3", "0", "3"), None, True)
    first, second = page.records[0].vector, page.records[1].vector
    assert first == [0.0, 1.0]
    assert second is not None and math.isnan(second[0]) and second[1] == 1.0


def test_read_without_vectors_gives_none(source, inspected):
    page = source.read(inspected, Partition("0-2", "0", "2"), None, False)
    assert all(r.vector is None for r in page.records)


def test_ids_pages_without_includes(source, fake_client, inspected):
    fake_client.calls.clear()
    ids = list(source.ids(inspected))
    assert len(ids) == 650 and ids[0] == "id0" and ids[-1] == "id649"
    assert [c.include for c in fake_client.calls] == [[], [], []]


def test_count(source, inspected):
    assert source.count(inspected) == 650


def test_tenant_and_namespaces(source):
    assert source.tenant == "t"
    assert source.namespaces() == ["db"]


def test_sample_size_and_offsets(source, fake_client, inspected):
    fake_client.calls.clear()
    records = source.sample(inspected, 100, random.Random(1), False)
    assert len(records) == 100
    assert len({r.id for r in records}) == 100
    # 10 pages of 10 records; page i lies in the i-th tenth of the 650 records.
    assert [c.limit for c in fake_client.calls] == [10] * 10
    assert [c.offset // 65 for c in fake_client.calls] == list(range(10))
    assert [(c.offset + c.limit - 1) // 65 for c in fake_client.calls] == list(
        range(10)
    )


# A sample reads min(size, max(10, ceil(size / CHROMA_MAX_PAGE))) pages of at
# most ceil(size / pages) records.
@pytest.mark.parametrize(
    "count, size, pages",
    [(650, 301, 10), (650, 30, 10), (650, 3, 3), (5000, 4000, 14)],
)
def test_sample_reads_ceil_pages(source, fake_client, inspected, count, size, pages):
    fake_client.records = LazyRecords(count)
    fake_client.calls.clear()
    records = source.sample(
        replace(inspected, count=count), size, random.Random(1), True
    )
    limits = [c.limit for c in fake_client.calls]
    assert len(limits) == pages and sum(limits) == size
    assert max(limits) == math.ceil(size / pages) <= CHROMA_MAX_PAGE
    assert all("embeddings" in c.include for c in fake_client.calls)
    assert len({r.id for r in records}) == size


def test_sample_reads_the_live_count_not_the_inspected_one(
    source, fake_client, inspected
):
    fake_client.records = LazyRecords(1000)
    fake_client.count_calls = 0
    records = source.sample(
        replace(inspected, count=100_000), 300, random.Random(1), False
    )
    assert len({r.id for r in records}) == len(records) == 300
    assert fake_client.count_calls == 1


def test_sample_of_300_from_100000_comes_from_10_places(source, fake_client, inspected):
    fake_client.records = LazyRecords(100_000)
    fake_client.calls.clear()
    records = source.sample(
        replace(inspected, count=100_000), 300, random.Random(1), False
    )
    assert len({r.id for r in records}) == 300
    assert [c.limit for c in fake_client.calls] == [30] * 10
    assert [c.offset // 10_000 for c in fake_client.calls] == list(range(10))
    assert [(c.offset + c.limit - 1) // 10_000 for c in fake_client.calls] == list(
        range(10)
    )


def test_sample_of_10000_from_12000_returns_10000(source, fake_client, inspected):
    fake_client.records = LazyRecords(12_000)
    records = source.sample(
        replace(inspected, count=12_000), 10_000, random.Random(1), False
    )
    assert len({r.id for r in records}) == len(records) == 10_000


def test_sample_of_at_least_the_count_reads_every_record(
    source, fake_client, inspected
):
    fake_client.calls.clear()
    records = source.sample(inspected, 650, random.Random(1), False)
    assert sorted(r.id for r in records) == sorted(f"id{i}" for i in range(650))
    assert [(c.offset, c.limit) for c in fake_client.calls] == [
        (0, 300),
        (300, 300),
        (600, 50),
    ]


def test_sample_of_an_empty_source_reads_no_page(source, fake_client, inspected):
    fake_client.records = []
    fake_client.calls.clear()
    assert source.sample(inspected, 300, random.Random(1), True) == []
    assert fake_client.calls == []


def test_vector_search(source, fake_client, inspected):
    assert source.vector_search(inspected, [1.0, 2.0], 5) == ["r1", "r2"]
    assert fake_client.queries == [([1.0, 2.0], 5)]


def test_filter_equal(source, fake_client, inspected):
    ids = source.filter_equal(inspected, "year", 2001, 10)
    assert ids == ["id1"]
    assert fake_client.calls[-1].where == {"year": {"$eq": 2001}}
    assert fake_client.calls[-1].include == []


def test_capabilities():
    assert ChromaSource.name == "chroma"
    assert ChromaSource.capabilities == frozenset(
        {"vectors", "vector_search", "filter"}
    )
