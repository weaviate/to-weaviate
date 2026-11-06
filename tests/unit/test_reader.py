import uuid
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from weaviate.exceptions import UnexpectedStatusCodeError, WeaviateConnectionError

from to_weaviate import plan as p
from to_weaviate.sources.base import Record
from to_weaviate.target.reader import FETCH_BATCH, WeaviateReader
from to_weaviate.target.writer import TargetObject, to_object


def make_entry(tenant: str | None = None) -> p.CollectionPlan:
    return p.CollectionPlan(
        source=p.SourceRef(
            ref="r",
            namespace="n",
            name="docs",
            count=1,
            dimension=2,
            distance=None,
            embedding=None,
        ),
        target=p.TargetPlan(
            name="Docs",
            tenant=tenant,
            vectors=[
                p.VectorPlan(name="source", mode="keep"),
                p.VectorPlan(
                    name="default",
                    mode="embed",
                    vectorizer=p.VectorizerPlan(module="text2vec-weaviate"),
                ),
            ],
            properties=[
                p.PropertyPlan(source_key="#document", name="document", type="text"),
                p.PropertyPlan(source_key="#id", name="source_id", type="text"),
                p.PropertyPlan(source_key="#uri", name="uri", type="text"),
                p.PropertyPlan(source_key="year", name="year", type="int"),
            ],
        ),
    )


ENTRY = make_entry()


def stored(source_id: str | None, year: int | None = None) -> SimpleNamespace:
    """An object as the client returns it: unset properties read as None."""
    return SimpleNamespace(
        uuid=uuid.uuid5(uuid.NAMESPACE_URL, str(source_id)),
        properties={
            "document": "text",
            "source_id": source_id,
            "uri": None,
            "year": year,
        },
        vector={"source": [0.25, 0.5], "default": [9.0, 9.0]},
    )


class FakeIterator:
    """Object iterator that raises `error` once before the object at `fail_at`."""

    def __init__(
        self, objects: list[Any], fail_at: int | None, error: Exception
    ) -> None:
        self.objects = objects
        self.position = 0
        self.fail_at = fail_at
        self.error = error

    def __iter__(self) -> "FakeIterator":
        return self

    def __next__(self) -> Any:
        if self.position == self.fail_at:
            self.fail_at = None
            raise self.error
        if self.position >= len(self.objects):
            raise StopIteration
        self.position += 1
        return self.objects[self.position - 1]


class FakeCollection:
    def __init__(self, objects: list[Any], failures: list[Exception]) -> None:
        self.objects = objects
        self.failures = failures
        self.applied_tenants: list[str] = []
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.aggregate = SimpleNamespace(over_all=self._over_all)
        self.query = SimpleNamespace(
            fetch_objects=self._fetch_objects, near_vector=self._near_vector
        )
        self.iterator_fail_at: int | None = None
        self.tenant_names: set[str] = set()
        self.tenants = SimpleNamespace(exists=self._tenant_exists)

    def _tenant_exists(self, tenant: str) -> bool:
        self.calls.append(("tenants.exists", {"tenant": tenant}))
        return tenant in self.tenant_names

    def _call(self, name: str, kwargs: dict[str, Any]) -> None:
        self.calls.append((name, kwargs))
        if self.failures:
            raise self.failures.pop(0)

    def with_tenant(self, tenant: str) -> "FakeCollection":
        self.applied_tenants.append(tenant)
        return self

    def _over_all(self, **kwargs: Any) -> Any:
        self._call("over_all", kwargs)
        return SimpleNamespace(total_count=len(self.objects))

    def _fetch_objects(self, **kwargs: Any) -> Any:
        self._call("fetch_objects", kwargs)
        f = kwargs["filters"]
        if f.target == "_id":
            found = [o for o in self.objects if str(o.uuid) in f.value]
        else:
            found = [o for o in self.objects if o.properties.get(f.target) == f.value]
        return SimpleNamespace(objects=found[: kwargs["limit"]])

    def _near_vector(self, **kwargs: Any) -> Any:
        self._call("near_vector", kwargs)
        return SimpleNamespace(objects=self.objects[: kwargs["limit"]])

    def iterator(self, **kwargs: Any) -> FakeIterator:
        self.calls.append(("iterator", kwargs))
        error = WeaviateConnectionError("connection lost")
        return FakeIterator(self.objects, self.iterator_fail_at, error)


class FakeClient:
    def __init__(
        self, objects: list[Any], failures: list[Exception] | None = None
    ) -> None:
        self.collection = FakeCollection(objects, failures or [])
        self.names: list[str] = []
        self.existing = {"Docs"}
        self.collections = SimpleNamespace(get=self._get, exists=self._exists)

    def _exists(self, name: str) -> bool:
        return name in self.existing

    def _get(self, name: str) -> FakeCollection:
        self.names.append(name)
        return self.collection


def reader(client: FakeClient) -> WeaviateReader:
    return WeaviateReader(client, sleep=lambda s: None)  # type: ignore[arg-type]


def test_count_reads_the_total_count_of_the_tenant():
    client = FakeClient([stored("a"), stored("b")])
    assert reader(client).count(make_entry(tenant="t1")) == 2
    assert client.names == ["Docs"] and client.collection.applied_tenants == ["t1"]
    assert client.collection.calls == [("over_all", {"total_count": True})]


def test_without_a_tenant_none_is_applied():
    client = FakeClient([stored("a")])
    target = reader(client)
    target.count(ENTRY)
    target.fetch(ENTRY, [str(stored("a").uuid)])
    target.near_vector(ENTRY, [0.1, 0.2], 1)
    target.filter_equal(ENTRY, "year", 1, 1)
    list(target.source_ids(ENTRY))
    assert client.collection.applied_tenants == []


def test_every_method_applies_the_tenant():
    client = FakeClient([stored("a")])
    target = reader(client)
    entry = make_entry(tenant="t1")
    target.count(entry)
    target.fetch(entry, [str(stored("a").uuid)])
    target.near_vector(entry, [0.1, 0.2], 1)
    target.filter_equal(entry, "year", 1, 1)
    list(target.source_ids(entry))
    assert client.collection.applied_tenants == ["t1"] * 5


def test_fetch_asks_for_at_most_100_uuids_per_request():
    objects = [stored(f"s{i}") for i in range(250)]
    client = FakeClient(objects)
    uuids = [str(o.uuid) for o in objects]
    found = reader(client).fetch(ENTRY, uuids)
    assert FETCH_BATCH == 100 and sorted(found) == sorted(uuids)
    calls = [kwargs for name, kwargs in client.collection.calls]
    assert [len(c["filters"].value) for c in calls] == [100, 100, 50]
    assert [c["limit"] for c in calls] == [100, 100, 50]
    assert all(c["include_vector"] is True for c in calls)


def test_fetch_compares_equal_to_the_written_object():
    obj = stored("a", year=2001)
    found = reader(FakeClient([obj])).fetch(ENTRY, [str(obj.uuid)])
    expected = TargetObject(
        str(obj.uuid),
        {"document": "text", "source_id": "a", "year": 2001},
        {"source": [0.25, 0.5]},
    )
    assert found == {str(obj.uuid): expected}
    written = to_object(Record("a", [0.25, 0.5], "text", None, {"year": 2001}), ENTRY)
    assert isinstance(written, TargetObject)
    assert (found[str(obj.uuid)].properties, found[str(obj.uuid)].vectors) == (
        written.properties,
        written.vectors,
    )


def test_fetch_of_no_uuids_sends_no_request():
    client = FakeClient([stored("a")])
    assert reader(client).fetch(ENTRY, []) == {}
    assert client.collection.calls == []


def test_near_vector_searches_the_kept_vector():
    client = FakeClient([stored("a"), stored("b"), stored("c")])
    assert reader(client).near_vector(ENTRY, [0.1, 0.2], 2) == ["a", "b"]
    name, kwargs = client.collection.calls[0]
    assert name == "near_vector" and kwargs["target_vector"] == "source"
    assert kwargs["near_vector"] == [0.1, 0.2] and kwargs["limit"] == 2
    assert kwargs["return_properties"] == ["source_id"]


def test_near_vector_needs_a_kept_vector():
    entry = make_entry()
    entry.target.vectors.pop(0)
    with pytest.raises(ValueError, match="no kept vector"):
        reader(FakeClient([])).near_vector(entry, [0.1], 1)


def test_filter_equal_returns_source_ids():
    client = FakeClient([stored("a", 2000), stored("b", 2001), stored("c", 2001)])
    assert reader(client).filter_equal(ENTRY, "year", 2001, 10) == ["b", "c"]
    _, kwargs = client.collection.calls[0]
    assert (kwargs["filters"].target, kwargs["filters"].value) == ("year", 2001)
    assert kwargs["limit"] == 10 and kwargs["return_properties"] == ["source_id"]


def test_source_ids_yields_every_pair_across_a_transient_error():
    objects = [stored("a"), stored(None), stored("c")]
    client = FakeClient(objects)
    client.collection.iterator_fail_at = 1
    pairs = list(reader(client).source_ids(ENTRY))
    assert pairs == [
        (str(objects[0].uuid), "a"),
        (str(objects[1].uuid), str(objects[1].uuid)),
        (str(objects[2].uuid), "c"),
    ]
    assert client.collection.calls == [
        ("iterator", {"return_properties": ["source_id"]})
    ]


def test_transient_errors_are_retried_and_others_raised():
    busy = UnexpectedStatusCodeError("x", httpx.Response(503))
    client = FakeClient([stored("a")], failures=[busy])
    assert reader(client).count(ENTRY) == 1
    assert len(client.collection.calls) == 2

    refused = UnexpectedStatusCodeError("x", httpx.Response(422))
    client = FakeClient([stored("a")], failures=[refused])
    with pytest.raises(UnexpectedStatusCodeError):
        reader(client).count(ENTRY)
    assert len(client.collection.calls) == 1


def test_exists_checks_the_collection_and_then_the_tenant():
    client = FakeClient([])
    client.collection.tenant_names = {"t1"}
    target = reader(client)
    assert target.exists(ENTRY) is True
    assert target.exists(make_entry(tenant="t1")) is True
    assert target.exists(make_entry(tenant="t2")) is False
    assert [kwargs for name, kwargs in client.collection.calls] == [
        {"tenant": "t1"},
        {"tenant": "t2"},
    ]
    client.existing = set()
    assert target.exists(make_entry(tenant="t1")) is False
    assert len(client.collection.calls) == 2
