from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from fakes import all_failed, raise_grpc_failure
from weaviate.exceptions import (
    UnexpectedStatusCodeError,
    WeaviateBatchError,
    WeaviateInsertManyAllFailedError,
)
from weaviate.util import generate_uuid5

from to_weaviate import plan as p
from to_weaviate.sources.base import Record
from to_weaviate.target.writer import (
    BatchRefused,
    Reject,
    TargetObject,
    WeaviateWriter,
    chunks,
    to_object,
)

SPARSE = {"#type": "sparse_vector", "indices": [1], "values": [0.5]}


def make_entry(
    modes: tuple[str, ...] = ("keep",),
    dropped: list[p.DroppedKey] | None = None,
    tenant: str | None = None,
) -> p.CollectionPlan:
    vectors = []
    for mode in modes:
        name = (
            "default"
            if len(modes) == 1
            else ("source" if mode == "keep" else "default")
        )
        vectors.append(p.VectorPlan(name=name, mode=mode))  # type: ignore[arg-type]
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
            name="My_docs",
            tenant=tenant,
            vectors=vectors,
            properties=[
                p.PropertyPlan(source_key="#document", name="document", type="text"),
                p.PropertyPlan(source_key="#id", name="source_id", type="text"),
                p.PropertyPlan(source_key="#uri", name="uri", type="text"),
                p.PropertyPlan(source_key="user id", name="user_id", type="text"),
                p.PropertyPlan(source_key="year", name="year", type="text"),
            ],
            dropped=dropped
            if dropped is not None
            else [p.DroppedKey(source_key="sparse", reason="sparse")],
        ),
    )


ENTRY_YEAR_TEXT = make_entry()
ENTRY_KEEP = make_entry()
ENTRY_EMBED = make_entry(("embed",))
ENTRY_BOTH = make_entry(("keep", "embed"))
REC = Record("doc-1", [0.1, 0.2], "hello", None, {})
REC_NO_DOC = Record("doc-2", [0.1, 0.2], None, None, {})
REC_NO_VEC = Record("doc-3", None, "text", None, {})


def test_to_object_maps_and_coerces():
    o = to_object(
        Record(
            "doc-1",
            [0.1, 0.2],
            "hello",
            None,
            {"user id": "u1", "year": 2020, "sparse": SPARSE},
        ),
        ENTRY_YEAR_TEXT,
    )
    assert isinstance(o, TargetObject)
    assert o.uuid == generate_uuid5("doc-1") and o.vectors == {"default": [0.1, 0.2]}
    assert o.properties == {
        "source_id": "doc-1",
        "document": "hello",
        "user_id": "u1",
        "year": "2020",
    }


def test_to_object_reads_uri_and_skips_none():
    o = to_object(Record("a", [1.0], None, "file://x", {"year": None}), ENTRY_KEEP)
    assert isinstance(o, TargetObject)
    assert o.properties == {"source_id": "a", "uri": "file://x"}


def test_to_object_coercion_error_rejects():
    entry = make_entry()
    entry.target.properties[-1] = p.PropertyPlan(
        source_key="year", name="year", type="int"
    )
    r = to_object(Record("a", [1.0], "d", None, {"year": "x"}), entry)
    assert (
        isinstance(r, Reject) and r.source_id == "a" and r.reason.startswith("year: ")
    )


def test_to_object_unknown_metadata_key_rejects():
    r = to_object(Record("a", [1.0], "d", None, {"mystery": 1}), ENTRY_KEEP)
    assert r == Reject(
        "a",
        "metadata key mystery is not in plan.yaml; add a property for it or list "
        "it under dropped",
    )


def test_embed_mode_sends_no_vector():
    o = to_object(REC, ENTRY_EMBED)
    assert isinstance(o, TargetObject) and o.vectors == {}


def test_both_sends_only_the_kept_vector():
    o = to_object(REC, ENTRY_BOTH)
    assert isinstance(o, TargetObject) and o.vectors == {"source": [0.1, 0.2]}


def test_embed_mode_rejects_missing_document():
    assert to_object(REC_NO_DOC, ENTRY_EMBED) == Reject("doc-2", "no document to embed")
    assert to_object(Record("d", None, "", None, {}), ENTRY_EMBED) == Reject(
        "d", "no document to embed"
    )


def test_keep_mode_rejects_missing_vector():
    assert to_object(REC_NO_VEC, ENTRY_KEEP) == Reject("doc-3", "no vector")


def test_chunks_respect_size():
    objs = [
        TargetObject(str(i), {"document": "x" * 16384}, {"default": [0.5] * 4096})
        for i in range(300)
    ]
    out = list(chunks(objs))
    assert sum(len(c) for c in out) == 300 and len(out) > 1


def test_chunks_oversized_object_goes_alone():
    big = TargetObject("b", {"document": "x" * 500}, {})
    small = TargetObject("s", {"document": "x"}, {})
    assert list(chunks([small, big, small], max_bytes=200)) == [[small], [big], [small]]


def test_chunks_empty():
    assert list(chunks([])) == []


class FakeData:
    def __init__(self, outcomes: list[Any]) -> None:
        self.outcomes = outcomes
        self.calls: list[list[Any]] = []

    def insert_many(self, objects: list[Any]) -> Any:
        self.calls.append(objects)
        outcome = self.outcomes.pop(0) if self.outcomes else SimpleNamespace(errors={})
        if isinstance(outcome, str):
            raise_grpc_failure(outcome, WeaviateBatchError, batch=True)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class FakeCollection:
    def __init__(self, data: FakeData) -> None:
        self.data = data
        self.applied: list[tuple[str, Any]] = []

    def with_tenant(self, tenant: str) -> "FakeCollection":
        self.applied.append(("tenant", tenant))
        return self

    def with_consistency_level(self, level: Any) -> "FakeCollection":
        self.applied.append(("consistency", level))
        return self


class FakeClient:
    def __init__(self, data: FakeData) -> None:
        self.collection = FakeCollection(data)
        self.names: list[str] = []
        self.collections = SimpleNamespace(get=self._get)

    def _get(self, name: str) -> FakeCollection:
        self.names.append(name)
        return self.collection


def objects(n: int) -> list[TargetObject]:
    return [
        TargetObject(
            f"u{i}", {"source_id": f"s{i}", "document": "d"}, {"default": [0.1]}
        )
        for i in range(n)
    ]


def test_writer_inserts_and_maps_errors():
    data = FakeData([SimpleNamespace(errors={1: SimpleNamespace(message="boom")})])
    client = FakeClient(data)
    rejects = WeaviateWriter(client).write(ENTRY_KEEP, objects(3))  # type: ignore[arg-type]
    assert rejects == [Reject("s1", "boom")]
    assert client.names == ["My_docs"] and client.collection.applied == []
    sent = data.calls[0]
    assert [o.uuid for o in sent] == ["u0", "u1", "u2"]
    assert sent[0].properties["source_id"] == "s0" and sent[0].vector == {
        "default": [0.1]
    }


def test_writer_applies_tenant_and_consistency():
    client = FakeClient(FakeData([]))
    entry = make_entry(tenant="t1")
    WeaviateWriter(client, consistency="QUORUM").write(entry, objects(1))  # type: ignore[arg-type]
    kinds = dict(client.collection.applied)
    assert kinds["tenant"] == "t1" and kinds["consistency"].value == "QUORUM"


def test_all_failed_exception_becomes_rejects():
    data = FakeData([WeaviateInsertManyAllFailedError("nope")])
    rejects = WeaviateWriter(FakeClient(data)).write(ENTRY_KEEP, objects(1))  # type: ignore[arg-type]
    assert [r.source_id for r in rejects] == ["s0"]
    assert all("nope" in r.reason for r in rejects)
    assert len(data.calls) == 1


def test_all_failed_batch_is_retried_and_then_raises():
    data = FakeData([all_failed("bad vector")] * 10)
    writer = WeaviateWriter(FakeClient(data), sleep=lambda s: None)  # type: ignore[arg-type]
    with pytest.raises(BatchRefused) as info:
        writer.write(ENTRY_KEEP, objects(2))
    assert str(info.value) == (
        "Weaviate refused every object of a batch of 2 in My_docs: bad vector"
    )
    assert len(data.calls) == 6


def test_all_failed_batch_that_a_retry_writes_gives_no_rejects():
    data = FakeData([all_failed("busy"), SimpleNamespace(errors={})])
    writer = WeaviateWriter(FakeClient(data), sleep=lambda s: None)  # type: ignore[arg-type]
    assert writer.write(ENTRY_KEEP, objects(2)) == []
    assert len(data.calls) == 2


def test_writer_one_call_per_chunk(monkeypatch):
    import to_weaviate.target.writer as w

    monkeypatch.setattr(w, "MAX_INSERT_BYTES", 200)
    data = FakeData([])
    WeaviateWriter(FakeClient(data)).write(ENTRY_KEEP, objects(3))  # type: ignore[arg-type]
    assert [len(c) for c in data.calls] == [1, 1, 1]


def test_error_in_second_chunk_maps_to_the_right_object(monkeypatch):
    import to_weaviate.target.writer as w

    monkeypatch.setattr(w, "MAX_INSERT_BYTES", 200)
    failed = SimpleNamespace(errors={0: SimpleNamespace(message="boom")})
    data = FakeData([SimpleNamespace(errors={}), failed, SimpleNamespace(errors={})])
    rejects = WeaviateWriter(FakeClient(data)).write(ENTRY_KEEP, objects(3))  # type: ignore[arg-type]
    assert [[o.uuid for o in c] for c in data.calls] == [["u0"], ["u1"], ["u2"]]
    assert rejects == [Reject("s1", "boom")]


def test_exhausted_retries_raise():
    err = UnexpectedStatusCodeError("x", httpx.Response(503))
    data = FakeData([err] * 10)
    writer = WeaviateWriter(FakeClient(data), sleep=lambda s: None)  # type: ignore[arg-type]
    with pytest.raises(UnexpectedStatusCodeError) as info:
        writer.write(ENTRY_KEEP, objects(2))
    assert info.value is err
    assert len(data.calls) == 6


def test_writer_retries_transient_and_raises_others():
    err = UnexpectedStatusCodeError("x", httpx.Response(503))
    data = FakeData([err, SimpleNamespace(errors={})])
    writer = WeaviateWriter(FakeClient(data), sleep=lambda s: None)  # type: ignore[arg-type]
    assert writer.write(ENTRY_KEEP, objects(1)) == []
    assert len(data.calls) == 2

    bad = FakeData([ValueError("bad")])
    with pytest.raises(ValueError):
        WeaviateWriter(FakeClient(bad), sleep=lambda s: None).write(  # type: ignore[arg-type]
            ENTRY_KEEP, objects(1)
        )


def test_writer_retries_a_transient_grpc_failure():
    data = FakeData(["UNAVAILABLE", SimpleNamespace(errors={})])
    writer = WeaviateWriter(FakeClient(data), sleep=lambda s: None)  # type: ignore[arg-type]
    assert writer.write(ENTRY_KEEP, objects(1)) == []
    assert len(data.calls) == 2
