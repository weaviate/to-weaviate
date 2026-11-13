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
    ProviderRefused,
    Reject,
    TargetObject,
    WeaviateWriter,
    chunks,
    provider_refusal,
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


def test_both_mode_keeps_the_vector_of_a_record_without_a_document():
    o = to_object(REC_NO_DOC, ENTRY_BOTH)
    assert isinstance(o, TargetObject) and o.vectors == {"source": [0.1, 0.2]}
    o = to_object(Record("d", [0.3], "", None, {}), ENTRY_BOTH)
    assert isinstance(o, TargetObject) and o.vectors == {"source": [0.3]}


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


# Per-object errors of Weaviate 1.40.4 during a rolling restart of a 3-node
# cluster: the coordinator could not reach the replica of the shard.
OUTAGE_ERRORS = [
    'resolve node name "weaviate-0" to host',
    'connect: Post "http://10.244.1.30:7001/indices/My_docs/shards/mXrkWx66gdpV/'
    'objects?schema_version=570": dial tcp 10.244.1.30:7001: connect: connection '
    "refused",
    'local index "My_docs" not found: deadline exceeded for waiting for update: '
    "version got=569 want=570",
]
CONTENT_ERROR = (
    "invalid integer property 'year' on class 'My_docs': requires an integer, "
    "the given value is '1.5'"
)


def partial(*failed: tuple[int, str]) -> Any:
    return SimpleNamespace(errors={i: SimpleNamespace(message=m) for i, m in failed})


@pytest.mark.parametrize("error", OUTAGE_ERRORS)
def test_objects_that_a_node_outage_refused_are_sent_again(error):
    data = FakeData([partial((1, error)), SimpleNamespace(errors={})])
    writer = WeaviateWriter(FakeClient(data), sleep=lambda s: None)  # type: ignore[arg-type]
    assert writer.write(ENTRY_KEEP, objects(3)) == []
    assert [[o.uuid for o in call] for call in data.calls] == [
        ["u0", "u1", "u2"],
        ["u1"],
    ]


def test_objects_that_a_node_outage_keeps_refusing_are_a_refused_batch():
    first = partial((0, OUTAGE_ERRORS[0]), (2, OUTAGE_ERRORS[1]))
    data = FakeData([first] + [all_failed(OUTAGE_ERRORS[1])] * 10)
    writer = WeaviateWriter(FakeClient(data), sleep=lambda s: None)  # type: ignore[arg-type]
    with pytest.raises(BatchRefused) as info:
        writer.write(ENTRY_KEEP, objects(3))
    assert str(info.value) == (
        "Weaviate refused 2 objects of a batch of 3 in My_docs: " + OUTAGE_ERRORS[1]
    )
    assert [[o.uuid for o in call] for call in data.calls] == [["u0", "u1", "u2"]] + [
        ["u0", "u2"]
    ] * 7


def test_objects_that_a_node_outage_refuses_on_every_attempt_are_a_refused_batch():
    data = FakeData(
        [partial((1, OUTAGE_ERRORS[2]))] + [partial((0, OUTAGE_ERRORS[2]))] * 10
    )
    writer = WeaviateWriter(FakeClient(data), sleep=lambda s: None)  # type: ignore[arg-type]
    with pytest.raises(BatchRefused) as info:
        writer.write(ENTRY_KEEP, objects(3))
    assert str(info.value) == (
        "Weaviate refused 1 object of a batch of 3 in My_docs: " + OUTAGE_ERRORS[2]
    )
    assert len(data.calls) == 7


def test_objects_refused_for_their_content_are_rejects_at_once():
    data = FakeData(
        [partial((1, CONTENT_ERROR), (2, OUTAGE_ERRORS[0])), SimpleNamespace(errors={})]
    )
    writer = WeaviateWriter(FakeClient(data), sleep=lambda s: None)  # type: ignore[arg-type]
    assert writer.write(ENTRY_KEEP, objects(3)) == [Reject("s1", CONTENT_ERROR)]
    assert [[o.uuid for o in call] for call in data.calls] == [
        ["u0", "u1", "u2"],
        ["u2"],
    ]


def test_a_single_object_that_a_node_outage_refuses_is_a_refused_batch():
    data = FakeData([all_failed(OUTAGE_ERRORS[0])] * 10)
    writer = WeaviateWriter(FakeClient(data), sleep=lambda s: None)  # type: ignore[arg-type]
    with pytest.raises(BatchRefused) as info:
        writer.write(ENTRY_KEEP, objects(1))
    assert str(info.value) == (
        "Weaviate refused every object of a batch of 1 in My_docs: " + OUTAGE_ERRORS[0]
    )
    assert len(data.calls) == 7


PROVIDER_401 = (
    "connection to: OpenAI API failed with status: 401 request-id: req-1 error: "
    "Incorrect API key provided: sk-made-***************3456. You can find your "
    "API key at https://platform.openai.com/account/api-keys."
)


@pytest.mark.parametrize(
    ("message", "provider", "status"),
    [
        (PROVIDER_401, "OpenAI API", 401),
        (
            "connection to Cohere API failed with status 403: forbidden",
            "Cohere API",
            403,
        ),
        ("connection to Google failed with status: 401 error: key", "Google", 401),
        (
            "connection to Contextual AI API request failed with status: 401",
            "Contextual AI API",
            401,
        ),
        (
            "connection to: https://api-inference.huggingface.co/models/m failed "
            "with status: 401 error: Authorization header is correct, but the "
            "token seems invalid",
            "api-inference.huggingface.co",
            401,
        ),
    ],
)
def test_a_provider_refusal_names_the_provider_and_the_status(
    message, provider, status
):
    refused = provider_refusal(message)
    assert refused is not None
    assert (refused.provider, refused.status) == (provider, status)
    assert "sk-made" not in str(refused) and "3456" not in str(refused)


@pytest.mark.parametrize(
    "message",
    [
        "connection to: OpenAI API failed with status: 400 error: too long",
        "connection to: OpenAI API failed with status: 429",
        "invalid integer property 'year'",
        "unauthorized: invalid api key",
    ],
)
def test_other_errors_are_not_a_provider_refusal(message):
    assert provider_refusal(message) is None


def test_a_provider_that_refuses_the_key_stops_the_batch_without_a_retry():
    data = FakeData([all_failed(PROVIDER_401)] * 10)
    writer = WeaviateWriter(FakeClient(data), sleep=lambda s: None)  # type: ignore[arg-type]
    with pytest.raises(ProviderRefused) as info:
        writer.write(ENTRY_EMBED, objects(3))
    assert (info.value.provider, info.value.status) == ("OpenAI API", 401)
    assert "sk-made" not in str(info.value)
    assert len(data.calls) == 1


def test_a_provider_that_refuses_the_key_of_a_single_object_stops_the_batch():
    data = FakeData([all_failed(PROVIDER_401)] * 10)
    writer = WeaviateWriter(FakeClient(data), sleep=lambda s: None)  # type: ignore[arg-type]
    with pytest.raises(ProviderRefused):
        writer.write(ENTRY_EMBED, objects(1))
    assert len(data.calls) == 1


def test_a_provider_that_refuses_the_key_of_one_object_stops_the_batch():
    data = FakeData([partial((0, CONTENT_ERROR), (1, PROVIDER_401))] * 10)
    writer = WeaviateWriter(FakeClient(data), sleep=lambda s: None)  # type: ignore[arg-type]
    with pytest.raises(ProviderRefused) as info:
        writer.write(ENTRY_EMBED, objects(3))
    assert (info.value.provider, info.value.status) == ("OpenAI API", 401)
    assert "sk-made" not in str(info.value)
    assert len(data.calls) == 1


def test_all_failed_batch_is_retried_and_then_raises():
    data = FakeData([all_failed("bad vector")] * 10)
    writer = WeaviateWriter(FakeClient(data), sleep=lambda s: None)  # type: ignore[arg-type]
    with pytest.raises(BatchRefused) as info:
        writer.write(ENTRY_KEEP, objects(2))
    assert str(info.value) == (
        "Weaviate refused every object of a batch of 2 in My_docs: bad vector"
    )
    assert len(data.calls) == 7


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
    assert len(data.calls) == 7


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


def test_one_object_that_grpc_refuses_for_its_size_is_a_reject():
    data = FakeData(["RESOURCE_EXHAUSTED", SimpleNamespace(errors={})])
    writer = WeaviateWriter(FakeClient(data), sleep=lambda s: None)  # type: ignore[arg-type]
    rejects = writer.write(ENTRY_KEEP, objects(1))
    assert rejects == [Reject("s0", "fake gRPC failure RESOURCE_EXHAUSTED")]
    assert len(data.calls) == 1


def test_a_chunk_of_two_that_grpc_refuses_for_its_size_is_retried():
    data = FakeData(["RESOURCE_EXHAUSTED"] * 10)
    writer = WeaviateWriter(FakeClient(data), sleep=lambda s: None)  # type: ignore[arg-type]
    with pytest.raises(WeaviateBatchError):
        writer.write(ENTRY_KEEP, objects(2))
    assert len(data.calls) == 7


def _int_entry(kind: str) -> p.CollectionPlan:
    entry = make_entry()
    entry.target.properties.append(
        p.PropertyPlan(source_key="int_edge", name="int_edge", type=kind)
    )
    return entry


def _int_record(value: Any) -> Record:
    return Record("edge-3", [0.1], "doc", None, {"int_edge": value, "year": 2020})


def _int_reject(value: int) -> str:
    return (
        f"int_edge: int {value} is above 2**53 in magnitude and Weaviate would "
        "store it rounded; set the property type to text, or text[] for a list"
    )


@pytest.mark.parametrize(
    ("kind", "value", "named"),
    [
        ("int", 2**53 + 1, 2**53 + 1),
        ("int", -(2**53) - 1, -(2**53) - 1),
        ("int", 2**63 - 1, 2**63 - 1),
        ("int", -(2**63), -(2**63)),
        ("number", 2**53 + 1, 2**53 + 1),
        ("number", 2**63 - 1, 2**63 - 1),
        ("int[]", [3, 2**63 - 1], 2**63 - 1),
        ("int[]", [-(2**63), 2**53 + 1], -(2**63)),
        ("number[]", [3, 2**53 + 1], 2**53 + 1),
    ],
)
def test_an_int_above_2_to_the_53_is_a_reject(kind, value, named):
    assert to_object(_int_record(value), _int_entry(kind)) == Reject(
        "edge-3", _int_reject(named)
    )


@pytest.mark.parametrize(
    ("kind", "value", "stored"),
    [
        ("int", 2**53, 2**53),
        ("int", -(2**53), -(2**53)),
        ("int[]", [-(2**53), 2**53], [-(2**53), 2**53]),
        ("number", 2**53, 2.0**53),
        ("number", 1e300, 1e300),
        ("text", 2**53 + 1, "9007199254740993"),
        ("text", 2**63 - 1, "9223372036854775807"),
        ("text[]", [3, -(2**63)], ["3", "-9223372036854775808"]),
    ],
)
def test_a_value_that_the_property_type_holds_exactly_is_stored(kind, value, stored):
    obj = to_object(_int_record(value), _int_entry(kind))
    assert isinstance(obj, TargetObject)
    assert obj.properties == {
        "source_id": "edge-3",
        "document": "doc",
        "year": "2020",
        "int_edge": stored,
    }
    assert type(obj.properties["int_edge"]) is type(stored)
