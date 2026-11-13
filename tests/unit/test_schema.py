import functools
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from fakes import FakeRpcError, raise_grpc_failure
from weaviate.classes.config import Configure, DataType, Tokenization, VectorDistances
from weaviate.collections.classes.config import (
    PQEncoderDistribution,
    PQEncoderType,
    _BQConfig,
    _PQConfig,
    _PQEncoderConfig,
    _RQConfig,
    _SQConfig,
    _VectorIndexConfigDynamic,
    _VectorIndexConfigFlat,
    _VectorIndexConfigHFresh,
    _VectorIndexConfigHNSW,
)
from weaviate.exceptions import (
    InsufficientPermissionsError,
    UnexpectedStatusCodeError,
    WeaviateBaseError,
    WeaviateBatchError,
    WeaviateConnectionError,
    WeaviateQueryError,
    WeaviateRetryError,
    WeaviateTenantGetError,
    WeaviateTimeoutError,
)

from to_weaviate import plan as p
from to_weaviate.settings import WeaviateSettings
from to_weaviate.target import connect as connect_module
from to_weaviate.target import schema
from to_weaviate.target.schema import (
    Created,
    PlanMismatch,
    PropertyRejected,
    SchemaRejected,
    WeaviateSchema,
    collection_kwargs,
    differences,
    ensure_collection,
    missing_properties,
    node_count,
    quantizer_updates,
    replication_factor,
)

PROPS = [
    p.PropertyPlan(
        source_key="#document",
        name="document",
        type="text",
        tokenization="word",
        searchable=True,
        filterable=False,
    ),
    p.PropertyPlan(
        source_key="#id", name="source_id", type="text", tokenization="field"
    ),
    p.PropertyPlan(source_key="year", name="year", type="int", range_filters=True),
]


def entry(
    name: str = "My_docs",
    tenant: str | None = None,
    mode: str = "keep",
    vectorizer: p.VectorizerPlan | None = None,
    distance: str | None = None,
    index: p.IndexPlan | None = None,
    properties: list[p.PropertyPlan] | None = None,
) -> p.CollectionPlan:
    return p.CollectionPlan(
        source=p.SourceRef(
            ref="r",
            namespace="ns",
            name="docs",
            count=1,
            dimension=2,
            distance=None,
            embedding=None,
        ),
        target=p.TargetPlan(
            name=name,
            tenant=tenant,
            vectors=[
                p.VectorPlan(
                    mode=mode,  # type: ignore[arg-type]
                    vectorizer=vectorizer,
                    distance=distance,
                    index=index or p.IndexPlan(),
                )
            ],
            properties=properties or PROPS,
        ),
    )


def vec(kwargs: dict[str, Any], i: int = 0) -> Any:
    return kwargs["vector_config"][i]


def test_kwargs_keep_without_vectorizer():
    k = collection_kwargs(
        [
            entry(
                mode="keep",
                vectorizer=None,
                distance="l2-squared",
                index=p.IndexPlan(type="hnsw"),
            )
        ]
    )
    assert k["name"] == "My_docs" and "multi_tenancy_config" not in k
    assert {x.name for x in k["properties"]} >= {"document", "source_id"}
    assert vec(k).vectorIndexConfig.distance == VectorDistances.L2_SQUARED


def test_kwargs_property_flags():
    props = {x.name: x for x in collection_kwargs([entry()])["properties"]}
    assert props["document"].dataType == DataType.TEXT
    assert props["document"].indexSearchable is True
    assert props["document"].indexFilterable is False
    assert props["source_id"].indexSearchable is False
    assert props["year"].dataType == DataType.INT
    assert props["year"].indexRangeFilters is True
    assert props["year"].tokenization is None and props["year"].indexSearchable is None


def test_kwargs_default_index_sends_no_index_config():
    v = vec(collection_kwargs([entry()]))
    assert v.vectorIndexConfig is None


def test_kwargs_rq_bits_reach_the_quantizer():
    index = p.IndexPlan(
        type="hnsw", quantizer=p.QuantizerPlan(type="rq", bits=8, rescore_limit=20)
    )
    q = vec(collection_kwargs([entry(index=index)])).vectorIndexConfig.quantizer
    assert q.bits == 8 and q.rescoreLimit == 20


def test_kwargs_index_options_are_passed():
    index = p.IndexPlan(type="hnsw", options={"ef_construction": 256})
    cfg = vec(collection_kwargs([entry(index=index)])).vectorIndexConfig
    assert cfg.efConstruction == 256


def test_kwargs_none_quantizer_disables_default_compression():
    index = p.IndexPlan(type="hnsw", quantizer=p.QuantizerPlan(type="none"))
    q = vec(collection_kwargs([entry(index=index)])).vectorIndexConfig.quantizer
    assert type(q).__name__ == "_UncompressedConfigCreate"


def test_kwargs_vectorizer_options():
    v = p.VectorizerPlan(module="text2vec-openai", options={"model": "m"})
    k = collection_kwargs([entry(mode="embed", vectorizer=v)])
    assert vec(k).vectorizer.vectorizer.value == "text2vec-openai"
    assert vec(k).properties == ["document"]
    assert vec(k).vectorizer.vectorizeClassName is False


# (options as plan.yaml holds them, as the client gets them). A text option
# is sent as the type that the factory declares for it.
SENT_OPTIONS = [
    ({"model": "007", "dimensions": "0256"}, {"model": "007", "dimensions": 256}),
    ({"dimensions": 256}, {"dimensions": 256}),
    ({"type_": "text"}, {"type_": "text"}),
]


@pytest.mark.parametrize(("planned", "sent"), SENT_OPTIONS)
def test_kwargs_send_a_text_option_as_the_declared_type(planned, sent):
    v = p.VectorizerPlan(module="text2vec-openai", options=planned)
    config = vec(collection_kwargs([entry(mode="embed", vectorizer=v)])).vectorizer
    assert {k: getattr(config, k) for k in sent} == sent


def test_kwargs_send_a_text_boolean_option_as_a_boolean():
    v = p.VectorizerPlan(
        module="text2vec-huggingface", options={"wait_for_model": "true"}
    )
    config = vec(collection_kwargs([entry(mode="embed", vectorizer=v)])).vectorizer
    assert config.waitForModel is True


def test_kwargs_unknown_vectorizer_module():
    v = p.VectorizerPlan(module="text2vec-nope")
    with pytest.raises(PlanMismatch, match="text2vec-nope is not supported"):
        collection_kwargs([entry(mode="embed", vectorizer=v)])


def test_kwargs_tenants_enable_multi_tenancy():
    k = collection_kwargs([entry(tenant="a"), entry(tenant="b")])
    assert k["multi_tenancy_config"].enabled is True
    assert k["multi_tenancy_config"].autoTenantCreation is None


def test_kwargs_group_unions_properties_in_first_seen_order():
    extra = p.PropertyPlan(source_key="k", name="k", type="text")
    k = collection_kwargs(
        [entry(tenant="a"), entry(tenant="b", properties=[*PROPS, extra])]
    )
    assert [x.name for x in k["properties"]] == ["document", "source_id", "year", "k"]


class FakeTenants:
    def __init__(self) -> None:
        self.created: list[list[str]] = []
        self.fail: list[Exception] = []

    def create(self, tenants: list[Any]) -> None:
        if self.fail:
            raise self.fail.pop(0)
        self.created.append([t.name for t in tenants])


class FakeCollection:
    def __init__(self, cfg: Any = None) -> None:
        self._cfg = cfg
        self.updates: list[Any] = []
        self.update_fail: list[Exception] = []
        self.added: list[Any] = []
        self.add_fail: list[Exception] = []
        self.config = SimpleNamespace(
            get=lambda: self._cfg, update=self._update, add_property=self._add
        )
        self.tenants = FakeTenants()

    def _add(self, prop: Any) -> None:
        if self.add_fail:
            raise self.add_fail.pop(0)
        self.added.append(prop)

    def _update(self, *, vector_config: Any) -> None:
        if self.update_fail:
            raise self.update_fail.pop(0)
        self.updates.append(vector_config)


class FakeCollections:
    def __init__(self, existing: FakeCollection | None) -> None:
        self.existing = existing
        self.created: list[dict[str, Any]] = []
        self.fail: list[Exception] = []
        # The config of a collection that `create` makes.
        self.created_cfg: Any = None

    def exists(self, name: str) -> bool:
        return self.existing is not None

    def get(self, name: str) -> FakeCollection:
        assert self.existing is not None
        return self.existing

    def create(self, **kwargs: Any) -> FakeCollection:
        if self.fail:
            raise self.fail.pop(0)
        self.created.append(kwargs)
        self.existing = FakeCollection(self.created_cfg)
        return self.existing


class FakeCluster:
    """`nodes` raises each error of `fail` in turn, then returns `count` nodes."""

    def __init__(self, count: int) -> None:
        self.count = count
        self.fail: list[Exception] = []
        self.calls = 0

    def nodes(self) -> list[Any]:
        self.calls += 1
        if self.fail:
            raise self.fail.pop(0)
        return [SimpleNamespace(name=f"node-{i}") for i in range(self.count)]


class FakeClient:
    def __init__(self, existing: FakeCollection | None, nodes: int = 1) -> None:
        self.collections = FakeCollections(existing)
        self.cluster = FakeCluster(nodes)


def hnsw(distance: Any = VectorDistances.COSINE, quantizer: Any = None) -> Any:
    return _VectorIndexConfigHNSW(
        multi_vector=None,
        quantizer=quantizer,
        cleanup_interval_seconds=0,
        distance_metric=distance,
        dynamic_ef_min=0,
        dynamic_ef_max=0,
        dynamic_ef_factor=0,
        ef=0,
        ef_construction=0,
        filter_strategy=None,  # type: ignore[arg-type]
        flat_search_cutoff=0,
        max_connections=0,
        skip=False,
        vector_cache_max_objects=0,
    )


def config(
    index: Any | None = None,
    module: str = "none",
    vector_name: str = "default",
    properties: dict[str, str] | None = None,
    tenancy: bool = False,
    factor: int = 1,
) -> Any:
    props = {"document": "text", "source_id": "text", "year": "int"}
    props.update(properties or {})
    return SimpleNamespace(
        vector_config={
            vector_name: SimpleNamespace(
                vectorizer=SimpleNamespace(vectorizer=SimpleNamespace(value=module)),
                vector_index_config=index or hnsw(),
            )
        },
        properties=[
            SimpleNamespace(name=n, data_type=SimpleNamespace(value=t))
            for n, t in props.items()
        ],
        multi_tenancy_config=SimpleNamespace(enabled=tenancy),
        replication_config=SimpleNamespace(factor=factor),
    )


def test_differences_none_when_missing():
    assert differences(FakeClient(None), [entry()]) is None  # type: ignore[arg-type]


def test_differences_empty_when_matching():
    client = FakeClient(FakeCollection(config()))
    assert (
        differences(client, [entry(distance="cosine", index=p.IndexPlan(type="hnsw"))])
        == []
    )  # type: ignore[arg-type]


def test_differences_extra_target_items_are_ignored():
    cfg = config(properties={"extra": "text"})
    cfg.vector_config["other"] = cfg.vector_config["default"]
    assert differences(FakeClient(FakeCollection(cfg)), [entry()]) == []  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "plan_kwargs, cfg, expected",
    [
        (
            {"distance": "dot", "index": p.IndexPlan(type="hnsw")},
            config(),
            "distance",
        ),
        (
            {
                "mode": "embed",
                "vectorizer": p.VectorizerPlan(module="text2vec-openai"),
            },
            config(),
            "text2vec-openai",
        ),
        ({}, config(module="text2vec-openai"), "vectorizer"),
        ({"index": p.IndexPlan(type="flat")}, config(), "index type"),
        (
            {"index": p.IndexPlan(type="hnsw", quantizer=p.QuantizerPlan(type="bq"))},
            config(),
            "quantizer type",
        ),
        (
            {
                "index": p.IndexPlan(
                    type="hnsw", quantizer=p.QuantizerPlan(type="rq", bits=8)
                )
            },
            config(
                index=hnsw(
                    quantizer=_RQConfig(
                        cache=None,
                        bits=1,
                        rescore_limit=0,
                        centering=None,
                        training_limit=0,
                    )
                )
            ),
            "bits",
        ),
        ({}, config(vector_name="other"), "default"),
        ({}, config(properties={"year": "text"}), "year"),
    ],
)
def test_differences_reports_mismatch(plan_kwargs, cfg, expected):
    found = differences(FakeClient(FakeCollection(cfg)), [entry(**plan_kwargs)])  # type: ignore[arg-type]
    assert found and expected in " ".join(found)


def test_differences_skip_a_property_type_without_a_mapping():
    props = [*PROPS[:2], p.PropertyPlan(source_key="year", name="year", type="date")]
    found = differences(FakeClient(FakeCollection(config())), [entry(properties=props)])  # type: ignore[arg-type]
    assert found == []


def test_a_missing_property_is_no_difference_and_is_listed_as_missing():
    cfg = config()
    cfg.properties = [x for x in cfg.properties if x.name != "year"]
    client = FakeClient(FakeCollection(cfg))
    assert differences(client, [entry()]) == []  # type: ignore[arg-type]
    assert missing_properties(client, [entry()]) == ["year"]  # type: ignore[arg-type]


def test_no_property_is_missing_from_a_collection_that_does_not_exist():
    assert missing_properties(FakeClient(None), [entry()]) == []  # type: ignore[arg-type]


def test_ensure_adds_a_missing_property_to_the_existing_collection():
    cfg = config()
    cfg.properties = [x for x in cfg.properties if x.name != "year"]
    existing = FakeCollection(cfg)
    existing.add_fail = [WeaviateConnectionError("down")]
    client = FakeClient(existing)
    result = ensure_collection(client, [entry()], sleep=lambda s: None)  # type: ignore[arg-type]
    assert result == "exists" and client.collections.created == []
    assert [(x.name, x.dataType.value) for x in existing.added] == [("year", "int")]


@pytest.mark.parametrize("status", [400, 422])
def test_a_refused_property_addition_is_schema_rejected(status):
    cfg = config()
    cfg.properties = [x for x in cfg.properties if x.name != "year"]
    existing = FakeCollection(cfg)
    body = {"error": [{"message": "property year already exists"}]}
    existing.add_fail = [
        UnexpectedStatusCodeError("refused", httpx.Response(status, json=body))
    ]
    with pytest.raises(PropertyRejected) as info:
        WeaviateSchema(FakeClient(existing)).ensure([entry()])  # type: ignore[arg-type]
    assert str(info.value) == "property year already exists"
    assert info.value.action == "add the property year to My_docs"


def stored_property(
    name: str,
    data_type: str,
    tokenization: Tokenization | None = None,
    searchable: bool = False,
    filterable: bool = True,
    range_filters: bool = False,
) -> Any:
    return SimpleNamespace(
        name=name,
        data_type=SimpleNamespace(value=data_type),
        tokenization=tokenization,
        index_searchable=searchable,
        index_filterable=filterable,
        index_range_filters=range_filters,
    )


def stored_props(**changed: Any) -> list[Any]:
    """The properties of a collection made from PROPS, with `changed` flags by name."""
    made = {
        "document": {
            "tokenization": Tokenization.WORD,
            "searchable": True,
            "filterable": False,
        },
        "source_id": {"tokenization": Tokenization.FIELD, "searchable": False},
        "year": {"range_filters": True},
    }
    types = {"document": "text", "source_id": "text", "year": "int"}
    return [
        stored_property(n, types[n], **{**flags, **changed.get(n, {})})
        for n, flags in made.items()
    ]


def test_differences_accept_the_property_flags_the_plan_sends():
    cfg = config()
    cfg.properties = stored_props()
    assert differences(FakeClient(FakeCollection(cfg)), [entry()]) == []  # type: ignore[arg-type]


# (flags of one stored property that differ from PROPS, expected difference)
PROPERTY_FLAG_MISMATCHES = {
    "tokenization": (
        {"document": {"tokenization": Tokenization.FIELD}},
        "property document: tokenization is field, plan has word",
    ),
    "searchable": (
        {"document": {"searchable": False}},
        "property document: searchable is False, plan has True",
    ),
    "filterable": (
        {"document": {"filterable": True}},
        "property document: filterable is True, plan has False",
    ),
    "range-filters": (
        {"year": {"range_filters": False}},
        "property year: range_filters is False, plan has True",
    ),
}


@pytest.mark.parametrize(
    ("changed", "expected"),
    PROPERTY_FLAG_MISMATCHES.values(),
    ids=PROPERTY_FLAG_MISMATCHES,
)
def test_differences_compare_each_property_flag_the_plan_sends(changed, expected):
    cfg = config()
    cfg.properties = stored_props(**changed)
    found = differences(FakeClient(FakeCollection(cfg)), [entry()])  # type: ignore[arg-type]
    assert found == [expected]


def test_differences_skip_the_property_flags_that_the_plan_does_not_send():
    # _property sends neither searchable for an int nor an unset tokenization.
    props = [
        p.PropertyPlan(source_key="k", name="k", type="text", searchable=True),
        p.PropertyPlan(source_key="n", name="n", type="int", searchable=True),
    ]
    cfg = config()
    cfg.properties = [
        stored_property("k", "text", tokenization=Tokenization.WORD, searchable=True),
        stored_property("n", "int", searchable=False),
    ]
    assert differences(FakeClient(FakeCollection(cfg)), [entry(properties=props)]) == []  # type: ignore[arg-type]


def test_differences_multi_tenancy():
    found = differences(FakeClient(FakeCollection(config())), [entry(tenant="a")])  # type: ignore[arg-type]
    assert found and "multi-tenancy" in found[0]
    found = differences(
        FakeClient(FakeCollection(config(tenancy=True))),
        [entry()],  # type: ignore[arg-type]
    )
    assert found and "multi-tenancy" in found[0]


def test_differences_none_quantizer_matches_uncompressed_target():
    index = p.IndexPlan(type="hnsw", quantizer=p.QuantizerPlan(type="none"))
    assert differences(FakeClient(FakeCollection(config())), [entry(index=index)]) == []  # type: ignore[arg-type]


# Weaviate 1.40.4 created an hfresh collection with rq and 1 bit for a plan
# with quantizer none. Such a plan is refused now; after the person sets rq or
# default, the copy resumes against that collection.
@pytest.mark.parametrize("quantizer", ["rq", "default"])
def test_differences_hfresh_rq_collection_matches_a_plan_changed_from_none(
    quantizer,
):
    found = _VectorIndexConfigHFresh(
        multi_vector=None,
        quantizer=_RQConfig(
            cache=None, bits=1, rescore_limit=350, centering=False, training_limit=0
        ),
        distance_metric=VectorDistances.COSINE,
        max_posting_size_kb=48,
        replicas=4,
        search_probe=256,
    )
    index = p.IndexPlan(type="hfresh", quantizer=p.QuantizerPlan(type=quantizer))
    client = FakeClient(FakeCollection(config(index=found)))
    assert differences(client, [entry(index=index)]) == []  # type: ignore[arg-type]


def test_differences_dynamic_reads_quantizer_from_both_parts():
    dyn = _VectorIndexConfigDynamic(
        distance_metric=VectorDistances.COSINE,
        hnsw=hnsw(quantizer=_BQConfig(cache=None, rescore_limit=0)),
        flat=SimpleNamespace(quantizer=_BQConfig(cache=None, rescore_limit=0)),  # type: ignore[arg-type]
        threshold=1,
    )
    index = p.IndexPlan(type="dynamic", quantizer=p.QuantizerPlan(type="bq"))
    assert (
        differences(FakeClient(FakeCollection(config(index=dyn))), [entry(index=index)])
        == []
    )  # type: ignore[arg-type]


def test_ensure_collection_creates_and_adds_tenants():
    client = FakeClient(None)
    result = ensure_collection(client, [entry(tenant="a"), entry(tenant="b")])  # type: ignore[arg-type]
    assert result == "created"
    assert client.collections.created[0]["name"] == "My_docs"
    assert client.collections.existing.tenants.created == [["a", "b"]]  # type: ignore[union-attr]


def test_ensure_collection_exists_adds_tenants():
    existing = FakeCollection(config(tenancy=True))
    result = ensure_collection(FakeClient(existing), [entry(tenant="a")])  # type: ignore[arg-type]
    assert result == "exists" and existing.tenants.created == [["a"]]


@pytest.mark.parametrize("status", [400, 422])
def test_refused_tenants_are_schema_rejected_and_name_the_tenants(status):
    existing = FakeCollection(config(tenancy=True))
    body = {"error": [{"message": "'a.b' is not a valid tenant name"}]}
    existing.tenants.fail = [
        UnexpectedStatusCodeError("refused", httpx.Response(status, json=body))
    ]
    with pytest.raises(SchemaRejected) as info:
        WeaviateSchema(FakeClient(existing)).ensure([entry(tenant="a.b")])  # type: ignore[arg-type]
    assert str(info.value) == "'a.b' is not a valid tenant name"
    assert info.value.action == "create the tenants of My_docs"
    assert existing.tenants.created == []


def test_ensure_collection_mismatch_raises():
    client = FakeClient(FakeCollection(config(properties={"year": "text"})))
    with pytest.raises(PlanMismatch, match="year"):
        ensure_collection(client, [entry()])  # type: ignore[arg-type]
    assert client.collections.created == []


def test_ensure_collection_retries_transient_creation():
    client = FakeClient(None)
    client.collections.fail = [WeaviateConnectionError("down")]
    assert ensure_collection(client, [entry()], sleep=lambda s: None) == "created"  # type: ignore[arg-type]


def test_ensure_collection_does_not_retry_other_errors():
    client = FakeClient(None)
    client.collections.fail = [ValueError("bad"), ValueError("again")]
    with pytest.raises(ValueError, match="bad"):
        ensure_collection(client, [entry()], sleep=lambda s: None)  # type: ignore[arg-type]
    assert len(client.collections.fail) == 1


def _status_error(status: int) -> UnexpectedStatusCodeError:
    return UnexpectedStatusCodeError("x", httpx.Response(status))


def _already_exists() -> UnexpectedStatusCodeError:
    body = {"error": [{"message": 'class name "My_docs" already exists'}]}
    return UnexpectedStatusCodeError("refused", httpx.Response(422, json=body))


def _applied_then_lost(client: FakeClient, cfg: Any, lost: Exception) -> None:
    """The first create stores a collection with `cfg` and raises `lost`.

    Every later create is refused with 422 "already exists".
    """
    calls: list[dict[str, Any]] = []

    def create(**kwargs: Any) -> FakeCollection:
        calls.append(kwargs)
        if len(calls) == 1:
            client.collections.existing = FakeCollection(cfg)
            raise lost
        raise _already_exists()

    client.collections.create = create  # type: ignore[method-assign]


@pytest.mark.parametrize(
    "lost", [WeaviateTimeoutError("read timeout"), WeaviateConnectionError("reset")]
)
def test_a_create_applied_before_its_answer_was_lost_counts_as_created(lost):
    client = FakeClient(None)
    _applied_then_lost(client, config(tenancy=True), lost)
    schema_ = WeaviateSchema(client, sleep=lambda s: None)  # type: ignore[arg-type]
    assert schema_.ensure([entry(tenant="a")]) == "created"
    assert client.collections.existing.tenants.created == [["a"]]  # type: ignore[union-attr]


def test_a_lost_create_that_left_another_collection_is_a_plan_mismatch():
    client = FakeClient(None)
    _applied_then_lost(
        client, config(properties={"year": "text"}), WeaviateTimeoutError("t")
    )
    with pytest.raises(PlanMismatch, match="property year: type is text"):
        WeaviateSchema(client, sleep=lambda s: None).ensure([entry()])  # type: ignore[arg-type]


def test_the_read_after_a_refused_retried_create_is_retried():
    client = FakeClient(None)
    _applied_then_lost(client, config(), WeaviateTimeoutError("read timeout"))
    exists = client.collections.exists
    fail: list[Exception] = [WeaviateConnectionError("reset")]

    def flaky_exists(name: str) -> bool:
        if client.collections.existing is not None and fail:
            raise fail.pop(0)
        return exists(name)

    client.collections.exists = flaky_exists  # type: ignore[method-assign]
    schema_ = WeaviateSchema(client, sleep=lambda s: None)  # type: ignore[arg-type]
    assert schema_.ensure([entry()]) == "created"
    assert fail == []


def test_a_recovered_collection_gets_the_planned_properties_it_lacks():
    client = FakeClient(None)
    cfg = config()
    cfg.properties = [x for x in cfg.properties if x.name != "year"]
    _applied_then_lost(client, cfg, WeaviateTimeoutError("read timeout"))
    schema_ = WeaviateSchema(client, sleep=lambda s: None)  # type: ignore[arg-type]
    assert schema_.ensure([entry()]) == "created"
    added = client.collections.existing.added  # type: ignore[union-attr]
    assert [(x.name, x.dataType.value) for x in added] == [("year", "int")]


def test_already_exists_on_the_first_create_attempt_is_schema_rejected():
    client = FakeClient(None)

    def create(**kwargs: Any) -> FakeCollection:
        client.collections.existing = FakeCollection(config())
        raise _already_exists()

    client.collections.create = create  # type: ignore[method-assign]
    with pytest.raises(SchemaRejected, match="already exists"):
        WeaviateSchema(client, sleep=lambda s: None).ensure([entry()])  # type: ignore[arg-type]


def test_a_create_refused_for_another_reason_after_a_lost_answer_is_schema_rejected():
    client = FakeClient(None)
    body = {"error": [{"message": "vector index dynamic needs async indexing"}]}
    # Under `replication_factor: auto` a refused create is sent once more
    # without a factor, so the server refuses two creates.
    client.collections.fail = [
        WeaviateTimeoutError("read timeout"),
        UnexpectedStatusCodeError("refused", httpx.Response(422, json=body)),
        UnexpectedStatusCodeError("refused", httpx.Response(422, json=body)),
    ]
    with pytest.raises(SchemaRejected) as info:
        WeaviateSchema(client, sleep=lambda s: None).ensure([entry()])  # type: ignore[arg-type]
    assert str(info.value) == "vector index dynamic needs async indexing"
    assert client.collections.existing is None


@pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
def test_transient_statuses(status):
    assert schema.transient(_status_error(status)) == 0.0


@pytest.mark.parametrize("status", [400, 401, 404, 422])
def test_non_transient_statuses(status):
    assert schema.transient(_status_error(status)) is None


def _grpc_error(status: str, error: Any, batch: bool = False) -> Exception:
    with pytest.raises(WeaviateBaseError) as info:
        raise_grpc_failure(status, error, batch=batch)
    return info.value


@pytest.mark.parametrize(
    ("status", "delay"),
    [
        ("DEADLINE_EXCEEDED", 0.0),
        ("RESOURCE_EXHAUSTED", 0.0),
        ("UNKNOWN", None),
        ("NOT_FOUND", None),
        ("INVALID_ARGUMENT", None),
    ],
)
def test_transient_grpc_statuses(status, delay):
    query = _grpc_error(status, lambda m: WeaviateQueryError(m, "GRPC search"))
    batch = _grpc_error(status, WeaviateBatchError, batch=True)
    assert isinstance(query.__context__, FakeRpcError)
    assert schema.transient(query) == schema.transient(batch) == delay


def test_transient_accepts_what_the_client_raises_after_its_unavailable_retries():
    query = _grpc_error("UNAVAILABLE", lambda m: WeaviateQueryError(m, "GRPC search"))
    batch = _grpc_error("UNAVAILABLE", WeaviateBatchError, batch=True)
    assert isinstance(query, WeaviateQueryError)
    assert isinstance(query.__context__, WeaviateRetryError)
    assert isinstance(batch, WeaviateRetryError)
    assert schema.transient(query) == schema.transient(batch) == 0.0


def test_transient_accepts_a_retry_error():
    assert schema.transient(WeaviateRetryError("fake", 3)) == 0.0


def test_transient_reads_the_cause_of_a_grpc_failure():
    error = WeaviateTenantGetError("fake")
    error.__cause__ = FakeRpcError("DEADLINE_EXCEEDED")
    assert schema.transient(error) == 0.0


def _chain(depth: int) -> Exception:
    """An error with an UNAVAILABLE FakeRpcError `depth` links below it.

    The links alternate between `__context__` and `__cause__`.
    """
    error: Exception = FakeRpcError("UNAVAILABLE")
    for level in range(depth):
        outer = ValueError(f"link {level}")
        if level % 2:
            outer.__cause__ = error
        else:
            outer.__context__ = error
        error = outer
    return error


def test_transient_reads_the_chain_up_to_5_levels():
    assert schema.transient(_chain(5)) == 0.0
    assert schema.transient(_chain(6)) is None


@pytest.mark.parametrize("status", ["INVALID_ARGUMENT", "PERMISSION_DENIED"])
def test_transient_judges_the_nearest_grpc_error_only(status):
    # The client's retry makes each attempt inside the except block of the
    # attempt before, so an earlier UNAVAILABLE is the context of the last one.
    with pytest.raises(WeaviateBatchError) as info:
        try:
            try:
                raise FakeRpcError("UNAVAILABLE")
            except FakeRpcError:
                raise FakeRpcError(status)  # noqa: B904
        except FakeRpcError:
            raise WeaviateBatchError("fake")  # noqa: B904
    earlier = info.value.__context__.__context__
    assert isinstance(earlier, FakeRpcError) and earlier.status == "UNAVAILABLE"
    assert schema.transient(info.value) is None


def test_transient_stops_on_a_cyclic_chain_and_objects_without_code():
    first = ValueError("first")
    second = KeyError("second")
    first.__context__ = second
    second.__context__ = first
    assert schema.transient(first) is None
    no_call = ValueError("code is a number")
    no_call.code = 7  # type: ignore[attr-defined]
    assert schema.transient(no_call) is None


def test_query_error_without_a_grpc_error_is_not_transient():
    error = WeaviateQueryError("could not find class Probe_Missing", "GRPC search")
    assert schema.transient(error) is None


def test_connect_with_grpc_url_uses_custom(monkeypatch):
    seen: dict[str, Any] = {}
    monkeypatch.setattr(
        connect_module.weaviate,
        "connect_to_custom",
        lambda **kw: seen.update(kw) or "client",
    )
    s = WeaviateSettings(
        url="http://localhost:8099",
        api_key="k",
        grpc_url="https://grpc.example.com",
        headers={"X-OpenAI-Api-Key": "o"},
    )
    assert connect_module.connect(s) == "client"
    assert seen["http_host"] == "localhost" and seen["http_port"] == 8099
    assert seen["http_secure"] is False
    assert seen["grpc_host"] == "grpc.example.com" and seen["grpc_port"] == 443
    assert seen["grpc_secure"] is True
    assert seen["headers"] == {"X-OpenAI-Api-Key": "o"}
    assert seen["auth_credentials"] is not None


def test_connect_default_ports(monkeypatch):
    seen: dict[str, Any] = {}
    monkeypatch.setattr(
        connect_module.weaviate, "connect_to_custom", lambda **kw: seen.update(kw)
    )
    s = WeaviateSettings(url="http://h", api_key=None, grpc_url="h:50051", headers={})
    connect_module.connect(s)
    assert seen["http_port"] == 80 and seen["grpc_port"] == 50051
    assert seen["headers"] is None and seen["auth_credentials"] is None


def test_connect_cloud(monkeypatch):
    seen: dict[str, Any] = {}
    monkeypatch.setattr(
        connect_module.weaviate,
        "connect_to_weaviate_cloud",
        lambda **kw: seen.update(kw) or "client",
    )
    s = WeaviateSettings(
        url="https://x.weaviate.cloud", api_key="k", grpc_url=None, headers={}
    )
    assert connect_module.connect(s) == "client"
    assert seen["cluster_url"] == "https://x.weaviate.cloud"
    assert seen["auth_credentials"] is not None and seen["headers"] is None


HAND_WRITTEN_PLAN = """\
version: 1
source:
  type: chroma
  url: http://localhost:8001
  tenant: null
collections:
- source:
    ref: r
    namespace: ns
    name: docs
    count: 1
    dimension: 2
    distance: cosine
    embedding: null
  target:
    name: Docs
    vectors:
    - name: default
      mode: embed
      vectorizer:
        module: text2vec-openai
        options:
          model: text-embedding-3-small
          base_url: "https://bob:hunter2x@proxy.example.com/v1?token=abc#frag"
    properties: []
"""


def test_kwargs_send_a_url_option_of_a_hand_written_plan_without_secrets(tmp_path):
    path = tmp_path / "plan.yaml"
    path.write_text(HAND_WRITTEN_PLAN)
    k = collection_kwargs(p.load_plan(path).collections)
    vectorizer = vec(k).vectorizer
    assert str(vectorizer.baseURL) == "https://proxy.example.com/v1"
    assert vectorizer.model == "text-embedding-3-small"
    sent = repr(k)
    for sentinel in ("bob", "hunter2x", "token", "abc", "frag"):
        assert sentinel not in sent


@pytest.mark.parametrize(
    ("module", "options", "expected"),
    [
        (
            "text2vec-openai",
            {
                "base_url": "https://bob:made-up-a@proxy.example.com/v1?k=made-up-b#c",
                "endpoint": "https://bob:made-up-d@embed.example.com/e?k=made-up-e#f",
            },
            {
                "baseURL": "https://proxy.example.com/v1",
                "endpoint": "https://embed.example.com/e",
            },
        ),
        (
            "text2vec-transformers",
            {
                "passage_inference_url": "http://bob:made-up-a@passage:8080?k=made-up-b",
                "query_inference_url": "http://bob:made-up-d@query:8080#made-up-e",
            },
            {
                "passageInferenceUrl": "http://passage:8080",
                "queryInferenceUrl": "http://query:8080",
            },
        ),
        (
            "text2vec-ollama",
            {"api_endpoint": "http://bob:made-up@ollama:11434?k=made-up#f"},
            {"apiEndpoint": "http://ollama:11434"},
        ),
    ],
)
def test_kwargs_send_every_url_option_without_userinfo_query_or_fragment(
    module: str, options: dict[str, Any], expected: dict[str, str]
) -> None:
    v = p.VectorizerPlan(module=module, options=options)
    k = collection_kwargs([entry(mode="embed", vectorizer=v)])
    found = vec(k).vectorizer
    assert {key: str(getattr(found, key)) for key in expected} == expected
    assert "made-up" not in repr(k) and "bob" not in repr(k)


@pytest.mark.parametrize("char", "/?#")
def test_kwargs_refuse_a_url_option_that_keeps_its_userinfo(char: str) -> None:
    options = {
        "model": "text-embedding-3-small",
        "base_url": f"https://bob:made{char}up-pw@proxy.example.com/v1",
    }
    v = p.VectorizerPlan(module="text2vec-openai", options=options)
    with pytest.raises(PlanMismatch) as e:
        collection_kwargs([entry(mode="embed", vectorizer=v)])
    assert str(e.value) == (
        "vectorizer option base_url of text2vec-openai in My_docs holds userinfo "
        "that cannot be removed"
    )


@pytest.mark.parametrize("model", ["voyage-3@2025-01", "bob:x@host", "m?v=1#tag"])
def test_kwargs_send_options_without_a_scheme_unchanged(model: str) -> None:
    options = {"model": model, "dimensions": 256, "truncate": True}
    v = p.VectorizerPlan(module="text2vec-voyageai", options=options)
    found = vec(collection_kwargs([entry(mode="embed", vectorizer=v)])).vectorizer
    assert (found.model, found.dimensions, found.truncate) == (model, 256, True)
    assert found.truncate is True


def test_kwargs_pass_a_list_option_to_the_factory_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sent: dict[str, Any] = {}

    # No text2vec factory takes a list, so this one records its arguments.
    # `wraps` keeps the signature that `_vector` checks the options against.
    @functools.wraps(Configure.Vectors.text2vec_voyageai)
    def record(**kwargs: Any) -> None:
        sent.update(kwargs)

    monkeypatch.setattr(Configure.Vectors, "text2vec_voyageai", record)
    value = ["bob:x@host", "m?v=1#tag"]
    v = p.VectorizerPlan(module="text2vec-voyageai", options={"model": value})
    collection_kwargs([entry(mode="embed", vectorizer=v)])
    assert sent["model"] is value
    assert value == ["bob:x@host", "m?v=1#tag"]


def test_kwargs_google_with_project_id_uses_vertex():
    v = p.VectorizerPlan(module="text2vec-google", options={"project_id": "proj"})
    k = collection_kwargs([entry(mode="embed", vectorizer=v)])
    assert vec(k).vectorizer.projectId == "proj"


def test_kwargs_google_without_project_id_uses_studio():
    v = p.VectorizerPlan(module="text2vec-google")
    k = collection_kwargs([entry(mode="embed", vectorizer=v)])
    assert vec(k).vectorizer.projectId is None
    assert vec(k).vectorizer.apiEndpoint == "generativelanguage.googleapis.com"


@pytest.mark.parametrize("server_module", ["text2vec-google", "text2vec-palm"])
def test_differences_accepts_both_google_names(server_module):
    v = p.VectorizerPlan(module="text2vec-google")
    cfg = config(module=server_module)
    found = differences(
        FakeClient(FakeCollection(cfg)),  # type: ignore[arg-type]
        [entry(mode="embed", vectorizer=v)],
    )
    assert found == []


@pytest.mark.parametrize(
    "key",
    [
        "colour",
        "name",
        "source_properties",
        "vectorize_collection_name",
        "quantizer",
        "vector_index_config",
    ],
)
def test_kwargs_refuses_a_vectorizer_option_key(key):
    v = p.VectorizerPlan(module="text2vec-openai", options={key: "x"})
    message = f"vectorizer option {key} is not accepted by text2vec-openai"
    with pytest.raises(PlanMismatch, match=message):
        collection_kwargs([entry(mode="embed", vectorizer=v)])


# The client warns on every text2vec_aws call.
@pytest.mark.filterwarnings("ignore:`text2vec_aws` is deprecated:DeprecationWarning")
@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({}, "vectorizer text2vec-aws needs the options model, region"),
        ({"model": "m"}, "vectorizer text2vec-aws needs the option region"),
    ],
)
def test_kwargs_name_the_missing_options_of_a_vectorizer(options, message):
    v = p.VectorizerPlan(module="text2vec-aws", options=options)
    with pytest.raises(PlanMismatch) as e:
        collection_kwargs([entry(mode="embed", vectorizer=v)])
    assert str(e.value) == message


def test_kwargs_report_another_type_error_of_a_vectorizer_factory(monkeypatch):
    def factory(**kwargs: Any) -> Any:
        raise TypeError("unsupported operand")

    monkeypatch.setattr(Configure.Vectors, "text2vec_openai", factory)
    v = p.VectorizerPlan(module="text2vec-openai")
    with pytest.raises(PlanMismatch) as e:
        collection_kwargs([entry(mode="embed", vectorizer=v)])
    assert str(e.value) == (
        "vectorizer text2vec-openai refused its options: unsupported operand"
    )


def test_kwargs_refuses_a_vectorizer_option_value():
    v = p.VectorizerPlan(module="text2vec-openai", options={"dimensions": "many"})
    with pytest.raises(PlanMismatch, match="dimensions"):
        collection_kwargs([entry(mode="embed", vectorizer=v)])


class MetaClient(FakeClient):
    def get_meta(self) -> dict[str, Any]:
        return {
            "version": "1.40.4",
            "modules": {"text2vec-openai": {}, "backup-s3": {}},
        }


def test_weaviate_schema_modules():
    schema_ = WeaviateSchema(MetaClient(None))  # type: ignore[arg-type]
    assert schema_.modules() == {"text2vec-openai", "backup-s3"}


def test_weaviate_schema_differences_and_ensure_use_the_client():
    client = FakeClient(None)
    schema_ = WeaviateSchema(client)  # type: ignore[arg-type]
    assert schema_.differences([entry()]) is None
    assert schema_.ensure([entry()]) == "created"
    assert client.collections.created[0]["name"] == "My_docs"


@pytest.mark.parametrize("status", [400, 422])
def test_weaviate_schema_turns_a_refused_creation_into_schema_rejected(status):
    client = FakeClient(None)
    body = {"error": [{"message": "vector index dynamic needs async indexing"}]}
    # Under `replication_factor: auto` a refused create is sent once more
    # without a factor, so the server refuses two creates.
    client.collections.fail = [
        UnexpectedStatusCodeError(
            "Collection may not have been created properly.",
            httpx.Response(status, json=body),
        )
        for _ in range(2)
    ]
    with pytest.raises(SchemaRejected) as info:
        WeaviateSchema(client).ensure([entry()])  # type: ignore[arg-type]
    assert str(info.value) == "vector index dynamic needs async indexing"
    assert client.collections.fail == []


def test_weaviate_schema_keeps_other_status_errors():
    client = FakeClient(None)
    client.collections.fail = [_status_error(401)]
    with pytest.raises(UnexpectedStatusCodeError):
        WeaviateSchema(client).ensure([entry()])  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "url",
    [
        "https://TOKEN@X.Weaviate.Cloud:443",
        "http://admin:pw@x.weaviate.cloud/",
        "x.weaviate.cloud",
    ],
)
def test_connect_cloud_passes_the_host_without_userinfo_or_port(monkeypatch, url):
    seen: dict[str, Any] = {}
    monkeypatch.setattr(
        connect_module.weaviate,
        "connect_to_weaviate_cloud",
        lambda **kw: seen.update(kw),
    )
    connect_module.connect(WeaviateSettings(url, None, None, {}))
    assert seen["cluster_url"] == "https://x.weaviate.cloud"


def test_connect_cloud_keeps_a_host_that_starts_with_http(monkeypatch):
    seen: dict[str, Any] = {}
    monkeypatch.setattr(
        connect_module.weaviate,
        "connect_to_weaviate_cloud",
        lambda **kw: seen.update(kw),
    )
    connect_module.connect(WeaviateSettings("httpx1.weaviate.cloud", None, None, {}))
    assert seen["cluster_url"] == "https://httpx1.weaviate.cloud"


def test_connect_cloud_without_api_key(monkeypatch):
    seen: dict[str, Any] = {}
    monkeypatch.setattr(
        connect_module.weaviate,
        "connect_to_weaviate_cloud",
        lambda **kw: seen.update(kw),
    )
    s = WeaviateSettings(
        url="https://x.weaviate.cloud", api_key=None, grpc_url=None, headers={}
    )
    connect_module.connect(s)
    assert seen["auth_credentials"] is None


def _trained_entry() -> p.CollectionPlan:
    """An entry with a vector whose sq needs training and one with plain rq."""
    plain = entry()
    vectors = [
        p.VectorPlan(
            name="a",
            index=p.IndexPlan(
                type="hnsw",
                quantizer=p.QuantizerPlan(
                    type="sq", training_limit=100, rescore_limit=20
                ),
            ),
        ),
        p.VectorPlan(
            name="b",
            mode="embed",
            vectorizer=p.VectorizerPlan(module="text2vec-weaviate"),
            index=p.IndexPlan(
                type="hnsw", quantizer=p.QuantizerPlan(type="rq", bits=8)
            ),
        ),
    ]
    return plain.model_copy(
        update={"target": plain.target.model_copy(update={"vectors": vectors})}
    )


@pytest.mark.parametrize(
    ("quantizer", "fields"),
    [
        (
            p.QuantizerPlan(type="sq", training_limit=100, rescore_limit=20),
            {"enabled": True, "trainingLimit": 100, "rescoreLimit": 20},
        ),
        (
            p.QuantizerPlan(type="pq", training_limit=100, segments=8, centroids=16),
            {"enabled": True, "trainingLimit": 100, "segments": 8, "centroids": 16},
        ),
        (
            p.QuantizerPlan(type="rq", bits=4, centering=True, training_limit=100),
            {"enabled": True, "bits": 4, "centering": True, "trainingLimit": 100},
        ),
    ],
)
def test_quantizer_updates_send_the_planned_quantizer_again(quantizer, fields):
    index = p.IndexPlan(type="hnsw", quantizer=quantizer)
    (update,) = quantizer_updates(entry(index=index))
    assert update.name == "default"
    sent = update.vectorIndexConfig.quantizer.model_dump(exclude_none=True)
    assert sent == fields


@pytest.mark.parametrize(
    "quantizer",
    [
        p.QuantizerPlan(type="rq", bits=8),
        p.QuantizerPlan(type="bq"),
        p.QuantizerPlan(type="none"),
        p.QuantizerPlan(),
    ],
)
def test_quantizer_updates_skip_a_quantizer_without_training(quantizer):
    index = p.IndexPlan(type="hnsw", quantizer=quantizer)
    assert quantizer_updates(entry(index=index)) == []


def test_quantizer_updates_name_only_the_vectors_that_need_training():
    assert [u.name for u in quantizer_updates(_trained_entry())] == ["a"]


def test_weaviate_schema_sends_the_quantizer_updates_to_the_collection():
    collection = FakeCollection()
    WeaviateSchema(FakeClient(collection)).enable_trained_quantizers(  # type: ignore[arg-type]
        _trained_entry()
    )
    assert [[u.name for u in sent] for sent in collection.updates] == [["a"]]


def test_weaviate_schema_sends_no_update_without_a_trained_quantizer():
    collection = FakeCollection()
    WeaviateSchema(FakeClient(collection)).enable_trained_quantizers(entry())  # type: ignore[arg-type]
    assert collection.updates == []


@pytest.mark.parametrize("status", [400, 422])
def test_weaviate_schema_turns_a_refused_update_into_schema_rejected(status):
    collection = FakeCollection()
    body = {"error": [{"message": "sq is not supported"}]}
    collection.update_fail = [
        UnexpectedStatusCodeError(
            "Collection configuration may not have been updated.",
            httpx.Response(status, json=body),
        )
    ]
    with pytest.raises(SchemaRejected) as info:
        WeaviateSchema(FakeClient(collection)).enable_trained_quantizers(  # type: ignore[arg-type]
            _trained_entry()
        )
    assert str(info.value) == "sq is not supported"
    assert info.value.action == "enable the quantizers of My_docs"


def test_weaviate_schema_retries_a_transient_update_failure():
    collection = FakeCollection()
    collection.update_fail = [_status_error(503)]
    client = FakeClient(collection)
    WeaviateSchema(client, sleep=lambda _: None).enable_trained_quantizers(  # type: ignore[arg-type]
        _trained_entry()
    )
    assert len(collection.updates) == 1


def _rq(**fields: Any) -> _RQConfig:
    values = {
        "cache": None,
        "bits": 4,
        "rescore_limit": 20,
        "centering": False,
        "training_limit": 10000,
    }
    return _RQConfig(**{**values, **fields})


def _pq(**fields: Any) -> _PQConfig:
    encoder = _PQEncoderConfig(
        type_=PQEncoderType.KMEANS, distribution=PQEncoderDistribution.LOG_NORMAL
    )
    values = {
        "internal_bit_compression": False,
        "segments": 8,
        "centroids": 16,
        "training_limit": 100,
        "encoder": encoder,
    }
    return _PQConfig(**{**values, **fields})


def _sq(**fields: Any) -> _SQConfig:
    return _SQConfig(**{"rescore_limit": 25, "training_limit": 100, **fields})


# (planned quantizer, quantizer the server reports, expected difference)
QUANTIZER_FIELD_MISMATCHES = {
    "rq-centering": (
        p.QuantizerPlan(type="rq", bits=4, centering=True),
        _rq(centering=False),
        "vector default: centering is False, plan has True",
    ),
    "rq-rescore-limit": (
        p.QuantizerPlan(type="rq", bits=4, rescore_limit=40),
        _rq(),
        "vector default: rescore_limit is 20, plan has 40",
    ),
    "rq-training-limit": (
        p.QuantizerPlan(type="rq", training_limit=100),
        _rq(),
        "vector default: training_limit is 10000, plan has 100",
    ),
    "pq-segments": (
        p.QuantizerPlan(type="pq", segments=4),
        _pq(),
        "vector default: segments is 8, plan has 4",
    ),
    "pq-centroids": (
        p.QuantizerPlan(type="pq", centroids=256),
        _pq(),
        "vector default: centroids is 16, plan has 256",
    ),
    "pq-training-limit": (
        p.QuantizerPlan(type="pq", training_limit=1000),
        _pq(),
        "vector default: training_limit is 100, plan has 1000",
    ),
    "sq-rescore-limit": (
        p.QuantizerPlan(type="sq", rescore_limit=20),
        _sq(),
        "vector default: rescore_limit is 25, plan has 20",
    ),
    "sq-training-limit": (
        p.QuantizerPlan(type="sq", training_limit=200),
        _sq(),
        "vector default: training_limit is 100, plan has 200",
    ),
}


@pytest.mark.parametrize(
    ("planned", "stored", "expected"),
    QUANTIZER_FIELD_MISMATCHES.values(),
    ids=QUANTIZER_FIELD_MISMATCHES,
)
def test_differences_compare_each_planned_quantizer_field(planned, stored, expected):
    index = p.IndexPlan(type="hnsw", quantizer=planned)
    cfg = config(index=hnsw(quantizer=stored))
    found = differences(FakeClient(FakeCollection(cfg)), [entry(index=index)])  # type: ignore[arg-type]
    assert found == [expected]


@pytest.mark.parametrize(
    ("planned", "stored"),
    [
        (p.QuantizerPlan(type="rq", bits=4), _rq(centering=True, rescore_limit=7)),
        (p.QuantizerPlan(type="pq"), _pq(segments=4, centroids=256)),
        (p.QuantizerPlan(type="sq"), _sq(rescore_limit=1, training_limit=9)),
    ],
)
def test_differences_skip_a_quantizer_field_that_the_plan_does_not_set(planned, stored):
    index = p.IndexPlan(type="hnsw", quantizer=planned)
    cfg = config(index=hnsw(quantizer=stored))
    assert differences(FakeClient(FakeCollection(cfg)), [entry(index=index)]) == []  # type: ignore[arg-type]


def test_differences_skip_a_field_that_the_index_does_not_store():
    # A flat index stores rq without centering and without a training limit.
    stored = _rq(bits=8, centering=None, training_limit=None, rescore_limit=40)
    flat = _VectorIndexConfigFlat(
        multi_vector=None,
        quantizer=stored,
        distance_metric=VectorDistances.COSINE,
        vector_cache_max_objects=0,
    )
    quantizer = p.QuantizerPlan(type="rq", bits=8, centering=False, rescore_limit=40)
    index = p.IndexPlan(type="flat", quantizer=quantizer)
    cfg = config(index=flat)
    assert differences(FakeClient(FakeCollection(cfg)), [entry(index=index)]) == []  # type: ignore[arg-type]


def _flat_with(quantizer: Any) -> _VectorIndexConfigFlat:
    return _VectorIndexConfigFlat(
        multi_vector=None,
        quantizer=quantizer,
        distance_metric=VectorDistances.COSINE,
        vector_cache_max_objects=0,
    )


@pytest.mark.parametrize(
    ("stored", "planned"),
    [
        (
            _BQConfig(cache=False, rescore_limit=-1),
            p.QuantizerPlan(type="bq", cache=True),
        ),
        (
            _rq(bits=8, cache=False, centering=None, training_limit=None),
            p.QuantizerPlan(type="rq", bits=8, cache=True),
        ),
    ],
)
def test_differences_compare_the_quantizer_cache(stored, planned):
    index = p.IndexPlan(type="flat", quantizer=planned)
    cfg = config(index=_flat_with(stored))
    assert differences(FakeClient(FakeCollection(cfg)), [entry(index=index)]) == [  # type: ignore[arg-type]
        "vector default: cache is False, plan has True"
    ]


@pytest.mark.parametrize(
    ("stored", "planned"),
    [
        (
            _BQConfig(cache=True, rescore_limit=-1),
            p.QuantizerPlan(type="bq", cache=True),
        ),
        (
            _rq(bits=8, cache=True, centering=None, training_limit=None),
            p.QuantizerPlan(type="rq", bits=8, cache=True),
        ),
    ],
)
def test_differences_accept_an_equal_quantizer_cache(stored, planned):
    index = p.IndexPlan(type="flat", quantizer=planned)
    cfg = config(index=_flat_with(stored))
    assert differences(FakeClient(FakeCollection(cfg)), [entry(index=index)]) == []  # type: ignore[arg-type]


def test_differences_compare_the_hnsw_part_of_a_dynamic_index():
    flat = SimpleNamespace(quantizer=_rq(bits=8, centering=None, training_limit=None))
    dyn = _VectorIndexConfigDynamic(
        distance_metric=VectorDistances.COSINE,
        hnsw=hnsw(quantizer=_rq(bits=8, training_limit=500)),
        flat=flat,  # type: ignore[arg-type]
        threshold=1,
    )
    quantizer = p.QuantizerPlan(type="rq", bits=8, training_limit=100)
    index = p.IndexPlan(type="dynamic", quantizer=quantizer)
    cfg = config(index=dyn)
    assert differences(FakeClient(FakeCollection(cfg)), [entry(index=index)]) == [
        "vector default: training_limit is 500, plan has 100"
    ]  # type: ignore[arg-type]


def _hfresh() -> _VectorIndexConfigHFresh:
    return _VectorIndexConfigHFresh(
        multi_vector=None,
        quantizer=_rq(bits=1, centering=None),
        distance_metric=VectorDistances.COSINE,
        max_posting_size_kb=48,
        replicas=4,
        search_probe=256,
    )


def _dynamic() -> _VectorIndexConfigDynamic:
    return _VectorIndexConfigDynamic(
        distance_metric=VectorDistances.COSINE,
        hnsw=hnsw(),
        flat=_flat_with(None),
        threshold=10000,
    )


# (planned index, index the server reports, expected difference)
INDEX_OPTION_MISMATCHES = {
    "hnsw-ef-construction": (
        p.IndexPlan(type="hnsw", options={"ef_construction": 256}),
        hnsw(),
        "vector default: ef_construction is 0, plan has 256",
    ),
    "flat-vector-cache": (
        p.IndexPlan(type="flat", options={"vector_cache_max_objects": 5}),
        _flat_with(None),
        "vector default: vector_cache_max_objects is 0, plan has 5",
    ),
    "dynamic-threshold": (
        p.IndexPlan(type="dynamic", options={"threshold": 500}),
        _dynamic(),
        "vector default: threshold is 10000, plan has 500",
    ),
    "hfresh-search-probe": (
        p.IndexPlan(type="hfresh", options={"search_probe": 64}),
        _hfresh(),
        "vector default: search_probe is 256, plan has 64",
    ),
}


@pytest.mark.parametrize(
    ("planned", "stored", "expected"),
    INDEX_OPTION_MISMATCHES.values(),
    ids=INDEX_OPTION_MISMATCHES,
)
def test_differences_compare_each_planned_index_option(planned, stored, expected):
    cfg = config(index=stored)
    found = differences(FakeClient(FakeCollection(cfg)), [entry(index=planned)])  # type: ignore[arg-type]
    assert found == [expected]


def test_differences_accept_equal_index_options():
    # hnsw() stores 0 for every option; the plan sets two of them.
    index = p.IndexPlan(type="hnsw", options={"ef": 0, "max_connections": 0})
    cfg = config(index=hnsw())
    assert differences(FakeClient(FakeCollection(cfg)), [entry(index=index)]) == []  # type: ignore[arg-type]


def test_differences_skip_an_index_option_that_the_stored_index_does_not_report():
    stored = _VectorIndexConfigDynamic(
        distance_metric=VectorDistances.COSINE,
        hnsw=hnsw(),
        flat=_flat_with(None),
        threshold=None,  # type: ignore[arg-type]
    )
    index = p.IndexPlan(type="dynamic", options={"threshold": 500})
    cfg = config(index=stored)
    assert differences(FakeClient(FakeCollection(cfg)), [entry(index=index)]) == []  # type: ignore[arg-type]


def test_ensure_refuses_added_centering_before_any_page():
    # rq 4 bits copied without centering, then centering added to plan.yaml:
    # the server refuses to change centering, so copy must stop before pages.
    client = FakeClient(FakeCollection(config(index=hnsw(quantizer=_rq()))))
    quantizer = p.QuantizerPlan(type="rq", bits=4, centering=True)
    index = p.IndexPlan(type="hnsw", quantizer=quantizer)
    with pytest.raises(PlanMismatch, match="centering is False, plan has True"):
        ensure_collection(client, [entry(index=index)])  # type: ignore[arg-type]
    assert client.collections.created == []


def test_ensure_resumes_into_a_collection_made_from_the_same_plan():
    stored = _rq(centering=True, rescore_limit=40, training_limit=100)
    client = FakeClient(FakeCollection(config(index=hnsw(quantizer=stored))))
    quantizer = p.QuantizerPlan(
        type="rq", bits=4, centering=True, rescore_limit=40, training_limit=100
    )
    index = p.IndexPlan(type="hnsw", quantizer=quantizer)
    assert ensure_collection(client, [entry(index=index)]) == "exists"  # type: ignore[arg-type]


def factor_entry(factor: Any) -> p.CollectionPlan:
    made = entry()
    target = made.target.model_copy(update={"replication_factor": factor})
    return made.model_copy(update={"target": target})


def sent_factors(client: FakeClient) -> list[int | None]:
    """The factor of each create that passed; None for one that sent no factor."""
    sent = [kwargs.get("replication_config") for kwargs in client.collections.created]
    return [None if config is None else config.factor for config in sent]


def ensure_reported(client: FakeClient, group: list[p.CollectionPlan]) -> list[Created]:
    reported: list[Created] = []
    ensure_collection(client, group, sleep=lambda s: None, created=reported.append)  # type: ignore[arg-type]
    return reported


def test_kwargs_send_no_replication_config_for_auto_without_a_factor():
    assert "replication_config" not in collection_kwargs([entry()])


def test_kwargs_send_only_the_factor():
    for sent in (
        collection_kwargs([entry()], 3)["replication_config"],
        collection_kwargs([factor_entry(3)])["replication_config"],
    ):
        assert sent.model_dump(exclude_none=True) == {"factor": 3}


def test_kwargs_send_the_planned_factor_whatever_the_auto_factor_is():
    assert collection_kwargs([factor_entry(2)], 3)["replication_config"].factor == 2


@pytest.mark.parametrize(("nodes", "factor"), [(1, 1), (2, 2), (3, 3), (5, 3)])
def test_auto_creates_with_the_node_count_up_to_3(nodes, factor):
    client = FakeClient(None, nodes=nodes)
    assert ensure_reported(client, [entry()]) == [
        Created("My_docs", factor, "nodes", nodes)
    ]
    assert sent_factors(client) == [factor]
    assert client.cluster.calls == 1


def test_auto_retries_a_transient_failure_of_the_node_read():
    client = FakeClient(None, nodes=3)
    client.cluster.fail = [WeaviateConnectionError("down")]
    assert ensure_reported(client, [entry()]) == [Created("My_docs", 3, "nodes", 3)]


def _forbidden() -> InsufficientPermissionsError:
    return InsufficientPermissionsError(httpx.Response(403, json={}))


def test_auto_sends_no_factor_when_the_node_read_is_forbidden():
    client = FakeClient(None, nodes=3)
    client.cluster.fail = [_forbidden()]
    client.collections.created_cfg = config(factor=2)
    assert ensure_reported(client, [entry()]) == [Created("My_docs", 2, "server")]
    assert sent_factors(client) == [None]


def test_auto_keeps_another_error_of_the_node_read():
    client = FakeClient(None)
    client.cluster.fail = [_status_error(401)]
    with pytest.raises(UnexpectedStatusCodeError):
        ensure_reported(client, [entry()])
    assert client.collections.created == []


def _refusal(status: int, message: str) -> UnexpectedStatusCodeError:
    body = {"error": [{"message": message}]}
    return UnexpectedStatusCodeError("refused", httpx.Response(status, json=body))


MINIMUM = "invalid replication factor: setup requires a minimum of 5: got 3"


@pytest.mark.parametrize("status", [400, 422])
def test_auto_creates_again_without_a_factor_when_the_factor_is_refused(status):
    client = FakeClient(None, nodes=3)
    client.collections.fail = [_refusal(status, MINIMUM)]
    client.collections.created_cfg = config(factor=5)
    assert ensure_reported(client, [entry()]) == [
        Created("My_docs", 5, "server", 3, MINIMUM)
    ]
    assert sent_factors(client) == [None]


def test_auto_reports_the_refusal_of_the_create_without_a_factor():
    client = FakeClient(None, nodes=3)
    client.collections.fail = [
        _refusal(422, MINIMUM),
        _refusal(422, "second refusal"),
        _refusal(422, "third refusal"),
    ]
    with pytest.raises(SchemaRejected) as info:
        WeaviateSchema(client, sleep=lambda s: None).ensure([entry()])  # type: ignore[arg-type]
    assert str(info.value) == "second refusal"
    assert len(client.collections.fail) == 1 and client.collections.created == []


def _count_creates(client: FakeClient) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    create = client.collections.create

    def counted(**kwargs: Any) -> FakeCollection:
        calls.append(kwargs)
        return create(**kwargs)

    client.collections.create = counted  # type: ignore[method-assign]
    return calls


def test_auto_counts_a_create_applied_before_its_answer_was_lost_as_created():
    client = FakeClient(None, nodes=3)
    _applied_then_lost(client, config(factor=3), WeaviateTimeoutError("read timeout"))
    calls = _count_creates(client)
    assert ensure_reported(client, [entry()]) == [Created("My_docs", 3, "nodes", 3)]
    assert [kwargs["replication_config"].factor for kwargs in calls] == [3, 3]


def test_auto_without_a_factor_counts_a_lost_create_as_created():
    client = FakeClient(None, nodes=3)
    client.cluster.fail = [_forbidden()]
    _applied_then_lost(client, config(factor=2), WeaviateTimeoutError("read timeout"))
    calls = _count_creates(client)
    assert ensure_reported(client, [entry()]) == [Created("My_docs", 2, "server", None)]
    assert ["replication_config" in kwargs for kwargs in calls] == [False, False]


def test_auto_lost_create_that_left_another_collection_is_a_plan_mismatch():
    client = FakeClient(None, nodes=3)
    _applied_then_lost(
        client, config(properties={"year": "text"}), WeaviateTimeoutError("t")
    )
    calls = _count_creates(client)
    with pytest.raises(PlanMismatch, match="property year: type is text"):
        ensure_reported(client, [entry()])
    assert len(calls) == 2


def test_auto_does_not_create_again_after_another_status():
    client = FakeClient(None, nodes=3)
    client.collections.fail = [_status_error(401), _status_error(401)]
    with pytest.raises(UnexpectedStatusCodeError):
        ensure_reported(client, [entry()])
    assert len(client.collections.fail) == 1


def test_auto_does_not_create_again_when_it_sent_no_factor():
    client = FakeClient(None)
    client.cluster.fail = [_forbidden()]
    client.collections.fail = [_refusal(422, "first"), _refusal(422, "second")]
    with pytest.raises(UnexpectedStatusCodeError):
        ensure_reported(client, [entry()])
    assert len(client.collections.fail) == 1


@pytest.mark.parametrize("nodes", [1, 3, 5])
def test_a_planned_factor_is_sent_as_it_is_without_a_node_read(nodes):
    client = FakeClient(None, nodes=nodes)
    assert ensure_reported(client, [factor_entry(2)]) == [Created("My_docs", 2, "plan")]
    assert sent_factors(client) == [2]
    assert client.cluster.calls == 0


def test_a_refused_planned_factor_is_not_created_again():
    client = FakeClient(None, nodes=1)
    message = (
        "could not find enough weaviate nodes for replication: 1 available, 3 requested"
    )
    client.collections.fail = [_refusal(422, message), _refusal(422, "second")]
    with pytest.raises(SchemaRejected) as info:
        WeaviateSchema(client, sleep=lambda s: None).ensure([factor_entry(3)])  # type: ignore[arg-type]
    assert str(info.value) == message
    assert len(client.collections.fail) == 1 and client.collections.created == []


@pytest.mark.parametrize("factor", ["auto", 2])
def test_an_existing_collection_is_not_reported_and_needs_no_node_read(factor):
    client = FakeClient(FakeCollection(config(factor=2)), nodes=3)
    assert ensure_reported(client, [factor_entry(factor)]) == []
    assert client.cluster.calls == 0 and client.collections.created == []


@pytest.mark.parametrize("stored", [1, 2, 3, 5])
def test_differences_skip_the_factor_of_a_plan_with_auto(stored):
    client = FakeClient(FakeCollection(config(factor=stored)), nodes=3)
    assert differences(client, [entry()]) == []  # type: ignore[arg-type]
    assert client.cluster.calls == 0


def test_differences_compare_a_planned_factor():
    client = FakeClient(FakeCollection(config(factor=1)))
    assert differences(client, [factor_entry(3)]) == [  # type: ignore[arg-type]
        "replication factor is 1, plan has 3"
    ]
    assert differences(client, [factor_entry(1)]) == []  # type: ignore[arg-type]


def test_ensure_refuses_an_existing_collection_with_another_factor():
    client = FakeClient(FakeCollection(config(factor=1)))
    with pytest.raises(PlanMismatch, match="replication factor is 1, plan has 3"):
        ensure_collection(client, [factor_entry(3)])  # type: ignore[arg-type]


def test_the_schema_reads_the_node_count_and_the_factor_of_a_collection():
    client = FakeClient(FakeCollection(config(factor=2)), nodes=5)
    made = WeaviateSchema(client, sleep=lambda s: None)  # type: ignore[arg-type]
    assert made.nodes() == 5
    assert made.replication_factor([entry()]) == 2
    assert replication_factor(FakeClient(None), [entry()]) is None  # type: ignore[arg-type]
    client.cluster.fail = [_forbidden()]
    assert node_count(client, lambda s: None) is None  # type: ignore[arg-type]
