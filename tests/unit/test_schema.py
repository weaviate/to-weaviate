import functools
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from fakes import FakeRpcError, raise_grpc_failure
from weaviate.classes.config import Configure, DataType, VectorDistances
from weaviate.collections.classes.config import (
    _BQConfig,
    _RQConfig,
    _VectorIndexConfigDynamic,
    _VectorIndexConfigHNSW,
)
from weaviate.exceptions import (
    UnexpectedStatusCodeError,
    WeaviateBaseError,
    WeaviateBatchError,
    WeaviateConnectionError,
    WeaviateQueryError,
    WeaviateRetryError,
    WeaviateTenantGetError,
)

from to_weaviate import plan as p
from to_weaviate.settings import WeaviateSettings
from to_weaviate.target import connect as connect_module
from to_weaviate.target import schema
from to_weaviate.target.schema import (
    PlanMismatch,
    SchemaRejected,
    WeaviateSchema,
    collection_kwargs,
    differences,
    dropped_options,
    ensure_collection,
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

    def create(self, tenants: list[Any]) -> None:
        self.created.append([t.name for t in tenants])


class FakeCollection:
    def __init__(self, cfg: Any = None) -> None:
        self._cfg = cfg
        self.config = SimpleNamespace(get=lambda: self._cfg)
        self.tenants = FakeTenants()


class FakeCollections:
    def __init__(self, existing: FakeCollection | None) -> None:
        self.existing = existing
        self.created: list[dict[str, Any]] = []
        self.fail: list[Exception] = []

    def exists(self, name: str) -> bool:
        return self.existing is not None

    def get(self, name: str) -> FakeCollection:
        assert self.existing is not None
        return self.existing

    def create(self, **kwargs: Any) -> FakeCollection:
        if self.fail:
            raise self.fail.pop(0)
        self.created.append(kwargs)
        self.existing = FakeCollection()
        return self.existing


class FakeClient:
    def __init__(self, existing: FakeCollection | None) -> None:
        self.collections = FakeCollections(existing)


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


def test_differences_missing_property():
    cfg = config()
    cfg.properties = [x for x in cfg.properties if x.name != "year"]
    found = differences(FakeClient(FakeCollection(cfg)), [entry()])  # type: ignore[arg-type]
    assert found and "year" in found[0]


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
def test_kwargs_leave_out_a_url_option_that_keeps_its_userinfo(char: str) -> None:
    options = {
        "model": "text-embedding-3-small",
        "base_url": f"https://bob:made{char}up-pw@proxy.example.com/v1",
        "endpoint": "https://bob:made-up-pw@embed.example.com/e",
    }
    v = p.VectorizerPlan(module="text2vec-openai", options=options)
    assert dropped_options(v) == ["base_url"]
    k = collection_kwargs([entry(mode="embed", vectorizer=v)])
    found = vec(k).vectorizer
    assert (
        found.baseURL is None and str(found.endpoint) == "https://embed.example.com/e"
    )
    for value in ("bob", "made", "up-pw", "@"):
        assert value not in repr(k)


def test_dropped_options_skip_values_that_clean():
    options = {
        "base_url": "https://bob:pw@proxy.example.com/v1?k=x#f",
        "model": "bob:x@host",
        "dimensions": 256,
    }
    v = p.VectorizerPlan(module="text2vec-openai", options=options)
    assert dropped_options(v) == []


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
    client.collections.fail = [
        UnexpectedStatusCodeError(
            "Collection may not have been created properly.",
            httpx.Response(status, json=body),
        )
    ]
    with pytest.raises(SchemaRejected) as info:
        WeaviateSchema(client).ensure([entry()])  # type: ignore[arg-type]
    assert str(info.value) == "vector index dynamic needs async indexing"


def test_weaviate_schema_keeps_other_status_errors():
    client = FakeClient(None)
    client.collections.fail = [_status_error(401)]
    with pytest.raises(UnexpectedStatusCodeError):
        WeaviateSchema(client).ensure([entry()])  # type: ignore[arg-type]


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
