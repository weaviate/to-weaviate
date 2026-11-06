import time
from collections.abc import Callable
from typing import Any, Literal

import weaviate
from weaviate.classes.config import (
    Configure,
    DataType,
    Property,
    Tokenization,
    VectorDistances,
)
from weaviate.classes.tenants import Tenant
from weaviate.collections.classes.config import (
    _BQConfig,
    _PQConfig,
    _RQConfig,
    _SQConfig,
)
from weaviate.exceptions import (
    UnexpectedStatusCodeError,
    WeaviateConnectionError,
    WeaviateTimeoutError,
)

from to_weaviate.plan import (
    QUANTIZER_FIELDS,
    CollectionPlan,
    PropertyPlan,
    VectorPlan,
)
from to_weaviate.retry import with_retry

RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})

_DATA_TYPES = {
    "text": DataType.TEXT,
    "int": DataType.INT,
    "number": DataType.NUMBER,
    "boolean": DataType.BOOL,
    "text[]": DataType.TEXT_ARRAY,
    "int[]": DataType.INT_ARRAY,
    "number[]": DataType.NUMBER_ARRAY,
    "boolean[]": DataType.BOOL_ARRAY,
}
_QUANTIZER_CLASSES: dict[type, str] = {
    _RQConfig: "rq",
    _PQConfig: "pq",
    _BQConfig: "bq",
    _SQConfig: "sq",
}


# The server reports both Google factories as the legacy module name.
_GOOGLE_ALIASES = {"text2vec-google": "text2vec-palm"}


class PlanMismatch(Exception):
    pass


def _val(x: Any) -> str:
    return str(getattr(x, "value", x))


def transient(exc: Exception) -> float | None:
    if isinstance(exc, WeaviateConnectionError | WeaviateTimeoutError):
        return 0.0
    if isinstance(exc, UnexpectedStatusCodeError) and exc.status_code in RETRY_STATUSES:
        return 0.0
    return None


def _property(plan: PropertyPlan) -> Property:
    is_text = plan.type in ("text", "text[]")
    return Property(
        name=plan.name,
        data_type=_DATA_TYPES[plan.type],
        tokenization=Tokenization(plan.tokenization)
        if is_text and plan.tokenization
        else None,
        index_filterable=plan.filterable,
        index_range_filters=plan.range_filters,
        index_searchable=plan.searchable if is_text else None,
    )


def _quantizer(plan: VectorPlan) -> Any:
    q = plan.index.quantizer
    if q.type == "default":
        return None
    if q.type == "none":
        return Configure.VectorIndex.Quantizer.none()
    fields = {
        f: getattr(q, f) for f in QUANTIZER_FIELDS[q.type] if getattr(q, f) is not None
    }
    return getattr(Configure.VectorIndex.Quantizer, q.type)(**fields)


def _vector(plan: VectorPlan) -> Any:
    kwargs: dict[str, Any] = {"name": plan.name}
    index = plan.index
    if index.type != "default":
        options: dict[str, Any] = dict(index.options)
        if plan.distance is not None:
            options["distance_metric"] = VectorDistances(plan.distance)
        kwargs["vector_index_config"] = getattr(Configure.VectorIndex, index.type)(
            **options
        )
        quantizer = _quantizer(plan)
        if quantizer is not None:
            kwargs["quantizer"] = quantizer
    if plan.vectorizer is None:
        return Configure.Vectors.self_provided(**kwargs)
    module = plan.vectorizer.module
    factory_name = module.replace("-", "_")
    if module == "text2vec-google":
        factory_name = (
            "text2vec_google_vertex"
            if "project_id" in plan.vectorizer.options
            else "text2vec_google_gemini"
        )
    factory = getattr(Configure.Vectors, factory_name, None)
    if factory is None:
        raise PlanMismatch(
            f"vectorizer module {module} is not supported by this client"
        )
    return factory(
        **kwargs,
        source_properties=["document"],
        vectorize_collection_name=False,
        **plan.vectorizer.options,
    )


def _properties(group: list[CollectionPlan]) -> list[PropertyPlan]:
    seen: dict[str, PropertyPlan] = {}
    for entry in group:
        for prop in entry.target.properties:
            seen.setdefault(prop.name, prop)
    return list(seen.values())


def _tenants(group: list[CollectionPlan]) -> list[str]:
    return list(dict.fromkeys(e.target.tenant for e in group if e.target.tenant))


def collection_kwargs(group: list[CollectionPlan]) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "name": group[0].target.name,
        "properties": [_property(p) for p in _properties(group)],
        "vector_config": [_vector(v) for v in group[0].target.vectors],
    }
    if _tenants(group):
        kwargs["multi_tenancy_config"] = Configure.multi_tenancy(enabled=True)
    return kwargs


def _index_quantizers(index: Any) -> list[str]:
    """Quantizer types of an index; a dynamic index has one per part."""
    parts = (
        [index.hnsw, index.flat]
        if _val(index.vector_index_type()) == "dynamic"
        else [index]
    )
    return [
        _QUANTIZER_CLASSES.get(type(p.quantizer), "none") if p.quantizer else "none"
        for p in parts
    ]


def _vector_differences(plan: VectorPlan, found: Any) -> list[str]:
    name = plan.name
    index = found.vector_index_config
    out: list[str] = []
    if plan.distance is not None and _val(index.distance_metric) != plan.distance:
        out.append(
            f"vector {name}: distance is {_val(index.distance_metric)}, "
            f"plan has {plan.distance}"
        )
    planned = plan.vectorizer.module if plan.vectorizer else "none"
    actual = _val(found.vectorizer.vectorizer)
    if actual != _GOOGLE_ALIASES.get(planned, planned):
        out.append(f"vector {name}: vectorizer is {actual}, plan has {planned}")
    if plan.index.type != "default":
        actual_type = _val(index.vector_index_type())
        if actual_type != plan.index.type:
            out.append(
                f"vector {name}: index type is {actual_type}, "
                f"plan has {plan.index.type}"
            )
    q = plan.index.quantizer
    if q.type != "default":
        types = _index_quantizers(index)
        if any(t != q.type for t in types):
            out.append(
                f"vector {name}: quantizer type is {'/'.join(types)}, plan has {q.type}"
            )
        elif q.bits is not None:
            parts = [index.hnsw, index.flat] if len(types) == 2 else [index]
            bits = [getattr(p.quantizer, "bits", None) for p in parts]
            if any(b != q.bits for b in bits):
                out.append(f"vector {name}: bits is {bits[0]}, plan has {q.bits}")
    return out


def differences(
    client: weaviate.WeaviateClient, group: list[CollectionPlan]
) -> list[str] | None:
    name = group[0].target.name
    if not client.collections.exists(name):
        return None
    cfg = client.collections.get(name).config.get()
    out: list[str] = []
    for plan in group[0].target.vectors:
        found = cfg.vector_config.get(plan.name) if cfg.vector_config else None
        if found is None:
            out.append(f"vector {plan.name}: missing in the collection")
        else:
            out.extend(_vector_differences(plan, found))
    actual = {p.name: _val(p.data_type) for p in cfg.properties}
    for prop in _properties(group):
        if prop.name not in actual:
            out.append(f"property {prop.name}: missing in the collection")
        elif actual[prop.name] != _DATA_TYPES[prop.type].value:
            out.append(
                f"property {prop.name}: type is {actual[prop.name]}, "
                f"plan has {prop.type}"
            )
    planned_mt = bool(_tenants(group))
    if planned_mt != cfg.multi_tenancy_config.enabled:
        found_mt = "on" if cfg.multi_tenancy_config.enabled else "off"
        out.append(
            f"multi-tenancy is {found_mt}, plan has {'on' if planned_mt else 'off'}"
        )
    return out


def ensure_collection(
    client: weaviate.WeaviateClient,
    group: list[CollectionPlan],
    sleep: Callable[[float], None] = time.sleep,
) -> Literal["created", "exists"]:
    found = differences(client, group)
    if found:
        raise PlanMismatch("; ".join(found))
    name = group[0].target.name
    result: Literal["created", "exists"] = "exists"
    if found is None:
        kwargs = collection_kwargs(group)
        with_retry(lambda: client.collections.create(**kwargs), transient, sleep=sleep)
        result = "created"
    tenants = _tenants(group)
    if tenants:
        collection = client.collections.get(name)
        with_retry(
            lambda: collection.tenants.create([Tenant(name=t) for t in tenants]),
            transient,
            sleep=sleep,
        )
    return result
