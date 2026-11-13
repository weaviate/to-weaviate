import inspect
import re
import time
import typing
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, Protocol, Union

import weaviate
from pydantic import ValidationError
from weaviate.classes.config import (
    Configure,
    DataType,
    Property,
    Reconfigure,
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
    WeaviateRetryError,
    WeaviateTimeoutError,
)

from to_weaviate.plan import (
    QUANTIZER_FIELDS,
    CollectionPlan,
    PropertyPlan,
    QuantizerPlan,
    VectorizerPlan,
    VectorPlan,
)
from to_weaviate.retry import with_retry
from to_weaviate.settings import stored_url

RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
GRPC_RETRY_STATUSES = frozenset(
    {"UNAVAILABLE", "DEADLINE_EXCEEDED", "RESOURCE_EXHAUSTED"}
)
CHAIN_DEPTH = 5
USERINFO_TEXT = "holds userinfo that cannot be removed"
_MISSING_ARGUMENTS = re.compile(r"missing \d+ required .*?arguments?: (.*)")

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
PROPERTY_TYPES = tuple(sorted(_DATA_TYPES))
_QUANTIZER_CLASSES: dict[type, str] = {
    _RQConfig: "rq",
    _PQConfig: "pq",
    _BQConfig: "bq",
    _SQConfig: "sq",
}


# Quantizer fields that a plan can set and the server reports. `differences`
# compares them with the existing collection. copy's config update after the
# pages sends them, and the server refuses a change of bits or centering.
COMPARED_QUANTIZER_FIELDS = (
    "cache",
    "bits",
    "centering",
    "segments",
    "centroids",
    "training_limit",
    "rescore_limit",
)

# The attribute of a stored property that holds each PropertyPlan flag.
_STORED_FLAGS = {
    "tokenization": "tokenization",
    "searchable": "index_searchable",
    "filterable": "index_filterable",
    "range_filters": "index_range_filters",
}

# A server reports a Google vectorizer as text2vec-google or as the legacy
# name text2vec-palm.
_MODULE_NAMES = {"text2vec-google": frozenset({"text2vec-google", "text2vec-palm"})}

# Arguments that `_vector` sets itself.
_RESERVED_OPTIONS = frozenset(
    {
        "name",
        "source_properties",
        "vectorize_collection_name",
        "quantizer",
        "vector_index_config",
    }
)


class PlanMismatch(Exception):
    pass


class SchemaRejected(Exception):
    """The server refused a schema change; the message is the server's.

    `action` names the change, as in "create the collection".
    """

    def __init__(self, message: str, action: str = "create the collection") -> None:
        super().__init__(message)
        self.action = action


class PropertyRejected(SchemaRejected):
    """The server refused to add a property to an existing collection."""


MAX_AUTO_FACTOR = 3


def auto_factor(nodes: int) -> int:
    """The factor that `replication_factor: auto` sends to a cluster of `nodes`."""
    return max(1, min(MAX_AUTO_FACTOR, nodes))


@dataclass(frozen=True)
class Created:
    """A collection that `ensure_collection` created, and its replication factor.

    `chosen_by` is "plan" for a factor of the plan and "nodes" for the factor
    that `auto` took from the node count. It is "server" when the create sent
    no factor: the key may not read the nodes (`nodes` is None), or the server
    refused the factor of `auto` with the message `refused`.
    """

    name: str
    factor: int
    chosen_by: Literal["plan", "nodes", "server"]
    nodes: int | None = None
    refused: str | None = None


class Schema(Protocol):
    """Target schema operations. `group` holds the entries of one target name."""

    def modules(self) -> set[str]: ...

    def differences(self, group: list[CollectionPlan]) -> list[str] | None: ...

    def missing_properties(self, group: list[CollectionPlan]) -> list[str]: ...

    def nodes(self) -> int | None:
        """The number of nodes; None when the key may not read them."""
        ...

    def replication_factor(self, group: list[CollectionPlan]) -> int | None:
        """The factor of the existing collection; None without one."""
        ...

    def ensure(
        self,
        group: list[CollectionPlan],
        created: Callable[[Created], None] | None = None,
    ) -> Literal["created", "exists"]: ...

    def enable_trained_quantizers(self, entry: CollectionPlan) -> None: ...


def module_names(module: str) -> frozenset[str]:
    """Names a server can report for the planned vectorizer `module`."""
    return _MODULE_NAMES.get(module, frozenset({module}))


def _val(x: Any) -> str:
    return str(getattr(x, "value", x))


def _grpc_status(exc: BaseException | None) -> str | None:
    """The status name of a gRPC error, read through its `code()` method."""
    code = getattr(exc, "code", None)
    return getattr(code(), "name", None) if callable(code) else None


def _chain(exc: BaseException) -> list[BaseException]:
    """`exc` and its causes and contexts, up to CHAIN_DEPTH links below it.

    Nearer links come first. On one level a cause comes before a context.
    """
    found = [exc]
    level = [exc]
    for _ in range(CHAIN_DEPTH):
        links = (e for error in level for e in (error.__cause__, error.__context__))
        level = list({id(e): e for e in links if e is not None}.values())
        found.extend(level)
    return found


def nearest_grpc_status(exc: Exception) -> str | None:
    """The status name of the nearest gRPC error in the chain of `exc`, or None.

    The client raises its own error inside the except block of the gRPC
    error, so the gRPC error is in the chain. Only the nearest gRPC error
    decides: the client's retry makes each attempt inside the except block of
    the one before, so a deeper link can be an earlier attempt. After its own
    retries of UNAVAILABLE, the client raises WeaviateRetryError, or a query
    error inside the except block of one; that link counts as UNAVAILABLE.
    """
    for link in _chain(exc):
        if isinstance(link, WeaviateRetryError):
            return "UNAVAILABLE"
        status = _grpc_status(link)
        if status is not None:
            return status
    return None


def transient(exc: Exception) -> float | None:
    if isinstance(exc, WeaviateConnectionError | WeaviateTimeoutError):
        return 0.0
    if isinstance(exc, UnexpectedStatusCodeError) and exc.status_code in RETRY_STATUSES:
        return 0.0
    return 0.0 if nearest_grpc_status(exc) in GRPC_RETRY_STATUSES else None


def refused_for_size(exc: Exception) -> bool:
    """Whether the nearest gRPC error in the chain of `exc` is RESOURCE_EXHAUSTED.

    gRPC refuses a message above its size limit with this status.
    """
    return nearest_grpc_status(exc) == "RESOURCE_EXHAUSTED"


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


def _quantizer_fields(plan: VectorPlan) -> dict[str, Any]:
    """The planned fields of the quantizer of `plan` that are set."""
    q = plan.index.quantizer
    return {
        f: getattr(q, f) for f in QUANTIZER_FIELDS[q.type] if getattr(q, f) is not None
    }


def _quantizer(plan: VectorPlan) -> Any:
    q = plan.index.quantizer
    if q.type == "default":
        return None
    if q.type == "none":
        return Configure.VectorIndex.Quantizer.none()
    return getattr(Configure.VectorIndex.Quantizer, q.type)(**_quantizer_fields(plan))


def _is_url(value: Any) -> bool:
    return isinstance(value, str) and "://" in value


def _sent_options(vectorizer: VectorizerPlan, collection: str) -> dict[str, Any]:
    """The options of `vectorizer` as Weaviate gets them.

    Weaviate stores the options in the schema, so a URL option is sent
    without its userinfo, query and fragment, whatever the plan holds. A URL
    option that `stored_url` does not store raises PlanMismatch: without it,
    the provider would get the text at its default endpoint.
    """
    sent: dict[str, Any] = {}
    for key, value in vectorizer.options.items():
        if _is_url(value):
            value = stored_url(value)
            if value is None:
                raise PlanMismatch(
                    f"vectorizer option {key} of {vectorizer.module} in "
                    f"{collection} {USERINFO_TEXT}"
                )
        sent[key] = value
    return sent


def cleaned_options(vectorizer: VectorizerPlan) -> list[str]:
    """Keys of the URL options that are sent without userinfo, a query or a fragment.

    A URL option whose userinfo cannot be removed is an error, not one of these.
    """
    return [
        key
        for key, value in vectorizer.options.items()
        if _is_url(value) and stored_url(value) not in (None, value)
    ]


_SCALAR_TYPES = (bool, int, float)


def _declared_scalar(hint: Any) -> type | None:
    """bool, int or float when `hint` is that type, optional or not; else None."""
    types = [hint]
    if typing.get_origin(hint) is Union:
        types = [t for t in typing.get_args(hint) if t is not type(None)]
    if len(types) == 1 and types[0] in _SCALAR_TYPES:
        return types[0]  # type: ignore[no-any-return]
    return None


def _scalar(text: str, kind: type) -> Any:
    """`text` as `kind`, or `text` when it does not read as one."""
    if kind is bool:
        return {"true": True, "false": False}.get(text.lower(), text)
    try:
        return kind(text)
    except ValueError:
        return text


def _declared_options(factory: Any, options: dict[str, Any]) -> dict[str, Any]:
    """`options` with each text value as the bool, int or float the factory declares.

    plan.yaml holds a --vectorizer-option value as the text given, and the
    client refuses a text where it declares a number or a boolean.
    """
    hints = typing.get_type_hints(factory)
    out: dict[str, Any] = {}
    for key, value in options.items():
        kind = _declared_scalar(hints.get(key))
        out[key] = _scalar(value, kind) if isinstance(value, str) and kind else value
    return out


def _vector(plan: VectorPlan, collection: str) -> Any:
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
    options = _sent_options(plan.vectorizer, collection)
    accepted = inspect.signature(factory).parameters
    for key in options:
        if key in _RESERVED_OPTIONS or key not in accepted:
            raise PlanMismatch(f"vectorizer option {key} is not accepted by {module}")
    options = _declared_options(factory, options)
    try:
        return factory(
            **kwargs,
            source_properties=["document"],
            vectorize_collection_name=False,
            **options,
        )
    except ValidationError as exc:
        error = exc.errors()[0]
        key = ".".join(str(part) for part in error["loc"])
        raise PlanMismatch(
            f"vectorizer option {key} of {module}: {error['msg']}"
        ) from exc
    except TypeError as exc:
        raise PlanMismatch(_type_error_text(module, exc)) from exc


def _type_error_text(module: str, exc: TypeError) -> str:
    """The message for a TypeError of a factory; names the missing options.

    Python names the missing arguments of a call as in "missing 2 required
    keyword-only arguments: 'model' and 'region'".
    """
    missing = _MISSING_ARGUMENTS.search(str(exc))
    names = re.findall(r"'([^']+)'", missing.group(1)) if missing else []
    if not names:
        return f"vectorizer {module} refused its options: {exc}"
    noun = "option" if len(names) == 1 else "options"
    return f"vectorizer {module} needs the {noun} {', '.join(names)}"


def _properties(group: list[CollectionPlan]) -> list[PropertyPlan]:
    seen: dict[str, PropertyPlan] = {}
    for entry in group:
        for prop in entry.target.properties:
            seen.setdefault(prop.name, prop)
    return list(seen.values())


def _tenants(group: list[CollectionPlan]) -> list[str]:
    return list(dict.fromkeys(e.target.tenant for e in group if e.target.tenant))


def collection_kwargs(
    group: list[CollectionPlan], auto: int | None = None
) -> dict[str, Any]:
    """The arguments that create the collection of `group`.

    `auto` is the factor to send for `replication_factor: auto`. Without it
    and without a factor in the plan, the server chooses the factor.
    """
    name = group[0].target.name
    planned = group[0].target.replication_factor
    factor = planned if isinstance(planned, int) else auto
    kwargs: dict[str, Any] = {
        "name": name,
        "properties": [_property(p) for p in _properties(group)],
        "vector_config": [_vector(v, name) for v in group[0].target.vectors],
    }
    if _tenants(group):
        kwargs["multi_tenancy_config"] = Configure.multi_tenancy(enabled=True)
    if factor is not None:
        kwargs["replication_config"] = Configure.replication(factor=factor)
    return kwargs


def quantizer_updates(entry: CollectionPlan) -> list[Any]:
    """Config updates that enable each planned quantizer that needs training.

    They hold the same quantizer as `collection_kwargs`. `differences`
    compares the fields they send with an existing collection, so the update
    does not change its config. Only hnsw takes a quantizer that needs
    training.
    """
    return [
        Reconfigure.Vectors.update(
            name=vector.name,
            vector_index_config=getattr(Reconfigure.VectorIndex, vector.index.type)(
                quantizer=getattr(
                    Reconfigure.VectorIndex.Quantizer, vector.index.quantizer.type
                )(**_quantizer_fields(vector))
            ),
        )
        for vector in entry.target.vectors
        if vector.index.quantizer.needs_training()
    ]


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
    if actual not in module_names(planned):
        out.append(f"vector {name}: vectorizer is {actual}, plan has {planned}")
    if plan.index.type != "default":
        actual_type = _val(index.vector_index_type())
        if actual_type != plan.index.type:
            out.append(
                f"vector {name}: index type is {actual_type}, "
                f"plan has {plan.index.type}"
            )
        else:
            out.extend(_index_option_differences(name, plan.index.options, index))
    q = plan.index.quantizer
    if q.type != "default":
        types = _index_quantizers(index)
        if any(t != q.type for t in types):
            out.append(
                f"vector {name}: quantizer type is {'/'.join(types)}, plan has {q.type}"
            )
        else:
            parts = [index.hnsw, index.flat] if len(types) == 2 else [index]
            out.extend(_quantizer_field_differences(name, q, parts))
    return out


def _index_option_differences(
    name: str, options: dict[str, int], index: Any
) -> list[str]:
    """Planned index options that the index holds otherwise.

    An option that the index does not report is not compared.
    """
    out: list[str] = []
    for option, value in options.items():
        found = getattr(index, option, None)
        if found is not None and found != value:
            out.append(f"vector {name}: {option} is {found}, plan has {value}")
    return out


def _quantizer_field_differences(
    name: str, planned: QuantizerPlan, parts: list[Any]
) -> list[str]:
    """Planned COMPARED_QUANTIZER_FIELDS that the index parts hold otherwise.

    A field that the plan does not set is not compared, nor a field that a
    part does not report: that index type does not store it.
    """
    out: list[str] = []
    for field in COMPARED_QUANTIZER_FIELDS:
        value = getattr(planned, field)
        if value is None:
            continue
        found = [getattr(part.quantizer, field, None) for part in parts]
        differing = [f for f in found if f is not None and f != value]
        if differing:
            out.append(f"vector {name}: {field} is {differing[0]}, plan has {value}")
    return out


def _property_flag_differences(plan: PropertyPlan, found: Any) -> list[str]:
    """The flags that `_property` sends for `plan` and `found` holds otherwise.

    A flag that the server does not report is not compared.
    """
    planned: dict[str, Any] = {}
    is_text = plan.type in ("text", "text[]")
    if is_text and plan.tokenization:
        planned["tokenization"] = plan.tokenization
    if is_text:
        planned["searchable"] = plan.searchable
    planned["filterable"] = plan.filterable
    planned["range_filters"] = plan.range_filters
    out: list[str] = []
    for flag, value in planned.items():
        reported = getattr(found, _STORED_FLAGS[flag], None)
        if reported is not None and _val(reported) != _val(value):
            out.append(
                f"property {plan.name}: {flag} is {_val(reported)}, plan has {value}"
            )
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
    stored = {p.name: p for p in cfg.properties}
    for prop in _properties(group):
        # A type without a mapping is an invalid_type finding of the plan. A
        # missing property is no difference: ensure adds it.
        planned_type = _DATA_TYPES.get(prop.type)
        if prop.name not in stored or planned_type is None:
            continue
        actual = _val(stored[prop.name].data_type)
        if actual != planned_type.value:
            out.append(f"property {prop.name}: type is {actual}, plan has {prop.type}")
        else:
            out.extend(_property_flag_differences(prop, stored[prop.name]))
    planned_mt = bool(_tenants(group))
    if planned_mt != cfg.multi_tenancy_config.enabled:
        found_mt = "on" if cfg.multi_tenancy_config.enabled else "off"
        out.append(
            f"multi-tenancy is {found_mt}, plan has {'on' if planned_mt else 'off'}"
        )
    planned_factor = group[0].target.replication_factor
    if isinstance(planned_factor, int):
        found_factor = cfg.replication_config.factor
        if found_factor != planned_factor:
            out.append(
                f"replication factor is {found_factor}, plan has {planned_factor}"
            )
    return out


def node_count(
    client: weaviate.WeaviateClient, sleep: Callable[[float], None] = time.sleep
) -> int | None:
    """The number of nodes; None when the key may not read them."""
    try:
        return len(with_retry(client.cluster.nodes, transient, sleep=sleep))
    except UnexpectedStatusCodeError as exc:
        if exc.status_code == 403:
            return None
        raise


def replication_factor(
    client: weaviate.WeaviateClient,
    group: list[CollectionPlan],
    sleep: Callable[[float], None] = time.sleep,
) -> int | None:
    """The factor of the existing collection of `group`; None without one."""
    name = group[0].target.name
    if not client.collections.exists(name):
        return None
    config = client.collections.get(name).config
    return with_retry(config.get, transient, sleep=sleep).replication_config.factor


def _create(
    client: weaviate.WeaviateClient,
    group: list[CollectionPlan],
    sleep: Callable[[float], None],
) -> Created:
    """Create the collection of `group` with the factor that the plan asks for.

    `auto` sends min(3, nodes). It sends no factor when the key may not read
    the nodes. When the server refuses the factor with 400 or 422, it creates
    once more without a factor; the error of that create is raised.
    """
    name = group[0].target.name

    def create(auto: int | None) -> None:
        kwargs = collection_kwargs(group, auto)
        with_retry(lambda: client.collections.create(**kwargs), transient, sleep=sleep)

    planned = group[0].target.replication_factor
    if isinstance(planned, int):
        create(None)
        return Created(name, planned, "plan")
    nodes = node_count(client, sleep)
    refused = None
    if nodes is not None:
        factor = auto_factor(nodes)
        try:
            create(factor)
        except UnexpectedStatusCodeError as exc:
            if exc.status_code not in (400, 422):
                raise
            refused = _server_message(exc)
        else:
            return Created(name, factor, "nodes", nodes)
    create(None)
    stored = replication_factor(client, group, sleep)
    assert stored is not None
    return Created(name, stored, "server", nodes, refused)


def _missing_properties(
    client: weaviate.WeaviateClient, group: list[CollectionPlan]
) -> list[PropertyPlan]:
    name = group[0].target.name
    if not client.collections.exists(name):
        return []
    stored = {p.name for p in client.collections.get(name).config.get().properties}
    return [p for p in _properties(group) if p.name not in stored]


def missing_properties(
    client: weaviate.WeaviateClient, group: list[CollectionPlan]
) -> list[str]:
    """Planned properties that the existing collection lacks; [] without one."""
    return [p.name for p in _missing_properties(client, group)]


def _add_property(
    client: weaviate.WeaviateClient,
    name: str,
    prop: PropertyPlan,
    sleep: Callable[[float], None],
) -> None:
    config = client.collections.get(name).config
    try:
        with_retry(lambda: config.add_property(_property(prop)), transient, sleep=sleep)
    except UnexpectedStatusCodeError as exc:
        if exc.status_code in (400, 422):
            raise PropertyRejected(
                _server_message(exc), f"add the property {prop.name} to {name}"
            ) from exc
        raise


def ensure_collection(
    client: weaviate.WeaviateClient,
    group: list[CollectionPlan],
    sleep: Callable[[float], None] = time.sleep,
    created: Callable[[Created], None] | None = None,
) -> Literal["created", "exists"]:
    """Create the collection of `group`, or add the planned properties it lacks.

    Raises PlanMismatch when the existing collection has other settings.
    `created` gets the collection that this call created.
    """
    found = differences(client, group)
    if found:
        raise PlanMismatch("; ".join(found))
    name = group[0].target.name
    result: Literal["created", "exists"] = "exists"
    if found is None:
        made = _create(client, group, sleep)
        if created is not None:
            created(made)
        result = "created"
    else:
        for prop in _missing_properties(client, group):
            _add_property(client, name, prop, sleep)
    tenants = _tenants(group)
    if tenants:
        collection = client.collections.get(name)
        try:
            with_retry(
                lambda: collection.tenants.create([Tenant(name=t) for t in tenants]),
                transient,
                sleep=sleep,
            )
        except UnexpectedStatusCodeError as exc:
            if exc.status_code in (400, 422):
                raise SchemaRejected(
                    _server_message(exc), f"create the tenants of {name}"
                ) from exc
            raise
    return result


def _server_message(exc: UnexpectedStatusCodeError) -> str:
    body: Any = exc.error
    if isinstance(body, dict) and isinstance(body.get("error"), list):
        texts = [
            str(item["message"])
            for item in body["error"]
            if isinstance(item, dict) and item.get("message")
        ]
        if texts:
            return "; ".join(texts)
    return exc.message


class WeaviateSchema:
    def __init__(
        self,
        client: weaviate.WeaviateClient,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._client = client
        self._sleep = sleep

    def modules(self) -> set[str]:
        return set(self._client.get_meta().get("modules") or {})

    def differences(self, group: list[CollectionPlan]) -> list[str] | None:
        return differences(self._client, group)

    def missing_properties(self, group: list[CollectionPlan]) -> list[str]:
        return missing_properties(self._client, group)

    def nodes(self) -> int | None:
        return node_count(self._client, self._sleep)

    def replication_factor(self, group: list[CollectionPlan]) -> int | None:
        return replication_factor(self._client, group, self._sleep)

    def ensure(
        self,
        group: list[CollectionPlan],
        created: Callable[[Created], None] | None = None,
    ) -> Literal["created", "exists"]:
        try:
            return ensure_collection(self._client, group, self._sleep, created)
        except UnexpectedStatusCodeError as exc:
            if exc.status_code in (400, 422):
                raise SchemaRejected(_server_message(exc)) from exc
            raise

    def enable_trained_quantizers(self, entry: CollectionPlan) -> None:
        """Send `quantizer_updates` of `entry`, so that the server trains them."""
        updates = quantizer_updates(entry)
        if not updates:
            return
        name = entry.target.name
        config = self._client.collections.get(name).config
        try:
            with_retry(
                lambda: config.update(vector_config=updates),
                transient,
                sleep=self._sleep,
            )
        except UnexpectedStatusCodeError as exc:
            if exc.status_code in (400, 422):
                raise SchemaRejected(
                    _server_message(exc), f"enable the quantizers of {name}"
                ) from exc
            raise
