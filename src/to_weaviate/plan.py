from collections.abc import Callable, Hashable, Mapping
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, field_validator, model_validator
from weaviate.classes.config import VectorDistances

from to_weaviate.naming import collection_name, property_name
from to_weaviate.sources.base import SourceCollection
from to_weaviate.types import resolve_type

INDEX_OPTIONS: dict[str, set[str]] = {
    "default": set(),
    "flat": {"vector_cache_max_objects"},
    "dynamic": {"threshold"},
    "hfresh": {"max_posting_size_kb", "replicas", "search_probe"},
    "hnsw": {
        "ef",
        "ef_construction",
        "max_connections",
        "dynamic_ef_min",
        "dynamic_ef_max",
        "dynamic_ef_factor",
        "flat_search_cutoff",
        "vector_cache_max_objects",
    },
}

QUANTIZER_FIELDS: dict[str, set[str]] = {
    "default": set(),
    "none": set(),
    "rq": {"bits", "rescore_limit", "training_limit", "centering", "cache"},
    "sq": {"rescore_limit", "training_limit"},
    "pq": {"training_limit", "segments", "centroids"},
    "bq": {"rescore_limit", "cache"},
}

# The quantizer types that Weaviate 1.40.4 creates for each named index type.
# It refuses the others, except none with hfresh, which it stores as rq. A
# dynamic index sends its quantizer to its hnsw part and its flat part, so it
# takes the types that both parts take.
INDEX_QUANTIZERS: dict[str, tuple[str, ...]] = {
    "hnsw": ("default", "none", "rq", "sq", "pq", "bq"),
    "flat": ("default", "none", "rq", "bq"),
    "dynamic": ("default", "none", "rq", "bq"),
    "hfresh": ("default", "rq"),
}
# The rq bits that Weaviate 1.40.4 accepts for each named index type. It
# accepts rq centering only with hnsw and 4 bits; without bits, hnsw uses 8.
RQ_BITS: dict[str, tuple[int, ...]] = {
    "hnsw": (1, 4, 8),
    "flat": (1, 8),
    "dynamic": (1, 8),
    "hfresh": (1,),
}
CENTERING_INDEX = "hnsw"
CENTERING_BITS = 4
# The quantizer fields that Weaviate 1.40.4 drops without an error, by index
# type and quantizer type. A dynamic index applies a field that its hnsw part
# or its flat part applies, so it drops none.
IGNORED_FIELDS: dict[str, dict[str, tuple[str, ...]]] = {
    "hnsw": {"rq": ("cache",), "bq": ("cache", "rescore_limit")},
    "flat": {"rq": ("training_limit",)},
    "hfresh": {"rq": ("cache", "training_limit")},
}
# Quantizer types that Weaviate trains on the vectors of the index; rq with
# centering is trained too. Without ASYNC_INDEXING, Weaviate 1.40.4 does not
# train such a quantizer that the collection was created with, and trains it
# when a config update enables it on an index that holds vectors. With
# ASYNC_INDEXING that update changes nothing.
TRAINED_QUANTIZERS = ("sq", "pq")
# The vectors a shard needs before Weaviate 1.40.4 trains the quantizer, when
# the plan sets no training_limit (entities/vectorindex/hnsw/*_config.go).
DEFAULT_TRAINING_LIMITS = {"sq": 100000, "pq": 100000, "rq": 10000}
# The distances that Weaviate 1.40.4 accepts for an index type that does not
# take every distance.
INDEX_DISTANCES: dict[str, tuple[str, ...]] = {"hfresh": ("cosine", "l2-squared")}

# The distances that target.schema maps to VectorDistances.
DISTANCES = tuple(sorted(d.value for d in VectorDistances))
KEEP_DISTANCES_WITHOUT_INDEX = (None, "cosine")
EMBED_INSTEAD = (
    "run to-weaviate inspect chroma --force --vectors embed --vectorizer MODULE"
)
NO_VECTORS_NOTE = f"the source has no vectors; {EMBED_INSTEAD}"
MERGE_TAG = "tag:yaml.org,2002:merge"


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PropertyPlan(_Model):
    source_key: str
    name: str
    type: str
    tokenization: Literal["word", "field"] | None = None
    searchable: bool = False
    filterable: bool = True
    range_filters: bool = False
    coerced_from: list[str] = []


class VectorizerPlan(_Model):
    module: str
    options: dict[str, Any] = {}


class QuantizerPlan(_Model):
    type: Literal["default", "none", "rq", "sq", "pq", "bq"] = "default"
    bits: Literal[1, 4, 8] | None = None
    rescore_limit: int | None = None
    training_limit: int | None = None
    centering: bool | None = None
    cache: bool | None = None
    segments: int | None = None
    centroids: int | None = None

    @model_validator(mode="after")
    def _fields_match_type(self) -> "QuantizerPlan":
        allowed = QUANTIZER_FIELDS[self.type]
        for field in QUANTIZER_FIELDS["rq"] | QUANTIZER_FIELDS["pq"]:
            value = getattr(self, field)
            if value is not None and field not in allowed:
                raise ValueError(
                    f"{field}={value!r} is not valid for quantizer type {self.type!r}"
                )
        return self

    def needs_training(self) -> bool:
        """Whether Weaviate trains this quantizer on the vectors of the index."""
        return self.type in TRAINED_QUANTIZERS or (
            self.type == "rq" and bool(self.centering)
        )

    def training_vectors(self) -> int:
        """The vectors a shard needs before Weaviate trains this quantizer."""
        return self.training_limit or DEFAULT_TRAINING_LIMITS[self.type]


class IndexPlan(_Model):
    type: Literal["default", "hnsw", "flat", "dynamic", "hfresh"] = "default"
    options: dict[str, int] = {}
    quantizer: QuantizerPlan = QuantizerPlan()

    @model_validator(mode="after")
    def _check(self) -> "IndexPlan":
        unknown = sorted(set(self.options) - INDEX_OPTIONS[self.type])
        if unknown:
            raise ValueError(
                f"index option(s) {unknown} are not valid for index type {self.type!r}"
            )
        quantizer = self.quantizer.type
        if quantizer != "default" and self.type == "default":
            raise ValueError(
                f"quantizer type {quantizer!r} needs a named index type, not 'default'"
            )
        supported = INDEX_QUANTIZERS.get(self.type)
        if supported is not None and quantizer not in supported:
            raise ValueError(
                f"quantizer type {quantizer!r} is not supported by index type "
                f"{self.type!r}; it supports {', '.join(supported)}"
            )
        for field in IGNORED_FIELDS.get(self.type, {}).get(quantizer, ()):
            value = getattr(self.quantizer, field)
            if value is not None:
                raise ValueError(
                    f"{field}={value!r} of quantizer type {quantizer!r} is not "
                    f"applied by index type {self.type!r}; remove it"
                )
        if quantizer == "rq":
            self._check_rq()
        return self

    def _check_rq(self) -> None:
        bits = self.quantizer.bits
        supported = RQ_BITS[self.type]
        if bits is not None and bits not in supported:
            raise ValueError(
                f"bits={bits} of quantizer type 'rq' is not supported by index "
                f"type {self.type!r}; it supports {', '.join(map(str, supported))}"
            )
        if self.quantizer.centering and (
            self.type != CENTERING_INDEX or bits != CENTERING_BITS
        ):
            raise ValueError(
                "centering=True is supported only by index type "
                f"{CENTERING_INDEX!r} with bits={CENTERING_BITS}"
            )


class VectorPlan(_Model):
    name: str = "default"
    mode: Literal["keep", "embed"] = "keep"
    vectorizer: VectorizerPlan | None = None
    distance: str | None = None
    index: IndexPlan = IndexPlan()

    @model_validator(mode="after")
    def _distance_needs_named_index(self) -> "VectorPlan":
        # A distance outside DISTANCES is the invalid_distance finding of check.
        if (
            self.index.type == "default"
            and self.distance in DISTANCES
            and self.distance not in KEEP_DISTANCES_WITHOUT_INDEX
        ):
            raise ValueError(
                f"distance {self.distance!r} needs a named index type, not 'default'"
            )
        return self


class DroppedKey(_Model):
    source_key: str
    reason: str


class TargetPlan(_Model):
    name: str
    tenant: str | None = None
    replication_factor: Literal["auto"] | int = "auto"
    vectors: list[VectorPlan]
    properties: list[PropertyPlan]
    dropped: list[DroppedKey] = []

    @field_validator("replication_factor", mode="before")
    @classmethod
    def _factor(cls, value: Any) -> Any:
        number = isinstance(value, int) and not isinstance(value, bool)
        if value != "auto" and not (number and value >= 1):
            raise ValueError(
                f"replication_factor={value!r} is not valid; use auto or an "
                "integer of at least 1"
            )
        return value

    @model_validator(mode="after")
    def _one_kept_vector(self) -> "TargetPlan":
        kept = [v.name for v in self.vectors if v.mode == "keep"]
        if len(kept) > 1:
            raise ValueError(
                f"at most one vector can keep the source vector, got {kept}"
            )
        return self


class SourceRef(_Model):
    ref: str
    namespace: str
    name: str
    count: int
    dimension: int | None
    distance: str | None
    embedding: dict[str, Any] | None


class CollectionPlan(_Model):
    source: SourceRef
    target: TargetPlan
    notes: list[str] = []


class SourcePlan(_Model):
    type: str
    url: str
    tenant: str | None


class Plan(_Model):
    version: Literal[1] = 1
    source: SourcePlan
    collections: list[CollectionPlan]


VectorizerMapper = Callable[
    [dict[str, Any] | None], tuple[VectorizerPlan | None, list[str]]
]


def _properties(
    collection: SourceCollection,
) -> tuple[list[PropertyPlan], list[DroppedKey]]:
    props = [
        PropertyPlan(
            source_key="#document",
            name="document",
            type="text",
            tokenization="word",
            searchable=True,
            filterable=False,
        ),
        PropertyPlan(
            source_key="#id",
            name="source_id",
            type="text",
            tokenization="field",
        ),
        PropertyPlan(source_key="#uri", name="uri", type="text", tokenization="field"),
    ]
    taken = {p.name for p in props}
    for field in collection.fields:
        name = property_name(field.key, taken)
        taken.add(name)
        type_, coerced_from = resolve_type(field.types)
        props.append(
            PropertyPlan(
                source_key=field.key,
                name=name,
                type=type_,
                tokenization="field" if type_ in ("text", "text[]") else None,
                range_filters=type_ in ("int", "number"),
                coerced_from=coerced_from,
            )
        )
    dropped = [
        DroppedKey(source_key=k, reason="sparse vectors are not migrated")
        for k in collection.sparse_keys
    ]
    return props, dropped


def _draft_collection(
    source: SourceCollection,
    taken_names: set[str],
    vectors: Literal["keep", "embed", "both"],
    vectorizer: VectorizerPlan | None,
    index: IndexPlan,
    vectorizer_for: VectorizerMapper | None,
    source_notes: list[str],
) -> CollectionPlan:
    notes = list(source_notes)
    name = collection_name(source.name, taken_names)
    taken_names.add(name)
    props, dropped = _properties(source)

    def kept(vector_name: str, with_vectorizer: bool) -> VectorPlan:
        chosen = None
        if with_vectorizer:
            chosen = vectorizer
            if chosen is None and vectorizer_for is not None:
                chosen, mapper_notes = vectorizer_for(source.embedding)
                notes.extend(mapper_notes)
        vector_index = index
        if source.distance in ("l2-squared", "dot") and index.type == "default":
            vector_index = index.model_copy(update={"type": "hnsw"})
            notes.append(
                f"index type set to hnsw because distance {source.distance} "
                "needs a named index type"
            )
        return VectorPlan(
            name=vector_name,
            mode="keep",
            vectorizer=chosen,
            distance=source.distance,
            index=vector_index,
        )

    def embedded() -> VectorPlan:
        notes.append(
            f"source distance {source.distance} is not carried over to the "
            "embedded vector"
        )
        return VectorPlan(
            name="default",
            mode="embed",
            vectorizer=vectorizer or VectorizerPlan(module="text2vec-weaviate"),
            distance=None,
            index=index,
        )

    drafted: list[VectorPlan]
    if vectors == "keep":
        drafted = [kept("default", True)]
    elif source.document_bytes_estimate <= 0:
        notes.append("no documents to embed; the source vector is kept")
        drafted = [kept("default", True)]
    elif vectors == "embed":
        drafted = [embedded()]
    else:
        drafted = [kept("source", False), embedded()]

    if has_no_vectors(source) and any(v.mode == "keep" for v in drafted):
        notes.append(NO_VECTORS_NOTE)
    return CollectionPlan(
        source=SourceRef(
            ref=source.ref,
            namespace=source.namespace,
            name=source.name,
            count=source.count,
            dimension=source.dimension,
            distance=source.distance,
            embedding=source.embedding,
        ),
        target=TargetPlan(
            name=name, vectors=drafted, properties=props, dropped=dropped
        ),
        notes=notes,
    )


def has_no_vectors(source: SourceRef | SourceCollection) -> bool:
    """Whether the source holds records but no vector.

    Chroma sets the dimension of a collection with its first vector.
    """
    return source.count > 0 and source.dimension is None


def draft_plan(
    source: SourcePlan,
    collections: list[SourceCollection],
    vectors: Literal["keep", "embed", "both"] = "keep",
    vectorizer: VectorizerPlan | None = None,
    index: IndexPlan | None = None,
    vectorizer_for: VectorizerMapper | None = None,
    source_notes: Mapping[str, list[str]] | None = None,
) -> Plan:
    """A plan for `collections`. `source_notes` holds notes by source ref."""
    taken: set[str] = set()
    drafted = [
        _draft_collection(
            c,
            taken,
            vectors,
            vectorizer,
            index or IndexPlan(),
            vectorizer_for,
            (source_notes or {}).get(c.ref, []),
        )
        for c in collections
    ]
    return Plan(source=source, collections=drafted)


class _PlanLoader(yaml.SafeLoader):
    """A safe loader that refuses a key that appears twice in one mapping.

    Keys that a merge key (`<<`) brings in may be repeated, as YAML allows.
    """

    def construct_mapping(
        self, node: yaml.MappingNode, deep: bool = False
    ) -> dict[Hashable, Any]:
        first: dict[Any, yaml.Node] = {}
        for key_node, _ in node.value:
            if not isinstance(key_node, yaml.ScalarNode) or key_node.tag == MERGE_TAG:
                continue
            key = self.construct_object(key_node)
            if key in first:
                raise yaml.constructor.ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    f"found duplicate key {key!r}, first at line "
                    f"{first[key].start_mark.line + 1}",
                    key_node.start_mark,
                )
            first[key] = key_node
        return super().construct_mapping(node, deep)


def load_plan(path: Path) -> Plan:
    text = path.read_text(encoding="utf-8")
    return Plan.model_validate(yaml.load(text, Loader=_PlanLoader))


def save_plan(plan: Plan, path: Path) -> None:
    path.write_text(
        yaml.safe_dump(
            plan.model_dump(mode="json"), sort_keys=False, allow_unicode=True
        ),
        encoding="utf-8",
    )
