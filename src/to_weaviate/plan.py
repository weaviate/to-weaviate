from collections.abc import Callable, Hashable, Mapping
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, model_validator

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

KEEP_DISTANCES_WITHOUT_INDEX = (None, "cosine")
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
        if self.quantizer.type != "default" and self.type == "default":
            raise ValueError(
                f"quantizer type {self.quantizer.type!r} needs a named index "
                "type, not 'default'"
            )
        return self


class VectorPlan(_Model):
    name: str = "default"
    mode: Literal["keep", "embed"] = "keep"
    vectorizer: VectorizerPlan | None = None
    distance: str | None = None
    index: IndexPlan = IndexPlan()

    @model_validator(mode="after")
    def _distance_needs_named_index(self) -> "VectorPlan":
        if (
            self.index.type == "default"
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
    vectors: list[VectorPlan]
    properties: list[PropertyPlan]
    dropped: list[DroppedKey] = []

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
