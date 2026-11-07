from collections import Counter
from collections.abc import Iterator, Mapping
from dataclasses import dataclass

from to_weaviate.plan import (
    CollectionPlan,
    Plan,
    PropertyPlan,
    VectorizerMapper,
    VectorizerPlan,
)
from to_weaviate.sources.base import SourceCollection
from to_weaviate.target.schema import (
    DISTANCES,
    PROPERTY_TYPES,
    USERINFO_TEXT,
    PlanMismatch,
    collection_kwargs,
    module_names,
)
from to_weaviate.target.writer import unmapped_key_text

# Vectorizer module to the Weaviate header that carries its provider key and
# the environment variable that `weaviate_settings` reads it from.
PROVIDER_KEYS: dict[str, tuple[str, str]] = {
    "text2vec-openai": ("X-OpenAI-Api-Key", "OPENAI_API_KEY"),
    "text2vec-cohere": ("X-Cohere-Api-Key", "COHERE_API_KEY"),
    "text2vec-jinaai": ("X-JinaAI-Api-Key", "JINAAI_API_KEY"),
    "text2vec-voyageai": ("X-Voyage-Api-Key", "VOYAGE_API_KEY"),
    "text2vec-mistral": ("X-Mistral-Api-Key", "MISTRAL_API_KEY"),
    "text2vec-huggingface": ("X-HuggingFace-Api-Key", "HUGGINGFACE_API_KEY"),
}
GOOGLE_VERTEX_KEY = ("X-Goog-Vertex-Api-Key", "VERTEX_API_KEY")
GOOGLE_STUDIO_KEY = ("X-Goog-Studio-Api-Key", "STUDIO_API_KEY")
ID_KEY = "#id"
ID_PROPERTY = "source_id"

VECTOR_MODE_DETAIL = (
    "No Weaviate vectorizer matches the source embedding function. Either "
    "supply query vectors from the application, or run inspect with "
    "--vectors embed to re-vectorize the documents in Weaviate."
)


@dataclass(frozen=True)
class Finding:
    code: str
    collection: str  # target collection name
    message: str


@dataclass(frozen=True)
class Decision:
    collection: str  # source collection name
    kind: str
    detail: str


def target_groups(entries: list[CollectionPlan]) -> dict[str, list[CollectionPlan]]:
    """Entries by target name, both in plan order."""
    groups: dict[str, list[CollectionPlan]] = {}
    for entry in entries:
        groups.setdefault(entry.target.name, []).append(entry)
    return groups


def collection_key(name: str) -> str:
    """The name Weaviate stores for collection `name`: first letter capitalized."""
    return name[:1].upper() + name[1:]


def _describe(entry: CollectionPlan) -> str:
    tenant = f" (tenant {entry.target.tenant})" if entry.target.tenant else ""
    return f"{entry.source.name} -> {entry.target.name}{tenant}"


def _duplicate_findings(entries: list[CollectionPlan]) -> list[Finding]:
    """Entries that write into the same collection and tenant.

    Entries share a place when their collection keys are equal and either
    their tenants are equal (two None tenants are equal) or one of them has a
    tenant and another has none.
    """
    by_key: dict[str, list[CollectionPlan]] = {}
    for entry in entries:
        by_key.setdefault(collection_key(entry.target.name), []).append(entry)
    out: list[Finding] = []
    for key, group in by_key.items():
        tenants = [e.target.tenant for e in group]
        if len(group) > 1 and (None in tenants or len(set(tenants)) < len(tenants)):
            out.append(
                Finding(
                    "duplicate_target",
                    key,
                    f"entries {', '.join(_describe(e) for e in group)} write "
                    f"into one place in collection {key}; give each entry its "
                    "own tenant or its own target name",
                )
            )
    return out


def _duplicate_source_findings(entries: list[CollectionPlan]) -> list[Finding]:
    """Entries that copy one source ref: one finding per target collection.

    The state records one target per source ref, so a plan keeps one entry per
    ref. A finding for each target lets `copy --collection` refuse any of them.
    """
    by_ref: dict[str, list[CollectionPlan]] = {}
    for entry in entries:
        by_ref.setdefault(entry.source.ref, []).append(entry)
    out: list[Finding] = []
    for group in by_ref.values():
        if len(group) < 2:
            continue
        message = (
            f"entries {', '.join(_describe(e) for e in group)} copy one source "
            "collection; the state records one target per source collection, so "
            "keep one of these entries"
        )
        for key in dict.fromkeys(collection_key(e.target.name) for e in group):
            out.append(Finding("duplicate_source", key, message))
    return out


def _name_findings(entry: CollectionPlan) -> list[Finding]:
    """Property names that are equal ignoring case, and repeated vector names.

    Weaviate compares property names in lower case.
    """
    target = entry.target.name
    by_lower: dict[str, list[str]] = {}
    for prop in entry.target.properties:
        by_lower.setdefault(prop.name.lower(), []).append(prop.name)
    out = [
        Finding(
            "duplicate_property",
            target,
            f"properties {', '.join(names)} have the same name ignoring case; "
            "give each property its own name",
        )
        for names in by_lower.values()
        if len(names) > 1
    ]
    out.extend(
        Finding(
            "duplicate_vector",
            target,
            f"vector name {name} is used {n} times; give each vector its own name",
        )
        for name, n in Counter(v.name for v in entry.target.vectors).items()
        if n > 1
    )
    return out


def _id_findings(entry: CollectionPlan) -> list[Finding]:
    """The source ID needs one property, named source_id.

    The writer, the reader and verify find the source ID of an object in the
    property source_id.
    """
    target = entry.target.name
    properties = entry.target.properties
    ids = [p.name for p in properties if p.source_key == ID_KEY]
    out: list[Finding] = []
    if not ids:
        out.append(
            Finding(
                "missing_id_property",
                target,
                f"no property has source_key {ID_KEY}; add the property "
                f"{ID_PROPERTY} with source_key {ID_KEY}",
            )
        )
    elif len(ids) > 1:
        out.append(
            Finding(
                "duplicate_id_property",
                target,
                f"properties {', '.join(ids)} have source_key {ID_KEY}; keep one "
                f"property for {ID_KEY} and name it {ID_PROPERTY}",
            )
        )
    elif ids[0] != ID_PROPERTY:
        out.append(
            Finding(
                "missing_id_property",
                target,
                f"property {ids[0]} has source_key {ID_KEY}; name it {ID_PROPERTY}",
            )
        )
    out.extend(
        Finding(
            "duplicate_id_property",
            target,
            f"property {ID_PROPERTY} has source_key {p.source_key}; only the "
            f"property with source_key {ID_KEY} can be named {ID_PROPERTY}",
        )
        for p in properties
        if p.name == ID_PROPERTY and p.source_key != ID_KEY
    )
    return out


def load_findings(plan: Plan) -> list[Finding]:
    """Errors that refuse the plan in every command that loads it."""
    return [
        f
        for entry in plan.collections
        for f in (*_name_findings(entry), *_id_findings(entry))
    ]


def _vectorizers(plan: Plan) -> Iterator[tuple[CollectionPlan, VectorizerPlan]]:
    for entry in plan.collections:
        for vector in entry.target.vectors:
            if vector.vectorizer is not None:
                yield entry, vector.vectorizer


def _type_counts(
    entry: CollectionPlan,
    key: str,
    inspected: Mapping[str, SourceCollection] | None,
) -> str | None:
    collection = inspected.get(entry.source.ref) if inspected else None
    for field in collection.fields if collection else []:
        if field.key == key:
            return ", ".join(f"{t}: {n}" for t, n in sorted(field.types.items()))
    return None


def _coerced(
    entry: CollectionPlan,
    prop: PropertyPlan,
    inspected: Mapping[str, SourceCollection] | None,
) -> str:
    counts = _type_counts(entry, prop.source_key, inspected)
    if counts is None:
        return (
            f"key {prop.source_key} also has values of type "
            f"{', '.join(prop.coerced_from)}; values are stored as {prop.type}"
        )
    return (
        f"key {prop.source_key} has the types {counts}; "
        f"values are stored as {prop.type}"
    )


def decisions(plan: Plan, inspected: Mapping[str, SourceCollection]) -> list[Decision]:
    """Choices in a drafted plan that the user should review."""
    out: list[Decision] = []
    for entry in plan.collections:
        name = entry.source.name
        vectors = entry.target.vectors
        embedded = any(v.mode == "embed" for v in vectors)
        if not embedded and any(
            v.mode == "keep" and v.vectorizer is None for v in vectors
        ):
            out.append(Decision(name, "vector_mode", VECTOR_MODE_DETAIL))
        for prop in entry.target.properties:
            if prop.coerced_from:
                out.append(
                    Decision(name, "mixed_types", _coerced(entry, prop, inspected))
                )
        for dropped in entry.target.dropped:
            out.append(
                Decision(
                    name,
                    "dropped_sparse",
                    f"key {dropped.source_key} is dropped: {dropped.reason}",
                )
            )
    return out


def _model(vectorizer: VectorizerPlan) -> str:
    model = vectorizer.options.get("model")
    return vectorizer.module if model is None else f"{vectorizer.module} {model}"


def _keep_model_findings(
    entry: CollectionPlan, vectorizer_for: VectorizerMapper
) -> list[Finding]:
    """A kept vector whose query vectorizer differs from the source model.

    When the source model is unknown, the plan's choice is accepted. When the
    mapper gives no `model` option, only the modules are compared.
    """
    source, _ = vectorizer_for(entry.source.embedding)
    if source is None:
        return []
    out: list[Finding] = []
    for vector in entry.target.vectors:
        planned = vector.vectorizer
        if vector.mode != "keep" or planned is None:
            continue
        source_model = source.options.get("model")
        if planned.module == source.module and (
            source_model is None or planned.options.get("model") == source_model
        ):
            continue
        out.append(
            Finding(
                "keep_model_mismatch",
                entry.target.name,
                f"vector {vector.name} keeps vectors made by {_model(source)}, "
                f"but queries would be vectorized with {_model(planned)}; use "
                "embed mode to re-vectorize the documents, or set the "
                "vectorizer to the source model",
            )
        )
    return out


def _unmapped_findings(
    entry: CollectionPlan, inspected: Mapping[str, SourceCollection] | None
) -> list[Finding]:
    """Inspected metadata keys that the entry neither maps nor drops."""
    collection = inspected.get(entry.source.ref) if inspected else None
    if collection is None:
        return []
    known = {p.source_key for p in entry.target.properties}
    known.update(d.source_key for d in entry.target.dropped)
    keys = [f.key for f in collection.fields] + collection.sparse_keys
    return [
        Finding("unmapped_key", entry.target.name, unmapped_key_text(key))
        for key in keys
        if key not in known
    ]


def _value_findings(entry: CollectionPlan) -> list[Finding]:
    """Property types and distances that the schema mapping does not have."""
    name = entry.target.name
    out = [
        Finding(
            "invalid_type",
            name,
            f"property {prop.name}: type {prop.type} is not one of "
            f"{', '.join(PROPERTY_TYPES)}",
        )
        for prop in entry.target.properties
        if prop.type not in PROPERTY_TYPES
    ]
    out.extend(
        Finding(
            "invalid_distance",
            name,
            f"vector {vector.name}: distance {vector.distance} is not one of "
            f"{', '.join(DISTANCES)}",
        )
        for vector in entry.target.vectors
        if vector.distance is not None and vector.distance not in DISTANCES
    )
    return out


def vectorizer_findings(groups: Mapping[str, list[CollectionPlan]]) -> list[Finding]:
    """Vectorizer settings that the client cannot turn into a collection."""
    out: list[Finding] = []
    for name, group in groups.items():
        try:
            collection_kwargs(group)
        except PlanMismatch as exc:
            out.append(Finding("invalid_vectorizer", name, str(exc)))
    return out


def userinfo_only(errors: list[Finding]) -> bool:
    """Whether every one of `errors` is a URL option that keeps its userinfo."""
    return bool(errors) and all(
        e.code == "invalid_vectorizer" and e.message.endswith(USERINFO_TEXT)
        for e in errors
    )


def plan_findings(
    plan: Plan,
    inspected: Mapping[str, SourceCollection] | None,
    vectorizer_for: VectorizerMapper,
) -> tuple[list[Finding], list[Finding]]:
    """Errors and warnings that need no connection to the target."""
    errors: list[Finding] = []
    warnings: list[Finding] = []
    errors.extend(_duplicate_findings(plan.collections))
    errors.extend(_duplicate_source_findings(plan.collections))
    value_errors = [f for e in plan.collections for f in _value_findings(e)]
    errors.extend(value_errors)
    # collection_kwargs cannot map a group that has a value error.
    unmapped = {f.collection for f in value_errors}
    groups = target_groups(plan.collections)
    errors.extend(
        vectorizer_findings({n: g for n, g in groups.items() if n not in unmapped})
    )
    for entry in plan.collections:
        errors.extend(_keep_model_findings(entry, vectorizer_for))
        errors.extend(_unmapped_findings(entry, inspected))
        for prop in entry.target.properties:
            if prop.coerced_from:
                warnings.append(
                    Finding(
                        "coerced_keys",
                        entry.target.name,
                        _coerced(entry, prop, inspected),
                    )
                )
        for dropped in entry.target.dropped:
            warnings.append(
                Finding(
                    "dropped_sparse",
                    entry.target.name,
                    f"key {dropped.source_key} is dropped: {dropped.reason}",
                )
            )
    return errors, warnings


def _provider_key(vectorizer: VectorizerPlan) -> tuple[str, str] | None:
    if vectorizer.module == "text2vec-google":
        if "project_id" in vectorizer.options:
            return GOOGLE_VERTEX_KEY
        return GOOGLE_STUDIO_KEY
    return PROVIDER_KEYS.get(vectorizer.module)


def provider_keys(plan: Plan) -> list[tuple[str, str]]:
    """(header, environment variable) of each provider key the plan needs."""
    found: dict[tuple[str, str], None] = {}
    for _, vectorizer in _vectorizers(plan):
        key = _provider_key(vectorizer)
        if key is not None:
            found.setdefault(key)
    return list(found)


def module_findings(plan: Plan, enabled: set[str]) -> list[Finding]:
    """Plan vectorizer modules that the target does not have enabled."""
    listed = ", ".join(sorted(enabled)) or "none"
    seen: dict[tuple[str, str], None] = {}
    for entry, vectorizer in _vectorizers(plan):
        if not module_names(vectorizer.module) & enabled:
            seen.setdefault((entry.target.name, vectorizer.module))
    return [
        Finding(
            "module_not_enabled",
            name,
            f"module {module} is not enabled on the target; "
            f"enabled modules: {listed}",
        )
        for name, module in seen
    ]
