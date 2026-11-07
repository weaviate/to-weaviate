from typing import Any

import pytest

from to_weaviate import plan as p
from to_weaviate.engine.check import (
    Finding,
    decisions,
    load_findings,
    module_findings,
    plan_findings,
    provider_keys,
)
from to_weaviate.sources.chroma.mapping import vectorizer_for


def entry(
    vectors: list[p.VectorPlan],
    embedding: dict[str, Any] | None = None,
    name: str = "Docs",
) -> p.CollectionPlan:
    return p.CollectionPlan(
        source=p.SourceRef(
            ref="r",
            namespace="ns",
            name="docs",
            count=1,
            dimension=2,
            distance="cosine",
            embedding=embedding,
        ),
        target=p.TargetPlan(name=name, vectors=vectors, properties=[]),
    )


def plan_of(*entries: p.CollectionPlan) -> p.Plan:
    return p.Plan(
        source=p.SourcePlan(type="chroma", url="u", tenant=None),
        collections=list(entries),
    )


def embed(module: str, **options: Any) -> p.VectorPlan:
    vectorizer = p.VectorizerPlan(module=module, options=options)
    return p.VectorPlan(mode="embed", vectorizer=vectorizer)


def keep(vectorizer: p.VectorizerPlan | None = None) -> p.VectorPlan:
    return p.VectorPlan(mode="keep", vectorizer=vectorizer, distance="cosine")


@pytest.mark.parametrize(
    ("options", "expected"),
    [
        ({"project_id": "proj"}, [("X-Goog-Vertex-Api-Key", "VERTEX_API_KEY")]),
        ({}, [("X-Goog-Studio-Api-Key", "STUDIO_API_KEY")]),
    ],
)
def test_provider_keys_for_google_depend_on_project_id(options, expected):
    assert provider_keys(plan_of(entry([embed("text2vec-google", **options)]))) == (
        expected
    )


def test_provider_keys_skip_modules_without_a_key_and_repeats():
    plan = plan_of(
        entry([embed("text2vec-weaviate")]),
        entry([embed("text2vec-cohere")], name="B"),
        entry([embed("text2vec-cohere")], name="C"),
    )
    assert provider_keys(plan) == [("X-Cohere-Api-Key", "COHERE_API_KEY")]


def voyage_mapper(
    embedding: dict[str, Any] | None,
) -> tuple[p.VectorizerPlan | None, list[str]]:
    return p.VectorizerPlan(module="text2vec-voyageai"), []


@pytest.mark.parametrize(
    ("vectorizer", "codes"),
    [
        (p.VectorizerPlan(module="text2vec-voyageai", options={"model": "v3"}), []),
        (p.VectorizerPlan(module="text2vec-openai"), ["keep_model_mismatch"]),
    ],
)
def test_keep_model_compares_modules_when_the_source_model_is_unknown(
    vectorizer, codes
):
    errors, _ = plan_findings(plan_of(entry([keep(vectorizer)])), None, voyage_mapper)
    assert [e.code for e in errors] == codes


@pytest.mark.parametrize(
    ("embedding", "old_vectorizer"),
    [
        (
            {"type": "known", "name": "voyageai", "config": {"model_name": "v3l"}},
            p.VectorizerPlan(module="text2vec-voyageai"),
        ),
        (
            {
                "type": "known",
                "name": "google_vertex",
                "config": {"model_name": "m", "project_id": "p", "region": "r"},
            },
            p.VectorizerPlan(module="text2vec-google", options={"project_id": "p"}),
        ),
    ],
)
def test_keep_model_mismatch_for_a_plan_drafted_without_the_model(
    embedding, old_vectorizer
):
    plan = plan_of(entry([keep(old_vectorizer)], embedding=embedding))
    errors, _ = plan_findings(plan, None, vectorizer_for)
    assert [e.code for e in errors] == ["keep_model_mismatch"]


def test_module_findings_accept_the_legacy_google_name():
    plan = plan_of(entry([embed("text2vec-google")]))
    assert module_findings(plan, {"text2vec-palm"}) == []
    assert module_findings(plan, {"text2vec-openai"}) == [
        Finding(
            "module_not_enabled",
            "Docs",
            "module text2vec-google is not enabled on the target; "
            "enabled modules: text2vec-openai",
        )
    ]


def test_vector_mode_is_not_a_decision_when_the_documents_are_embedded():
    both = entry([p.VectorPlan(name="source", mode="keep"), embed("text2vec-weaviate")])
    assert decisions(plan_of(both), {}) == []
    assert [d.kind for d in decisions(plan_of(entry([keep()])), {})] == ["vector_mode"]


def targeted(ref: str, name: str, tenant: str | None) -> p.CollectionPlan:
    base = entry([keep()], name=name)
    return base.model_copy(
        update={
            "source": base.source.model_copy(update={"ref": ref, "name": ref}),
            "target": base.target.model_copy(update={"tenant": tenant}),
        }
    )


@pytest.mark.parametrize(
    ("a", "b", "duplicate"),
    [
        (("Docs", None), ("Docs", None), True),
        (("Docs", "t"), ("Docs", "t"), True),
        (("Docs", "t"), ("Docs", None), True),
        (("docs", None), ("Docs", None), True),
        (("docs", "t"), ("Docs", "t"), True),
        (("Docs", "t"), ("Docs", "u"), False),
        (("docs", "t"), ("Docs", "u"), False),
        (("Docs", None), ("Other", None), False),
        (("docs", None), ("dOcs", None), False),
    ],
)
def test_duplicate_target_compares_the_name_weaviate_stores(a, b, duplicate):
    plan = plan_of(targeted("r1", *a), targeted("r2", *b))
    errors, _ = plan_findings(plan, None, voyage_mapper)
    found = [(e.code, e.collection) for e in errors]
    assert found == ([("duplicate_target", "Docs")] if duplicate else [])


def test_duplicate_source_names_each_target_collection_of_the_ref():
    plan = plan_of(
        targeted("r1", "Docs", "a"),
        targeted("r1", "docs", "b"),
        targeted("r1", "Other", None),
        targeted("r2", "Third", None),
    )
    errors, _ = plan_findings(plan, None, voyage_mapper)
    assert [(e.code, e.collection) for e in errors] == [
        ("duplicate_source", "Docs"),
        ("duplicate_source", "Other"),
    ]
    assert "r1 -> Docs (tenant a), r1 -> docs (tenant b), r1 -> Other" in (
        errors[0].message
    )


def test_a_url_option_that_keeps_its_userinfo_is_an_error():
    base_url = "https://bob:made/up@proxy.example.com/v1"
    plan = plan_of(entry([embed("text2vec-openai", base_url=base_url)]))
    errors, warnings = plan_findings(plan, None, vectorizer_for)
    assert errors == [
        Finding(
            "invalid_vectorizer",
            "Docs",
            "vectorizer option base_url of text2vec-openai in Docs holds userinfo "
            "that cannot be removed",
        )
    ]
    assert warnings == []
    clean = plan_of(entry([embed("text2vec-openai", base_url="https://u:p@h/v1")]))
    assert plan_findings(clean, None, vectorizer_for) == ([], [])


ID_PROPERTY = p.PropertyPlan(source_key="#id", name="source_id", type="text")


def with_target(e: p.CollectionPlan, **update: Any) -> p.CollectionPlan:
    return e.model_copy(update={"target": e.target.model_copy(update=update)})


def prop(name: str, key: str | None = None) -> p.PropertyPlan:
    return p.PropertyPlan(source_key=key or name, name=name, type="text")


def test_load_findings_refuse_property_names_equal_ignoring_case():
    properties = [ID_PROPERTY, prop("year"), prop("Year"), prop("title")]
    properties.append(prop("YEAR", "y2"))
    plan = plan_of(with_target(entry([keep()]), properties=properties))
    assert load_findings(plan) == [
        Finding(
            "duplicate_property",
            "Docs",
            "properties year, Year, YEAR have the same name ignoring case; give "
            "each property its own name",
        )
    ]


def test_load_findings_refuse_a_repeated_vector_name():
    both = entry([keep(), embed("text2vec-weaviate")])
    plan = plan_of(with_target(both, properties=[ID_PROPERTY]))
    assert load_findings(plan) == [
        Finding(
            "duplicate_vector",
            "Docs",
            "vector name default is used 2 times; give each vector its own name",
        )
    ]


@pytest.mark.parametrize(
    ("properties", "expected"),
    [
        ([ID_PROPERTY, prop("year")], []),
        (
            [prop("year")],
            [
                (
                    "missing_id_property",
                    "no property has source_key #id; add the property source_id "
                    "with source_key #id",
                )
            ],
        ),
        (
            [ID_PROPERTY, prop("chroma_id", "#id")],
            [
                (
                    "duplicate_id_property",
                    "properties source_id, chroma_id have source_key #id; keep one "
                    "property for #id and name it source_id",
                )
            ],
        ),
        (
            [prop("chroma_id", "#id"), prop("source_id", "year")],
            [
                (
                    "missing_id_property",
                    "property chroma_id has source_key #id; name it source_id",
                ),
                (
                    "duplicate_id_property",
                    "property source_id has source_key year; only the property with "
                    "source_key #id can be named source_id",
                ),
            ],
        ),
    ],
)
def test_load_findings_need_one_id_property_named_source_id(properties, expected):
    plan = plan_of(with_target(entry([keep()]), properties=properties))
    assert [(f.code, f.message) for f in load_findings(plan)] == expected
