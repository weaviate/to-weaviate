from typing import Any

import pytest

from to_weaviate import plan as p
from to_weaviate.engine.check import (
    Finding,
    decisions,
    module_findings,
    plan_findings,
    provider_keys,
)


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
