import io

import pytest
from pydantic import ValidationError

from to_weaviate.plan import (
    CollectionPlan,
    IndexPlan,
    Plan,
    QuantizerPlan,
    SourcePlan,
    SourceRef,
    TargetPlan,
    VectorizerPlan,
    VectorPlan,
    draft_plan,
    load_plan,
    save_plan,
)
from to_weaviate.sources.base import FieldStats, SourceCollection

SRC = SourcePlan(type="chroma", url="http://localhost:8000", tenant=None)


def make(
    name="docs", fields=None, sparse=None, distance="cosine", doc_bytes=1000, **kw
):
    return SourceCollection(
        ref=kw.get("ref", f"default/{name}"),
        namespace="default",
        name=name,
        count=10,
        dimension=3,
        distance=distance,
        embedding=kw.get("embedding"),
        fields=fields or [],
        sparse_keys=sparse or [],
        document_bytes_estimate=doc_bytes,
    )


@pytest.fixture
def collection_with_fields():
    return make(
        fields=[
            FieldStats("category", {"text": 10}),
            FieldStats("year", {"int": 10}),
            FieldStats("tags", {"text[]": 10}),
        ],
        sparse=["sparse"],
    )


def target(plan):
    return plan.collections[0].target


def test_draft_properties(collection_with_fields):
    t = target(draft_plan(SRC, [collection_with_fields], "keep"))
    by = {p.name: p for p in t.properties}
    assert [p.name for p in t.properties][:3] == ["document", "source_id", "uri"]
    assert (by["document"].tokenization, by["document"].searchable) == ("word", True)
    assert by["document"].filterable is False
    assert (
        by["source_id"].tokenization == "field"
        and by["category"].tokenization == "field"
    )
    assert by["uri"].source_key == "#uri" and by["uri"].tokenization == "field"
    assert by["year"].range_filters is True and by["tags"].type == "text[]"
    assert by["category"].searchable is False and by["category"].filterable is True
    assert [d.source_key for d in t.dropped] == ["sparse"]
    assert t.dropped[0].reason == "sparse vectors are not migrated"
    assert [(v.name, v.mode) for v in t.vectors] == [("default", "keep")]


def test_draft_property_names_avoid_reserved_ones():
    c = make(
        fields=[FieldStats("document", {"text": 1}), FieldStats("my key", {"int": 1})]
    )
    names = [p.name for p in target(draft_plan(SRC, [c], "keep")).properties]
    assert names == ["document", "source_id", "uri", "document_meta", "my_key"]


def test_draft_mixed_type_becomes_text():
    c = make(fields=[FieldStats("year", {"int": 600, "text": 50})])
    p = {p.name: p for p in target(draft_plan(SRC, [c], "keep")).properties}["year"]
    assert p.type == "text" and p.coerced_from == ["int"]
    assert p.tokenization == "field" and p.range_filters is False


def test_draft_unique_collection_names():
    a, b = make("my-docs", ref="a"), make("my_docs", ref="b")
    names = [c.target.name for c in draft_plan(SRC, [a, b], "keep").collections]
    assert names == ["My_docs", "My_docs_2"]


def test_draft_source_ref():
    c = make(embedding={"type": "known", "name": "x", "config": {}})
    ref = draft_plan(SRC, [c], "keep").collections[0].source
    assert ref == SourceRef(
        ref="default/docs",
        namespace="default",
        name="docs",
        count=10,
        dimension=3,
        distance="cosine",
        embedding=c.embedding,
    )


def test_draft_embed_defaults_to_weaviate_embeddings():
    v = target(draft_plan(SRC, [make()], "embed")).vectors[0]
    assert v.mode == "embed" and v.vectorizer is not None
    assert v.vectorizer.module == "text2vec-weaviate" and v.distance is None


def test_draft_embed_notes_the_source_distance():
    plan = draft_plan(SRC, [make(distance="l2-squared")], "embed")
    assert any("l2-squared" in n for n in plan.collections[0].notes)


def test_draft_embed_with_chosen_vectorizer():
    v = VectorizerPlan(
        module="text2vec-openai", options={"model": "text-embedding-3-large"}
    )
    assert target(draft_plan(SRC, [make()], "embed", v)).vectors[0].vectorizer == v


def test_draft_embed_without_documents_stays_on_keep():
    plan = draft_plan(SRC, [make(doc_bytes=0)], "embed")
    v = target(plan).vectors[0]
    assert (v.name, v.mode) == ("default", "keep")
    assert any("no documents" in n for n in plan.collections[0].notes)


def test_draft_both_without_documents_has_one_default_vector():
    vs = target(draft_plan(SRC, [make(doc_bytes=0)], "both")).vectors
    assert [(x.name, x.mode) for x in vs] == [("default", "keep")]


def test_draft_both_keeps_the_source_vector_and_adds_a_new_one():
    vs = target(draft_plan(SRC, [make()], "both")).vectors
    assert [(x.name, x.mode) for x in vs] == [("source", "keep"), ("default", "embed")]
    assert vs[0].vectorizer is None and vs[0].distance == "cosine"
    assert vs[1].distance is None


def test_draft_both_puts_the_chosen_vectorizer_on_the_new_vector_only():
    chosen = VectorizerPlan(
        module="text2vec-openai", options={"model": "text-embedding-3-large"}
    )
    plan = draft_plan(
        SRC,
        [make(embedding={"name": "x"})],
        "both",
        chosen,
        vectorizer_for=lambda e: (VectorizerPlan(module="m"), []),
    )
    vs = target(plan).vectors
    assert [(x.name, x.mode) for x in vs] == [("source", "keep"), ("default", "embed")]
    assert vs[0].vectorizer is None
    assert vs[1].vectorizer == chosen


def test_draft_keep_uses_the_mapper_and_collects_notes():
    def mapper(embedding):
        return VectorizerPlan(module="text2vec-x"), ["mapped"]

    c = make(embedding={"name": "x"})
    plan = draft_plan(SRC, [c], "keep", vectorizer_for=mapper)
    assert target(plan).vectors[0].vectorizer == VectorizerPlan(module="text2vec-x")
    assert "mapped" in plan.collections[0].notes


def test_draft_keep_vectorizer_argument_wins_over_the_mapper():
    chosen = VectorizerPlan(module="text2vec-openai")
    plan = draft_plan(
        SRC,
        [make()],
        "keep",
        chosen,
        vectorizer_for=lambda e: (VectorizerPlan(module="m"), []),
    )
    assert target(plan).vectors[0].vectorizer == chosen


@pytest.mark.parametrize("vectors", ["keep", "embed", "both"])
def test_draft_puts_the_source_notes_first(vectors):
    c, other = make(), make("other", ref="r2")
    plan = draft_plan(
        SRC, [c, other], vectors, source_notes={c.ref: ["from the source"]}
    )
    assert plan.collections[0].notes[0] == "from the source"
    assert "from the source" not in plan.collections[1].notes


def test_load_accepts_a_plan_with_the_full_embedding_config(tmp_path):
    full = {
        "type": "known",
        "name": "openai",
        "config": {
            "api_key_env_var": "OPENAI_API_KEY",
            "model_name": "m",
            "api_base": "https://bob:made-up-pw@proxy.example.com/v1",
            "default_headers": {"X-Custom": "made-up-header"},
            "organization_id": None,
        },
    }
    plan = draft_plan(SRC, [make(embedding=full)], "keep")
    save_plan(plan, tmp_path / "plan.yaml")
    assert load_plan(tmp_path / "plan.yaml").collections[0].source.embedding == full


def test_draft_keep_without_mapper_has_no_vectorizer():
    assert target(draft_plan(SRC, [make()], "keep")).vectors[0].vectorizer is None


def test_plan_refuses_two_kept_vectors():
    with pytest.raises(ValidationError):
        TargetPlan(
            name="A",
            vectors=[VectorPlan(name="a"), VectorPlan(name="b")],
            properties=[],
        )


def test_plan_refuses_unknown_fields():
    with pytest.raises(ValidationError):
        SourcePlan(type="chroma", url="u", tenant=None, bogus=1)


def test_rq_bits():
    assert QuantizerPlan(type="rq", bits=4).bits == 4
    with pytest.raises(ValidationError):
        QuantizerPlan(type="rq", bits=2)


def test_quantizer_refuses_a_field_of_another_type():
    with pytest.raises(ValidationError) as e:
        QuantizerPlan(type="pq", bits=8)
    assert "bits" in str(e.value) and "pq" in str(e.value)


def test_quantizer_refuses_unknown_fields():
    with pytest.raises(ValidationError):
        QuantizerPlan(type="rq", bogus=1)


def test_quantizer_needs_a_named_index_type():
    with pytest.raises(ValidationError):
        IndexPlan(type="default", quantizer=QuantizerPlan(type="rq"))


def test_index_refuses_an_unknown_option():
    with pytest.raises(ValidationError) as e:
        IndexPlan(type="hnsw", options={"search_probe": 4})
    assert "search_probe" in str(e.value)


def test_index_accepts_a_listed_option():
    assert IndexPlan(type="hfresh", options={"search_probe": 4}).options == {
        "search_probe": 4
    }


def test_default_index_refused_for_non_cosine_distance():
    with pytest.raises(ValidationError):
        VectorPlan(distance="l2-squared", index=IndexPlan())
    VectorPlan(distance="cosine", index=IndexPlan())
    VectorPlan(distance=None, index=IndexPlan())


def test_draft_names_hnsw_when_distance_is_not_cosine():
    plan = draft_plan(SRC, [make(distance="l2-squared")], "keep")
    assert target(plan).vectors[0].index.type == "hnsw"
    assert any("l2-squared" in n for n in plan.collections[0].notes)


def test_draft_keeps_the_chosen_index_type_when_distance_is_not_cosine():
    ix = IndexPlan(type="flat")
    plan = draft_plan(SRC, [make(distance="dot")], "keep", index=ix)
    assert target(plan).vectors[0].index == ix


def test_draft_applies_index_to_every_vector():
    ix = IndexPlan(type="hnsw", quantizer=QuantizerPlan(type="rq", bits=8))
    vs = target(draft_plan(SRC, [make()], "both", index=ix)).vectors
    assert len(vs) == 2 and all(v.index == ix for v in vs)


def test_round_trip(tmp_path, collection_with_fields):
    plan = draft_plan(SRC, [collection_with_fields, make("b", distance="dot")], "both")
    save_plan(plan, tmp_path / "plan.yaml")
    assert load_plan(tmp_path / "plan.yaml") == plan


def test_plan_is_written_and_read_as_utf8_in_any_locale(tmp_path, monkeypatch):
    # pathlib asks io.text_encoding for the locale encoding when none is given.
    monkeypatch.setattr(
        io, "text_encoding", lambda encoding, stacklevel=2: encoding or "latin-1"
    )
    plan = draft_plan(SRC, [make(fields=[FieldStats("名前 año", {"text": 1})])])
    save_plan(plan, tmp_path / "plan.yaml")
    assert "名前 año" in (tmp_path / "plan.yaml").read_bytes().decode("utf-8")
    assert load_plan(tmp_path / "plan.yaml") == plan


def test_save_keeps_model_key_order(tmp_path):
    plan = Plan(source=SRC, collections=[])
    save_plan(plan, tmp_path / "p.yaml")
    assert (tmp_path / "p.yaml").read_text().splitlines()[0] == "version: 1"


def test_load_refuses_a_typo(tmp_path):
    plan = draft_plan(SRC, [make()], "keep")
    save_plan(plan, tmp_path / "p.yaml")
    text = (
        (tmp_path / "p.yaml").read_text().replace("tokenization:", "tokenisation:", 1)
    )
    (tmp_path / "p.yaml").write_text(text)
    with pytest.raises(ValidationError):
        load_plan(tmp_path / "p.yaml")


def test_models_are_exported():
    assert CollectionPlan and TargetPlan
