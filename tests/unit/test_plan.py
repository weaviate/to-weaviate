import io
import re
from dataclasses import replace

import pytest
import yaml
from pydantic import ValidationError

from to_weaviate.plan import (
    NO_VECTORS_NOTE,
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


def test_draft_names_differ_ignoring_case_and_avoid_reserved_suffixes():
    keys = ["myKey", "mykey", "max_temp", "title_searchable"]
    fields = [FieldStats(key, {"text": 1}) for key in keys]
    sources = [make("my_docs", ref="a", fields=fields)]
    sources += [make(name, ref=name) for name in ("my_Docs", "MY_DOCS")]
    plan = draft_plan(SRC, sources, "keep")
    names = [c.target.name for c in plan.collections]
    assert names == ["My_docs", "My_Docs_2", "MY_DOCS_3"]
    properties = [p.name for p in target(plan).properties]
    assert properties[3:] == [
        "myKey",
        "mykey_2",
        "max_temp_meta",
        "title_searchable_meta",
    ]


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


# The quantizer types that Weaviate 1.40.4 creates for each named index type.
SUPPORTED_QUANTIZERS = {
    "hnsw": ["default", "none", "rq", "sq", "pq", "bq"],
    "flat": ["default", "none", "rq", "bq"],
    "dynamic": ["default", "none", "rq", "bq"],
    "hfresh": ["default", "rq"],
}
ALL_QUANTIZERS = ["default", "none", "rq", "sq", "pq", "bq"]


@pytest.mark.parametrize(
    ("index", "quantizer"),
    [(i, q) for i, qs in SUPPORTED_QUANTIZERS.items() for q in qs],
)
def test_index_accepts_the_quantizer_types_that_it_supports(index, quantizer):
    plan = IndexPlan(type=index, quantizer=QuantizerPlan(type=quantizer))
    assert plan.quantizer.type == quantizer


@pytest.mark.parametrize(
    ("index", "quantizer"),
    [
        (i, q)
        for i, qs in SUPPORTED_QUANTIZERS.items()
        for q in ALL_QUANTIZERS
        if q not in qs
    ],
)
def test_index_refuses_a_quantizer_type_that_it_does_not_support(index, quantizer):
    with pytest.raises(ValidationError) as e:
        IndexPlan(type=index, quantizer=QuantizerPlan(type=quantizer))
    supported = ", ".join(SUPPORTED_QUANTIZERS[index])
    assert (
        f"quantizer type {quantizer!r} is not supported by index type {index!r}; "
        f"it supports {supported}"
    ) in str(e.value)


# The rq bits that Weaviate 1.40.4 accepts for each named index type.
SUPPORTED_RQ_BITS = {
    "hnsw": [1, 4, 8],
    "flat": [1, 8],
    "dynamic": [1, 8],
    "hfresh": [1],
}


@pytest.mark.parametrize(
    ("index", "bits"),
    [(i, b) for i, bits in SUPPORTED_RQ_BITS.items() for b in bits],
)
def test_index_accepts_the_rq_bits_that_it_supports(index, bits):
    plan = IndexPlan(type=index, quantizer=QuantizerPlan(type="rq", bits=bits))
    assert plan.quantizer.bits == bits


@pytest.mark.parametrize(
    ("index", "bits"),
    [
        (i, b)
        for i, bits in SUPPORTED_RQ_BITS.items()
        for b in (1, 4, 8)
        if b not in bits
    ],
)
def test_index_refuses_rq_bits_that_it_does_not_support(index, bits):
    with pytest.raises(ValidationError) as e:
        IndexPlan(type=index, quantizer=QuantizerPlan(type="rq", bits=bits))
    supported = ", ".join(map(str, SUPPORTED_RQ_BITS[index]))
    assert (
        f"bits={bits} of quantizer type 'rq' is not supported by index type "
        f"{index!r}; it supports {supported}"
    ) in str(e.value)


def test_rq_centering_with_hnsw_and_4_bits_is_accepted():
    quantizer = QuantizerPlan(type="rq", bits=4, centering=True)
    assert IndexPlan(type="hnsw", quantizer=quantizer).quantizer.centering is True


# A server without a bits value uses 8 bits with hnsw.
@pytest.mark.parametrize(
    ("index", "bits"),
    [
        ("hnsw", None),
        ("hnsw", 1),
        ("hnsw", 8),
        ("flat", 8),
        ("dynamic", 1),
        ("hfresh", 1),
    ],
)
def test_rq_centering_needs_hnsw_and_4_bits(index, bits):
    with pytest.raises(ValidationError) as e:
        IndexPlan(
            type=index, quantizer=QuantizerPlan(type="rq", bits=bits, centering=True)
        )
    assert "centering=True is supported only by index type 'hnsw' with bits=4" in (
        str(e.value)
    )


@pytest.mark.parametrize("index", ["hnsw", "flat", "dynamic", "hfresh"])
def test_rq_without_centering_is_accepted_by_every_index(index):
    quantizer = QuantizerPlan(type="rq", bits=1, centering=False)
    assert IndexPlan(type=index, quantizer=quantizer).quantizer.centering is False


# (index type, quantizer type, field, value). Weaviate 1.40.4 stores the
# collection without the field and returns no error.
IGNORED_FIELDS = [
    ("hnsw", "rq", "cache", True),
    ("hnsw", "bq", "cache", True),
    ("hnsw", "bq", "rescore_limit", 30),
    ("flat", "rq", "training_limit", 100),
    ("hfresh", "rq", "cache", True),
    ("hfresh", "rq", "training_limit", 100),
]
# A dynamic index applies a field that its hnsw part or its flat part applies.
APPLIED_FIELDS = [
    ("hnsw", "rq", "rescore_limit", 40),
    ("hnsw", "rq", "training_limit", 100),
    ("hnsw", "sq", "rescore_limit", 25),
    ("hnsw", "sq", "training_limit", 100),
    ("hnsw", "pq", "segments", 8),
    ("hnsw", "pq", "centroids", 16),
    ("hnsw", "pq", "training_limit", 100),
    ("flat", "rq", "cache", True),
    ("flat", "rq", "rescore_limit", 40),
    ("flat", "bq", "cache", True),
    ("flat", "bq", "rescore_limit", 50),
    ("hfresh", "rq", "rescore_limit", 66),
    ("dynamic", "rq", "cache", True),
    ("dynamic", "rq", "rescore_limit", 40),
    ("dynamic", "rq", "training_limit", 100),
    ("dynamic", "bq", "cache", True),
    ("dynamic", "bq", "rescore_limit", 50),
]


@pytest.mark.parametrize(("index", "quantizer", "field", "value"), APPLIED_FIELDS)
def test_index_accepts_a_quantizer_field_that_it_applies(
    index, quantizer, field, value
):
    plan = IndexPlan(
        type=index, quantizer=QuantizerPlan(type=quantizer, **{field: value})
    )
    assert getattr(plan.quantizer, field) == value


@pytest.mark.parametrize(("index", "quantizer", "field", "value"), IGNORED_FIELDS)
def test_index_refuses_a_quantizer_field_that_it_does_not_apply(
    index, quantizer, field, value
):
    with pytest.raises(ValidationError) as e:
        IndexPlan(type=index, quantizer=QuantizerPlan(type=quantizer, **{field: value}))
    assert (
        f"{field}={value!r} of quantizer type {quantizer!r} is not applied by "
        f"index type {index!r}"
    ) in str(e.value)


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


def test_load_refuses_a_key_that_appears_twice_in_one_mapping(tmp_path):
    save_plan(draft_plan(SRC, [make()], "keep"), tmp_path / "p.yaml")
    lines = (tmp_path / "p.yaml").read_text().splitlines(keepends=True)
    at = next(i for i, line in enumerate(lines) if line.strip() == "type: text")
    lines.insert(at + 1, lines[at].replace("text", "int"))
    (tmp_path / "p.yaml").write_text("".join(lines))
    with pytest.raises(yaml.YAMLError) as e:
        load_plan(tmp_path / "p.yaml")
    assert e.value.problem == f"found duplicate key 'type', first at line {at + 1}"
    assert e.value.problem_mark.line == at + 1


def test_load_accepts_a_key_that_a_merge_key_brings_in_again(tmp_path):
    plan = draft_plan(SRC, [make()], "keep")
    save_plan(plan, tmp_path / "p.yaml")
    text = (tmp_path / "p.yaml").read_text()
    uri = re.search(r"( *)- source_key: '#uri'\n(?:\1  .*\n)+", text)
    assert uri is not None
    indent = uri.group(1)
    merged = (
        f"{indent}- <<: *field\n{indent}  source_key: '#uri'\n{indent}  name: uri\n"
    )
    text = text.replace(uri.group(0), merged)
    text = text.replace(
        "- source_key: '#id'\n", f"- &field\n{indent}  source_key: '#id'\n"
    )
    (tmp_path / "p.yaml").write_text(text)
    assert "<<: *field" in text and "&field" in text
    assert load_plan(tmp_path / "p.yaml") == plan


def test_models_are_exported():
    assert CollectionPlan and TargetPlan


@pytest.mark.parametrize(
    ("quantizer", "needed"),
    [
        (QuantizerPlan(type="sq"), True),
        (QuantizerPlan(type="pq"), True),
        (QuantizerPlan(type="rq", bits=4, centering=True), True),
        (QuantizerPlan(type="rq", bits=8), False),
        (QuantizerPlan(type="rq", bits=4, centering=False), False),
        (QuantizerPlan(type="bq"), False),
        (QuantizerPlan(type="none"), False),
        (QuantizerPlan(), False),
    ],
)
def test_quantizer_needs_training(quantizer, needed):
    assert quantizer.needs_training() is needed


# (count, dimension, vectors mode, whether the draft notes that the source
# has no vectors)
NO_VECTOR_NOTES = [
    (10, None, "keep", True),
    (10, None, "both", True),
    (0, None, "keep", False),
    (10, 3, "keep", False),
    (10, None, "embed", False),
]


@pytest.mark.parametrize(("count", "dimension", "vectors", "noted"), NO_VECTOR_NOTES)
def test_draft_notes_a_source_without_vectors(count, dimension, vectors, noted):
    source = replace(make(), count=count, dimension=dimension)
    entry = draft_plan(SRC, [source], vectors).collections[0]
    assert (NO_VECTORS_NOTE in entry.notes) is noted


def test_the_draft_writes_the_replication_factor_auto(tmp_path):
    plan = draft_plan(SRC, [make()], "keep")
    assert target(plan).replication_factor == "auto"
    save_plan(plan, tmp_path / "p.yaml")
    saved = yaml.safe_load((tmp_path / "p.yaml").read_text())
    assert saved["collections"][0]["target"]["replication_factor"] == "auto"


def test_a_plan_without_a_replication_factor_loads_as_auto(tmp_path):
    save_plan(draft_plan(SRC, [make()], "keep"), tmp_path / "p.yaml")
    document = yaml.safe_load((tmp_path / "p.yaml").read_text())
    document["collections"][0]["target"].pop("replication_factor", None)
    (tmp_path / "p.yaml").write_text(yaml.safe_dump(document))
    assert target(load_plan(tmp_path / "p.yaml")).replication_factor == "auto"


@pytest.mark.parametrize("value", ["auto", 1, 3, 7])
def test_replication_factor_accepts_auto_and_an_integer_from_1(value):
    made = TargetPlan(name="A", vectors=[], properties=[], replication_factor=value)
    assert made.replication_factor == value
    assert type(made.replication_factor) is type(value)


@pytest.mark.parametrize("value", [0, -1, True, False, "3", "AUTO", "none", 2.0, None])
def test_replication_factor_refuses_other_values(value):
    with pytest.raises(ValidationError) as e:
        TargetPlan(name="A", vectors=[], properties=[], replication_factor=value)
    assert [error["msg"] for error in e.value.errors()] == [
        f"Value error, replication_factor={value!r} is not valid; use auto or an "
        "integer of at least 1"
    ]


LARGE_INT_NOTE = (
    "key {key} holds ints above 2**53 in magnitude, which Weaviate would store "
    "rounded as int or number; the property is typed {type} and holds the "
    "decimal digits"
)


def test_draft_types_a_key_with_an_int_above_2_to_the_53_as_text_and_notes_it():
    fields = [
        FieldStats("hash", {"large_int": 10}),
        FieldStats("mixed", {"int": 9, "large_int": 1}),
        FieldStats("stamps", {"large_int[]": 10}),
        FieldStats("year", {"int": 10}),
    ]
    entry = draft_plan(SRC, [make(fields=fields)], "keep").collections[0]
    drafted = {
        x.source_key: (x.type, x.tokenization, x.range_filters, x.coerced_from)
        for x in entry.target.properties
        if not x.source_key.startswith("#")
    }
    assert drafted == {
        "hash": ("text", "field", False, ["large_int"]),
        "mixed": ("text", "field", False, ["int", "large_int"]),
        "stamps": ("text[]", "field", False, ["large_int[]"]),
        "year": ("int", None, True, []),
    }
    assert entry.notes == [
        LARGE_INT_NOTE.format(key="hash", type="text"),
        LARGE_INT_NOTE.format(key="mixed", type="text"),
        LARGE_INT_NOTE.format(key="stamps", type="text[]"),
    ]
