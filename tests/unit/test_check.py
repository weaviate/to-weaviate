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
from to_weaviate.sources.base import FieldStats, SourceCollection
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
    assert plan_findings(clean, None, vectorizer_for) == (
        [],
        [
            Finding(
                "cleaned_option",
                "Docs",
                "option base_url of vector default in Docs holds userinfo, a query "
                "or a fragment; it is sent without them.",
            )
        ],
    )


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


def tenant_entry(tenant: str, *properties: p.PropertyPlan) -> p.CollectionPlan:
    base = entry([keep()])
    return base.model_copy(
        update={
            "source": base.source.model_copy(update={"ref": tenant}),
            "target": base.target.model_copy(
                update={"tenant": tenant, "properties": [ID_PROPERTY, *properties]}
            ),
        }
    )


def test_entries_of_one_collection_may_not_name_a_property_in_two_cases():
    plan = plan_of(tenant_entry("a", prop("title")), tenant_entry("b", prop("tItle")))
    errors, _ = plan_findings(plan, None, vectorizer_for)
    assert errors == [
        Finding(
            "duplicate_property",
            "Docs",
            "properties title in docs -> Docs (tenant a) and tItle in docs -> Docs "
            "(tenant b) have the same name ignoring case; Weaviate compares "
            "property names in lower case, so give the property one name in every "
            "entry of Docs",
        )
    ]


def test_entries_of_one_collection_may_share_a_property_name():
    plan = plan_of(tenant_entry("a", prop("title")), tenant_entry("b", prop("title")))
    errors, _ = plan_findings(plan, None, vectorizer_for)
    assert errors == []


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


def inspected_as(fields: list[str], sparse: list[str]) -> dict[str, SourceCollection]:
    return {
        "r": SourceCollection(
            ref="r",
            namespace="ns",
            name="docs",
            count=1,
            dimension=2,
            distance="cosine",
            embedding=None,
            fields=[FieldStats(key, {"text": 1}) for key in fields],
            sparse_keys=sparse,
            document_bytes_estimate=1,
        )
    }


def test_unmapped_key_names_each_inspected_key_that_is_not_mapped_or_dropped():
    e = with_target(
        entry([keep()]),
        properties=[ID_PROPERTY, prop("title")],
        dropped=[p.DroppedKey(source_key="skipped", reason="r")],
    )
    found = inspected_as(["title", "year", "skipped"], ["sparse"])
    errors, _ = plan_findings(plan_of(e), found, voyage_mapper)
    assert [(f.code, f.collection, f.message) for f in errors] == [
        (
            "unmapped_key",
            "Docs",
            f"metadata key {key} is not in plan.yaml; add a property for it or "
            "list it under dropped",
        )
        for key in ("year", "sparse")
    ]
    assert plan_findings(plan_of(e), None, voyage_mapper)[0] == []


@pytest.mark.parametrize(
    ("names", "found"),
    [
        (["docs", "Docs"], []),
        (["Docs", "Other"], []),
        (["docs", "dOcs"], [("Docs", "Docs, DOcs")]),
        (
            ["my_docs", "My_Docs", "MY_DOCS", "x"],
            [("My_docs", "My_docs, My_Docs, MY_DOCS")],
        ),
    ],
)
def test_load_findings_refuse_collection_names_that_differ_only_in_case(names, found):
    entries = [
        with_target(targeted(f"r{i}", name, f"t{i}"), properties=[ID_PROPERTY])
        for i, name in enumerate(names)
    ]
    assert load_findings(plan_of(*entries)) == [
        Finding(
            "duplicate_collection_name",
            collection,
            f"target collections {keys} differ only in case; Weaviate compares "
            "collection names ignoring case, so give each its own name",
        )
        for collection, keys in found
    ]


def quantized(count: int, **quantizer: Any) -> p.CollectionPlan:
    index = p.IndexPlan(type="hnsw", quantizer=p.QuantizerPlan(**quantizer))
    e = entry([p.VectorPlan(mode="keep", distance="cosine", index=index)])
    return e.model_copy(update={"source": e.source.model_copy(update={"count": count})})


@pytest.mark.parametrize(
    ("quantizer", "count", "limit"),
    [
        ({"type": "sq"}, 300, 100000),
        ({"type": "pq"}, 99999, 100000),
        ({"type": "rq", "centering": True, "bits": 4}, 9999, 10000),
        ({"type": "sq", "training_limit": 500}, 300, 500),
    ],
)
def test_a_trained_quantizer_above_the_record_count_is_a_warning(
    quantizer, count, limit
):
    plan = plan_of(quantized(count, **quantizer))
    errors, warnings = plan_findings(plan, None, voyage_mapper)
    assert errors == []
    assert warnings == [
        Finding(
            "quantizer_training",
            "Docs",
            f"vector default: Weaviate trains the {quantizer['type']} quantizer of "
            f"a shard once the shard holds {limit} vectors, and the source has "
            f"{count} records; with ASYNC_INDEXING on the server the vectors stay "
            "uncompressed. Set training_limit below the count of a shard.",
        )
    ]


@pytest.mark.parametrize(
    ("quantizer", "count"),
    [
        ({"type": "sq"}, 100000),
        ({"type": "sq", "training_limit": 100}, 300),
        ({"type": "rq"}, 10),
        ({"type": "rq", "centering": True, "bits": 4}, 10000),
        ({"type": "bq"}, 1),
        ({"type": "none"}, 1),
        ({"type": "default"}, 1),
    ],
)
def test_other_quantizers_give_no_training_warning(quantizer, count):
    plan = plan_of(quantized(count, **quantizer))
    errors, warnings = plan_findings(plan, None, voyage_mapper)
    assert errors == [] and warnings == []


def test_a_hand_written_property_name_with_a_reserved_suffix_is_a_warning():
    properties = [ID_PROPERTY, prop("max_temp"), prop("title_searchable_meta")]
    plan = plan_of(with_target(entry([keep()]), properties=properties))
    errors, warnings = plan_findings(plan, None, voyage_mapper)
    assert errors == []
    assert warnings == [
        Finding(
            "reserved_suffix",
            "Docs",
            "property max_temp ends in _temp; Weaviate 1.35.20, 1.36.15, 1.37.5 "
            "and later refuse to create a property with this suffix",
        )
    ]


@pytest.mark.parametrize(
    "url",
    [
        "https://proxy.example.com/v1?api-version=made-up",
        "https://proxy.example.com/v1#made-up",
        "https://u:p@proxy.example.com/v1?key=made-up",
        "https://u:p@proxy.example.com/v1",
    ],
)
def test_a_url_option_sent_in_another_form_gives_a_warning(url):
    plan = plan_of(entry([embed("text2vec-openai", base_url=url)]))
    errors, warnings = plan_findings(plan, None, vectorizer_for)
    assert errors == []
    assert warnings == [
        Finding(
            "cleaned_option",
            "Docs",
            "option base_url of vector default in Docs holds userinfo, a query or "
            "a fragment; it is sent without them.",
        )
    ]


@pytest.mark.parametrize(
    ("name", "word"),
    [
        ("geo__map", "map"),
        ("doc__backup", "backup"),
        ("x__reindex_3", "reindex"),
        ("a__b_ingest", "ingest"),
    ],
)
def test_a_hand_written_sidecar_shaped_property_name_is_a_warning(name, word):
    properties = [ID_PROPERTY, prop(name), prop("geo_map"), prop("mapping__x")]
    plan = plan_of(with_target(entry([keep()]), properties=properties))
    errors, warnings = plan_findings(plan, None, voyage_mapper)
    assert errors == []
    assert warnings == [
        Finding(
            "reserved_suffix",
            "Docs",
            f"property {name} has the shape of a migration directory name (__ and "
            f"the word {word}); Weaviate 1.39.3 and later refuse to create a "
            "property with this shape",
        )
    ]


# (target name, property name, the refused name that the finding names).
# Weaviate 1.40.4 refuses each of these names; the client refuses vector.
REFUSED_NAMES = [
    ("tiny-l2", "position", "collection tiny-l2"),
    ("C" * 256, "position", f"collection {'C' * 256}"),
    ("Docs", "has space", "property has space"),
    ("Docs", "p" * 232, f"property {'p' * 232}"),
    ("Docs", "Id", "property Id"),
    ("Docs", "_additional", "property _additional"),
    ("Docs", "vector", "property vector"),
    ("Docs", "Position", "property Position"),
    ("Docs", "ID", "property ID"),
    ("Docs", "Vector", "property Vector"),
]


@pytest.mark.parametrize(("target", "name", "named"), REFUSED_NAMES)
def test_a_name_that_weaviate_refuses_is_an_invalid_name(target, name, named):
    found = entry([keep()], name=target)
    found.target.properties.append(
        p.PropertyPlan(source_key="position", name=name, type="int")
    )
    errors, _ = plan_findings(plan_of(found), None, vectorizer_for)
    assert [(e.code, e.collection) for e in errors] == [("invalid_name", target)]
    assert errors[0].message.startswith(f"{named}: Weaviate ")


@pytest.mark.parametrize("name", ["iD", "__", "class", "p" * 231, "_x", "_X"])
def test_a_name_that_weaviate_accepts_is_no_invalid_name(name):
    found = entry([keep()], name="C" * 255)
    found.target.properties.append(
        p.PropertyPlan(source_key="position", name=name, type="int")
    )
    errors, _ = plan_findings(plan_of(found), None, vectorizer_for)
    assert errors == []


@pytest.mark.parametrize("name", ["mt.dot", "a b", "t" * 65])
def test_a_tenant_name_that_weaviate_refuses_is_an_invalid_name(name):
    found = entry([keep()], name="Docs")
    found.target.tenant = name
    errors, _ = plan_findings(plan_of(found), None, vectorizer_for)
    assert [(e.code, e.collection, e.message) for e in errors] == [
        (
            "invalid_name",
            "Docs",
            f"tenant {name}: Weaviate accepts a tenant name that matches "
            "^[-_0-9A-Za-z]{1,64}$",
        )
    ]


@pytest.mark.parametrize("name", ["mt_dot", "a-b", "T" * 64, "1"])
def test_a_tenant_name_that_weaviate_accepts_is_no_invalid_name(name):
    found = entry([keep()], name="Docs")
    found.target.tenant = name
    errors, _ = plan_findings(plan_of(found), None, vectorizer_for)
    assert errors == []


def test_a_kept_vector_of_a_source_without_vectors_is_an_error():
    found = entry([keep()])
    found.source.dimension = None
    errors, _ = plan_findings(plan_of(found), None, vectorizer_for)
    assert [(e.code, e.collection) for e in errors] == [("no_source_vectors", "Docs")]
    assert "--vectors embed --vectorizer" in errors[0].message


@pytest.mark.parametrize(
    ("count", "vector"), [(0, keep()), (1, embed("text2vec-weaviate"))]
)
def test_an_empty_source_or_an_embedded_vector_needs_no_source_vectors(count, vector):
    found = entry([vector])
    found.source.dimension, found.source.count = None, count
    errors, _ = plan_findings(plan_of(found), None, vectorizer_for)
    assert errors == []


def test_a_property_name_with_an_upper_case_first_letter_names_the_stored_name():
    found = entry([keep()])
    found.target.properties.append(
        p.PropertyPlan(source_key="position", name="Position", type="int")
    )
    errors, _ = plan_findings(plan_of(found), None, vectorizer_for)
    assert [(e.code, e.message) for e in errors] == [
        (
            "invalid_name",
            "property Position: Weaviate stores a property name with its first "
            "letter in lower case, as position; use that name",
        )
    ]
