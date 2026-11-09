import pytest

from to_weaviate.naming import (
    collection_name,
    collection_name_problem,
    property_name,
    property_name_problem,
    reserved_suffix,
    sidecar_word,
)

COLLECTIONS = [
    ("my-docs", set(), "My_docs"),
    ("my.docs", {"My_docs"}, "My_docs_2"),
    ("my.docs", {"My_docs", "My_docs_2"}, "My_docs_3"),
    ("123abc", set(), "C_123abc"),
    ("_x", set(), "C__x"),
    ("docs", set(), "Docs"),
    ("my_Docs", {"My_docs"}, "My_Docs_2"),
    ("MY_DOCS", {"My_docs", "My_Docs_2"}, "MY_DOCS_3"),
    ("my_docs", {"MY_DOCS"}, "My_docs_2"),
]

PROPERTIES = [
    ("user id", set(), "user_id"),
    ("id", set(), "id_meta"),
    ("document", set(), "document_meta"),
    ("source_id", set(), "source_id_meta"),
    ("9lives", set(), "p_9lives"),
    ("a-b", {"a_b"}, "a_b_2"),
    ("a-b", {"a_b", "a_b_2"}, "a_b_3"),
    ("Year", set(), "year"),
    ("mykey", {"myKey"}, "mykey_2"),
    ("myKey", {"mykey", "MYKEY_2"}, "myKey_3"),
    ("max_temp", set(), "max_temp_meta"),
    ("title_searchable", set(), "title_searchable_meta"),
    ("price_rangeable", set(), "price_rangeable_meta"),
    ("x__meta_count", set(), "x__meta_count_meta"),
    ("x_propertyLength", set(), "x_propertyLength_meta"),
    ("x_nullState", set(), "x_nullState_meta"),
    ("max_temp", {"max_temp_meta"}, "max_temp_meta_2"),
    ("geo__map", set(), "geo__map_meta"),
    ("doc__backup", set(), "doc__backup_meta"),
    ("x__reindex_3", set(), "x__reindex_3_meta"),
    ("a__b_ingest", set(), "a__b_ingest_meta"),
    ("geo__map", {"geo__map_meta"}, "geo__map_meta_2"),
    ("geo_map", set(), "geo_map"),
    ("mapping__x", set(), "mapping__x"),
    ("__map_x", set(), "__map_x"),
]

# (name, the role word that makes it sidecar-shaped)
SIDECAR_NAMES = [
    ("geo__map", "map"),
    ("doc__backup", "backup"),
    ("x__reindex_3", "reindex"),
    ("a__b_ingest", "ingest"),
    ("x___map", "map"),
    ("title__enable_filterable_ingest_3", "ingest"),
    ("geo_map", None),
    ("mapping__x", None),
    ("__map_x", None),
    ("x__map_", None),
    ("x__map_3a", None),
    ("geo__map_meta", None),
    ("x__reindex_3_meta_2", None),
]


@pytest.mark.parametrize(("source", "taken", "expected"), COLLECTIONS)
def test_collection_name(source, taken, expected):
    snapshot = set(taken)
    assert collection_name(source, taken) == expected
    assert taken == snapshot


@pytest.mark.parametrize(("source", "taken", "expected"), PROPERTIES)
def test_property_name(source, taken, expected):
    snapshot = set(taken)
    assert property_name(source, taken) == expected
    assert taken == snapshot


@pytest.mark.parametrize(("name", "word"), SIDECAR_NAMES)
def test_sidecar_word_follows_the_server_rule(name, word):
    assert sidecar_word(name) == word


@pytest.mark.parametrize(
    "key", ["geo__map", "doc__backup", "x__reindex_3", "a__b_ingest"]
)
def test_a_drafted_name_is_neither_sidecar_shaped_nor_reserved(key):
    name = property_name(key, set())
    assert sidecar_word(name) is None and reserved_suffix(name) is None


LONG_KEY = "k" * 300
# (source key, drafted name), drafted in this order into one collection, as
# the draft does after document, source_id and uri. Weaviate 1.40.4 accepts
# every drafted name; it refuses the property names vector (the client), id
# in any case of its first letter, and names longer than 231 characters.
DRAFTED_IN_ORDER = [
    ("日本", "__"),
    ("中国", "___2"),
    ("", "p_"),
    ("1st key", "p_1st_key"),
    ("my key", "my_key"),
    ("ID", "iD"),
    ("Id", "id_meta"),
    ("id", "id_meta_2"),
    ("_additional", "_additional_meta"),
    ("__meta", "__meta"),
    ("vector", "vector_meta"),
    ("Vector", "vector_meta_2"),
    (LONG_KEY, "key_1"),
    (LONG_KEY + "x", "key_2"),
]


def test_drafted_property_names_are_names_the_server_accepts():
    taken = {"document", "source_id", "uri"}
    for key, expected in DRAFTED_IN_ORDER:
        name = property_name(key, taken)
        assert name == expected, key
        taken.add(name)


def test_a_reserved_name_near_the_length_limit_falls_back_to_a_numbered_key():
    assert property_name("t" * 226 + "_temp", set()) == "key_1"


def test_a_long_collection_name_is_cut_to_the_server_limit():
    long = "c" * 300
    first = collection_name(long, set())
    second = collection_name(long, {first})
    assert len(first) == 255 and len(second) <= 255
    assert second.lower() != first.lower()
    assert collection_name_problem(first) is None
    assert collection_name_problem(second) is None


def test_drafted_collection_names_in_order():
    taken: set[str] = set()
    found = []
    for source in ["文档", "1-leading-digit", "Mixed-Case", "mixed-case", "class"]:
        found.append(collection_name(source, taken))
        taken.add(found[-1])
    assert found == ["C___", "C_1_leading_digit", "Mixed_Case", "Mixed_case_2", "Class"]


@pytest.mark.parametrize(
    "key", ["Position", "ID", "Year", "_X", "日本", "1st", "Vector"]
)
def test_a_drafted_property_name_is_stored_as_drafted(key):
    name = property_name(key, set())
    assert property_name_problem(name) is None, name
