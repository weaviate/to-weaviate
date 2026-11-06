import pytest

from to_weaviate.naming import collection_name, property_name

COLLECTIONS = [
    ("my-docs", set(), "My_docs"),
    ("my.docs", {"My_docs"}, "My_docs_2"),
    ("my.docs", {"My_docs", "My_docs_2"}, "My_docs_3"),
    ("123abc", set(), "C_123abc"),
    ("_x", set(), "C__x"),
    ("docs", set(), "Docs"),
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
