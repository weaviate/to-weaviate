import pytest

from to_weaviate.naming import collection_name, property_name

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
