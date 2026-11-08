import pytest

from to_weaviate.types import coerce, resolve_type, value_type

SPARSE = {"#type": "sparse_vector", "indices": [1], "values": [0.5]}

VALUE_TYPES = [
    (True, "boolean"),
    (1, "int"),
    (1.0, "number"),
    ("a", "text"),
    ([1, 2], "int[]"),
    ([True], "boolean[]"),
    ([], "text[]"),
    ([1, "a"], "text[]"),
    (SPARSE, "sparse_vector"),
]

RESOLVE = [
    ({"int": 5}, ("int", [])),
    ({"int": 5, "number": 2}, ("number", ["int"])),
    ({"int": 5, "text": 1}, ("text", ["int"])),
    ({"text": 4, "text[]": 1}, ("text[]", ["text"])),
    ({"int": 3, "boolean": 1}, ("text", ["boolean", "int"])),
    ({"int[]": 2, "number": 1}, ("number[]", ["int[]", "number"])),
    ({"int[]": 2, "number[]": 1}, ("number[]", ["int[]"])),
    ({"int[]": 2, "text[]": 1}, ("text[]", ["int[]"])),
]

COERCE = [
    (3, "number", 3.0),
    (3, "text", "3"),
    (True, "text", "true"),
    (False, "text", "false"),
    ("a", "text[]", ["a"]),
    (3, "number[]", [3.0]),
    ([1, 2], "number[]", [1.0, 2.0]),
    ([1, True], "text[]", ["1", "true"]),
    (2.0, "int", 2),
    ("x", "text", "x"),
    (2**53, "number", float(2**53)),
    (SPARSE, "sparse_vector", SPARSE),
]


@pytest.mark.parametrize(("value", "expected"), VALUE_TYPES)
def test_value_type(value, expected):
    assert value_type(value) == expected


@pytest.mark.parametrize(("counts", "expected"), RESOLVE)
def test_resolve_type(counts, expected):
    assert resolve_type(counts) == expected


@pytest.mark.parametrize(("value", "target", "expected"), COERCE)
def test_coerce(value, target, expected):
    result = coerce(value, target)
    assert result == expected and type(result) is type(expected)


@pytest.mark.parametrize(
    ("value", "target"),
    [
        ("x", "int"),
        ("x", "number"),
        (2.5, "int"),
        (True, "int"),
        ("x", "boolean"),
        (2**53 + 1, "number"),
        (10**400, "number"),
        (2**53 + 1, "number[]"),
        (SPARSE, "text"),
        (SPARSE, "text[]"),
        ([SPARSE], "text[]"),
        ([[1]], "text[]"),
        ({"a": 1}, "sparse_vector"),
        ("x", "sparse_vector"),
    ],
)
def test_coerce_refuses_lossy(value, target):
    with pytest.raises(ValueError):
        coerce(value, target)
