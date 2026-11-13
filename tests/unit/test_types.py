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
    (2**53, "int"),
    (-(2**53), "int"),
    (2**53 + 1, "large_int"),
    (-(2**53) - 1, "large_int"),
    (2**63 - 1, "large_int"),
    (2.0**60, "number"),
    ([2**53 + 1], "large_int[]"),
    ([1, 2**53 + 1], "large_int[]"),
    ([1.5, 2**53 + 1], "large_int[]"),
    (["a", True, 2**60], "large_int[]"),
    ([1.5, 2**53], "text[]"),
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
    ({"large_int": 1}, ("text", ["large_int"])),
    ({"int": 5, "large_int": 1}, ("text", ["int", "large_int"])),
    ({"number": 5, "large_int": 1}, ("text", ["large_int", "number"])),
    ({"large_int[]": 2}, ("text[]", ["large_int[]"])),
    ({"int[]": 2, "large_int": 1}, ("text[]", ["int[]", "large_int"])),
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
    (2**53, "int", 2**53),
    (-(2**53), "int", -(2**53)),
    (2.0**53, "int", 2**53),
    (2**53, "number", 2.0**53),
    (1e300, "number", 1e300),
    (2**53 + 1, "text", "9007199254740993"),
    (-(2**53) - 1, "text", "-9007199254740993"),
    (2**63 - 1, "text", "9223372036854775807"),
    ([1, 2**63 - 1], "text[]", ["1", "9223372036854775807"]),
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
    [("x", "int"), ("x", "number"), (2.5, "int"), (True, "int"), ("x", "boolean")],
)
def test_coerce_refuses_lossy(value, target):
    with pytest.raises(ValueError):
        coerce(value, target)


# (value, target type). Weaviate keeps an int and a number as a double, which
# holds every int up to 2**53 in magnitude.
ABOVE_THE_DOUBLE_LIMIT = [
    (2**53 + 1, "int"),
    (-(2**53) - 1, "int"),
    (2**63 - 1, "int"),
    (-(2**63), "int"),
    (2.0**60, "int"),
    (2**53 + 1, "number"),
    (-(2**53) - 1, "number"),
    ([1, 2**53 + 1], "int[]"),
    ([1, 2**53 + 1], "number[]"),
]


@pytest.mark.parametrize(("value", "target"), ABOVE_THE_DOUBLE_LIMIT)
def test_coerce_refuses_an_int_above_2_to_the_53(value, target):
    with pytest.raises(ValueError, match=r"above 2\*\*53"):
        coerce(value, target)


@pytest.mark.parametrize(
    ("value", "target", "message"),
    [
        (
            2**53 + 1,
            "int",
            "int 9007199254740993 is above 2**53 in magnitude and Weaviate would "
            "store it rounded; set the property type to text, or text[] for a list",
        ),
        (
            2.0**60,
            "int",
            "number 1.152921504606847e+18 is above 2**53 in magnitude, which an "
            "int property does not hold; set the property type to number or text",
        ),
        (
            -1e300,
            "int[]",
            "number -1e+300 is above 2**53 in magnitude, which an int property "
            "does not hold; set the property type to number or text",
        ),
    ],
)
def test_the_refusal_names_the_value_as_the_source_holds_it(value, target, message):
    with pytest.raises(ValueError) as raised:
        coerce(value, target)
    assert str(raised.value) == message


@pytest.mark.parametrize(
    ("value", "target", "message"),
    [
        (
            2**53 + 1,
            "boolean",
            "cannot convert int value 9007199254740993 to boolean without "
            "losing data",
        ),
        ([2**53 + 1], "int", "cannot convert int[] value to int"),
        ([1, 2**53 + 1], "text", "cannot convert int[] value to text"),
    ],
)
def test_a_refusal_names_an_int_above_2_to_the_53_as_int(value, target, message):
    with pytest.raises(ValueError) as raised:
        coerce(value, target)
    assert str(raised.value) == message
