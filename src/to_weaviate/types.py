from collections.abc import Iterable
from typing import Any

SCALARS = ("text", "int", "number", "boolean")
SPARSE_VECTOR = "sparse_vector"
# The client sends property values in a protobuf Struct, whose numbers are
# doubles, and Weaviate keeps an int as a double. A double holds every int up
# to this magnitude; above it the stored value can differ from the source.
EXACT_INT_LIMIT = 2**53
# Observed type of an int above EXACT_INT_LIMIT. It is not a property type.
LARGE_INT = "large_int"


def _scalar_type(value: Any) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "int" if abs(value) <= EXACT_INT_LIMIT else LARGE_INT
    if isinstance(value, float):
        return "number"
    return "text"


def value_type(value: Any) -> str:
    """Observed type of a metadata value. Empty or mixed lists are `text[]`.

    It is a Weaviate type, or `large_int` for an int that no property type
    but text holds exactly. A list of ints with such an int is `large_int[]`.
    """
    if isinstance(value, dict) and value.get("#type") == SPARSE_VECTOR:
        return SPARSE_VECTOR
    if isinstance(value, list):
        element_types = {_scalar_type(v) for v in value}
        if element_types == {"int", LARGE_INT}:
            element_types = {LARGE_INT}
        return (element_types.pop() if len(element_types) == 1 else "text") + "[]"
    return _scalar_type(value)


def has_large_int(types: Iterable[str]) -> bool:
    """Whether the observed `types` include an int above 2**53 in magnitude."""
    return any(t.removesuffix("[]") == LARGE_INT for t in types)


def _merge_scalars(types: set[str]) -> str:
    if LARGE_INT in types:
        return "text"
    if len(types) == 1:
        return next(iter(types))
    if types == {"int", "number"}:
        return "number"
    return "text"


def resolve_type(counts: dict[str, int]) -> tuple[str, list[str]]:
    """Pick one target type for a key observed with several types.

    Returns the target type and the sorted observed types that differ from it.
    """
    observed = set(counts)
    is_array = any(t.endswith("[]") for t in observed)
    base = _merge_scalars({t.removesuffix("[]") for t in observed})
    target = base + "[]" if is_array else base
    return target, sorted(observed - {target})


def _exact_int(value: int) -> int:
    if abs(value) > EXACT_INT_LIMIT:
        raise ValueError(
            f"int {value} is above 2**53 in magnitude and Weaviate would store "
            "it rounded; set the property type to text, or text[] for a list"
        )
    return value


def _coerce_scalar(value: Any, target: str) -> Any:
    kind = _scalar_type(value)
    if target == "text":
        if kind == "boolean":
            return "true" if value else "false"
        if kind == "text":
            return value
        return str(value)
    if target == "number" and kind == "number":
        return float(value)
    if target == "number" and kind in ("int", LARGE_INT):
        return float(_exact_int(value))
    if target == "int" and kind in ("int", LARGE_INT):
        return _exact_int(value)
    if target == "int" and kind == "number" and value.is_integer():
        return _exact_int(int(value))
    if target == "boolean" and kind == "boolean":
        return value
    raise ValueError(
        f"cannot convert {kind} value {value!r} to {target} without losing data"
    )


def coerce(value: Any, target_type: str) -> Any:
    """Convert a value to `target_type`. Raises ValueError if data would be lost."""
    if target_type.endswith("[]"):
        element_type = target_type.removesuffix("[]")
        items = value if isinstance(value, list) else [value]
        return [_coerce_scalar(v, element_type) for v in items]
    if isinstance(value, list | dict):
        raise ValueError(f"cannot convert {value_type(value)} value to {target_type}")
    return _coerce_scalar(value, target_type)
