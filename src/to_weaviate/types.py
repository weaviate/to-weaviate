from typing import Any

SCALARS = ("text", "int", "number", "boolean")
SPARSE_VECTOR = "sparse_vector"
# The client sends property values in a protobuf Struct, whose numbers are
# doubles. Weaviate checks that a double for an int property survives Go's
# int64 conversion, which for 2**63 depends on the CPU: amd64 refuses it,
# arm64 stores 2**63 - 1. An int whose double reaches this limit is refused
# here, so the outcome does not depend on the server.
INT_DOUBLE_LIMIT = 2.0**63


def _scalar_type(value: Any) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "number"
    return "text"


def value_type(value: Any) -> str:
    """Weaviate type of a metadata value. Empty or mixed lists are `text[]`."""
    if isinstance(value, dict) and value.get("#type") == SPARSE_VECTOR:
        return SPARSE_VECTOR
    if isinstance(value, list):
        element_types = {_scalar_type(v) for v in value}
        return (element_types.pop() if len(element_types) == 1 else "text") + "[]"
    return _scalar_type(value)


def _merge_scalars(types: set[str]) -> str:
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


def _stored_int(value: int) -> int:
    if float(value) >= INT_DOUBLE_LIMIT:
        raise ValueError(
            f"int {value} cannot be stored: Weaviate keeps ints as doubles and "
            "refuses this value; type the property text"
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
    if target == "number" and kind in ("int", "number"):
        return float(value)
    if target == "int" and kind == "int":
        return _stored_int(value)
    if target == "int" and kind == "number" and value.is_integer():
        return _stored_int(int(value))
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
