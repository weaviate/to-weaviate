from typing import Any

SCALARS = ("text", "int", "number", "boolean")
SPARSE_VECTOR = "sparse_vector"


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


def _to_float(value: int | float) -> float:
    try:
        result = float(value)
    except OverflowError:
        raise ValueError(f"int value {value!r} is too large for number") from None
    if result != value:
        raise ValueError(f"int value {value!r} loses precision as number")
    return result


def _coerce_scalar(value: Any, target: str) -> Any:
    if isinstance(value, list | dict):
        raise ValueError(f"cannot convert {value_type(value)} value to {target}")
    kind = _scalar_type(value)
    if target == "text":
        if kind == "boolean":
            return "true" if value else "false"
        if kind == "text":
            return value
        return str(value)
    if target == "number" and kind in ("int", "number"):
        return _to_float(value)
    if target == "int" and kind == "int":
        return value
    if target == "int" and kind == "number" and value.is_integer():
        return int(value)
    if target == "boolean" and kind == "boolean":
        return value
    raise ValueError(
        f"cannot convert {kind} value {value!r} to {target} without losing data"
    )


def coerce(value: Any, target_type: str) -> Any:
    """Convert a value to `target_type`. Raises ValueError if data would be lost."""
    if target_type == SPARSE_VECTOR:
        if value_type(value) == SPARSE_VECTOR:
            return value
        raise ValueError(f"cannot convert {value_type(value)} value to {target_type}")
    if target_type.endswith("[]"):
        element_type = target_type.removesuffix("[]")
        items = value if isinstance(value, list) else [value]
        return [_coerce_scalar(v, element_type) for v in items]
    if isinstance(value, list | dict):
        raise ValueError(f"cannot convert {value_type(value)} value to {target_type}")
    return _coerce_scalar(value, target_type)
