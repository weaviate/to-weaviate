from typing import Any

DISTANCES = {"l2": "l2-squared", "cosine": "cosine", "ip": "dot"}
DEFAULT_SPACE = "l2"


def _config(collection: dict[str, Any]) -> dict[str, Any]:
    return collection.get("configuration_json") or {}


def _space(section: Any) -> str | None:
    if isinstance(section, dict):
        space = section.get("space")
        if isinstance(space, str):
            return space
    return None


def _chroma_space(collection: dict[str, Any]) -> str:
    config = _config(collection)
    legacy = config.get("hnsw_configuration")
    if legacy is not None:
        metadata = collection.get("metadata") or {}
        space = metadata.get("hnsw:space")
        if isinstance(space, str):
            return space
        return _space(legacy) or DEFAULT_SPACE
    return _space(config.get("hnsw")) or _space(config.get("spann")) or DEFAULT_SPACE


def distance_of(collection: dict[str, Any]) -> str:
    """Weaviate distance name of a Chroma collection."""
    space = _chroma_space(collection)
    try:
        return DISTANCES[space]
    except KeyError:
        raise ValueError(f"unsupported Chroma distance {space!r}") from None


def embedding_of(collection: dict[str, Any]) -> dict[str, Any] | None:
    """The embedding function when it is a known one, otherwise None."""
    function = _config(collection).get("embedding_function")
    if isinstance(function, dict) and function.get("type") == "known":
        return function
    return None


def sparse_keys_of(collection: dict[str, Any]) -> list[str]:
    """Metadata keys the collection schema declares as sparse vectors."""
    schema = collection.get("schema")
    keys = schema.get("keys") if isinstance(schema, dict) else None
    if not isinstance(keys, dict):
        return []
    return sorted(
        key
        for key, entry in keys.items()
        if isinstance(entry, dict) and "sparse_vector" in entry
    )
