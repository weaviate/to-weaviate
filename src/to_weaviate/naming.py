import re

RESERVED_PROPERTIES = frozenset(
    {"id", "_id", "_additional", "document", "source_id", "uri"}
)

_INVALID = re.compile(r"[^A-Za-z0-9_]")


def _unique(base: str, taken: set[str]) -> str:
    if base not in taken:
        return base
    n = 2
    while f"{base}_{n}" in taken:
        n += 1
    return f"{base}_{n}"


def collection_name(source_name: str, taken: set[str]) -> str:
    """Return a valid collection name not in `taken`. Does not modify `taken`."""
    name = _INVALID.sub("_", source_name)
    if not name or not ("A" <= name[0] <= "Z" or "a" <= name[0] <= "z"):
        name = "C_" + name
    return _unique(name[0].upper() + name[1:], taken)


def property_name(source_key: str, taken: set[str]) -> str:
    """Return a valid, non-reserved property name not in `taken`. Pure."""
    name = _INVALID.sub("_", source_key)
    if not name or name[0].isdigit():
        name = "p_" + name
    name = name[0].lower() + name[1:]
    if name in RESERVED_PROPERTIES:
        name += "_meta"
    return _unique(name, taken)
