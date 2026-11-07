import re

RESERVED_PROPERTIES = frozenset(
    {"id", "_id", "_additional", "document", "source_id", "uri"}
)

# Weaviate refuses to create a property whose name ends in one of these:
# reservedPropertyNameSuffixes in entities/schema/validation.go, v1.40.4.
# Releases before 1.35.20, 1.36.15 and 1.37.5 accept such names.
RESERVED_SUFFIXES = (
    "_searchable",
    "_rangeable",
    "_temp",
    "__meta_count",
    "_propertyLength",
    "_nullState",
)

# Weaviate also refuses a property name that reads as the working directory of
# a migration: PropertyNameIsSidecarShaped in entities/schema/validation.go,
# v1.40.4. Releases before 1.39.3 accept such names.
SIDECAR_ROLE_WORDS = ("reindex", "ingest", "backup", "map")
_DIGITS = "0123456789"

_INVALID = re.compile(r"[^A-Za-z0-9_]")


def reserved_suffix(name: str) -> str | None:
    """The reserved suffix that `name` ends in, or None.

    The comparison is case-sensitive, as in the server.
    """
    return next((s for s in RESERVED_SUFFIXES if name.endswith(s)), None)


def sidecar_word(name: str) -> str | None:
    """The role word that makes `name` sidecar-shaped, or None.

    The server's rule: take the part of `name` after its first `__`, drop a
    `_<digits>` tail, and read the word after the last `_`, or the whole part
    when it has no `_`. The name is sidecar-shaped when that word is in
    SIDECAR_ROLE_WORDS.
    """
    start = name.find("__")
    if start < 0:
        return None
    part = name[start + 2 :]
    head, sep, tail = part.rpartition("_")
    if sep and tail and not tail.lstrip(_DIGITS):
        part = head
    word = part[part.rfind("_") + 1 :]
    return word if word in SIDECAR_ROLE_WORDS else None


def _unique(base: str, taken: set[str]) -> str:
    """`base`, or `base_<n>` with the lowest n from 2, not in `taken` ignoring case.

    Weaviate compares collection names and property names ignoring case.
    """
    lowered = {name.lower() for name in taken}
    if base.lower() not in lowered:
        return base
    n = 2
    while f"{base}_{n}".lower() in lowered:
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
    # "_meta" makes "meta" the last word, so the name is not sidecar-shaped.
    if name in RESERVED_PROPERTIES or reserved_suffix(name) or sidecar_word(name):
        name += "_meta"
    return _unique(name, taken)
