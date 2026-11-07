import re

# The names that Weaviate 1.40.4 accepts. It stores a collection name with
# its first letter in upper case and a property name with its first letter
# in lower case, and compares the stored property name with SERVER_RESERVED.
# The client refuses the property name vector.
MAX_COLLECTION_NAME = 255
MAX_PROPERTY_NAME = 231
COLLECTION_NAME_RULE = f"^[A-Z][_0-9A-Za-z]{{0,{MAX_COLLECTION_NAME - 1}}}$"
PROPERTY_NAME_RULE = f"^[_A-Za-z][_0-9A-Za-z]{{0,{MAX_PROPERTY_NAME - 1}}}$"
SERVER_RESERVED = ("id", "_id", "_additional")
CLIENT_RESERVED = ("vector",)
# The draft uses document, source_id and uri itself.
RESERVED_PROPERTIES = frozenset(
    {*SERVER_RESERVED, *CLIENT_RESERVED, "document", "source_id", "uri"}
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


def collection_name_problem(name: str) -> str | None:
    """Why Weaviate refuses the collection name `name`, or None."""
    stored = name[:1].upper() + name[1:]
    if re.fullmatch(COLLECTION_NAME_RULE, stored):
        return None
    return (
        f"Weaviate accepts a collection name that matches {COLLECTION_NAME_RULE} "
        "once its first letter is in upper case"
    )


def property_name_problem(name: str) -> str | None:
    """Why Weaviate or its client refuses the property name `name`, or None."""
    if not re.fullmatch(PROPERTY_NAME_RULE, name):
        return f"Weaviate accepts a property name that matches {PROPERTY_NAME_RULE}"
    stored = name[:1].lower() + name[1:]
    if stored in SERVER_RESERVED or name in CLIENT_RESERVED:
        return f"Weaviate reserves the property name {stored}"
    # The plan would then name a property that the collection does not hold
    # under that name.
    if stored != name:
        return (
            "Weaviate stores a property name with its first letter in lower "
            f"case, as {stored}; use that name"
        )
    return None


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


def _numbered(base: str, taken: set[str]) -> str:
    """`base_<n>` with the lowest n from 1 not in `taken` ignoring case."""
    lowered = {name.lower() for name in taken}
    n = 1
    while f"{base}_{n}".lower() in lowered:
        n += 1
    return f"{base}_{n}"


def collection_name(source_name: str, taken: set[str]) -> str:
    """Return a valid collection name not in `taken`. Does not modify `taken`.

    A name longer than MAX_COLLECTION_NAME is cut, so that it stays within
    the limit with its `_<n>` suffix.
    """
    name = _INVALID.sub("_", source_name)
    if not name or not ("A" <= name[0] <= "Z" or "a" <= name[0] <= "z"):
        name = "C_" + name
    base = name[0].upper() + name[1:MAX_COLLECTION_NAME]
    found = _unique(base, taken)
    while len(found) > MAX_COLLECTION_NAME:
        base = base[: len(base) - (len(found) - MAX_COLLECTION_NAME)]
        found = _unique(base, taken)
    return found


def property_name(source_key: str, taken: set[str]) -> str:
    """Return a valid, non-reserved property name not in `taken`. Pure."""
    name = _INVALID.sub("_", source_key)
    if not name or name[0].isdigit():
        name = "p_" + name
    name = name[0].lower() + name[1:]
    # "_meta" makes "meta" the last word, so the name is not sidecar-shaped.
    if name in RESERVED_PROPERTIES or reserved_suffix(name) or sidecar_word(name):
        name += "_meta"
    name = _unique(name, taken)
    return name if len(name) <= MAX_PROPERTY_NAME else _numbered("key", taken)
