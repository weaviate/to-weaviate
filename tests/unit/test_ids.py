from weaviate.util import generate_uuid5

from to_weaviate.ids import to_uuid


def test_to_uuid():
    canonical = "3f2b6c1e-8d4a-4b7e-9c1d-2a5e6f7a8b9c"
    assert to_uuid(canonical) == canonical
    assert to_uuid(canonical.upper()) == generate_uuid5(canonical.upper())
    assert to_uuid("doc-1") == generate_uuid5("doc-1")
