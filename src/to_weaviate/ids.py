import uuid

from weaviate.util import generate_uuid5


def to_uuid(source_id: str) -> str:
    """Keep canonical lowercase UUIDs; derive a UUIDv5 from any other id."""
    try:
        if str(uuid.UUID(source_id)) == source_id:
            return source_id
    except ValueError:
        pass
    return str(generate_uuid5(source_id))
