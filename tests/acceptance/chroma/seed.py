"""Seed the local Chroma test server with the collections of the acceptance suite.

Run it as a script, or call `seed()`. Each run deletes and recreates the
collections, so every run sends the same data. `add_to_tiny_l2` and
`delete_from_tiny_l2` add records to tiny.l2 and delete them again. Every
function here that sends a request reads CHROMA_URL and runs only when it is
TEST_URL.
"""

import math
import os
import random
import sys
import uuid
from dataclasses import dataclass
from typing import Any

import httpx

DATABASE_PATH = "/api/v2/tenants/default_tenant/databases/default_database"
ADD_BATCH = 300
TEST_URL = "http://localhost:8001"
# Every collection that `seed` makes, in the order it makes them.
COLLECTIONS = ("docs-cosine", "tiny.l2", "docs-rq", "docs-ip")
CATEGORIES = ("guide", "reference", "tutorial")
TAGS = ("red", "green", "blue", "small", "large")


@dataclass(frozen=True)
class Record:
    id: str
    embedding: list[float]
    document: str | None
    metadata: dict[str, Any]
    uri: str


def _vector(rng: random.Random, dimension: int) -> list[float]:
    return [rng.uniform(-1, 1) for _ in range(dimension)]


def docs_cosine(rng: random.Random) -> list[Record]:
    """700 records of 8 dimensions.

    `code` is text in 50 records and int in the others. `reviewer` is in 70
    records only. 20 records have no document.
    """
    records = []
    for i in range(700):
        category = CATEGORIES[i % len(CATEGORIES)]
        metadata: dict[str, Any] = {
            "category": category,
            "year": 2000 + i % 25,
            "tags": rng.sample(TAGS, k=1 + i % 3),
            "code": f"C-{i}" if i % 14 == 0 else i,
            "score": rng.random(),
            "weights": [rng.random() for _ in range(3)],
            "published": i % 2 == 0,
        }
        if i % 10 == 3:
            metadata["reviewer"] = f"reviewer-{i % 7}"
        records.append(
            Record(
                id=f"doc-{i:04d}",
                embedding=_vector(rng, 8),
                document=None if i % 35 == 0 else f"Document {i} is a {category}.",
                metadata=metadata,
                uri=f"https://example.com/docs/{i}",
            )
        )
    return records


def tiny_l2(rng: random.Random, start: int = 0, count: int = 10) -> list[Record]:
    """`count` records of 4 dimensions, numbered from `start`. Their ids are UUIDs."""
    return [
        Record(
            id=str(uuid.UUID(int=rng.getrandbits(128), version=4)),
            embedding=_vector(rng, 4),
            document=f"Tiny record {i}.",
            metadata={"position": i},
            uri=f"https://example.com/tiny/{i}",
        )
        for i in range(start, start + count)
    ]


def docs_rq(rng: random.Random) -> list[Record]:
    """300 records of 32 dimensions."""
    return [
        Record(
            id=f"rq-{i:03d}",
            embedding=_vector(rng, 32),
            document=f"Record {i} of the rq collection.",
            metadata={"group": i % 5},
            uri=f"https://example.com/rq/{i}",
        )
        for i in range(300)
    ]


def docs_ip(rng: random.Random) -> list[Record]:
    """40 records of 8 dimensions with unit length.

    With unit vectors the largest inner product with a record's own vector is
    that record's, so a nearest-neighbor query by its vector returns it first.
    """
    records = []
    for i in range(40):
        vector = _vector(rng, 8)
        norm = math.sqrt(sum(x * x for x in vector))
        records.append(
            Record(
                id=f"ip-{i:02d}",
                embedding=[x / norm for x in vector],
                document=f"Record {i} of the ip collection.",
                metadata={"group": i % 4},
                uri=f"https://example.com/ip/{i}",
            )
        )
    return records


def _recreate(
    http: httpx.Client, name: str, space: str | None, records: list[Record]
) -> None:
    # Chroma 1.5.9 keeps the array metadata rows of a deleted collection. A
    # record created later on the same server can then show an array key that
    # was never sent for it. A fresh container has no such rows.
    deleted = http.delete(f"/collections/{name}")
    if deleted.status_code != 404:
        deleted.raise_for_status()
    body: dict[str, Any] = {"name": name}
    if space is not None:
        body["configuration"] = {"hnsw": {"space": space}}
    created = http.post("/collections", json=body)
    created.raise_for_status()
    _add(http, created.json()["id"], records)


def _add(http: httpx.Client, collection_id: str, records: list[Record]) -> None:
    for start in range(0, len(records), ADD_BATCH):
        batch = records[start : start + ADD_BATCH]
        added = http.post(
            f"/collections/{collection_id}/add",
            json={
                "ids": [r.id for r in batch],
                "embeddings": [r.embedding for r in batch],
                "documents": [r.document for r in batch],
                "metadatas": [r.metadata for r in batch],
                "uris": [r.uri for r in batch],
            },
        )
        added.raise_for_status()


def _test_server() -> httpx.Client:
    """A client of the test server. Raises ValueError for any other CHROMA_URL."""
    if os.environ.get("CHROMA_URL") != TEST_URL:
        raise ValueError(
            "the seed changes collections, so it runs only against the "
            f"test server; set CHROMA_URL={TEST_URL}"
        )
    return httpx.Client(base_url=TEST_URL + DATABASE_PATH, timeout=60)


def _collection_id(http: httpx.Client, name: str) -> str:
    found = http.get(f"/collections/{name}")
    found.raise_for_status()
    return str(found.json()["id"])


def seed() -> None:
    """Delete and recreate the COLLECTIONS on the test server."""
    rng = random.Random(0)
    with _test_server() as http:
        _recreate(http, "docs-cosine", "cosine", docs_cosine(rng))
        _recreate(http, "tiny.l2", None, tiny_l2(rng))
        _recreate(http, "docs-rq", "cosine", docs_rq(rng))
        _recreate(http, "docs-ip", "ip", docs_ip(rng))


def _extra_tiny_l2(start: int, count: int) -> list[Record]:
    return tiny_l2(random.Random(start), start, count)


def extra_tiny_l2_ids(start: int, count: int) -> list[str]:
    """The ids that `add_to_tiny_l2(start, count)` adds. Sends no request."""
    return [r.id for r in _extra_tiny_l2(start, count)]


def add_to_tiny_l2(start: int, count: int) -> None:
    """Add `count` records numbered from `start` to tiny.l2."""
    with _test_server() as http:
        _add(http, _collection_id(http, "tiny.l2"), _extra_tiny_l2(start, count))


def delete_from_tiny_l2(ids: list[str]) -> None:
    """Delete the records with `ids` from tiny.l2. An id that is missing is no error.

    Chroma answers 200 for ids that do not exist and 400 for an empty list,
    so an empty list sends no request.
    """
    if not ids:
        return
    with _test_server() as http:
        collection_id = _collection_id(http, "tiny.l2")
        deleted = http.post(f"/collections/{collection_id}/delete", json={"ids": ids})
        deleted.raise_for_status()


def records(name: str, limit: int = 10) -> list[Record]:
    """The first `limit` records of the collection `name`, read from the server."""
    with _test_server() as http:
        found = http.post(
            f"/collections/{_collection_id(http, name)}/get",
            json={
                "limit": limit,
                "include": ["embeddings", "documents", "metadatas", "uris"],
            },
        )
        found.raise_for_status()
        rows = found.json()
    return [
        Record(
            id=rows["ids"][i],
            embedding=rows["embeddings"][i],
            document=rows["documents"][i],
            metadata=rows["metadatas"][i] or {},
            uri=rows["uris"][i],
        )
        for i in range(len(rows["ids"]))
    ]


def count(name: str) -> int:
    """The number of records in the collection `name`."""
    with _test_server() as http:
        collection_id = _collection_id(http, name)
        counted = http.get(f"/collections/{collection_id}/count")
        counted.raise_for_status()
        return int(counted.json())


def tiny_l2_count() -> int:
    """The number of records in tiny.l2."""
    return count("tiny.l2")


if __name__ == "__main__":
    try:
        seed()
    except (ValueError, httpx.HTTPError) as exc:
        sys.exit(f"seed.py: {exc}")
    print(f"Seeded {', '.join(COLLECTIONS)}.")
