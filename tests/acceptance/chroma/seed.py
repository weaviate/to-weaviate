"""Seed the local Chroma test server with the collections of the acceptance suite.

Run it as a script, or call `seed()`. Both read CHROMA_URL and run only when it
is TEST_URL. Each run deletes and recreates the collections, so every run sends
the same data.
"""

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

    `code` is text in 50 records and int in the others. 20 records have no
    document.
    """
    records = []
    for i in range(700):
        category = CATEGORIES[i % len(CATEGORIES)]
        records.append(
            Record(
                id=f"doc-{i:04d}",
                embedding=_vector(rng, 8),
                document=None if i % 35 == 0 else f"Document {i} is a {category}.",
                metadata={
                    "category": category,
                    "year": 2000 + i % 25,
                    "tags": rng.sample(TAGS, k=1 + i % 3),
                    "code": f"C-{i}" if i % 14 == 0 else i,
                    "score": rng.random(),
                    "weights": [rng.random() for _ in range(3)],
                },
                uri=f"https://example.com/docs/{i}",
            )
        )
    return records


def tiny_l2(rng: random.Random) -> list[Record]:
    """10 records of 4 dimensions. Their ids are UUIDs."""
    return [
        Record(
            id=str(uuid.UUID(int=rng.getrandbits(128), version=4)),
            embedding=_vector(rng, 4),
            document=f"Tiny record {i}.",
            metadata={"position": i},
            uri=f"https://example.com/tiny/{i}",
        )
        for i in range(10)
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
    collection_id = created.json()["id"]
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


def seed() -> None:
    """Delete and recreate docs-cosine, tiny.l2 and docs-rq on the test server."""
    if os.environ.get("CHROMA_URL") != TEST_URL:
        raise ValueError(
            "the seed deletes and recreates collections, so it runs only "
            f"against the test server; set CHROMA_URL={TEST_URL}"
        )
    rng = random.Random(0)
    with httpx.Client(base_url=TEST_URL + DATABASE_PATH, timeout=60) as http:
        _recreate(http, "docs-cosine", "cosine", docs_cosine(rng))
        _recreate(http, "tiny.l2", None, tiny_l2(rng))
        _recreate(http, "docs-rq", "cosine", docs_rq(rng))


if __name__ == "__main__":
    try:
        seed()
    except (ValueError, httpx.HTTPError) as exc:
        sys.exit(f"seed.py: {exc}")
    print("Seeded docs-cosine, tiny.l2 and docs-rq.")
