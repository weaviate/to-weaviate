"""Create the Chroma collection to-weaviate-ci that the cloud suite reads.

A person runs this script once by hand, before Chroma Cloud stops accepting
writes on 2026-12-07: export CHROMA_API_KEY, CHROMA_TENANT and
CHROMA_DATABASE, then run `python tests/cloud/chroma/seed.py`. CHROMA_URL and
CHROMA_TOKEN_HEADER have the defaults of to-weaviate. The script reads the
environment only, not `.env`.

It stops without a change when the collection exists, and it deletes nothing.
"""

import os
import random
import sys
from collections.abc import Mapping
from typing import Any
from urllib.parse import quote

import httpx

from to_weaviate.settings import MissingCredential, chroma_settings
from to_weaviate.sources.chroma.config import distance_of

COLLECTION = "to-weaviate-ci"
# This request body is not verified against Chroma Cloud. Chroma 1.5.9 answers
# it with an HNSW collection whose space is cosine.
CREATE_BODY = {"name": COLLECTION, "configuration": {"spann": {"space": "cosine"}}}
RECORDS = 2000
DIMENSION = 384
ADD_BATCH = 300
MAX_MESSAGE_CHARS = 500
CATEGORIES = ("guide", "reference", "tutorial")
TAGS = ("red", "green", "blue", "small", "large")


class SeedError(Exception):
    pass


def records() -> dict[str, list[Any]]:
    """The columns of the add calls for all records.

    Every record has a document and the metadata keys `category` (text),
    `position` (int) and `tags` (list of strings).
    """
    rng = random.Random(0)
    columns: dict[str, list[Any]] = {
        "ids": [],
        "embeddings": [],
        "documents": [],
        "metadatas": [],
    }
    for i in range(RECORDS):
        category = CATEGORIES[i % len(CATEGORIES)]
        columns["ids"].append(f"ci-{i:04d}")
        columns["embeddings"].append([rng.uniform(-1, 1) for _ in range(DIMENSION)])
        columns["documents"].append(f"Record {i} of the cloud suite is a {category}.")
        columns["metadatas"].append(
            {"category": category, "position": i, "tags": rng.sample(TAGS, k=1 + i % 3)}
        )
    return columns


def _checked(response: httpx.Response) -> httpx.Response:
    if response.status_code >= 400:
        raise SeedError(
            f"{response.request.method} {response.request.url.path} returned "
            f"HTTP {response.status_code}: {response.text[:MAX_MESSAGE_CHARS]}"
        )
    return response


def seed(env: Mapping[str, str]) -> None:
    """Create to-weaviate-ci in the database that `env` names and add the records."""
    settings = chroma_settings(env)
    if settings.tenant is None:
        raise SeedError("CHROMA_TENANT is not set")
    if settings.database is None:
        raise SeedError("CHROMA_DATABASE is not set")
    headers: dict[str, str] = {}
    if settings.api_key is not None:
        value = settings.api_key
        if settings.token_header.lower() == "authorization":
            value = f"Bearer {value}"
        headers[settings.token_header] = value
    base_url = (
        f"{settings.url.rstrip('/')}/api/v2/tenants/{quote(settings.tenant, safe='')}"
        f"/databases/{quote(settings.database, safe='')}"
    )
    with httpx.Client(base_url=base_url, headers=headers, timeout=60) as http:
        existing = http.get(f"/collections/{COLLECTION}")
        if existing.status_code != 404:
            _checked(existing)
            raise SeedError(
                f"{COLLECTION} exists in the database {settings.database}. "
                "The seed changed nothing."
            )
        # Without get_or_create, Chroma refuses a collection that exists.
        _checked(http.post("/collections", json=CREATE_BODY))
        collection = _checked(http.get(f"/collections/{COLLECTION}")).json()
        distance = distance_of(collection)
        if distance != "cosine":
            raise SeedError(
                f"Chroma created {COLLECTION} with the distance {distance}, not "
                "cosine. It holds no records. Delete it, change CREATE_BODY, and "
                "run the seed again."
            )
        columns = records()
        for start in range(0, RECORDS, ADD_BATCH):
            batch = {k: v[start : start + ADD_BATCH] for k, v in columns.items()}
            try:
                _checked(http.post(f"/collections/{collection['id']}/add", json=batch))
            except (SeedError, httpx.HTTPError) as exc:
                raise SeedError(
                    f"{exc}\n{COLLECTION} is incomplete. Delete it, then run the "
                    "seed again."
                ) from exc


if __name__ == "__main__":
    try:
        seed(os.environ)
    except (SeedError, MissingCredential, ValueError, httpx.HTTPError) as exc:
        sys.exit(f"seed.py: {exc}")
    print(f"Created {COLLECTION} with {RECORDS} records.")
