"""Create and change the Chroma collection `support-articles` of the README demo.

Usage:
  python docs/demo/seed.py [COUNT]   create the collection with COUNT records
  python docs/demo/seed.py change    change it, as an application would after
                                     the copy: add 5 records, update 5, delete 5

Reads CHROMA_URL (default http://localhost:8001) and refuses any host other
than localhost, because it deletes the collection first. Each record has a
document, four scalar metadata keys and a 1536-dimension unit vector, the
size of OpenAI's text-embedding-3-small. The vectors are random, with a
fixed seed, so every run creates the same collection.
"""

import math
import os
import random
import sys
from urllib.parse import urlsplit

import httpx

NAME = "support-articles"
DIMENSION = 1536
BATCH = 300
DATABASE_PATH = "/api/v2/tenants/default_tenant/databases/default_database"
DEFAULT_COUNT = 5000
# `change` adds the records numbered from DEFAULT_COUNT, updates the document
# of the first CHANGED records and deletes the CHANGED records after those.
CHANGED = 5

PRODUCTS = [
    "Billing",
    "Dashboards",
    "Exports",
    "Alerts",
    "Single sign-on",
    "Mobile app",
]
TASKS = [
    "set up {p} for a new team",
    "fix a {p} error after an upgrade",
    "move {p} settings between workspaces",
    "limit who can change {p}",
    "audit recent changes to {p}",
    "connect {p} to an external tool",
]
DETAILS = [
    "The steps apply to every plan.",
    "An administrator role is required.",
    "Changes take effect within five minutes.",
    "The old settings stay available for 30 days.",
    "Each workspace keeps its own copy.",
]


def unit_vector(rng: random.Random) -> list[float]:
    values = [rng.gauss(0.0, 1.0) for _ in range(DIMENSION)]
    norm = math.sqrt(sum(v * v for v in values))
    return [round(v / norm, 6) for v in values]


def records(count: int) -> list[dict]:
    rng = random.Random(7)
    out = []
    for i in range(count):
        product = PRODUCTS[i % len(PRODUCTS)]
        task = TASKS[(i // len(PRODUCTS)) % len(TASKS)].format(p=product.lower())
        out.append(
            {
                "id": f"kb-{i:05d}",
                "document": f"How to {task}. {rng.choice(DETAILS)}",
                "metadata": {
                    "product": product,
                    "year": 2019 + i % 7,
                    "views": rng.randint(10, 50_000),
                    "verified": i % 3 != 0,
                },
                "embedding": unit_vector(rng),
            }
        )
    return out


def _url() -> str:
    url = os.environ.get("CHROMA_URL", "http://localhost:8001")
    if urlsplit(url).hostname not in ("localhost", "127.0.0.1"):
        sys.exit(
            f"seed.py changes {NAME}, so it runs only against localhost, not {url}"
        )
    return url


def _client() -> httpx.Client:
    return httpx.Client(base_url=_url() + DATABASE_PATH, timeout=60)


def _add(http: httpx.Client, collection_id: str, rows: list[dict]) -> None:
    for start in range(0, len(rows), BATCH):
        batch = rows[start : start + BATCH]
        added = http.post(
            f"/collections/{collection_id}/add",
            json={
                key + "s": [r[key] for r in batch]
                for key in ("id", "document", "metadata", "embedding")
            },
        )
        added.raise_for_status()


def create(count: int) -> None:
    with _client() as http:
        deleted = http.delete(f"/collections/{NAME}")
        if deleted.status_code != 404:
            deleted.raise_for_status()
        created = http.post(
            "/collections",
            json={"name": NAME, "configuration": {"hnsw": {"space": "cosine"}}},
        )
        created.raise_for_status()
        _add(http, created.json()["id"], records(count))
    print(f"Created {NAME} with {count} records of dimension {DIMENSION} at {_url()}")


def change() -> None:
    """Add CHANGED records, update the documents of CHANGED and delete CHANGED."""
    rows = records(DEFAULT_COUNT + CHANGED)
    updated = rows[:CHANGED]
    removed = rows[CHANGED : 2 * CHANGED]
    with _client() as http:
        found = http.get(f"/collections/{NAME}")
        found.raise_for_status()
        collection_id = found.json()["id"]
        _add(http, collection_id, rows[DEFAULT_COUNT:])
        changed = http.post(
            f"/collections/{collection_id}/update",
            json={
                "ids": [r["id"] for r in updated],
                "documents": ["Updated: " + r["document"] for r in updated],
            },
        )
        changed.raise_for_status()
        deleted = http.post(
            f"/collections/{collection_id}/delete",
            json={"ids": [r["id"] for r in removed]},
        )
        deleted.raise_for_status()
    print(
        f"Changed {NAME}: added {CHANGED} records, updated {CHANGED}, deleted {CHANGED}"
    )


def main() -> None:
    if sys.argv[1:] == ["change"]:
        change()
    else:
        create(int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_COUNT)


if __name__ == "__main__":
    main()
