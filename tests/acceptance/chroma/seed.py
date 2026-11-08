"""Seed the local Chroma test server with the collections of the acceptance suite.

Run it as a script, or call `seed()`. Each run deletes and recreates the
collections, so every run sends the same data. On a distributed Chroma the
suite then calls `wait_indexed`. `add_to_tiny_l2` and `delete_from_tiny_l2`
add records to tiny.l2 and delete them again; `restore_tiny_l2` makes tiny.l2
again from records read before. `make_collection`, `add_records`,
`delete_records` and `delete_collection` serve the cases that make their own
collections. Every function here that sends a request reads CHROMA_URL and
runs only when it is TEST_URL.
"""

import math
import os
import random
import sys
import time
import uuid
from collections.abc import Iterable
from dataclasses import dataclass, replace
from typing import Any

import httpx

DATABASE_PATH = "/api/v2/tenants/default_tenant/databases/default_database"
ADD_BATCH = 300
# The records of one add when a collection is made. Chroma 1.5.9 accepts at
# most 5461 (max_batch_size of /api/v2/pre-flight-checks).
LARGE_ADD_BATCH = 5000
# Records of the resume collection. A copy in both mode embeds about 300
# records a second on the CPU of a development machine, so a full copy runs
# about 28 seconds there.
RESUME_COUNT = 8000
TOPICS = ("rivers", "mountains", "cities", "forests", "deserts", "oceans")
TEST_URL = "http://localhost:8001"
# The collections that `seed` makes and that copy without a reject, in the
# order it makes them. It also makes EDGE and BULK.
COLLECTIONS = (
    "docs-cosine",
    "tiny.l2",
    "docs-rq",
    "docs-ip",
    "tenant-a",
    "tenant-b",
    "rejects",
    "openai-proxy",
    "resume",
    "empty",
    "Mixed-Case",
    "mixed-case",
    "1-leading-digit",
    "class",
)
# Collections whose names need a change for Weaviate, made in this order.
RENAMED_COLLECTIONS = ("Mixed-Case", "mixed-case", "1-leading-digit", "class")
# Values at the limits of their type. Its records with the int 2**63 - 1 are
# rejects by design, so it is not in COLLECTIONS.
EDGE = "edge"
# The large collection. seed makes it, but it is not in COLLECTIONS, so the
# test that copies every collection does not copy it.
BULK = "bulk"
BULK_COUNT = 50_000
# Every collection that `seed` makes.
SEEDED = (*COLLECTIONS, EDGE, BULK)
# The collections of the cases of test_source_order.py, which make them.
ORDER = "order"
ORDER_COUNT = 1000
BURST = "burst"
BURST_COUNT = 12_000
# A distributed Chroma compacts every 10 s, at most 10,000 records of a
# collection per run (compaction_interval_sec and max_compaction_size of
# rust/worker/chroma_config.yaml in the Chroma repository), so bulk needs five
# runs. Its frontend keeps a collection, with the log position that
# indexing_status reads, for 60 s (cache_ttl_secs of
# rust/frontend/sample_configs/distributed.yaml), so the status shows a
# finished run up to 60 s late.
INDEXING_TIMEOUT = 300.0
INDEXING_POLL = 2.0
CATEGORIES = ("guide", "reference", "tutorial")
TAGS = ("red", "green", "blue", "small", "large")


@dataclass(frozen=True)
class Record:
    id: str
    embedding: list[float]
    document: str | None
    metadata: dict[str, Any]
    uri: str | None


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


def tenant_records(rng: random.Random, tenant: str, count: int) -> list[Record]:
    """`count` records of 8 dimensions for the tenant collection `tenant-<tenant>`.

    Only the records of tenant b have the key `region`.
    """
    records = []
    for i in range(count):
        metadata: dict[str, Any] = {"position": i}
        if tenant == "b":
            metadata["region"] = ("north", "south")[i % 2]
        records.append(
            Record(
                id=f"{tenant}-{i:02d}",
                embedding=_vector(rng, 8),
                document=f"Record {i} of tenant {tenant}.",
                metadata=metadata,
                uri=f"https://example.com/{tenant}/{i}",
            )
        )
    return records


def rejects(rng: random.Random) -> list[Record]:
    """301 records of 4 dimensions; the int key `rank` holds 300.5 in rej-300.

    rej-300 is the first record of the second page, so an inspection of the
    first page only sees int values. rej-005 has no document.
    """
    return [
        Record(
            id=f"rej-{i:03d}",
            embedding=_vector(rng, 4),
            document=None if i == 5 else f"Record {i} of the rejects collection.",
            metadata={"rank": 300.5 if i == 300 else i},
            uri=f"https://example.com/rejects/{i}",
        )
        for i in range(301)
    ]


def resume(rng: random.Random) -> list[Record]:
    """RESUME_COUNT records of 64 dimensions, each with a document."""
    return [
        Record(
            id=f"res-{i:05d}",
            embedding=_vector(rng, 64),
            document=f"Record {i} of the resume collection is about "
            f"{TOPICS[i % len(TOPICS)]}.",
            metadata={"position": i},
            uri=f"https://example.com/resume/{i}",
        )
        for i in range(RESUME_COUNT)
    ]


# Values at the limits of their type. Chroma returns -0.0 as 0.0. Weaviate
# 1.40.4 returns an int above 2**53 rounded to a double. copy rejects
# 2**63 - 1, whose double is 2**63.
INT_EDGES = (0, -1, 2**53 + 1, 2**63 - 1, -(2**63))
FLOAT_EDGES = (1e308, -1e308, -0.0, 5e-324, 0.1)
FLAGS = (True, False, "true", "false")
# Metadata keys that are reserved words or that need a change to be property names.
ODD_KEYS = (
    "ID",
    "Id",
    "id",
    "_additional",
    "document",
    "uri",
    "vector",
    "class",
    "1st key",
    "my key",
    "日本",
    "created_temp",
    "a__backup",
    "x_nullState",
)
LONG_DOCUMENT_BYTES = 200_000
UNICODE_IDS = ("edge-ключ-1", "edge-日本-2", "edge-🙂-3")
# CJK text, an e and an a with combining marks, and Hebrew and Arabic words.
UNICODE_DOCUMENT = (
    "日本語のテキスト e\u0301 a\u0308 "
    "\u05e9\u05dc\u05d5\u05dd \u0627\u0644\u0633\u0644\u0627\u0645"
)


def edge(rng: random.Random) -> list[Record]:
    """40 records of 8 dimensions with values at the limits of their type.

    The first three ids hold non-ASCII text. Record 0 has an empty document,
    record 1 a long one, record 2 a document with combining marks and right
    to left text, record 3 none. `flag` is a boolean in some records and text
    in others; `num` is an int in some and a float in others.
    """
    documents: dict[int, str | None] = {
        0: "",
        1: "x" * LONG_DOCUMENT_BYTES,
        2: UNICODE_DOCUMENT,
        3: None,
    }
    records = []
    for i in range(40):
        metadata: dict[str, Any] = {
            "int_edge": INT_EDGES[i % len(INT_EDGES)],
            "float_edge": FLOAT_EDGES[i % len(FLOAT_EDGES)],
            "flag": FLAGS[i % len(FLAGS)],
            "num": (3, 2.5)[i % 2],
            "empty": "",
            "tags": (["a"], ["a", "b", "c"])[i % 2],
        }
        metadata.update({key: f"{key} {i}" for key in ODD_KEYS})
        records.append(
            Record(
                id=UNICODE_IDS[i] if i < len(UNICODE_IDS) else f"edge-{i:02d}",
                embedding=_vector(rng, 8),
                document=documents.get(i, f"Edge record {i}."),
                metadata=metadata,
                uri=f"https://example.com/edge/{i}",
            )
        )
    return records


def bulk(rng: random.Random) -> list[Record]:
    """BULK_COUNT records of 32 dimensions, each with a document."""
    return [
        Record(
            id=f"bulk-{i:05d}",
            embedding=_vector(rng, 32),
            document=f"Record {i} of the bulk collection.",
            metadata={"position": i},
            uri=None,
        )
        for i in range(BULK_COUNT)
    ]


def renamed(rng: random.Random, name: str) -> list[Record]:
    """5 records of 8 dimensions for the collection `name`."""
    return [
        Record(
            id=f"{name}-{i}",
            embedding=_vector(rng, 8),
            document=f"Record {i} of {name}.",
            metadata={"position": i},
            uri=None,
        )
        for i in range(5)
    ]


def order_records(start: int, count: int) -> list[Record]:
    """`count` records of 4 dimensions for ORDER, numbered from `start`.

    Their ids are UUIDs, so the order of the ids differs from the order of
    the records.
    """
    rng = random.Random(start)
    return [
        Record(
            id=str(uuid.UUID(int=rng.getrandbits(128), version=4)),
            embedding=_vector(rng, 4),
            document=f"Record {i} of the order collection.",
            metadata={"position": i},
            uri=None,
        )
        for i in range(start, start + count)
    ]


def burst_records() -> list[Record]:
    """BURST_COUNT records of 16 dimensions for BURST."""
    rng = random.Random(BURST_COUNT)
    return [
        Record(
            id=f"burst-{i:05d}",
            embedding=_vector(rng, 16),
            document=f"Record {i} of the burst collection.",
            metadata={"position": i},
            uri=None,
        )
        for i in range(BURST_COUNT)
    ]


# Made-up secrets of the embedding function of openai-proxy. The tool must
# neither store nor print them.
PROXY_USER = "b8-proxy-user"
PROXY_PASSWORD = "made-up-password-b8"
PROXY_QUERY_TOKEN = "made-up-query-token-b8"
PROXY_HEADER_TOKEN = "made-up-header-token-b8"
PROXY_API_BASE = (
    f"https://{PROXY_USER}:{PROXY_PASSWORD}@proxy.example.invalid/openai/v1"
    f"?sig={PROXY_QUERY_TOKEN}"
)
# The configuration that chromadb 1.5.9 sends for an OpenAIEmbeddingFunction
# (get_config). With api_type azure the tool maps it to no vectorizer, so no
# provider key is needed.
OPENAI_PROXY_FUNCTION = {
    "type": "known",
    "name": "openai",
    "config": {
        "api_key_env_var": "CHROMA_OPENAI_API_KEY",
        "model_name": "text-embedding-3-small",
        "organization_id": None,
        "api_base": PROXY_API_BASE,
        "api_type": "azure",
        "api_version": "2024-02-01",
        "deployment_id": "b8-deployment",
        "default_headers": {"X-Proxy-Token": PROXY_HEADER_TOKEN},
        "dimensions": None,
    },
}


def openai_proxy(rng: random.Random) -> list[Record]:
    """10 records of 8 dimensions."""
    return [
        Record(
            id=f"proxy-{i}",
            embedding=_vector(rng, 8),
            document=f"Record {i} of the openai-proxy collection.",
            metadata={"position": i},
            uri=None,
        )
        for i in range(10)
    ]


def _recreate(
    http: httpx.Client,
    name: str,
    space: str | None,
    records: list[Record],
    embedding_function: dict[str, Any] | None = None,
    batch_size: int = LARGE_ADD_BATCH,
) -> None:
    # Chroma 1.5.9 keeps the array metadata rows of a deleted collection. A
    # record created later on the same server can then show an array key that
    # was never sent for it. A fresh container has no such rows.
    _delete_collection(http, name)
    body: dict[str, Any] = {"name": name}
    if space is not None:
        body["configuration"] = {"hnsw": {"space": space}}
    if embedding_function is not None:
        body.setdefault("configuration", {})["embedding_function"] = embedding_function
    created = http.post("/collections", json=body)
    created.raise_for_status()
    _add(http, created.json()["id"], records, batch_size)


def _delete_collection(http: httpx.Client, name: str) -> None:
    deleted = http.delete(f"/collections/{name}")
    if deleted.status_code != 404:
        deleted.raise_for_status()


def _add(
    http: httpx.Client,
    collection_id: str,
    records: list[Record],
    batch_size: int = ADD_BATCH,
) -> None:
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
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
    """Delete and recreate the SEEDED collections on the test server."""
    rng = random.Random(0)
    with _test_server() as http:
        _recreate(http, "docs-cosine", "cosine", docs_cosine(rng))
        _recreate(http, "tiny.l2", None, tiny_l2(rng))
        _recreate(http, "docs-rq", "cosine", docs_rq(rng))
        _recreate(http, "docs-ip", "ip", docs_ip(rng))
        _recreate(http, "tenant-a", "cosine", tenant_records(rng, "a", 30))
        _recreate(http, "tenant-b", "cosine", tenant_records(rng, "b", 20))
        _recreate(http, "rejects", "cosine", rejects(rng))
        _recreate(
            http, "openai-proxy", "cosine", openai_proxy(rng), OPENAI_PROXY_FUNCTION
        )
        _recreate(http, "resume", "cosine", resume(rng))
        _recreate(http, "empty", "cosine", [])
        _recreate(http, EDGE, "cosine", edge(rng))
        for name in RENAMED_COLLECTIONS:
            _recreate(http, name, "cosine", renamed(rng, name))
        _recreate(http, BULK, "cosine", bulk(rng))


def wait_indexed(names: Iterable[str], timeout: float = INDEXING_TIMEOUT) -> None:
    """Wait until distributed Chroma has compacted every write to collections `names`.

    Distributed Chroma reads a write from its log until a compaction run
    moves it into the segments of the collection. A run of more than 5,000
    records numbers them again in another order (max_partition_size of the
    worker config), so pages read before and after such a run can skip or
    repeat records. indexing_status counts the writes that no run has
    compacted. It reads the collection from the frontend's cache, as `get`
    does, so at 0 `get` reads the compacted collection, in one order until
    the next write. A single-node Chroma answers indexing_status with HTTP
    500, so the suite calls this on distributed Chroma only. After `timeout`
    s, raises TimeoutError with the collections that still have such writes.
    """
    with _test_server() as http:
        ids = {name: _collection_id(http, name) for name in names}
        deadline = time.monotonic() + timeout
        while True:
            pending = {}
            for name, collection_id in ids.items():
                found = http.get(f"/collections/{collection_id}/indexing_status")
                found.raise_for_status()
                status = found.json()
                if status["num_unindexed_ops"] > 0:
                    pending[name] = status
            if not pending:
                return
            if time.monotonic() > deadline:
                raise TimeoutError(
                    "distributed Chroma has not compacted every write after "
                    f"{timeout:.0f} s: "
                    + ", ".join(
                        f"{name} has {s['num_unindexed_ops']} of {s['total_ops']} "
                        "operations not compacted"
                        for name, s in pending.items()
                    )
                )
            # A compacted collection stays compacted while nothing writes.
            ids = {name: ids[name] for name in pending}
            time.sleep(INDEXING_POLL)


def _extra_tiny_l2(start: int, count: int) -> list[Record]:
    return tiny_l2(random.Random(start), start, count)


def extra_tiny_l2_ids(start: int, count: int) -> list[str]:
    """The ids that `add_to_tiny_l2(start, count)` adds. Sends no request."""
    return [r.id for r in _extra_tiny_l2(start, count)]


def add_to_tiny_l2(
    start: int, count: int, metadata: dict[str, Any] | None = None
) -> None:
    """Add `count` records numbered from `start` to tiny.l2.

    The keys of `metadata` are added to the metadata of each record.
    """
    added = [
        replace(r, metadata={**r.metadata, **(metadata or {})})
        for r in _extra_tiny_l2(start, count)
    ]
    add_records("tiny.l2", added)


def delete_from_tiny_l2(ids: list[str]) -> None:
    """Delete the records with `ids` from tiny.l2. An id that is missing is no error."""
    delete_records("tiny.l2", ids)


def restore_tiny_l2(records: list[Record]) -> None:
    """Delete tiny.l2 and make it again with `records`, in their order.

    Chroma puts a record that is added again after the others, so only a new
    collection gives the records back their order.
    """
    with _test_server() as http:
        _recreate(http, "tiny.l2", None, records)


def make_collection(name: str, records: list[Record], batch_size: int) -> None:
    """Delete the collection `name` and make it again with `records`, in their order.

    The records go in adds of `batch_size`. The collection has Chroma's
    default distance.
    """
    with _test_server() as http:
        _recreate(http, name, None, records, batch_size=batch_size)


def add_records(name: str, records: list[Record]) -> None:
    """Add `records` to the collection `name`, in adds of ADD_BATCH."""
    with _test_server() as http:
        _add(http, _collection_id(http, name), records)


def delete_records(name: str, ids: list[str]) -> None:
    """Delete the records with `ids` from the collection `name`.

    An id that is missing is no error: Chroma answers 200 for ids that do not
    exist and 400 for an empty list, so an empty list sends no request.
    """
    if not ids:
        return
    with _test_server() as http:
        collection_id = _collection_id(http, name)
        deleted = http.post(f"/collections/{collection_id}/delete", json={"ids": ids})
        deleted.raise_for_status()


def delete_collection(name: str) -> None:
    """Delete the collection `name`. A collection that does not exist is no error."""
    with _test_server() as http:
        _delete_collection(http, name)


def page_ids(name: str, offset: int, limit: int) -> list[str]:
    """The ids that `get` with `offset` and `limit` returns for collection `name`."""
    with _test_server() as http:
        found = http.post(
            f"/collections/{_collection_id(http, name)}/get",
            json={"offset": offset, "limit": limit, "include": []},
        )
        found.raise_for_status()
        ids: list[str] = found.json()["ids"]
        return ids


def collection_state(name: str) -> dict[str, Any]:
    """The `version` and the `log_position` of the collection `name`.

    A distributed Chroma raises both with each compaction run. A single-node
    Chroma answers 0 for both.
    """
    with _test_server() as http:
        found = http.get(f"/collections/{name}")
        found.raise_for_status()
        document = found.json()
    return {key: document[key] for key in ("version", "log_position")}


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
    print(f"Seeded {', '.join(SEEDED)}.")
