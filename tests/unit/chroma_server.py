"""A fake Chroma server for httpx.MockTransport.

The CLI tests that use it run the real ChromaClient and ChromaSource, so the
paging, the retries and the paths of the client are under test.
"""

import json
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import parse_qs, unquote

import httpx
from fakes import FakeFactories

from to_weaviate.settings import ChromaSettings
from to_weaviate.sources.base import Record
from to_weaviate.sources.chroma.client import ChromaClient
from to_weaviate.sources.chroma.source import ChromaSource

# A fault is an answer or an error that replaces the normal answer of one
# request. Its key is the request number (1-based), a (kind, offset) pair or a
# kind alone. Kinds: identity, databases, collections, count, get, query.
Fault = httpx.Response | httpx.TransportError
FaultKey = int | str | tuple[str, int | None]


@dataclass(frozen=True)
class Call:
    kind: str
    offset: int | None
    limit: int | None
    include: list[str] | None
    raw_path: str
    headers: dict[str, str]


@dataclass
class ChromaServer:
    """One tenant, one database and one collection over `records`.

    A /get answer holds at most `page_cap` rows when it is set. Each key of
    `faults` maps to a list; a request that matches takes the first item of
    the list. `calls` records every request. Without `dimension_listed`, the
    collection listing gives the dimension as null.
    """

    records: list[Record]
    tenant: str = "default_tenant"
    database: str = "default_database"
    collection_id: str = "chroma-docs"
    name: str = "docs"
    space: str = "cosine"
    page_cap: int | None = None
    dimension_listed: bool = True
    faults: dict[FaultKey, list[Fault]] = field(default_factory=dict)
    calls: list[Call] = field(default_factory=list)

    def collection(self) -> dict[str, Any]:
        vectors = [r.vector for r in self.records if r.vector is not None]
        return {
            "id": self.collection_id,
            "name": self.name,
            "dimension": (
                len(vectors[0]) if vectors and self.dimension_listed else None
            ),
            "configuration_json": {"hnsw": {"space": self.space}},
            "metadata": None,
            "tenant": self.tenant,
            "database": self.database,
        }

    def gets(self, include: str | None = None) -> list[Call]:
        """The /get calls, only those that include `include` when it is given."""
        return [
            c
            for c in self.calls
            if c.kind == "get" and (include is None or include in (c.include or []))
        ]

    def handler(self, request: httpx.Request) -> httpx.Response:
        raw_path = request.url.raw_path.decode().partition("?")[0]
        parts = [unquote(p) for p in raw_path.split("/")[3:]]
        body = json.loads(request.content) if request.content else {}
        query = {k: v[0] for k, v in parse_qs(request.url.query.decode()).items()}
        kind, answer = self._route(parts, body, query)
        offset = body.get("offset") if kind == "get" else None
        if kind == "collections":
            offset = int(query.get("offset", 0))
        self.calls.append(
            Call(
                kind,
                offset,
                body.get("limit"),
                body.get("include"),
                raw_path,
                dict(request.headers),
            )
        )
        fault = self._fault(len(self.calls), kind, offset)
        if isinstance(fault, httpx.TransportError):
            raise fault
        if fault is not None:
            return fault
        return answer()

    def _fault(self, number: int, kind: str, offset: int | None) -> Fault | None:
        for key in (number, (kind, offset), kind):
            queued = self.faults.get(key)
            if queued:
                return queued.pop(0)
        return None

    def _route(
        self, parts: list[str], body: dict[str, Any], query: dict[str, str]
    ) -> tuple[str, Any]:
        if parts == ["auth", "identity"]:
            identity = {
                "user_id": "u",
                "tenant": self.tenant,
                "databases": [self.database],
            }
            return "identity", lambda: httpx.Response(200, json=identity)
        if len(parts) < 3 or parts[:1] != ["tenants"] or parts[1] != self.tenant:
            return "unknown", lambda: _not_found("tenant")
        if parts[2:] == ["databases"]:
            rows = [{"id": "db-1", "name": self.database, "tenant": self.tenant}]
            return "databases", lambda: httpx.Response(200, json=rows)
        if len(parts) < 5 or parts[3] != self.database:
            return "unknown", lambda: _not_found("database")
        if parts[4:] == ["collections"]:
            offset, limit = int(query.get("offset", 0)), int(query["limit"])
            rows = [self.collection()][offset : offset + limit]
            return "collections", lambda: httpx.Response(200, json=rows)
        if len(parts) != 7 or parts[5] != self.collection_id:
            return "unknown", lambda: _not_found("collection")
        kind = parts[6]
        if kind == "count":
            return kind, lambda: httpx.Response(200, json=len(self.records))
        if kind == "get":
            return kind, lambda: httpx.Response(200, json=self._get(body))
        if kind == "query":
            return kind, lambda: httpx.Response(200, json={"ids": [[]]})
        return "unknown", lambda: _not_found("route")

    def _get(self, body: dict[str, Any]) -> dict[str, Any]:
        rows = self.records
        where = body.get("where")
        if where:
            ((key, condition),) = where.items()
            rows = [r for r in rows if r.metadata.get(key) == condition["$eq"]]
        limit = body["limit"]
        if self.page_cap is not None:
            limit = min(limit, self.page_cap)
        offset = body["offset"]
        rows = rows[offset : offset + limit]
        include = body["include"]

        def column(name: str, values: list[Any]) -> list[Any] | None:
            return values if name in include else None

        return {
            "ids": [r.id for r in rows],
            "embeddings": column("embeddings", [r.vector for r in rows]),
            "documents": column("documents", [r.document for r in rows]),
            "uris": column("uris", [r.uri for r in rows]),
            "metadatas": column("metadatas", [r.metadata or None for r in rows]),
            "include": include,
        }


def _not_found(what: str) -> httpx.Response:
    return httpx.Response(
        404, json={"error": "NotFoundError", "message": f"{what} not found"}
    )


class ServerFactories(FakeFactories):
    """FakeFactories whose source is a ChromaSource over `server`.

    The tenant comes from CHROMA_TENANT or from /auth/identity, as in the CLI.
    `sleeps` holds the retry delays of the client.
    """

    def __init__(self, server: ChromaServer, **kwargs: Any) -> None:
        super().__init__(source=None, **kwargs)  # type: ignore[arg-type]
        self.server = server
        self.sleeps: list[float] = []

    def _source(self, settings: ChromaSettings) -> ChromaSource:  # type: ignore[override]
        self.source_calls += 1
        client = ChromaClient(
            settings,
            transport=httpx.MockTransport(self.server.handler),
            sleep=self.sleeps.append,
        )
        return ChromaSource(client, settings.tenant or client.identity()["tenant"])
