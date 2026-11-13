import time
from collections.abc import Callable
from typing import Any
from urllib.parse import quote

import httpx

from to_weaviate.output import redact
from to_weaviate.retry import with_retry
from to_weaviate.settings import ChromaSettings, url_passwords, url_tokens

CHROMA_MAX_PAGE = 300
COLLECTIONS_PAGE = 1000
RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
MAX_MESSAGE_CHARS = 500
MAX_CONTENT_TYPE_CHARS = 100
# The names of the tenant and the database that every Chroma server creates.
DEFAULT_TENANT = "default_tenant"
DEFAULT_DATABASE = "default_database"
# Chroma 0.5.16 to 0.6.3 with token authentication answer the identity with
# this value for the tenant and for each database. It is not a name: a
# request for the tenant or the database `*` finds no collection.
ANY = "*"
REDIRECT_HINT = "Set CHROMA_URL to the address the server redirects to."
NOT_JSON_HINT = (
    "Check that CHROMA_URL points at the Chroma API and not at a proxy or a web page."
)
DATABASES_FORBIDDEN_HINT = (
    "Check CHROMA_API_KEY in the environment or in .env. A key for one database "
    "cannot list the databases: set CHROMA_DATABASE or use --database."
)
COLLECTION_GONE_HINT = (
    "The collection is gone from Chroma, or it was created again with a new id. "
    "Run to-weaviate inspect chroma --force to inspect it again."
)
DATABASES_NOT_LISTED = (
    "The Chroma server cannot list databases (HTTP 405, as Chroma before 0.6.3 "
    "answers)"
)
DATABASES_NOT_LISTED_HINT = "Pass --database or set CHROMA_DATABASE."
NO_ROUTE_HINT = (
    "Set CHROMA_URL to the root address of the Chroma server, without a path "
    "such as /api/v2. The server must be Chroma 0.5.16 or later."
)
NOT_FOUND_HINT = (
    "Check CHROMA_TENANT, CHROMA_DATABASE and --database. The message names "
    "what Chroma did not find."
)
QUOTA_HINT = (
    "The request exceeds a quota of Chroma Cloud. The message names the quota. "
    "Ask Chroma to raise it."
)
# Chroma Cloud answers a request over a quota with HTTP 422 and this text.
QUOTA_TEXT = "exceeded quota limit"
# Chroma 0.5 and 0.6 answer a missing collection with HTTP 400 and this error
# name. Chroma 1.x answers HTTP 404.
INVALID_COLLECTION = "InvalidCollection"
NO_BODY = "the answer has no body"
# Single-node Chroma 1.5.9 answers indexing_status with HTTP 500 and
# "Method scout_logs is not implemented".
NOT_IMPLEMENTED = "is not implemented"


class ChromaError(Exception):
    """A Chroma request failed. `status` is None for a network failure.

    `hint` is None when the caller's general hint applies.
    """

    def __init__(
        self,
        status: int | None,
        message: str,
        retry_after: float | None = None,
        hint: str | None = None,
    ) -> None:
        super().__init__(
            message if status is None else f"Chroma returned HTTP {status}: {message}"
        )
        self.status = status
        self.message = message
        self.retry_after = retry_after
        self.hint = hint


class DatabasesNotListed(Exception):
    """The server cannot list databases and its identity names none."""

    def __init__(self) -> None:
        super().__init__(
            f"{DATABASES_NOT_LISTED}, and its identity answer names no database."
        )


def _json(response: httpx.Response) -> Any:
    try:
        return response.json()
    except ValueError:
        return None


def _detail(detail: Any) -> str | None:
    """The text of the `detail` field of a Chroma 0.5 or 0.6 error body.

    The field is a string, or a list of validation errors with `loc` and
    `msg`. None when it holds no text.
    """
    if isinstance(detail, str):
        return detail or None
    if not isinstance(detail, list):
        return None
    parts: list[str] = []
    for item in detail:
        if isinstance(item, str):
            parts.append(item)
        elif isinstance(item, dict) and isinstance(item.get("msg"), str):
            loc = item.get("loc")
            where = ".".join(str(part) for part in loc) if isinstance(loc, list) else ""
            parts.append(f"{where}: {item['msg']}" if where else item["msg"])
    return "; ".join(parts) or None


def _message(response: httpx.Response, body: Any, secrets: list[str]) -> str:
    """The error text of `response`, redacted and then cut.

    A cut before the redaction would leave a part of a secret that `redact`
    no longer matches.
    """
    text = response.text
    if isinstance(body, dict):
        found = body.get("message") or body.get("error")
        if not (isinstance(found, str) and found):
            found = _detail(body.get("detail"))
        if found:
            text = found
    if not text.strip():
        text = NO_BODY
    return redact(text, secrets)[:MAX_MESSAGE_CHARS]


def _no_route(response: httpx.Response, body: Any) -> bool:
    """Whether the server has no route for the request.

    Chroma 1.x answers 404 or 405 with an empty body. Chroma 0.5 and 0.6
    answer 404 with `{"detail": "Not Found"}`. Their 405 has a JSON body and
    does not count: Chroma 0.5.16 to 0.6.2 answer it for the database list of
    a correct CHROMA_URL. A 404 or 405 whose body is not JSON comes from a
    proxy or another server, since Chroma answers an error of an existing
    route with JSON.
    """
    if response.status_code not in (404, 405):
        return False
    if body is None:
        return True
    return response.status_code == 404 and body == {"detail": "Not Found"}


def _error(
    response: httpx.Response, secrets: list[str], on_collection: bool
) -> ChromaError:
    """The ChromaError of an answer with HTTP 400 or higher."""
    status = response.status_code
    body = _json(response)
    retry_after = _retry_after(response)
    if _no_route(response, body):
        request = response.request
        route = redact(f"no route for {request.method} {request.url.path}", secrets)
        return ChromaError(
            status, route[:MAX_MESSAGE_CHARS], retry_after, NO_ROUTE_HINT
        )
    hint = None
    invalid = isinstance(body, dict) and body.get("error") == INVALID_COLLECTION
    if on_collection and (status == 404 or (status == 400 and invalid)):
        hint = COLLECTION_GONE_HINT
    elif status == 404:
        hint = NOT_FOUND_HINT
    elif status == 422 and QUOTA_TEXT in response.text:
        hint = QUOTA_HINT
    return ChromaError(status, _message(response, body, secrets), retry_after, hint)


def _redirect_message(response: httpx.Response, secrets: list[str]) -> str:
    """The scheme, host and port that `response` redirects to.

    The userinfo, path, query and fragment of the `Location` are left out.
    httpx raises RemoteProtocolError for a `Location` that is not a valid
    URL before the response gets here.
    """
    location = response.headers.get("Location")
    if not location:
        return "a redirect without a Location header"
    url = response.request.url.join(location)
    return redact(f"a redirect to {url.scheme}://{url.netloc.decode()}", secrets)


def _not_json_message(response: httpx.Response, secrets: list[str]) -> str:
    content_type = response.headers.get("Content-Type")
    if content_type is None:
        return "the answer is not JSON (no Content-Type)"
    shown = redact(content_type, secrets)[:MAX_CONTENT_TYPE_CHARS]
    return f"the answer is not JSON (Content-Type: {shown})"


def _retry_after(response: httpx.Response) -> float | None:
    try:
        return max(0.0, float(response.headers["Retry-After"]))
    except (KeyError, ValueError):
        return None


def _transient(exc: Exception) -> float | None:
    if isinstance(exc, httpx.TransportError):
        return 0.0
    if isinstance(exc, ChromaError) and exc.status in RETRY_STATUSES:
        return exc.retry_after or 0.0
    return None


def _not_implemented(exc: ChromaError) -> bool:
    return exc.status == 500 and NOT_IMPLEMENTED in exc.message


def _transient_unless_not_implemented(exc: Exception) -> float | None:
    """`_transient`, except that a 500 of a route the server lacks is final.

    A retry of that 500 would cost 15.75 s to 31.5 s per collection and
    command on single-node Chroma.
    """
    if isinstance(exc, ChromaError) and _not_implemented(exc):
        return None
    return _transient(exc)


def _seg(value: str) -> str:
    return quote(value, safe="")


class ChromaClient:
    def __init__(
        self,
        settings: ChromaSettings,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        headers: dict[str, str] = {}
        if settings.api_key is not None:
            value = settings.api_key
            if settings.token_header.lower() == "authorization":
                value = f"Bearer {value}"
            headers[settings.token_header] = value
        # On a redirect to another host, httpx drops only an Authorization
        # header, so a followed redirect would send another key header there.
        self._http = httpx.Client(
            base_url=settings.url.rstrip("/"),
            headers=headers,
            timeout=60,
            transport=transport,
            follow_redirects=False,
        )
        self._sleep = sleep
        self._secrets = [
            *([settings.api_key] if settings.api_key else []),
            *url_passwords(settings.url),
            *url_tokens(settings.url),
        ]

    def close(self) -> None:
        self._http.close()

    def _request(
        self,
        method: str,
        path: str,
        transient: Callable[[Exception], float | None] = _transient,
        on_collection: bool = False,
        **kwargs: Any,
    ) -> Any:
        def call() -> Any:
            response = self._http.request(method, path, **kwargs)
            if response.status_code >= 400:
                raise _error(response, self._secrets, on_collection)
            if response.status_code >= 300:
                raise ChromaError(
                    response.status_code,
                    _redirect_message(response, self._secrets),
                    hint=REDIRECT_HINT,
                )
            try:
                return response.json()
            except ValueError:
                raise ChromaError(
                    response.status_code,
                    _not_json_message(response, self._secrets),
                    hint=NOT_JSON_HINT,
                ) from None

        try:
            return with_retry(call, transient, sleep=self._sleep)
        except httpx.TransportError as exc:
            raise ChromaError(None, f"{type(exc).__name__}: {exc}") from exc

    def _collection_path(self, tenant: str, database: str, collection_id: str) -> str:
        return (
            f"/api/v2/tenants/{_seg(tenant)}/databases/{_seg(database)}"
            f"/collections/{_seg(collection_id)}"
        )

    def _hinted_request(
        self,
        status: int,
        hint: str,
        method: str,
        path: str,
        transient: Callable[[Exception], float | None] = _transient,
        **kwargs: Any,
    ) -> Any:
        """`_request`, whose ChromaError of HTTP `status` carries `hint`."""
        try:
            return self._request(method, path, transient, **kwargs)
        except ChromaError as exc:
            if exc.status != status:
                raise
            raise ChromaError(status, exc.message, hint=hint) from exc

    def _collection_request(
        self,
        method: str,
        path: str,
        transient: Callable[[Exception], float | None] = _transient,
        **kwargs: Any,
    ) -> Any:
        """`_request` on a collection path.

        A 404 of an existing route, or a 400 `InvalidCollection`, means the
        collection id is gone.
        """
        return self._request(method, path, transient, on_collection=True, **kwargs)

    def identity(self) -> dict[str, Any]:
        result: dict[str, Any] = self._request("GET", "/api/v2/auth/identity")
        return result

    def tenant(self) -> str:
        """The tenant of the identity, DEFAULT_TENANT when it is `ANY`."""
        tenant: str = self.identity()["tenant"]
        return DEFAULT_TENANT if tenant == ANY else tenant

    def identity_databases(self) -> list[str]:
        """The databases that the identity names, DEFAULT_DATABASE for `ANY`.

        Chroma 0.5.16 to 0.6.2 answer `["default_database"]` here for every
        tenant, also after another database was created.
        """
        listed = self.identity().get("databases")
        names = [
            DEFAULT_DATABASE if name == ANY else name
            for name in (listed if isinstance(listed, list) else [])
            if isinstance(name, str) and name
        ]
        return list(dict.fromkeys(names))

    def databases(self, tenant: str) -> list[str] | None:
        """The database names of `tenant`. None when the server cannot list them.

        Chroma 0.5.16 to 0.6.2 serve this path for POST only and answer 405.
        """
        path = f"/api/v2/tenants/{_seg(tenant)}/databases"
        try:
            rows = self._hinted_request(403, DATABASES_FORBIDDEN_HINT, "GET", path)
        except ChromaError as exc:
            if exc.status == 405:
                return None
            raise
        return [row["name"] for row in rows]

    def collections(self, tenant: str, database: str) -> list[dict[str, Any]]:
        """Every collection of `database`. Each page starts after the rows read.

        The listing ends only at an empty page: a server can return fewer rows
        than the limit before its last page. A page without a new id raises
        ChromaError, because a server that ignores `offset` would otherwise
        be read without end.
        """
        path = f"/api/v2/tenants/{_seg(tenant)}/databases/{_seg(database)}/collections"
        found: list[dict[str, Any]] = []
        ids: set[Any] = set()
        while True:
            offset = len(found)
            page = self._request(
                "GET", path, params={"limit": COLLECTIONS_PAGE, "offset": offset}
            )
            if not page:
                return found
            new = {row["id"] for row in page} - ids
            if not new:
                raise ChromaError(
                    200, f"the collection list did not advance at offset {offset}"
                )
            ids |= new
            found.extend(page)

    def count(self, tenant: str, database: str, collection_id: str) -> int:
        path = self._collection_path(tenant, database, collection_id) + "/count"
        return int(self._collection_request("GET", path))

    def indexing_status(
        self, tenant: str, database: str, collection_id: str
    ) -> dict[str, Any] | None:
        """The compaction counts of a collection; None when the server has none.

        Single-node Chroma 1.5.9 answers HTTP 500 "Method scout_logs is not
        implemented", which is not retried. Any other 500 is retried, so a
        transient 500 of a distributed frontend does not switch the guard off.
        """
        path = self._collection_path(tenant, database, collection_id)
        try:
            result: dict[str, Any] = self._request(
                "GET",
                path + "/indexing_status",
                transient=_transient_unless_not_implemented,
            )
        except ChromaError as exc:
            # The route exists since Chroma 1.4.1. Older servers answer 404
            # with an empty body for every collection, and a missing
            # collection also answers 404. The next request of the command
            # reports a missing collection with COLLECTION_GONE_HINT.
            if exc.status == 404 or _not_implemented(exc):
                return None
            raise
        return result

    def collection_by_name(
        self, tenant: str, database: str, name: str
    ) -> dict[str, Any]:
        """The collection document, with `id`, `version` and `log_position`.

        The frontend reads it from sysdb, not from its collection cache, so a
        compaction shows here before `get` serves the new order.
        """
        path = (
            f"/api/v2/tenants/{_seg(tenant)}/databases/{_seg(database)}"
            f"/collections/{_seg(name)}"
        )
        result: dict[str, Any] = self._collection_request("GET", path)
        return result

    def get(
        self,
        tenant: str,
        database: str,
        collection_id: str,
        *,
        limit: int,
        offset: int,
        include: list[str],
        where: dict[str, Any] | None = None,
        ids: list[str] | None = None,
    ) -> dict[str, Any]:
        if limit > CHROMA_MAX_PAGE:
            raise ValueError(
                f"limit {limit} exceeds the Chroma page maximum of {CHROMA_MAX_PAGE}"
            )
        body: dict[str, Any] = {"limit": limit, "offset": offset, "include": include}
        if where is not None:
            body["where"] = where
        if ids is not None:
            body["ids"] = ids
        path = self._collection_path(tenant, database, collection_id) + "/get"
        result: dict[str, Any] = self._collection_request("POST", path, json=body)
        return result

    def query(
        self,
        tenant: str,
        database: str,
        collection_id: str,
        *,
        embedding: list[float],
        n_results: int,
    ) -> list[str]:
        body = {"query_embeddings": [embedding], "n_results": n_results, "include": []}
        path = self._collection_path(tenant, database, collection_id) + "/query"
        result = self._collection_request("POST", path, json=body)
        ids: list[str] = result["ids"][0]
        return ids
