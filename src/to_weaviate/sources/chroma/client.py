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
REDIRECT_HINT = "Set CHROMA_URL to the address the server redirects to."
NOT_JSON_HINT = (
    "Check that CHROMA_URL points at the Chroma API and not at a proxy or a web page."
)


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


def _message(response: httpx.Response, secrets: list[str]) -> str:
    """The error text of `response`, redacted and then cut.

    A cut before the redaction would leave a part of a secret that `redact`
    no longer matches.
    """
    try:
        body = response.json()
    except ValueError:
        body = None
    text = response.text
    if isinstance(body, dict):
        found = body.get("message") or body.get("error")
        if isinstance(found, str) and found:
            text = found
    return redact(text, secrets)[:MAX_MESSAGE_CHARS]


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

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        def call() -> Any:
            response = self._http.request(method, path, **kwargs)
            if response.status_code >= 400:
                raise ChromaError(
                    response.status_code,
                    _message(response, self._secrets),
                    _retry_after(response),
                )
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
            return with_retry(call, _transient, sleep=self._sleep)
        except httpx.TransportError as exc:
            raise ChromaError(None, f"{type(exc).__name__}: {exc}") from exc

    def _collection_path(self, tenant: str, database: str, collection_id: str) -> str:
        return (
            f"/api/v2/tenants/{_seg(tenant)}/databases/{_seg(database)}"
            f"/collections/{_seg(collection_id)}"
        )

    def identity(self) -> dict[str, Any]:
        result: dict[str, Any] = self._request("GET", "/api/v2/auth/identity")
        return result

    def databases(self, tenant: str) -> list[str]:
        rows = self._request("GET", f"/api/v2/tenants/{_seg(tenant)}/databases")
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
        return int(self._request("GET", path))

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
    ) -> dict[str, Any]:
        if limit > CHROMA_MAX_PAGE:
            raise ValueError(
                f"limit {limit} exceeds the Chroma page maximum of {CHROMA_MAX_PAGE}"
            )
        body: dict[str, Any] = {"limit": limit, "offset": offset, "include": include}
        if where is not None:
            body["where"] = where
        path = self._collection_path(tenant, database, collection_id) + "/get"
        result: dict[str, Any] = self._request("POST", path, json=body)
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
        result = self._request("POST", path, json=body)
        ids: list[str] = result["ids"][0]
        return ids
