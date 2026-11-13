import json
from unittest.mock import patch

import httpx
import pytest

from to_weaviate.settings import ChromaSettings
from to_weaviate.sources.chroma.client import (
    CHROMA_MAX_PAGE,
    COLLECTION_GONE_HINT,
    DATABASES_FORBIDDEN_HINT,
    ChromaClient,
    ChromaError,
)


def settings(**kw):
    base = dict(
        url="http://chroma.test",
        api_key="ck-x",
        tenant=None,
        database=None,
        token_header="x-chroma-token",
    )
    base.update(kw)
    return ChromaSettings(**base)


def make(handler, **kw):
    sleeps: list[float] = []
    client = ChromaClient(
        settings(**kw), transport=httpx.MockTransport(handler), sleep=sleeps.append
    )
    return client, sleeps


@pytest.fixture(autouse=True)
def no_jitter():
    with patch("to_weaviate.retry.random.uniform", return_value=0.0):
        yield


def test_sends_token_header():
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json={"user_id": "u"})

    client, _ = make(handler)
    assert client.identity() == {"user_id": "u"}
    assert seen[0].headers["x-chroma-token"] == "ck-x"
    assert seen[0].url.path == "/api/v2/auth/identity"


def test_authorization_header_uses_bearer():
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json={})

    client, _ = make(handler, token_header="Authorization")
    client.identity()
    assert seen[0].headers["Authorization"] == "Bearer ck-x"


def test_no_auth_header_without_api_key():
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json={})

    client, _ = make(handler, api_key=None)
    client.identity()
    assert "x-chroma-token" not in seen[0].headers
    assert "authorization" not in seen[0].headers


def test_get_refuses_page_over_300():
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json={})

    client, _ = make(handler)
    assert CHROMA_MAX_PAGE == 300
    with pytest.raises(ValueError):
        client.get("t", "d", "c", limit=301, offset=0, include=[])
    assert seen == []


def test_get_posts_body_and_returns_json():
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json={"ids": ["a"]})

    client, _ = make(handler)
    out = client.get("t", "d", "c", limit=300, offset=600, include=["documents"])
    assert out == {"ids": ["a"]}
    assert seen[0].url.path == "/api/v2/tenants/t/databases/d/collections/c/get"
    assert json.loads(seen[0].content) == {
        "limit": 300,
        "offset": 600,
        "include": ["documents"],
    }


def test_get_posts_the_ids_when_given():
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json={"ids": ["a"]})

    client, _ = make(handler)
    client.get("t", "d", "c", limit=2, offset=0, include=[], ids=["a", "b"])
    assert json.loads(seen[0].content) == {
        "limit": 2,
        "offset": 0,
        "include": [],
        "ids": ["a", "b"],
    }

    client.get("t", "d", "c", limit=1, offset=0, include=[], where={"k": "v"})
    assert json.loads(seen[1].content)["where"] == {"k": "v"}


def test_query_returns_first_id_list():
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json={"ids": [["a", "b"]]})

    client, _ = make(handler)
    ids = client.query("t", "d", "c", embedding=[0.1, 0.2], n_results=2)
    assert ids == ["a", "b"]
    assert seen[0].url.path.endswith("/collections/c/query")
    assert json.loads(seen[0].content) == {
        "query_embeddings": [[0.1, 0.2]],
        "n_results": 2,
        "include": [],
    }


def test_count():
    client, _ = make(lambda r: httpx.Response(200, json=42))
    assert client.count("t", "d", "c") == 42


def test_databases_returns_names():
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json=[{"name": "a"}, {"name": "b"}])

    client, _ = make(handler)
    assert client.databases("t") == ["a", "b"]
    assert seen[0].url.path == "/api/v2/tenants/t/databases"


def test_retries_429_then_succeeds():
    responses = [httpx.Response(429), httpx.Response(200, json={"ok": True})]
    client, sleeps = make(lambda r: responses.pop(0))
    assert client.identity() == {"ok": True}
    assert len(sleeps) == 1


def test_honors_retry_after():
    responses = [
        httpx.Response(429, headers={"Retry-After": "2"}),
        httpx.Response(200, json={}),
    ]
    client, sleeps = make(lambda r: responses.pop(0))
    client.identity()
    assert sleeps == [2]


def test_retry_after_is_capped_at_30_seconds():
    responses = [
        httpx.Response(503, headers={"Retry-After": "600"}),
        httpx.Response(200, json={}),
    ]
    client, sleeps = make(lambda r: responses.pop(0))
    client.identity()
    assert sleeps == [30]


def test_retries_transport_errors():
    calls = []

    def handler(request):
        calls.append(1)
        if len(calls) == 1:
            raise httpx.ConnectError("down")
        return httpx.Response(200, json={})

    client, sleeps = make(handler)
    client.identity()
    assert len(calls) == 2 and len(sleeps) == 1


def test_401_fails_without_retry():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(401, json={"message": "bad token"})

    client, sleeps = make(handler)
    with pytest.raises(ChromaError) as e:
        client.identity()
    assert e.value.status == 401 and e.value.message == "bad token"
    assert len(calls) == 1 and sleeps == []


def test_gives_up_after_seven_attempts():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(503, text="unavailable")

    client, sleeps = make(handler)
    with pytest.raises(ChromaError) as e:
        client.identity()
    assert e.value.status == 503 and e.value.message == "unavailable"
    assert len(calls) == 7 and len(sleeps) == 6


def test_error_message_from_error_field_and_truncation():
    client, _ = make(lambda r: httpx.Response(400, json={"error": "boom"}))
    with pytest.raises(ChromaError) as e:
        client.identity()
    assert e.value.message == "boom"

    client, _ = make(lambda r: httpx.Response(400, text="x" * 900))
    with pytest.raises(ChromaError) as e:
        client.identity()
    assert len(e.value.message) == 500


CUT_KEY = "B5-MADE-UP-CHROMA-KEY-0123456789abcdefghij"
CUT_PASSWORD = "B5-MADE-UP-URL-PASSWORD-0123456789abcdefghij"


@pytest.mark.parametrize("secret", [CUT_KEY, CUT_PASSWORD])
@pytest.mark.parametrize("as_json", [False, True])
def test_error_message_is_redacted_before_the_cut(secret, as_json):
    text = "x" * 480 + secret + "y" * 100
    assert text.index(secret) < 490 and text.index(secret) + len(secret) > 510

    def handler(request):
        if as_json:
            return httpx.Response(400, json={"message": text})
        return httpx.Response(400, text=text)

    client, _ = make(
        handler,
        api_key=CUT_KEY,
        url=f"http://user:{CUT_PASSWORD}@chroma.test",
    )
    with pytest.raises(ChromaError) as e:
        client.identity()
    assert len(e.value.message) <= 500
    slices = [secret[i : i + 8] for i in range(len(secret) - 7)]
    assert not [s for s in slices if s in str(e.value)]
    assert "***" in e.value.message


def test_api_key_not_in_repr_or_error():
    client, _ = make(lambda r: httpx.Response(401, text="nope"))
    with pytest.raises(ChromaError) as e:
        client.identity()
    assert "ck-x" not in repr(e.value) and "ck-x" not in str(e.value)
    assert "ck-x" not in repr(client)


def test_collections_reads_all_pages():
    rows = [{"id": str(i)} for i in range(1003)]
    seen = []

    def handler(request):
        seen.append(request)
        offset = int(request.url.params["offset"])
        limit = int(request.url.params["limit"])
        return httpx.Response(200, json=rows[offset : offset + limit])

    client, _ = make(handler)
    assert client.collections("t", "d") == rows
    assert [r.url.params["limit"] for r in seen] == ["1000", "1000", "1000"]
    assert [r.url.params["offset"] for r in seen] == ["0", "1000", "1003"]
    assert seen[0].url.path == "/api/v2/tenants/t/databases/d/collections"


def collection_rows(total, cap):
    """A handler that serves `total` collections, at most `cap` per page."""
    rows = [{"id": str(i)} for i in range(total)]

    def handler(request):
        offset = int(request.url.params["offset"])
        limit = int(request.url.params["limit"])
        return httpx.Response(200, json=rows[offset : offset + min(limit, cap)])

    return handler


@pytest.mark.parametrize("total", [250, 200])
def test_collections_advance_by_the_rows_received(total):
    seen = []
    serve = collection_rows(total, cap=100)

    def handler(request):
        seen.append(request)
        return serve(request)

    client, _ = make(handler)
    found = client.collections("t", "d")
    assert [row["id"] for row in found] == [str(i) for i in range(total)]
    offsets = [str(offset) for offset in range(0, total, 100)] + [str(total)]
    assert [r.url.params["offset"] for r in seen] == offsets
    assert {r.url.params["limit"] for r in seen} == {"1000"}


def test_collections_read_past_a_page_shorter_than_the_one_before():
    rows = [{"id": str(i)} for i in range(250)]
    sizes = [100, 50, 100, 100]
    seen = []

    def handler(request):
        seen.append(request)
        offset = int(request.url.params["offset"])
        return httpx.Response(200, json=rows[offset : offset + sizes.pop(0)])

    client, _ = make(handler)
    assert client.collections("t", "d") == rows
    assert [r.url.params["offset"] for r in seen] == ["0", "100", "150", "250"]


def test_collections_page_over_the_limit_is_read_once():
    rows = [{"id": str(i)} for i in range(2500)]
    seen = []

    def handler(request):
        seen.append(request)
        offset = int(request.url.params["offset"])
        return httpx.Response(200, json=rows[offset : offset + 1200])

    client, _ = make(handler)
    assert client.collections("t", "d") == rows
    offsets = ["0", "1200", "2400", "2500"]
    assert [r.url.params["offset"] for r in seen] == offsets


def test_collections_that_do_not_advance_raise_chroma_error():
    rows = [{"id": str(i)} for i in range(3)]
    seen = []

    def handler(request):
        seen.append(request)
        if len(seen) > 5:
            raise AssertionError("the listing did not stop")
        return httpx.Response(200, json=rows)

    client, sleeps = make(handler)
    with pytest.raises(ChromaError) as e:
        client.collections("t", "d")
    assert e.value.message == "the collection list did not advance at offset 3"
    assert [r.url.params["offset"] for r in seen] == ["0", "3"]
    assert sleeps == []


REDIRECT_HINT = "Set CHROMA_URL to the address the server redirects to."
NOT_JSON_HINT = (
    "Check that CHROMA_URL points at the Chroma API and not at a proxy or a web page."
)


@pytest.mark.parametrize(
    ("status", "headers", "message"),
    [
        (
            301,
            {"Location": "https://user:MADE-UP@Chroma.Example:8443/v2?token=T#frag"},
            "a redirect to https://chroma.example:8443",
        ),
        (302, {"Location": "/login?next=%2F"}, "a redirect to http://chroma.test"),
        (300, {}, "a redirect without a Location header"),
    ],
)
def test_redirect_raises_chroma_error_without_following_it(status, headers, message):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(status, headers=headers)

    client, sleeps = make(handler)
    with pytest.raises(ChromaError) as e:
        client.identity()
    assert e.value.status == status
    assert str(e.value) == f"Chroma returned HTTP {status}: {message}"
    assert e.value.hint == REDIRECT_HINT
    assert len(calls) == 1 and sleeps == []


def test_answer_that_is_not_json_raises_chroma_error():
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, html="<html><body>Sign in</body></html>")

    client, sleeps = make(handler)
    with pytest.raises(ChromaError) as e:
        client.identity()
    assert e.value.status == 200
    assert str(e.value) == (
        "Chroma returned HTTP 200: the answer is not JSON "
        "(Content-Type: text/html; charset=utf-8)"
    )
    assert e.value.hint == NOT_JSON_HINT
    assert len(calls) == 1 and sleeps == []


def test_answer_that_is_not_json_names_a_missing_content_type():
    client, _ = make(lambda r: httpx.Response(204))
    with pytest.raises(ChromaError) as e:
        client.identity()
    assert e.value.message == "the answer is not JSON (no Content-Type)"


def test_content_type_of_an_answer_that_is_not_json_is_redacted_then_cut():
    # A cut before the redaction would keep the first 15 characters of the key.
    content_type = "text/html; x=" + "a" * 72 + CUT_KEY + "b" * 100
    client, _ = make(
        lambda r: httpx.Response(200, headers={"Content-Type": content_type}),
        api_key=CUT_KEY,
    )
    with pytest.raises(ChromaError) as e:
        client.identity()
    shown = "text/html; x=" + "a" * 72 + "***" + "b" * 12
    assert e.value.message == f"the answer is not JSON (Content-Type: {shown})"
    slices = [CUT_KEY[i : i + 8] for i in range(len(CUT_KEY) - 7)]
    assert not [s for s in slices if s in str(e.value)]


def test_close_closes_the_http_client():
    client, _ = make(lambda request: httpx.Response(200, json=1))
    client.close()
    with pytest.raises(RuntimeError, match="closed"):
        client.count("t", "d", "c")


def test_transport_errors_after_the_retries_raise_chroma_error():
    calls = []

    def handler(request):
        calls.append(1)
        raise httpx.ConnectError("down")

    client, sleeps = make(handler)
    with pytest.raises(ChromaError) as e:
        client.identity()
    assert e.value.status is None and isinstance(e.value.__cause__, httpx.ConnectError)
    assert str(e.value) == "ConnectError: down"
    assert len(calls) == 7 and len(sleeps) == 6


def _answer(status: int):
    def handler(request):
        return httpx.Response(status, json={"message": "no"})

    return handler


def test_a_403_on_the_database_list_names_the_database_settings():
    client, _ = make(_answer(403))
    with pytest.raises(ChromaError) as info:
        client.databases("t")
    assert (info.value.status, info.value.hint) == (403, DATABASES_FORBIDDEN_HINT)


@pytest.mark.parametrize("call", ["count", "get", "query"])
def test_a_404_on_a_collection_says_to_inspect_again(call):
    client, _ = make(_answer(404))
    calls = {
        "count": lambda: client.count("t", "d", "c"),
        "get": lambda: client.get("t", "d", "c", limit=1, offset=0, include=[]),
        "query": lambda: client.query("t", "d", "c", embedding=[1.0], n_results=1),
    }
    with pytest.raises(ChromaError) as info:
        calls[call]()
    assert (str(info.value), info.value.hint) == (
        "Chroma returned HTTP 404: no",
        COLLECTION_GONE_HINT,
    )


def test_other_answers_keep_the_general_hint():
    client, _ = make(_answer(404))
    with pytest.raises(ChromaError) as info:
        client.databases("t")
    assert info.value.hint is None
    client, _ = make(_answer(403))
    with pytest.raises(ChromaError) as info:
        client.count("t", "d", "c")
    assert info.value.hint is None


NOT_IMPLEMENTED_ANSWER = {
    "error": "InternalError",
    "message": "Method scout_logs is not implemented",
}


def test_indexing_status_returns_the_counts():
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json={"num_unindexed_ops": 5, "total_ops": 9})

    client, _ = make(handler)
    assert client.indexing_status("t", "d", "c") == {
        "num_unindexed_ops": 5,
        "total_ops": 9,
    }
    assert (seen[0].method, seen[0].url.path) == (
        "GET",
        "/api/v2/tenants/t/databases/d/collections/c/indexing_status",
    )


def test_indexing_status_of_a_single_node_server_is_none_and_not_retried():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(500, json=NOT_IMPLEMENTED_ANSWER)

    client, sleeps = make(handler)
    assert client.indexing_status("t", "d", "c") is None
    assert len(calls) == 1 and sleeps == []


@pytest.mark.parametrize(
    "answer",
    [
        httpx.Response(404),
        httpx.Response(
            404,
            json={
                "error": "NotFoundError",
                "message": "Collection [c] does not exist.",
            },
        ),
    ],
    ids=["no route", "missing collection"],
)
def test_a_404_on_indexing_status_is_none_and_not_retried(answer):
    calls = []

    def handler(request):
        calls.append(1)
        return answer

    client, sleeps = make(handler)
    assert client.indexing_status("t", "d", "c") is None
    assert len(calls) == 1 and sleeps == []


def test_another_500_on_indexing_status_is_retried():
    responses = [
        httpx.Response(500, json={"error": "InternalError", "message": "boom"}),
        httpx.Response(200, json={"num_unindexed_ops": 0, "total_ops": 9}),
    ]
    client, sleeps = make(lambda r: responses.pop(0))
    assert client.indexing_status("t", "d", "c") == {
        "num_unindexed_ops": 0,
        "total_ops": 9,
    }
    assert sleeps == [0.25]


def test_collection_by_name_reads_the_collection_document():
    seen = []

    def handler(request):
        seen.append(request)
        document = {"id": "c-1", "name": "my docs", "version": 3, "log_position": 650}
        return httpx.Response(200, json=document)

    client, _ = make(handler)
    assert client.collection_by_name("t", "d", "my docs")["log_position"] == 650
    assert seen[0].method == "GET"
    assert (
        seen[0].url.raw_path.decode()
        == "/api/v2/tenants/t/databases/d/collections/my%20docs"
    )


# Chroma 0.5.16 to 0.6.2 answer the first body, a Rust server the empty one.
@pytest.mark.parametrize(
    "answer",
    [
        httpx.Response(
            405, json={"detail": "Method Not Allowed"}, headers={"Allow": "POST"}
        ),
        httpx.Response(405, headers={"Allow": "POST"}),
    ],
)
def test_databases_is_none_when_the_server_answers_405(answer):
    client, sleeps = make(lambda r: answer)
    assert client.databases("t") is None
    assert sleeps == []


IDENTITY_NO_AUTH = {
    "user_id": "",
    "tenant": "default_tenant",
    "databases": ["default_database"],
    "attributes": None,
}
IDENTITY_TOKEN_AUTH = {
    "user_id": "anonymous",
    "tenant": "*",
    "databases": ["*"],
    "attributes": None,
}


@pytest.mark.parametrize(
    ("identity", "tenant"),
    [
        (IDENTITY_NO_AUTH, "default_tenant"),
        (IDENTITY_TOKEN_AUTH, "default_tenant"),
        ({"tenant": "of-the-key", "databases": ["a"]}, "of-the-key"),
    ],
)
def test_tenant_is_the_identity_tenant_and_the_default_for_a_star(identity, tenant):
    client, _ = make(lambda r: httpx.Response(200, json=identity))
    assert client.tenant() == tenant


@pytest.mark.parametrize(
    ("identity", "databases"),
    [
        (IDENTITY_NO_AUTH, ["default_database"]),
        (IDENTITY_TOKEN_AUTH, ["default_database"]),
        (
            {"tenant": "t", "databases": ["a", "*", "b", "a"]},
            ["a", "default_database", "b"],
        ),
        ({"tenant": "t", "databases": ["*", "default_database"]}, ["default_database"]),
        ({"tenant": "t", "databases": []}, []),
        ({"tenant": "t", "databases": None}, []),
        ({"tenant": "t"}, []),
        ({"tenant": "t", "databases": ["", 3, None]}, []),
    ],
)
def test_identity_databases_names_the_default_for_a_star(identity, databases):
    seen = []

    def handler(request):
        seen.append(request.url.path)
        return httpx.Response(200, json=identity)

    client, _ = make(handler)
    assert client.identity_databases() == databases
    assert seen == ["/api/v2/auth/identity"]
