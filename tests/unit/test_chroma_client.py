import json
from unittest.mock import patch

import httpx
import pytest

from to_weaviate.settings import ChromaSettings
from to_weaviate.sources.chroma.client import CHROMA_MAX_PAGE, ChromaClient, ChromaError


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


def test_gives_up_after_six_attempts():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(503, text="unavailable")

    client, sleeps = make(handler)
    with pytest.raises(ChromaError) as e:
        client.identity()
    assert e.value.status == 503 and e.value.message == "unavailable"
    assert len(calls) == 6 and len(sleeps) == 5


def test_error_message_from_error_field_and_truncation():
    client, _ = make(lambda r: httpx.Response(400, json={"error": "boom"}))
    with pytest.raises(ChromaError) as e:
        client.identity()
    assert e.value.message == "boom"

    client, _ = make(lambda r: httpx.Response(400, text="x" * 900))
    with pytest.raises(ChromaError) as e:
        client.identity()
    assert len(e.value.message) == 500


def test_api_key_not_in_repr_or_error():
    client, _ = make(lambda r: httpx.Response(401, text="nope"))
    with pytest.raises(ChromaError) as e:
        client.identity()
    assert "ck-x" not in repr(e.value) and "ck-x" not in str(e.value)
    assert "ck-x" not in repr(client)


def test_collections_reads_all_pages():
    seen = []

    def handler(request):
        seen.append(request)
        offset = int(request.url.params["offset"])
        n = 1000 if offset == 0 else 3
        return httpx.Response(200, json=[{"id": str(offset + i)} for i in range(n)])

    client, _ = make(handler)
    assert len(client.collections("t", "d")) == 1003
    assert [r.url.params["limit"] for r in seen] == ["1000", "1000"]
    assert [r.url.params["offset"] for r in seen] == ["0", "1000"]
    assert seen[0].url.path == "/api/v2/tenants/t/databases/d/collections"


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
    assert len(calls) == 6 and len(sleeps) == 5
