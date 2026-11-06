import pytest

from to_weaviate.output import redact, redact_strings, secret_values


def test_redact_replaces_the_value():
    assert redact("key SENTINEL-A was refused", ["SENTINEL-A"]) == "key *** was refused"


@pytest.mark.parametrize(
    "message",
    [
        "Illegal header value {encoded!r}",
        "value {value!r} is not allowed",
    ],
)
def test_redact_replaces_the_escaped_forms(message):
    value = "top\nsecret-é"
    text = message.format(encoded=value.encode(), value=value)
    assert "secret" in text
    assert "secret" not in redact(text, [value])


def test_redact_ignores_empty_values():
    assert redact("nothing here", ["", "x" * 0]) == "nothing here"


def test_redact_strings_walks_nested_values():
    document = {"message": "a S1", "errors": [{"message": "b S1", "n": 2}], "ok": True}
    assert redact_strings(document, ["S1"]) == {
        "message": "a ***",
        "errors": [{"message": "b ***", "n": 2}],
        "ok": True,
    }


def test_secret_values_reads_the_key_variables():
    env = {
        "CHROMA_API_KEY": "c",
        "WEAVIATE_API_KEY": " w ",
        "OPENAI_API_KEY": "o",
        "COHERE_APIKEY": "",
        "WEAVIATE_URL": "http://localhost:8099",
    }
    assert sorted(secret_values(env)) == ["c", "o", "w"]


def test_redact_strings_keeps_status_and_code():
    document = {
        "status": "error",
        "code": "usage",
        "message": "an error",
        "errors": [{"code": "usage", "message": "usage error"}],
    }
    assert redact_strings(document, ["error", "usage"]) == {
        "status": "error",
        "code": "usage",
        "message": "an ***",
        "errors": [{"code": "usage", "message": "*** ***"}],
    }


def test_secret_values_include_url_passwords():
    env = {
        "CHROMA_URL": "https://user:p%40ss@chroma.example.com",
        "WEAVIATE_URL": "http://admin:wpass@localhost:8099",
        "WEAVIATE_GRPC_URL": "grpc-user:gpass@localhost:50099",
    }
    values = secret_values(env)
    assert {"p%40ss", "p@ss", "wpass", "gpass"} <= set(values)
    assert not {"user", "admin", "grpc-user"} & set(values)
