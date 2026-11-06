import pytest

from to_weaviate.settings import (
    InvalidSetting,
    MissingCredential,
    chroma_settings,
    load_env,
    normalized_host,
    stored_url,
    url_passwords,
    url_tokens,
    weaviate_settings,
    without_userinfo,
)


def test_chroma_defaults():
    s = chroma_settings({"CHROMA_API_KEY": "ck-x"})
    assert s.url == "https://api.trychroma.com" and s.token_header == "x-chroma-token"


def test_cloud_url_requires_key():
    with pytest.raises(MissingCredential) as e:
        chroma_settings({})
    assert e.value.var == "CHROMA_API_KEY"


def test_custom_url_needs_no_key():
    assert chroma_settings({"CHROMA_URL": "http://localhost:8001"}).api_key is None


def test_missing_weaviate_url_names_variable():
    with pytest.raises(MissingCredential) as e:
        weaviate_settings({})
    assert e.value.var == "WEAVIATE_URL"


def test_provider_key_becomes_header():
    s = weaviate_settings(
        {"WEAVIATE_URL": "https://c.weaviate.cloud", "OPENAI_API_KEY": "sk-SENTINEL"}
    )
    assert s.headers["X-OpenAI-Api-Key"] == "sk-SENTINEL"


def test_repr_hides_secrets():
    s = weaviate_settings(
        {
            "WEAVIATE_URL": "u",
            "WEAVIATE_API_KEY": "SENTINEL",
            "OPENAI_API_KEY": "SENTINEL2",
        }
    )
    assert "SENTINEL" not in repr(s)


KEY_VARIABLES = ["CHROMA_API_KEY", "WEAVIATE_API_KEY", "OPENAI_API_KEY", "XAI_APIKEY"]


@pytest.mark.parametrize("var", KEY_VARIABLES)
@pytest.mark.parametrize(
    "value",
    [
        "made\nup",
        "made\tup",
        "made\x00up",
        "made\x7fup",
        "made up",
        " made-up",
        "made-up\n",
        "made-up\r",
    ],
)
def test_key_with_a_control_character_is_refused(var, value):
    env = {"WEAVIATE_URL": "http://localhost:8099", var: value}
    load = chroma_settings if var == "CHROMA_API_KEY" else weaviate_settings
    with pytest.raises(InvalidSetting) as e:
        load(env)
    assert str(e.value) == f"{var} holds a control character."
    assert e.value.hint == (
        f"Set {var} without control characters or leading and trailing whitespace."
    )


@pytest.mark.parametrize("var", KEY_VARIABLES)
@pytest.mark.parametrize(
    ("value", "expected"),
    [("made up", "made up"), ("madé-up", "madé-up"), ("", None), (" \n ", None)],
)
def test_key_without_a_control_character_is_kept(var, value, expected):
    env = {"WEAVIATE_URL": "http://localhost:8099", var: value}
    if var == "CHROMA_API_KEY":
        env["CHROMA_URL"] = "http://localhost:8001"
        assert chroma_settings(env).api_key == expected
    elif var == "WEAVIATE_API_KEY":
        assert weaviate_settings(env).api_key == expected
    else:
        headers = weaviate_settings(env).headers
        assert list(headers.values()) == ([expected] if expected else [])


def test_environ_wins_over_dotenv(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("WEAVIATE_URL=from-file\n")
    monkeypatch.setenv("WEAVIATE_URL", "from-env")
    assert load_env(tmp_path)["WEAVIATE_URL"] == "from-env"


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://user:pw@host:8000/path?q=1", "https://host:8000/path?q=1"),
        ("https://user@host", "https://host"),
        ("user:pw@host:8099", "host:8099"),
        ("http://host:8099", "http://host:8099"),
        ("host:8099", "host:8099"),
        ("http://[::1", "http://[::1"),
        ("user:pw@[::1:50051", "[::1:50051"),
        ("http://user:p%40ss@[::1", "http://[::1"),
        ("http://user:pw@[::1/path@x", "http://[::1/path@x"),
    ],
)
def test_without_userinfo(url, expected):
    assert without_userinfo(url) == expected


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://user:pw@host:8000/path?q=1#f", "https://host:8000/path"),
        ("https://host/v1?api-key=k", "https://host/v1"),
        ("https://host#f?q", "https://host"),
        ("https://host/v1", "https://host/v1"),
        ("user:pw@host:8099/p?q", "host:8099/p"),
        ("http://user:pw@[::1?q", "http://[::1"),
    ],
)
def test_stored_url(url, expected):
    assert stored_url(url) == expected


@pytest.mark.parametrize(
    "url",
    [
        "https://bob:pa/ss@proxy.example.com/v1",
        "https://bob:pa?ss@proxy.example.com/v1",
        "https://bob:pa#ss@proxy.example.com/v1",
        "http://bob:pa#ss@ollama:11434",
        "https://bob:pa?ss@proxy.example.com",
        "bob:pa/ss@proxy.example.com",
        "https://proxy.example.com/users/a@b",
        "https://proxy.example.com/v1?mail=a@b",
        "https://proxy.example.com/v1#a@b",
        "http://user:pw@[::1/path@x",
    ],
)
def test_stored_url_stores_no_value_that_holds_an_at_sign_after_cleaning(url):
    assert stored_url(url) is None


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://user:pw@host:8000", ["pw"]),
        ("https://user@host", []),
        ("http://host:8099", []),
        ("http://[::1", []),
        ("user:pw@[::1:50051", ["pw"]),
        ("http://user:p%40ss@[::1", ["p%40ss", "p@ss"]),
        ("http://user@[::1", []),
    ],
)
def test_url_passwords(url, expected):
    assert url_passwords(url) == expected


@pytest.mark.parametrize(
    ("url", "default_scheme", "expected"),
    [
        ("http://localhost:8099", "http", "localhost:8099"),
        ("HTTP://LocalHost:8099", "http", "localhost:8099"),
        ("http://admin:pw@localhost:8099", "http", "localhost:8099"),
        ("http://localhost:8099/v1/schema", "http", "localhost:8099"),
        ("http://localhost:8099/?a=1#top", "http", "localhost:8099"),
        ("http://localhost.:8099", "http", "localhost:8099"),
        ("http://weaviate.test", "http", "weaviate.test:80"),
        ("http://weaviate.test:80", "http", "weaviate.test:80"),
        ("https://c.weaviate.cloud", "http", "c.weaviate.cloud:443"),
        ("https://C.Weaviate.Cloud.:443/", "http", "c.weaviate.cloud:443"),
        ("localhost:8099", "http", "localhost:8099"),
        ("user:pw@weaviate.test", "http", "weaviate.test:80"),
        ("c.weaviate.cloud", "https", "c.weaviate.cloud:443"),
        ("http://[::1]:8099", "http", "[::1]:8099"),
    ],
)
def test_normalized_host(url, default_scheme, expected):
    assert normalized_host(url, default_scheme) == expected


def test_normalized_host_defaults_to_http():
    assert normalized_host("weaviate.test") == "weaviate.test:80"


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://TOKEN@host", ["TOKEN"]),
        ("https://to%40ken@host:8000/p", ["to%40ken", "to@ken"]),
        ("TOKEN@host:8099", ["TOKEN"]),
        ("http://TOKEN@[::1", ["TOKEN"]),
        ("https://user:pw@host", []),
        ("https://user:@host", []),
        ("http://host:8099", []),
        ("", []),
    ],
)
def test_url_tokens(url, expected):
    assert url_tokens(url) == expected


def load(var, url):
    """The settings that read `var`, loaded with `var` set to `url`."""
    env = {"WEAVIATE_URL": "http://localhost:8099", var: url}
    return chroma_settings(env) if var == "CHROMA_URL" else weaviate_settings(env)


@pytest.mark.parametrize(
    ("var", "url"),
    [
        *[
            (var, f"http://admin:hunter2{char}x@localhost:{port}")
            for var, port in (
                ("CHROMA_URL", 8001),
                ("WEAVIATE_URL", 8099),
                ("WEAVIATE_GRPC_URL", 50099),
            )
            for char in "#?/"
        ],
        ("CHROMA_URL", "http://admin:1234#x@localhost:8001"),
        ("CHROMA_URL", "http://admin:1234?x@localhost:8001"),
        ("CHROMA_URL", "http://admin:1234/x@localhost:8001"),
        ("CHROMA_URL", "https://chroma.example.com/#top"),
        ("CHROMA_URL", "https://chroma.example.com#"),
        ("CHROMA_URL", "https://chroma.example.com/a@b"),
        ("CHROMA_URL", "https://chroma.example.com/?a=b@c"),
        ("CHROMA_URL", "localhost:8001"),
        ("CHROMA_URL", "ftp://localhost:8001"),
        ("CHROMA_URL", "http://[::1"),
        ("CHROMA_URL", "http://localhost:80a"),
        ("CHROMA_URL", "http://localhost:99999"),
        ("WEAVIATE_URL", "http://localhost:8099/v1"),
        ("WEAVIATE_URL", "localhost:8099/v1/schema"),
        ("WEAVIATE_URL", "http://localhost:8099#top"),
        ("WEAVIATE_URL", "http://localhost:port"),
        ("WEAVIATE_URL", "http://[::1"),
        ("WEAVIATE_GRPC_URL", "grpcs://localhost:50099"),
        ("WEAVIATE_GRPC_URL", "grpc://localhost:50099"),
        ("WEAVIATE_GRPC_URL", "http://localhost:50099/grpc"),
        ("WEAVIATE_GRPC_URL", "http://[::1:50099"),
    ],
)
def test_refused_url(var, url):
    with pytest.raises(InvalidSetting) as e:
        load(var, url)
    assert str(e.value) == f"{var} is not a valid URL."
    assert e.value.hint == "Percent-encode #, ?, / and @ in a password."
    assert e.value.__cause__ is None
    assert e.value.__context__ is None or e.value.__suppress_context__


@pytest.mark.parametrize(
    ("var", "url"),
    [
        ("CHROMA_URL", "https://user:p%40ss@chroma.example.com/prefix?a=1"),
        ("CHROMA_URL", "HTTP://LocalHost:8001/"),
        ("CHROMA_URL", "https://TOKEN@chroma.example.com"),
        ("WEAVIATE_URL", "localhost:8099"),
        ("WEAVIATE_URL", "admin:pw@localhost:8099/"),
        ("WEAVIATE_URL", "https://TOKEN@c.weaviate.cloud"),
        ("WEAVIATE_URL", "http://[::1]:8099"),
        ("WEAVIATE_GRPC_URL", "localhost:50099"),
        ("WEAVIATE_GRPC_URL", "https://grpc-user:pw@grpc.example.com/"),
    ],
)
def test_accepted_url(var, url):
    assert load(var, url) is not None


@pytest.mark.parametrize(
    "url",
    [
        "https://api.trychroma.com.",
        "https://API.TryChroma.com",
        "https://EU.API.TRYCHROMA.COM.:443/",
    ],
)
def test_chroma_cloud_host_ignores_case_and_a_trailing_dot(url):
    with pytest.raises(MissingCredential) as e:
        chroma_settings({"CHROMA_URL": url})
    assert e.value.var == "CHROMA_API_KEY"


def test_settings_hold_the_parsed_endpoints():
    s = weaviate_settings(
        {
            "WEAVIATE_URL": "admin:pw@LocalHost:8099",
            "WEAVIATE_GRPC_URL": "https://Grpc.Example.com",
        }
    )
    assert (s.http.host, s.http.port, s.http.secure) == ("localhost", 8099, False)
    assert s.grpc is not None
    assert (s.grpc.host, s.grpc.port, s.grpc.secure) == ("grpc.example.com", 443, True)
    assert weaviate_settings({"WEAVIATE_URL": "https://c.weaviate.cloud"}).grpc is None


@pytest.mark.parametrize(
    ("url", "grpc", "expected"),
    [
        ("http://c.weaviate.cloud", None, ("c.weaviate.cloud", 443, True)),
        ("https://c.weaviate.cloud", None, ("c.weaviate.cloud", 443, True)),
        ("c.weaviate.cloud", None, ("c.weaviate.cloud", 443, True)),
        ("http://localhost:8099", None, ("localhost", 8099, True)),
        ("http://weaviate.test", "weaviate.test:50051", ("weaviate.test", 80, False)),
        ("weaviate.test", "weaviate.test:50051", ("weaviate.test", 80, False)),
        ("https://weaviate.test", "weaviate.test:50051", ("weaviate.test", 443, True)),
    ],
)
def test_http_endpoint_without_grpc_url_is_port_443_over_tls(url, grpc, expected):
    env = {"WEAVIATE_URL": url, "WEAVIATE_GRPC_URL": grpc or ""}
    http = weaviate_settings(env).http
    assert (http.host, http.port, http.secure) == expected
