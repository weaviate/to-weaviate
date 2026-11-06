import pytest

from to_weaviate.settings import (
    MissingCredential,
    chroma_settings,
    load_env,
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
    ],
)
def test_without_userinfo(url, expected):
    assert without_userinfo(url) == expected
