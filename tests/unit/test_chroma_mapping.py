import pytest

from to_weaviate.sources.chroma.mapping import vectorizer_for


def known(name, **config):
    return {"type": "known", "name": name, "config": config}


def test_openai_mapping():
    v, _ = vectorizer_for(
        known("openai", model_name="text-embedding-3-small", dimensions=512)
    )
    assert v is not None
    assert v.module == "text2vec-openai"
    assert v.options == {"model": "text-embedding-3-small", "dimensions": 512}


def test_openai_never_copies_the_key_variable():
    v, _ = vectorizer_for(
        known(
            "openai",
            api_key_env_var="OPENAI_API_KEY",
            model_name="m",
            api_base="https://x",
        )
    )
    assert v is not None
    assert v.options == {"model": "m", "base_url": "https://x"}


def test_openai_azure_has_no_mapping():
    v, notes = vectorizer_for(known("openai", api_type="azure", model_name="m"))
    assert v is None and notes


def test_none_embedding():
    assert vectorizer_for(None) == (None, [])


def test_null_options_are_left_out():
    v, _ = vectorizer_for(known("cohere", model_name=None))
    assert v is not None
    assert v.module == "text2vec-cohere" and v.options == {}


@pytest.mark.parametrize(
    ("name", "config", "module", "options"),
    [
        (
            "cohere",
            {"model_name": "embed-v4"},
            "text2vec-cohere",
            {"model": "embed-v4"},
        ),
        (
            "jina",
            {"model_name": "j", "dimensions": 8, "extra": 1},
            "text2vec-jinaai",
            {"model": "j", "dimensions": 8},
        ),
        ("voyageai", {"model_name": "v"}, "text2vec-voyageai", {}),
        ("mistral", {"model": "mm"}, "text2vec-mistral", {"model": "mm"}),
        (
            "huggingface",
            {"model_name": "h"},
            "text2vec-huggingface",
            {"model": "h"},
        ),
        (
            "ollama",
            {"url": "http://o", "model_name": "n"},
            "text2vec-ollama",
            {"api_endpoint": "http://o", "model": "n"},
        ),
    ],
)
def test_simple_mappings(name, config, module, options):
    v, _ = vectorizer_for(known(name, **config))
    assert v is not None
    assert (v.module, v.options) == (module, options)


@pytest.mark.parametrize("name", ["google_generative_ai", "google_gemini"])
def test_google_ai_studio(name):
    v, _ = vectorizer_for(known(name, model_name="x"))
    assert v is not None
    assert v.module == "text2vec-google"
    assert v.options.get("project_id") is None


def test_google_vertex():
    v, _ = vectorizer_for(known("google_vertex", project_id="p"))
    assert v is not None
    assert v.module == "text2vec-google" and v.options["project_id"] == "p"


def test_default_model_has_no_mapping():
    v, notes = vectorizer_for(known("default"))
    assert v is None and any("all-MiniLM-L6-v2" in n for n in notes)


def test_chroma_hosted_model_note():
    v, notes = vectorizer_for(known("chroma-cloud-qwen"))
    assert v is None and any("embed.trychroma.com" in n for n in notes)


def test_unmapped_name_is_named_in_a_note():
    v, notes = vectorizer_for(known("onnx_thing"))
    assert v is None and any("onnx_thing" in n for n in notes)
