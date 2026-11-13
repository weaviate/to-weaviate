import pytest

from to_weaviate.plan import CollectionPlan, SourceRef, TargetPlan, VectorPlan
from to_weaviate.sources.chroma.mapping import (
    CONFIG_KEYS,
    stored_embedding,
    vectorizer_for,
)
from to_weaviate.target.schema import collection_kwargs


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


@pytest.mark.parametrize(
    ("api_base", "base_url"),
    [
        ("https://proxy.example.com/v1", "https://proxy.example.com"),
        ("https://proxy.example.com/v1/", "https://proxy.example.com"),
        ("https://proxy.example.com/", "https://proxy.example.com"),
        ("https://proxy.example.com/openai/v1", "https://proxy.example.com/openai"),
        ("https://proxy.example.com/v1/v1", "https://proxy.example.com/v1"),
    ],
)
def test_openai_api_base_loses_one_trailing_v1(api_base, base_url):
    v, notes = vectorizer_for(known("openai", model_name="m", api_base=api_base))
    assert v is not None
    assert v.options == {"model": "m", "base_url": base_url}
    assert [n for n in notes if "Weaviate server must reach" in n and base_url in n]


@pytest.mark.parametrize(
    "api_base",
    [
        "https://api.openai.com/v1",
        "https://api.openai.com/v1/",
        "https://api.openai.com",
    ],
)
def test_openai_default_base_gives_no_base_url(api_base):
    v, notes = vectorizer_for(known("openai", model_name="m", api_base=api_base))
    assert v is not None
    assert v.options == {"model": "m"}
    assert notes == []


# chromadb's OpenAIEmbeddingFunction uses text-embedding-ada-002 when no
# model is given.
ADA_002 = "text-embedding-ada-002"


def _sent_vectorizer(v):
    entry = CollectionPlan(
        source=SourceRef(
            ref="r",
            namespace="n",
            name="docs",
            count=1,
            dimension=2,
            distance="cosine",
            embedding=None,
        ),
        target=TargetPlan(
            name="Docs", vectors=[VectorPlan(vectorizer=v)], properties=[]
        ),
    )
    return collection_kwargs([entry])["vector_config"][0].vectorizer


@pytest.mark.parametrize(
    "config",
    [
        {"model_name": ADA_002},
        {"model_name": ADA_002, "api_base": None, "dimensions": None},
        {"model_name": ADA_002, "api_base": "https://api.openai.com/v1"},
    ],
)
def test_openai_ada_002_gets_the_weaviate_model_name(config):
    v, notes = vectorizer_for(known("openai", **config))
    assert v is not None
    assert v.module == "text2vec-openai"
    assert v.options == {"model": "ada"}
    assert notes == []
    sent = _sent_vectorizer(v)
    assert (sent.model, sent.modelVersion, sent.baseURL) == ("ada", None, None)


def test_openai_ada_002_keeps_its_name_with_a_base_url():
    v, _ = vectorizer_for(
        known("openai", model_name=ADA_002, api_base="https://proxy.example.com/v1")
    )
    assert v is not None
    assert v.options == {"model": ADA_002, "base_url": "https://proxy.example.com"}
    assert _sent_vectorizer(v).model == ADA_002


def test_openai_ada_002_gets_the_weaviate_model_name_on_a_regional_openai_url():
    # Weaviate validates the model name for every base URL that contains
    # api.openai.com, not only the default one.
    v, _ = vectorizer_for(
        known("openai", model_name=ADA_002, api_base="https://eu.api.openai.com/v1")
    )
    assert v is not None
    assert v.options == {"model": "ada", "base_url": "https://eu.api.openai.com"}


@pytest.mark.parametrize("model", ["text-embedding-3-small", "text-embedding-3-large"])
@pytest.mark.parametrize("api_base", [None, "https://proxy.example.com/v1"])
def test_openai_v3_models_keep_their_name(model, api_base):
    v, _ = vectorizer_for(known("openai", model_name=model, api_base=api_base))
    assert v is not None
    assert v.options["model"] == model
    assert _sent_vectorizer(v).model == model


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
        ("voyageai", {"model_name": "v"}, "text2vec-voyageai", {"model": "v"}),
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
    assert v.options == {"model": "x"}


def test_google_vertex():
    v, _ = vectorizer_for(
        known("google_vertex", model_name="m", project_id="p", region="europe-west1")
    )
    assert v is not None
    assert v.module == "text2vec-google" and v.options["project_id"] == "p"
    assert v.options == {"model": "m", "project_id": "p", "location": "europe-west1"}


@pytest.mark.parametrize(
    ("name", "config", "options"),
    [
        ("voyageai", {"model_name": "voyage-3-large"}, {"model": "voyage-3-large"}),
        (
            "google_generative_ai",
            {"model_name": "models/text-embedding-004", "dimension": 768},
            {"model": "text-embedding-004", "dimensions": 768},
        ),
        (
            "google_gemini",
            {"model_name": "gemini-embedding-001"},
            {"model": "gemini-embedding-001"},
        ),
        (
            "google_gemini",
            {"model_name": "gemini-embedding-001", "vertexai": False, "project": "p"},
            {"model": "gemini-embedding-001"},
        ),
        (
            "google_gemini",
            {
                "model_name": "gemini-embedding-001",
                "vertexai": True,
                "project": "p",
                "location": "europe-west4",
            },
            {
                "model": "gemini-embedding-001",
                "project_id": "p",
                "location": "europe-west4",
            },
        ),
        (
            "google_vertex",
            {
                "model_name": "text-embedding-005",
                "project_id": "p",
                "region": "us-central1",
            },
            {
                "model": "text-embedding-005",
                "project_id": "p",
                "location": "us-central1",
            },
        ),
    ],
)
def test_voyage_and_google_keep_the_model(name, config, options):
    v, _ = vectorizer_for(known(name, **config))
    assert v is not None
    assert v.options == options


@pytest.mark.parametrize(
    ("name", "config", "missing"),
    [
        ("voyageai", {}, "model_name"),
        ("google_generative_ai", {"dimension": 768}, "model_name"),
        ("google_gemini", {}, "model_name"),
        ("google_gemini", {"vertexai": True, "project": "p"}, "model_name"),
        ("google_gemini", {"model_name": "m", "vertexai": True}, "project"),
        ("google_vertex", {"project_id": "p"}, "model_name"),
        ("google_vertex", {"model_name": "m", "project_id": None}, "project_id"),
    ],
)
def test_voyage_and_google_without_a_required_key_have_no_vectorizer(
    name, config, missing
):
    v, notes = vectorizer_for(known(name, **config))
    assert v is None
    assert len(notes) == 1 and missing in notes[0] and name in notes[0]


@pytest.mark.parametrize(
    ("name", "config", "expected"),
    [
        ("voyageai", {"model_name": "voyage-3-large"}, {"model": "voyage-3-large"}),
        (
            "google_generative_ai",
            {"model_name": "models/text-embedding-004", "dimension": 768},
            {"modelId": "text-embedding-004", "dimensions": 768, "projectId": None},
        ),
        (
            "google_gemini",
            {"model_name": "m", "vertexai": True, "project": "p", "location": "l"},
            {"modelId": "m", "projectId": "p", "location": "l"},
        ),
        (
            "google_gemini",
            {"model_name": "m", "dimension": 256},
            {"modelId": "m", "dimensions": 256, "projectId": None},
        ),
        (
            "google_gemini",
            {"model_name": "m", "dimension": 256, "vertexai": True, "project": "p"},
            {"modelId": "m", "dimensions": 256, "projectId": "p"},
        ),
        (
            "google_vertex",
            {"model_name": "m", "project_id": "p", "region": "l"},
            {"modelId": "m", "projectId": "p", "location": "l"},
        ),
    ],
)
def test_voyage_and_google_options_reach_the_collection_config(name, config, expected):
    v, _ = vectorizer_for(known(name, **config))
    entry = CollectionPlan(
        source=SourceRef(
            ref="r",
            namespace="n",
            name="docs",
            count=1,
            dimension=2,
            distance="cosine",
            embedding=None,
        ),
        target=TargetPlan(
            name="Docs", vectors=[VectorPlan(vectorizer=v)], properties=[]
        ),
    )
    found = collection_kwargs([entry])["vector_config"][0].vectorizer
    assert {key: getattr(found, key) for key in expected} == expected


@pytest.mark.parametrize(
    "variant",
    [{}, {"vertexai": True, "project": "p", "location": "europe-west4"}],
)
def test_google_gemini_maps_the_dimension(variant):
    v, _ = vectorizer_for(
        known(
            "google_gemini", model_name="gemini-embedding-001", dimension=256, **variant
        )
    )
    assert v is not None
    assert v.options["dimensions"] == 256


def test_default_model_has_no_mapping():
    v, notes = vectorizer_for(known("default"))
    assert v is None and any("all-MiniLM-L6-v2" in n for n in notes)


def test_chroma_hosted_model_note():
    v, notes = vectorizer_for(known("chroma-cloud-qwen"))
    assert v is None and any("embed.trychroma.com" in n for n in notes)


def test_unmapped_name_is_named_in_a_note():
    v, notes = vectorizer_for(known("onnx_thing"))
    assert v is None and any("onnx_thing" in n for n in notes)


def test_config_keys_are_the_keys_the_mapping_reads():
    assert CONFIG_KEYS == {
        "openai": {"model_name", "dimensions", "api_base", "api_type"},
        "cohere": {"model_name"},
        "jina": {"model_name", "dimensions"},
        "voyageai": {"model_name"},
        "mistral": {"model"},
        "huggingface": {"model_name"},
        "google_generative_ai": {"model_name", "dimension"},
        "google_gemini": {
            "model_name",
            "dimension",
            "vertexai",
            "project",
            "location",
        },
        "google_vertex": {"model_name", "project_id", "region"},
        "ollama": {"url", "model_name"},
    }


def test_stored_embedding_keeps_only_the_listed_keys():
    embedding = {
        **known(
            "openai",
            api_key_env_var="OPENAI_API_KEY",
            model_name="m",
            api_base="https://bob:made-up-pw@proxy.example.com/v1?k=made-up-q#f",
            default_headers={"Authorization": "Bearer made-up-header"},
            organization_id=None,
            api_type=None,
        ),
        "extra": "made-up-extra",
    }
    stored, notes = stored_embedding(embedding)
    assert stored == {
        "type": "known",
        "name": "openai",
        "config": {
            "model_name": "m",
            "api_base": "https://proxy.example.com/v1",
            "api_type": None,
        },
    }
    assert len(notes) == 1
    for key in ("api_key_env_var", "default_headers", "extra"):
        assert key in notes[0]
    for value in ("made-up", "bob", "OPENAI_API_KEY", "organization_id"):
        assert value not in notes[0]


def test_stored_embedding_names_only_the_dropped_keys_with_a_value():
    nulls = {
        "api_key_env_var": None,
        "organization_id": None,
        "api_version": None,
        "deployment_id": None,
        "default_headers": None,
    }
    stored, notes = stored_embedding(known("openai", model_name="m", **nulls))
    assert stored == known("openai", model_name="m")
    assert notes == []
    _, notes = stored_embedding(
        known("openai", model_name="m", **{**nulls, "organization_id": "made-up"})
    )
    assert len(notes) == 1 and "organization_id" in notes[0]
    assert [key for key in nulls if key in notes[0]] == ["organization_id"]


def test_stored_embedding_of_an_unmapped_function_keeps_no_config_key():
    stored, notes = stored_embedding(known("onnx_thing", token="made-up-token"))
    assert stored == {"type": "known", "name": "onnx_thing", "config": {}}
    assert len(notes) == 1 and "token" in notes[0] and "made-up" not in notes[0]


def test_stored_embedding_cleans_the_ollama_url():
    stored, _ = stored_embedding(
        known("ollama", url="http://u:made-up@ollama:11434/api?x=1#f", model_name="n")
    )
    assert stored is not None
    assert stored["config"] == {"url": "http://ollama:11434/api", "model_name": "n"}


@pytest.mark.parametrize("char", "/?#")
def test_stored_embedding_drops_a_url_that_keeps_its_userinfo(char):
    password = f"made{char}up-pw"
    for name, key in (("openai", "api_base"), ("ollama", "url")):
        url = f"https://bob:{password}@proxy.example.com/v1"
        stored, notes = stored_embedding(known(name, model_name="m", **{key: url}))
        assert stored == known(name, model_name="m")
        assert notes == [
            f"The keys {key} of the embedding function {name!r} are not stored. "
            f"The value of {key} holds userinfo that cannot be removed."
        ]
        v, _ = vectorizer_for(stored)
        assert v is not None
        for value in ("bob", "made", "up-pw", "@"):
            assert value not in repr(stored) + notes[0] + repr(v)


def test_stored_embedding_note_names_unread_and_unclean_keys():
    embedding = known(
        "openai",
        model_name="m",
        api_base="https://bob:made/up@proxy.example.com/v1",
        organization_id="made-up-org",
    )
    _, notes = stored_embedding(embedding)
    assert notes == [
        "The keys api_base, organization_id of the embedding function 'openai' are "
        "not stored; the mapping does not read organization_id. The value of "
        "api_base holds userinfo that cannot be removed."
    ]


def test_stored_embedding_without_dropped_keys_has_no_note():
    embedding = known("cohere", model_name="c")
    assert stored_embedding(embedding) == (embedding, [])


def test_stored_embedding_of_none():
    assert stored_embedding(None) == (None, [])


def test_keys_outside_the_table_are_ignored():
    full = known(
        "openai",
        api_key_env_var="OPENAI_API_KEY",
        model_name="m",
        api_base="https://proxy.example.com/v1",
        default_headers={"X-Custom": "made-up"},
        organization_id="o",
    )
    stored, _ = stored_embedding(full)
    assert vectorizer_for(full) == vectorizer_for(stored)


@pytest.mark.parametrize(
    ("name", "config", "options"),
    [
        (
            "openai",
            {
                "model_name": "m",
                "api_base": "https://bob:made-up-pw@proxy.example.com/v1?k=made-up#f",
            },
            {"model": "m", "base_url": "https://proxy.example.com"},
        ),
        (
            "ollama",
            {"model_name": "n", "url": "http://bob:made-up-pw@ollama:11434?k=made-up"},
            {"model": "n", "api_endpoint": "http://ollama:11434"},
        ),
    ],
)
def test_vectorizer_options_have_no_userinfo_query_or_fragment(name, config, options):
    v, notes = vectorizer_for(known(name, **config))
    assert v is not None
    assert v.options == options
    assert not [n for n in notes if "made-up" in n or "bob" in n]
