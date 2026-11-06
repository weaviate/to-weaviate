import pytest

from to_weaviate.plan import CollectionPlan, SourceRef, TargetPlan, VectorPlan
from to_weaviate.sources.chroma.mapping import vectorizer_for
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


def test_default_model_has_no_mapping():
    v, notes = vectorizer_for(known("default"))
    assert v is None and any("all-MiniLM-L6-v2" in n for n in notes)


def test_chroma_hosted_model_note():
    v, notes = vectorizer_for(known("chroma-cloud-qwen"))
    assert v is None and any("embed.trychroma.com" in n for n in notes)


def test_unmapped_name_is_named_in_a_note():
    v, notes = vectorizer_for(known("onnx_thing"))
    assert v is None and any("onnx_thing" in n for n in notes)
