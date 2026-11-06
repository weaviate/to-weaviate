from typing import Any, NamedTuple

from to_weaviate.plan import VectorizerPlan


class _Function(NamedTuple):
    module: str
    renames: dict[str, str]  # Chroma config key to Weaviate option name
    # Keys without which there is no vectorizer: without a model, Weaviate
    # would use its own default model, not the one that made the vectors.
    required: tuple[str, ...] = ()


_FUNCTIONS: dict[str, _Function] = {
    "openai": _Function(
        "text2vec-openai",
        {"model_name": "model", "dimensions": "dimensions", "api_base": "base_url"},
    ),
    "cohere": _Function("text2vec-cohere", {"model_name": "model"}),
    "jina": _Function(
        "text2vec-jinaai", {"model_name": "model", "dimensions": "dimensions"}
    ),
    "voyageai": _Function(
        "text2vec-voyageai", {"model_name": "model"}, ("model_name",)
    ),
    "mistral": _Function("text2vec-mistral", {"model": "model"}),
    "huggingface": _Function("text2vec-huggingface", {"model_name": "model"}),
    "google_generative_ai": _Function(
        "text2vec-google",
        {"model_name": "model", "dimension": "dimensions"},
        ("model_name",),
    ),
    "google_gemini": _Function(
        "text2vec-google", {"model_name": "model"}, ("model_name",)
    ),
    "google_vertex": _Function(
        "text2vec-google",
        {"model_name": "model", "project_id": "project_id", "region": "location"},
        ("model_name", "project_id"),
    ),
    "ollama": _Function(
        "text2vec-ollama", {"url": "api_endpoint", "model_name": "model"}
    ),
}

# google_gemini with `vertexai` true.
_GEMINI_VERTEX = _Function(
    "text2vec-google",
    {"model_name": "model", "project": "project_id", "location": "location"},
    ("model_name", "project"),
)


def _function(name: Any, config: dict[str, Any]) -> _Function | None:
    if name == "google_gemini" and config.get("vertexai") is True:
        return _GEMINI_VERTEX
    return _FUNCTIONS.get(name)


def vectorizer_for(
    embedding: dict[str, Any] | None,
) -> tuple[VectorizerPlan | None, list[str]]:
    """Map a Chroma embedding function to a Weaviate vectorizer and notes."""
    if embedding is None:
        return None, []
    name = embedding.get("name")
    config = embedding.get("config") or {}

    if name == "default":
        return None, [
            "Chroma's default embedding model is all-MiniLM-L6-v2, which has no "
            "Weaviate module. Options: keep the vectors and embed queries in the "
            "application, use text2vec-huggingface with that model, or "
            "re-vectorize with --vectors embed."
        ]
    if name == "chroma-cloud-qwen":
        return None, [
            "The embedding model chroma-cloud-qwen is served by "
            "embed.trychroma.com and has no Weaviate module."
        ]
    if name == "openai" and config.get("api_type") == "azure":
        return None, [
            "The OpenAI embedding function uses Azure; it is not mapped to a "
            "Weaviate vectorizer."
        ]
    function = _function(name, config)
    if function is None:
        return None, [f"The embedding function {name!r} has no Weaviate vectorizer."]
    missing = [key for key in function.required if config.get(key) is None]
    if missing:
        return None, [
            f"The embedding function {name!r} has no {', '.join(missing)} in its "
            "config; it is not mapped to a Weaviate vectorizer."
        ]

    options = {
        target: config[source]
        for source, target in function.renames.items()
        if config.get(source) is not None
    }
    model = options.get("model")
    if name == "google_generative_ai" and isinstance(model, str):
        options["model"] = model.removeprefix("models/")
    return VectorizerPlan(module=function.module, options=options), []
