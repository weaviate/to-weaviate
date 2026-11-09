from typing import Any

from to_weaviate.plan import VectorizerPlan

# Chroma config key to Weaviate option name, per Chroma embedding function.
_SIMPLE: dict[str, tuple[str, dict[str, str]]] = {
    "openai": (
        "text2vec-openai",
        {"model_name": "model", "dimensions": "dimensions", "api_base": "base_url"},
    ),
    "cohere": ("text2vec-cohere", {"model_name": "model"}),
    "jina": ("text2vec-jinaai", {"model_name": "model", "dimensions": "dimensions"}),
    "voyageai": ("text2vec-voyageai", {}),
    "mistral": ("text2vec-mistral", {"model": "model"}),
    "huggingface": ("text2vec-huggingface", {"model_name": "model"}),
    "google_generative_ai": ("text2vec-google", {}),
    "google_gemini": ("text2vec-google", {}),
    "google_vertex": ("text2vec-google", {"project_id": "project_id"}),
    "ollama": (
        "text2vec-ollama",
        {"url": "api_endpoint", "model_name": "model"},
    ),
}


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
    if name not in _SIMPLE:
        return None, [f"The embedding function {name!r} has no Weaviate vectorizer."]

    module, renames = _SIMPLE[name]
    options = {
        target: config[source]
        for source, target in renames.items()
        if config.get(source) is not None
    }
    return VectorizerPlan(module=module, options=options), []
