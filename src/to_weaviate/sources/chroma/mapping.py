from typing import Any, NamedTuple
from urllib.parse import urlsplit

from to_weaviate.plan import VectorizerPlan
from to_weaviate.settings import stored_url

OPENAI_HOST = "api.openai.com"
# The OpenAI client's default base URL, https://api.openai.com/v1, after
# `_openai_base`. Weaviate uses it when base_url is not set.
OPENAI_DEFAULT_BASE = f"https://{OPENAI_HOST}"
# The model of chromadb's OpenAI embedding function when none is given.
OPENAI_ADA_002 = "text-embedding-ada-002"


class _Function(NamedTuple):
    module: str
    renames: dict[str, str]  # Chroma config key to Weaviate option name
    # Keys without which there is no vectorizer. Without the model, Weaviate
    # would use its own default model, not the one that made the vectors.
    # A Vertex mapping also needs the project: `project` for google_gemini,
    # `project_id` for google_vertex.
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
        "text2vec-google",
        {"model_name": "model", "dimension": "dimensions"},
        ("model_name",),
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
    {
        "model_name": "model",
        "dimension": "dimensions",
        "project": "project_id",
        "location": "location",
    },
    ("model_name", "project"),
)

# Config keys that `vectorizer_for` reads, per Chroma embedding function.
# inspect.json and plan.yaml store only these keys. default_headers is never
# one of them, because a header can hold a token.
CONFIG_KEYS: dict[str, frozenset[str]] = {
    **{name: frozenset(function.renames) for name, function in _FUNCTIONS.items()},
    "openai": frozenset({*_FUNCTIONS["openai"].renames, "api_type"}),
    "google_gemini": frozenset(
        {*_FUNCTIONS["google_gemini"].renames, *_GEMINI_VERTEX.renames, "vertexai"}
    ),
}
_URL_KEYS = frozenset({"api_base", "url"})
_EMBEDDING_KEYS = frozenset({"type", "name", "config"})


def _config_of(embedding: dict[str, Any]) -> dict[str, Any]:
    config = embedding.get("config")
    return config if isinstance(config, dict) else {}


def _table_keys(name: Any) -> frozenset[str]:
    return CONFIG_KEYS.get(name, frozenset())


def _stored_config(name: Any, config: dict[str, Any]) -> dict[str, Any]:
    """The keys of `config` in CONFIG_KEYS; URL values via `stored_url`.

    A URL value that `stored_url` does not store is left out.
    """
    keys = _table_keys(name)
    stored: dict[str, Any] = {}
    for key, value in config.items():
        if key not in keys:
            continue
        if key in _URL_KEYS and isinstance(value, str):
            value = stored_url(value)
            if value is None:
                continue
        stored[key] = value
    return stored


def stored_embedding(
    embedding: dict[str, Any] | None,
) -> tuple[dict[str, Any] | None, list[str]]:
    """`embedding` as inspect.json and plan.yaml store it, and notes.

    The stored form has the type, the name and the config keys in
    CONFIG_KEYS. A URL value has no userinfo, query or fragment; a URL value
    that keeps its userinfo is dropped. The note names the dropped keys whose
    value is not null, never their values.
    """
    if embedding is None:
        return None, []
    name = embedding.get("name")
    config = _config_of(embedding)
    kept = _stored_config(name, config)
    stored = {"type": embedding.get("type"), "name": name, "config": kept}
    dropped = sorted(
        {k for k, v in embedding.items() if k not in _EMBEDDING_KEYS and v is not None}
        | {k for k, v in config.items() if k not in kept and v is not None}
    )
    if not dropped:
        return stored, []
    # A key that the mapping reads is dropped only for its URL value.
    unclean = [k for k in dropped if k in _table_keys(name)]
    unread = [k for k in dropped if k not in unclean]
    reason = ""
    if unread:
        reason = "; the mapping does not read "
        reason += ", ".join(unread) if unclean else "them"
    note = (
        f"The keys {', '.join(dropped)} of the embedding function {name!r} are "
        f"not stored{reason}."
    )
    note += "".join(
        f" The value of {key} holds userinfo that cannot be removed." for key in unclean
    )
    return stored, [note]


def _openai_host(base_url: str) -> bool:
    """Whether the host of `base_url` is api.openai.com or a subdomain of it.

    Weaviate applies its OpenAI model names to every base URL that contains
    api.openai.com. This rule is narrower: a URL that holds the name outside
    the host, or in a longer host name, is another server.
    """
    try:
        host = urlsplit(base_url).hostname or ""
    except ValueError:
        return False
    return host == OPENAI_HOST or host.endswith("." + OPENAI_HOST)


def _openai_base(api_base: str) -> str:
    """`api_base` without one trailing `/v1` and without a trailing slash.

    The OpenAI client takes a base URL that ends in `/v1`. Weaviate adds
    `/v1/embeddings` to its base_url.
    """
    return api_base.removesuffix("/").removesuffix("/v1").removesuffix("/")


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
    config = _stored_config(name, _config_of(embedding))

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
    notes: list[str] = []
    model = options.get("model")
    if name == "google_generative_ai" and isinstance(model, str):
        options["model"] = model.removeprefix("models/")
    base_url = options.get("base_url")
    if isinstance(base_url, str):
        base_url = _openai_base(base_url)
        if base_url == OPENAI_DEFAULT_BASE:
            del options["base_url"]
        else:
            options["base_url"] = base_url
            notes.append(
                f"text2vec-openai uses base_url {base_url}; the Weaviate server "
                "must reach this host."
            )
    # On OpenAI's own endpoint Weaviate accepts this model only as `ada`,
    # which with its default modelVersion 002 requests text-embedding-ada-002.
    # With any other base_url it sends the model name unchanged.
    if (
        function.module == "text2vec-openai"
        and model == OPENAI_ADA_002
        and _openai_host(options.get("base_url", OPENAI_DEFAULT_BASE))
    ):
        options["model"] = "ada"
    return VectorizerPlan(module=function.module, options=options), notes
