import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

from dotenv import dotenv_values

DEFAULT_CHROMA_URL = "https://api.trychroma.com"
DEFAULT_TOKEN_HEADER = "x-chroma-token"
CHROMA_CLOUD_DOMAIN = "trychroma.com"

# Provider environment variable to Weaviate header. Copied from
# agent-skills/skills/weaviate/scripts/weaviate_conn.py. When both the
# APIKEY and API_KEY variants are set, the first one listed wins.
API_KEY_MAP = {
    "ANTHROPIC_APIKEY": "X-Anthropic-Api-Key",
    "ANTHROPIC_API_KEY": "X-Anthropic-Api-Key",
    "ANYSCALE_APIKEY": "X-Anyscale-Api-Key",
    "ANYSCALE_API_KEY": "X-Anyscale-Api-Key",
    "AWS_ACCESS_KEY": "X-Aws-Access-Key",
    "AWS_SECRET_KEY": "X-Aws-Secret-Key",
    "COHERE_APIKEY": "X-Cohere-Api-Key",
    "COHERE_API_KEY": "X-Cohere-Api-Key",
    "DATABRICKS_TOKEN": "X-Databricks-Token",
    "FRIENDLI_TOKEN": "X-Friendli-Api-Key",
    "VERTEX_APIKEY": "X-Goog-Vertex-Api-Key",
    "VERTEX_API_KEY": "X-Goog-Vertex-Api-Key",
    "STUDIO_APIKEY": "X-Goog-Studio-Api-Key",
    "STUDIO_API_KEY": "X-Goog-Studio-Api-Key",
    "HUGGINGFACE_APIKEY": "X-HuggingFace-Api-Key",
    "HUGGINGFACE_API_KEY": "X-HuggingFace-Api-Key",
    "JINAAI_APIKEY": "X-JinaAI-Api-Key",
    "JINAAI_API_KEY": "X-JinaAI-Api-Key",
    "MISTRAL_APIKEY": "X-Mistral-Api-Key",
    "MISTRAL_API_KEY": "X-Mistral-Api-Key",
    "NVIDIA_APIKEY": "X-Nvidia-Api-Key",
    "NVIDIA_API_KEY": "X-Nvidia-Api-Key",
    "OPENAI_APIKEY": "X-OpenAI-Api-Key",
    "OPENAI_API_KEY": "X-OpenAI-Api-Key",
    "AZURE_APIKEY": "X-Azure-Api-Key",
    "AZURE_API_KEY": "X-Azure-Api-Key",
    "VOYAGE_APIKEY": "X-Voyage-Api-Key",
    "VOYAGE_API_KEY": "X-Voyage-Api-Key",
    "XAI_APIKEY": "X-Xai-Api-Key",
    "XAI_API_KEY": "X-Xai-Api-Key",
}


class MissingCredential(Exception):
    def __init__(self, var: str) -> None:
        super().__init__(f"{var} is not set")
        self.var = var


@dataclass(frozen=True)
class ChromaSettings:
    url: str
    api_key: str | None = field(repr=False)
    tenant: str | None
    database: str | None
    token_header: str


@dataclass(frozen=True)
class WeaviateSettings:
    url: str
    api_key: str | None = field(repr=False)
    grpc_url: str | None
    headers: dict[str, str] = field(repr=False)


def load_env(cwd: Path) -> dict[str, str]:
    """Merge `.env` in `cwd` with `os.environ`; `os.environ` wins, unchanged."""
    merged = {k: v for k, v in dotenv_values(cwd / ".env").items() if v is not None}
    merged.update(os.environ)
    return merged


def _get(env: Mapping[str, str], name: str) -> str | None:
    value = env.get(name, "").strip()
    return value or None


def _is_chroma_cloud(url: str) -> bool:
    host = urlparse(url).hostname or ""
    return host == CHROMA_CLOUD_DOMAIN or host.endswith("." + CHROMA_CLOUD_DOMAIN)


def chroma_settings(env: Mapping[str, str]) -> ChromaSettings:
    url = _get(env, "CHROMA_URL") or DEFAULT_CHROMA_URL
    api_key = _get(env, "CHROMA_API_KEY")
    if api_key is None and _is_chroma_cloud(url):
        raise MissingCredential("CHROMA_API_KEY")
    return ChromaSettings(
        url=url,
        api_key=api_key,
        tenant=_get(env, "CHROMA_TENANT"),
        database=_get(env, "CHROMA_DATABASE"),
        token_header=_get(env, "CHROMA_TOKEN_HEADER") or DEFAULT_TOKEN_HEADER,
    )


def weaviate_settings(env: Mapping[str, str]) -> WeaviateSettings:
    url = _get(env, "WEAVIATE_URL")
    if url is None:
        raise MissingCredential("WEAVIATE_URL")
    headers: dict[str, str] = {}
    for var, header in API_KEY_MAP.items():
        value = _get(env, var)
        if value is not None and header not in headers:
            headers[header] = value
    return WeaviateSettings(
        url=url,
        api_key=_get(env, "WEAVIATE_API_KEY"),
        grpc_url=_get(env, "WEAVIATE_GRPC_URL"),
        headers=headers,
    )
