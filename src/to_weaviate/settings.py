import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import SplitResult, unquote, urlsplit, urlunsplit

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


URL_VARIABLES = ("CHROMA_URL", "WEAVIATE_URL", "WEAVIATE_GRPC_URL")
URL_HINT = "Percent-encode #, ?, / and @ in a password."
HTTP_SCHEMES = frozenset({"http", "https"})


class InvalidSetting(Exception):
    """A variable holds a value that the tool refuses.

    The message names the variable and never holds the value. `hint` says
    how to fix it.
    """

    def __init__(self, message: str, hint: str) -> None:
        super().__init__(message)
        self.hint = hint


def _split(url: str) -> tuple[SplitResult, str]:
    """`url` split, with `//` put in front of a URL without a scheme."""
    prefix = "" if "://" in url else "//"
    return urlsplit(prefix + url), prefix


def _plain_userinfo(url: str) -> tuple[str | None, str]:
    """The userinfo of `url` and `url` without it, by plain string search.

    For URLs that `urlsplit` refuses. The authority runs from after `://`, or
    from the start, to the first `/`. The userinfo is the part of the
    authority before its last `@`.
    """
    scheme_end = url.find("://")
    start = 0 if scheme_end < 0 else scheme_end + 3
    end = url.find("/", start)
    at = url.rfind("@", start, len(url) if end < 0 else end)
    if at < 0:
        return None, url
    return url[start:at], url[:start] + url[at + 1 :]


def without_userinfo(url: str) -> str:
    """`url` without its `user:password@` part. Never raises."""
    try:
        parts, prefix = _split(url)
    except ValueError:
        return _plain_userinfo(url)[1]
    if "@" not in parts.netloc:
        return url
    netloc = parts.netloc.rsplit("@", 1)[1]
    return urlunsplit(parts._replace(netloc=netloc))[len(prefix) :]


def stored_url(url: str) -> str:
    """`url` without its userinfo, query and fragment. Never raises.

    The form of a URL that the tool writes to a file or prints, because a
    query or a fragment can hold a token.
    """
    bare = without_userinfo(url)
    ends = [i for i in (bare.find("?"), bare.find("#")) if i >= 0]
    return bare[: min(ends)] if ends else bare


@dataclass(frozen=True)
class Endpoint:
    """A host, a port and whether TLS is used.

    `host` is in lower case, without userinfo. An IPv6 host has no brackets.
    """

    host: str
    port: int
    secure: bool

    @property
    def normalized(self) -> str:
        """`host:port`, equal for two ways to write one server.

        The host has no trailing dot. An IPv6 host is in brackets.
        """
        host = self.host.removesuffix(".")
        if ":" in host:
            host = f"[{host}]"
        return f"{host}:{self.port}"


def _endpoint(parts: SplitResult, default_scheme: str) -> Endpoint:
    """The endpoint of `parts`; a URL without a scheme gets `default_scheme`.

    A URL without a port gets 443 for https and 80 for any other scheme.
    """
    secure = (parts.scheme or default_scheme) == "https"
    return Endpoint(parts.hostname or "", parts.port or (443 if secure else 80), secure)


def normalized_host(url: str, default_scheme: str = "http") -> str:
    """The `host:port` of `url`, equal for two ways to write one server.

    See `Endpoint.normalized`. The userinfo, path, query and fragment are
    dropped. A URL without a port gets 443 for https and 80 for any other
    scheme. A URL without a scheme gets `default_scheme`.

    Raises ValueError when `url` cannot be split or its port is not a number.
    """
    return _endpoint(_split(url)[0], default_scheme).normalized


def _parsed_url(
    var: str, url: str, schemes: frozenset[str] | None, any_path: bool
) -> SplitResult:
    """`url` split, or InvalidSetting when the tool refuses it.

    Refused for every variable: a value that cannot be split, a port that is
    not a number, a fragment, and an `@` in the path or the query. An
    unencoded `#`, `?` or `/` in a password gives these shapes. `schemes` are
    the accepted schemes, with "" for none; None accepts any scheme. Without
    `any_path`, the path is empty or `/`.
    """
    message = f"{var} is not a valid URL."
    try:
        parts, _ = _split(url)
        parts.port  # noqa: B018 - raises ValueError for a port that is not a number
    except ValueError:
        # The ValueError can quote the port text, which can be a password.
        raise InvalidSetting(message, URL_HINT) from None
    if (
        "#" in url
        or "@" in parts.path + parts.query
        or (schemes is not None and parts.scheme not in schemes)
        or (not any_path and parts.path not in ("", "/"))
    ):
        raise InvalidSetting(message, URL_HINT)
    return parts


def _userinfo(url: str) -> str | None:
    """The userinfo of `url`, or None. Never raises."""
    try:
        netloc = _split(url)[0].netloc
    except ValueError:
        return _plain_userinfo(url)[0]
    return netloc.rpartition("@")[0] if "@" in netloc else None


def url_tokens(url: str) -> list[str]:
    """The userinfo of `url` when it has no colon, as written and decoded.

    A userinfo without a colon, as in `https://TOKEN@host`, is a token.
    Never raises.
    """
    userinfo = _userinfo(url)
    if not userinfo or ":" in userinfo:
        return []
    return list(dict.fromkeys([userinfo, unquote(userinfo)]))


def url_passwords(url: str) -> list[str]:
    """The password of `url`, as written and percent-decoded. Never raises."""
    password: str | None
    try:
        password = _split(url)[0].password
    except ValueError:
        userinfo = _plain_userinfo(url)[0] or ""
        password = userinfo.partition(":")[2] if ":" in userinfo else None
    if not password:
        return []
    return list(dict.fromkeys([password, unquote(password)]))


class MissingCredential(Exception):
    def __init__(self, var: str) -> None:
        super().__init__(f"{var} is not set")
        self.var = var


@dataclass(frozen=True)
class ChromaSettings:
    """Raises InvalidSetting when `url` is refused. CHROMA_URL needs http or https.

    `host` is the host of `url` in lower case, without a trailing dot.
    """

    url: str
    api_key: str | None = field(repr=False)
    tenant: str | None
    database: str | None
    token_header: str
    host: str = field(init=False)

    def __post_init__(self) -> None:
        parts = _parsed_url("CHROMA_URL", self.url, HTTP_SCHEMES, any_path=True)
        object.__setattr__(self, "host", (parts.hostname or "").removesuffix("."))


@dataclass(frozen=True)
class WeaviateSettings:
    """Raises InvalidSetting when `url` or `grpc_url` is refused.

    Both have no path other than `/`. A `grpc_url` scheme is http or https.
    `http` is the endpoint of `url` and `grpc` the endpoint of `grpc_url`; a
    URL without a scheme is http. `target.connect` and the stored target
    host read both. Without `grpc_url`, `connect` uses
    `connect_to_weaviate_cloud`, which connects over TLS to port 443 whatever
    the scheme says, so `http` is secure and a URL without a port gets 443.
    """

    url: str
    api_key: str | None = field(repr=False)
    grpc_url: str | None
    headers: dict[str, str] = field(repr=False)
    http: Endpoint = field(init=False)
    grpc: Endpoint | None = field(init=False)

    def __post_init__(self) -> None:
        http = _parsed_url("WEAVIATE_URL", self.url, None, any_path=False)
        grpc = None
        if self.grpc_url:
            parts = _parsed_url(
                "WEAVIATE_GRPC_URL", self.grpc_url, HTTP_SCHEMES | {""}, any_path=False
            )
            grpc = _endpoint(parts, "http")
        if grpc is None:
            endpoint = Endpoint(http.hostname or "", http.port or 443, True)
        else:
            endpoint = _endpoint(http, "http")
        object.__setattr__(self, "http", endpoint)
        object.__setattr__(self, "grpc", grpc)


def load_env(cwd: Path) -> dict[str, str]:
    """Merge `.env` in `cwd` with `os.environ`; `os.environ` wins, unchanged."""
    merged = {k: v for k, v in dotenv_values(cwd / ".env").items() if v is not None}
    merged.update(os.environ)
    return merged


def _get(env: Mapping[str, str], name: str) -> str | None:
    value = env.get(name, "").strip()
    return value or None


def _is_chroma_cloud(host: str) -> bool:
    return host == CHROMA_CLOUD_DOMAIN or host.endswith("." + CHROMA_CLOUD_DOMAIN)


def chroma_settings(env: Mapping[str, str]) -> ChromaSettings:
    settings = ChromaSettings(
        url=_get(env, "CHROMA_URL") or DEFAULT_CHROMA_URL,
        api_key=_get(env, "CHROMA_API_KEY"),
        tenant=_get(env, "CHROMA_TENANT"),
        database=_get(env, "CHROMA_DATABASE"),
        token_header=_get(env, "CHROMA_TOKEN_HEADER") or DEFAULT_TOKEN_HEADER,
    )
    if settings.api_key is None and _is_chroma_cloud(settings.host):
        raise MissingCredential("CHROMA_API_KEY")
    return settings


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
