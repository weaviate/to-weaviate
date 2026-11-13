import re

import weaviate
from weaviate.auth import Auth

from to_weaviate.settings import WeaviateSettings

# The client's error for a version text that is not (v)x.y.z.
_UNPARSED_VERSION = re.compile(
    r"Unable to parse a version from the input string: (.*)\. Is it in the format",
    re.DOTALL,
)


class VersionUnreadable(Exception):
    """The server reports a version text that the client cannot parse, or none.

    `reported` is that text; None when the server's answer has no version.
    """

    def __init__(self, reported: str | None) -> None:
        super().__init__(reported)
        self.reported = reported


def connect(settings: WeaviateSettings) -> weaviate.WeaviateClient:
    """Raises VersionUnreadable for a server whose version the client cannot parse.

    The client raises a bare ValueError or KeyError for it during startup.
    """
    try:
        return _connect(settings)
    except ValueError as exc:
        unparsed = _UNPARSED_VERSION.search(str(exc))
        if unparsed is None:
            raise
        raise VersionUnreadable(unparsed.group(1)) from exc
    except KeyError as exc:
        if exc.args != ("version",):
            raise
        raise VersionUnreadable(None) from exc


def _connect(settings: WeaviateSettings) -> weaviate.WeaviateClient:
    auth = Auth.api_key(settings.api_key) if settings.api_key else None
    headers = settings.headers or None
    if settings.grpc is not None:
        http, grpc = settings.http, settings.grpc
        return weaviate.connect_to_custom(
            http_host=http.host,
            http_port=http.port,
            http_secure=http.secure,
            grpc_host=grpc.host,
            grpc_port=grpc.port,
            grpc_secure=grpc.secure,
            headers=headers,
            auth_credentials=auth,
        )
    # The client takes the host from a value that starts with "http" by
    # parsing it as a URL, so a bare host such as "httpx1.weaviate.cloud"
    # would become empty. Its port is always 443.
    return weaviate.connect_to_weaviate_cloud(
        cluster_url=f"https://{settings.http.host}",
        auth_credentials=auth,  # type: ignore[arg-type]
        headers=headers,
    )
