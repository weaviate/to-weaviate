from urllib.parse import urlparse

import weaviate
from weaviate.auth import Auth

from to_weaviate.settings import WeaviateSettings


def _endpoint(url: str) -> tuple[str, int, bool]:
    parsed = urlparse(url if "://" in url else f"http://{url}")
    secure = parsed.scheme == "https"
    port = parsed.port or (443 if secure else 80)
    return parsed.hostname or "", port, secure


def connect(settings: WeaviateSettings) -> weaviate.WeaviateClient:
    auth = Auth.api_key(settings.api_key) if settings.api_key else None
    headers = settings.headers or None
    if settings.grpc_url:
        http_host, http_port, http_secure = _endpoint(settings.url)
        grpc_host, grpc_port, grpc_secure = _endpoint(settings.grpc_url)
        return weaviate.connect_to_custom(
            http_host=http_host,
            http_port=http_port,
            http_secure=http_secure,
            grpc_host=grpc_host,
            grpc_port=grpc_port,
            grpc_secure=grpc_secure,
            headers=headers,
            auth_credentials=auth,
        )
    return weaviate.connect_to_weaviate_cloud(
        cluster_url=settings.url,
        auth_credentials=auth,  # type: ignore[arg-type]
        headers=headers,
    )
