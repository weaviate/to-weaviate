import weaviate
from weaviate.auth import Auth

from to_weaviate.settings import WeaviateSettings


def connect(settings: WeaviateSettings) -> weaviate.WeaviateClient:
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
