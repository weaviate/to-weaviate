#!/bin/sh
# Starts and stops the Weaviate of the acceptance suite in Docker, for a
# machine where `make acceptance-up` cannot create its kind cluster. Weaviate
# listens on 127.0.0.1 ports 8099 and 50099, accepts the API key admin-key and
# vectorizes through a text2vec-transformers container.
#
# Usage: weaviate-docker.sh up|down. Weaviate runs with ASYNC_INDEXING=true,
# as every Weaviate Cloud instance does; ASYNC=0 turns it off.
set -eu

# With CDPATH set, cd prints the directory, and ROOT would get two lines.
ROOT="$(CDPATH='' cd -- "$(dirname -- "$0")/../.." && pwd)"
VERSION_FILE="$ROOT/tests/acceptance/weaviate-version"
# The model of the CI action: 384 dimensions.
TRANSFORMERS_IMAGE=semitechnologies/transformers-inference:baai-bge-small-en-v1.5-onnx
WEAVIATE=to-weaviate-dev
TRANSFORMERS=to-weaviate-t2v
NETWORK=to-weaviate-dev-net
KEY=admin-key
READY_URL=http://localhost:8099/v1/.well-known/ready
READY_TRIES=60
READY_INTERVAL=2

# exists reads every failed inspect as "does not exist", so the daemon is
# checked first.
require_docker() {
    if ! docker info >/dev/null 2>&1; then
        echo "docker info failed, so Docker is not running or not reachable. Nothing was $1." >&2
        exit 1
    fi
}

exists() {
    docker "$1" inspect "$2" >/dev/null 2>&1
}

up() {
    case "${ASYNC:-1}" in
    1) async_indexing=true ;;
    0) async_indexing=false ;;
    *)
        echo "ASYNC must be 0 or 1. Nothing was started." >&2
        exit 2
        ;;
    esac
    version=""
    # read returns 1 for a last line without a newline, and still sets version.
    read -r version <"$VERSION_FILE" || true
    if [ -z "$version" ]; then
        echo "$VERSION_FILE holds no Weaviate version. Nothing was started." >&2
        exit 1
    fi
    require_docker started
    found=""
    exists container "$WEAVIATE" && found="$found $WEAVIATE"
    exists container "$TRANSFORMERS" && found="$found $TRANSFORMERS"
    exists network "$NETWORK" && found="$found $NETWORK"
    if [ -n "$found" ]; then
        echo "These exist:$found. Run make weaviate-down first. Nothing was started." >&2
        exit 1
    fi

    docker network create "$NETWORK" >/dev/null
    docker run -d --name "$TRANSFORMERS" --network "$NETWORK" \
        -e ENABLE_CUDA=0 \
        "$TRANSFORMERS_IMAGE" >/dev/null
    docker run -d --name "$WEAVIATE" --network "$NETWORK" \
        -p 127.0.0.1:8099:8080 -p 127.0.0.1:50099:50051 \
        -e AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=false \
        -e AUTHENTICATION_APIKEY_ENABLED=true \
        -e AUTHENTICATION_APIKEY_ALLOWED_KEYS="$KEY" \
        -e AUTHENTICATION_APIKEY_USERS=admin-user \
        -e AUTHORIZATION_RBAC_ENABLED=true \
        -e AUTHORIZATION_RBAC_ROOT_USERS=admin-user \
        -e PERSISTENCE_DATA_PATH=/var/lib/weaviate \
        -e DEFAULT_VECTORIZER_MODULE=none \
        -e ENABLE_MODULES=text2vec-transformers \
        -e TRANSFORMERS_INFERENCE_API="http://$TRANSFORMERS:8080" \
        -e CLUSTER_HOSTNAME=node1 \
        -e ASYNC_INDEXING="$async_indexing" \
        "cr.weaviate.io/semitechnologies/weaviate:$version" >/dev/null

    tries=0
    until curl -fs -o /dev/null --max-time 5 \
        -H "Authorization: Bearer $KEY" "$READY_URL"; do
        tries=$((tries + 1))
        if [ "$tries" -ge "$READY_TRIES" ]; then
            echo "Weaviate is not ready after $READY_TRIES tries, $READY_INTERVAL s apart." \
                "Read docker logs $WEAVIATE, then run make weaviate-down." >&2
            exit 1
        fi
        sleep "$READY_INTERVAL"
    done
    echo "Weaviate $version is ready on ports 8099 and 50099" \
        "(ASYNC_INDEXING=$async_indexing)."
}

down() {
    require_docker removed
    removed=""
    for name in "$WEAVIATE" "$TRANSFORMERS"; do
        if exists container "$name"; then
            docker rm -f "$name" >/dev/null
            removed="$removed $name"
        fi
    done
    if exists network "$NETWORK"; then
        docker network rm "$NETWORK" >/dev/null
        removed="$removed $NETWORK"
    fi
    if [ -n "$removed" ]; then
        echo "Removed:$removed."
    else
        echo "Docker lists none of $WEAVIATE, $TRANSFORMERS and $NETWORK. Nothing was removed."
    fi
}

case "${1:-}" in
up) up ;;
down) down ;;
*)
    echo "usage: $0 up|down" >&2
    exit 2
    ;;
esac
