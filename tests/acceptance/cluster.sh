#!/usr/bin/env bash
# Starts and stops the servers of the acceptance suite: Chroma from the compose
# file, and Weaviate in the kind cluster that weaviate-local-k8s creates.
#
# Usage: cluster.sh up|down. LOCAL_K8S names the weaviate-local-k8s checkout.
#
# `local-k8s.sh clean` runs `pkill -f` with two patterns, and Linux pkill also
# matches the ancestors of the pkill process. No ancestor of `clean` may hold
# either pattern in its command line, so the text that names them stays in
# this file and is printed by builtins and heredocs only.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOCAL_K8S="${LOCAL_K8S:-../weaviate-local-k8s}"
SCRIPT="$LOCAL_K8S/local-k8s.sh"
COMPOSE="$ROOT/tests/acceptance/chroma/docker-compose.yml"
VERSION_FILE="$ROOT/tests/acceptance/weaviate-version"
CLUSTER=weaviate-k8s
CONTROL_PLANE="$CLUSTER-control-plane"
# Holds the id of the control-plane container that `up` created. `down`
# deletes the cluster only while that container exists.
MARKER="$ROOT/.acceptance-kind-cluster"

control_plane_id() {
    docker inspect -f '{{.Id}}' "$CONTROL_PLANE" 2>/dev/null
}

require_script() {
    if [ ! -x "$SCRIPT" ]; then
        echo "$SCRIPT does not exist or is not executable." \
            "Set LOCAL_K8S to a clone of weaviate-local-k8s." >&2
        exit 1
    fi
}

up() {
    if ! command -v kind >/dev/null; then
        echo "kind is not on PATH. Install kind, then run make acceptance-up again." >&2
        exit 1
    fi
    require_script
    local clusters
    if ! clusters="$(kind get clusters)"; then
        echo "kind get clusters failed, so it is not known whether the cluster" \
            "$CLUSTER exists. Nothing was started." >&2
        exit 1
    fi
    if printf '%s\n' "$clusters" | grep -qx "$CLUSTER"; then
        echo "The kind cluster $CLUSTER exists. local-k8s.sh setup would reuse" \
            "its name. Nothing was started." >&2
        exit 1
    fi
    local version
    read -r version <"$VERSION_FILE"

    docker compose -f "$COMPOSE" up -d --wait
    # The default OBSERVABILITY=true of local-k8s.sh would take ports 3000 and 9091.
    if ! WEAVIATE_VERSION="$version" WORKERS=1 REPLICAS=1 \
        WEAVIATE_PORT=8099 WEAVIATE_GRPC_PORT=50099 RBAC=true \
        MODULES=text2vec-transformers EXPOSE_PODS=false OBSERVABILITY=false \
        "$SCRIPT" setup; then
        cat >&2 <<EOF
local-k8s.sh setup failed. A partly created kind cluster $CLUSTER may exist.
make acceptance-down will not remove it, because the marker is written only
after setup succeeds. To remove the cluster by hand, run:
  kind delete cluster --name $CLUSTER
EOF
        exit 1
    fi
    local id
    if ! id="$(control_plane_id)"; then
        cat >&2 <<EOF
local-k8s.sh setup succeeded, but the container $CONTROL_PLANE cannot be
inspected, so no marker was written and make acceptance-down will not remove
the cluster. To remove it by hand, run:
  kind delete cluster --name $CLUSTER
EOF
        exit 1
    fi
    printf '%s\n' "$id" >"$MARKER"
    echo "Created the kind cluster $CLUSTER. $MARKER holds the id of $CONTROL_PLANE."
}

down() {
    docker compose -f "$COMPOSE" down
    if [ ! -f "$MARKER" ]; then
        echo "This checkout did not create the kind cluster $CLUSTER" \
            "($MARKER does not exist). The cluster was left alone."
        return 0
    fi
    local recorded current
    read -r recorded <"$MARKER" || recorded=""
    current="$(control_plane_id)" || current=""
    if [ -z "$current" ]; then
        echo "The container $CONTROL_PLANE does not exist, so the cluster this" \
            "checkout created is gone. Removed the stale marker $MARKER."
        rm -f "$MARKER"
        return 0
    fi
    if [ "$current" != "$recorded" ]; then
        echo "The container $CONTROL_PLANE is not the one this checkout created." \
            "The cluster was left alone. Removed the stale marker $MARKER."
        rm -f "$MARKER"
        return 0
    fi
    require_script
    # The effects follow clean() in local-k8s.sh.
    cat <<EOF
local-k8s.sh clean does all of this:
  - kills every process on this machine whose command line contains kubectl-relay
  - kills every process on this machine whose command line contains 'kubectl get pods -n weaviate -w'
  - removes /tmp/weaviate_pod_states.log and /tmp/weaviate_pod_states.jq
  - switches the kubectl context to kind-$CLUSTER
  - uninstalls the Helm release weaviate and deletes the namespace weaviate in that context
  - deletes the kind cluster $CLUSTER; after that no kubectl context is current
  - when ENABLE_BACKUP, S3_OFFLOAD, USAGE_S3 or COLLECTION_EXPORT is true in your shell, deletes the MinIO manifests in that context
EOF
    local answer
    printf 'Type y to continue: '
    if ! read -r answer; then
        echo
        answer=""
    fi
    if [ "$answer" != y ]; then
        echo "local-k8s.sh clean was not run."
        return 0
    fi
    "$SCRIPT" clean
    rm -f "$MARKER"
    echo "Deleted the kind cluster $CLUSTER. Select your kubectl context again" \
        "with kubectl config use-context NAME."
}

case "${1:-}" in
up) up ;;
down) down ;;
*)
    echo "usage: $0 up|down" >&2
    exit 2
    ;;
esac
