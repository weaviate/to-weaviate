#!/bin/sh
# Starts and stops the open-source distributed Chroma of the cluster leg of
# .github/workflows/ci.yml, in namespace chroma of the kind cluster that the
# current kubectl context names. Its HTTP API is forwarded to localhost:8001,
# the CHROMA_URL of the acceptance suite. Runs only on Linux amd64, in that
# leg after the weaviate-local-k8s action: the service images exist for amd64
# only.
#
# Usage: chroma-k8s.sh up|down. up renders the Helm chart
# k8s/distributed-chroma of the Chroma repository with helm template and
# applies it with Postgres and MinIO from k8s/test, as Chroma's Tiltfile
# does. down stops the forward and deletes the namespace.
set -eu

# The pins of Chroma release 1.5.9. The tag holds the chart and the config
# files. The service images on Docker Hub carry the short commit of the tag.
# The release's MinIO image, minio/minio:latest, is gone from Docker Hub, so
# MinIO is the digest that Chroma's CI pins after the release.
CHROMA_TAG=1.5.9
CHROMA_COMMIT=11f3c74
MINIO_IMAGE=ghcr.io/chroma-core/ci-minio@sha256:14cea493d9a34af32f524e538b8346cf79f3321eff8e708c1e2960462bd8936e
# Built from k8s/test/postgres of the tag. A tag other than latest keeps the
# pull policy IfNotPresent, so the nodes use the loaded image.
POSTGRES_IMAGE=chroma-postgres:15
REPOSITORY=https://github.com/chroma-core/chroma.git
NAMESPACE=chroma
PORT=8001
HEALTH_URL="http://localhost:$PORT/api/v2/healthcheck"
FORWARD_LOG=/tmp/port-forward-chroma.log
FORWARD_PID=/tmp/port-forward-chroma.pid
FORWARD_PATTERN="kubectl port-forward -n $NAMESPACE svc/rust-frontend-service"
# The chart's dev values limit each query node to 100m CPU, which slows every
# read of the suite.
QUERY_CPU_LIMIT=1000m
QUERY_CPU_REQUEST=250m
WAIT_TIMEOUT=600
PULL_GRACE=60
POLL_INTERVAL=5
READY_TRIES=30
READY_INTERVAL=2
STOP_TRIES=10

# The images run on amd64 only, so the check runs before any other command.
case "$(uname -m)" in
x86_64 | amd64) ;;
*)
    echo "the distributed Chroma images are amd64 only; this leg runs in CI. Nothing was started or deleted." >&2
    exit 2
    ;;
esac
if [ "$(uname -s)" != Linux ]; then
    echo "chroma-k8s.sh runs on Linux only; this leg runs in CI. Nothing was started or deleted." >&2
    exit 2
fi

usage() {
    echo "usage: chroma-k8s.sh up|down" >&2
    exit 2
}

require() {
    for tool in "$@"; do
        if ! command -v "$tool" >/dev/null 2>&1; then
            echo "chroma-k8s.sh needs $tool on PATH. Nothing was started or deleted." >&2
            exit 1
        fi
    done
}

# Sets CLUSTER to the kind cluster of the current kubectl context. Any other
# context stops the script, so it deploys into and deletes from a kind
# cluster only.
kind_cluster() {
    context=$(kubectl config current-context 2>/dev/null) || context=
    case "$context" in
    kind-?*) CLUSTER=${context#kind-} ;;
    *)
        echo "The current kubectl context '$context' is not a kind context. Nothing was started or deleted." >&2
        exit 1
        ;;
    esac
    if ! kind get clusters 2>/dev/null | grep -qx "$CLUSTER"; then
        echo "kind get clusters does not list $CLUSTER, the cluster of the context $context. Nothing was started or deleted." >&2
        exit 1
    fi
}

# pull_failures: the image, reason and message of each container of the
# namespace whose image pull fails.
pull_failures() {
    kubectl -n "$NAMESPACE" get pods -o jsonpath='{range .items[*]}{range .status.containerStatuses[*]}{.image}{" "}{.state.waiting.reason}{" "}{.state.waiting.message}{"\n"}{end}{end}' 2>/dev/null |
        grep -E '^[^ ]+ (ErrImagePull|ImagePullBackOff|InvalidImageName) ' || true
}

# diagnose [RESOURCE]: prints the pods and the last events of the namespace,
# and the last log lines of RESOURCE.
diagnose() {
    {
        echo "--- kubectl -n $NAMESPACE get pods"
        kubectl -n "$NAMESPACE" get pods -o wide || true
        echo "--- the last events of namespace $NAMESPACE"
        kubectl -n "$NAMESPACE" get events --sort-by=.lastTimestamp 2>&1 | tail -n 40 || true
        if [ $# -gt 0 ]; then
            echo "--- kubectl -n $NAMESPACE logs $1"
            kubectl -n "$NAMESPACE" logs "$1" --all-containers --tail=50 || true
        fi
    } >&2
}

# wait_until WHAT RESOURCE COMMAND...: runs COMMAND until it succeeds. Stops
# after WAIT_TIMEOUT s, or when an image pull of the namespace has failed for
# PULL_GRACE s.
wait_until() {
    what=$1
    resource=$2
    shift 2
    start=$(date +%s)
    failing_since=
    until "$@" >/dev/null 2>&1; do
        now=$(date +%s)
        failures=$(pull_failures)
        if [ -z "$failures" ]; then
            failing_since=
        elif [ -z "$failing_since" ]; then
            failing_since=$now
        elif [ $((now - failing_since)) -ge "$PULL_GRACE" ]; then
            echo "$resource is not $what: an image pull in namespace $NAMESPACE has failed for $PULL_GRACE s." >&2
            echo "$failures" >&2
            diagnose "$resource"
            exit 1
        fi
        if [ $((now - start)) -ge "$WAIT_TIMEOUT" ]; then
            echo "$resource is not $what after $WAIT_TIMEOUT s." >&2
            diagnose "$resource"
            exit 1
        fi
        sleep "$POLL_INTERVAL"
    done
    echo "$resource is $what after $(($(date +%s) - start)) s."
}

rolled_out() {
    wait_until "rolled out" "$1" kubectl -n "$NAMESPACE" rollout status "$1" --timeout=10s
}

# check_images FILE...: stops unless every image of FILE is a pin of this
# script.
check_images() {
    images=$(sed -n 's/^[[:space:]-]*image:[[:space:]]*//p' "$@" | tr -d "\"'" | sort -u | paste -s -d ' ' -)
    for image in $images; do
        case "$image" in
        chromadb/*:"$CHROMA_COMMIT" | "$POSTGRES_IMAGE" | "$MINIO_IMAGE") ;;
        *)
            echo "The manifests name the image $image, which is not a pin of chroma-k8s.sh. Nothing was applied." >&2
            exit 1
            ;;
        esac
    done
    echo "Images: $images"
}

# render DIR: clones the release into DIR/chroma and writes its manifests to
# DIR.
render() {
    src=$1/chroma
    git clone --quiet --depth 1 --branch "$CHROMA_TAG" "$REPOSITORY" "$src"
    commit=$(git -C "$src" rev-parse HEAD)
    case "$commit" in
    "$CHROMA_COMMIT"*) ;;
    *)
        echo "The tag $CHROMA_TAG of $REPOSITORY is the commit $commit, not $CHROMA_COMMIT. Nothing was applied." >&2
        exit 1
        ;;
    esac
    chart=$src/k8s/distributed-chroma
    worker=$src/rust/worker/chroma_config.yaml
    helm template chroma "$chart" \
        -f "$chart/values.yaml" -f "$chart/values.dev.yaml" \
        --set-file rustFrontendService.configuration="$src/rust/frontend/sample_configs/distributed.yaml" \
        --set-file queryService.configuration="$worker" \
        --set-file compactionService.configuration="$worker" \
        --set-file rustLogService.configuration="$worker" \
        --set-file garbageCollector.configuration="$worker" \
        --set rustSysdbService.replicaCount=0 \
        --set queryService.resources.limits.cpu="$QUERY_CPU_LIMIT" \
        --set queryService.resources.requests.cpu="$QUERY_CPU_REQUEST" \
        --set-string rustFrontendService.image.repository=chromadb/rust-frontend-service-oss \
        --set-string rustFrontendService.image.tag="$CHROMA_COMMIT" \
        --set-string queryService.image.repository=chromadb/query-service \
        --set-string queryService.image.tag="$CHROMA_COMMIT" \
        --set-string compactionService.image.repository=chromadb/compactor-service \
        --set-string compactionService.image.tag="$CHROMA_COMMIT" \
        --set-string rustLogService.image.repository=chromadb/rust-log-service \
        --set-string rustLogService.image.tag="$CHROMA_COMMIT" \
        --set-string sysdb.image.repository=chromadb/sysdb-service \
        --set-string sysdb.image.tag="$CHROMA_COMMIT" \
        --set-string sysdbMigration.image.repository=chromadb/sysdb-migration \
        --set-string sysdbMigration.image.tag="$CHROMA_COMMIT" \
        --set-string garbageCollector.image.repository=chromadb/garbage-collector-service \
        --set-string garbageCollector.image.tag="$CHROMA_COMMIT" \
        >"$1/chroma.yaml"
    sed "s#image: chroma-postgres\$#image: $POSTGRES_IMAGE#" \
        "$src/k8s/test/postgres.yaml" >"$1/postgres.yaml"
    sed "s#image: minio/minio:latest\$#image: $MINIO_IMAGE#" \
        "$src/k8s/test/minio.yaml" >"$1/minio.yaml"
    check_images "$1/chroma.yaml" "$1/postgres.yaml" "$1/minio.yaml"
}

# answers: whether any server answers on PORT.
answers() {
    if curl -s --max-time 2 -o /dev/null "http://localhost:$PORT/"; then
        return 0
    else
        status=$?
    fi
    # curl exits 7 when it cannot connect.
    [ "$status" -ne 7 ]
}

# forward: runs until the job ends. A kubectl port-forward to a service stays
# on one pod and exits when that pod goes away; the loop then starts a new
# one.
forward() {
    while :; do
        echo "$(date +%T) starting $FORWARD_PATTERN $PORT:8000" >>"$FORWARD_LOG"
        if kubectl port-forward -n "$NAMESPACE" svc/rust-frontend-service "$PORT:8000" >>"$FORWARD_LOG" 2>&1; then
            status=0
        else
            status=$?
        fi
        echo "$(date +%T) kubectl port-forward exited with $status" >>"$FORWARD_LOG"
        sleep 1
    done
}

up() {
    require git docker kind kubectl helm curl
    kind_cluster
    if kubectl get namespace "$NAMESPACE" >/dev/null 2>&1; then
        echo "The namespace $NAMESPACE exists in the kind cluster $CLUSTER; run chroma-k8s.sh down first. Nothing was started." >&2
        exit 1
    fi
    if answers; then
        echo "A server answers on port $PORT, so the forward cannot listen there. Nothing was started." >&2
        exit 1
    fi
    began=$(date +%s)
    work=$(mktemp -d)
    trap 'rm -rf "$work"' EXIT
    echo "Rendering the manifests of Chroma $CHROMA_TAG ($CHROMA_COMMIT)."
    render "$work"
    docker build --quiet -t "$POSTGRES_IMAGE" "$work/chroma/k8s/test/postgres"
    kind load docker-image "$POSTGRES_IMAGE" --name "$CLUSTER"
    kubectl apply -f "$work/chroma/k8s/distributed-chroma/crds/memberlist_crd.yaml"
    kubectl wait --for=condition=established crd/memberlists.chroma.cluster --timeout=60s
    kubectl create namespace "$NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -
    # sysdb, its migration and the log service fail until these answer.
    kubectl apply -f "$work/postgres.yaml" -f "$work/minio.yaml"
    rolled_out deploy/postgres
    rolled_out deploy/minio-deployment
    kubectl apply -f "$work/chroma.yaml"
    wait_until complete "job/sysdb-migration-$CHROMA_COMMIT" \
        kubectl -n "$NAMESPACE" wait --for=condition=complete "job/sysdb-migration-$CHROMA_COMMIT" --timeout=10s
    for resource in deploy/sysdb sts/rust-log-service sts/query-service \
        sts/compaction-service sts/garbage-collector deploy/rust-frontend-service; do
        rolled_out "$resource"
    done
    rm -rf "$work"
    trap - EXIT

    : >"$FORWARD_LOG"
    # The loop keeps no descriptor of this step open, so the step ends while
    # it runs.
    forward </dev/null >/dev/null 2>&1 &
    echo $! >"$FORWARD_PID"
    tries=0
    until curl -fsS --max-time 5 -o /dev/null "$HEALTH_URL" 2>/dev/null; do
        tries=$((tries + 1))
        if [ "$tries" -ge "$READY_TRIES" ]; then
            echo "Distributed Chroma does not answer $HEALTH_URL after $READY_TRIES tries, $READY_INTERVAL s apart." >&2
            tail -n 20 "$FORWARD_LOG" >&2 || true
            diagnose deploy/rust-frontend-service
            exit 1
        fi
        sleep "$READY_INTERVAL"
    done
    echo "Distributed Chroma $CHROMA_TAG answers on port $PORT through kubectl port-forward, $(($(date +%s) - began)) s after the start. Log: $FORWARD_LOG."
}

down() {
    require kubectl kind pkill pgrep
    kind_cluster
    if [ -f "$FORWARD_PID" ]; then
        kill "$(cat "$FORWARD_PID")" 2>/dev/null || true
        rm -f "$FORWARD_PID"
    fi
    pkill -f "$FORWARD_PATTERN" || true
    tries=0
    while pgrep -f "$FORWARD_PATTERN" >/dev/null; do
        tries=$((tries + 1))
        if [ "$tries" -ge "$STOP_TRIES" ]; then
            echo "A process that matches '$FORWARD_PATTERN' still runs after $STOP_TRIES s." >&2
            exit 1
        fi
        sleep 1
    done
    kubectl delete namespace "$NAMESPACE" --ignore-not-found --timeout=180s
    echo "The forward on port $PORT is stopped and the namespace $NAMESPACE of the kind cluster $CLUSTER is deleted."
}

[ $# -eq 1 ] || usage
case "$1" in
up) up ;;
down) down ;;
*) usage ;;
esac
