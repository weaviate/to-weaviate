#!/bin/sh
# Replaces the Weaviate port-forwards of the weaviate-local-k8s action with
# kubectl port-forward processes that a loop starts again when they exit.
# Runs only in the acceptance job of .github/workflows/ci.yml, right after the
# action.
#
# The action forwards ports 8099 and 50099 through kubectl-relay. After a test
# kills a copy with SIGKILL, kubectl-relay can stop forwarding gRPC while HTTP
# still works, and every later command exits 4. A kubectl port-forward to a
# service stays on one pod, so it exits when that pod goes away; the loop then
# starts a new one.
#
# The ports are the action's weaviate-port and weaviate-grpc-port inputs in
# ci.yml. The key is WEAVIATE_API_KEY of the job.
set -eu

HTTP_PORT=8099
GRPC_PORT=50099
KEY="${WEAVIATE_API_KEY:-admin-key}"
HTTP_LOG=/tmp/port-forward-http.log
GRPC_LOG=/tmp/port-forward-grpc.log
READY_URL="http://localhost:$HTTP_PORT/v1/.well-known/ready"
HEALTH_URL="http://localhost:$GRPC_PORT/grpc.health.v1.Health/Check"
READY_TRIES=30
READY_INTERVAL=2
STOP_TRIES=10

# The script stops processes and binds ports of the machine it runs on, so the
# check runs before any other command.
if [ "${GITHUB_ACTIONS:-}" != "true" ]; then
    echo "GITHUB_ACTIONS is not true: port-forward.sh runs only in CI. Nothing was stopped or started." >&2
    exit 2
fi

# stop_relay PATTERN: stops the kubectl-relay process whose command line
# holds PATTERN and waits until it is gone.
stop_relay() {
    pkill -f "$1" || true
    tries=0
    while pgrep -f "$1" >/dev/null; do
        tries=$((tries + 1))
        if [ "$tries" -ge "$STOP_TRIES" ]; then
            echo "A process that matches '$1' still runs after $STOP_TRIES s." >&2
            exit 1
        fi
        sleep 1
    done
}

# forward LOCAL_PORT SERVICE REMOTE_PORT LOG: runs until the job ends.
forward() {
    while :; do
        echo "$(date +%T) starting kubectl port-forward svc/$2 $1:$3" >>"$4"
        if kubectl port-forward -n weaviate "svc/$2" "$1:$3" >>"$4" 2>&1; then
            status=0
        else
            status=$?
        fi
        echo "$(date +%T) kubectl port-forward exited with $status" >>"$4"
        sleep 1
    done
}

http_ready() {
    curl -fsS --max-time 5 -o /dev/null -H "Authorization: Bearer $KEY" "$READY_URL" 2>/dev/null
}

# The request is an empty grpc.health.v1.HealthCheckRequest: a gRPC frame of
# five zero bytes (not compressed, length 0). The answer SERVING is the frame
# 00 00 00 00 02 08 01: not compressed, length 2, field 1 (status) = 1.
grpc_serving() {
    answer=$(printf '\000\000\000\000\000' |
        curl -sS --max-time 5 --http2-prior-knowledge \
            -H 'content-type: application/grpc' -H 'te: trailers' \
            --data-binary @- "$HEALTH_URL" 2>/dev/null |
        od -A n -t x1 | tr -d ' \n')
    [ "$answer" = 00000000020801 ]
}

stop_relay "kubectl-relay svc/weaviate -n weaviate $HTTP_PORT:80"
stop_relay "kubectl-relay svc/weaviate-grpc -n weaviate $GRPC_PORT:50051"

# The loops keep no descriptor of this step open, so the step ends while they
# run.
forward "$HTTP_PORT" weaviate 80 "$HTTP_LOG" </dev/null >/dev/null 2>&1 &
forward "$GRPC_PORT" weaviate-grpc 50051 "$GRPC_LOG" </dev/null >/dev/null 2>&1 &

tries=0
until http_ready && grpc_serving; do
    tries=$((tries + 1))
    if [ "$tries" -ge "$READY_TRIES" ]; then
        echo "Weaviate does not answer through ports $HTTP_PORT and $GRPC_PORT after $READY_TRIES tries, $READY_INTERVAL s apart." >&2
        tail -n 20 "$HTTP_LOG" "$GRPC_LOG" >&2 || true
        exit 1
    fi
    sleep "$READY_INTERVAL"
done
echo "Weaviate answers on ports $HTTP_PORT (HTTP ready) and $GRPC_PORT (gRPC SERVING) through kubectl port-forward. Logs: $HTTP_LOG, $GRPC_LOG."
