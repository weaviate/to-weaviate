LOCAL_K8S ?= ../weaviate-local-k8s
WEAVIATE_VERSION := $(shell cat tests/acceptance/weaviate-version)
CHROMA_COMPOSE := tests/acceptance/chroma/docker-compose.yml
KIND_CLUSTER := weaviate-k8s
# acceptance-up writes this file before local-k8s.sh setup runs.
# acceptance-down deletes the cluster only when the file exists.
CLUSTER_MARKER := .acceptance-kind-cluster

.PHONY: help acceptance-up acceptance acceptance-down
.DEFAULT_GOAL := help

help:
	@echo "make acceptance-up    start Chroma and a Weaviate kind cluster for the acceptance suite"
	@echo "make acceptance       run the acceptance suite against them"
	@echo "make acceptance-down  stop Chroma; delete the kind cluster that acceptance-up created, after a prompt"

# local-k8s.sh setup does not check for an existing cluster of the same name.
# Its default OBSERVABILITY=true would take ports 3000 and 9091.
acceptance-up:
	@command -v kind >/dev/null || { \
		echo "kind is not on PATH. Install kind, then run make acceptance-up again."; \
		exit 1; \
	}
	@test -x '$(LOCAL_K8S)/local-k8s.sh' || { \
		echo "$(LOCAL_K8S)/local-k8s.sh does not exist or is not executable. Set LOCAL_K8S to a clone of weaviate-local-k8s."; \
		exit 1; \
	}
	@if kind get clusters 2>/dev/null | grep -qx '$(KIND_CLUSTER)'; then \
		echo "The kind cluster $(KIND_CLUSTER) exists. local-k8s.sh setup would reuse its name. Nothing was started."; \
		exit 1; \
	fi
	docker compose -f $(CHROMA_COMPOSE) up -d --wait
	touch $(CLUSTER_MARKER)
	WEAVIATE_VERSION=$(WEAVIATE_VERSION) WORKERS=1 REPLICAS=1 \
		WEAVIATE_PORT=8099 WEAVIATE_GRPC_PORT=50099 RBAC=true \
		MODULES=text2vec-transformers EXPOSE_PODS=false OBSERVABILITY=false \
		$(LOCAL_K8S)/local-k8s.sh setup

# The suite deletes its collections on these servers, so a value from the
# shell does not replace these.
acceptance: export WEAVIATE_URL := http://localhost:8099
acceptance: export WEAVIATE_GRPC_URL := http://localhost:50099
acceptance: export WEAVIATE_API_KEY := admin-key
acceptance: export CHROMA_URL := http://localhost:8001
acceptance:
	pytest tests/acceptance -v

# The list of effects follows clean() in local-k8s.sh.
acceptance-down:
	docker compose -f $(CHROMA_COMPOSE) down
	@if [ ! -f $(CLUSTER_MARKER) ]; then \
		echo "This checkout did not create the kind cluster $(KIND_CLUSTER) ($(CLUSTER_MARKER) does not exist). The cluster was left alone."; \
		exit 0; \
	fi; \
	echo "local-k8s.sh clean does all of this:"; \
	echo "  - kills every process on this machine whose command line contains kubectl-relay"; \
	echo "  - kills every process on this machine whose command line contains 'kubectl get pods -n weaviate -w'"; \
	echo "  - removes /tmp/weaviate_pod_states.log and /tmp/weaviate_pod_states.jq"; \
	echo "  - switches the kubectl context to kind-$(KIND_CLUSTER)"; \
	echo "  - uninstalls the Helm release weaviate and deletes the namespace weaviate in that context"; \
	echo "  - deletes the kind cluster $(KIND_CLUSTER)"; \
	echo "  - when ENABLE_BACKUP, S3_OFFLOAD, USAGE_S3 or COLLECTION_EXPORT is true in your shell, deletes the MinIO manifests in that context"; \
	printf 'Type y to continue: '; read answer; \
	if [ "$$answer" = y ]; then \
		$(LOCAL_K8S)/local-k8s.sh clean && rm -f $(CLUSTER_MARKER); \
	else \
		echo "local-k8s.sh clean was not run."; \
	fi
