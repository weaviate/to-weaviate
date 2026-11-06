LOCAL_K8S ?= ../weaviate-local-k8s
WEAVIATE_VERSION := $(shell cat tests/acceptance/weaviate-version)
CHROMA_COMPOSE := tests/acceptance/chroma/docker-compose.yml
KIND_CLUSTER := weaviate-k8s

.PHONY: acceptance-up acceptance acceptance-down

# local-k8s.sh setup does not check for an existing cluster of the same name.
# Its default OBSERVABILITY=true would take ports 3000 and 9091.
acceptance-up:
	@if kind get clusters 2>/dev/null | grep -qx '$(KIND_CLUSTER)'; then \
		echo "The kind cluster $(KIND_CLUSTER) exists. local-k8s.sh setup would reuse its name. Nothing was started."; \
		exit 1; \
	fi
	docker compose -f $(CHROMA_COMPOSE) up -d --wait
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

acceptance-down:
	docker compose -f $(CHROMA_COMPOSE) down
	@echo "local-k8s.sh clean deletes the kind cluster $(KIND_CLUSTER) and kills every kubectl-relay process on this machine."
	@printf 'Type y to continue: '; read answer; \
	if [ "$$answer" = y ]; then \
		$(LOCAL_K8S)/local-k8s.sh clean; \
	else \
		echo "local-k8s.sh clean was not run."; \
	fi
