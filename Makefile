LOCAL_K8S ?= ../weaviate-local-k8s
# cluster.sh reads LOCAL_K8S from the environment, so a path with a space
# needs no quoting here.
export LOCAL_K8S
CHROMA_COMPOSE := tests/acceptance/chroma/docker-compose.yml

.PHONY: help acceptance-up acceptance acceptance-down chroma-up chroma-down weaviate-up weaviate-down acceptance-cloud
.DEFAULT_GOAL := help

help:
	@echo "make acceptance-up    start Chroma and a Weaviate kind cluster for the acceptance suite"
	@echo "make acceptance       run the acceptance suite against them"
	@echo "make acceptance-down  stop Chroma; delete the kind cluster that acceptance-up created, after a prompt"
	@echo "make chroma-up        start the Chroma of the acceptance suite on port 8001"
	@echo "make chroma-down      stop that Chroma"
	@echo "make weaviate-up      start Weaviate and text2vec-transformers in Docker on ports 8099 and 50099 (ASYNC=0: no async indexing)"
	@echo "make weaviate-down    remove those containers and their network"
	@echo "make acceptance-cloud run the API-level tests against the Weaviate Cloud cluster at WEAVIATE_URL"

# tests/acceptance/cluster.sh holds the checks and the steps of both targets.
acceptance-up:
	bash tests/acceptance/cluster.sh up

# The suite deletes its collections on these servers, so a value from the
# shell does not replace these.
acceptance: export WEAVIATE_URL := http://localhost:8099
acceptance: export WEAVIATE_GRPC_URL := http://localhost:50099
acceptance: export WEAVIATE_API_KEY := admin-key
acceptance: export CHROMA_URL := http://localhost:8001
acceptance:
	pytest tests/acceptance -v

acceptance-down:
	bash tests/acceptance/cluster.sh down

chroma-up:
	docker compose -f $(CHROMA_COMPOSE) up -d --wait

chroma-down:
	docker compose -f $(CHROMA_COMPOSE) down

# tests/acceptance/weaviate-docker.sh reads ASYNC from the environment.
weaviate-up:
	sh tests/acceptance/weaviate-docker.sh up

weaviate-down:
	sh tests/acceptance/weaviate-docker.sh down

# WEAVIATE_URL and WEAVIATE_API_KEY come from the environment only. A
# variable on the make command line also reaches every child process in
# MAKEFLAGS. The recipe reads them as shell variables, so make never prints
# a value. The tool reads the empty WEAVIATE_GRPC_URL as unset.
acceptance-cloud: export ACCEPTANCE_TARGET := cloud
acceptance-cloud: export CHROMA_URL := http://localhost:8001
acceptance-cloud: export WEAVIATE_ASYNC_INDEXING := true
acceptance-cloud: export ACCEPTANCE_VECTORIZER := text2vec-weaviate
acceptance-cloud: export WEAVIATE_GRPC_URL :=
acceptance-cloud:
	@if [ -z "$$WEAVIATE_URL" ] || [ -z "$$WEAVIATE_API_KEY" ] \
		|| [ "$(origin WEAVIATE_URL)" = "command line" ] \
		|| [ "$(origin WEAVIATE_API_KEY)" = "command line" ]; then \
		echo "set WEAVIATE_URL and WEAVIATE_API_KEY in the environment (not on the make command line)" >&2; \
		exit 2; \
	fi
	pytest tests/acceptance -v -m "not infra"
