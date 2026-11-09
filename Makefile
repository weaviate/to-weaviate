LOCAL_K8S ?= ../weaviate-local-k8s
# cluster.sh reads LOCAL_K8S from the environment, so a path with a space
# needs no quoting here.
export LOCAL_K8S

.PHONY: help acceptance-up acceptance acceptance-down
.DEFAULT_GOAL := help

help:
	@echo "make acceptance-up    start Chroma and a Weaviate kind cluster for the acceptance suite"
	@echo "make acceptance       run the acceptance suite against them"
	@echo "make acceptance-down  stop Chroma; delete the kind cluster that acceptance-up created, after a prompt"

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
