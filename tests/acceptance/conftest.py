"""Helpers and fixtures of the acceptance suite.

The suite runs the installed to-weaviate command against a real Chroma server
and a real Weaviate server. It starts neither of them. It runs only when the
environment names the test endpoints, because it deletes and recreates
collections on both servers.
"""

import os
from collections.abc import Iterator
from pathlib import Path

import e2e
import pytest
from chroma.seed import seed
from e2e import (
    Result,
    close_weaviate,
    delete_target_collection,
    weaviate_config,
    weaviate_count,
    weaviate_object,
)

# test_migration.py imports these names from this module.
__all__ = ["Result", "run", "weaviate_config", "weaviate_count", "weaviate_object"]

TEST_ENDPOINTS = {
    "CHROMA_URL": "http://localhost:8001",
    "WEAVIATE_URL": "http://localhost:8099",
    "WEAVIATE_GRPC_URL": "http://localhost:50099",
    "WEAVIATE_API_KEY": "admin-key",
}
REQUIRE_VARIABLE = "TO_WEAVIATE_REQUIRE_ACCEPTANCE"
# The local Chroma needs none of these. A value from the shell would change
# the tenant, the database or the key header that the command uses.
CLEARED_VARIABLES = (
    "CHROMA_DATABASE",
    "CHROMA_TENANT",
    "CHROMA_API_KEY",
    "CHROMA_TOKEN_HEADER",
)
# The target collections the suite creates. The suite touches no other one.
SUITE_COLLECTIONS = ["Docs_cosine", "Tiny_l2", "Docs_rq"]


def endpoint_problems() -> list[str]:
    """Each suite variable that is missing or differs from the test endpoint.

    No value from the environment appears in the text.
    """
    problems = []
    for name, expected in TEST_ENDPOINTS.items():
        value = os.environ.get(name)
        if not value:
            problems.append(f"{name} is not set")
        elif value != expected:
            shown = "the test key" if name == "WEAVIATE_API_KEY" else expected
            problems.append(f"{name} is not {shown}")
    return problems


def run(args: list[str], workdir: Path, env: dict[str, str] | None = None) -> Result:
    """Run the installed to-weaviate with `args`, --json and --dir `workdir`.

    The working directory is `workdir`, so no `.env` of the repository is read.
    The environment is os.environ with CLEARED_VARIABLES empty and `env` on top.
    """
    cleared = dict.fromkeys(CLEARED_VARIABLES, "")
    return e2e.run(args, workdir, {**cleared, **(env or {})})


@pytest.fixture(scope="session", autouse=True)
def _test_endpoints() -> Iterator[None]:
    problems = endpoint_problems()
    if problems:
        reason = (
            "the acceptance suite runs only against the test endpoints: "
            + "; ".join(problems)
            + ". See README.md."
        )
        # CI sets it, so that a suite that skips every test does not pass.
        if os.environ.get(REQUIRE_VARIABLE) == "1":
            pytest.fail(reason)
        pytest.skip(reason)
    yield
    close_weaviate()


@pytest.fixture(scope="session")
def seeded(_test_endpoints: None) -> None:
    seed()


@pytest.fixture(autouse=True)
def _delete_suite_collections(_test_endpoints: None) -> None:
    for name in SUITE_COLLECTIONS:
        delete_target_collection(name)
