"""Helpers and fixtures of the acceptance suite.

The suite runs the installed to-weaviate command against a real Chroma server
and a real Weaviate server. It starts neither of them. It runs only when the
environment names the test endpoints, because it deletes and recreates
collections on both servers.
"""

import json
import os
import subprocess
import sysconfig
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import weaviate
from chroma.seed import seed
from weaviate.collections.classes.config import CollectionConfig
from weaviate.collections.classes.internal import Object

from to_weaviate.settings import weaviate_settings
from to_weaviate.target.connect import connect

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
COMMAND = Path(sysconfig.get_path("scripts")) / "to-weaviate"
COMMAND_TIMEOUT = 600

_client: weaviate.WeaviateClient | None = None


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


@dataclass(frozen=True)
class Result:
    exit_code: int
    data: dict[str, Any]
    stderr: str


def run(args: list[str], workdir: Path, env: dict[str, str] | None = None) -> Result:
    """Run the installed to-weaviate with `args`, --json and --dir `workdir`.

    The working directory is `workdir`, so no `.env` of the repository is read.
    The environment is os.environ with CLEARED_VARIABLES empty and `env` on top.
    """
    cleared = dict.fromkeys(CLEARED_VARIABLES, "")
    completed = subprocess.run(
        [str(COMMAND), *args, "--json", "--dir", str(workdir)],
        cwd=workdir,
        env={**os.environ, **cleared, **(env or {})},
        capture_output=True,
        text=True,
        timeout=COMMAND_TIMEOUT,
        check=False,
    )
    try:
        data = json.loads(completed.stdout)
    except ValueError:
        data = None
    if not isinstance(data, dict):
        pytest.fail(
            f"to-weaviate {' '.join(args)} exited {completed.returncode} without "
            f"one JSON document on stdout.\nstdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    return Result(completed.returncode, data, completed.stderr)


def _weaviate() -> weaviate.WeaviateClient:
    global _client
    if _client is None:
        _client = connect(weaviate_settings(os.environ))
    return _client


def weaviate_count(name: str) -> int:
    """The number of objects in the target collection `name`."""
    result = _weaviate().collections.use(name).aggregate.over_all(total_count=True)
    assert result.total_count is not None
    return result.total_count


def weaviate_object(name: str) -> Object[Any, Any]:
    """One object of the target collection `name`."""
    objects = _weaviate().collections.use(name).query.fetch_objects(limit=1).objects
    assert objects, f"{name} has no objects"
    return objects[0]


def weaviate_config(name: str) -> CollectionConfig:
    """The configuration of the target collection `name`, read from the server."""
    return _weaviate().collections.use(name).config.get()


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
    if _client is not None:
        _client.close()


@pytest.fixture(scope="session")
def seeded(_test_endpoints: None) -> None:
    seed()


@pytest.fixture(autouse=True)
def _delete_suite_collections(_test_endpoints: None) -> None:
    _weaviate().collections.delete(SUITE_COLLECTIONS)
