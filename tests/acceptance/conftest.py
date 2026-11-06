"""Helpers and fixtures of the acceptance suite.

The suite runs the installed to-weaviate command against a real Chroma server
and a real Weaviate server. It starts neither of them.
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

from to_weaviate.settings import weaviate_settings
from to_weaviate.target.connect import connect

SUITE_VARIABLES = (
    "WEAVIATE_URL",
    "WEAVIATE_GRPC_URL",
    "WEAVIATE_API_KEY",
    "CHROMA_URL",
)
MISSING_VARIABLES = [name for name in SUITE_VARIABLES if not os.environ.get(name)]
# The target collections the suite creates. The suite touches no other one.
SUITE_COLLECTIONS = ["Docs_cosine", "Tiny_l2", "Docs_rq"]
COMMAND = Path(sysconfig.get_path("scripts")) / "to-weaviate"
COMMAND_TIMEOUT = 600

_client: weaviate.WeaviateClient | None = None


@dataclass(frozen=True)
class Result:
    exit_code: int
    data: dict[str, Any]
    stderr: str


def run(args: list[str], workdir: Path, env: dict[str, str] | None = None) -> Result:
    """Run the installed to-weaviate with `args`, --json and --dir `workdir`.

    The working directory is `workdir`, so no `.env` of the repository is read.
    The environment is os.environ with `env` on top.
    """
    completed = subprocess.run(
        [str(COMMAND), *args, "--json", "--dir", str(workdir)],
        cwd=workdir,
        env={**os.environ, **(env or {})},
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


def weaviate_config(name: str) -> CollectionConfig:
    """The configuration of the target collection `name`, read from the server."""
    return _weaviate().collections.use(name).config.get()


@pytest.fixture(scope="session", autouse=True)
def _servers() -> Iterator[None]:
    if MISSING_VARIABLES:
        pytest.skip(
            f"{', '.join(MISSING_VARIABLES)} not set; the acceptance suite needs "
            "a Chroma and a Weaviate server, see README.md"
        )
    yield
    if _client is not None:
        _client.close()


@pytest.fixture(scope="session")
def seeded() -> None:
    seed()


@pytest.fixture(autouse=True)
def _delete_suite_collections() -> None:
    _weaviate().collections.delete(SUITE_COLLECTIONS)
