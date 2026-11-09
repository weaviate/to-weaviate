"""Code that the acceptance suite and the cloud suite share.

Both suites run the installed to-weaviate command and read the target through
the Weaviate client. Each suite decides which servers it accepts and which
environment the command gets.
"""

import json
import os
import subprocess
import sysconfig
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import weaviate
from weaviate.collections.classes.config import CollectionConfig
from weaviate.collections.classes.internal import Object

from to_weaviate.settings import weaviate_settings
from to_weaviate.target.connect import connect

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


def close_weaviate() -> None:
    """Close the client of the helpers below, when one is open."""
    global _client
    if _client is not None:
        _client.close()
        _client = None


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


def delete_target_collection(name: str) -> None:
    """Delete the target collection `name` and no other. A missing one is no error."""
    _weaviate().collections.delete(name)
