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
from weaviate.classes.query import Filter
from weaviate.collections import Collection
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


def _collection(name: str, tenant: str | None = None) -> Collection[Any, Any]:
    collection = _weaviate().collections.use(name)
    return collection.with_tenant(tenant) if tenant else collection


def weaviate_count(name: str, tenant: str | None = None) -> int:
    """The number of objects in the target collection `name`, or in its `tenant`."""
    result = _collection(name, tenant).aggregate.over_all(total_count=True)
    assert result.total_count is not None
    return result.total_count


def weaviate_near_vector(
    name: str, vector: list[float], target_vector: str = "default", limit: int = 3
) -> list[str]:
    """The source ids of the objects nearest to `vector`, nearest first."""
    response = _collection(name).query.near_vector(
        near_vector=vector,
        target_vector=target_vector,
        limit=limit,
        return_properties=["source_id"],
    )
    return [str(o.properties["source_id"]) for o in response.objects]


def weaviate_near_text(
    name: str, text: str, target_vector: str = "default", limit: int = 3
) -> list[str]:
    """The source ids of the objects nearest to `text`, nearest first."""
    response = _collection(name).query.near_text(
        query=text,
        target_vector=target_vector,
        limit=limit,
        return_properties=["source_id"],
    )
    return [str(o.properties["source_id"]) for o in response.objects]


def weaviate_by_source_id(
    name: str, source_id: str, tenant: str | None = None
) -> Object[Any, Any] | None:
    """The object whose source_id is `source_id`, with its vectors, or None."""
    response = _collection(name, tenant).query.fetch_objects(
        filters=Filter.by_property("source_id").equal(source_id),
        include_vector=True,
        limit=2,
    )
    assert len(response.objects) < 2, f"{name} has two objects of {source_id}"
    return response.objects[0] if response.objects else None


def weaviate_object(name: str) -> Object[Any, Any]:
    """One object of the target collection `name`."""
    objects = _weaviate().collections.use(name).query.fetch_objects(limit=1).objects
    assert objects, f"{name} has no objects"
    return objects[0]


def weaviate_delete_object(name: str, uuid: Any) -> None:
    """Delete the object `uuid` of the target collection `name`."""
    assert _collection(name).data.delete_by_id(uuid), f"{name} has no {uuid}"


def weaviate_insert_object(name: str, obj: Object[Any, Any]) -> None:
    """Insert `obj`, an object read with its vectors, into `name` again."""
    properties = {k: v for k, v in obj.properties.items() if v is not None}
    _collection(name).data.insert(
        properties=properties, vector=obj.vector, uuid=obj.uuid
    )


def weaviate_update_object(name: str, uuid: Any, properties: dict[str, Any]) -> None:
    """Set `properties` of the object `uuid` of `name`; other properties stay."""
    _collection(name).data.update(uuid=uuid, properties=properties)


def weaviate_tenants(name: str) -> set[str]:
    """The tenant names of the target collection `name`."""
    return set(_weaviate().collections.use(name).tenants.get())


def weaviate_exists(name: str) -> bool:
    """Whether the target collection `name` exists."""
    return _weaviate().collections.exists(name)


def weaviate_compressed(name: str) -> list[bool]:
    """Whether each shard of the target collection `name` holds compressed vectors."""
    nodes = _weaviate().cluster.nodes(collection=name, output="verbose")
    return [shard.compressed for node in nodes for shard in node.shards or []]


def weaviate_config(name: str) -> CollectionConfig:
    """The configuration of the target collection `name`, read from the server."""
    return _weaviate().collections.use(name).config.get()


def delete_target_collection(name: str) -> None:
    """Delete the target collection `name` and no other. A missing one is no error."""
    _weaviate().collections.delete(name)
