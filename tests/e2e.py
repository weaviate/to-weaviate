"""Code that the acceptance suite and the cloud suite share.

Both suites run the installed to-weaviate command and read the target through
the Weaviate client. Each suite decides which servers it accepts and which
environment the command gets. The target rules of the acceptance suite are
here, so that the unit tests can import them.
"""

import json
import os
import subprocess
import sysconfig
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import pytest
import weaviate
from weaviate.classes.query import Filter
from weaviate.classes.tenants import Tenant, TenantActivityStatus
from weaviate.collections import Collection
from weaviate.collections.classes.config import CollectionConfig
from weaviate.collections.classes.internal import Object

from to_weaviate.output import ERROR_EXIT, secret_values
from to_weaviate.settings import weaviate_settings
from to_weaviate.target.connect import connect
from to_weaviate.target.schema import module_names

COMMAND = Path(sysconfig.get_path("scripts")) / "to-weaviate"
COMMAND_TIMEOUT = 600
MAX_MESSAGE_CHARS = 500
ERROR_KEYS = {"status", "code", "message", "hint"}

# "local" or "cloud": the servers that the acceptance suite runs against.
TARGET_VARIABLE = "ACCEPTANCE_TARGET"
# The module that the tests of the embed and both modes plan.
VECTORIZER_VARIABLE = "ACCEPTANCE_VECTORIZER"
DEFAULT_VECTORIZER = "text2vec-transformers"
# The number of Weaviate nodes of the target.
NODES_VARIABLE = "ACCEPTANCE_NODES"
LOCAL_ENDPOINTS = {
    "CHROMA_URL": "http://localhost:8001",
    "WEAVIATE_URL": "http://localhost:8099",
    "WEAVIATE_GRPC_URL": "http://localhost:50099",
    "WEAVIATE_API_KEY": "admin-key",
}
CLOUD_HOST_SUFFIXES = (".weaviate.cloud", ".weaviate.network")

_client: weaviate.WeaviateClient | None = None
# The secret values of the environments that `run` and `start` gave a command.
_secrets: set[str] = set()


def take_secrets() -> set[str]:
    """The secret values given to commands since the last call, then forget them."""
    found = set(_secrets)
    _secrets.clear()
    return found


def _environment(env: dict[str, str] | None) -> dict[str, str]:
    merged = {**os.environ, **(env or {})}
    _secrets.update(secret_values(merged))
    return merged


def _contract_problems(exit_code: int, data: dict[str, Any], stderr: str) -> list[str]:
    """How the JSON document, the exit code and stderr break the CLI contract."""
    problems = []
    if data.get("status") == "error":
        missing = ERROR_KEYS - set(data)
        if missing:
            problems.append(f"the error document has no {', '.join(sorted(missing))}")
        code = data.get("code")
        if code not in ERROR_EXIT:
            problems.append(f"the error code {code!r} is not a known code")
        elif exit_code != ERROR_EXIT[code]:
            problems.append(f"exit code {exit_code} for the error code {code}")
        message = str(data.get("message"))
        if "\n" in message or len(message) > MAX_MESSAGE_CHARS:
            problems.append("the message is not one line of at most 500 characters")
    elif data.get("status") == "success":
        if exit_code != 0:
            problems.append(f"exit code {exit_code} for a success document")
    else:
        problems.append("the document has no status success or error")
    if '"status"' in stderr:
        problems.append("stderr holds a JSON document")
    return problems


@dataclass(frozen=True)
class Result:
    exit_code: int
    data: dict[str, Any]
    stderr: str


def run(args: list[str], workdir: Path, env: dict[str, str] | None = None) -> Result:
    """Run the installed to-weaviate with `args`, --json and --dir `workdir`.

    The working directory is `workdir`, so no `.env` of the repository is read.
    The environment is os.environ with `env` on top. The run fails the test
    when stdout or stderr holds the value of a secret variable, or when the
    document, the exit code and stderr break the CLI contract.
    """
    merged = _environment(env)
    completed = subprocess.run(
        [str(COMMAND), *args, "--json", "--dir", str(workdir)],
        cwd=workdir,
        env=merged,
        capture_output=True,
        text=True,
        timeout=COMMAND_TIMEOUT,
        check=False,
    )
    output = completed.stdout + completed.stderr
    if any(secret in output for secret in secret_values(merged)):
        # The value is not printed here either.
        pytest.fail(f"to-weaviate {' '.join(args)} printed a secret value")
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
    problems = _contract_problems(completed.returncode, data, completed.stderr)
    if problems:
        pytest.fail(
            f"to-weaviate {' '.join(args)}: {'; '.join(problems)}.\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    return Result(completed.returncode, data, completed.stderr)


def run_text(
    args: list[str], workdir: Path, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    """Run the installed to-weaviate like `run` does, without --json.

    The run fails the test when stdout or stderr holds the value of a secret
    variable.
    """
    merged = _environment(env)
    completed = subprocess.run(
        [str(COMMAND), *args, "--dir", str(workdir)],
        cwd=workdir,
        env=merged,
        capture_output=True,
        text=True,
        timeout=COMMAND_TIMEOUT,
        check=False,
    )
    if any(s in completed.stdout + completed.stderr for s in secret_values(merged)):
        pytest.fail(f"to-weaviate {' '.join(args)} printed a secret value")
    return completed


def start(
    args: list[str], workdir: Path, stdout: Path, env: dict[str, str] | None = None
) -> subprocess.Popen[bytes]:
    """Start the installed to-weaviate like `run` does, without waiting for it.

    Its stdout goes to the file `stdout`; its stderr is discarded.
    """
    with stdout.open("wb") as handle:
        return subprocess.Popen(
            [str(COMMAND), *args, "--json", "--dir", str(workdir)],
            cwd=workdir,
            env=_environment(env),
            stdout=handle,
            stderr=subprocess.DEVNULL,
        )


def _variable(name: str) -> str:
    return os.environ.get(name, "").strip()


def target() -> str:
    """The value of TARGET_VARIABLE in lower case; unset or empty is "local".

    endpoint_problems reports a value other than "local" and "cloud".
    """
    return _variable(TARGET_VARIABLE).lower() or "local"


def vectorizer() -> str:
    """The module that VECTORIZER_VARIABLE names.

    Unset or empty is DEFAULT_VECTORIZER, the module of the local target.
    """
    return _variable(VECTORIZER_VARIABLE) or DEFAULT_VECTORIZER


def nodes() -> int:
    """The node count that NODES_VARIABLE names; unset or empty is 1.

    A value that is not a whole number of at least 1 raises ValueError.
    """
    value = _variable(NODES_VARIABLE)
    if not value:
        return 1
    if not value.isdecimal() or int(value) < 1:
        raise ValueError(f"{NODES_VARIABLE} is not a whole number of at least 1")
    return int(value)


def endpoint_problems() -> list[str]:
    """Each suite variable that breaks a rule of the target that target() names.

    The local target needs the four LOCAL_ENDPOINTS. The cloud target needs a
    Weaviate Cloud URL, a key, no WEAVIATE_GRPC_URL and the local Chroma. No
    value from the environment appears in the text.
    """
    if target() == "local":
        return _local_problems()
    if target() == "cloud":
        return _cloud_problems()
    return [f"{TARGET_VARIABLE} is neither local nor cloud"]


def _local_problems() -> list[str]:
    problems = []
    for name, expected in LOCAL_ENDPOINTS.items():
        value = os.environ.get(name)
        if not value:
            problems.append(f"{name} is not set")
        elif value != expected:
            shown = "the test key" if name == "WEAVIATE_API_KEY" else expected
            problems.append(f"{name} is not {shown}")
    return problems


def _cloud_problems() -> list[str]:
    problems = []
    url = _variable("WEAVIATE_URL")
    if not url:
        problems.append("WEAVIATE_URL is not set")
    else:
        problems.extend(_cloud_url_problems(url))
    if not _variable("WEAVIATE_API_KEY"):
        problems.append("WEAVIATE_API_KEY is not set")
    # The tool reads an empty value as unset.
    if _variable("WEAVIATE_GRPC_URL"):
        problems.append(
            "WEAVIATE_GRPC_URL is set (the cloud target connects without it)"
        )
    chroma = os.environ.get("CHROMA_URL")
    local_chroma = LOCAL_ENDPOINTS["CHROMA_URL"]
    if not chroma:
        problems.append("CHROMA_URL is not set")
    elif chroma != local_chroma:
        problems.append(
            f"CHROMA_URL is not {local_chroma} (the source stays the local test Chroma)"
        )
    return problems


def _cloud_url_problems(url: str) -> list[str]:
    try:
        parts = urlsplit(url)
        host = parts.hostname or ""
    except ValueError:
        return ["WEAVIATE_URL is not a URL"]
    problems = []
    if parts.scheme != "https":
        problems.append("WEAVIATE_URL does not use https (the cloud target needs TLS)")
    if not host.endswith(CLOUD_HOST_SUFFIXES):
        problems.append(
            "the host of WEAVIATE_URL does not end with "
            + " or ".join(CLOUD_HOST_SUFFIXES)
        )
    return problems


def fail_infra_on_cloud(item: pytest.Item) -> None:
    """Fail `item` when it is marked infra and target() is "cloud".

    An infra test runs docker or kubectl, which reach only the local servers.
    The acceptance conftest calls it from pytest_runtest_setup, so it runs
    before any fixture of the item is set up.
    """
    if item.get_closest_marker("infra") is not None and target() == "cloud":
        pytest.fail("infra tests run only against the local servers")


def dedicated_cluster_problem(
    listed: Iterable[str], owned: Iterable[str]
) -> str | None:
    """The failure text when `listed` holds a collection outside `owned`, else None.

    The suite deletes every collection that `owned` names. A target with other
    collections is shared, and a collection of such a name on it can hold the
    data of someone else.
    """
    foreign = sorted(set(listed) - set(owned))
    if not foreign:
        return None
    return (
        "the target holds collections the suite does not own: "
        f"{', '.join(foreign)}; use a dedicated cluster"
    )


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


def weaviate_set_shard_status(name: str, status: str) -> None:
    """Set every shard of the target collection `name` to READY or READONLY."""
    _weaviate().collections.use(name).config.update_shards(status=status)  # type: ignore[arg-type]


def weaviate_set_tenant_status(name: str, tenant: str, status: str) -> None:
    """Set `tenant` of the target collection `name` to ACTIVE or INACTIVE."""
    _weaviate().collections.use(name).tenants.update(
        Tenant(name=tenant, activity_status=TenantActivityStatus(status))
    )


def weaviate_tenants(name: str) -> set[str]:
    """The tenant names of the target collection `name`."""
    return set(_weaviate().collections.use(name).tenants.get())


def weaviate_exists(name: str) -> bool:
    """Whether the target collection `name` exists."""
    return _weaviate().collections.exists(name)


def weaviate_modules() -> set[str]:
    """The names of the modules that the target lists in /v1/meta."""
    return set(_weaviate().get_meta().get("modules") or {})


def enabled_vectorizer() -> str:
    """vectorizer(), after the test fails when the target does not enable it."""
    module = vectorizer()
    if not module_names(module) & weaviate_modules():
        pytest.fail(
            f"/v1/meta of the target does not list {module}; set "
            f"{VECTORIZER_VARIABLE} to a module that the target enables"
        )
    return module


def embedded_dimension(name: str, vector: str = "default") -> int:
    """The length of the vector `vector` of the first object of `name`."""
    response = _collection(name).query.fetch_objects(limit=1, include_vector=True)
    assert response.objects, f"{name} has no objects"
    return len(response.objects[0].vector[vector])


def weaviate_wait_indexed(name: str, timeout: float = 120.0) -> None:
    """Wait until every shard of `name` has indexed its vector queue.

    A server with ASYNC_INDEXING indexes vectors after the insert answers, and
    a vector search does not find a vector that is still in the queue. Without
    it the queue is always empty.
    """
    deadline = time.monotonic() + timeout
    while True:
        nodes = _weaviate().cluster.nodes(collection=name, output="verbose")
        shards = [shard for node in nodes for shard in node.shards or []]
        if shards and all(
            shard.vector_queue_length == 0 and shard.vector_indexing_status == "READY"
            for shard in shards
        ):
            return
        if time.monotonic() > deadline:
            raise AssertionError(
                f"{name} still indexes vectors after {timeout:.0f} s: "
                + ", ".join(
                    f"{s.name} {s.vector_indexing_status} queue {s.vector_queue_length}"
                    for s in shards
                )
            )
        time.sleep(0.5)


def weaviate_compressed(name: str) -> list[bool]:
    """Whether each shard of the target collection `name` holds compressed vectors."""
    nodes = _weaviate().cluster.nodes(collection=name, output="verbose")
    return [shard.compressed for node in nodes for shard in node.shards or []]


def weaviate_config(name: str) -> CollectionConfig:
    """The configuration of the target collection `name`, read from the server."""
    return _weaviate().collections.use(name).config.get()


def create_target_collection(kwargs: dict[str, Any]) -> None:
    """Create a target collection with the arguments of `collections.create`."""
    _weaviate().collections.create(**kwargs)


def delete_target_collection(name: str) -> None:
    """Delete the target collection `name` and no other. A missing one is no error."""
    _weaviate().collections.delete(name)


def weaviate_collection_names() -> list[str]:
    """The names of every collection of the target."""
    return list(_weaviate().collections.list_all(simple=True))


def delete_target_collections(names: list[str]) -> None:
    """Delete each of the target collections `names` that exists, and no other.

    The server takes about 0.2 seconds to delete a collection that does not
    exist, so the existing ones are listed first.
    """
    existing = set(weaviate_collection_names())
    for name in names:
        if name in existing:
            _weaviate().collections.delete(name)
