"""Helpers and fixtures of the acceptance suite.

The suite runs the installed to-weaviate command against a real Chroma server
and a real Weaviate server. It starts neither of them. It runs only when the
environment names the endpoints of its target, because it deletes and
recreates collections on both servers. `endpoint_problems` in tests/e2e.py
holds the rules of each target.
"""

import os
import subprocess
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import e2e
import pytest
import yaml
from chroma.seed import SEEDED, seed, wait_indexed
from e2e import (
    Result,
    chroma_kind,
    close_weaviate,
    compaction_properties,
    create_target_collection,
    delete_target_collections,
    embedded_dimension,
    enabled_vectorizer,
    endpoint_problems,
    nodes,
    shard_replicas,
    target,
    vectorizer,
    weaviate_by_source_id,
    weaviate_compressed,
    weaviate_config,
    weaviate_count,
    weaviate_delete_object,
    weaviate_exists,
    weaviate_fetch,
    weaviate_insert,
    weaviate_insert_object,
    weaviate_modules,
    weaviate_near_text,
    weaviate_near_vector,
    weaviate_object,
    weaviate_set_shard_status,
    weaviate_set_tenant_status,
    weaviate_tenants,
    weaviate_update_object,
    weaviate_wait_indexed,
)

# The test modules import these names from this module.
__all__ = [
    "ASYNC_VARIABLE",
    "KnownFinding",
    "Result",
    "SUITE_COLLECTIONS",
    "async_indexing",
    "chroma_kind",
    "compaction_properties",
    "compression_wait",
    "copy_and_verify",
    "create_target_collection",
    "embedded_dimension",
    "enabled_vectorizer",
    "endpoint_problems",
    "nodes",
    "read_plan",
    "run",
    "run_text",
    "shard_replicas",
    "start",
    "target",
    "vectorizer",
    "weaviate_by_source_id",
    "weaviate_compressed",
    "weaviate_config",
    "weaviate_count",
    "weaviate_delete_object",
    "weaviate_exists",
    "weaviate_fetch",
    "weaviate_insert",
    "weaviate_insert_object",
    "weaviate_modules",
    "weaviate_near_text",
    "weaviate_near_vector",
    "weaviate_object",
    "weaviate_set_shard_status",
    "weaviate_set_tenant_status",
    "weaviate_tenants",
    "weaviate_update_object",
    "weaviate_wait_indexed",
    "write_plan",
]

REQUIRE_VARIABLE = "TO_WEAVIATE_REQUIRE_ACCEPTANCE"
# Whether the Weaviate server runs with ASYNC_INDEXING: "true" or "false",
# as the CI matrix sets it. Unset, a test decides from what the server does.
ASYNC_VARIABLE = "WEAVIATE_ASYNC_INDEXING"
# The local Chroma needs none of these. A value from the shell would change
# the tenant, the database or the key header that the command uses.
CLEARED_VARIABLES = (
    "CHROMA_DATABASE",
    "CHROMA_TENANT",
    "CHROMA_API_KEY",
    "CHROMA_TOKEN_HEADER",
)
# The target collections the suite creates. The suite touches no other one.
SUITE_COLLECTIONS = [
    "Docs_cosine",
    "Tiny_l2",
    "Docs_rq",
    "Docs_ip",
    "Tenant_a",
    "Tenant_b",
    "Tenants",
    "Rejects",
    "Tiny_l2_renamed",
    "Openai_proxy",
    "Openai_ada",
    "Resume",
    "Empty",
    "Edge",
    # The draft names the two mixed-case collections in the order Chroma
    # lists them.
    "Mixed_Case",
    "Mixed_case",
    "Mixed_Case_2",
    "Mixed_case_2",
    "C_1_leading_digit",
    "Class",
    "Bulk",
    "Burst",
    "Sync",
]


class KnownFinding(AssertionError):
    """A test reached the failure of a finding that is recorded and not fixed.

    A test marked xfail(raises=KnownFinding) fails for any other assertion.
    """


def async_indexing() -> bool | None:
    """What ASYNC_VARIABLE says about the server, or None when it is unset."""
    value = os.environ.get(ASYNC_VARIABLE, "").strip().lower()
    if value in ("1", "true", "yes", "on"):
        return True
    if value in ("0", "false", "no", "off"):
        return False
    return None


def compression_wait() -> float:
    """Seconds to wait for a trained quantizer that copy enabled after the pages.

    A server without ASYNC_INDEXING compresses at once. A server with it
    indexes the queue first, which took up to 85 s on a 3-node cluster.
    """
    return 3.0 if async_indexing() is False else 180.0


def read_plan(workdir: Path) -> dict[str, Any]:
    """The plan.yaml of `workdir` as plain data."""
    data = yaml.safe_load((workdir / "plan.yaml").read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


def write_plan(workdir: Path, plan: dict[str, Any]) -> None:
    """Replace the plan.yaml of `workdir` with `plan`."""
    (workdir / "plan.yaml").write_text(
        yaml.safe_dump(plan, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )


def run(args: list[str], workdir: Path, env: dict[str, str] | None = None) -> Result:
    """Run the installed to-weaviate with `args`, --json and --dir `workdir`.

    The working directory is `workdir`, so no `.env` of the repository is read.
    The environment is os.environ with CLEARED_VARIABLES empty and `env` on top.
    """
    cleared = dict.fromkeys(CLEARED_VARIABLES, "")
    return e2e.run(args, workdir, {**cleared, **(env or {})})


def run_text(
    args: list[str], workdir: Path, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    """Run the installed to-weaviate as `run` does, without --json."""
    cleared = dict.fromkeys(CLEARED_VARIABLES, "")
    return e2e.run_text(args, workdir, {**cleared, **(env or {})})


def copy_and_verify(workdir: Path) -> Result:
    """Run check, copy and verify in `workdir`; each must succeed.

    Returns the result of verify.
    """
    for cmd in (["check"], ["copy"], ["verify"]):
        r = run(cmd, workdir)
        assert r.exit_code == 0 and r.data["status"] == "success", r
    return r


def start(
    args: list[str], workdir: Path, stdout: Path, env: dict[str, str] | None = None
) -> subprocess.Popen[bytes]:
    """Start the installed to-weaviate as `run` does, without waiting for it.

    Its stdout goes to the file `stdout`.
    """
    cleared = dict.fromkeys(CLEARED_VARIABLES, "")
    return e2e.start(args, workdir, stdout, {**cleared, **(env or {})})


@pytest.fixture(scope="session", autouse=True)
def _test_endpoints() -> Iterator[None]:
    problems = endpoint_problems()
    if problems:
        reason = (
            "the acceptance suite runs only against the test endpoints: "
            + "; ".join(problems)
            + ". See docs/development/testing.md."
        )
        # CI sets it, so that a suite that skips every test does not pass.
        if os.environ.get(REQUIRE_VARIABLE) == "1":
            pytest.fail(reason)
        pytest.skip(reason)
    if target() == "cloud":
        listed = e2e.weaviate_collection_names()
        problem = e2e.dedicated_cluster_problem(listed, SUITE_COLLECTIONS)
        if problem:
            close_weaviate()
            pytest.fail(problem)
    yield
    close_weaviate()


# tryfirst: the guard runs before the hook that sets up the fixtures of the
# item, so a fixture of an infra test never runs on the cloud target.
@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item: pytest.Item) -> None:
    e2e.fail_infra_on_cloud(item)


@pytest.fixture(scope="session")
def seeded(_test_endpoints: None) -> None:
    seed()
    if chroma_kind() == "distributed":
        wait_indexed(SEEDED)


@pytest.fixture(autouse=True)
def _no_secret_in_a_file(tmp_path: Path) -> Iterator[None]:
    """After the test, no file under its directory holds a secret value.

    The secrets are the values of the secret variables of every command the
    test ran. state.sqlite is read as bytes like the other files.
    """
    e2e.take_secrets()
    yield
    secrets = [s.encode("utf-8") for s in e2e.take_secrets()]
    for path in sorted(p for p in tmp_path.rglob("*") if p.is_file()):
        data = path.read_bytes()
        assert not any(
            s in data for s in secrets
        ), f"{path.relative_to(tmp_path)} holds a secret value"


@pytest.fixture(autouse=True)
def _delete_suite_collections(_test_endpoints: None) -> Iterator[None]:
    delete_target_collections(SUITE_COLLECTIONS)
    yield
    delete_target_collections(SUITE_COLLECTIONS)
