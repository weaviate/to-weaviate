"""Helpers and fixtures of the acceptance suite.

The suite runs the installed to-weaviate command against a real Chroma server
and a real Weaviate server. It starts neither of them. It runs only when the
environment names the test endpoints, because it deletes and recreates
collections on both servers.
"""

import os
import subprocess
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import e2e
import pytest
import yaml
from chroma.seed import seed
from e2e import (
    Result,
    close_weaviate,
    delete_target_collections,
    weaviate_by_source_id,
    weaviate_compressed,
    weaviate_config,
    weaviate_count,
    weaviate_delete_object,
    weaviate_exists,
    weaviate_insert_object,
    weaviate_near_text,
    weaviate_near_vector,
    weaviate_object,
    weaviate_set_shard_status,
    weaviate_tenants,
    weaviate_update_object,
)

# The test modules import these names from this module.
__all__ = [
    "SUITE_COLLECTIONS",
    "KnownFinding",
    "Result",
    "copy_and_verify",
    "read_plan",
    "run",
    "start",
    "weaviate_by_source_id",
    "weaviate_compressed",
    "weaviate_config",
    "weaviate_count",
    "weaviate_delete_object",
    "weaviate_exists",
    "weaviate_insert_object",
    "weaviate_near_text",
    "weaviate_near_vector",
    "weaviate_object",
    "weaviate_set_shard_status",
    "weaviate_tenants",
    "weaviate_update_object",
    "write_plan",
]

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
]


class KnownFinding(AssertionError):
    """A test reached the failure of a finding that is recorded and not fixed.

    A test marked xfail(raises=KnownFinding) fails for any other assertion.
    """


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
def _delete_suite_collections(_test_endpoints: None) -> Iterator[None]:
    delete_target_collections(SUITE_COLLECTIONS)
    yield
    delete_target_collections(SUITE_COLLECTIONS)
