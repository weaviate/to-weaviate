"""Cases that need a Weaviate cluster of 3 nodes with replication factor 3.

Two of them delete a pod with kubectl. They run in the cluster setup of CI
and skip on a cluster of fewer than 3 nodes. The server sets no minimum
factor: copy creates each collection with `replication_factor: auto`.
"""

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import pytest
from chroma.seed import BULK, BULK_COUNT, count
from conftest import (
    enabled_vectorizer,
    nodes,
    run,
    start,
    weaviate_config,
    weaviate_count,
)
from e2e import COMMAND_TIMEOUT, close_weaviate

from to_weaviate.settings import weaviate_settings
from to_weaviate.target.connect import connect

pytestmark = pytest.mark.infra


@pytest.fixture(scope="module", autouse=True)
def _needs_three_nodes(_test_endpoints: None) -> None:
    # A fixture, not a module-level skipif: nodes() connects to the target,
    # which only the session fixture _test_endpoints may allow.
    count = nodes()
    if count < 3:
        pytest.skip(
            "needs a Weaviate cluster of at least 3 nodes (the cluster's count, "
            f"or ACCEPTANCE_NODES); it has {count}"
        )


TARGET = "Resume"
# The factor that `replication_factor: auto` sends on 3 nodes or more.
REPLICATION_FACTOR = 3
NAMESPACE = "weaviate"
POD = "weaviate-1"
DELETE_AFTER_SECONDS = 3
REPLACE_TIMEOUT = 60
ROLLOUT_TIMEOUT = 60
READY_TIMEOUT = 60


def _kubectl(*args: str, timeout: float = 30) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["kubectl", *args, "-n", NAMESPACE],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _pod_uid() -> str | None:
    """The uid of POD, or None while the pod does not exist.

    Any other kubectl failure fails the test with the stderr of kubectl.
    """
    completed = _kubectl("get", "pod", POD, "-o", "jsonpath={.metadata.uid}")
    if completed.returncode != 0:
        if "(NotFound)" in completed.stderr:
            return None
        pytest.fail(f"kubectl get pod {POD} -n {NAMESPACE}: {completed.stderr}")
    uid = completed.stdout.strip()
    assert uid, f"kubectl get pod {POD} -n {NAMESPACE} printed no uid"
    return uid


def _delete_pod() -> str:
    """Delete POD without waiting for it to stop; return the uid it had."""
    uid = _pod_uid()
    assert uid, f"kubectl finds no pod {POD} in namespace {NAMESPACE}"
    completed = _kubectl("delete", "pod", POD, "--wait=false")
    assert completed.returncode == 0, f"kubectl delete pod {POD}: {completed.stderr}"
    return uid


def _wait_for_the_cluster(deleted_uid: str) -> None:
    """Wait until POD is replaced and ready, and Weaviate answers through the forwards.

    A kubectl port-forward to a service stays on one pod, and its loop starts
    a new one when that pod goes away. A command that runs before that exits 4
    at once.
    """
    deadline = time.monotonic() + REPLACE_TIMEOUT
    while _pod_uid() in (deleted_uid, None):
        if time.monotonic() > deadline:
            pytest.fail(f"{POD} is not replaced {REPLACE_TIMEOUT} s after its deletion")
        time.sleep(1)
    completed = _kubectl(
        "rollout",
        "status",
        "sts/weaviate",
        f"--timeout={ROLLOUT_TIMEOUT}s",
        timeout=ROLLOUT_TIMEOUT + 30,
    )
    assert completed.returncode == 0, (
        f"the statefulset weaviate is not ready {ROLLOUT_TIMEOUT} s after {POD} "
        f"was replaced: {completed.stdout} {completed.stderr}"
    )
    # The helpers of the suite open a new client after this.
    close_weaviate()
    deadline = time.monotonic() + READY_TIMEOUT
    while True:
        # connect checks HTTP and the gRPC health service.
        try:
            client = connect(weaviate_settings(os.environ))
        except Exception as exc:
            problem = f"{type(exc).__name__}: {exc}"
        else:
            try:
                if client.is_ready():
                    return
            finally:
                client.close()
            problem = "/v1/.well-known/ready is not ready"
        if time.monotonic() > deadline:
            pytest.fail(
                f"Weaviate does not answer through the forwards {READY_TIMEOUT} s "
                f"after the rollout: {problem}"
            )
        time.sleep(1)


def _delete_a_pod_during(
    process: subprocess.Popen[bytes], command: str, stdout: Path
) -> dict[str, Any]:
    """Delete POD DELETE_AFTER_SECONDS into `process`; return its JSON document.

    Waits for `process`, then for the cluster. Fails when `process` ends
    before the deletion, because then the case tests nothing.
    """
    uid = None
    try:
        time.sleep(DELETE_AFTER_SECONDS)
        if process.poll() is not None:
            pytest.fail(
                f"{command} exited {process.returncode} within "
                f"{DELETE_AFTER_SECONDS} s, before the deletion of {POD}:\n"
                + stdout.read_text(encoding="utf-8")
            )
        uid = _delete_pod()
        process.wait(timeout=COMMAND_TIMEOUT)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        if uid is not None:
            _wait_for_the_cluster(uid)
    text = stdout.read_text(encoding="utf-8")
    assert process.returncode == 0, f"{command} exited {process.returncode}:\n{text}"
    document = json.loads(text)
    assert isinstance(document, dict) and document["status"] == "success", document
    return document


def _assert_replicated(name: str) -> None:
    factor = weaviate_config(name).replication_config.factor
    assert factor == REPLICATION_FACTOR, (
        f"{name} has replication factor {factor}; copy creates it with "
        f"{REPLICATION_FACTOR} on a cluster of {nodes()} nodes"
    )


def _events(workdir: Path) -> list[str]:
    lines = (workdir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line)["event"] for line in lines]


@pytest.mark.parametrize("level", ["QUORUM", "ALL"])
def test_a_pod_deleted_during_copy_leaves_no_reject(seeded, tmp_path, level):
    # Both mode embeds every document, so the copy still runs when the pod is
    # deleted.
    args = ["--vectors", "both", "--vectorizer", enabled_vectorizer()]
    r = run(["inspect", "chroma", "--collection", "resume", *args], tmp_path)
    assert r.exit_code == 0, r
    total = count("resume")
    stdout = tmp_path / "first-copy.json"
    process = start(["copy", "--consistency", level], tmp_path, stdout)
    document = _delete_a_pod_during(process, "copy", stdout)
    row = document["collections"][0]
    assert (row["copied"], row["total"], row["rejects"], row["done"]) == (
        total,
        total,
        0,
        True,
    ), row
    _assert_replicated(TARGET)
    assert weaviate_count(TARGET) == total
    # The next copy reaches the collection and writes no page.
    before = len(_events(tmp_path))
    r = run(["copy", "--consistency", level], tmp_path)
    assert r.exit_code == 0 and r.data["status"] == "success", r
    added = _events(tmp_path)[before:]
    assert "collection_finished" in added and "progress" not in added, added
    assert weaviate_count(TARGET) == total
    r = run(["verify"], tmp_path)
    assert r.exit_code == 0 and r.data["status"] == "success", r
    assert {c["name"] for c in r.data["checks"]} == {"count", "ids", "records"}


@pytest.mark.parametrize("level", ["ONE", "QUORUM", "ALL"])
def test_copy_with_a_consistency_level_on_three_replicas(seeded, tmp_path, level):
    r = run(["inspect", "chroma", "--collection", "resume"], tmp_path)
    assert r.exit_code == 0, r
    r = run(["copy", "--consistency", level], tmp_path)
    assert r.exit_code == 0 and r.data["status"] == "success", r
    assert r.data["created_collections"] == [
        {
            "collection": TARGET,
            "replication_factor": REPLICATION_FACTOR,
            "chosen_by": "nodes",
            "nodes": nodes(),
        }
    ], r
    _assert_replicated(TARGET)
    assert weaviate_count(TARGET) == count("resume")
    r = run(["verify"], tmp_path)
    assert r.exit_code == 0 and r.data["status"] == "success", r


def test_verify_passes_while_a_pod_is_deleted(seeded, tmp_path):
    # A full verify of 50,000 records runs long enough for the deletion.
    r = run(
        ["inspect", "chroma", "--collection", BULK, "--sample-pages", "5"], tmp_path
    )
    assert r.exit_code == 0, r
    r = run(["copy", "--parallelism", "8"], tmp_path)
    assert r.exit_code == 0 and r.data["status"] == "success", r
    assert weaviate_count("Bulk") == BULK_COUNT
    stdout = tmp_path / "verify.json"
    process = start(["verify"], tmp_path, stdout)
    document = _delete_a_pod_during(process, "verify", stdout)
    assert {c["name"] for c in document["checks"]} == {"count", "ids", "records"}
    assert all(c["passed"] for c in document["checks"]), document["checks"]
