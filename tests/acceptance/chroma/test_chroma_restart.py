"""A restart of the suite's Chroma during a copy.

The test restarts the container of tests/acceptance/chroma/docker-compose.yml,
so it runs only where the suite owns that container.
"""

import json
import subprocess
import time
from pathlib import Path

import httpx
import pytest
from chroma.seed import count
from conftest import enabled_vectorizer, run, start, weaviate_count
from e2e import COMMAND_TIMEOUT

pytestmark = pytest.mark.infra

TARGET = "Resume"
COMPOSE_FILE = Path(__file__).with_name("docker-compose.yml")
RESTART_AFTER_SECONDS = 2
RESTART_TIMEOUT = 120
READY_TIMEOUT = 60


def _restart_chroma() -> None:
    completed = subprocess.run(
        ["docker", "compose", "-f", str(COMPOSE_FILE), "restart"],
        capture_output=True,
        text=True,
        timeout=RESTART_TIMEOUT,
        check=False,
    )
    assert completed.returncode == 0, f"docker compose restart: {completed.stderr}"


def _wait_for_chroma(total: int) -> None:
    """Wait until Chroma answers again and holds the `total` records of resume.

    docker compose restart returns once the container runs, before the server
    in it answers.
    """
    deadline = time.monotonic() + READY_TIMEOUT
    while True:
        try:
            found = count("resume")
        except httpx.HTTPError as exc:
            problem = f"{type(exc).__name__}: {exc}"
        else:
            assert found == total, f"resume holds {found} of {total} after the restart"
            return
        if time.monotonic() > deadline:
            pytest.fail(
                f"Chroma does not answer {READY_TIMEOUT} s after the restart: {problem}"
            )
        time.sleep(1)


def test_a_chroma_restart_during_copy_leaves_no_reject(seeded, tmp_path):
    # Both mode embeds every document, so the copy still runs when Chroma
    # restarts.
    args = ["--vectors", "both", "--vectorizer", enabled_vectorizer()]
    r = run(["inspect", "chroma", "--collection", "resume", *args], tmp_path)
    assert r.exit_code == 0, r
    total = count("resume")
    stdout = tmp_path / "copy.json"
    process = start(["copy"], tmp_path, stdout)
    restarted = False
    try:
        time.sleep(RESTART_AFTER_SECONDS)
        if process.poll() is not None:
            pytest.fail(
                f"copy exited {process.returncode} within {RESTART_AFTER_SECONDS} s, "
                "before the restart of Chroma:\n" + stdout.read_text(encoding="utf-8")
            )
        restarted = True
        _restart_chroma()
        process.wait(timeout=COMMAND_TIMEOUT)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        if restarted:
            _wait_for_chroma(total)
    text = stdout.read_text(encoding="utf-8")
    assert process.returncode == 0, f"copy exited {process.returncode}:\n{text}"
    row = json.loads(text)["collections"][0]
    assert (row["copied"], row["total"], row["rejects"], row["done"]) == (
        total,
        total,
        0,
        True,
    ), row
    assert weaviate_count(TARGET) == total
    r = run(["verify"], tmp_path)
    assert r.exit_code == 0 and r.data["status"] == "success", r
    assert {c["name"] for c in r.data["checks"]} == {"count", "ids", "records"}
