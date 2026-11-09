"""Chroma stopped and started again during a copy.

The test stops and starts the container of
tests/acceptance/chroma/docker-compose.yml, so it runs only where the suite
owns that container. A page read while Chroma is stopped goes through the
source retry, but the tool writes no event for a retry, so the test cannot
show that one ran.
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
STOP_AFTER_SECONDS = 2
STOPPED_SECONDS = 3
COMPOSE_TIMEOUT = 120
READY_TIMEOUT = 60


def _compose(command: str) -> None:
    completed = subprocess.run(
        ["docker", "compose", "-f", str(COMPOSE_FILE), command],
        capture_output=True,
        text=True,
        timeout=COMPOSE_TIMEOUT,
        check=False,
    )
    assert completed.returncode == 0, f"docker compose {command}: {completed.stderr}"


def _wait_for_chroma(total: int) -> None:
    """Wait until Chroma answers again and holds the `total` records of resume.

    docker compose start returns once the container runs, before the server
    in it answers.
    """
    deadline = time.monotonic() + READY_TIMEOUT
    while True:
        try:
            found = count("resume")
        except httpx.HTTPError as exc:
            problem = f"{type(exc).__name__}: {exc}"
        else:
            assert found == total, f"resume holds {found} of {total} after the start"
            return
        if time.monotonic() > deadline:
            pytest.fail(
                f"Chroma does not answer {READY_TIMEOUT} s after the start: {problem}"
            )
        time.sleep(1)


def test_a_chroma_restart_during_copy_leaves_no_reject(seeded, tmp_path):
    # Both mode embeds every document, so the copy still runs when Chroma
    # stops.
    args = ["--vectors", "both", "--vectorizer", enabled_vectorizer()]
    r = run(["inspect", "chroma", "--collection", "resume", *args], tmp_path)
    assert r.exit_code == 0, r
    total = count("resume")
    stdout = tmp_path / "copy.json"
    process = start(["copy"], tmp_path, stdout)
    stopped = started = False
    try:
        time.sleep(STOP_AFTER_SECONDS)
        if process.poll() is not None:
            pytest.fail(
                f"copy exited {process.returncode} within {STOP_AFTER_SECONDS} s, "
                "before Chroma was stopped:\n" + stdout.read_text(encoding="utf-8")
            )
        stopped = True
        _compose("stop")
        time.sleep(STOPPED_SECONDS)
        _compose("start")
        started = True
        process.wait(timeout=COMMAND_TIMEOUT)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        if stopped:
            # The rest of the suite needs Chroma.
            if not started:
                _compose("start")
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
