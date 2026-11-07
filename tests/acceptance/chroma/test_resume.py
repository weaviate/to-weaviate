import signal
import time

import pytest
from chroma.seed import count
from conftest import run, start, weaviate_count

TARGET = "Resume"
# The copy is killed once it has copied this part of the records.
KILL_AFTER = 1 / 3
POLL_SECONDS = 0.2
# A copy that has not reached KILL_AFTER by then has stopped making progress.
KILL_DEADLINE_SECONDS = 300


def _progress(workdir) -> dict:
    r = run(["status"], workdir)
    assert r.exit_code == 0, r
    row: dict = r.data["collections"][0]
    return row


def test_copy_resumes_after_the_process_is_killed(seeded, tmp_path):
    # Both mode embeds every document, which makes the copy slow enough to
    # kill in the middle without a large source.
    args = ["--vectors", "both", "--vectorizer", "text2vec-transformers"]
    r = run(["inspect", "chroma", "--collection", "resume", *args], tmp_path)
    assert r.exit_code == 0, r
    r = run(["check"], tmp_path)
    assert r.exit_code == 0, r
    total = count("resume")

    first = tmp_path / "first-copy.json"
    process = start(["copy"], tmp_path, first)
    try:
        deadline = time.monotonic() + KILL_DEADLINE_SECONDS
        copied = 0
        while process.poll() is None and time.monotonic() < deadline:
            if (tmp_path / "state.sqlite").exists():
                copied = _progress(tmp_path)["copied"]
                if copied >= total * KILL_AFTER:
                    break
            time.sleep(POLL_SECONDS)
        if process.returncode == 0:
            pytest.fail(
                f"copy finished after copying {copied} of {total} records, "
                "before it could be killed; raise RESUME_COUNT in "
                "tests/acceptance/chroma/seed.py"
            )
        if process.returncode is not None:
            pytest.fail(
                f"copy exited {process.returncode} before it could be killed:\n"
                + first.read_text(encoding="utf-8")
            )
        assert copied >= total * KILL_AFTER, f"copied {copied} of {total} in time"
        # A second copy on the directory is refused while the first one runs,
        # and the first one goes on.
        second = run(["copy"], tmp_path)
        assert second.exit_code == 2 and second.data["code"] == "usage", second
        assert second.data["message"] == f"Another to-weaviate copy uses {tmp_path}."
        while process.poll() is None and _progress(tmp_path)["copied"] == copied:
            assert time.monotonic() < deadline, "the first copy made no progress"
            time.sleep(POLL_SECONDS)
        assert process.poll() is None, "the first copy ended before it was killed"
        process.send_signal(signal.SIGKILL)
        process.wait()
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
    assert process.returncode == -signal.SIGKILL

    row = _progress(tmp_path)
    assert 0 < row["copied"] < total and not row["done"], row

    # The killed process held copy.lock; the operating system released it.
    r = run(["copy"], tmp_path)
    assert r.exit_code == 0 and r.data["status"] == "success", r
    assert weaviate_count(TARGET) == total
    row = _progress(tmp_path)
    assert (row["copied"], row["total"], row["rejects"], row["done"]) == (
        total,
        total,
        0,
        True,
    ), row
    r = run(["verify"], tmp_path)
    assert r.exit_code == 0 and r.data["status"] == "success", r
    assert {c["name"] for c in r.data["checks"]} == {"count", "ids", "records"}
