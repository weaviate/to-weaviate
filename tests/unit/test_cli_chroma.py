"""CLI commands over the real ChromaClient and ChromaSource and a fake server."""

import json
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest
from chroma_server import ChromaServer, ServerFactories
from fakes import invoke

from to_weaviate.sources.base import Record
from to_weaviate.state import State

DIR = Path(".to-weaviate")


@pytest.fixture(autouse=True)
def in_tmp(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


@pytest.fixture(autouse=True)
def no_jitter():
    with patch("to_weaviate.retry.random.uniform", return_value=0.0):
        yield


def records(start: int, stop: int) -> list[Record]:
    return [
        Record(f"id-{i:04d}", [float(i), 1.0], f"doc {i}", None, {"n": i})
        for i in range(start, stop)
    ]


def inspected(server: ChromaServer, **kwargs) -> ServerFactories:
    fakes = ServerFactories(server, **kwargs)
    r = invoke(fakes, "inspect", "chroma")
    assert r.exit_code == 0, r.output
    return fakes


def copied(fakes: ServerFactories, *args: str) -> dict:
    r = invoke(fakes, "copy", "--json", *args)
    assert r.exit_code == 0, r.output
    row: dict = json.loads(r.stdout)["collections"][0]
    return row


@pytest.mark.parametrize(("cap", "reads"), [(100, 10), (None, 4)])
def test_a_page_shorter_than_the_limit_does_not_end_a_partition(cap, reads):
    server = ChromaServer(records(0, 1000), page_cap=cap)
    fakes = inspected(server)
    before = len(server.gets("embeddings"))
    row = copied(fakes, "--parallelism", "1")
    assert (row["copied"], row["total"], row["done"]) == (1000, 1000, True)
    assert len(fakes.writer.objects) == 1000
    # A bounded partition ends at its stop without one more read.
    assert len(server.gets("embeddings")) - before == reads
    r = invoke(fakes, "verify", "--json")
    assert r.exit_code == 0, r.output
    ids = next(c for c in json.loads(r.stdout)["checks"] if c["name"] == "ids")
    assert ids["detail"] == "missing 0, extra 0"
    # A tail ends only at an empty page.
    server.records += records(1000, 1250)
    before = len(server.gets("embeddings"))
    row = copied(fakes, "--parallelism", "1")
    assert (row["copied"], row["total"]) == (1250, 1250)
    assert len(fakes.writer.objects) == 1250
    tail_reads = 4 if cap else 2
    assert len(server.gets("embeddings")) - before == tail_reads
    assert set(fakes.writer.writes_per_uuid.values()) == {1}


def partitions(directory: Path) -> dict[str, tuple[int, int, str | None, bool]]:
    """(start, stop, cursor, finished) of each stored partition, by key."""
    state = State(directory / "state.sqlite")
    try:
        pending = {p.key: cursor for p, cursor in state.pending("chroma-docs")}
        rows = {}
        for p in state.partitions("chroma-docs"):
            done = p.key not in pending
            cursor = None if done else pending[p.key]
            rows[p.key] = (int(p.start), int(p.stop), cursor, done)
        return rows
    finally:
        state.close()


def assert_resumable(directory: Path, writer) -> None:
    """Each partition's cursor follows the pages it wrote, and only those."""
    written = {int(o.properties["n"]) for o in writer.objects.values()}
    for start, stop, cursor, done in partitions(directory).values():
        end = stop if done else int(cursor or start)
        assert set(range(start, end)) <= written, (start, stop, cursor)
        assert not set(range(end, stop)) & written, (start, stop, cursor)


ERROR = {"error": "InternalError", "message": "boom"}
# Faults on the second /get of the second of three partitions of 900 records.
MID_PAGE = {
    "transport-error": ([httpx.ReadError("connection reset")], [0.0]),
    "503-then-429": (
        [
            httpx.Response(503, json=ERROR),
            httpx.Response(429, json=ERROR, headers={"Retry-After": "2"}),
        ],
        [0.0, 2.0],
    ),
}


@pytest.mark.parametrize(("faults", "sleeps"), MID_PAGE.values(), ids=MID_PAGE)
def test_a_fault_in_the_middle_of_a_page_is_retried(faults, sleeps):
    server = ChromaServer(records(0, 2700))
    fakes = inspected(server)
    server.faults[("get", 1200)] = list(faults)
    row = copied(fakes, "--parallelism", "3")
    assert (row["copied"], row["total"], row["done"]) == (2700, 2700, True)
    assert fakes.sleeps == sleeps
    assert len(fakes.writer.writes_per_uuid) == 2700
    assert set(fakes.writer.writes_per_uuid.values()) == {1}


def test_a_page_that_fails_every_attempt_stops_copy_and_resumes(tmp_path):
    server = ChromaServer(records(0, 2700))
    fakes = inspected(server)
    server.faults[("get", 1200)] = [httpx.Response(500, json=ERROR)] * 6
    r = invoke(fakes, "copy", "--parallelism", "3", "--json")
    assert r.exit_code == 4, r.output
    d = json.loads(r.stdout)
    assert d["code"] == "source_unreachable"
    assert d["message"] == "Chroma returned HTTP 500: boom"
    assert fakes.sleeps == [0.0] * 5
    stored = partitions(tmp_path / DIR)
    assert stored["900-1800"][2:] == ("1200", False)
    assert_resumable(tmp_path / DIR, fakes.writer)
    row = copied(fakes, "--parallelism", "3")
    assert (row["copied"], row["total"], row["done"]) == (2700, 2700, True)
    assert len(fakes.writer.writes_per_uuid) == 2700
    assert set(fakes.writer.writes_per_uuid.values()) == {1}


def test_a_count_that_fails_after_the_pages_resumes_without_pages(tmp_path):
    server = ChromaServer(records(0, 2700))
    fakes = inspected(server)
    server.faults["count"] = [httpx.Response(500, json=ERROR)] * 6
    r = invoke(fakes, "copy", "--parallelism", "3", "--json")
    assert r.exit_code == 4 and json.loads(r.stdout)["code"] == "source_unreachable"
    assert all(done for *_, done in partitions(tmp_path / DIR).values())
    assert len(partitions(tmp_path / DIR)) == 3
    assert len(fakes.writer.objects) == 2700
    server.records += records(2700, 2750)
    before = len(server.calls)
    row = copied(fakes, "--parallelism", "3")
    assert (row["copied"], row["total"], row["done"]) == (2750, 2750, True)
    offsets = [c.offset for c in server.calls[before:] if c.kind == "get"]
    assert offsets == [2700, 2750]
    assert len(fakes.writer.writes_per_uuid) == 2750
    assert set(fakes.writer.writes_per_uuid.values()) == {1}
