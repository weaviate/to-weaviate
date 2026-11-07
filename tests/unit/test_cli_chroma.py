"""CLI commands over the real ChromaClient and ChromaSource and a fake server."""

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import httpx
import pytest
from chroma_server import ChromaServer, ServerFactories
from fakes import FakeFactories, cli_env, invoke
from typer.testing import CliRunner

from to_weaviate import cli
from to_weaviate.cli import build_app
from to_weaviate.settings import ChromaSettings
from to_weaviate.sources.base import Record
from to_weaviate.sources.chroma.client import ChromaClient
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
    gets = [c for c in server.calls[before:] if c.kind == "get"]
    # The id at offset 2699, read without a payload, shows that nothing shifted.
    assert (gets[0].limit, gets[0].include) == (1, [])
    assert [c.offset for c in gets] == [2699, 2700, 2750]
    assert len(fakes.writer.writes_per_uuid) == 2750
    assert set(fakes.writer.writes_per_uuid.values()) == {1}


def cli_app(monkeypatch, server: ChromaServer, fakes: FakeFactories) -> Any:
    """The CLI app with its own Chroma source factory over `server`."""

    def client(settings: ChromaSettings) -> ChromaClient:
        transport = httpx.MockTransport(server.handler)
        return ChromaClient(settings, transport=transport, sleep=lambda s: None)

    monkeypatch.setattr(cli, "ChromaClient", client)
    return build_app(cli._chroma_source, fakes._writer, fakes._reader, fakes._schema)


def run_cli(app: Any, *args: str, **env: str | None) -> Any:
    return CliRunner().invoke(app, list(args), env=cli_env(**env))


TENANT_FROM_KEY = "tenant-of-the-key"


def test_inspect_reads_the_tenant_from_the_key_only_without_chroma_tenant(
    monkeypatch,
):
    server = ChromaServer(records(0, 10), tenant=TENANT_FROM_KEY)
    app = cli_app(monkeypatch, server, FakeFactories(None))  # type: ignore[arg-type]
    r = run_cli(app, "inspect", "chroma", "--json", "--force")
    assert r.exit_code == 0, r.output
    identity = [c for c in server.calls if c.kind == "identity"]
    assert len(identity) == 1 and identity[0].headers["x-chroma-token"] == "chroma-key"
    paths = [c.raw_path for c in server.calls if c.kind != "identity"]
    assert paths and all(
        p.startswith(f"/api/v2/tenants/{TENANT_FROM_KEY}/") for p in paths
    )
    server.calls.clear()
    r = run_cli(
        app,
        "inspect",
        "chroma",
        "--json",
        "--force",
        CHROMA_TENANT=TENANT_FROM_KEY,
        CHROMA_TOKEN_HEADER="Authorization",
    )
    assert r.exit_code == 0, r.output
    assert [c.kind for c in server.calls if c.kind == "identity"] == []
    assert {c.headers["authorization"] for c in server.calls} == {"Bearer chroma-key"}


def test_a_key_that_the_identity_refuses_exits_3(monkeypatch):
    server = ChromaServer(records(0, 10))
    server.faults["identity"] = [httpx.Response(401, json={"message": "chroma-key"})]
    app = cli_app(monkeypatch, server, FakeFactories(None))  # type: ignore[arg-type]
    r = run_cli(app, "inspect", "chroma", "--json")
    assert r.exit_code == 3, r.output
    d = json.loads(r.stdout)
    assert d["code"] == "auth_failed" and "CHROMA_API_KEY" in d["hint"]
    assert "chroma-key" not in r.output


def test_tenant_database_and_collection_id_are_percent_encoded_in_paths(tmp_path):
    server = ChromaServer(
        records(0, 10), tenant="t x", database="my db/1", collection_id="id+plus"
    )
    env = {"CHROMA_TENANT": "t x", "CHROMA_DATABASE": "my db/1"}
    fakes = ServerFactories(server)
    for args in (["inspect", "chroma"], ["copy"], ["verify"]):
        r = invoke(fakes, *args, "--json", **env)
        assert r.exit_code == 0, (args, r.output)
    prefix = "/api/v2/tenants/t%20x/databases/my%20db%2F1/collections"
    assert {c.raw_path for c in server.calls if c.kind != "unknown"} >= {
        prefix,
        f"{prefix}/id%2Bplus/count",
        f"{prefix}/id%2Bplus/get",
    }
    assert all(c.kind != "unknown" for c in server.calls), server.calls
    assert len(fakes.writer.objects) == 10
    settings = ChromaSettings(
        url="http://chroma.test",
        api_key=None,
        tenant=None,
        database=None,
        token_header="x-chroma-token",
    )
    client = ChromaClient(settings, transport=httpx.MockTransport(server.handler))
    assert (
        client.query("t x", "my db/1", "id+plus", embedding=[1.0, 0.0], n_results=1)
        == []
    )
    assert server.calls[-1].raw_path == f"{prefix}/id%2Bplus/query"


def test_a_403_on_the_database_list_names_the_database_settings():
    server = ChromaServer(records(0, 10))
    server.faults["databases"] = [httpx.Response(403, json={"message": "forbidden"})]
    r = invoke(ServerFactories(server), "inspect", "chroma", "--json")
    assert r.exit_code == 3, r.output
    d = json.loads(r.stdout)
    assert d["code"] == "auth_failed"
    assert d["message"] == "Chroma refused the request: the API key lacks a permission."
    assert d["hint"] == (
        "Check CHROMA_API_KEY in the environment or in .env. A key for one "
        "database cannot list the databases: set CHROMA_DATABASE or use --database."
    )


def test_a_source_collection_that_is_gone_names_inspect_force():
    server = ChromaServer(records(0, 10))
    fakes = inspected(server)
    server.collection_id = "created-again"
    r = invoke(fakes, "copy", "--json")
    assert r.exit_code == 4, r.output
    d = json.loads(r.stdout)
    assert (d["code"], d["message"]) == (
        "source_unreachable",
        "Chroma returned HTTP 404: collection not found",
    )
    assert d["hint"] == (
        "The collection is gone from Chroma, or it was created again with a new "
        "id. Run to-weaviate inspect chroma --force to inspect it again."
    )
