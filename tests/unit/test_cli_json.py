import json
from collections.abc import Iterator
from typing import Any

import httpx
import pytest
import typer
from fakes import (
    FakeFactories,
    FakeSchema,
    FakeSource,
    FakeWriter,
    cli_records,
    invoke,
)
from weaviate.exceptions import (
    AuthenticationFailedError,
    InsufficientPermissionsError,
    UnexpectedStatusCodeError,
    WeaviateConnectionError,
    WeaviateGRPCUnavailableError,
    WeaviateStartUpError,
    WeaviateTimeoutError,
)

from to_weaviate.cli import app
from to_weaviate.output import ERROR_EXIT, CliError
from to_weaviate.plan import load_plan
from to_weaviate.sources.chroma.client import ChromaError

COMMANDS = [["inspect", "chroma"], ["check"], ["copy"], ["status"]]
PROGRESS = {
    "inspect": "Inspecting",
    "check": "Checking",
    "copy": "Copying",
    "verify": "Verifying",
}


@pytest.fixture(autouse=True)
def in_tmp(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


def leaf_commands(command: Any, path: tuple[str, ...] = ()) -> Iterator[Any]:
    children = getattr(command, "commands", None)
    if children is None:
        yield " ".join(path), command
        return
    for name, child in children.items():
        yield from leaf_commands(child, (*path, name))


def test_every_command_has_a_json_option():
    commands = dict(leaf_commands(typer.main.get_command(app)))
    assert set(commands) >= {"inspect chroma", "check", "copy", "status", "verify"}
    missing = [
        name
        for name, command in commands.items()
        if not any("--json" in param.opts for param in command.params)
    ]
    assert missing == []


def test_every_command_has_a_dir_option():
    commands = dict(leaf_commands(typer.main.get_command(app)))
    missing = [
        name
        for name, command in commands.items()
        if not any("--dir" in param.opts for param in command.params)
    ]
    assert missing == []


@pytest.mark.parametrize(
    "cmd", [["inspect", "chroma"], ["check"], ["copy"], ["status"], ["verify"]]
)
def test_json_stdout_is_one_document(cmd):
    fakes = FakeFactories(FakeSource(cli_records(10)))
    if cmd[0] != "inspect":
        assert invoke(fakes, "inspect", "chroma").exit_code == 0
    if cmd[0] == "verify":
        assert invoke(fakes, "copy").exit_code == 0
    r = invoke(fakes, *cmd, "--json")
    assert r.exit_code == 0, r.output
    assert json.loads(r.stdout)["status"] == "success"
    progress = PROGRESS.get(cmd[0])
    if progress is None:
        assert r.stderr == ""
    else:
        assert progress in r.stderr and progress not in r.stdout


def test_json_error_document():
    fakes = FakeFactories(FakeSource(cli_records(10)))
    r = invoke(fakes, "inspect", "chroma", "--json", CHROMA_API_KEY=None)
    assert r.exit_code == 3
    d = json.loads(r.stdout)
    assert (d["status"], d["code"]) == ("error", "missing_credential")
    assert "CHROMA_API_KEY" in d["hint"] and d["message"]


def test_text_error_goes_to_stderr():
    fakes = FakeFactories(FakeSource(cli_records(10)))
    r = invoke(fakes, "inspect", "chroma", CHROMA_API_KEY=None)
    assert r.exit_code == 3 and r.stdout == ""
    assert "Error: CHROMA_API_KEY" in r.stderr and "Hint: " in r.stderr


def test_exit_code_does_not_depend_on_json():
    fakes = FakeFactories(
        FakeSource(cli_records(10)), writer=FakeWriter(reject_ids={"id-3"})
    )
    assert invoke(fakes, "inspect", "chroma").exit_code == 0
    text = invoke(fakes, "copy")
    js = invoke(fakes, "copy", "--json")
    assert text.exit_code == js.exit_code == 1
    d = json.loads(js.stdout)
    assert (d["status"], d["code"]) == ("error", "rejects")
    assert "rejects.jsonl" in d["hint"]
    assert d["collections"][0]["rejects"] == 1 and d["total"] == 10


def test_inspect_json_lists_decisions():
    fakes = FakeFactories(FakeSource(cli_records(10, year_text_at=3)))
    r = invoke(fakes, "inspect", "chroma", "--json")
    assert r.exit_code == 0, r.output
    d = json.loads(r.stdout)
    assert {x["kind"] for x in d["decisions"]} == {"vector_mode", "mixed_types"}
    assert all(x["collection"] == "docs" and x["detail"] for x in d["decisions"])
    assert d["plan"].endswith("plan.yaml") and d["total"] == 10
    assert d["collections"] == [
        {
            "ref": FakeSource.REF,
            "source": "docs",
            "target": "Docs",
            "count": 10,
            "dimension": 2,
            "distance": "cosine",
            "vectors": [{"name": "default", "mode": "keep", "vectorizer": None}],
            "notes": [],
        }
    ]


def test_no_secret_in_any_output(tmp_path):
    env = {
        "CHROMA_API_KEY": "SENTINEL-A",
        "WEAVIATE_API_KEY": "SENTINEL-B",
        "OPENAI_API_KEY": "SENTINEL-C",
        "CHROMA_URL": "https://user:SENTINEL-D@chroma.example.com",
        "WEAVIATE_URL": "http://admin:SENTINEL-E@localhost:8099",
    }
    fakes = FakeFactories(
        FakeSource(cli_records(10)), writer=FakeWriter(reject_ids={"id-3"})
    )
    embed = ["--vectors", "embed", "--vectorizer", "text2vec-openai"]
    all_output = ""
    for args, exit_code in (
        (["inspect", "chroma", *embed], 0),
        (["check"], 0),
        (["copy"], 1),
        (["status"], 0),
        (["verify"], 1),
        (["inspect", "chroma", "--force", "--json", *embed], 0),
        (["check", "--json"], 0),
        (["copy", "--json"], 1),
        (["status", "--json"], 0),
        (["verify", "--json"], 1),
    ):
        r = invoke(fakes, *args, **env)
        assert r.exit_code == exit_code, r.output
        all_output += r.output
    assert (tmp_path / ".to-weaviate" / "rejects.jsonl").exists()
    server_body = {"message": "bad SENTINEL-B for X-OpenAI-Api-Key SENTINEL-C at E"}
    server_body["message"] += " http://admin:SENTINEL-E@localhost:8099"
    failures: list[tuple[list[str], Exception | None, Exception | None, int]] = [
        (
            ["inspect", "chroma", "--force"],
            ChromaError(None, "LocalProtocolError: Illegal header value b'SENTINEL-A'"),
            None,
            4,
        ),
        (
            ["check"],
            None,
            UnexpectedStatusCodeError("x", httpx.Response(500, json=server_body)),
            4,
        ),
        (
            ["copy"],
            None,
            RuntimeError("SENTINEL-B SENTINEL-C https://user:SENTINEL-D@chroma"),
            5,
        ),
        (["status", "--dir", "missing"], None, None, 2),
    ]
    for args, source_error, target_error, exit_code in failures:
        fakes.source_error, fakes.target_error = source_error, target_error
        for mode in ([], ["--json"]):
            r = invoke(fakes, *args, *mode, TO_WEAVIATE_DEBUG="1", **env)
            assert r.exit_code == exit_code, r.output
            all_output += r.output
    for f in tmp_path.rglob("*"):
        if f.is_file():
            assert b"SENTINEL" not in f.read_bytes()
    assert "SENTINEL" not in all_output


@pytest.mark.parametrize("mode", [[], ["--json"]])
def test_secret_in_an_exception_is_redacted(mode):
    error = ChromaError(None, "LocalProtocolError: Illegal header value b'SENTINEL-A'")
    fakes = FakeFactories(FakeSource(cli_records(10)), source_error=error)
    r = invoke(fakes, "inspect", "chroma", *mode, CHROMA_API_KEY="SENTINEL-A")
    assert r.exit_code == 4
    assert "SENTINEL" not in r.stdout and "SENTINEL" not in r.stderr
    assert "***" in r.output


LONG_KEY = (
    "".join(chr(ord("a") + i * 7 % 26) for i in range(230))
    + "\n"
    + "".join(chr(ord("A") + i * 11 % 26) for i in range(230))
)


@pytest.mark.parametrize("mode", [[], ["--json"]])
def test_long_secret_is_redacted_before_shortening(mode):
    assert len(LONG_KEY) == 461
    fakes = FakeFactories(FakeSource(cli_records(10)))
    assert invoke(fakes, "inspect", "chroma").exit_code == 0
    fakes.target_error = httpx.LocalProtocolError(
        f"Illegal header value {LONG_KEY.encode()!r}"
    )
    r = invoke(fakes, "check", *mode, WEAVIATE_API_KEY=LONG_KEY)
    assert r.exit_code == 4
    escaped = repr(LONG_KEY.encode())[2:-1]
    for form in (LONG_KEY, escaped):
        leaked = [form[i : i + 20] for i in range(len(form) - 19)]
        assert not [part for part in leaked if part in r.output]


@pytest.mark.parametrize("mode", [[], ["--json"]])
def test_secret_with_whitespace_is_redacted_before_collapsing(mode):
    fakes = FakeFactories(FakeSource(cli_records(10)))
    assert invoke(fakes, "inspect", "chroma").exit_code == 0
    fakes.target_error = RuntimeError("headers: top\nsecret")
    r = invoke(fakes, "check", *mode, WEAVIATE_API_KEY="top\nsecret")
    assert r.exit_code == 5
    assert "top secret" not in r.output and "top\nsecret" not in r.output
    assert "secret" not in r.output


def test_url_password_is_not_stored_or_printed(tmp_path):
    fakes = FakeFactories(FakeSource(cli_records(10)))
    env = {
        "CHROMA_URL": "https://user:SENTINEL-D@chroma.example.com",
        "WEAVIATE_URL": "http://admin:SENTINEL-E@localhost:8099",
    }
    inspect_run = invoke(fakes, "inspect", "chroma", **env)
    check_run = invoke(fakes, "check", **env)
    assert inspect_run.exit_code == check_run.exit_code == 0
    assert "Inspecting Chroma at https://chroma.example.com" in inspect_run.stdout
    assert "Checking the target at http://localhost:8099" in check_run.stdout
    plan = load_plan(tmp_path / ".to-weaviate" / "plan.yaml")
    assert plan.source.url == "https://chroma.example.com"
    assert "SENTINEL" not in inspect_run.output + check_run.output


@pytest.mark.parametrize("debug", [None, "1"])
def test_malformed_url_variable_does_not_break_the_error_report(debug):
    fakes = FakeFactories(FakeSource(cli_records(10)))
    assert invoke(fakes, "inspect", "chroma").exit_code == 0
    fakes.target_error = RuntimeError("headers: SENTINEL-B")
    r = invoke(
        fakes,
        "check",
        "--json",
        WEAVIATE_GRPC_URL="http://[::1:50099",
        WEAVIATE_API_KEY="SENTINEL-B",
        TO_WEAVIATE_DEBUG=debug,
    )
    assert r.exit_code == 5
    assert json.loads(r.stdout)["code"] == "internal_error"
    assert "SENTINEL" not in r.stdout and "SENTINEL" not in r.stderr


def test_status_ignores_a_malformed_weaviate_url():
    fakes = FakeFactories(FakeSource(cli_records(10)))
    r = invoke(fakes, "status", "--json", WEAVIATE_URL="http://[::1")
    assert r.exit_code == 2
    assert json.loads(r.stdout)["code"] == "usage"


FALLBACK = {
    "status": "error",
    "code": "internal_error",
    "message": "internal error: the error could not be redacted, so its details "
    "are not printed",
    "hint": "Report it at https://github.com/weaviate/to-weaviate/issues.",
}


@pytest.mark.parametrize("mode", [[], ["--json"]])
def test_a_failing_error_report_prints_a_fixed_error(monkeypatch, mode):
    def broken(env):
        raise ValueError("redaction broke on SENTINEL-X")

    monkeypatch.setattr("to_weaviate.cli.secret_values", broken)
    fakes = FakeFactories(FakeSource(cli_records(10)))
    assert invoke(fakes, "inspect", "chroma").exit_code == 0
    fakes.target_error = RuntimeError("headers: SENTINEL-B")
    r = invoke(fakes, "check", *mode, TO_WEAVIATE_DEBUG="1")
    assert r.exit_code == 5
    if mode:
        assert json.loads(r.stdout) == FALLBACK
    else:
        assert r.stderr.splitlines() == [
            f"Error: {FALLBACK['message']}",
            f"Hint: {FALLBACK['hint']}",
        ]
    for text in ("SENTINEL", "headers", "redaction broke", "Traceback"):
        assert text not in r.output


def test_unexpected_error_gives_an_internal_error_document(tmp_path):
    fakes = FakeFactories(
        FakeSource(cli_records(10)), writer=FakeWriter(fail_on_call=1)
    )
    assert invoke(fakes, "inspect", "chroma").exit_code == 0
    r = invoke(fakes, "copy", "--json")
    assert r.exit_code == 5
    d = json.loads(r.stdout)
    assert (d["status"], d["code"]) == ("error", "internal_error")
    assert d["message"] == "internal error: RuntimeError: fake write failure on call 1"
    assert "https://github.com/weaviate/to-weaviate/issues" in d["hint"]
    assert "TO_WEAVIATE_DEBUG=1" in d["hint"]
    assert "Traceback" not in r.output
    assert (tmp_path / ".to-weaviate" / "state.sqlite").exists()


def test_unexpected_error_in_text_mode():
    fakes = FakeFactories(
        FakeSource(cli_records(10)), writer=FakeWriter(fail_on_call=1)
    )
    assert invoke(fakes, "inspect", "chroma").exit_code == 0
    r = invoke(fakes, "copy")
    assert r.exit_code == 5
    assert "Error: internal error: RuntimeError" in r.stderr
    assert "Traceback" not in r.output


def test_debug_prints_the_traceback_on_stderr():
    fakes = FakeFactories(
        FakeSource(cli_records(10)), writer=FakeWriter(fail_on_call=1)
    )
    assert invoke(fakes, "inspect", "chroma").exit_code == 0
    r = invoke(fakes, "copy", "--json", TO_WEAVIATE_DEBUG="1")
    assert r.exit_code == 5
    assert "Traceback (most recent call last)" in r.stderr
    assert "fake write failure on call 1" in r.stderr
    assert json.loads(r.stdout)["code"] == "internal_error"


def test_other_weaviate_errors_are_target_unreachable():
    error = UnexpectedStatusCodeError("Create collection", httpx.Response(500))
    fakes = FakeFactories(
        FakeSource(cli_records(10)), schema=FakeSchema(ensure_error=error)
    )
    assert invoke(fakes, "inspect", "chroma").exit_code == 0
    r = invoke(fakes, "copy", "--json")
    assert r.exit_code == 4
    d = json.loads(r.stdout)
    assert d["code"] == "target_unreachable"
    assert "UnexpectedStatusCodeError" in d["message"] and "500" in d["message"]


def test_click_errors_and_version_keep_their_behavior():
    fakes = FakeFactories(FakeSource(cli_records(10)))
    r = invoke(fakes, "copy", "--parallelism", "many", "--json")
    assert r.exit_code == 2 and r.stdout == "" and "Invalid value" in r.stderr
    r = invoke(fakes, "--version")
    assert r.exit_code == 0 and r.stdout.startswith("to-weaviate ")


def test_auth_error_does_not_repeat_the_server_text():
    body = {"code": 401, "message": "invalid api key SENTINEL-B"}
    fakes = FakeFactories(FakeSource(cli_records(10)))
    assert invoke(fakes, "inspect", "chroma").exit_code == 0
    fakes.target_error = UnexpectedStatusCodeError(
        "Meta endpoint", httpx.Response(401, json=body)
    )
    r = invoke(fakes, "check", WEAVIATE_API_KEY="SENTINEL-B")
    assert r.exit_code == 3 and "SENTINEL" not in r.output


@pytest.mark.parametrize(
    ("error", "code", "named"),
    [
        (ChromaError(401, "bad key"), "auth_failed", "CHROMA_API_KEY"),
        (ChromaError(403, "forbidden"), "auth_failed", "CHROMA_API_KEY"),
        (ChromaError(404, "no such tenant"), "source_unreachable", "CHROMA_URL"),
        (
            ChromaError(None, "ConnectError: refused"),
            "source_unreachable",
            "CHROMA_URL",
        ),
    ],
)
def test_source_errors_map_to_codes(error, code, named):
    fakes = FakeFactories(FakeSource(cli_records(10)), source_error=error)
    r = invoke(fakes, "inspect", "chroma", "--json")
    assert r.exit_code == ERROR_EXIT[code]
    d = json.loads(r.stdout)
    assert (d["status"], d["code"]) == ("error", code) and named in d["hint"]


@pytest.mark.parametrize(
    ("error", "code", "named"),
    [
        (
            UnexpectedStatusCodeError("Meta endpoint", httpx.Response(401)),
            "auth_failed",
            "WEAVIATE_API_KEY",
        ),
        (
            UnexpectedStatusCodeError("Meta endpoint", httpx.Response(403)),
            "auth_failed",
            "WEAVIATE_API_KEY",
        ),
        (
            InsufficientPermissionsError(httpx.Response(403)),
            "auth_failed",
            "WEAVIATE_API_KEY",
        ),
        (AuthenticationFailedError("no"), "auth_failed", "WEAVIATE_API_KEY"),
        (WeaviateStartUpError("down"), "target_unreachable", "WEAVIATE_URL"),
        (WeaviateConnectionError("down"), "target_unreachable", "WEAVIATE_URL"),
        (WeaviateTimeoutError("slow"), "target_unreachable", "WEAVIATE_URL"),
        (WeaviateGRPCUnavailableError(), "target_unreachable", "WEAVIATE_URL"),
        (httpx.ConnectTimeout("timed out"), "target_unreachable", "WEAVIATE_URL"),
        (
            httpx.LocalProtocolError("Illegal header value b'x'"),
            "target_unreachable",
            "WEAVIATE_URL",
        ),
    ],
)
@pytest.mark.parametrize("cmd", ["check", "copy", "verify"])
def test_target_errors_map_to_codes(error, code, named, cmd):
    fakes = FakeFactories(FakeSource(cli_records(10)))
    assert invoke(fakes, "inspect", "chroma").exit_code == 0
    fakes.target_error = error
    r = invoke(fakes, cmd, "--json")
    assert r.exit_code == ERROR_EXIT[code]
    d = json.loads(r.stdout)
    assert (d["status"], d["code"]) == ("error", code) and named in d["hint"]
    assert "\n" not in d["message"]


def test_copy_stops_when_the_collection_exists_with_other_settings():
    difference = "vector default: distance is dot, plan has cosine"
    fakes = FakeFactories(
        FakeSource(cli_records(10)),
        schema=FakeSchema(existing={"Docs": [difference]}),
    )
    assert invoke(fakes, "inspect", "chroma").exit_code == 0
    r = invoke(fakes, "copy", "--json")
    assert r.exit_code == 2
    d = json.loads(r.stdout)
    assert d["code"] == "plan_mismatch" and difference in d["message"]
    assert "exists" in d["hint"] and fakes.source.read_calls == 0


def test_copy_refuses_an_invalid_vectorizer_before_connecting():
    fakes = FakeFactories(FakeSource(cli_records(10)))
    args = ["--vectors", "embed", "--vectorizer", "text2vec-openai"]
    args += ["--vectorizer-option", "colour=blue"]
    assert invoke(fakes, "inspect", "chroma", *args).exit_code == 0
    fakes.target_error = WeaviateStartUpError("down")
    r = invoke(fakes, "copy", "--json")
    assert r.exit_code == 2
    d = json.loads(r.stdout)
    assert d["code"] == "invalid_plan"
    assert [e["code"] for e in d["errors"]] == ["invalid_vectorizer"]
    assert "colour" in d["errors"][0]["message"]


@pytest.mark.parametrize("cmd", ["check", "copy", "status", "verify"])
def test_missing_plan_is_a_usage_error(cmd):
    r = invoke(FakeFactories(FakeSource(cli_records(10))), cmd, "--json")
    assert r.exit_code == 2
    d = json.loads(r.stdout)
    assert d["code"] == "usage" and "to-weaviate inspect chroma" in d["hint"]


@pytest.mark.parametrize("content", ["collections: [", "version: 1\n", ""])
@pytest.mark.parametrize("cmd", ["check", "copy", "status", "verify"])
def test_invalid_plan_is_reported(tmp_path, content, cmd):
    fakes = FakeFactories(FakeSource(cli_records(10)))
    assert invoke(fakes, "inspect", "chroma").exit_code == 0
    (tmp_path / ".to-weaviate" / "plan.yaml").write_text(content)
    r = invoke(fakes, cmd, "--json")
    assert r.exit_code == 2
    assert json.loads(r.stdout)["code"] == "invalid_plan"


def test_copy_without_inspect_json_is_a_usage_error(tmp_path):
    fakes = FakeFactories(FakeSource(cli_records(10)))
    assert invoke(fakes, "inspect", "chroma").exit_code == 0
    (tmp_path / ".to-weaviate" / "inspect.json").unlink()
    r = invoke(fakes, "copy", "--json")
    assert r.exit_code == 2
    d = json.loads(r.stdout)
    assert d["code"] == "usage" and "to-weaviate inspect chroma" in d["hint"]


def test_copy_needs_an_inspected_collection_for_every_entry(tmp_path):
    fakes = FakeFactories(FakeSource(cli_records(10)))
    assert invoke(fakes, "inspect", "chroma").exit_code == 0
    (tmp_path / ".to-weaviate" / "inspect.json").write_text('{"collections": []}')
    r = invoke(fakes, "copy", "--json")
    assert r.exit_code == 2
    d = json.loads(r.stdout)
    assert d["code"] == "usage" and "inspect" in d["hint"]


def test_check_without_weaviate_url_names_it():
    fakes = FakeFactories(FakeSource(cli_records(10)))
    assert invoke(fakes, "inspect", "chroma").exit_code == 0
    r = invoke(fakes, "check", "--json", WEAVIATE_URL=None)
    assert r.exit_code == 3
    d = json.loads(r.stdout)
    assert d["code"] == "missing_credential" and "WEAVIATE_URL" in d["hint"]


def test_cli_error_document_and_exit_code():
    error = CliError("rejects", "2 records were rejected", "See rejects.jsonl.")
    assert error.exit_code == 1
    assert error.document() == {
        "status": "error",
        "code": "rejects",
        "message": "2 records were rejected",
        "hint": "See rejects.jsonl.",
    }


def test_error_exit_covers_every_code():
    assert ERROR_EXIT == {
        "rejects": 1,
        "verification_failed": 1,
        "prune_limit": 1,
        "usage": 2,
        "invalid_plan": 2,
        "plan_mismatch": 2,
        "target_rejected_schema": 2,
        "state_mismatch": 2,
        "missing_credential": 3,
        "auth_failed": 3,
        "source_unreachable": 4,
        "target_unreachable": 4,
        "internal_error": 5,
    }
