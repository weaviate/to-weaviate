import errno
import json
import os
import sqlite3
from collections.abc import Iterator
from typing import Any

import httpx
import pytest
import typer
from fakes import (
    FakeFactories,
    FakeInsertClient,
    FakeSchema,
    FakeSource,
    FakeWriter,
    cli_env,
    cli_records,
    invoke,
    raise_grpc_failure,
)
from typer.testing import CliRunner
from weaviate.exceptions import (
    AuthenticationFailedError,
    InsufficientPermissionsError,
    UnexpectedStatusCodeError,
    WeaviateBatchError,
    WeaviateConnectionError,
    WeaviateGRPCUnavailableError,
    WeaviateRetryError,
    WeaviateStartUpError,
    WeaviateTimeoutError,
)

from to_weaviate import cli
from to_weaviate.cli import app
from to_weaviate.output import ERROR_EXIT, CliError
from to_weaviate.plan import load_plan, save_plan
from to_weaviate.sources.base import SourceNotCompacted
from to_weaviate.sources.chroma.client import ChromaClient, ChromaError
from to_weaviate.target.schema import collection_kwargs
from to_weaviate.target.writer import WeaviateWriter

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


# A made-up WEAVIATE_API_KEY that is also the name of the source and the target.
KEY_AS_NAME = "MADE_UP_KEY_42"


def key_named_fakes() -> FakeFactories:
    source = FakeSource(cli_records(3))
    source.COLLECTION = KEY_AS_NAME
    return FakeFactories(source)


@pytest.mark.parametrize(
    "cmd",
    [["inspect", "chroma", "--force"], ["check"], ["copy"], ["status"], ["verify"]],
    ids=lambda cmd: cmd[0],
)
def test_secret_in_a_success_document_is_redacted(cmd):
    fakes = key_named_fakes()
    assert invoke(fakes, "inspect", "chroma").exit_code == 0
    assert invoke(fakes, "copy").exit_code == 0
    # check warns about each planned property that the collection lacks.
    fakes.schema.properties[KEY_AS_NAME] = set()
    r = invoke(fakes, *cmd, "--json", WEAVIATE_API_KEY=KEY_AS_NAME)
    assert r.exit_code == 0, r.output
    assert json.loads(r.stdout)["status"] == "success"
    assert KEY_AS_NAME not in r.stdout and "***" in r.stdout


def test_secret_in_a_success_text_is_redacted():
    r = invoke(key_named_fakes(), "inspect", "chroma", WEAVIATE_API_KEY=KEY_AS_NAME)
    assert r.exit_code == 0, r.output
    assert KEY_AS_NAME not in r.stdout and "*** -> ***" in r.stdout


@pytest.mark.parametrize("mode", [["--json"], []], ids=["json", "text"])
def test_secret_in_a_progress_line_is_redacted(mode):
    fakes = key_named_fakes()
    sq = ["--index", "hnsw", "--quantizer", "sq"]
    assert invoke(fakes, "inspect", "chroma", *sq).exit_code == 0
    # copy adds each planned property that the collection lacks.
    fakes.schema.properties[KEY_AS_NAME] = set()
    r = invoke(fakes, "copy", *mode, WEAVIATE_API_KEY=KEY_AS_NAME)
    assert r.exit_code == 0, r.output
    assert KEY_AS_NAME not in r.stdout and KEY_AS_NAME not in r.stderr
    progress = (r.stderr if mode else r.stdout).splitlines()
    for line in (
        "Added the property document to ***",
        "Copying *** to ***",
        "Enabling the trained quantizers of ***",
        "Copied ***: 3 records, 0 rejects",
    ):
        assert line in progress, progress


def test_secret_in_the_source_changed_warning_is_redacted():
    fakes = key_named_fakes()
    source = fakes.source
    assert invoke(fakes, "inspect", "chroma").exit_code == 0
    assert invoke(fakes, "copy", "--parallelism", "1").exit_code == 0
    # id-2, the last record read, is deleted; id-3 and id-4 are added.
    source.records = source.records[:2] + cli_records(5)[3:]
    r = invoke(fakes, "copy", "--json", WEAVIATE_API_KEY=KEY_AS_NAME)
    assert r.exit_code == 0, r.output
    assert json.loads(r.stdout)["collections"][0]["reread_from"] == 0
    assert KEY_AS_NAME not in r.stdout and KEY_AS_NAME not in r.stderr
    assert (
        "Warning: ***: the source changed since the last run; re-reading from the start"
    ) in r.stderr.splitlines()


def test_secret_in_a_progress_line_of_verify_is_redacted():
    fakes = key_named_fakes()
    assert invoke(fakes, "inspect", "chroma").exit_code == 0
    assert invoke(fakes, "copy").exit_code == 0
    fakes.source.waits = [(5, 10)]
    fakes.source.compaction = [{"version": 2, "log_position": 10}]
    r = invoke(fakes, "verify", "--json", WEAVIATE_API_KEY=KEY_AS_NAME)
    assert r.exit_code == 0, r.output
    assert KEY_AS_NAME not in r.stdout and KEY_AS_NAME not in r.stderr
    progress = r.stderr.splitlines()
    assert "Verifying *** in ***" in progress, progress
    assert (
        "Waiting for Chroma to compact ***: 5 of 10 operations not compacted"
        in progress
    ), progress


LONG_KEY = (
    "".join(chr(ord("a") + i * 7 % 26) for i in range(230))
    + "é"
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
    fakes.target_error = RuntimeError("headers: top  secret")
    r = invoke(fakes, "check", *mode, WEAVIATE_API_KEY="top  secret")
    assert r.exit_code == 5
    assert "top secret" not in r.output and "top\nsecret" not in r.output
    assert "secret" not in r.output


@pytest.mark.parametrize("mode", [[], ["--json"]])
@pytest.mark.parametrize(
    ("cmd", "var"),
    [
        (["check"], "WEAVIATE_API_KEY"),
        (["check"], "OPENAI_API_KEY"),
        (["inspect", "chroma", "--force"], "CHROMA_API_KEY"),
    ],
)
def test_key_with_a_control_character_is_a_usage_error(cmd, var, mode):
    key = "B5-MADE-UP\nKEY-VALUE"
    fakes = FakeFactories(FakeSource(cli_records(10)))
    assert invoke(fakes, "inspect", "chroma").exit_code == 0
    fakes.source_error = ChromaError(None, f"Illegal header value {key.encode()!r}")
    fakes.target_error = httpx.LocalProtocolError(f"Illegal header {key.encode()!r}")
    r = invoke(fakes, *cmd, *mode, TO_WEAVIATE_DEBUG="1", **{var: key})
    assert r.exit_code == 2, r.output
    message = f"{var} holds a control character."
    if mode:
        d = json.loads(r.stdout)
        assert (d["code"], d["message"]) == ("usage", message)
    else:
        assert f"Error: {message}" in r.stderr
    for part in ("B5-MADE", "UP\\nKEY", "KEY-VALUE"):
        assert part not in r.output


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


def test_embedding_config_secrets_are_not_stored_or_sent(tmp_path):
    password, header_token, query_token = (
        "F7-MADE-UP-PASSWORD",
        "F7-MADE-UP-HEADER-TOKEN",
        "F7-MADE-UP-QUERY-TOKEN",
    )
    embedding = {
        "type": "known",
        "name": "openai",
        "config": {
            "api_key_env_var": "OPENAI_API_KEY",
            "model_name": "text-embedding-3-small",
            "api_base": f"https://bob:{password}@proxy.example.com/v1"
            f"?token={query_token}#{query_token}",
            "default_headers": {"Authorization": f"Bearer {header_token}"},
        },
    }
    fakes = FakeFactories(FakeSource(cli_records(10), embedding=embedding))
    runs = [
        invoke(fakes, "inspect", "chroma"),
        invoke(fakes, "inspect", "chroma", "--force", "--json"),
    ]
    assert [r.exit_code for r in runs] == [0, 0], runs[0].output
    directory = tmp_path / ".to-weaviate"
    plan = load_plan(directory / "plan.yaml")
    sent = repr(collection_kwargs(plan.collections))
    texts = [(directory / name).read_text() for name in ("plan.yaml", "inspect.json")]
    for text in [*texts, *(r.output for r in runs), sent]:
        for value in (password, header_token, query_token, "bob"):
            assert value not in text
    entry = plan.collections[0]
    assert entry.source.embedding == {
        "type": "known",
        "name": "openai",
        "config": {
            "model_name": "text-embedding-3-small",
            "api_base": "https://proxy.example.com/v1",
        },
    }
    vectorizer = entry.target.vectors[0].vectorizer
    assert vectorizer is not None
    assert vectorizer.options == {
        "model": "text-embedding-3-small",
        "base_url": "https://proxy.example.com",
    }
    assert [n for n in entry.notes if "api_key_env_var, default_headers" in n]


@pytest.mark.parametrize("char", "/?#")
def test_embedding_url_that_keeps_its_userinfo_is_not_stored_or_sent(tmp_path, char):
    password = f"B5-MADE{char}UP-PW"
    embedding = {
        "type": "known",
        "name": "openai",
        "config": {
            "model_name": "text-embedding-3-small",
            "api_base": f"https://bob:{password}@proxy.example.com/v1",
        },
    }
    fakes = FakeFactories(FakeSource(cli_records(10), embedding=embedding))
    runs = [
        invoke(fakes, "inspect", "chroma"),
        invoke(fakes, "inspect", "chroma", "--force", "--json"),
    ]
    assert [r.exit_code for r in runs] == [0, 0], runs[0].output
    directory = tmp_path / ".to-weaviate"
    plan = load_plan(directory / "plan.yaml")
    sent = repr(collection_kwargs(plan.collections))
    texts = [(directory / name).read_text() for name in ("plan.yaml", "inspect.json")]
    for text in [*texts, *(r.output for r in runs), sent]:
        for value in ("@", "bob", password, *password.split(char)):
            assert value not in text
    entry = plan.collections[0]
    assert entry.source.embedding == {
        "type": "known",
        "name": "openai",
        "config": {"model_name": "text-embedding-3-small"},
    }
    assert [n for n in entry.notes if "api_base holds userinfo" in n]


USERINFO_HINT = "Percent-encode the password in plan.yaml or remove the option."


def plan_with_url_option(tmp_path, fakes, password):
    """Inspect with a text2vec-openai base_url that holds `password`."""
    embed = ["--vectors", "embed", "--vectorizer", "text2vec-openai"]
    option = f"base_url=https://bob:{password}@proxy.example.com/v1"
    args = [*embed, "--vectorizer-option", option]
    assert invoke(fakes, "inspect", "chroma", *args).exit_code == 0


@pytest.mark.parametrize("mode", [[], ["--json"]])
@pytest.mark.parametrize("cmd", ["check", "copy"])
@pytest.mark.parametrize("char", "/?#")
def test_check_and_copy_refuse_a_url_option_that_keeps_its_userinfo(
    tmp_path, char, cmd, mode
):
    password = f"B5-MADE{char}UP-PW"
    fakes = FakeFactories(FakeSource(cli_records(10)))
    plan_with_url_option(tmp_path, fakes, password)
    r = invoke(fakes, cmd, *mode, OPENAI_API_KEY="openai-key")
    assert r.exit_code == 2, r.output
    message = (
        "vectorizer option base_url of text2vec-openai in Docs holds userinfo "
        "that cannot be removed"
    )
    if mode:
        d = json.loads(r.stdout)
        assert (d["code"], d["hint"]) == ("invalid_plan", USERINFO_HINT)
        assert d["errors"] == [
            {"code": "invalid_vectorizer", "collection": "Docs", "message": message}
        ]
        assert d["warnings"] == []
    else:
        assert f"error invalid_vectorizer Docs: {message}" in r.stdout
        assert f"Hint: {USERINFO_HINT}" in r.stderr
    assert fakes.schema.ensured == [] and fakes.writer.calls == 0
    assert fakes.source.read_calls == 0
    if cmd == "copy":
        assert fakes.schema_calls == 0
    for value in ("bob", password, *password.split(char)):
        assert value not in r.output


def test_plan_errors_of_other_kinds_keep_the_plan_hint(tmp_path):
    fakes = FakeFactories(FakeSource(cli_records(10)))
    plan_with_url_option(tmp_path, fakes, "B5-MADE/UP-PW")
    path = tmp_path / ".to-weaviate" / "plan.yaml"
    plan = load_plan(path)
    first = plan.collections[0]
    twin = first.model_copy(
        update={"target": first.target.model_copy(update={"name": "Docs2"})}
    )
    save_plan(plan.model_copy(update={"collections": [first, twin]}), path)
    r = invoke(fakes, "check", "--json", OPENAI_API_KEY="openai-key")
    assert r.exit_code == 2, r.output
    d = json.loads(r.stdout)
    assert {e["code"] for e in d["errors"]} == {
        "duplicate_source",
        "invalid_vectorizer",
    }
    assert d["hint"] == (
        "Edit plan.yaml, or run to-weaviate inspect chroma --force to draft it again."
    )


def test_source_url_query_and_fragment_are_not_stored_or_printed(tmp_path):
    fakes = FakeFactories(FakeSource(cli_records(10)))
    url = "https://user:SENTINEL-D@chroma.example.com/?token=SENTINEL-Q"
    runs = [
        invoke(fakes, "inspect", "chroma", CHROMA_URL=url),
        invoke(fakes, "inspect", "chroma", "--force", "--json", CHROMA_URL=url),
    ]
    assert [r.exit_code for r in runs] == [0, 0], runs[0].output
    assert "Inspecting Chroma at https://chroma.example.com/\n" in runs[0].output
    plan = load_plan(tmp_path / ".to-weaviate" / "plan.yaml")
    assert plan.source.url == "https://chroma.example.com/"
    for f in tmp_path.rglob("*"):
        if f.is_file():
            assert b"SENTINEL" not in f.read_bytes()
    assert "SENTINEL" not in runs[0].output + runs[1].output


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
    assert r.exit_code == 2
    assert json.loads(r.stdout)["code"] == "usage"
    assert "SENTINEL" not in r.stdout and "SENTINEL" not in r.stderr


URL_HINT = "Percent-encode #, ?, / and @ in a password."
# (variable, value, sentinel): a password with an unencoded #, ? or /, and a
# source URL with a fragment.
REFUSED_URLS = [
    *[
        (var, f"http://admin:hunter2{char}x@localhost:{port}", "hunter2")
        for var, port in (
            ("WEAVIATE_URL", 8099),
            ("WEAVIATE_GRPC_URL", 50099),
            ("CHROMA_URL", 8001),
        )
        for char in "#?/"
    ],
    (
        "CHROMA_URL",
        "https://user:SENTINEL-D@chroma.example.com/?token=SENTINEL-Q#SENTINEL-F",
        "SENTINEL",
    ),
]


@pytest.mark.parametrize("debug", [None, "1"])
@pytest.mark.parametrize(("var", "url", "sentinel"), REFUSED_URLS)
def test_refused_url_variable_is_a_usage_error_without_its_value(
    tmp_path, var, url, sentinel, debug
):
    fakes = FakeFactories(FakeSource(cli_records(10)))
    assert invoke(fakes, "inspect", "chroma").exit_code == 0
    cmd = ["inspect", "chroma", "--force"] if var == "CHROMA_URL" else ["check"]
    js = invoke(fakes, *cmd, "--json", TO_WEAVIATE_DEBUG=debug, **{var: url})
    text = invoke(fakes, *cmd, TO_WEAVIATE_DEBUG=debug, **{var: url})
    assert js.exit_code == text.exit_code == 2, js.output
    d = json.loads(js.stdout)
    assert (d["code"], d["message"], d["hint"]) == (
        "usage",
        f"{var} is not a valid URL.",
        URL_HINT,
    )
    assert f"Error: {var} is not a valid URL." in text.stderr
    for r in (js, text):
        assert sentinel not in r.stdout and sentinel not in r.stderr
    for f in tmp_path.rglob("*"):
        if f.is_file():
            assert sentinel.encode() not in f.read_bytes()


NO_HOST_HINT = "Write it as scheme://host, with an optional :port."


@pytest.mark.parametrize(
    ("var", "cmd"),
    [
        ("WEAVIATE_URL", ["check"]),
        ("CHROMA_URL", ["inspect", "chroma", "--force"]),
    ],
)
def test_a_url_variable_without_a_host_is_a_usage_error(var, cmd):
    fakes = FakeFactories(FakeSource(cli_records(10)))
    assert invoke(fakes, "inspect", "chroma").exit_code == 0
    fakes.source_calls = 0
    js = invoke(fakes, *cmd, "--json", **{var: "http://"})
    text = invoke(fakes, *cmd, **{var: "http://"})
    assert js.exit_code == text.exit_code == 2, js.output
    d = json.loads(js.stdout)
    assert (d["code"], d["message"], d["hint"]) == (
        "usage",
        f"{var} is not a valid URL.",
        NO_HOST_HINT,
    )
    assert f"Error: {var} is not a valid URL." in text.stderr
    assert fakes.source_calls == 0 and fakes.schema_calls == 0


@pytest.mark.parametrize(
    ("cmd", "var", "error"),
    [
        (
            ["check"],
            "WEAVIATE_URL",
            WeaviateConnectionError(
                "no answer from https://MADE-UP-TOKEN@weaviate.test"
            ),
        ),
        (
            ["inspect", "chroma", "--force"],
            "CHROMA_URL",
            ChromaError(None, "ConnectError: https://MADE-UP-TOKEN@chroma.example.com"),
        ),
    ],
)
def test_url_userinfo_without_a_colon_is_not_printed(cmd, var, error):
    fakes = FakeFactories(FakeSource(cli_records(10)))
    assert invoke(fakes, "inspect", "chroma").exit_code == 0
    fakes.source_error = fakes.target_error = error
    host = "weaviate.test" if var == "WEAVIATE_URL" else "chroma.example.com"
    r = invoke(fakes, *cmd, "--json", **{var: f"https://MADE-UP-TOKEN@{host}"})
    assert r.exit_code == 4, r.output
    assert "MADE-UP-TOKEN" not in r.output and "***" in r.output


@pytest.mark.parametrize(
    "url", ["https://api.trychroma.com.", "https://API.TryChroma.com."]
)
def test_chroma_cloud_url_with_a_trailing_dot_needs_the_key(url):
    fakes = FakeFactories(FakeSource(cli_records(10)))
    r = invoke(
        fakes, "inspect", "chroma", "--json", CHROMA_URL=url, CHROMA_API_KEY=None
    )
    assert r.exit_code == 3, r.output
    d = json.loads(r.stdout)
    assert d["code"] == "missing_credential" and "CHROMA_API_KEY" in d["hint"]


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

    fakes = FakeFactories(FakeSource(cli_records(10)))
    assert invoke(fakes, "inspect", "chroma").exit_code == 0
    monkeypatch.setattr("to_weaviate.cli.secret_values", broken)
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


def test_copy_retries_a_retry_error_and_then_reports_target_unreachable():
    client = FakeInsertClient(lambda n: WeaviateRetryError("UNAVAILABLE", 3))
    writer = WeaviateWriter(client, sleep=lambda s: None)  # type: ignore[arg-type]
    fakes = FakeFactories(FakeSource(cli_records(10)), writer=writer, store=client)
    assert invoke(fakes, "inspect", "chroma").exit_code == 0
    r = invoke(fakes, "copy", "--parallelism", "1", "--json")
    assert r.exit_code == 4
    d = json.loads(r.stdout)
    assert d["code"] == "target_unreachable" and "WeaviateRetryError" in d["message"]
    assert client.calls == 7


def test_click_errors_and_version_keep_their_behavior():
    fakes = FakeFactories(FakeSource(cli_records(10)))
    r = invoke(fakes, "copy", "--parallelism", "many", "--json")
    assert r.exit_code == 2 and r.stdout == "" and "Invalid value" in r.stderr
    r = invoke(fakes, "--version")
    assert r.exit_code == 0 and r.stdout.startswith("to-weaviate ")


def batch_error(status: str, message: str | None = None) -> WeaviateBatchError:
    """What a gRPC batch call raises when it fails with `status`."""
    try:
        raise_grpc_failure(
            status, lambda m: WeaviateBatchError(message or m), batch=True
        )
    except WeaviateBatchError as exc:
        return exc


def test_auth_error_does_not_repeat_the_server_text():
    body = {"code": 401, "message": "invalid api key SENTINEL-B"}
    fakes = FakeFactories(FakeSource(cli_records(10)))
    assert invoke(fakes, "inspect", "chroma").exit_code == 0
    fakes.target_error = UnexpectedStatusCodeError(
        "Meta endpoint", httpx.Response(401, json=body)
    )
    r = invoke(fakes, "check", WEAVIATE_API_KEY="SENTINEL-B")
    assert r.exit_code == 3 and "SENTINEL" not in r.output


def test_auth_error_of_a_batch_does_not_repeat_the_server_text():
    fakes = FakeFactories(FakeSource(cli_records(10)))
    assert invoke(fakes, "inspect", "chroma").exit_code == 0
    fakes.target_error = batch_error(
        "UNAUTHENTICATED", "extract auth: unauthorized: invalid api key SENTINEL-B"
    )
    r = invoke(fakes, "copy", "--json", WEAVIATE_API_KEY="SENTINEL-B")
    assert r.exit_code == 3 and "SENTINEL" not in r.output
    d = json.loads(r.stdout)
    assert d["message"] == (
        "Weaviate refused the request: the API key is missing or wrong."
    )


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
    ("response", "named", "hint"),
    [
        (
            httpx.Response(301, headers={"Location": "https://chroma.example/v2"}),
            "HTTP 301: a redirect to https://chroma.example",
            "Set CHROMA_URL to the address the server redirects to.",
        ),
        (
            httpx.Response(200, html="<html><body>Sign in</body></html>"),
            "HTTP 200: the answer is not JSON (Content-Type: text/html",
            "Check that CHROMA_URL points at the Chroma API and not at a proxy or "
            "a web page.",
        ),
    ],
    ids=["redirect", "html"],
)
def test_chroma_answer_that_is_not_json_is_source_unreachable(
    monkeypatch, response, named, hint
):
    def client(settings):
        transport = httpx.MockTransport(lambda request: response)
        return ChromaClient(settings, transport=transport)

    monkeypatch.setattr(cli, "ChromaClient", client)
    r = CliRunner().invoke(app, ["inspect", "chroma", "--json"], env=cli_env())
    assert r.exit_code == 4, r.output
    d = json.loads(r.stdout)
    assert (d["status"], d["code"], d["hint"]) == ("error", "source_unreachable", hint)
    assert named in d["message"]


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
        (batch_error("UNAUTHENTICATED"), "auth_failed", "WEAVIATE_API_KEY"),
        (batch_error("PERMISSION_DENIED"), "auth_failed", "WEAVIATE_API_KEY"),
        (batch_error("INTERNAL"), "target_unreachable", "WEAVIATE_URL"),
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


def test_check_refuses_a_weaviate_port_without_a_grpc_url():
    fakes = FakeFactories(FakeSource(cli_records(10)))
    url = "http://admin:B7-MADE-UP-PW@localhost:8099"
    assert invoke(fakes, "inspect", "chroma").exit_code == 0
    r = invoke(fakes, "check", "--json", WEAVIATE_URL=url, WEAVIATE_GRPC_URL=None)
    assert r.exit_code == 2, r.output
    assert json.loads(r.stdout) == {
        "status": "error",
        "code": "usage",
        "message": "WEAVIATE_URL names port 8099, but without WEAVIATE_GRPC_URL "
        "the client connects to port 443.",
        "hint": "Set WEAVIATE_GRPC_URL for a Weaviate that is not on Weaviate Cloud.",
    }
    assert "localhost" not in r.output and "B7-MADE-UP" not in r.output
    grpc = "http://localhost:50099"
    r = invoke(fakes, "check", "--json", WEAVIATE_URL=url, WEAVIATE_GRPC_URL=grpc)
    assert r.exit_code == 0, r.output


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
        "invalid_state": 2,
        "directory_unwritable": 2,
        "missing_credential": 3,
        "auth_failed": 3,
        "source_unreachable": 4,
        "target_unreachable": 4,
        "source_compacting": 4,
        "source_writing": 4,
        "delete_failed": 4,
        "foreign_objects": 2,
        "internal_error": 5,
    }


def test_a_compaction_during_copy_is_source_compacting_exit_4():
    fakes = FakeFactories(FakeSource(cli_records(10)))
    assert invoke(fakes, "inspect", "chroma").exit_code == 0
    fakes.source.compaction = [
        {"version": 0, "log_position": 0},
        {"version": 1, "log_position": 10000},
    ]
    r = invoke(fakes, "copy", "--json")
    assert r.exit_code == 4, r.output
    assert json.loads(r.stdout) == {
        "status": "error",
        "code": "source_compacting",
        "message": "Chroma compacted the collection docs while it was read "
        "(version 0 to 1, log_position 0 to 10000), so a page can have skipped "
        "records. Run the command again.",
        "hint": "Stop the writes to the collection, then run the command again.",
        "collection": "docs",
    }


def test_a_collection_that_chroma_does_not_compact_is_source_compacting_exit_4():
    fakes = FakeFactories(FakeSource(cli_records(10)))
    assert invoke(fakes, "inspect", "chroma").exit_code == 0
    fakes.source.wait_error = SourceNotCompacted("docs", 1200, 40000, 600.0)
    r = invoke(fakes, "copy", "--json")
    assert r.exit_code == 4, r.output
    d = json.loads(r.stdout)
    assert (d["code"], d["message"], d["hint"]) == (
        "source_compacting",
        "Chroma has not compacted every write to the collection docs after 600 s: "
        "1200 of 40000 operations are not compacted. Run the command again.",
        "Stop the writes to the collection and run the command again once Chroma "
        "has compacted them.",
    )


UNWRITABLE_HINT = "Make --dir and its files writable, or use another --dir."
DISK_FULL_HINT = "Free space on the disk of --dir, or use another --dir."


@pytest.mark.parametrize(
    ("code", "hint"),
    [
        (errno.ENOSPC, DISK_FULL_HINT),
        (errno.EDQUOT, DISK_FULL_HINT),
        (errno.EACCES, UNWRITABLE_HINT),
        (errno.EPERM, UNWRITABLE_HINT),
        (errno.EROFS, UNWRITABLE_HINT),
    ],
)
def test_a_file_that_the_disk_or_its_permissions_refuse_is_directory_unwritable(
    code, hint
):
    error = cli._as_cli_error(OSError(code, os.strerror(code), "mig/rejects.jsonl"))
    assert (error.code, error.message, error.hint) == (
        "directory_unwritable",
        f"rejects.jsonl in mig cannot be written: {os.strerror(code)}.",
        hint,
    )
    assert error.exit_code == 2


def test_a_disk_error_without_a_file_names_the_migration_directory():
    error = cli._as_cli_error(OSError(errno.ENOSPC, "No space left on device"))
    assert (error.code, error.message) == (
        "directory_unwritable",
        "The migration directory cannot be written: No space left on device.",
    )


@pytest.mark.parametrize(
    "error",
    [
        ConnectionRefusedError(errno.ECONNREFUSED, "refused"),
        FileNotFoundError(errno.ENOENT, "gone", "mig/plan.yaml"),
        sqlite3.OperationalError("no such table: partitions"),
    ],
)
def test_other_os_and_sqlite_errors_stay_internal_errors(error):
    assert cli._as_cli_error(error).code == "internal_error"
