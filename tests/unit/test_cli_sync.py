import json
import re
import shutil
from dataclasses import replace

import pytest
from fakes import (
    FakeFactories,
    FakeSource,
    FakeWriter,
    all_failed,
    cli_records,
    invoke,
)
from test_cli_commands import (
    DIR,
    LOCK_HINT,
    LOCK_MESSAGE,
    PROVIDER_429,
    REFUSED,
    _set_index,
    edit_target,
    error_of,
    held_lock,
    inspected,
    refusing,
    two_entries,
)
from typer.testing import Result

from to_weaviate import cli
from to_weaviate.cli import PRUNE_HINT
from to_weaviate.engine.copy import REJECTS_HINT, plan_hash
from to_weaviate.engine.sync import PLAN_CHANGED
from to_weaviate.engine.verify import COUNT_ABOVE_SOURCE, TARGET_ABOVE_SOURCE
from to_weaviate.ids import to_uuid
from to_weaviate.plan import DroppedKey, load_plan
from to_weaviate.state import State
from to_weaviate.target.deleter import DeleteFailed
from to_weaviate.target.writer import TargetObject


@pytest.fixture(autouse=True)
def in_tmp(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


def copied(fakes: FakeFactories) -> FakeFactories:
    assert invoke(fakes, "inspect", "chroma").exit_code == 0
    assert invoke(fakes, "copy").exit_code == 0
    return fakes


def change_source(source: FakeSource) -> None:
    """Change id-1, add id-10, delete id-2."""
    source.records[1] = replace(source.records[1], document="changed")
    source.records.extend(cli_records(11)[10:])
    del source.records[2]


def row_of(r: Result) -> dict:
    assert r.exit_code == 0, r.output
    d = json.loads(r.stdout)
    assert d["status"] == "success" and len(d["collections"]) == 1
    return d["collections"][0]


def counts(row: dict) -> tuple[int, int, int, int, int]:
    return tuple(row[k] for k in ("copied", "unchanged", "rejects", "stale", "deleted"))


def test_sync_writes_the_changed_records_and_reports_the_stale_ones():
    fakes = copied(FakeFactories(FakeSource(cli_records(10))))
    change_source(fakes.source)
    r = invoke(fakes, "sync", "--json")
    row = row_of(r)
    assert counts(row) == (2, 8, 0, 1, 0)
    assert (row["plan_changed"], row["prune_refused"]) == (False, False)
    assert json.loads(r.stdout)["prune"] is False
    assert fakes.writer.objects[to_uuid("id-1")].properties["document"] == "changed"
    assert to_uuid("id-10") in fakes.writer.objects
    assert to_uuid("id-2") in fakes.writer.objects and fakes.deleter.calls == 0
    r = invoke(fakes, "sync")
    assert r.exit_code == 0, r.output
    assert r.stdout.splitlines()[-3:] == [
        "docs -> Docs: copied 0, unchanged 10, rejects 0, stale 1, deleted 0",
        "Total: copied 0, unchanged 10, rejects 0, stale 1, deleted 0",
        "Run to-weaviate sync --prune to delete the 1 stale object.",
    ]


def test_sync_prune_deletes_the_stale_objects():
    fakes = copied(FakeFactories(FakeSource(cli_records(10))))
    change_source(fakes.source)
    r = invoke(fakes, "sync", "--prune", "--json")
    assert counts(row_of(r)) == (2, 8, 0, 1, 1)
    assert json.loads(r.stdout)["prune"] is True
    assert len(fakes.writer.objects) == 10
    assert to_uuid("id-2") not in fakes.writer.objects
    assert fakes.deleter.deleted == [("Docs", None, to_uuid("id-2"))]
    assert invoke(fakes, "verify", "--json").exit_code == 0
    r = invoke(fakes, "sync", "--prune")
    assert r.exit_code == 0, r.output
    assert r.stdout.splitlines()[-1] == (
        "Total: copied 0, unchanged 10, rejects 0, stale 0, deleted 0"
    )


PRUNE_HINT_TEXT = (
    "More than 10% of the objects with a source_id have no source record. "
    "Check that the records were deleted on purpose and that no other migration "
    "directory or application writes objects with a source_id to the collection. "
    "To delete them, run to-weaviate sync --prune --force."
)


def test_sync_prune_refuses_more_than_ten_percent_without_force():
    fakes = copied(FakeFactories(FakeSource(cli_records(10))))
    del fakes.source.records[:2]
    r = invoke(fakes, "sync", "--prune", "--json")
    assert error_of(r) == (
        1,
        "prune_limit",
        "Stale objects above 10% in Docs; nothing was deleted there.",
        PRUNE_HINT_TEXT,
    )
    assert PRUNE_HINT == PRUNE_HINT_TEXT
    row = json.loads(r.stdout)["collections"][0]
    assert counts(row) == (0, 8, 0, 2, 0) and row["prune_refused"] is True
    assert len(fakes.writer.objects) == 10 and fakes.deleter.calls == 0
    r = invoke(fakes, "sync", "--prune")
    assert (
        "docs -> Docs: copied 0, unchanged 8, rejects 0, stale 2, deleted 0, "
        "prune refused"
    ) in r.stdout.splitlines()
    r = invoke(fakes, "sync", "--prune", "--force", "--json")
    row = row_of(r)
    assert counts(row) == (0, 8, 0, 2, 2) and row["prune_refused"] is False
    assert len(fakes.writer.objects) == 8


def test_sync_with_rejects_and_a_refused_prune_reports_both():
    fakes = copied(FakeFactories(FakeSource(cli_records(10)), writer=FakeWriter()))
    del fakes.source.records[:2]
    fakes.writer.reject_ids = {"id-3"}
    fakes.source.records[1] = replace(fakes.source.records[1], document="changed")
    r = invoke(fakes, "sync", "--prune", "--json")
    code, name, message, hint = error_of(r)
    assert (code, name) == (1, "rejects")
    assert message == (
        "1 record rejected. Stale objects above 10% in Docs; nothing was deleted "
        "there."
    )
    assert hint.endswith(PRUNE_HINT_TEXT)


def test_sync_force_needs_prune():
    fakes = copied(FakeFactories(FakeSource(cli_records(10))))
    r = invoke(fakes, "sync", "--force", "--json")
    assert error_of(r) == (
        2,
        "usage",
        "--force needs --prune.",
        "Run to-weaviate sync --prune --force.",
    )


class ShrinkingSource(FakeSource):
    """A source that loses its last record while its ids are listed."""

    def ids(self, collection):
        yield from [r.id for r in self.records]
        self.records.pop()


@pytest.mark.parametrize(
    ("flags", "command"),
    [
        ((), "to-weaviate sync --prune"),
        (("--force",), "to-weaviate sync --prune --force"),
    ],
)
def test_sync_prune_stops_when_the_source_count_changes_during_the_listing(
    flags, command
):
    fakes = copied(FakeFactories(ShrinkingSource(cli_records(10))))
    r = invoke(fakes, "sync", "--prune", *flags, "--json")
    assert error_of(r) == (
        4,
        "source_writing",
        "docs changed while sync listed its ids: 10 records before, 10 listed, "
        "9 after.",
        f"Stop the writes to the source, then run {command} again.",
    )
    assert json.loads(r.stdout)["collection"] == "docs"
    assert len(fakes.writer.objects) == 10 and fakes.deleter.calls == 0


def test_a_sync_whose_batch_is_refused_names_sync_in_the_hint():
    fakes, _ = refusing(10, lambda n: all_failed(REFUSED))
    r = invoke(fakes, "sync", "--parallelism", "1", "--json")
    assert r.exit_code == 4
    d = json.loads(r.stdout)
    assert d["code"] == "target_unreachable"
    assert d["hint"] == "Fix the cause and run to-weaviate sync again."


def test_a_sync_that_a_busy_provider_stops_names_the_provider_in_the_hint():
    fakes, _ = refusing(10, lambda n: all_failed(PROVIDER_429))
    r = invoke(fakes, "sync", "--parallelism", "1", "--json")
    assert r.exit_code == 4
    d = json.loads(r.stdout)
    assert d["code"] == "target_unreachable"
    assert d["hint"] == (
        "Wait for the provider's rate limit to reset or for the provider to "
        "answer again, or lower --parallelism. Then run to-weaviate sync again."
    )


def test_a_failed_sync_after_a_plan_edit_keeps_the_target_guard(tmp_path):
    fakes = copied(FakeFactories(FakeSource(cli_records(10))))
    target = load_plan(tmp_path / DIR / "plan.yaml").collections[0].target
    edit_target(
        tmp_path / DIR,
        properties=[p for p in target.properties if p.name != "year"],
        dropped=[*target.dropped, DroppedKey(source_key="year", reason="not needed")],
    )
    fakes.writer.fail_on_call = fakes.writer.calls + 1
    assert invoke(fakes, "sync", "--json").exit_code == 5
    fakes.writer.fail_on_call = None
    r = invoke(fakes, "sync", "--json", WEAVIATE_URL="http://other:8099")
    assert error_of(r)[:2] == (2, "state_mismatch")


def test_a_sync_that_fails_before_its_first_page_keeps_status_and_the_target_guard():
    fakes = copied(FakeFactories(FakeSource(cli_records(10))))
    fakes.source.wait_error = RuntimeError("source down")
    r = invoke(fakes, "sync", "--json")
    assert r.exit_code == 5, r.output
    fakes.source.wait_error = None
    row = json.loads(invoke(fakes, "status", "--json").stdout)["collections"][0]
    assert (row["copied"], row["total"], row["done"]) == (10, 10, True)
    r = invoke(fakes, "sync", "--json", WEAVIATE_URL="http://other:8099")
    assert error_of(r)[:2] == (2, "state_mismatch")
    assert len(fakes.writer.objects) == 10


@pytest.mark.parametrize(
    ("flags", "command"),
    [
        ((), "to-weaviate sync --prune"),
        (("--force",), "to-weaviate sync --prune --force"),
    ],
)
def test_sync_prune_stops_when_weaviate_reports_failed_deletions(flags, command):
    fakes = copied(FakeFactories(FakeSource(cli_records(10))))
    del fakes.source.records[2]
    fakes.deleter.error = DeleteFailed("Docs", 1, 2)
    r = invoke(fakes, "sync", "--prune", *flags, "--json")
    assert error_of(r) == (
        4,
        "delete_failed",
        "Weaviate did not delete 1 of 2 objects of docs in Docs.",
        f"Run {command} again.",
    )
    assert json.loads(r.stdout)["collection"] == "docs"


def test_sync_rejects_exit_1_with_the_rejects_hint():
    fakes = copied(FakeFactories(FakeSource(cli_records(10)), writer=FakeWriter()))
    fakes.writer.reject_ids = {"id-3"}
    fakes.source.records[3] = replace(fakes.source.records[3], document="changed")
    r = invoke(fakes, "sync", "--json")
    code, name, message, hint = error_of(r)
    assert (code, name, message) == (1, "rejects", "1 record rejected.")
    assert hint == (
        f"See {DIR / 'rejects.jsonl'} for each record and the reason. {REJECTS_HINT}"
    )
    assert counts(json.loads(r.stdout)["collections"][0]) == (0, 9, 1, 0, 0)
    # The cause is fixed: the next sync writes the record.
    fakes.writer.reject_ids = set()
    assert counts(row_of(invoke(fakes, "sync", "--json"))) == (1, 9, 0, 0, 0)
    assert fakes.writer.objects[to_uuid("id-3")].properties["document"] == "changed"


def test_sync_warns_when_the_plan_entry_changed(tmp_path):
    fakes = copied(FakeFactories(FakeSource(cli_records(10))))
    target = load_plan(tmp_path / DIR / "plan.yaml").collections[0].target
    edit_target(
        tmp_path / DIR,
        properties=[p for p in target.properties if p.name != "year"],
        dropped=[*target.dropped, DroppedKey(source_key="year", reason="not needed")],
    )
    r = invoke(fakes, "sync", "--json")
    row = row_of(r)
    assert counts(row) == (10, 0, 0, 0, 0) and row["plan_changed"] is True
    assert f"Warning: docs: {PLAN_CHANGED}" in r.stderr.splitlines()
    assert all("year" not in o.properties for o in fakes.writer.objects.values())
    r = invoke(fakes, "sync", "--json")
    row = row_of(r)
    assert counts(row) == (0, 10, 0, 0, 0) and row["plan_changed"] is False
    assert "Warning" not in r.stderr


@pytest.mark.parametrize(("name", "exit_code"), [("docs", 0), ("Docs", 0), ("x", 2)])
def test_sync_selects_collections_like_copy(name, exit_code):
    fakes = copied(FakeFactories(FakeSource(cli_records(10))))
    fakes.source.records.extend(cli_records(11)[10:])
    r = invoke(fakes, "sync", "--collection", name, "--json")
    assert r.exit_code == exit_code, r.output
    if exit_code:
        assert json.loads(r.stdout)["code"] == "usage"
    assert len(fakes.writer.objects) == (11 if exit_code == 0 else 10)


def test_sync_holds_the_copy_lock(tmp_path):
    pytest.importorskip("fcntl")
    fakes = copied(FakeFactories(FakeSource(cli_records(10))))
    del fakes.source.records[0]
    with held_lock(tmp_path / DIR):
        r = invoke(fakes, "sync", "--prune", "--json")
    assert error_of(r) == (2, "usage", LOCK_MESSAGE, LOCK_HINT)
    assert len(fakes.writer.objects) == 10


def test_sync_passes_the_consistency_to_the_writer_and_the_deleter():
    fakes = copied(FakeFactories(FakeSource(cli_records(10))))
    r = invoke(fakes, "sync", "--prune", "--consistency", "QUORUM")
    assert r.exit_code == 0, r.output
    assert fakes.consistency[-1] == "QUORUM" and fakes.delete_consistency == ["QUORUM"]


def test_sync_enables_the_trained_quantizers_per_entry(tmp_path):
    fakes = inspected(FakeFactories(FakeSource(cli_records(10))))
    _set_index(
        tmp_path / DIR / "plan.yaml", {"type": "hnsw", "quantizer": {"type": "sq"}}
    )
    two_entries(tmp_path / DIR, ("Docs", "a"), ("Docs", "b"))
    r = invoke(fakes, "sync", "--json")
    assert r.exit_code == 0, r.output
    assert fakes.schema.updated == [("Docs", "a"), ("Docs", "b")]
    rows = json.loads(r.stdout)["collections"]
    assert [(row["target"], row["tenant"], row["copied"]) for row in rows] == [
        ("Docs", "a", 10),
        ("Docs", "b", 10),
    ]


def test_sync_text_output_lists_each_entry_and_the_total(tmp_path):
    fakes = inspected(FakeFactories(FakeSource(cli_records(10))))
    two_entries(tmp_path / DIR, ("Docs", "a"), ("Docs", "b"))
    r = invoke(fakes, "sync")
    assert r.exit_code == 0, r.output
    lines = r.stdout.splitlines()
    entry = "copied 10, unchanged 0, rejects 0, stale 0, deleted 0"
    assert lines[-3:] == [
        f"docs -> Docs (tenant a): {entry}",
        f"docs -> Docs (tenant b): {entry}",
        "Total: copied 20, unchanged 0, rejects 0, stale 0, deleted 0",
    ]
    assert "Syncing docs to Docs (tenant a)" in lines
    for tenant in ("a", "b"):
        assert (
            f"Synced docs to Docs (tenant {tenant}): 10 records written, "
            "0 unchanged, 0 rejects, 0 stale, 0 deleted"
        ) in lines


def test_status_after_a_sync_shows_every_record_copied():
    fakes = copied(FakeFactories(FakeSource(cli_records(10))))
    fakes.source.records[1] = replace(fakes.source.records[1], document="changed")
    assert invoke(fakes, "sync").exit_code == 0
    row = json.loads(invoke(fakes, "status", "--json").stdout)["collections"][0]
    assert (row["copied"], row["total"], row["done"]) == (10, 10, True)


def test_verify_after_a_deletion_names_sync_prune():
    fakes = copied(FakeFactories(FakeSource(cli_records(10))))
    del fakes.source.records[2]
    r = invoke(fakes, "verify", "--json")
    assert r.exit_code == 1, r.output
    checks = {c["name"]: c for c in json.loads(r.stdout)["checks"]}
    assert checks["count"]["detail"].endswith(COUNT_ABOVE_SOURCE)
    assert checks["ids"]["detail"].endswith(TARGET_ABOVE_SOURCE)
    assert '["id-2"]' in checks["ids"]["detail"]


def test_copy_stores_the_plan_hash(tmp_path):
    copied(FakeFactories(FakeSource(cli_records(10))))
    entry = load_plan(tmp_path / DIR / "plan.yaml").collections[0]
    state = State(tmp_path / DIR / "state.sqlite")
    try:
        assert state.set_plan_hash(entry.source.ref, plan_hash(entry)) is False
        # Another hash is reported as a change only when one was stored.
        assert state.set_plan_hash(entry.source.ref, b"\0" * 16) is True
    finally:
        state.close()


def test_sync_without_a_copy_before_creates_the_collection_and_copies():
    fakes = FakeFactories(FakeSource(cli_records(10)))
    assert invoke(fakes, "inspect", "chroma").exit_code == 0
    r = invoke(fakes, "sync", "--prune", "--json")
    assert counts(row_of(r)) == (10, 0, 0, 0, 0)
    assert fakes.schema.ensured == ["Docs"] and len(fakes.writer.objects) == 10


def test_sync_help_names_prune():
    r = invoke(FakeFactories(FakeSource(cli_records(1))), "sync", "--help")
    # A CI runner can make Rich color the help, which splits the option names.
    text = re.sub(r"\x1b\[[0-9;]*m", "", r.stdout)
    assert r.exit_code == 0 and "--prune" in text and "--force" in text
    assert "Read the source again and write the new and changed records." in text


def test_sync_rejects_a_parallelism_outside_the_range():
    fakes = copied(FakeFactories(FakeSource(cli_records(10))))
    r = invoke(fakes, "sync", "--parallelism", "0", "--json")
    assert error_of(r)[:2] == (2, "usage")
    assert json.loads(r.stdout)["message"] == "--parallelism 0 is outside 1 to 8."


def test_sync_refuses_an_invalid_plan_before_it_writes(tmp_path):
    fakes = inspected(FakeFactories(FakeSource(cli_records(10))))
    target = load_plan(tmp_path / DIR / "plan.yaml").collections[0].target
    properties = [p for p in target.properties if p.name != "year"]
    edit_target(tmp_path / DIR, properties=properties)
    r = invoke(fakes, "sync", "--json")
    assert r.exit_code == 2, r.output
    d = json.loads(r.stdout)
    assert (d["code"], d["errors"][0]["code"]) == ("invalid_plan", "unmapped_key")
    assert fakes.writer.calls == 0 and not (tmp_path / DIR / "state.sqlite").exists()


def test_sync_fails_as_internal_error_when_a_write_fails_and_deletes_nothing():
    fakes = copied(FakeFactories(FakeSource(cli_records(10))))
    del fakes.source.records[2]
    fakes.source.records[1] = replace(fakes.source.records[1], document="changed")
    fakes.writer.fail_on_call = fakes.writer.calls + 1
    r = invoke(fakes, "sync", "--prune", "--json")
    assert r.exit_code == 5, r.output
    assert len(fakes.writer.objects) == 10 and fakes.deleter.calls == 0


def test_sync_events_hold_plan_changed_and_sync_finished(tmp_path):
    fakes = copied(FakeFactories(FakeSource(cli_records(10))))
    target = load_plan(tmp_path / DIR / "plan.yaml").collections[0].target
    edit_target(
        tmp_path / DIR,
        properties=[p for p in target.properties if p.name != "year"],
        dropped=[*target.dropped, DroppedKey(source_key="year", reason="not needed")],
    )
    del fakes.source.records[0]
    assert invoke(fakes, "sync", "--prune").exit_code == 0
    lines = (tmp_path / DIR / "events.jsonl").read_text().splitlines()
    events = [json.loads(line) for line in lines]
    assert sum(e["event"] == "plan_changed" for e in events) == 1
    last = events[-1]
    assert last["event"] == "sync_finished"
    counted = (last["copied"], last["unchanged"], last["stale"], last["deleted"])
    assert counted == (9, 0, 1, 1)


def other_objects(fakes: FakeFactories, source_ids: list[str | None]) -> None:
    """Objects in Docs that this directory did not write."""
    place = fakes.writer.stored.setdefault(("Docs", None), {})
    for i, source_id in enumerate(source_ids):
        uuid = f"00000000-0000-0000-0000-{i + 1:012d}"
        properties = {} if source_id is None else {"source_id": source_id}
        place[uuid] = TargetObject(uuid, properties, {})


FOREIGN_MESSAGE = (
    "2 objects have a source_id that docs does not hold; sync --prune deletes "
    "them. Unless their records were deleted from docs, copy into another "
    "collection or tenant."
)


@pytest.mark.parametrize("command", ["copy", "sync"])
def test_a_first_run_into_a_collection_with_other_objects_warns(command):
    fakes = inspected(FakeFactories(FakeSource(cli_records(10))))
    other_objects(fakes, ["other-1", "other-2", None])
    r = invoke(fakes, command, "--yes", "--json")
    assert r.exit_code == 0, r.output
    assert json.loads(r.stdout)["warnings"] == [
        {"code": "foreign_objects", "collection": "Docs", "message": FOREIGN_MESSAGE}
    ]
    assert f"Warning: docs: {FOREIGN_MESSAGE}" in r.stderr.splitlines()
    # The next run has progress, so it warns no more; a copy lists no id.
    calls = len(fakes.source.calls)
    r = invoke(fakes, command, "--json")
    assert json.loads(r.stdout)["warnings"] == []
    if command == "copy":
        assert "ids" not in fakes.source.calls[calls:]


def test_a_first_copy_into_a_new_collection_lists_no_id():
    fakes = inspected(FakeFactories(FakeSource(cli_records(10))))
    r = invoke(fakes, "copy", "--json")
    assert r.exit_code == 0 and json.loads(r.stdout)["warnings"] == []
    assert "ids" not in fakes.source.calls


def test_a_copy_again_into_the_same_collection_from_a_new_directory_does_not_warn(
    tmp_path,
):
    fakes = copied(FakeFactories(FakeSource(cli_records(10))))
    shutil.rmtree(tmp_path / DIR)
    fakes = inspected(fakes)
    r = invoke(fakes, "copy", "--json")
    assert r.exit_code == 0 and json.loads(r.stdout)["warnings"] == []


def test_objects_without_a_source_id_do_not_warn():
    fakes = inspected(FakeFactories(FakeSource(cli_records(10))))
    other_objects(fakes, [None, None])
    r = invoke(fakes, "copy", "--json")
    assert r.exit_code == 0 and json.loads(r.stdout)["warnings"] == []


FOREIGN_HINT = (
    "Copy into another collection or tenant, or, when these objects may be "
    "deleted, run the same command with --yes."
)


def stopped(r: Result, command: str) -> None:
    # CliRunner writes the typed answer to stdout; a terminal shows it instead.
    d = json.loads(r.stdout[r.stdout.index("{") :])
    assert (r.exit_code, d["code"], d["message"], d["hint"]) == (
        2,
        "foreign_objects",
        f"Docs holds objects that sync --prune would delete; the {command} wrote "
        "nothing.",
        FOREIGN_HINT,
    )
    assert d["warnings"][0]["code"] == "foreign_objects"


@pytest.mark.parametrize("command", ["copy", "sync"])
def test_a_first_run_with_other_objects_stops_without_a_terminal_or_yes(command):
    fakes = inspected(FakeFactories(FakeSource(cli_records(10))))
    other_objects(fakes, ["other-1", "other-2"])
    r = invoke(fakes, command, "--json")
    stopped(r, command)
    assert fakes.schema.ensured == [] and len(fakes.writer.objects) == 2
    assert f"Warning: docs: {FOREIGN_MESSAGE}" in r.stderr.splitlines()


@pytest.mark.parametrize("command", ["copy", "sync"])
@pytest.mark.parametrize(
    ("answer", "goes_on"), [("y\n", True), ("n\n", False), ("", False)]
)
def test_a_first_run_with_other_objects_asks_in_a_terminal(
    monkeypatch, command, answer, goes_on
):
    monkeypatch.setattr(cli, "_interactive", lambda: True)
    fakes = inspected(FakeFactories(FakeSource(cli_records(10))))
    other_objects(fakes, ["other-1", "other-2"])
    r = invoke(fakes, command, "--json", input=answer)
    assert f"Go on with the {command}? [y/N]" in r.stderr
    if goes_on:
        assert r.exit_code == 0, r.output
        assert len(fakes.writer.objects) == 12
    else:
        stopped(r, command)
        assert len(fakes.writer.objects) == 2


def test_yes_changes_nothing_without_other_objects(monkeypatch):
    monkeypatch.setattr(cli, "_interactive", lambda: True)
    fakes = inspected(FakeFactories(FakeSource(cli_records(10))))
    r = invoke(fakes, "copy", "--yes", "--json")
    assert r.exit_code == 0 and json.loads(r.stdout)["warnings"] == []
    assert "Go on" not in r.stderr


def test_a_first_run_into_an_inactive_tenant_lists_no_id(tmp_path):
    fakes = inspected(FakeFactories(FakeSource(cli_records(10))))
    two_entries(tmp_path / DIR, ("Docs", "a"), ("Docs", "b"))
    uuid = "00000000-0000-0000-0000-000000000001"
    other = TargetObject(uuid, {"source_id": "other-1"}, {})
    fakes.writer.stored[("Docs", "b")] = {uuid: other}
    fakes.writer.inactive = {("Docs", "b")}  # type: ignore[attr-defined]
    r = invoke(fakes, "copy", "--json")
    # Weaviate refuses to read or write an inactive tenant; the write reports it.
    assert r.exit_code == 0, r.output
    assert json.loads(r.stdout)["warnings"] == []
    assert "ids" not in fakes.source.calls


def test_a_first_run_into_a_collection_without_source_id_lists_no_id():
    fakes = inspected(FakeFactories(FakeSource(cli_records(10))))
    other_objects(fakes, [None])
    # Weaviate refuses a query for a property the collection does not have.
    fakes.schema.properties["Docs"] = {"document"}
    r = invoke(fakes, "copy", "--json")
    assert r.exit_code == 0, r.output
    assert "ids" not in fakes.source.calls
    added = json.loads(r.stdout)["added_properties"]
    assert {"collection": "Docs", "property": "source_id"} in added
