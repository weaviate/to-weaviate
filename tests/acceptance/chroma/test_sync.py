"""sync after a copy: changed and added records, stale objects and --prune."""

from collections.abc import Iterator
from dataclasses import replace

import pytest
from chroma.seed import (
    ADD_BATCH,
    SYNC,
    SYNC_COUNT,
    add_records,
    delete_collection,
    delete_records,
    make_collection,
    order_records,
    records,
    update_records,
    wait_indexed,
)
from conftest import (
    chroma_kind,
    copy_and_verify,
    read_plan,
    run,
    weaviate_by_source_id,
    weaviate_count,
    weaviate_fetch,
    weaviate_insert,
    weaviate_object,
    write_plan,
)

from to_weaviate.engine.verify import TARGET_ABOVE_SOURCE

TARGET = "Sync"
BASE = order_records(0, SYNC_COUNT)
SYNC_COUNTS = ("copied", "unchanged", "rejects", "stale", "deleted")


def _written(name: str) -> None:
    """On a distributed Chroma, wait until the writes to `name` are compacted."""
    if chroma_kind() == "distributed":
        wait_indexed([name])


@pytest.fixture
def sync_collection(seeded) -> Iterator[None]:
    make_collection(SYNC, BASE, ADD_BATCH)
    _written(SYNC)
    yield
    delete_collection(SYNC)


def _counts(row: dict) -> tuple[int, ...]:
    return tuple(row[key] for key in SYNC_COUNTS)


def _inspect(workdir, *names: str) -> None:
    args = [arg for name in names for arg in ("--collection", name)]
    r = run(["inspect", "chroma", *args], workdir)
    assert r.exit_code == 0, r


def test_sync_writes_the_changed_records_and_prune_deletes_the_stale_ones(
    sync_collection, tmp_path
):
    _inspect(tmp_path, SYNC)
    copy_and_verify(tmp_path)
    add_records(SYNC, order_records(SYNC_COUNT, 5))
    update_records(SYNC, [replace(r, document="changed") for r in BASE[:5]])
    delete_records(SYNC, [r.id for r in BASE[5:10]])
    _written(SYNC)
    r = run(["sync"], tmp_path)
    assert r.exit_code == 0, r
    row = r.data["collections"][0]
    assert _counts(row) == (10, 90, 0, 5, 0), r
    assert (row["plan_changed"], row["prune_refused"], r.data["prune"]) == (
        False,
        False,
        False,
    )
    assert weaviate_count(TARGET) == SYNC_COUNT + 5
    r = run(["verify"], tmp_path)
    assert r.exit_code == 1 and r.data["code"] == "verification_failed", r
    checks = {c["name"]: c for c in r.data["checks"]}
    assert checks["count"]["detail"].endswith(TARGET_ABOVE_SOURCE), r
    assert checks["ids"]["detail"].endswith(TARGET_ABOVE_SOURCE), r
    r = run(["sync", "--prune"], tmp_path)
    assert r.exit_code == 0, r
    assert _counts(r.data["collections"][0]) == (0, 100, 0, 5, 5), r
    assert r.data["prune"] is True
    assert weaviate_count(TARGET) == SYNC_COUNT
    r = run(["verify"], tmp_path)
    assert r.exit_code == 0, r
    changed = weaviate_by_source_id(TARGET, BASE[0].id)
    assert changed is not None and changed.properties["document"] == "changed"
    assert weaviate_by_source_id(TARGET, BASE[5].id) is None
    r = run(["status"], tmp_path)
    assert r.exit_code == 0, r
    status = r.data["collections"][0]
    assert (status["copied"], status["done"]) == (SYNC_COUNT, True), r


def test_prune_above_ten_percent_needs_force(sync_collection, tmp_path):
    _inspect(tmp_path, SYNC)
    copy_and_verify(tmp_path)
    delete_records(SYNC, [r.id for r in BASE[:20]])
    _written(SYNC)
    r = run(["sync", "--prune"], tmp_path)
    assert r.exit_code == 1 and r.data["code"] == "prune_limit", r
    row = r.data["collections"][0]
    assert _counts(row) == (0, 80, 0, 20, 0) and row["prune_refused"] is True, r
    assert "sync --prune --force" in r.data["hint"], r
    assert weaviate_count(TARGET) == SYNC_COUNT
    r = run(["sync", "--prune", "--force"], tmp_path)
    assert r.exit_code == 0, r
    row = r.data["collections"][0]
    assert _counts(row) == (0, 80, 0, 20, 20) and row["prune_refused"] is False, r
    assert weaviate_count(TARGET) == SYNC_COUNT - 20
    r = run(["verify"], tmp_path)
    assert r.exit_code == 0, r


def test_prune_deletes_only_in_its_tenant_and_keeps_objects_without_source_id(
    seeded, tmp_path
):
    tenants = {"tenant-a": "a", "tenant-b": "b"}
    _inspect(tmp_path, *tenants)
    plan = read_plan(tmp_path)
    for entry in plan["collections"]:
        entry["target"]["name"] = "Tenants"
        entry["target"]["tenant"] = tenants[entry["source"]["name"]]
    write_plan(tmp_path, plan)
    copy_and_verify(tmp_path)
    foreign = weaviate_insert("Tenants", {"position": 1}, [0.1] * 8, tenant="a")
    removed = records("tenant-a", 2)
    try:
        delete_records("tenant-a", [r.id for r in removed])
        _written("tenant-a")
        r = run(["sync", "--prune", "--collection", "tenant-a"], tmp_path)
        assert r.exit_code == 0, r
        rows = r.data["collections"]
        assert [(row["source"], row["tenant"]) for row in rows] == [("tenant-a", "a")]
        assert _counts(rows[0]) == (0, 28, 0, 2, 2), r
        assert weaviate_count("Tenants", "a") == 30 - 2 + 1
        assert weaviate_count("Tenants", "b") == 20
        assert weaviate_fetch("Tenants", foreign, tenant="a") is not None
        for record in removed:
            assert weaviate_by_source_id("Tenants", record.id, tenant="a") is None
    finally:
        # The other tests of the session read tenant-a with its 30 records.
        add_records("tenant-a", removed)
        _written("tenant-a")


def test_sync_after_a_plan_edit_writes_every_record(sync_collection, tmp_path):
    _inspect(tmp_path, SYNC)
    copy_and_verify(tmp_path)
    plan = read_plan(tmp_path)
    target = plan["collections"][0]["target"]
    target["properties"] = [p for p in target["properties"] if p["name"] != "position"]
    target["dropped"].append({"source_key": "position", "reason": "test"})
    write_plan(tmp_path, plan)
    r = run(["check"], tmp_path)
    assert r.exit_code == 0, r
    r = run(["sync"], tmp_path)
    assert r.exit_code == 0, r
    row = r.data["collections"][0]
    assert _counts(row) == (100, 0, 0, 0, 0) and row["plan_changed"] is True, r
    assert "Warning: sync: the plan entry changed" in r.stderr, r.stderr
    r = run(["verify"], tmp_path)
    assert r.exit_code == 0, r
    assert weaviate_object(TARGET).properties.get("position") is None
    r = run(["sync"], tmp_path)
    assert r.exit_code == 0, r
    row = r.data["collections"][0]
    assert _counts(row) == (0, 100, 0, 0, 0) and row["plan_changed"] is False, r
    assert "Warning" not in r.stderr, r.stderr


def test_sync_writes_the_rejects_once_the_plan_is_fixed(seeded, tmp_path):
    sampled = ["--collection", "rejects", "--sample-pages", "1"]
    r = run(["inspect", "chroma", *sampled], tmp_path)
    assert r.exit_code == 0, r
    r = run(["copy"], tmp_path)
    assert r.exit_code == 1 and r.data["code"] == "rejects", r
    assert r.data["collections"][0]["rejects"] == 1, r
    assert "to-weaviate sync writes the record" in r.data["hint"], r
    # Weaviate keeps the type of a property, so the plan drops rank instead.
    plan = read_plan(tmp_path)
    target = plan["collections"][0]["target"]
    target["properties"] = [p for p in target["properties"] if p["name"] != "rank"]
    target["dropped"].append({"source_key": "rank", "reason": "mixed types"})
    write_plan(tmp_path, plan)
    r = run(["check"], tmp_path)
    assert r.exit_code == 0, r
    r = run(["sync"], tmp_path)
    assert r.exit_code == 0, r
    row = r.data["collections"][0]
    assert _counts(row) == (301, 0, 0, 0, 0) and row["plan_changed"] is True, r
    assert weaviate_count("Rejects") == 301
    assert weaviate_by_source_id("Rejects", "rej-300") is not None
    r = run(["verify"], tmp_path)
    assert r.exit_code == 0, r
    r = run(["status"], tmp_path)
    assert r.exit_code == 0, r
    status = r.data["collections"][0]
    assert (status["copied"], status["rejects"], status["done"]) == (301, 0, True), r
