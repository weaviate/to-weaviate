import time

import pytest
from chroma.seed import count
from conftest import (
    read_plan,
    run,
    weaviate_compressed,
    weaviate_config,
    weaviate_count,
    weaviate_exists,
    weaviate_tenants,
    write_plan,
)

TARGET = "Tenants"
COMPRESSION_WAIT = 3.0
# Source collection to tenant of TARGET.
TENANTS = {"tenant-a": "a", "tenant-b": "b"}


def test_two_collections_copy_into_two_tenants_of_one_collection(seeded, tmp_path):
    names = [arg for source in TENANTS for arg in ("--collection", source)]
    r = run(["inspect", "chroma", *names], tmp_path)
    assert r.exit_code == 0, r
    plan = read_plan(tmp_path)
    assert sorted(e["source"]["name"] for e in plan["collections"]) == sorted(TENANTS)
    for entry in plan["collections"]:
        entry["target"]["name"] = TARGET
        entry["target"]["tenant"] = TENANTS[entry["source"]["name"]]
    write_plan(tmp_path, plan)
    for cmd in (["check"], ["copy"], ["verify"]):
        r = run(cmd, tmp_path)
        assert r.exit_code == 0 and r.data["status"] == "success", r
    checks = r.data["checks"]
    assert sorted({c["tenant"] for c in checks}) == ["a", "b"]
    assert all(c["passed"] and c["collection"] == TARGET for c in checks), checks
    assert weaviate_config(TARGET).multi_tenancy_config.enabled
    assert weaviate_tenants(TARGET) == {"a", "b"}
    # Only the records of tenant b have region, and the collection has it.
    properties = {p.name for p in weaviate_config(TARGET).properties}
    assert {"position", "region"} <= properties
    for source, tenant in TENANTS.items():
        assert weaviate_count(TARGET, tenant) == count(source), tenant
    r = run(["status"], tmp_path)
    assert r.exit_code == 0, r
    stored = {row["ref"]: row for row in r.data["collections"]}
    for entry in plan["collections"]:
        row = stored[entry["source"]["ref"]]
        expected = count(entry["source"]["name"])
        assert (row["copied"], row["total"], row["done"]) == (expected, expected, True)
        assert row["stored_target"]["collection"] == TARGET
        assert row["stored_target"]["tenant"] == entry["target"]["tenant"]
    # A second run finds both tenants and copies nothing more.
    r = run(["copy"], tmp_path)
    assert r.exit_code == 0, r
    for source, tenant in TENANTS.items():
        assert weaviate_count(TARGET, tenant) == count(source), tenant


def _position_as_text(entry: dict) -> None:
    for prop in entry["target"]["properties"]:
        if prop["name"] == "position":
            prop["type"] = "text"
            prop["tokenization"] = "field"
            prop["range_filters"] = False


# (source collections, change to the plan entry of the second one, code).
# copy creates the collection from the first entry, so a second entry with
# other vectors or other property settings would be written into a collection
# that its plan does not describe.
CONFLICTS = {
    # tiny.l2 has l2 vectors and drafts hnsw with l2-squared.
    "other-vectors": (["tenant-a", "tiny.l2"], None, "conflicting_vectors"),
    "other-property-type": (
        ["tenant-a", "tenant-b"],
        _position_as_text,
        "conflicting_property",
    ),
}


@pytest.mark.parametrize(
    ("sources", "change", "code"), CONFLICTS.values(), ids=CONFLICTS
)
def test_entries_of_one_collection_must_agree_on_its_settings(
    seeded, tmp_path, sources, change, code
):
    names = [arg for source in sources for arg in ("--collection", source)]
    r = run(["inspect", "chroma", *names], tmp_path)
    assert r.exit_code == 0, r
    plan = read_plan(tmp_path)
    entries = {e["source"]["name"]: e for e in plan["collections"]}
    for tenant, source in zip("ab", sources, strict=True):
        entries[source]["target"]["name"] = TARGET
        entries[source]["target"]["tenant"] = tenant
    if change is not None:
        change(entries[sources[1]])
    plan["collections"] = [entries[source] for source in sources]
    write_plan(tmp_path, plan)
    for cmd in (["check"], ["copy"]):
        r = run(cmd, tmp_path)
        assert r.exit_code == 2 and r.data["code"] == "invalid_plan", r
        assert [e["code"] for e in r.data["errors"]] == [code], r
    assert not weaviate_exists(TARGET)


def test_a_trained_quantizer_compresses_the_vectors_of_every_tenant(seeded, tmp_path):
    names = [arg for source in TENANTS for arg in ("--collection", source)]
    r = run(["inspect", "chroma", *names], tmp_path)
    assert r.exit_code == 0, r
    plan = read_plan(tmp_path)
    for entry in plan["collections"]:
        entry["target"]["name"] = TARGET
        entry["target"]["tenant"] = TENANTS[entry["source"]["name"]]
        index = {"type": "hnsw", "quantizer": {"type": "sq"}}
        entry["target"]["vectors"][0]["index"] = index
    write_plan(tmp_path, plan)
    for cmd in (["check"], ["copy"], ["verify"]):
        r = run(cmd, tmp_path)
        assert r.exit_code == 0 and r.data["status"] == "success", r
    # One shard per tenant; copy enabled sq after the pages of each tenant.
    deadline = time.monotonic() + COMPRESSION_WAIT
    while weaviate_compressed(TARGET) != [True, True] and time.monotonic() < deadline:
        time.sleep(0.5)
    assert weaviate_compressed(TARGET) == [True, True]
