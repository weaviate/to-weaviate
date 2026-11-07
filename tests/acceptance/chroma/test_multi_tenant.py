from chroma.seed import count
from conftest import (
    read_plan,
    run,
    weaviate_config,
    weaviate_count,
    weaviate_tenants,
    write_plan,
)

TARGET = "Tenants"
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
