from chroma.seed import (
    add_to_tiny_l2,
    delete_from_tiny_l2,
    extra_tiny_l2_ids,
    tiny_l2_count,
)
from conftest import (
    read_plan,
    run,
    weaviate_count,
    weaviate_set_shard_status,
    weaviate_set_tenant_status,
    write_plan,
)

TARGET = "Tiny_l2"
BATCH_HINT = "Fix the cause and run to-weaviate copy again. It resumes at this page."


def test_a_read_only_shard_stops_copy_and_the_next_copy_resumes(seeded, tmp_path):
    r = run(["inspect", "chroma", "--collection", "tiny.l2"], tmp_path)
    assert r.exit_code == 0, r
    r = run(["copy"], tmp_path)
    assert r.exit_code == 0, r
    added = extra_tiny_l2_ids(10, 5)
    try:
        add_to_tiny_l2(10, 5)
        weaviate_set_shard_status(TARGET, "READONLY")
        # Weaviate refuses each object of the batch; copy retries the batch,
        # then stops before the page is advanced.
        r = run(["copy"], tmp_path)
        assert r.exit_code == 4 and r.data["code"] == "target_unreachable", r
        assert r.data["message"] == (
            "Weaviate refused every object of a batch of 5 in Tiny_l2: store is "
            "read-only due to: manually set by user"
        ), r
        assert r.data["hint"] == BATCH_HINT
        row = run(["status"], tmp_path).data["collections"][0]
        assert (row["copied"], row["rejects"], row["done"]) == (10, 0, False), row
        assert not (tmp_path / "rejects.jsonl").exists()
        weaviate_set_shard_status(TARGET, "READY")
        r = run(["copy"], tmp_path)
        assert r.exit_code == 0, r
        assert weaviate_count(TARGET) == 15
        r = run(["verify"], tmp_path)
        assert r.exit_code == 0, r
    finally:
        delete_from_tiny_l2(added)
        assert tiny_l2_count() == 10


def test_an_unreachable_source_or_target_exits_4(seeded, tmp_path):
    inspect = ["inspect", "chroma", "--collection", "tiny.l2"]
    r = run(inspect, tmp_path, env={"CHROMA_URL": "http://localhost:1"})
    assert r.exit_code == 4 and r.data["code"] == "source_unreachable", r
    assert r.data["message"].startswith("Chroma could not be reached: ConnectError")
    assert "CHROMA_URL" in r.data["hint"]
    assert list(tmp_path.iterdir()) == []
    r = run(inspect, tmp_path)
    assert r.exit_code == 0, r
    unreachable = {
        "WEAVIATE_URL": "http://localhost:1",
        "WEAVIATE_GRPC_URL": "http://localhost:2",
    }
    r = run(["check"], tmp_path, env=unreachable)
    assert r.exit_code == 4 and r.data["code"] == "target_unreachable", r
    assert "WEAVIATE_URL" in r.data["hint"]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["inspect.json", "plan.yaml"]


def test_an_inactive_tenant_stops_copy_and_the_next_copy_resumes(seeded, tmp_path):
    names = ["--collection", "tenant-a", "--collection", "tenant-b"]
    r = run(["inspect", "chroma", *names], tmp_path)
    assert r.exit_code == 0, r
    plan = read_plan(tmp_path)
    for entry in plan["collections"]:
        entry["target"]["name"] = "Tenants"
        entry["target"]["tenant"] = entry["source"]["name"].removeprefix("tenant-")
    write_plan(tmp_path, plan)
    r = run(["copy", "--collection", "tenant-a"], tmp_path)
    assert r.exit_code == 0, r
    weaviate_set_tenant_status("Tenants", "b", "INACTIVE")
    r = run(["copy", "--collection", "tenant-b"], tmp_path)
    assert r.exit_code == 4 and r.data["code"] == "target_unreachable", r
    assert r.data["hint"] == BATCH_HINT, r
    # Five readers write chunks of four records of tenant-b.
    assert r.data["message"] == (
        "Weaviate refused every object of a batch of 4 in Tenants: tenant not "
        "active: 'b'"
    ), r
    weaviate_set_tenant_status("Tenants", "b", "ACTIVE")
    r = run(["copy", "--collection", "tenant-b"], tmp_path)
    assert r.exit_code == 0, r
    assert weaviate_count("Tenants", "b") == 20
    r = run(["verify"], tmp_path)
    assert r.exit_code == 0, r
