from chroma.seed import (
    COLLECTIONS,
    add_to_tiny_l2,
    count,
    delete_from_tiny_l2,
    extra_tiny_l2_ids,
    tiny_l2_count,
)
from conftest import (
    SUITE_COLLECTIONS,
    read_plan,
    run,
    weaviate_config,
    weaviate_count,
    weaviate_object,
    write_plan,
)


def test_every_seeded_collection_in_one_run(seeded, tmp_path):
    r = run(["inspect", "chroma"], tmp_path)
    assert r.exit_code == 0, r
    targets = {row["source"]: row["target"] for row in r.data["collections"]}
    assert set(COLLECTIONS) <= set(targets), r
    assert {targets[name] for name in COLLECTIONS} <= set(SUITE_COLLECTIONS)
    # The test server can hold collections that the seed does not make, and
    # the suite writes only to its own target collections.
    plan = read_plan(tmp_path)
    plan["collections"] = [
        e for e in plan["collections"] if e["source"]["name"] in COLLECTIONS
    ]
    write_plan(tmp_path, plan)
    for cmd in (["check"], ["copy"], ["verify"]):
        r = run(cmd, tmp_path)
        assert r.exit_code == 0 and r.data["status"] == "success", r
    checks = r.data["checks"]
    assert len(checks) == 3 * len(COLLECTIONS) and all(c["passed"] for c in checks)
    r = run(["status"], tmp_path)
    assert r.exit_code == 0, r
    rows = {row["target"]: row for row in r.data["collections"]}
    assert len(rows) == len(COLLECTIONS), r
    for name in COLLECTIONS:
        expected = count(name)
        row = rows[targets[name]]
        assert (row["total"], row["copied"], row["rejects"], row["done"]) == (
            expected,
            expected,
            0,
            True,
        ), row
        assert weaviate_count(targets[name]) == expected, name


def test_end_to_end(seeded, tmp_path):
    names = ["--collection", "docs-cosine", "--collection", "tiny.l2"]
    r = run(["inspect", "chroma", *names], tmp_path)
    assert r.exit_code == 0, r
    for cmd in (["check"], ["copy"], ["verify"]):
        r = run(cmd, tmp_path)
        assert r.exit_code == 0 and r.data["status"] == "success", r
    assert weaviate_count("Docs_cosine") == 700 and weaviate_count("Tiny_l2") == 10
    # The seed gives tiny.l2 UUID ids, and a source id that is a UUID is kept.
    tiny = weaviate_object("Tiny_l2")
    assert str(tiny.uuid) == tiny.properties["source_id"]
    types = {p.name: p.data_type for p in weaviate_config("Docs_cosine").properties}
    assert types["code"].value == "text"
    r = run(["copy"], tmp_path)
    assert r.exit_code == 0 and weaviate_count("Docs_cosine") == 700, r


# (start, count) of each add to tiny.l2, after its 10 seed records.
TINY_L2_ADDS = [(10, 5), (15, 5)]


def test_each_copy_adds_the_records_that_the_source_gained(seeded, tmp_path):
    r = run(["inspect", "chroma", "--collection", "tiny.l2"], tmp_path)
    assert r.exit_code == 0, r
    # The other tests expect tiny.l2 as the seed made it. The cleanup deletes
    # every id an add can store, also when that add failed or never ran.
    intended = [i for add in TINY_L2_ADDS for i in extra_tiny_l2_ids(*add)]
    try:
        r = run(["copy"], tmp_path)
        assert r.exit_code == 0, r
        for add in TINY_L2_ADDS:
            add_to_tiny_l2(*add)
            r = run(["copy"], tmp_path)
            assert r.exit_code == 0, r
        assert weaviate_count("Tiny_l2") == 20
        r = run(["status"], tmp_path)
        row = r.data["collections"][0]
        assert (row["copied"], row["total"]) == (20, 20), r
        r = run(["verify"], tmp_path)
        assert r.exit_code == 0 and r.data["status"] == "success", r
    finally:
        delete_from_tiny_l2(intended)
        assert tiny_l2_count() == 10


def test_wrong_or_missing_api_key_exits_3(seeded, tmp_path):
    r = run(["inspect", "chroma", "--collection", "tiny.l2"], tmp_path)
    assert r.exit_code == 0, r
    # An empty value counts as unset: run cannot remove a variable.
    for key in ("wrong", ""):
        r = run(["check"], tmp_path, env={"WEAVIATE_API_KEY": key})
        assert r.exit_code == 3 and r.data["code"] == "auth_failed", r
        assert "WEAVIATE_API_KEY" in r.data["hint"]


def test_rq_bits_reach_the_server(seeded, tmp_path):
    r = run(
        [
            "inspect",
            "chroma",
            "--collection",
            "docs-rq",
            "--index",
            "hnsw",
            "--quantizer",
            "rq",
            "--quantizer-bits",
            "4",
        ],
        tmp_path,
    )
    assert r.exit_code == 0, r
    for cmd in (["check"], ["copy"], ["verify"]):
        r = run(cmd, tmp_path)
        assert r.exit_code == 0, r
    vector = weaviate_config("Docs_rq").vector_config["default"]
    assert vector.vector_index_config.quantizer.bits == 4
