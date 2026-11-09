import json

from chroma.seed import (
    BULK,
    BULK_COUNT,
    EDGE,
    ODD_KEYS,
    RENAMED_COLLECTIONS,
    records,
)
from conftest import (
    copy_and_verify,
    read_plan,
    run,
    weaviate_by_source_id,
    weaviate_count,
    weaviate_exists,
)

from to_weaviate.engine.verify import MISSING_IDS


def test_an_empty_collection_copies_and_verifies(seeded, tmp_path):
    r = run(["inspect", "chroma", "--collection", "empty"], tmp_path)
    assert r.exit_code == 0, r
    row = r.data["collections"][0]
    assert (row["target"], row["count"], row["dimension"]) == ("Empty", 0, None), r
    r = copy_and_verify(tmp_path)
    assert all(c["passed"] for c in r.data["checks"]), r
    row = run(["status"], tmp_path).data["collections"][0]
    assert (row["total"], row["copied"], row["done"]) == (0, 0, True), row
    assert weaviate_exists("Empty") and weaviate_count("Empty") == 0


# The property name drafted for each of ODD_KEYS. Weaviate 1.40.4 accepts each.
DRAFTED_NAMES = {
    "ID": "iD",
    "Id": "id_meta",
    "id": "id_meta_2",
    "_additional": "_additional_meta",
    "document": "document_meta",
    "uri": "uri_meta",
    "vector": "vector_meta",
    "class": "class",
    "1st key": "p_1st_key",
    "my key": "my_key",
    "日本": "__",
    "created_temp": "created_temp_meta",
    "a__backup": "a__backup_meta",
    "x_nullState": "x_nullState_meta",
}
# Weaviate 1.40.4 returns an int above 2**53 rounded to a double.
STORED_INTS = {2**53 + 1: 2**53}
# copy rejects an int whose double is 2**63: the server's check of it depends
# on its CPU.
REFUSED_INT = 2**63 - 1
INT_REJECT = (
    f"int_edge: int {REFUSED_INT} cannot be stored: Weaviate keeps ints as "
    "doubles and refuses this value; type the property text"
)


def test_values_at_the_limits_of_their_type_are_copied(seeded, tmp_path):
    r = run(["inspect", "chroma", "--collection", EDGE], tmp_path)
    assert r.exit_code == 0, r
    properties = read_plan(tmp_path)["collections"][0]["target"]["properties"]
    names = {p["source_key"]: p["name"] for p in properties}
    assert {key: names[key] for key in ODD_KEYS} == DRAFTED_NAMES
    r = run(["check"], tmp_path)
    assert r.exit_code == 0, r
    warnings = sorted((w["code"], w["message"].split()[1]) for w in r.data["warnings"])
    assert warnings == [("coerced_keys", "flag"), ("coerced_keys", "num")], r
    source = records(EDGE, limit=40)
    refused = {r.id for r in source if r.metadata["int_edge"] == REFUSED_INT}
    assert len(refused) == 8
    r = run(["copy"], tmp_path)
    assert r.exit_code == 1 and r.data["code"] == "rejects", r
    row = r.data["collections"][0]
    assert (row["copied"], row["rejects"]) == (32, 8), r
    lines = (tmp_path / "rejects.jsonl").read_text(encoding="utf-8").splitlines()
    found = [json.loads(line) for line in lines]
    assert {f["source_id"] for f in found} == refused
    assert {f["reason"] for f in found} == {INT_REJECT}
    r = run(["verify", "--sample", "10000"], tmp_path)
    assert r.exit_code == 1 and r.data["code"] == "verification_failed", r
    checks = {c["name"]: c for c in r.data["checks"]}
    assert checks["count"]["detail"].startswith("source count 40, target count 32;")
    ids = checks["ids"]["detail"]
    assert ids.startswith("missing 8, extra 0; missing ids "), ids
    listed, _, fix = ids.partition("missing ids ")[2].partition("; ")
    assert set(json.loads(listed)) == refused
    assert fix == MISSING_IDS
    assert checks["records"]["passed"], checks["records"]
    assert "rejects skipped 8, differing 0" in checks["records"]["detail"]
    for record in source:
        found = weaviate_by_source_id("Edge", record.id)
        if record.id in refused:
            assert found is None, record.id
            continue
        assert found is not None, record.id
        stored = found.properties
        meta = record.metadata
        assert stored["source_id"] == record.id
        assert stored["document"] == record.document
        assert stored["int_edge"] == STORED_INTS.get(meta["int_edge"], meta["int_edge"])
        assert stored["float_edge"] == meta["float_edge"]
        assert stored["flag"] == str(meta["flag"]).lower()
        assert stored["num"] == float(meta["num"])
        assert stored["empty"] == ""
        assert stored["tags"] == meta["tags"]
        for key, name in DRAFTED_NAMES.items():
            assert stored[name] == meta[key], (record.id, key)


def test_collection_names_that_weaviate_refuses_are_renamed(seeded, tmp_path):
    names = [arg for name in RENAMED_COLLECTIONS for arg in ("--collection", name)]
    r = run(["inspect", "chroma", *names], tmp_path)
    assert r.exit_code == 0, r
    targets = {row["source"]: row["target"] for row in r.data["collections"]}
    # The draft names the first of the two mixed-case collections that Chroma
    # lists without a number.
    mixed = [row["source"] for row in r.data["collections"] if "ixed" in row["source"]]
    expected = {
        "Mixed-Case": "Mixed_Case",
        "mixed-case": "Mixed_case",
        "1-leading-digit": "C_1_leading_digit",
        "class": "Class",
    }
    expected[mixed[1]] += "_2"
    assert targets == expected, r
    r = copy_and_verify(tmp_path)
    assert all(c["passed"] for c in r.data["checks"]), r
    for target in targets.values():
        assert weaviate_count(target) == 5, target


def test_a_large_collection_copies_and_verifies(seeded, tmp_path):
    r = run(
        ["inspect", "chroma", "--collection", BULK, "--sample-pages", "5"], tmp_path
    )
    assert r.exit_code == 0, r
    assert r.data["collections"][0]["count"] == BULK_COUNT, r
    r = run(["check"], tmp_path)
    assert r.exit_code == 0, r
    r = run(["copy", "--parallelism", "8"], tmp_path)
    assert r.exit_code == 0, r
    assert weaviate_count("Bulk") == BULK_COUNT
    r = run(["verify"], tmp_path)
    assert r.exit_code == 0, r
    checks = {c["name"]: c["detail"] for c in r.data["checks"]}
    assert checks["count"] == f"source count {BULK_COUNT}, target count {BULK_COUNT}"
    assert checks["ids"] == "missing 0, extra 0"
    assert checks["records"].startswith("sampled 300, compared 300,")
    row = run(["status"], tmp_path).data["collections"][0]
    assert (row["total"], row["copied"], row["rejects"], row["done"]) == (
        BULK_COUNT,
        BULK_COUNT,
        0,
        True,
    ), row
