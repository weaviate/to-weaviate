import json
from typing import Any

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
    write_plan,
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
LARGE_INT_NOTE = (
    "key int_edge holds ints above 2**53 in magnitude, which Weaviate would "
    "store rounded as int or number; the property is typed text and holds the "
    "decimal digits"
)


def _int_reject(value: int) -> str:
    return (
        f"int_edge: int {value} is above 2**53 in magnitude and Weaviate would "
        "store it rounded; set the property type to text, or text[] for a list"
    )


def _int_edge(plan: dict[str, Any]) -> dict[str, Any]:
    properties = plan["collections"][0]["target"]["properties"]
    found: dict[str, Any] = next(p for p in properties if p["source_key"] == "int_edge")
    return found


def test_values_at_the_limits_of_their_type_are_copied(seeded, tmp_path):
    r = run(["inspect", "chroma", "--collection", EDGE], tmp_path)
    assert r.exit_code == 0, r
    plan = read_plan(tmp_path)
    properties = plan["collections"][0]["target"]["properties"]
    names = {p["source_key"]: p["name"] for p in properties}
    assert {key: names[key] for key in ODD_KEYS} == DRAFTED_NAMES
    # An int above 2**53 in the sample makes the key text.
    drafted = _int_edge(plan)
    assert drafted["type"] == "text", drafted
    assert drafted["coerced_from"] == ["int", "large_int"], drafted
    assert LARGE_INT_NOTE in plan["collections"][0]["notes"]
    r = run(["check"], tmp_path)
    assert r.exit_code == 0, r
    warnings = sorted((w["code"], w["message"].split()[1]) for w in r.data["warnings"])
    assert warnings == [
        ("coerced_keys", "flag"),
        ("coerced_keys", "int_edge"),
        ("coerced_keys", "num"),
    ], r
    source = records(EDGE, limit=40)
    r = run(["copy"], tmp_path)
    assert r.exit_code == 0, r
    row = r.data["collections"][0]
    assert (row["copied"], row["rejects"]) == (40, 0), r
    r = run(["verify", "--sample", "10000"], tmp_path)
    assert r.exit_code == 0, r
    checks = {c["name"]: c for c in r.data["checks"]}
    assert "compared 40, rejects skipped 0, differing 0" in checks["records"]["detail"]
    for record in source:
        found = weaviate_by_source_id("Edge", record.id)
        assert found is not None, record.id
        stored = found.properties
        meta = record.metadata
        assert stored["source_id"] == record.id
        assert stored["document"] == record.document
        assert stored["int_edge"] == str(meta["int_edge"])
        assert stored["float_edge"] == meta["float_edge"]
        assert stored["flag"] == str(meta["flag"]).lower()
        assert stored["num"] == float(meta["num"])
        assert stored["empty"] == ""
        assert stored["tags"] == meta["tags"]
        for key, name in DRAFTED_NAMES.items():
            assert stored[name] == meta[key], (record.id, key)


def test_an_int_above_2_to_the_53_for_an_int_property_is_a_reject(seeded, tmp_path):
    """A plan that types the key int, as 0.1.0 drafted it."""
    r = run(["inspect", "chroma", "--collection", EDGE], tmp_path)
    assert r.exit_code == 0, r
    plan = read_plan(tmp_path)
    _int_edge(plan).update(
        type="int", tokenization=None, range_filters=True, coerced_from=[]
    )
    write_plan(tmp_path, plan)
    r = run(["check"], tmp_path)
    assert r.exit_code == 0, r
    warned = [w["message"] for w in r.data["warnings"] if w["code"] == "large_ints"]
    assert len(warned) == 1 and warned[0].startswith("key int_edge has ints"), r
    source = records(EDGE, limit=40)
    refused = {r.id for r in source if abs(r.metadata["int_edge"]) > 2**53}
    assert len(refused) == 24
    r = run(["copy"], tmp_path)
    assert r.exit_code == 1 and r.data["code"] == "rejects", r
    row = r.data["collections"][0]
    assert (row["copied"], row["rejects"]) == (16, 24), r
    lines = (tmp_path / "rejects.jsonl").read_text(encoding="utf-8").splitlines()
    found = [json.loads(line) for line in lines]
    assert {f["source_id"] for f in found} == refused
    assert {f["reason"] for f in found} == {
        _int_reject(value) for value in (2**53 + 1, 2**63 - 1, -(2**63))
    }
    r = run(["verify", "--sample", "10000"], tmp_path)
    assert r.exit_code == 1 and r.data["code"] == "verification_failed", r
    checks = {c["name"]: c for c in r.data["checks"]}
    assert checks["count"]["detail"].startswith("source count 40, target count 16;")
    ids = checks["ids"]["detail"]
    assert ids.startswith("missing 24, extra 0; missing ids "), ids
    listed, _, fix = ids.partition("missing ids ")[2].partition("; ")
    assert set(json.loads(listed)) <= refused
    assert fix == MISSING_IDS
    assert checks["records"]["passed"], checks["records"]
    assert "rejects skipped 24, differing 0" in checks["records"]["detail"]
    for record in source:
        found_object = weaviate_by_source_id("Edge", record.id)
        if record.id in refused:
            assert found_object is None, record.id
            continue
        assert found_object is not None, record.id
        assert found_object.properties["int_edge"] == record.metadata["int_edge"]


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
