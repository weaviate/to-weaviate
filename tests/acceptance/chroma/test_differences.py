from chroma.seed import records
from conftest import (
    Result,
    run,
    weaviate_by_source_id,
    weaviate_delete_object,
    weaviate_insert_object,
    weaviate_update_object,
)

TARGET = "Tiny_l2"


def _failed(r: Result) -> dict[str, dict]:
    """The checks of a verify run that failed with verification_failed, by name."""
    assert r.exit_code == 1 and r.data["code"] == "verification_failed", r
    return {c["name"]: c for c in r.data["checks"]}


def test_verify_finds_a_missing_and_a_changed_object(seeded, tmp_path):
    # tiny.l2 has 10 records, so the default sample holds every record.
    r = run(["inspect", "chroma", "--collection", "tiny.l2"], tmp_path)
    assert r.exit_code == 0, r
    r = run(["copy"], tmp_path)
    assert r.exit_code == 0, r
    source_id = records("tiny.l2", limit=1)[0].id
    stored = weaviate_by_source_id(TARGET, source_id)
    assert stored is not None

    weaviate_delete_object(TARGET, stored.uuid)
    checks = _failed(run(["verify"], tmp_path))
    assert not checks["count"]["passed"]
    assert "source count 10, target count 9" in checks["count"]["detail"]
    assert not checks["ids"]["passed"]
    assert checks["ids"]["detail"] == f'missing 1, extra 0; missing ids ["{source_id}"]'
    assert f'"{source_id}": missing in the target' in checks["records"]["detail"]

    weaviate_insert_object(TARGET, stored)
    r = run(["verify"], tmp_path)
    assert r.exit_code == 0, r

    weaviate_update_object(TARGET, stored.uuid, {"uri": "https://example.com/changed"})
    checks = _failed(run(["verify"], tmp_path))
    assert checks["count"]["passed"] and checks["ids"]["passed"], checks
    assert not checks["records"]["passed"]
    assert checks["records"]["detail"].endswith(
        f'differing 1: "{source_id}": property uri differs'
    ), checks["records"]

    weaviate_update_object(TARGET, stored.uuid, {"uri": stored.properties["uri"]})
    weaviate_delete_object(TARGET, stored.uuid)
    checks = _failed(run(["verify", "--quick"], tmp_path))
    assert not checks["count"]["passed"]
    assert "ids" not in checks, checks
