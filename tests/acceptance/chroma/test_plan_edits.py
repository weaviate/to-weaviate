from typing import Any

from chroma.seed import records
from conftest import (
    copy_and_verify,
    create_target_collection,
    read_plan,
    run,
    weaviate_by_source_id,
    weaviate_config,
    weaviate_count,
    weaviate_exists,
    write_plan,
)
from weaviate.classes.config import DataType, Property

from to_weaviate.plan import load_plan
from to_weaviate.target.schema import collection_kwargs

TARGET = "Tiny_l2"


def _inspect_tiny_l2(workdir) -> None:
    r = run(["inspect", "chroma", "--collection", "tiny.l2"], workdir)
    assert r.exit_code == 0, r


def _planned_collection(workdir) -> dict[str, Any]:
    """The arguments with which copy creates the target of the plan in `workdir`."""
    return collection_kwargs(load_plan(workdir / "plan.yaml").collections)


def test_a_target_with_another_property_type_stops_check_and_copy(seeded, tmp_path):
    _inspect_tiny_l2(tmp_path)
    kwargs = _planned_collection(tmp_path)
    kwargs["properties"] = [
        Property(name="position", data_type=DataType.TEXT)
        if p.name == "position"
        else p
        for p in kwargs["properties"]
    ]
    create_target_collection(kwargs)
    r = run(["check"], tmp_path)
    assert r.exit_code == 2 and r.data["code"] == "invalid_plan", r
    assert r.data["errors"] == [
        {
            "code": "plan_mismatch",
            "collection": TARGET,
            "message": "property position: type is text, plan has int",
        }
    ], r
    r = run(["copy"], tmp_path)
    assert r.exit_code == 2 and r.data["code"] == "plan_mismatch", r
    assert weaviate_count(TARGET) == 0


def test_a_target_with_one_more_property_takes_the_copy(seeded, tmp_path):
    _inspect_tiny_l2(tmp_path)
    kwargs = _planned_collection(tmp_path)
    kwargs["properties"].append(Property(name="note", data_type=DataType.TEXT))
    create_target_collection(kwargs)
    copy_and_verify(tmp_path)
    assert weaviate_count(TARGET) == 10
    assert "note" in {p.name for p in weaviate_config(TARGET).properties}


def test_a_hand_edited_plan_is_copied_as_edited(seeded, tmp_path):
    _inspect_tiny_l2(tmp_path)
    plan = read_plan(tmp_path)
    target = plan["collections"][0]["target"]
    for prop in target["properties"]:
        if prop["name"] == "position":
            prop.update(name="pos", type="text", tokenization="field")
            prop["range_filters"] = False
    target["properties"] = [p for p in target["properties"] if p["name"] != "uri"]
    target["dropped"].append({"source_key": "#uri", "reason": "not needed"})
    write_plan(tmp_path, plan)
    copy_and_verify(tmp_path)
    properties = {p.name: p for p in weaviate_config(TARGET).properties}
    assert "uri" not in properties and "position" not in properties
    pos = properties["pos"]
    assert (pos.data_type.value, pos.tokenization.value) == ("text", "field")
    assert not pos.index_range_filters
    stored = [weaviate_by_source_id(TARGET, r) for r in _source_ids(tmp_path)]
    assert sorted(o.properties["pos"] for o in stored if o) == [
        str(i) for i in range(10)
    ]


def _source_ids(workdir) -> list[str]:
    r = run(["verify", "--sample", "10"], workdir)
    assert r.exit_code == 0, r

    return [record.id for record in records("tiny.l2")]


# (change to the drafted plan of tiny.l2, the name that Weaviate refuses)
REFUSED_NAMES = {
    "property": ("properties", "has space"),
    "collection": ("name", "tiny-l2"),
}


def test_a_name_that_weaviate_refuses_is_an_invalid_plan(seeded, tmp_path):
    for field, name in REFUSED_NAMES.values():
        workdir = tmp_path / field
        workdir.mkdir()
        _inspect_tiny_l2(workdir)
        plan = read_plan(workdir)
        target = plan["collections"][0]["target"]
        if field == "name":
            target["name"] = name
        else:
            next(p for p in target["properties"] if p["name"] == "position")["name"] = (
                name
            )
        write_plan(workdir, plan)
        for command in ("check", "copy"):
            r = run([command], workdir)
            assert r.exit_code == 2 and r.data["code"] == "invalid_plan", r
            errors = r.data["errors"]
            assert [e["code"] for e in errors] == ["invalid_name"], r
            assert name in errors[0]["message"], r
        assert not weaviate_exists("Tiny-l2")
