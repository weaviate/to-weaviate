from pathlib import Path
from typing import Any

from conftest import nodes, read_plan, run, weaviate_config, write_plan

from to_weaviate.target.schema import auto_factor

TARGET = "Tiny_l2"


def _inspect(workdir: Path, factor: Any = "auto") -> None:
    r = run(["inspect", "chroma", "--collection", "tiny.l2"], workdir)
    assert r.exit_code == 0, r
    _plan_factor(workdir, factor)


def _plan_factor(workdir: Path, factor: Any) -> None:
    plan = read_plan(workdir)
    assert plan["collections"][0]["target"]["replication_factor"] == "auto"
    plan["collections"][0]["target"]["replication_factor"] = factor
    write_plan(workdir, plan)


def _stored_factor() -> int:
    return weaviate_config(TARGET).replication_config.factor


def test_copy_creates_a_collection_with_the_factor_of_auto(seeded, tmp_path):
    _inspect(tmp_path)
    r = run(["check"], tmp_path)
    assert r.exit_code == 0 and r.data["warnings"] == [], r
    factor = auto_factor(nodes())
    r = run(["copy"], tmp_path)
    assert r.exit_code == 0 and r.data["status"] == "success", r
    assert r.data["created_collections"] == [
        {
            "collection": TARGET,
            "replication_factor": factor,
            "chosen_by": "nodes",
            "nodes": nodes(),
        }
    ], r
    assert f"Created {TARGET} with replication factor {factor} (" in r.stderr
    assert _stored_factor() == factor
    # The collection exists: the next copy creates nothing, and auto accepts
    # the factor that it has.
    r = run(["copy"], tmp_path)
    assert r.exit_code == 0 and r.data["created_collections"] == [], r
    r = run(["verify"], tmp_path)
    assert r.exit_code == 0 and r.data["status"] == "success", r


def test_a_planned_factor_above_the_node_count_is_refused(seeded, tmp_path):
    factor = nodes() + 1
    _inspect(tmp_path, factor)
    r = run(["check"], tmp_path)
    assert r.exit_code == 2 and r.data["code"] == "invalid_plan", r
    assert r.data["errors"] == [
        {
            "code": "too_few_nodes",
            "collection": TARGET,
            "message": f"replication_factor {factor} needs {factor} nodes and the "
            f"cluster has {nodes()}; lower replication_factor or set it to auto",
        }
    ], r
    # copy sends the factor as the plan has it, and the server refuses it.
    r = run(["copy"], tmp_path)
    assert r.data["code"] == "target_rejected_schema", r


def test_a_planned_factor_is_sent_and_compared_with_the_collection(seeded, tmp_path):
    _inspect(tmp_path, 1)
    r = run(["check"], tmp_path)
    assert r.exit_code == 0 and r.data["warnings"] == [], r
    r = run(["copy"], tmp_path)
    assert r.exit_code == 0 and r.data["status"] == "success", r
    assert r.data["created_collections"] == [
        {
            "collection": TARGET,
            "replication_factor": 1,
            "chosen_by": "plan",
            "nodes": None,
        }
    ], r
    assert _stored_factor() == 1

    plan = read_plan(tmp_path)
    plan["collections"][0]["target"]["replication_factor"] = 2
    write_plan(tmp_path, plan)
    r = run(["check"], tmp_path)
    assert r.exit_code == 2 and r.data["code"] == "invalid_plan", r
    assert r.data["errors"] == [
        {
            "code": "plan_mismatch",
            "collection": TARGET,
            "message": "replication factor is 1, plan has 2",
        }
    ], r
    r = run(["copy"], tmp_path)
    assert r.data["code"] == "plan_mismatch", r

    # auto accepts the collection. On more than one node check warns that it
    # has one replica.
    plan["collections"][0]["target"]["replication_factor"] = "auto"
    write_plan(tmp_path, plan)
    r = run(["check"], tmp_path)
    assert r.exit_code == 0, r
    expected = ["not_replicated"] if nodes() > 1 else []
    assert [w["code"] for w in r.data["warnings"]] == expected, r
    r = run(["copy"], tmp_path)
    assert r.exit_code == 0 and r.data["created_collections"] == [], r
    assert _stored_factor() == 1
