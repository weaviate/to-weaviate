import json
from pathlib import Path

import pytest
from fakes import (
    FakeFactories,
    FakeSchema,
    FakeSource,
    FakeWriter,
    cli_records,
    invoke,
)

from to_weaviate.plan import load_plan, save_plan
from to_weaviate.target.schema import SchemaRejected

DIR = Path(".to-weaviate")
OPENAI_SMALL = {
    "type": "known",
    "name": "openai",
    "config": {"model_name": "text-embedding-3-small"},
}


@pytest.fixture(autouse=True)
def in_tmp(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


def inspected(fakes: FakeFactories, *args: str) -> FakeFactories:
    r = invoke(fakes, "inspect", "chroma", *args)
    assert r.exit_code == 0, r.output
    return fakes


def test_inspect_without_key_exits_3_and_names_variable():
    fakes = FakeFactories(FakeSource(cli_records(10)))
    r = invoke(fakes, "inspect", "chroma", CHROMA_API_KEY=None)
    assert r.exit_code == 3 and "CHROMA_API_KEY" in r.output
    assert fakes.source_calls == 0


def test_inspect_writes_plan_and_inspect_json(tmp_path):
    r = invoke(FakeFactories(FakeSource(cli_records(10))), "inspect", "chroma")
    assert r.exit_code == 0, r.output
    plan = load_plan(tmp_path / DIR / "plan.yaml")
    assert [(e.source.ref, e.target.name) for e in plan.collections] == [
        (FakeSource.REF, "Docs")
    ]
    found = json.loads((tmp_path / DIR / "inspect.json").read_text())["collections"]
    assert [(c["ref"], c["count"]) for c in found] == [(FakeSource.REF, 10)]
    assert found[0]["fields"] == [{"key": "year", "types": {"int": 10}}]
    assert "plan.yaml" in r.stdout and "to-weaviate check" in r.stdout


def test_inspect_refuses_to_overwrite_plan():
    fakes = inspected(FakeFactories(FakeSource(cli_records(10))))
    r = invoke(fakes, "inspect", "chroma")
    assert r.exit_code == 2 and "--force" in r.output
    assert fakes.source_calls == 1
    assert invoke(fakes, "inspect", "chroma", "--force").exit_code == 0


def test_check_requires_provider_key():
    fakes = inspected(
        FakeFactories(FakeSource(cli_records(10))),
        "--vectors",
        "embed",
        "--vectorizer",
        "text2vec-openai",
    )
    r = invoke(fakes, "check")
    assert r.exit_code == 3 and "OPENAI_API_KEY" in r.output
    assert invoke(fakes, "check", OPENAI_API_KEY="openai-key").exit_code == 0


def test_check_rejects_duplicate_target_names_without_tenants(tmp_path):
    fakes = inspected(FakeFactories(FakeSource(cli_records(10))))
    path = tmp_path / DIR / "plan.yaml"
    plan = load_plan(path)
    first = plan.collections[0]
    twin = first.model_copy(
        update={"source": first.source.model_copy(update={"ref": "other"})}
    )
    save_plan(plan.model_copy(update={"collections": [first, twin]}), path)
    r = invoke(fakes, "check", "--json")
    assert r.exit_code == 2
    errors = json.loads(r.stdout)["errors"]
    assert [(e["code"], e["collection"]) for e in errors] == [
        ("duplicate_target", "Docs")
    ]
    tenants = [
        e.model_copy(update={"target": e.target.model_copy(update={"tenant": t})})
        for e, t in [(first, "a"), (twin, "b")]
    ]
    save_plan(plan.model_copy(update={"collections": tenants}), path)
    assert invoke(fakes, "check").exit_code == 0


def test_check_warns_on_coerced_keys():
    fakes = inspected(FakeFactories(FakeSource(cli_records(10, year_text_at=3))))
    r = invoke(fakes, "check")
    assert r.exit_code == 0, r.output
    line = next(x for x in r.stdout.splitlines() if "coerced_keys" in x)
    assert "year" in line and "int: 9" in line and "text: 1" in line


def test_check_without_inspect_json_names_the_coerced_types(tmp_path):
    fakes = inspected(FakeFactories(FakeSource(cli_records(10, year_text_at=3))))
    (tmp_path / DIR / "inspect.json").unlink()
    r = invoke(fakes, "check")
    assert r.exit_code == 0, r.output
    line = next(x for x in r.stdout.splitlines() if "coerced_keys" in x)
    assert "year" in line and "int" in line


def test_check_refuses_keep_with_a_different_model():
    fakes = inspected(
        FakeFactories(FakeSource(cli_records(10), embedding=OPENAI_SMALL)),
        "--vectorizer",
        "text2vec-openai",
        "--vectorizer-option",
        "model=text-embedding-3-large",
    )
    r = invoke(fakes, "check", OPENAI_API_KEY="openai-key")
    assert r.exit_code == 2 and "keep_model_mismatch" in r.output
    assert "embed mode" in r.output
    assert "text-embedding-3-small" in r.output
    assert "text-embedding-3-large" in r.output


def test_check_allows_keep_with_vectorizer_when_source_model_unknown(tmp_path):
    fakes = inspected(
        FakeFactories(FakeSource(cli_records(10))),
        "--vectorizer",
        "text2vec-openai",
        "--vectorizer-option",
        "model=m",
    )
    r = invoke(fakes, "check", OPENAI_API_KEY="openai-key")
    assert r.exit_code == 0, r.output
    assert fakes.schema.ensured == []
    assert not (tmp_path / DIR / "state.sqlite").exists()


def test_check_refuses_module_not_enabled_on_target():
    fakes = inspected(
        FakeFactories(
            FakeSource(cli_records(10)),
            schema=FakeSchema(modules=["text2vec-weaviate", "backup-s3"]),
        ),
        "--vectors",
        "embed",
        "--vectorizer",
        "text2vec-openai",
    )
    r = invoke(fakes, "check", OPENAI_API_KEY="openai-key")
    assert r.exit_code == 2 and "module_not_enabled" in r.output
    assert "backup-s3, text2vec-weaviate" in r.output


def test_check_reports_differences_of_an_existing_collection():
    difference = "vector default: distance is dot, plan has cosine"
    fakes = inspected(
        FakeFactories(
            FakeSource(cli_records(10)),
            schema=FakeSchema(existing={"Docs": [difference]}),
        )
    )
    r = invoke(fakes, "check", "--json")
    assert r.exit_code == 2
    d = json.loads(r.stdout)
    assert (d["status"], d["code"]) == ("error", "invalid_plan")
    assert d["errors"] == [
        {"code": "plan_mismatch", "collection": "Docs", "message": difference}
    ]


def test_check_reports_an_invalid_vectorizer_option():
    fakes = inspected(
        FakeFactories(FakeSource(cli_records(10))),
        "--vectors",
        "embed",
        "--vectorizer",
        "text2vec-openai",
        "--vectorizer-option",
        "colour=blue",
    )
    r = invoke(fakes, "check", "--json", OPENAI_API_KEY="openai-key")
    assert r.exit_code == 2
    errors = json.loads(r.stdout)["errors"]
    assert [e["code"] for e in errors] == ["invalid_vectorizer"]
    assert "colour" in errors[0]["message"]


def test_inspect_vectorizer_options(tmp_path):
    inspected(
        FakeFactories(FakeSource(cli_records(10))),
        "--vectors",
        "embed",
        "--vectorizer",
        "text2vec-openai",
        "--vectorizer-option",
        "model=m",
    )
    vector = load_plan(tmp_path / DIR / "plan.yaml").collections[0].target.vectors[0]
    assert vector.vectorizer is not None
    assert vector.vectorizer.module == "text2vec-openai"
    assert vector.vectorizer.options == {"model": "m"}


def test_inspect_vectorizer_option_values_are_typed(tmp_path):
    options = ["dimensions=256", "flag=true", "other=false", "model=3-small"]
    args = [x for o in options for x in ("--vectorizer-option", o)]
    inspected(
        FakeFactories(FakeSource(cli_records(10))),
        "--vectorizer",
        "text2vec-openai",
        *args,
    )
    vector = load_plan(tmp_path / DIR / "plan.yaml").collections[0].target.vectors[0]
    assert vector.vectorizer is not None
    assert vector.vectorizer.options == {
        "dimensions": 256,
        "flag": True,
        "other": False,
        "model": "3-small",
    }


def test_inspect_index_and_quantizer_options(tmp_path):
    inspected(
        FakeFactories(FakeSource(cli_records(10))),
        "--vectors",
        "both",
        "--index",
        "hnsw",
        "--quantizer",
        "rq",
        "--quantizer-bits",
        "4",
    )
    vectors = load_plan(tmp_path / DIR / "plan.yaml").collections[0].target.vectors
    assert len(vectors) == 2
    for vector in vectors:
        assert vector.index.type == "hnsw"
        assert (vector.index.quantizer.type, vector.index.quantizer.bits) == ("rq", 4)


def test_quantizer_bits_without_rq_is_refused(tmp_path):
    fakes = FakeFactories(FakeSource(cli_records(10)))
    r = invoke(
        fakes,
        "inspect",
        "chroma",
        "--index",
        "hnsw",
        "--quantizer",
        "pq",
        "--quantizer-bits",
        "8",
    )
    assert r.exit_code == 2 and "--quantizer-bits" in r.output
    assert not (tmp_path / DIR / "plan.yaml").exists()


@pytest.mark.parametrize(
    ("args", "named"),
    [
        (["--vectorizer", "text2vec-openai", "--vectorizer-option", "model"], "="),
        (["--vectorizer-option", "model=m"], "--vectorizer"),
        (["--quantizer", "rq"], "quantizer"),
        (["--index", "hnsw", "--quantizer", "rq", "--quantizer-bits", "2"], "bits"),
        (["--collection", "nope"], "nope"),
    ],
)
def test_inspect_usage_errors(tmp_path, args, named):
    r = invoke(FakeFactories(FakeSource(cli_records(10))), "inspect", "chroma", *args)
    assert r.exit_code == 2 and named in r.output
    assert not (tmp_path / DIR / "plan.yaml").exists()


M = "vector index dynamic needs async indexing"


def test_copy_stops_when_the_target_refuses_the_schema():
    source = FakeSource(cli_records(10))
    fakes = inspected(
        FakeFactories(source, schema=FakeSchema(ensure_error=SchemaRejected(M)))
    )
    r = invoke(fakes, "copy", "--json")
    assert r.exit_code == 2
    d = json.loads(r.stdout)
    assert d["code"] == "target_rejected_schema" and M in d["message"]
    assert "--index" in d["hint"]
    assert source.read_calls == 0


def test_copy_exits_1_when_rejects_exist():
    fakes = inspected(
        FakeFactories(
            FakeSource(cli_records(10)), writer=FakeWriter(reject_ids={"id-3"})
        )
    )
    r = invoke(fakes, "copy")
    assert r.exit_code == 1 and "rejects.jsonl" in r.output
    assert len(fakes.writer.objects) == 9


def test_copy_creates_the_collection_and_passes_the_consistency():
    fakes = inspected(FakeFactories(FakeSource(cli_records(10))))
    r = invoke(fakes, "copy", "--consistency", "QUORUM")
    assert r.exit_code == 0, r.output
    assert fakes.schema.ensured == ["Docs"] and fakes.consistency == ["QUORUM"]
    assert len(fakes.writer.objects) == 10


@pytest.mark.parametrize(("name", "exit_code"), [("docs", 0), ("Docs", 0), ("x", 2)])
def test_copy_collection_matches_source_or_target_name(name, exit_code):
    fakes = inspected(FakeFactories(FakeSource(cli_records(10))))
    r = invoke(fakes, "copy", "--collection", name)
    assert r.exit_code == exit_code, r.output
    assert len(fakes.writer.objects) == (10 if exit_code == 0 else 0)


def test_copy_refuses_duplicate_target_names_before_connecting(tmp_path):
    fakes = inspected(FakeFactories(FakeSource(cli_records(10))))
    directory = tmp_path / DIR
    plan = load_plan(directory / "plan.yaml")
    first = plan.collections[0]
    twin = first.model_copy(
        update={"source": first.source.model_copy(update={"ref": "other"})}
    )
    save_plan(
        plan.model_copy(update={"collections": [first, twin]}), directory / "plan.yaml"
    )
    found = json.loads((directory / "inspect.json").read_text())
    found["collections"].append({**found["collections"][0], "ref": "other"})
    (directory / "inspect.json").write_text(json.dumps(found))
    r = invoke(fakes, "copy", "--json")
    assert r.exit_code == 2
    d = json.loads(r.stdout)
    assert d["code"] == "invalid_plan"
    assert [e["code"] for e in d["errors"]] == ["duplicate_target"]
    assert fakes.schema_calls == 0 and fakes.source.read_calls == 0


def test_copy_requires_provider_key():
    fakes = inspected(
        FakeFactories(FakeSource(cli_records(10))),
        "--vectors",
        "embed",
        "--vectorizer",
        "text2vec-openai",
    )
    r = invoke(fakes, "copy")
    assert r.exit_code == 3 and "OPENAI_API_KEY" in r.output
    assert fakes.schema_calls == 0 and fakes.source.read_calls == 0


def test_parallelism_over_8_is_refused():
    fakes = inspected(FakeFactories(FakeSource(cli_records(10))))
    r = invoke(fakes, "copy", "--parallelism", "9")
    assert r.exit_code == 2 and "--parallelism" in r.output
    assert fakes.source.read_calls == 0


def test_parallelism_below_1_is_refused():
    fakes = inspected(FakeFactories(FakeSource(cli_records(10))))
    assert invoke(fakes, "copy", "--parallelism", "0").exit_code == 2


def test_status_json():
    fakes = inspected(FakeFactories(FakeSource(cli_records(1000))))
    assert invoke(fakes, "copy").exit_code == 0
    r = invoke(fakes, "status", "--json")
    assert r.exit_code == 0, r.output
    c = json.loads(r.stdout)["collections"][0]
    assert c["copied"] == 1000 and c["total"] == 1000
    assert c["done"] is True and json.loads(r.stdout)["total"] == 1000


def test_status_without_state_lists_the_plan(tmp_path):
    inspected(FakeFactories(FakeSource(cli_records(10))))
    fakes = FakeFactories(FakeSource([]))
    r = invoke(fakes, "status", "--json", CHROMA_API_KEY=None, WEAVIATE_URL=None)
    assert r.exit_code == 0, r.output
    assert json.loads(r.stdout)["collections"] == [
        {
            "ref": FakeSource.REF,
            "target": "Docs",
            "total": 10,
            "copied": 0,
            "rejects": 0,
            "done": False,
        }
    ]
    assert not (tmp_path / DIR / "state.sqlite").exists()
    assert fakes.source_calls == 0


def test_status_exits_0_with_rejects():
    fakes = inspected(
        FakeFactories(
            FakeSource(cli_records(10)), writer=FakeWriter(reject_ids={"id-3"})
        )
    )
    assert invoke(fakes, "copy").exit_code == 1
    r = invoke(fakes, "status")
    assert r.exit_code == 0 and "rejects 1" in r.stdout
