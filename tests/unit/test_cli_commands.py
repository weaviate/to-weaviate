import json
import sqlite3
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any

import httpx
import pytest
import yaml
from fakes import (
    FakeFactories,
    FakeInsertClient,
    FakeSchema,
    FakeSource,
    FakeWriter,
    all_failed,
    cli_records,
    invoke,
    old_state_file,
    raise_grpc_failure,
)
from typer.testing import Result
from weaviate.exceptions import InsufficientPermissionsError, WeaviateBatchError

from to_weaviate.engine.verify import TARGET_ABOVE_SOURCE, TARGET_BELOW_SOURCE
from to_weaviate.ids import to_uuid
from to_weaviate.lock import Locked, exclusive_lock
from to_weaviate.plan import (
    DroppedKey,
    IndexPlan,
    QuantizerPlan,
    VectorizerPlan,
    load_plan,
    save_plan,
)
from to_weaviate.sources.base import Page, Partition, Record, SourceCollection
from to_weaviate.sources.chroma.client import ChromaError
from to_weaviate.sources.chroma.source import ChromaSource
from to_weaviate.state import State
from to_weaviate.target.schema import PropertyRejected, SchemaRejected
from to_weaviate.target.writer import WeaviateWriter

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


def test_copy_resumes_with_a_plan_that_holds_the_full_embedding_config(tmp_path):
    source = FakeSource(cli_records(10), page_size=2, embedding=OPENAI_SMALL)
    fakes = inspected(FakeFactories(source, writer=FakeWriter(fail_on_call=3)))
    path = tmp_path / DIR / "plan.yaml"
    plan = load_plan(path)
    entry = plan.collections[0]
    full = {
        **OPENAI_SMALL,
        "config": {
            **OPENAI_SMALL["config"],
            "api_key_env_var": "OPENAI_API_KEY",
            "default_headers": {"X-Custom": "made-up-header"},
            "organization_id": None,
        },
    }
    old = entry.model_copy(
        update={"source": entry.source.model_copy(update={"embedding": full})}
    )
    save_plan(plan.model_copy(update={"collections": [old]}), path)
    assert load_plan(path).collections[0].source.embedding == full
    key = {"OPENAI_API_KEY": "openai-key"}
    assert invoke(fakes, "check", **key).exit_code == 0
    r = invoke(fakes, "copy", "--parallelism", "1", **key)
    assert r.exit_code == 5, r.output
    r = invoke(fakes, "copy", "--json", **key)
    assert r.exit_code == 0, r.output
    assert json.loads(r.stdout)["collections"][0]["copied"] == 10
    assert len(fakes.writer.objects) == 10


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


def test_inspect_vectorizer_option_values_are_strings(tmp_path):
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
        "dimensions": "256",
        "flag": "true",
        "other": "false",
        "model": "3-small",
    }


def test_an_all_digit_vectorizer_option_keeps_its_text(tmp_path):
    fakes = inspected(
        FakeFactories(FakeSource(cli_records(10))),
        "--vectors",
        "embed",
        "--vectorizer",
        "text2vec-openai",
        "--vectorizer-option",
        "model=007",
        "--vectorizer-option",
        "dimensions=0256",
    )
    text = (tmp_path / DIR / "plan.yaml").read_text()
    assert "model: '007'" in text and "dimensions: '0256'" in text
    r = invoke(fakes, "check", "--json", OPENAI_API_KEY="openai-key")
    assert r.exit_code == 0, r.output


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


def _set_index(path: Path, index: dict[str, Any]) -> None:
    """Give every vector of the plan.yaml at `path` the index `index`.

    The file is edited as YAML, so the plan model does not check `index`.
    """
    plan = yaml.safe_load(path.read_text(encoding="utf-8"))
    for entry in plan["collections"]:
        for vector in entry["target"]["vectors"]:
            vector["index"] = index
    path.write_text(yaml.safe_dump(plan, sort_keys=False), encoding="utf-8")


def test_inspect_refuses_a_quantizer_that_the_index_does_not_support(tmp_path):
    fakes = FakeFactories(FakeSource(cli_records(10)))
    r = invoke(
        fakes, "inspect", "chroma", "--index", "flat", "--quantizer", "sq", "--json"
    )
    assert r.exit_code == 2, r.output
    d = json.loads(r.stdout)
    assert d["code"] == "usage"
    assert "quantizer type 'sq' is not supported by index type 'flat'" in d["message"]
    assert not (tmp_path / DIR / "plan.yaml").exists()


@pytest.mark.parametrize("command", ["check", "copy"])
def test_a_quantizer_that_the_index_does_not_support_is_refused(tmp_path, command):
    fakes = inspected(FakeFactories(FakeSource(cli_records(10))))
    index = {"type": "hfresh", "quantizer": {"type": "none"}}
    _set_index(tmp_path / DIR / "plan.yaml", index)
    r = invoke(fakes, command, "--json")
    assert r.exit_code == 2, r.output
    d = json.loads(r.stdout)
    assert d["code"] == "invalid_plan"
    assert (
        "quantizer type 'none' is not supported by index type 'hfresh'"
        in (d["message"])
    )
    assert fakes.schema_calls == 0 and fakes.schema.ensured == []


def test_inspect_refuses_rq_bits_that_the_index_does_not_support(tmp_path):
    fakes = FakeFactories(FakeSource(cli_records(10)))
    args = ("--index", "hfresh", "--quantizer", "rq", "--quantizer-bits", "8")
    r = invoke(fakes, "inspect", "chroma", *args, "--json")
    assert r.exit_code == 2, r.output
    d = json.loads(r.stdout)
    assert d["code"] == "usage"
    assert (
        "bits=8 of quantizer type 'rq' is not supported by index type 'hfresh'"
        in (d["message"])
    )
    assert not (tmp_path / DIR / "plan.yaml").exists()


@pytest.mark.parametrize("command", ["check", "copy"])
def test_rq_centering_that_the_index_does_not_support_is_refused(tmp_path, command):
    fakes = inspected(FakeFactories(FakeSource(cli_records(10))))
    quantizer = {"type": "rq", "bits": 8, "centering": True}
    _set_index(tmp_path / DIR / "plan.yaml", {"type": "hnsw", "quantizer": quantizer})
    r = invoke(fakes, command, "--json")
    assert r.exit_code == 2, r.output
    d = json.loads(r.stdout)
    assert d["code"] == "invalid_plan"
    assert (
        "centering=True is supported only by index type 'hnsw' with bits=4"
        in (d["message"])
    )
    assert fakes.schema_calls == 0 and fakes.schema.ensured == []


@pytest.mark.parametrize("command", ["check", "copy"])
def test_a_quantizer_field_that_the_index_does_not_apply_is_refused(tmp_path, command):
    fakes = inspected(FakeFactories(FakeSource(cli_records(10))))
    quantizer = {"type": "bq", "rescore_limit": 30}
    _set_index(tmp_path / DIR / "plan.yaml", {"type": "hnsw", "quantizer": quantizer})
    r = invoke(fakes, command, "--json")
    assert r.exit_code == 2, r.output
    d = json.loads(r.stdout)
    assert d["code"] == "invalid_plan"
    assert (
        "rescore_limit=30 of quantizer type 'bq' is not applied by index type 'hnsw'"
    ) in d["message"]
    assert fakes.schema_calls == 0 and fakes.schema.ensured == []


def test_check_warns_that_a_dynamic_index_needs_async_indexing():
    fakes = inspected(FakeFactories(FakeSource(cli_records(10))), "--index", "dynamic")
    r = invoke(fakes, "check", "--json")
    assert r.exit_code == 0, r.output
    assert json.loads(r.stdout)["warnings"] == [
        {
            "code": "needs_async_indexing",
            "collection": "Docs",
            "message": "vector default: the dynamic index needs ASYNC_INDEXING=true "
            "on the server; check cannot read that setting",
        }
    ]


def test_check_gives_no_async_indexing_warning_without_a_dynamic_index():
    fakes = inspected(FakeFactories(FakeSource(cli_records(10))), "--index", "hnsw")
    r = invoke(fakes, "check", "--json")
    assert r.exit_code == 0, r.output
    assert json.loads(r.stdout)["warnings"] == []


def test_inspect_unknown_collection_hint_names_both_database_settings():
    fakes = FakeFactories(FakeSource(cli_records(10)))
    r = invoke(fakes, "inspect", "chroma", "--collection", "nope", "--json")
    assert r.exit_code == 2
    hint = json.loads(r.stdout)["hint"]
    assert "--database" in hint and "CHROMA_DATABASE" in hint


DB = FakeSource.NAMESPACE


# `asked` is the namespaces argument the source receives; None asks for all.
# `recorded` is the database of each plan entry. None means that inspect finds
# no collection there and exits 2.
@pytest.mark.parametrize(
    "variable, args, asked, recorded",
    [
        pytest.param(DB, [], [DB], [DB], id="variable"),
        pytest.param(DB, ["--database", "other"], ["other"], None, id="option-wins"),
        pytest.param(None, [], None, [DB], id="neither"),
    ],
)
def test_inspect_databases(tmp_path, variable, args, asked, recorded):
    source = FakeSource(cli_records(10))
    r = invoke(
        FakeFactories(source), "inspect", "chroma", *args, CHROMA_DATABASE=variable
    )
    assert source.inspected_namespaces == [asked]
    if recorded is None:
        assert r.exit_code == 2 and "in database other." in r.output, r.output
        return
    assert r.exit_code == 0, r.output
    plan = load_plan(tmp_path / DIR / "plan.yaml")
    assert [e.source.namespace for e in plan.collections] == recorded


NO_COLLECTION_HINT = "Check --database, CHROMA_DATABASE and CHROMA_TENANT."


class NoCollections:
    """The ChromaClient calls that inspect makes, for databases without collections."""

    def __init__(self, databases: list[str]) -> None:
        self.names = databases

    def databases(self, tenant: str) -> list[str]:
        return self.names

    def collections(self, tenant: str, database: str) -> list[dict[str, Any]]:
        return []


def error_of(r: Result) -> tuple[int, str, str, str]:
    d = json.loads(r.stdout)
    assert (r.exit_code, d["status"]) != (0, "success"), r.output
    return r.exit_code, d["code"], d["message"], d["hint"]


def test_inspect_that_finds_no_collection_exits_2_and_writes_no_file(tmp_path):
    fakes = FakeFactories(FakeSource(cli_records(10)))
    r = invoke(fakes, "inspect", "chroma", "--database", "nope", "--json")
    assert error_of(r) == (
        2,
        "usage",
        f"Chroma tenant {FakeSource.TENANT} has no collection in database nope.",
        NO_COLLECTION_HINT,
    )
    assert not (tmp_path / DIR).exists()
    for command in ("check", "copy", "verify", "status"):
        r = invoke(fakes, command, "--json")
        assert r.exit_code == 2 and json.loads(r.stdout)["code"] == "usage", r.output


def test_inspect_that_finds_no_collection_in_text_mode_writes_no_file(tmp_path):
    fakes = FakeFactories(FakeSource(cli_records(10)))
    r = invoke(fakes, "inspect", "chroma", "--database", "nope")
    assert r.exit_code == 2, r.output
    assert r.stderr.splitlines()[-2:] == [
        f"Error: Chroma tenant {FakeSource.TENANT} has no collection in database "
        "nope.",
        f"Hint: {NO_COLLECTION_HINT}",
    ]
    assert not (tmp_path / DIR).exists()


@pytest.mark.parametrize(
    "args, variable, searched",
    [
        pytest.param(
            ["--database", "zeta", "--database", "alpha"],
            None,
            "zeta, alpha",
            id="options",
        ),
        pytest.param([], "nope", "nope", id="variable"),
    ],
)
def test_inspect_that_finds_no_collection_names_the_databases_in_search_order(
    args, variable, searched
):
    fakes = FakeFactories(FakeSource(cli_records(10)))
    r = invoke(fakes, "inspect", "chroma", *args, "--json", CHROMA_DATABASE=variable)
    assert error_of(r)[:3] == (
        2,
        "usage",
        f"Chroma tenant {FakeSource.TENANT} has no collection in database "
        f"{searched}.",
    )


def test_inspect_of_every_database_names_them_in_the_order_chroma_lists_them():
    source = ChromaSource(NoCollections(["db-b", "db-a"]), "t")
    r = invoke(FakeFactories(source), "inspect", "chroma", "--json")  # type: ignore[arg-type]
    assert error_of(r) == (
        2,
        "usage",
        "Chroma tenant t has no collection in database db-b, db-a.",
        NO_COLLECTION_HINT,
    )
    assert not DIR.exists()


def test_inspect_of_a_tenant_without_a_database_exits_2():
    source = ChromaSource(NoCollections([]), "t")
    r = invoke(FakeFactories(source), "inspect", "chroma", "--json")  # type: ignore[arg-type]
    assert error_of(r) == (
        2,
        "usage",
        "Chroma tenant t has no database.",
        NO_COLLECTION_HINT,
    )
    assert not DIR.exists()


@pytest.mark.parametrize("command", ["check", "copy", "verify", "status"])
def test_plan_without_a_collection_is_refused_on_load(tmp_path, command):
    fakes = inspected(FakeFactories(FakeSource(cli_records(10))))
    path = tmp_path / DIR / "plan.yaml"
    save_plan(load_plan(path).model_copy(update={"collections": []}), path)
    r = invoke(fakes, command, "--json")
    assert error_of(r)[:3] == (
        2,
        "invalid_plan",
        f"{DIR / 'plan.yaml'} is not a valid plan: collections is empty.",
    )


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
    assert "plan.yaml" in d["hint"]
    assert source.read_calls == 0


def test_copy_refuses_a_module_that_the_target_lacks_before_it_creates_anything(
    tmp_path,
):
    source = FakeSource(cli_records(10))
    fakes = inspected(
        FakeFactories(source, schema=FakeSchema(modules=["text2vec-weaviate"])),
        "--vectors",
        "embed",
        "--vectorizer",
        "text2vec-openai",
    )
    r = invoke(fakes, "copy", "--json", OPENAI_API_KEY="openai-key")
    assert r.exit_code == 2, r.output
    d = json.loads(r.stdout)
    assert (d["code"], d["hint"]) == (
        "invalid_plan",
        "Edit plan.yaml, or run to-weaviate inspect chroma --force to draft it again.",
    )
    assert [(e["code"], e["message"]) for e in d["errors"]] == [
        (
            "module_not_enabled",
            "module text2vec-openai is not enabled on the target; enabled "
            "modules: text2vec-weaviate",
        )
    ]
    assert fakes.schema.ensured == [] and source.read_calls == 0
    assert not (tmp_path / DIR / "state.sqlite").exists()


def test_hint_of_a_refused_schema_tells_to_edit_the_plan():
    refused = SchemaRejected("'_additional' is a reserved property name")
    fakes = inspected(
        FakeFactories(
            FakeSource(cli_records(10)), schema=FakeSchema(ensure_error=refused)
        )
    )
    code, error, _, hint = error_of(invoke(fakes, "copy", "--json"))
    assert (code, error) == (2, "target_rejected_schema")
    assert hint == (
        "Read the server's message and edit plan.yaml. A free Weaviate Cloud "
        "cluster accepts only the index type hfresh."
    )
    assert "--force" not in hint


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


def test_second_copy_reads_only_the_records_added_since():
    fakes = inspected(FakeFactories(FakeSource(cli_records(10))))
    assert invoke(fakes, "copy").exit_code == 0
    reads, writes = fakes.source.read_calls, fakes.writer.calls
    fakes.source.records.extend(cli_records(11)[10:])
    r = invoke(fakes, "copy", "--json")
    assert r.exit_code == 0, r.output
    assert fakes.source.read_calls - reads == 1 and fakes.writer.calls - writes == 1
    assert len(fakes.writer.objects) == 11
    assert set(fakes.writer.writes_per_uuid.values()) == {1}
    assert json.loads(r.stdout)["collections"][0]["copied"] == 11


def test_every_copy_reads_the_records_added_since_the_last_run():
    fakes = inspected(FakeFactories(FakeSource(cli_records(10))))
    assert invoke(fakes, "copy").exit_code == 0
    fakes.source.records.extend(cli_records(15)[10:])
    assert invoke(fakes, "copy").exit_code == 0
    fakes.source.records.extend(cli_records(20)[15:])
    r = invoke(fakes, "copy", "--json")
    assert r.exit_code == 0, r.output
    row = json.loads(r.stdout)["collections"][0]
    assert (row["copied"], row["total"], len(fakes.writer.objects)) == (20, 20, 20)
    assert set(fakes.writer.writes_per_uuid.values()) == {1}


def copied_and_total(fakes: FakeFactories) -> tuple[int, int]:
    r = invoke(fakes, "status", "--json")
    assert r.exit_code == 0, r.output
    row = json.loads(r.stdout)["collections"][0]
    return row["copied"], row["total"]


def test_status_after_a_tail_shows_the_highest_source_count():
    fakes = inspected(FakeFactories(FakeSource(cli_records(10))))
    assert invoke(fakes, "copy").exit_code == 0
    fakes.source.records.extend(cli_records(15)[10:])
    assert invoke(fakes, "copy").exit_code == 0
    assert copied_and_total(fakes) == (15, 15)
    # Each run gives init_collection the count of inspect.json, 10.
    assert invoke(fakes, "copy").exit_code == 0
    assert copied_and_total(fakes) == (15, 15)


REFUSED = "vector lengths don't match"
BATCH_HINT = "Fix the cause and run to-weaviate copy again. It resumes at this page."


def refusing(
    records: int, fail: Callable[[int], Exception | None]
) -> tuple[FakeFactories, FakeInsertClient]:
    """Inspected fakes with a WeaviateWriter over a FakeInsertClient."""
    client = FakeInsertClient(fail)
    writer = WeaviateWriter(client, sleep=lambda s: None)  # type: ignore[arg-type]
    source = FakeSource(cli_records(records))
    return inspected(FakeFactories(source, writer=writer, store=client)), client


def test_copy_stops_when_weaviate_refuses_every_object_of_a_batch(tmp_path):
    fakes, client = refusing(10, lambda n: all_failed(f"{REFUSED}; key SENTINEL-K"))
    r = invoke(
        fakes, "copy", "--parallelism", "1", "--json", WEAVIATE_API_KEY="SENTINEL-K"
    )
    assert r.exit_code == 4
    d = json.loads(r.stdout)
    assert d["code"] == "target_unreachable" and d["hint"] == BATCH_HINT
    assert d["message"] == (
        f"Weaviate refused every object of a batch of 10 in Docs: {REFUSED}; key ***"
    )
    assert "SENTINEL-K" not in r.output
    assert client.calls == 6
    state = State(tmp_path / DIR / "state.sqlite")
    try:
        assert state.pending(FakeSource.REF) == [(Partition("0-10", "0", "10"), None)]
        assert state.progress()[0].rejects == 0
    finally:
        state.close()
    assert not (tmp_path / DIR / "rejects.jsonl").exists()


def test_copy_retries_a_batch_that_weaviate_refused_once():
    fakes, client = refusing(10, lambda n: all_failed(REFUSED) if n == 1 else None)
    r = invoke(fakes, "copy", "--json")
    assert r.exit_code == 0, r.output
    assert len(client.writes_per_uuid) == 10
    assert json.loads(r.stdout)["collections"][0]["copied"] == 10


def test_copy_after_a_refused_batch_copies_the_page():
    fakes, client = refusing(10, lambda n: all_failed(REFUSED))
    invoke(fakes, "copy", "--parallelism", "1")
    client.fail = lambda n: None
    calls = client.calls
    r = invoke(fakes, "copy", "--parallelism", "1", "--json")
    row = json.loads(r.stdout)["collections"][0]
    assert (client.calls - calls, row["copied"], row["rejects"]) == (1, 10, 0)
    assert r.exit_code == 0, r.output


class ChromaRows:
    """The ChromaClient calls of a ChromaSource, over rows of one collection."""

    COLLECTION = {"id": "chroma-docs", "name": "docs", "dimension": 2}
    COLUMNS = {
        "embeddings": "embedding",
        "documents": "document",
        "uris": "uri",
        "metadatas": "metadata",
    }

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows

    def databases(self, tenant: str) -> list[str]:
        return ["db"]

    def collections(self, tenant: str, database: str) -> list[dict[str, Any]]:
        return [self.COLLECTION]

    def count(self, tenant: str, database: str, ref: str) -> int:
        return len(self.rows)

    def get(
        self,
        tenant: str,
        database: str,
        ref: str,
        *,
        limit: int,
        offset: int,
        include: list[str],
        where: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        rows = self.rows[offset : offset + limit]
        response: dict[str, Any] = {"ids": [row["id"] for row in rows]}
        for name, key in self.COLUMNS.items():
            response[name] = [row[key] for row in rows] if name in include else None
        return response


def test_copy_rejects_a_record_whose_embedding_holds_a_null():
    rows = [
        {
            "id": f"id-{i}",
            "embedding": [float(i), 1.0],
            "document": f"doc {i}",
            "uri": None,
            "metadata": {"year": 2000 + i},
        }
        for i in range(10)
    ]
    rows[4]["embedding"] = [None, 1.0]
    fakes = inspected(FakeFactories(ChromaSource(ChromaRows(rows), "t")))  # type: ignore[arg-type]
    r = invoke(fakes, "copy", "--json")
    assert r.exit_code == 1, r.output
    row = json.loads(r.stdout)["collections"][0]
    assert (row["copied"], row["rejects"]) == (9, 1)
    assert len(fakes.writer.objects) == 9
    assert to_uuid("id-4") not in fakes.writer.objects
    reject = json.loads((DIR / "rejects.jsonl").read_text())
    assert (reject["source_id"], reject["reason"]) == (
        "id-4",
        "embedding holds a value that is not a number",
    )
    r = invoke(fakes, "verify", "--json")
    assert r.exit_code == 1 and json.loads(r.stdout)["code"] == "verification_failed"
    records = next(c for c in json.loads(r.stdout)["checks"] if c["name"] == "records")
    assert records["detail"] == (
        'sampled 10, compared 10, rejects skipped 0, differing 1: "id-4": '
        "missing in the target"
    )


def test_copy_resumes_at_the_refused_page_and_writes_each_object_once():
    fakes, client = refusing(250, lambda n: all_failed(REFUSED) if n > 1 else None)
    assert invoke(fakes, "copy", "--parallelism", "1").exit_code == 4
    assert len(client.writes_per_uuid) == 100
    client.fail = lambda n: None
    r = invoke(fakes, "copy", "--parallelism", "1", "--json")
    assert r.exit_code == 0, r.output
    assert len(client.writes_per_uuid) == 250
    assert set(client.writes_per_uuid.values()) == {1}


# (first target name, first tenant, second target name, second tenant)
ONE_PLACE = [
    ("Docs", None, "Docs", None),
    ("Docs", "a", "Docs", "a"),
    ("Docs", "a", "Docs", None),
    ("docs", None, "Docs", None),
]


def two_entries(
    directory: Path,
    first_target: tuple[str, str | None],
    second_target: tuple[str, str | None],
) -> None:
    """Rewrite plan.yaml and inspect.json with two entries from two sources."""
    plan = load_plan(directory / "plan.yaml")
    first = plan.collections[0]
    entries = []
    for ref, (name, tenant) in (
        (first.source.ref, first_target),
        ("other", second_target),
    ):
        target = first.target.model_copy(update={"name": name, "tenant": tenant})
        source = first.source.model_copy(update={"ref": ref})
        entries.append(first.model_copy(update={"source": source, "target": target}))
    save_plan(plan.model_copy(update={"collections": entries}), directory / "plan.yaml")
    found = json.loads((directory / "inspect.json").read_text())
    found["collections"].append({**found["collections"][0], "ref": "other"})
    (directory / "inspect.json").write_text(json.dumps(found))


@pytest.mark.parametrize(("name_a", "tenant_a", "name_b", "tenant_b"), ONE_PLACE)
def test_check_finds_entries_that_write_to_one_place(
    tmp_path, name_a, tenant_a, name_b, tenant_b
):
    fakes = inspected(FakeFactories(FakeSource(cli_records(10))))
    two_entries(tmp_path / DIR, (name_a, tenant_a), (name_b, tenant_b))
    r = invoke(fakes, "check", "--json")
    assert r.exit_code == 2
    errors = json.loads(r.stdout)["errors"]
    assert [(e["code"], e["collection"]) for e in errors] == [
        ("duplicate_target", "Docs")
    ]


@pytest.mark.parametrize(("name_a", "tenant_a", "name_b", "tenant_b"), ONE_PLACE)
def test_copy_refuses_entries_that_write_to_one_place(
    tmp_path, name_a, tenant_a, name_b, tenant_b
):
    fakes = inspected(FakeFactories(FakeSource(cli_records(10))))
    two_entries(tmp_path / DIR, (name_a, tenant_a), (name_b, tenant_b))
    r = invoke(fakes, "copy", "--json")
    assert r.exit_code == 2
    d = json.loads(r.stdout)
    assert d["code"] == "invalid_plan"
    assert [e["code"] for e in d["errors"]] == ["duplicate_target"]
    assert fakes.schema_calls == 0 and fakes.source.read_calls == 0


@pytest.mark.parametrize(
    ("name_a", "tenant_a", "name_b", "tenant_b"),
    [
        ("Docs", "a", "Docs", "b"),
        ("docs", "a", "Docs", "b"),
        ("Docs", None, "Other", None),
    ],
)
def test_check_accepts_entries_that_write_to_separate_places(
    tmp_path, name_a, tenant_a, name_b, tenant_b
):
    fakes = inspected(FakeFactories(FakeSource(cli_records(10))))
    two_entries(tmp_path / DIR, (name_a, tenant_a), (name_b, tenant_b))
    r = invoke(fakes, "check", "--json")
    assert r.exit_code == 0, r.output


def change_second_entry(directory: Path, **update: Any) -> None:
    """Replace fields of the target of the second plan entry."""
    plan = load_plan(directory / "plan.yaml")
    first, second = plan.collections
    target = second.target.model_copy(update=update)
    entries = [first, second.model_copy(update={"target": target})]
    save_plan(plan.model_copy(update={"collections": entries}), directory / "plan.yaml")


ENTRY_A = "docs -> Docs (tenant a)"
ENTRY_B = "docs -> Docs (tenant b)"


@pytest.mark.parametrize("cmd", ["check", "copy"])
def test_entries_of_one_collection_with_other_vectors_are_an_invalid_plan(
    tmp_path, cmd
):
    fakes = inspected(FakeFactories(FakeSource(cli_records(10))))
    two_entries(tmp_path / DIR, ("Docs", "a"), ("Docs", "b"))
    second = load_plan(tmp_path / DIR / "plan.yaml").collections[1].target
    update = {"distance": "dot", "index": IndexPlan(type="hnsw")}
    vectors = [v.model_copy(update=update) for v in second.vectors]
    change_second_entry(tmp_path / DIR, vectors=vectors)
    r = invoke(fakes, cmd, "--json")
    message = (
        f"vector default: distance cosine in {ENTRY_A}, dot in {ENTRY_B}; copy "
        "creates collection Docs with the vectors of the first entry, so give "
        "every entry of Docs the same vectors"
    )
    assert plan_errors(r) == [
        {"code": "conflicting_vectors", "collection": "Docs", "message": message}
    ]
    assert fakes.schema.ensured == [] and fakes.writer.calls == 0


def test_a_vectorizer_difference_is_named_without_its_values(tmp_path):
    fakes = inspected(FakeFactories(FakeSource(cli_records(10))))
    two_entries(tmp_path / DIR, ("Docs", "a"), ("Docs", "b"))
    second = load_plan(tmp_path / DIR / "plan.yaml").collections[1].target
    vectorizer = VectorizerPlan(
        module="text2vec-openai", options={"base_url": "https://proxy.example"}
    )
    vectors = [v.model_copy(update={"vectorizer": vectorizer}) for v in second.vectors]
    change_second_entry(tmp_path / DIR, vectors=vectors)
    r = invoke(fakes, "check", "--json", OPENAI_API_KEY="openai-key")
    (error,) = [e for e in plan_errors(r) if e["code"] == "conflicting_vectors"]
    assert error["message"].startswith(
        f"vector default: vectorizer differs between {ENTRY_A} and {ENTRY_B};"
    )
    assert "proxy.example" not in r.stdout


@pytest.mark.parametrize("cmd", ["check", "copy"])
def test_entries_of_one_collection_with_other_property_settings_are_an_invalid_plan(
    tmp_path, cmd
):
    fakes = inspected(FakeFactories(FakeSource(cli_records(10))))
    two_entries(tmp_path / DIR, ("Docs", "a"), ("Docs", "b"))
    second = load_plan(tmp_path / DIR / "plan.yaml").collections[1].target
    update = {"type": "text", "tokenization": "field", "range_filters": False}
    properties = [
        p.model_copy(update=update) if p.name == "year" else p
        for p in second.properties
    ]
    change_second_entry(tmp_path / DIR, properties=properties)
    r = invoke(fakes, cmd, "--json")
    message = (
        f"property year: type int in {ENTRY_A}, text in {ENTRY_B}; give it the "
        "same type and index settings in every entry of Docs"
    )
    assert plan_errors(r) == [
        {"code": "conflicting_property", "collection": "Docs", "message": message}
    ]
    assert fakes.schema.ensured == [] and fakes.writer.calls == 0


@pytest.mark.parametrize("cmd", ["check", "copy"])
def test_entries_of_one_collection_may_not_name_a_property_in_two_cases(tmp_path, cmd):
    fakes = inspected(FakeFactories(FakeSource(cli_records(10))))
    two_entries(tmp_path / DIR, ("Docs", "a"), ("Docs", "b"))
    second = load_plan(tmp_path / DIR / "plan.yaml").collections[1].target
    properties = [
        p.model_copy(update={"name": "yEar"}) if p.name == "year" else p
        for p in second.properties
    ]
    change_second_entry(tmp_path / DIR, properties=properties)
    r = invoke(fakes, cmd, "--json")
    message = (
        f"properties year in {ENTRY_A} and yEar in {ENTRY_B} have the same name "
        "ignoring case; Weaviate compares property names in lower case, so give "
        "the property one name in every entry of Docs"
    )
    assert r.exit_code == 2, r.output
    assert plan_errors(r) == [
        {"code": "duplicate_property", "collection": "Docs", "message": message}
    ]
    assert json.loads(r.stdout)["code"] == "invalid_plan"
    assert fakes.schema.ensured == [] and fakes.writer.calls == 0


def test_entries_of_one_collection_may_have_other_properties_and_source_keys(
    tmp_path,
):
    fakes = inspected(FakeFactories(FakeSource(cli_records(10))))
    two_entries(tmp_path / DIR, ("Docs", "a"), ("Docs", "b"))
    second = load_plan(tmp_path / DIR / "plan.yaml").collections[1].target
    properties = [
        p.model_copy(update={"coerced_from": ["text"]}) if p.name == "year" else p
        for p in second.properties
        if p.name != "uri"
    ]
    change_second_entry(tmp_path / DIR, properties=properties)
    r = invoke(fakes, "check", "--json")
    assert r.exit_code == 0, r.output


# (quantizer in plan.yaml, whether copy enables it again after the pages).
TRAINING = {
    "sq": ({"type": "sq", "training_limit": 100}, True),
    "pq": ({"type": "pq", "segments": 2, "training_limit": 100}, True),
    "rq-centering": ({"type": "rq", "bits": 4, "centering": True}, True),
    "rq": ({"type": "rq", "bits": 8}, False),
    "bq": ({"type": "bq"}, False),
    "none": ({"type": "none"}, False),
    "default": ({"type": "default"}, False),
}


@pytest.mark.parametrize(("quantizer", "sent"), TRAINING.values(), ids=TRAINING)
def test_copy_enables_a_quantizer_that_needs_training_after_the_pages(
    tmp_path, quantizer, sent
):
    fakes = inspected(FakeFactories(FakeSource(cli_records(10))))
    _set_index(tmp_path / DIR / "plan.yaml", {"type": "hnsw", "quantizer": quantizer})
    r = invoke(fakes, "copy", "--json")
    assert r.exit_code == 0, r.output
    assert fakes.schema.updated == ([("Docs", None)] if sent else [])
    assert len(fakes.writer.objects) == 10


def test_copy_enables_the_quantizer_once_per_tenant_entry(tmp_path):
    fakes = inspected(FakeFactories(FakeSource(cli_records(10))))
    _set_index(
        tmp_path / DIR / "plan.yaml", {"type": "hnsw", "quantizer": {"type": "sq"}}
    )
    two_entries(tmp_path / DIR, ("Docs", "a"), ("Docs", "b"))
    r = invoke(fakes, "copy", "--json")
    assert r.exit_code == 0, r.output
    assert fakes.schema.updated == [("Docs", "a"), ("Docs", "b")]


def test_a_refused_quantizer_update_stops_copy_and_resumes_without_pages(tmp_path):
    refused = SchemaRejected("sq is not supported", "enable the quantizers of Docs")
    fakes = inspected(
        FakeFactories(
            FakeSource(cli_records(10)), schema=FakeSchema(update_error=refused)
        )
    )
    _set_index(
        tmp_path / DIR / "plan.yaml", {"type": "hnsw", "quantizer": {"type": "sq"}}
    )
    r = invoke(fakes, "copy", "--json")
    assert r.exit_code == 2, r.output
    d = json.loads(r.stdout)
    assert d["code"] == "target_rejected_schema"
    assert d["message"] == (
        "Weaviate refused to enable the quantizers of Docs: sq is not supported"
    )
    status = json.loads(invoke(fakes, "status", "--json").stdout)["collections"][0]
    assert (status["copied"], status["done"]) == (10, False)
    reads = fakes.source.read_calls
    fakes.schema.update_error = None
    r = invoke(fakes, "copy", "--json")
    assert r.exit_code == 0, r.output
    assert fakes.source.read_calls == reads
    assert fakes.schema.updated == [("Docs", None)]
    status = json.loads(invoke(fakes, "status", "--json").stdout)["collections"][0]
    assert (status["copied"], status["done"]) == (10, True)


def test_an_entry_is_done_again_when_its_plan_stops_needing_the_update(tmp_path):
    fakes = inspected(FakeFactories(FakeSource(cli_records(10))))
    plan = tmp_path / DIR / "plan.yaml"
    rq = {"type": "rq", "bits": 4}
    _set_index(plan, {"type": "hnsw", "quantizer": rq})
    assert invoke(fakes, "copy", "--json").exit_code == 0
    # The server refuses to add centering to the existing rq index.
    fakes.schema.update_error = SchemaRejected(
        "rq centering is immutable", "enable the quantizers of Docs"
    )
    _set_index(plan, {"type": "hnsw", "quantizer": {**rq, "centering": True}})
    assert invoke(fakes, "copy", "--json").exit_code == 2
    _set_index(plan, {"type": "hnsw", "quantizer": rq})
    r = invoke(fakes, "copy", "--json")
    assert r.exit_code == 0, r.output
    assert json.loads(r.stdout)["collections"][0]["done"] is True
    status = json.loads(invoke(fakes, "status", "--json").stdout)["collections"][0]
    assert status["done"] is True


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


STATE_HINT = (
    "To resume, set WEAVIATE_URL and the target name back. To copy to the new "
    "target, use another --dir."
)


def retarget(directory: Path, name: str, tenant: str | None = None) -> None:
    """Set the target name and tenant of the only plan entry."""
    plan = load_plan(directory / "plan.yaml")
    entry = plan.collections[0]
    target = entry.target.model_copy(update={"name": name, "tenant": tenant})
    entry = entry.model_copy(update={"target": target})
    save_plan(plan.model_copy(update={"collections": [entry]}), directory / "plan.yaml")


def mismatch(r: Result) -> dict[str, Any]:
    """The error document of a state_mismatch run."""
    assert r.exit_code == 2, r.output
    d = json.loads(r.stdout)
    assert (d["status"], d["code"], d["hint"]) == (
        "error",
        "state_mismatch",
        STATE_HINT,
    )
    return d


def test_copy_refuses_a_renamed_target_after_a_full_copy(tmp_path):
    fakes = inspected(FakeFactories(FakeSource(cli_records(10))))
    assert invoke(fakes, "copy").exit_code == 0
    writes = fakes.writer.calls
    retarget(tmp_path / DIR, "Docs2")
    d = mismatch(invoke(fakes, "copy", "--json"))
    assert d["message"] == (
        "The state in .to-weaviate holds a copy of docs to localhost:8099, "
        "collection Docs. The plan and WEAVIATE_URL now name localhost:8099, "
        "collection Docs2."
    )
    assert fakes.writer.calls == writes and fakes.schema.ensured == ["Docs"]
    assert ("Docs2", None) not in fakes.writer.stored


def test_copy_refuses_another_weaviate_url(tmp_path):
    source = FakeSource(cli_records(10))
    assert invoke(inspected(FakeFactories(source)), "copy").exit_code == 0
    reads = source.read_calls
    other = FakeFactories(source)
    url = "https://admin:SENTINEL@Prod.Example.com"
    d = mismatch(invoke(other, "copy", "--json", WEAVIATE_URL=url))
    assert d["message"] == (
        "The state in .to-weaviate holds a copy of docs to localhost:8099, "
        "collection Docs. The plan and WEAVIATE_URL now name prod.example.com:443, "
        "collection Docs."
    )
    assert other.writer.calls == 0 and other.schema.ensured == []
    assert source.read_calls == reads
    assert b"SENTINEL" not in (tmp_path / DIR / "state.sqlite").read_bytes()


def test_copy_refuses_a_tenant_change(tmp_path):
    fakes = inspected(FakeFactories(FakeSource(cli_records(10))))
    retarget(tmp_path / DIR, "Docs", tenant="a")
    assert invoke(fakes, "copy").exit_code == 0
    retarget(tmp_path / DIR, "Docs", tenant="b")
    d = mismatch(invoke(fakes, "copy", "--json"))
    assert "collection Docs, tenant a. The plan" in d["message"]
    assert d["message"].endswith("collection Docs, tenant b.")


@pytest.mark.parametrize("partial", [True, False])
def test_copy_refuses_a_target_without_the_collection(partial):
    source = FakeSource(cli_records(10), page_size=2)
    writer = FakeWriter(fail_on_call=3 if partial else None)
    fakes = inspected(FakeFactories(source, writer=writer))
    first = invoke(fakes, "copy", "--parallelism", "1")
    assert first.exit_code == (5 if partial else 0), first.output
    reads = source.read_calls
    replaced = FakeFactories(source)
    d = mismatch(invoke(replaced, "copy", "--json"))
    assert d["message"] == (
        "The state in .to-weaviate holds a copy of docs to localhost:8099, "
        "collection Docs. That collection does not exist in the target now."
    )
    assert replaced.writer.calls == 0 and replaced.schema.ensured == []
    assert source.read_calls == reads


def test_copy_refuses_a_target_without_the_tenant(tmp_path):
    source = FakeSource(cli_records(10))
    fakes = inspected(FakeFactories(source))
    retarget(tmp_path / DIR, "Docs", tenant="a")
    assert invoke(fakes, "copy").exit_code == 0
    replaced = FakeFactories(source)
    replaced.writer.stored[("Docs", "b")] = {}
    d = mismatch(invoke(replaced, "copy", "--json"))
    assert d["message"].endswith(
        "collection Docs, tenant a. That collection or tenant does not exist in "
        "the target now."
    )
    assert replaced.writer.calls == 0 and replaced.schema.ensured == []


def test_copy_without_progress_accepts_a_target_without_the_collection():
    source = FakeSource(cli_records(10))
    fakes = inspected(FakeFactories(source, writer=FakeWriter(fail_on_call=1)))
    assert invoke(fakes, "copy", "--parallelism", "1").exit_code == 5
    replaced = FakeFactories(source)
    r = invoke(replaced, "copy", "--json")
    assert r.exit_code == 0, r.output
    assert len(replaced.writer.objects) == 10


class FirstReadFails(FakeSource):
    """FakeSource whose first read raises, like a source that went away."""

    def __init__(self, records: list[Record]) -> None:
        super().__init__(records)
        self.failed = False

    def read(
        self,
        collection: SourceCollection,
        partition: Partition,
        cursor: str | None,
        with_vectors: bool,
    ) -> Page:
        if not self.failed:
            self.failed = True
            raise ChromaError(None, "ConnectError: refused")
        return super().read(collection, partition, cursor, with_vectors)


def stored_target(fakes: FakeFactories) -> dict[str, Any]:
    status = json.loads(invoke(fakes, "status", "--json").stdout)
    return status["collections"][0]["stored_target"]


def test_copy_without_progress_replaces_the_stored_target(tmp_path):
    fakes = inspected(FakeFactories(FirstReadFails(cli_records(10))))
    first = invoke(fakes, "copy", "--parallelism", "1")
    assert first.exit_code == 4, first.output
    assert stored_target(fakes)["collection"] == "Docs"
    retarget(tmp_path / DIR, "Docs2")
    r = invoke(fakes, "copy", "--json")
    assert r.exit_code == 0, r.output
    assert json.loads(r.stdout)["collections"][0]["copied"] == 10
    assert len(fakes.writer.stored[("Docs2", None)]) == 10
    assert ("Docs", None) not in fakes.writer.stored
    assert stored_target(fakes) == {
        "host": "localhost:8099",
        "collection": "Docs2",
        "tenant": None,
    }


def test_copy_refuses_a_target_that_the_state_holds_for_another_source(tmp_path):
    fakes = inspected(FakeFactories(FakeSource(cli_records(10))))
    assert invoke(fakes, "copy").exit_code == 0
    writes = fakes.writer.calls
    plan = load_plan(tmp_path / DIR / "plan.yaml")
    entry = plan.collections[0]
    source = entry.source.model_copy(update={"ref": "other", "name": "other"})
    entries = [entry.model_copy(update={"source": source})]
    save_plan(
        plan.model_copy(update={"collections": entries}), tmp_path / DIR / "plan.yaml"
    )
    found = json.loads((tmp_path / DIR / "inspect.json").read_text())
    found["collections"].append(
        {**found["collections"][0], "ref": "other", "name": "other"}
    )
    (tmp_path / DIR / "inspect.json").write_text(json.dumps(found))
    d = mismatch(invoke(fakes, "copy", "--json"))
    assert d["message"] == (
        "The state in .to-weaviate holds a copy of docs to localhost:8099, "
        "collection Docs. The plan now copies other there."
    )
    assert fakes.writer.calls == writes


def duplicate_source(directory: Path) -> None:
    """Add a second plan entry for the source of the first, into Docs2."""
    plan = load_plan(directory / "plan.yaml")
    first = plan.collections[0]
    twin = first.model_copy(
        update={"target": first.target.model_copy(update={"name": "Docs2"})}
    )
    save_plan(
        plan.model_copy(update={"collections": [first, twin]}),
        directory / "plan.yaml",
    )


def test_check_refuses_two_entries_of_one_source(tmp_path):
    fakes = inspected(FakeFactories(FakeSource(cli_records(10))))
    duplicate_source(tmp_path / DIR)
    r = invoke(fakes, "check", "--json")
    assert r.exit_code == 2, r.output
    d = json.loads(r.stdout)
    assert d["code"] == "invalid_plan"
    assert [(e["code"], e["collection"]) for e in d["errors"]] == [
        ("duplicate_source", "Docs"),
        ("duplicate_source", "Docs2"),
    ]


@pytest.mark.parametrize("names", [[], ["--collection", "Docs2"]])
def test_copy_refuses_two_entries_of_one_source(tmp_path, names):
    fakes = inspected(FakeFactories(FakeSource(cli_records(10))))
    duplicate_source(tmp_path / DIR)
    r = invoke(fakes, "copy", "--json", *names)
    assert r.exit_code == 2, r.output
    d = json.loads(r.stdout)
    assert d["code"] == "invalid_plan"
    assert {e["code"] for e in d["errors"]} == {"duplicate_source"}
    assert fakes.schema_calls == 0 and fakes.source.read_calls == 0


# (URL of the first run, URL of the resume, WEAVIATE_GRPC_URL, stored host)
SAME_URLS = [
    (
        "http://localhost:8099",
        "HTTP://LocalHost:8099",
        "http://localhost:50099",
        "localhost:8099",
    ),
    (
        "http://localhost:8099",
        "http://localhost:8099/",
        "http://localhost:50099",
        "localhost:8099",
    ),
    (
        "http://localhost:8099",
        "http://localhost.:8099",
        "http://localhost:50099",
        "localhost:8099",
    ),
    (
        "http://weaviate.test",
        "http://weaviate.test:80",
        "http://weaviate.test:50099",
        "weaviate.test:80",
    ),
    ("https://c.weaviate.cloud", "c.weaviate.cloud", None, "c.weaviate.cloud:443"),
    # Without WEAVIATE_GRPC_URL the client connects to port 443 over TLS.
    (
        "http://c.weaviate.cloud",
        "https://c.weaviate.cloud",
        None,
        "c.weaviate.cloud:443",
    ),
    (
        "https://c.weaviate.cloud",
        "http://c.weaviate.cloud",
        None,
        "c.weaviate.cloud:443",
    ),
    (
        "https://c.weaviate.cloud/",
        "https://c.weaviate.cloud:443",
        None,
        "c.weaviate.cloud:443",
    ),
    (
        "localhost:8099",
        "http://localhost:8099",
        "http://localhost:50099",
        "localhost:8099",
    ),
]


@pytest.mark.parametrize(("first", "second", "grpc", "host"), SAME_URLS)
def test_copy_resumes_with_the_same_url_in_another_form(first, second, grpc, host):
    source = FakeSource(cli_records(10), page_size=2)
    fakes = inspected(FakeFactories(source, writer=FakeWriter(fail_on_call=3)))
    r = invoke(
        fakes, "copy", "--parallelism", "1", WEAVIATE_URL=first, WEAVIATE_GRPC_URL=grpc
    )
    assert r.exit_code == 5, r.output
    r = invoke(fakes, "copy", "--json", WEAVIATE_URL=second, WEAVIATE_GRPC_URL=grpc)
    assert r.exit_code == 0, r.output
    assert json.loads(r.stdout)["collections"][0]["copied"] == 10
    assert len(fakes.writer.objects) == 10
    status = json.loads(invoke(fakes, "status", "--json").stdout)
    assert status["collections"][0]["stored_target"]["host"] == host


def test_copy_resumes_a_state_without_targets_and_stores_the_target(tmp_path):
    fakes = inspected(FakeFactories(FakeSource(cli_records(10))))
    old_state_file(
        tmp_path / DIR / "state.sqlite",
        FakeSource.REF,
        10,
        Partition("0-10", "0", "10"),
        cursor="5",
        copied=5,
    )
    # The earlier run created the collection in the target.
    fakes.writer.stored[("Docs", None)] = {}
    r = invoke(fakes, "copy", "--json")
    assert r.exit_code == 0, r.output
    assert json.loads(r.stdout)["collections"][0]["copied"] == 10
    written = sorted(o.properties["source_id"] for o in fakes.writer.objects.values())
    assert written == [f"id-{i}" for i in range(5, 10)]
    status = json.loads(invoke(fakes, "status", "--json").stdout)
    assert status["collections"][0]["stored_target"] == {
        "host": "localhost:8099",
        "collection": "Docs",
        "tenant": None,
    }
    retarget(tmp_path / DIR, "Docs2")
    mismatch(invoke(fakes, "copy", "--json"))


def test_status_prints_the_stored_target(tmp_path):
    fakes = inspected(FakeFactories(FakeSource(cli_records(10))))
    retarget(tmp_path / DIR, "Docs", tenant="a")
    url = "http://admin:SENTINEL@LocalHost:8099/"
    assert invoke(fakes, "copy", WEAVIATE_URL=url).exit_code == 0
    text = invoke(fakes, "status", WEAVIATE_URL=None, CHROMA_API_KEY=None)
    assert text.exit_code == 0, text.output
    assert text.stdout.splitlines()[:2] == [
        "docs -> Docs: copied 10 of 10, rejects 0, done",
        "  stored target: localhost:8099, collection Docs, tenant a",
    ]
    js = invoke(fakes, "status", "--json", WEAVIATE_URL=None, CHROMA_API_KEY=None)
    assert json.loads(js.stdout)["collections"][0]["stored_target"] == {
        "host": "localhost:8099",
        "collection": "Docs",
        "tenant": "a",
    }
    assert "SENTINEL" not in text.output + js.output


LOCK_MESSAGE = "Another to-weaviate copy uses .to-weaviate."
LOCK_HINT = "Wait for it to end, or use another --dir."


@contextmanager
def held_lock(directory: Path) -> Iterator[None]:
    """Hold the copy lock of `directory` the way a running copy holds it."""
    fcntl = pytest.importorskip("fcntl")
    with (directory / "copy.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def test_copy_refuses_a_directory_that_another_copy_holds(tmp_path):
    pytest.importorskip("fcntl")
    source = FakeSource(cli_records(10), page_size=2)
    fakes = inspected(FakeFactories(source, writer=FakeWriter(fail_on_call=3)))
    assert invoke(fakes, "copy", "--parallelism", "1").exit_code == 5
    before = invoke(fakes, "status", "--json").stdout
    calls = (source.read_calls, fakes.writer.calls, fakes.schema_calls)
    with held_lock(tmp_path / DIR):
        r = invoke(fakes, "copy", "--json")
    assert r.exit_code == 2, r.output
    d = json.loads(r.stdout)
    assert (d["code"], d["message"], d["hint"]) == ("usage", LOCK_MESSAGE, LOCK_HINT)
    assert invoke(fakes, "status", "--json").stdout == before
    assert (source.read_calls, fakes.writer.calls, fakes.schema_calls) == calls
    # The lock file stays, and a lock file that nobody holds does not block.
    r = invoke(fakes, "copy", "--json")
    assert r.exit_code == 0, r.output
    assert json.loads(r.stdout)["collections"][0]["copied"] == 10


def test_status_and_verify_run_while_a_copy_holds_the_directory(tmp_path):
    pytest.importorskip("fcntl")
    fakes = inspected(FakeFactories(FakeSource(cli_records(10))))
    assert invoke(fakes, "copy").exit_code == 0
    with held_lock(tmp_path / DIR):
        status = invoke(fakes, "status", "--json")
        verify = invoke(fakes, "verify", "--json")
    assert status.exit_code == 0, status.output
    assert verify.exit_code == 0, verify.output


class LockProbe(FakeSource):
    """FakeSource that tries to take the copy lock on each read and counts it."""

    def __init__(self, records: list[Record], lock_path: Path) -> None:
        super().__init__(records, page_size=2)
        self.lock_path = lock_path
        self.refused = 0
        self.taken = 0

    def read(
        self,
        collection: SourceCollection,
        partition: Partition,
        cursor: str | None,
        with_vectors: bool,
    ) -> Page:
        try:
            with exclusive_lock(self.lock_path):
                self.taken += 1
        except Locked:
            self.refused += 1
        return super().read(collection, partition, cursor, with_vectors)


def test_copy_holds_the_lock_while_it_copies_records(tmp_path):
    pytest.importorskip("fcntl")
    source = LockProbe(cli_records(10), tmp_path / DIR / "copy.lock")
    fakes = inspected(FakeFactories(source))
    r = invoke(fakes, "copy", "--json", "--parallelism", "1")
    assert r.exit_code == 0, r.output
    assert json.loads(r.stdout)["collections"][0]["copied"] == 10
    assert len(fakes.writer.objects) == 10
    assert source.read_calls == 5
    assert (source.refused, source.taken) == (5, 0)


def edit_target(directory: Path, **update: Any) -> None:
    """Replace fields of the target of the first plan entry."""
    path = directory / "plan.yaml"
    plan = load_plan(path)
    first = plan.collections[0]
    entry = first.model_copy(update={"target": first.target.model_copy(update=update)})
    entries = [entry, *plan.collections[1:]]
    save_plan(plan.model_copy(update={"collections": entries}), path)


def plan_errors(r: Result) -> list[dict[str, str]]:
    d = json.loads(r.stdout)
    assert (r.exit_code, d.get("code")) == (2, "invalid_plan"), r.output
    return d["errors"]


TYPES = "boolean, boolean[], int, int[], number, number[], text, text[]"
DISTANCES = "cosine, dot, hamming, l2-squared, manhattan"


@pytest.mark.parametrize("cmd", ["check", "copy"])
def test_a_property_type_without_a_mapping_is_an_invalid_plan(tmp_path, cmd):
    fakes = inspected(FakeFactories(FakeSource(cli_records(10))))
    target = load_plan(tmp_path / DIR / "plan.yaml").collections[0].target
    properties = [
        p.model_copy(update={"type": "date"}) if p.name == "year" else p
        for p in target.properties
    ]
    edit_target(tmp_path / DIR, properties=properties)
    r = invoke(fakes, cmd, "--json")
    message = f"property year: type date is not one of {TYPES}"
    assert plan_errors(r) == [
        {"code": "invalid_type", "collection": "Docs", "message": message}
    ]
    assert fakes.writer.calls == 0


@pytest.mark.parametrize("cmd", ["check", "copy"])
def test_a_distance_without_a_mapping_is_an_invalid_plan(tmp_path, cmd):
    fakes = inspected(FakeFactories(FakeSource(cli_records(10))), "--index", "hnsw")
    target = load_plan(tmp_path / DIR / "plan.yaml").collections[0].target
    vectors = [v.model_copy(update={"distance": "Cosine"}) for v in target.vectors]
    edit_target(tmp_path / DIR, vectors=vectors)
    r = invoke(fakes, cmd, "--json")
    message = f"vector default: distance Cosine is not one of {DISTANCES}"
    assert plan_errors(r) == [
        {"code": "invalid_distance", "collection": "Docs", "message": message}
    ]
    assert fakes.writer.calls == 0


@pytest.mark.parametrize("cmd", ["check", "copy"])
def test_a_distance_without_a_mapping_and_the_default_index_is_an_invalid_plan(
    tmp_path, cmd
):
    fakes = inspected(FakeFactories(FakeSource(cli_records(10))))
    target = load_plan(tmp_path / DIR / "plan.yaml").collections[0].target
    assert [v.index.type for v in target.vectors] == ["default"]
    vectors = [v.model_copy(update={"distance": "Cosine"}) for v in target.vectors]
    edit_target(tmp_path / DIR, vectors=vectors)
    r = invoke(fakes, cmd, "--json")
    message = f"vector default: distance Cosine is not one of {DISTANCES}"
    assert plan_errors(r) == [
        {"code": "invalid_distance", "collection": "Docs", "message": message}
    ]
    assert fakes.writer.calls == 0


def _with_pq_segments(vectors: list[Any], segments: int) -> list[Any]:
    quantizer = QuantizerPlan(type="pq", segments=segments)
    return [
        v.model_copy(
            update={"index": v.index.model_copy(update={"quantizer": quantizer})}
        )
        for v in vectors
    ]


@pytest.mark.parametrize("cmd", ["check", "copy"])
def test_pq_segments_that_do_not_divide_the_dimension_are_an_invalid_plan(
    tmp_path, cmd
):
    fakes = inspected(FakeFactories(FakeSource(cli_records(10))), "--index", "hnsw")
    target = load_plan(tmp_path / DIR / "plan.yaml").collections[0].target
    edit_target(tmp_path / DIR, vectors=_with_pq_segments(target.vectors, 3))
    r = invoke(fakes, cmd, "--json")
    message = (
        "vector default: pq segments 3 do not divide the dimension 2 of the "
        "source vectors"
    )
    assert plan_errors(r) == [
        {"code": "invalid_quantizer", "collection": "Docs", "message": message}
    ]
    assert fakes.writer.calls == 0 and fakes.schema.ensured == []


def test_pq_segments_of_an_embedded_vector_are_not_compared_with_the_source(
    tmp_path,
):
    fakes = inspected(
        FakeFactories(FakeSource(cli_records(10))),
        "--vectors",
        "both",
        "--index",
        "hnsw",
    )
    target = load_plan(tmp_path / DIR / "plan.yaml").collections[0].target
    assert [v.mode for v in target.vectors] == ["keep", "embed"]
    kept, embedded = target.vectors
    vectors = [*_with_pq_segments([kept], 2), *_with_pq_segments([embedded], 3)]
    edit_target(tmp_path / DIR, vectors=vectors)
    r = invoke(fakes, "check", "--json")
    assert r.exit_code == 0, r.output


@pytest.mark.parametrize("cmd", ["check", "copy"])
def test_hfresh_with_a_dot_distance_is_an_invalid_plan(tmp_path, cmd):
    fakes = inspected(FakeFactories(FakeSource(cli_records(10))), "--index", "hfresh")
    target = load_plan(tmp_path / DIR / "plan.yaml").collections[0].target
    vectors = [v.model_copy(update={"distance": "dot"}) for v in target.vectors]
    edit_target(tmp_path / DIR, vectors=vectors)
    r = invoke(fakes, cmd, "--json")
    message = (
        "vector default: distance dot is not supported by index type hfresh; "
        "it supports cosine, l2-squared"
    )
    assert plan_errors(r) == [
        {"code": "invalid_distance", "collection": "Docs", "message": message}
    ]
    assert fakes.writer.calls == 0 and fakes.schema.ensured == []


@pytest.mark.parametrize("distance", ["cosine", "l2-squared", None])
def test_hfresh_with_a_cosine_or_l2_distance_passes_check(tmp_path, distance):
    fakes = inspected(FakeFactories(FakeSource(cli_records(10))), "--index", "hfresh")
    target = load_plan(tmp_path / DIR / "plan.yaml").collections[0].target
    vectors = [v.model_copy(update={"distance": distance}) for v in target.vectors]
    edit_target(tmp_path / DIR, vectors=vectors)
    r = invoke(fakes, "check", "--json")
    assert r.exit_code == 0, r.output


@pytest.mark.parametrize("cmd", ["check", "copy", "status", "verify"])
def test_a_plan_that_is_not_utf8_is_an_invalid_plan(tmp_path, cmd):
    fakes = inspected(FakeFactories(FakeSource(cli_records(10))))
    path = tmp_path / DIR / "plan.yaml"
    path.write_bytes(path.read_bytes().replace(b"name: docs", b"name: d\xf6cs"))
    r = invoke(fakes, cmd, "--json")
    assert error_of(r)[:3] == (
        2,
        "invalid_plan",
        f"{DIR / 'plan.yaml'} is not UTF-8 text.",
    )


@pytest.mark.parametrize("cmd", ["check", "copy", "verify"])
def test_an_inspection_that_is_not_utf8_is_an_invalid_plan(tmp_path, cmd):
    fakes = inspected(FakeFactories(FakeSource(cli_records(10))))
    path = tmp_path / DIR / "inspect.json"
    path.write_bytes(path.read_bytes().replace(b'"docs"', b'"d\xf6cs"'))
    r = invoke(fakes, cmd, "--json")
    assert error_of(r)[:3] == (
        2,
        "invalid_plan",
        f"{DIR / 'inspect.json'} is not UTF-8 text.",
    )


LOADING_COMMANDS = ["check", "copy", "status", "verify"]


@pytest.mark.parametrize("cmd", LOADING_COMMANDS)
def test_property_names_that_differ_only_in_case_are_an_invalid_plan(tmp_path, cmd):
    fakes = inspected(FakeFactories(FakeSource(cli_records(10))))
    target = load_plan(tmp_path / DIR / "plan.yaml").collections[0].target
    year = next(p for p in target.properties if p.name == "year")
    # A first letter in upper case would also be an invalid_name in check.
    twin = year.model_copy(update={"name": "yeaR"})
    edit_target(tmp_path / DIR, properties=[*target.properties, twin])
    r = invoke(fakes, cmd, "--json")
    message = (
        "properties year, yeaR have the same name ignoring case; give each "
        "property its own name"
    )
    assert plan_errors(r) == [
        {"code": "duplicate_property", "collection": "Docs", "message": message}
    ]
    assert fakes.writer.calls == 0


@pytest.mark.parametrize("cmd", LOADING_COMMANDS)
def test_a_repeated_vector_name_is_an_invalid_plan(tmp_path, cmd):
    fakes = inspected(FakeFactories(FakeSource(cli_records(10))), "--vectors", "both")
    target = load_plan(tmp_path / DIR / "plan.yaml").collections[0].target
    vectors = [v.model_copy(update={"name": "default"}) for v in target.vectors]
    edit_target(tmp_path / DIR, vectors=vectors)
    r = invoke(fakes, cmd, "--json")
    message = "vector name default is used 2 times; give each vector its own name"
    assert plan_errors(r) == [
        {"code": "duplicate_vector", "collection": "Docs", "message": message}
    ]
    assert fakes.writer.calls == 0


@pytest.mark.parametrize("cmd", LOADING_COMMANDS)
def test_a_key_that_appears_twice_in_one_mapping_is_an_invalid_plan(tmp_path, cmd):
    fakes = inspected(FakeFactories(FakeSource(cli_records(10))))
    path = tmp_path / DIR / "plan.yaml"
    path.write_text(path.read_text().replace("version: 1\n", "version: 1\n" * 2, 1))
    r = invoke(fakes, cmd, "--json")
    code, message = error_of(r)[1:3]
    assert (r.exit_code, code) == (2, "invalid_plan")
    assert "line 2, column 1" in message
    assert "found duplicate key ..., first at line 1" in message


@pytest.mark.parametrize(
    "args", [["check"], ["copy"], ["copy", "--collection", "Docs"], ["status"]]
)
def test_an_id_property_that_is_not_source_id_is_an_invalid_plan(tmp_path, args):
    fakes = inspected(FakeFactories(FakeSource(cli_records(10))))
    target = load_plan(tmp_path / DIR / "plan.yaml").collections[0].target
    renames = {"source_id": "chroma_id", "year": "source_id"}
    properties = [
        p.model_copy(update={"name": renames.get(p.name, p.name)})
        for p in target.properties
    ]
    edit_target(tmp_path / DIR, properties=properties)
    r = invoke(fakes, *args, "--json")
    assert plan_errors(r) == [
        {
            "code": "missing_id_property",
            "collection": "Docs",
            "message": "property chroma_id has source_key #id; name it source_id",
        },
        {
            "code": "duplicate_id_property",
            "collection": "Docs",
            "message": "property source_id has source_key year; only the property "
            "with source_key #id can be named source_id",
        },
    ]
    assert fakes.writer.calls == 0 and fakes.source.read_calls == 0


def test_copy_of_another_entry_refuses_a_plan_without_an_id_property(tmp_path):
    fakes = inspected(FakeFactories(FakeSource(cli_records(10))))
    two_entries(tmp_path / DIR, ("Docs", None), ("Other", None))
    target = load_plan(tmp_path / DIR / "plan.yaml").collections[0].target
    properties = [p for p in target.properties if p.source_key != "#id"]
    edit_target(tmp_path / DIR, properties=properties)
    r = invoke(fakes, "copy", "--collection", "Other", "--json")
    message = (
        "no property has source_key #id; add the property source_id with "
        "source_key #id"
    )
    assert plan_errors(r) == [
        {"code": "missing_id_property", "collection": "Docs", "message": message}
    ]
    assert fakes.writer.calls == 0 and fakes.source.read_calls == 0


UNMAPPED_YEAR = (
    "metadata key year is not in plan.yaml; add a property for it or list it "
    "under dropped"
)


@pytest.mark.parametrize("cmd", ["check", "copy"])
def test_a_key_that_the_plan_neither_maps_nor_drops_is_an_invalid_plan(tmp_path, cmd):
    fakes = inspected(FakeFactories(FakeSource(cli_records(10))))
    target = load_plan(tmp_path / DIR / "plan.yaml").collections[0].target
    properties = [p for p in target.properties if p.name != "year"]
    edit_target(tmp_path / DIR, properties=properties)
    r = invoke(fakes, cmd, "--json")
    assert plan_errors(r) == [
        {"code": "unmapped_key", "collection": "Docs", "message": UNMAPPED_YEAR}
    ]
    assert fakes.writer.calls == 0 and fakes.source.read_calls == 0


def test_copy_resumes_after_an_unmapped_key_is_listed_under_dropped(tmp_path):
    source = FakeSource(cli_records(10), page_size=2)
    fakes = inspected(FakeFactories(source, writer=FakeWriter(fail_on_call=3)))
    assert invoke(fakes, "copy", "--parallelism", "1").exit_code == 5
    target = load_plan(tmp_path / DIR / "plan.yaml").collections[0].target
    properties = [p for p in target.properties if p.name != "year"]
    edit_target(tmp_path / DIR, properties=properties)
    assert plan_errors(invoke(fakes, "copy", "--json"))[0]["code"] == "unmapped_key"
    dropped = [*target.dropped, DroppedKey(source_key="year", reason="not needed")]
    edit_target(tmp_path / DIR, properties=properties, dropped=dropped)
    assert invoke(fakes, "check").exit_code == 0
    r = invoke(fakes, "copy", "--json")
    assert r.exit_code == 0, r.output
    assert json.loads(r.stdout)["collections"][0]["copied"] == 10
    assert len(fakes.writer.objects) == 10


def test_check_refuses_target_names_that_differ_only_in_case(tmp_path):
    fakes = inspected(FakeFactories(FakeSource(cli_records(10))))
    two_entries(tmp_path / DIR, ("docs", None), ("dOcs", None))
    for args in (["check"], ["copy", "--collection", "dOcs"]):
        r = invoke(fakes, *args, "--json")
        assert [(e["code"], e["collection"]) for e in plan_errors(r)] == [
            ("duplicate_collection_name", "Docs")
        ]
    assert fakes.writer.calls == 0 and fakes.source.read_calls == 0


@pytest.mark.parametrize("mode", [[], ["--json"]])
def test_check_warns_on_a_property_name_with_a_reserved_suffix(tmp_path, mode):
    fakes = inspected(FakeFactories(FakeSource(cli_records(10))))
    target = load_plan(tmp_path / DIR / "plan.yaml").collections[0].target
    properties = [
        p.model_copy(update={"name": "year_temp"}) if p.name == "year" else p
        for p in target.properties
    ]
    edit_target(tmp_path / DIR, properties=properties)
    r = invoke(fakes, "check", *mode)
    assert r.exit_code == 0, r.output
    message = (
        "property year_temp ends in _temp; Weaviate 1.35.20, 1.36.15, 1.37.5 and "
        "later refuse to create a property with this suffix"
    )
    if mode:
        assert json.loads(r.stdout)["warnings"] == [
            {"code": "reserved_suffix", "collection": "Docs", "message": message}
        ]
    else:
        assert f"warning reserved_suffix Docs: {message}" in r.stdout


# The client warns on every text2vec_aws call.
@pytest.mark.filterwarnings("ignore:`text2vec_aws` is deprecated:DeprecationWarning")
@pytest.mark.parametrize("cmd", ["check", "copy"])
def test_a_vectorizer_without_its_required_options_is_an_invalid_plan(cmd):
    fakes = inspected(
        FakeFactories(
            FakeSource(cli_records(10)), schema=FakeSchema(modules=["text2vec-aws"])
        ),
        "--vectors",
        "embed",
        "--vectorizer",
        "text2vec-aws",
    )
    r = invoke(fakes, cmd, "--json")
    message = "vectorizer text2vec-aws needs the options model, region"
    assert plan_errors(r) == [
        {"code": "invalid_vectorizer", "collection": "Docs", "message": message}
    ]
    assert fakes.writer.calls == 0


@pytest.mark.parametrize("mode", [[], ["--json"]])
def test_check_warns_on_a_url_option_sent_without_its_query(mode):
    url = "https://proxy.example.com/v1?api-version=B6-MADE-UP-TOKEN"
    fakes = inspected(
        FakeFactories(FakeSource(cli_records(10))),
        "--vectors",
        "embed",
        "--vectorizer",
        "text2vec-openai",
        "--vectorizer-option",
        f"base_url={url}",
    )
    r = invoke(fakes, "check", *mode, OPENAI_API_KEY="openai-key")
    assert r.exit_code == 0, r.output
    message = (
        "option base_url of vector default in Docs holds userinfo, a query or a "
        "fragment; it is sent without them."
    )
    if mode:
        assert json.loads(r.stdout)["warnings"] == [
            {"code": "cleaned_option", "collection": "Docs", "message": message}
        ]
    else:
        assert f"warning cleaned_option Docs: {message}" in r.stdout
    assert "B6-MADE-UP-TOKEN" not in r.output


@pytest.mark.parametrize("debug", [None, "1"])
@pytest.mark.parametrize("mode", [[], ["--json"]])
@pytest.mark.parametrize("cmd", LOADING_COMMANDS)
def test_a_yaml_error_names_the_line_and_column_but_not_the_line_text(
    tmp_path, cmd, mode, debug
):
    fakes = inspected(FakeFactories(FakeSource(cli_records(10))))
    path = tmp_path / DIR / "plan.yaml"
    password = "B6-MADE-UP-PW"
    bad = f"notes: https://bob:{password}@proxy.example.com/v1: bad\n"
    path.write_text(bad + path.read_text())
    r = invoke(fakes, cmd, *mode, TO_WEAVIATE_DEBUG=debug)
    assert r.exit_code == 2, r.output
    message = (
        f"{DIR / 'plan.yaml'} is not a valid plan: YAML error at line 1, column "
        f"{bad.index(': bad') + 1}: mapping values are not allowed here"
    )
    if mode:
        d = json.loads(r.stdout)
        assert (d["code"], d["message"]) == ("invalid_plan", message)
    else:
        assert f"Error: {message}" in r.stderr
    for part in ["bob", *(password[i : i + 4] for i in range(len(password) - 3))]:
        assert part not in r.stdout and part not in r.stderr


@pytest.mark.parametrize("debug", [None, "1"])
@pytest.mark.parametrize("mode", [[], ["--json"]])
@pytest.mark.parametrize(
    ("bad", "problem"),
    [
        ("notes: *B6-MADE-UP-PW\n", "found undefined alias ..."),
        (
            "notes: !B6-MADE-UP-PW x\n",
            "could not determine a constructor for the tag ...",
        ),
    ],
)
def test_a_yaml_error_does_not_print_an_alias_or_tag_name(
    tmp_path, bad, problem, mode, debug
):
    fakes = inspected(FakeFactories(FakeSource(cli_records(10))))
    path = tmp_path / DIR / "plan.yaml"
    path.write_text(bad + path.read_text())
    r = invoke(fakes, "check", *mode, TO_WEAVIATE_DEBUG=debug)
    assert r.exit_code == 2, r.output
    message = (
        f"{DIR / 'plan.yaml'} is not a valid plan: YAML error at line 1, column 8: "
        f"{problem}"
    )
    if mode:
        d = json.loads(r.stdout)
        assert (d["code"], d["message"]) == ("invalid_plan", message)
    else:
        assert f"Error: {message}" in r.stderr
    password = "B6-MADE-UP-PW"
    for part in (password[i : i + 4] for i in range(len(password) - 3)):
        assert part not in r.stdout and part not in r.stderr


FORBIDDEN = "Weaviate refused the request: the API key lacks a permission."


def test_copy_creates_nothing_when_the_key_may_not_create_the_collection():
    forbidden = InsufficientPermissionsError(httpx.Response(403))
    source = FakeSource(cli_records(10))
    fakes = inspected(FakeFactories(source, schema=FakeSchema(ensure_error=forbidden)))
    r = invoke(fakes, "copy", "--json")
    assert r.exit_code == 3, r.output
    d = json.loads(r.stdout)
    assert (d["code"], d["message"]) == ("auth_failed", FORBIDDEN)
    assert "WEAVIATE_API_KEY" in d["hint"]
    assert fakes.writer.calls == 0
    row = json.loads(invoke(fakes, "status", "--json").stdout)["collections"][0]
    assert row["copied"] == 0 and "stored_target" not in row


def test_a_403_on_insert_is_not_retried_and_copy_resumes_at_its_page():
    def fail(n: int) -> Exception | None:
        return InsufficientPermissionsError(httpx.Response(403)) if n == 2 else None

    fakes, client = refusing(250, fail)
    r = invoke(fakes, "copy", "--parallelism", "1", "--json")
    assert r.exit_code == 3, r.output
    assert json.loads(r.stdout)["message"] == FORBIDDEN
    assert (client.calls, len(client.writes_per_uuid)) == (2, 100)
    client.fail = lambda n: None
    r = invoke(fakes, "copy", "--parallelism", "1", "--json")
    assert r.exit_code == 0, r.output
    assert len(client.writes_per_uuid) == 250
    assert set(client.writes_per_uuid.values()) == {1}


LENGTH_MESSAGE = "new node has a vector with length 1. Existing nodes have length 2"


def test_an_object_that_weaviate_refuses_is_a_reject_with_its_message(tmp_path):
    records = cli_records(21)
    records[7] = replace(records[7], vector=[7.0])

    def reject(obj: Any) -> str | None:
        return LENGTH_MESSAGE if len(obj.vector["default"]) != 2 else None

    client = FakeInsertClient(reject=reject)
    writer = WeaviateWriter(client, sleep=lambda s: None)  # type: ignore[arg-type]
    fakes = inspected(FakeFactories(FakeSource(records), writer=writer, store=client))
    r = invoke(fakes, "copy", "--json")
    assert r.exit_code == 1 and json.loads(r.stdout)["code"] == "rejects", r.output
    lines = (tmp_path / DIR / "rejects.jsonl").read_text().splitlines()
    found = [json.loads(line) for line in lines]
    assert [(f["source_id"], f["reason"]) for f in found] == [("id-7", LENGTH_MESSAGE)]
    assert len(client.writes_per_uuid) == 20
    assert to_uuid("id-7") not in client.writes_per_uuid
    r = invoke(fakes, "verify", "--json")
    assert r.exit_code == 1, r.output
    checks = {c["name"]: c for c in json.loads(r.stdout)["checks"]}
    assert not checks["count"]["passed"]
    assert checks["ids"]["detail"] == (
        f'missing 1, extra 0; missing ids ["id-7"]; {TARGET_BELOW_SOURCE}'
    )


def test_the_rejects_hint_says_how_to_copy_a_rejected_record_again():
    writer = FakeWriter(reject_ids={"id-3"})
    fakes = inspected(FakeFactories(FakeSource(cli_records(10)), writer=writer))
    r = invoke(fakes, "copy", "--json")
    assert r.exit_code == 1, r.output
    d = json.loads(r.stdout)
    assert (d["code"], d["hint"]) == (
        "rejects",
        "See .to-weaviate/rejects.jsonl for each record and the reason. Rejects "
        "are final for this directory. After fixing the cause, copy plan.yaml "
        "and inspect.json to a new --dir and run to-weaviate copy there; "
        "objects already in Weaviate are written again unchanged.",
    )


MIB = 1024 * 1024


def test_an_object_above_the_message_limit_is_a_reject(tmp_path):
    records = cli_records(6)
    records[2] = replace(records[2], document="x" * (5 * MIB))

    def refuse(objects: list[Any]) -> None:
        if any(len(o.properties.get("document") or "") > MIB for o in objects):
            raise_grpc_failure("RESOURCE_EXHAUSTED", WeaviateBatchError, batch=True)

    client = FakeInsertClient(refuse=refuse)
    writer = WeaviateWriter(client, sleep=lambda s: None)  # type: ignore[arg-type]
    fakes = inspected(FakeFactories(FakeSource(records), writer=writer, store=client))
    for _ in range(2):
        r = invoke(fakes, "copy", "--json")
        assert r.exit_code == 1, r.output
        row = json.loads(r.stdout)["collections"][0]
        assert (row["copied"], row["rejects"]) == (5, 1)
    reject = json.loads((tmp_path / DIR / "rejects.jsonl").read_text())
    assert (reject["source_id"], reject["reason"]) == (
        "id-2",
        "fake gRPC failure RESOURCE_EXHAUSTED",
    )
    assert to_uuid("id-2") not in client.writes_per_uuid
    assert len(client.writes_per_uuid) == 5


def test_two_source_ids_with_one_uuid_fail_verify(monkeypatch):
    def colliding(source_id: str) -> str:
        return to_uuid("a" if source_id == "b" else source_id)

    monkeypatch.setattr("to_weaviate.target.writer.to_uuid", colliding)
    monkeypatch.setattr("to_weaviate.engine.verify.to_uuid", colliding)
    records = [
        Record(i, [1.0, float(n)], f"doc {i}", None, {}) for n, i in enumerate("abc")
    ]
    fakes = inspected(FakeFactories(FakeSource(records)))
    r = invoke(fakes, "copy", "--json")
    assert r.exit_code == 0, r.output
    assert json.loads(r.stdout)["collections"][0]["copied"] == 3
    assert len(fakes.writer.objects) == 2
    r = invoke(fakes, "verify", "--json")
    assert r.exit_code == 1, r.output
    checks = {c["name"]: c for c in json.loads(r.stdout)["checks"]}
    assert checks["count"]["detail"].startswith("source count 3, target count 2;")
    # The id check compares uuids, so it does not see the collision.
    assert (checks["ids"]["passed"], checks["ids"]["detail"]) == (
        True,
        "missing 0, extra 0",
    )
    assert not checks["records"]["passed"]
    assert '"a": ' in checks["records"]["detail"]
    assert "property source_id differs" in checks["records"]["detail"]


class DeletingSource(FakeSource):
    """Deletes record `id-150` once the page that holds it is read."""

    def read(
        self,
        collection: SourceCollection,
        partition: Partition,
        cursor: str | None,
        with_vectors: bool,
    ) -> Page:
        page = super().read(collection, partition, cursor, with_vectors)
        if any(r.id == "id-150" for r in page.records):
            self.records = [r for r in self.records if r.id != "id-150"]
        return page


def test_a_record_deleted_during_the_copy_shifts_another_one_past_the_cursor():
    # The documented limit of copying by offset (finding F4): the record after
    # the gap moves into the page that was read already.
    fakes = inspected(FakeFactories(DeletingSource(cli_records(400))))
    r = invoke(fakes, "copy", "--parallelism", "1", "--json")
    assert r.exit_code == 0, r.output
    assert to_uuid("id-200") not in fakes.writer.objects
    assert len(fakes.writer.objects) == 399
    r = invoke(fakes, "verify", "--json")
    assert r.exit_code == 1, r.output
    checks = {c["name"]: c for c in json.loads(r.stdout)["checks"]}
    assert checks["ids"]["detail"] == (
        'missing 1, extra 1; missing ids ["id-200"]; extra ids ["id-150"]; '
        f"{TARGET_BELOW_SOURCE}; {TARGET_ABOVE_SOURCE}"
    )


def test_copy_warns_when_it_reads_the_source_again_from_the_start():
    source = FakeSource(cli_records(10))
    fakes = inspected(FakeFactories(source))
    r = invoke(fakes, "copy", "--parallelism", "1", "--json")
    assert r.exit_code == 0, r.output
    assert "reread_from" not in json.loads(r.stdout)["collections"][0]
    # id-9, the last record read, is deleted; id-10 and id-11 are added.
    source.records = source.records[:9] + cli_records(12)[10:]
    r = invoke(fakes, "copy", "--json")
    assert r.exit_code == 0, r.output
    assert json.loads(r.stdout)["collections"][0]["reread_from"] == 0
    assert (
        "Warning: docs: the source changed since the last run; re-reading from "
        "the start"
    ) in r.stderr.splitlines()
    assert len(fakes.writer.objects) == 12


def _prop(entry: Any, name: str) -> Any:
    return next(p for p in entry.target.properties if p.name == name)


def _rename_year(plan: Any) -> None:
    _prop(plan.collections[0], "year").name = "publication_year"


def _year_as_text(plan: Any) -> None:
    year = _prop(plan.collections[0], "year")
    year.type, year.tokenization, year.range_filters = "text", "field", False


def _year_as_list(plan: Any) -> None:
    _prop(plan.collections[0], "year").type = "int[]"


def _document_flags(plan: Any) -> None:
    document = _prop(plan.collections[0], "document")
    document.searchable, document.filterable = False, True


def _drop_uri(plan: Any) -> None:
    target = plan.collections[0].target
    target.properties = [p for p in target.properties if p.source_key != "#uri"]
    target.dropped.append(DroppedKey(source_key="#uri", reason="not needed"))


def _hnsw_l2(plan: Any) -> None:
    vector = plan.collections[0].target.vectors[0]
    vector.distance = "l2-squared"
    vector.index = IndexPlan(type="hnsw", options={"ef": 32})


def _edit_notes(plan: Any) -> None:
    plan.collections[0].notes = ["checked by hand"]


def _second_entry_first(plan: Any) -> None:
    first = plan.collections[0]
    target = first.target.model_copy(update={"name": "Docs2"})
    source = first.source.model_copy(update={"ref": "other"})
    plan.collections = [
        first.model_copy(update={"source": source, "target": target}),
        first,
    ]


# (edit of the draft, stored name of the year property, stored value of 2003)
HAND_EDITS = {
    "rename": (_rename_year, "publication_year", 2003),
    "int-to-text": (_year_as_text, "year", "2003"),
    "int-to-list": (_year_as_list, "year", [2003]),
    "document-flags": (_document_flags, "year", 2003),
    "drop-uri": (_drop_uri, "year", 2003),
    "hnsw-l2": (_hnsw_l2, "year", 2003),
    "notes": (_edit_notes, "year", 2003),
    "reordered": (_second_entry_first, "year", 2003),
}


@pytest.mark.parametrize(("edit", "name", "value"), HAND_EDITS.values(), ids=HAND_EDITS)
def test_an_allowed_edit_of_the_plan_is_applied_end_to_end(tmp_path, edit, name, value):
    records = [replace(r, uri=f"https://example.com/{r.id}") for r in cli_records(30)]
    fakes = inspected(FakeFactories(FakeSource(records)))
    path = tmp_path / DIR / "plan.yaml"
    plan = load_plan(path)
    edit(plan)
    save_plan(plan, path)
    found = json.loads((tmp_path / DIR / "inspect.json").read_text())
    found["collections"].append({**found["collections"][0], "ref": "other"})
    (tmp_path / DIR / "inspect.json").write_text(json.dumps(found))
    for command in ("check", "copy", "verify"):
        r = invoke(fakes, command, "--json")
        assert r.exit_code == 0, (command, r.output)
    targets = [e.target.name for e in plan.collections]
    assert sorted(fakes.schema.ensured) == sorted(targets)
    for target in targets:
        stored = fakes.writer.stored[(target, None)][to_uuid("id-3")].properties
        assert stored[name] == value
        dropped = edit is _drop_uri
        assert ("uri" in stored) is not dropped
        assert stored["source_id"] == "id-3"


@pytest.mark.parametrize("cmd", ["check", "copy"])
def test_a_property_name_that_the_client_refuses_is_an_invalid_plan(tmp_path, cmd):
    fakes = inspected(FakeFactories(FakeSource(cli_records(10))))
    path = tmp_path / DIR / "plan.yaml"
    plan = load_plan(path)
    _prop(plan.collections[0], "year").name = "vector"
    save_plan(plan, path)
    r = invoke(fakes, cmd, "--json")
    assert r.exit_code == 2, r.output
    errors = json.loads(r.stdout)["errors"]
    assert [(e["code"], e["message"]) for e in errors] == [
        ("invalid_name", "property vector: Weaviate reserves the property name vector")
    ]
    assert fakes.writer.calls == 0


STATE_HINT_UNREADABLE = "Start again with another --dir, or restore the file."


@pytest.mark.parametrize("cmd", ["status", "copy"])
def test_a_corrupted_state_file_is_an_invalid_state(tmp_path, cmd):
    fakes = inspected(FakeFactories(FakeSource(cli_records(10))))
    (tmp_path / DIR / "state.sqlite").write_bytes(bytes(range(256)) * 16)
    r = invoke(fakes, cmd, "--json")
    assert r.exit_code == 2, r.output
    d = json.loads(r.stdout)
    assert (d["code"], d["message"], d["hint"]) == (
        "invalid_state",
        f"state.sqlite in {DIR} cannot be read.",
        STATE_HINT_UNREADABLE,
    )
    assert fakes.writer.calls == 0


def test_a_state_file_of_a_newer_schema_resumes(tmp_path):
    fakes = inspected(FakeFactories(FakeSource(cli_records(250))))
    old_state_file(
        tmp_path / DIR / "state.sqlite",
        FakeSource.REF,
        250,
        Partition("0-250", "0", "250"),
        "100",
        100,
    )
    db = sqlite3.connect(tmp_path / DIR / "state.sqlite")
    with db:
        db.execute("CREATE TABLE later (x TEXT)")
        db.execute("ALTER TABLE partitions ADD COLUMN later TEXT")
        db.execute("UPDATE partitions SET later = 'kept'")
    db.close()
    # The earlier run created the collection in the target.
    fakes.writer.stored[("Docs", None)] = {}
    r = invoke(fakes, "copy", "--parallelism", "1", "--json")
    assert r.exit_code == 0, r.output
    row = json.loads(r.stdout)["collections"][0]
    assert (row["copied"], row["total"], row["done"]) == (250, 250, True)
    assert len(fakes.writer.objects) == 150
    r = invoke(fakes, "status", "--json")
    assert r.exit_code == 0 and json.loads(r.stdout)["collections"][0]["done"]
    db = sqlite3.connect(tmp_path / DIR / "state.sqlite")
    assert db.execute("SELECT later FROM partitions").fetchall() == [("kept",)]
    db.close()


def _with_lang(records: list[Record]) -> list[Record]:
    return [replace(r, metadata={**r.metadata, "lang": "en"}) for r in records]


def test_copy_adds_a_planned_property_that_the_collection_lacks(tmp_path):
    source = FakeSource(cli_records(10))
    fakes = inspected(FakeFactories(source))
    assert invoke(fakes, "copy").exit_code == 0
    source.records += _with_lang(cli_records(15)[10:])
    r = invoke(fakes, "copy", "--json")
    assert r.exit_code == 1, r.output
    reasons = {json.loads(x)["reason"] for x in (DIR / "rejects.jsonl").open()}
    assert reasons == {
        "metadata key lang is not in plan.yaml; add a property for it or list it "
        "under dropped"
    }
    assert invoke(fakes, "inspect", "chroma", "--force").exit_code == 0
    r = invoke(fakes, "check", "--json")
    assert r.exit_code == 0, r.output
    assert json.loads(r.stdout)["warnings"] == [
        {
            "code": "missing_property",
            "collection": "Docs",
            "message": "property lang: missing in the collection; copy will add it",
        }
    ]
    r = invoke(fakes, "copy", "--json")
    assert fakes.schema.added == [("Docs", "lang")]
    assert "Added the property lang to Docs" in r.stderr
    assert json.loads(r.stdout)["added_properties"] == [
        {"collection": "Docs", "property": "lang"}
    ]
    # The five records were rejected by the run before and are not read again.
    assert r.exit_code == 1 and json.loads(r.stdout)["code"] == "rejects", r.output
    # A copy in a new directory writes them with the new property.
    other = ["--dir", "other"]
    assert invoke(fakes, "inspect", "chroma", *other).exit_code == 0
    r = invoke(fakes, "copy", "--json", *other)
    assert r.exit_code == 0, r.output
    assert json.loads(r.stdout)["added_properties"] == []
    assert fakes.writer.objects[to_uuid("id-12")].properties["lang"] == "en"
    assert len(fakes.writer.objects) == 15


def test_inspect_notes_and_check_refuses_a_source_without_vectors(tmp_path):
    records = [replace(r, vector=None) for r in cli_records(10)]
    fakes = FakeFactories(FakeSource(records))
    r = invoke(fakes, "inspect", "chroma", "--json")
    assert r.exit_code == 0, r.output
    row = json.loads(r.stdout)["collections"][0]
    assert (row["dimension"], row["vectors"][0]["mode"]) == (None, "keep")
    assert NO_VECTORS_NOTE in row["notes"]
    for cmd in ("check", "copy"):
        r = invoke(fakes, cmd, "--json")
        assert r.exit_code == 2, r.output
        errors = json.loads(r.stdout)["errors"]
        assert [(e["code"], e["message"]) for e in errors] == [
            ("no_source_vectors", NO_VECTORS_ERROR)
        ]
    assert fakes.writer.calls == 0
    args = ["--force", "--vectors", "embed", "--vectorizer", "text2vec-weaviate"]
    r = invoke(fakes, "inspect", "chroma", *args, "--json")
    assert r.exit_code == 0, r.output
    assert NO_VECTORS_NOTE not in json.loads(r.stdout)["collections"][0]["notes"]
    for cmd in ("check", "copy"):
        assert invoke(fakes, cmd).exit_code == 0, cmd
    assert len(fakes.writer.objects) == 10


NO_VECTORS_NOTE = (
    "the source has no vectors; run to-weaviate inspect chroma --force "
    "--vectors embed --vectorizer MODULE"
)
NO_VECTORS_ERROR = (
    "docs has records without vectors, but vector default keeps the source "
    "vectors; run to-weaviate inspect chroma --force --vectors embed "
    "--vectorizer MODULE"
)


@pytest.mark.parametrize("cmd", ["check", "copy"])
def test_a_property_name_that_the_server_stores_otherwise_is_an_invalid_plan(
    tmp_path, cmd
):
    fakes = inspected(FakeFactories(FakeSource(cli_records(10))))
    path = tmp_path / DIR / "plan.yaml"
    plan = load_plan(path)
    _prop(plan.collections[0], "year").name = "Year"
    save_plan(plan, path)
    r = invoke(fakes, cmd, "--json")
    assert r.exit_code == 2, r.output
    errors = json.loads(r.stdout)["errors"]
    assert [(e["code"], e["message"].split(":")[0]) for e in errors] == [
        ("invalid_name", "property Year")
    ]
    assert fakes.writer.calls == 0 and fakes.schema.ensured == []


def test_a_state_file_that_another_process_locks_is_in_use(tmp_path, monkeypatch):
    fakes = inspected(FakeFactories(FakeSource(cli_records(10))))
    assert invoke(fakes, "copy").exit_code == 0
    real_connect = sqlite3.connect
    monkeypatch.setattr(
        "to_weaviate.state.sqlite3.connect",
        lambda *a, **k: _Locked(real_connect(*a, **k)),
    )
    r = invoke(fakes, "status", "--json")
    assert r.exit_code == 2, r.output
    d = json.loads(r.stdout)
    assert (d["code"], d["message"], d["hint"]) == (
        "usage",
        f"Another to-weaviate process is using {DIR}.",
        "Wait for it to end, then run the command again.",
    )


class _Locked:
    """A sqlite connection whose statements find the file locked."""

    def __init__(self, db: sqlite3.Connection) -> None:
        self._db = db

    def __enter__(self) -> "_Locked":
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def executescript(self, script: str) -> None:
        raise sqlite3.OperationalError("database is locked")

    def close(self) -> None:
        self._db.close()


def test_a_refused_property_addition_names_the_plan_in_its_hint():
    refused = PropertyRejected("conflict for property lang", "add the property lang")
    source = FakeSource(cli_records(10))
    fakes = inspected(FakeFactories(source, schema=FakeSchema(ensure_error=refused)))
    r = invoke(fakes, "copy", "--json")
    assert r.exit_code == 2, r.output
    d = json.loads(r.stdout)
    assert (d["code"], d["message"], d["hint"]) == (
        "target_rejected_schema",
        "Weaviate refused to add the property lang: conflict for property lang",
        "Weaviate refused to add the property; read its message and edit plan.yaml.",
    )
