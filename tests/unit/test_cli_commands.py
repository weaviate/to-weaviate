import json
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
from fakes import (
    FakeFactories,
    FakeInsertClient,
    FakeSchema,
    FakeSource,
    FakeWriter,
    cli_records,
    invoke,
    old_state_file,
)
from typer.testing import Result
from weaviate.exceptions import WeaviateInsertManyAllFailedError

from to_weaviate.ids import to_uuid
from to_weaviate.lock import Locked, exclusive_lock
from to_weaviate.plan import load_plan, save_plan
from to_weaviate.sources.base import Page, Partition, Record, SourceCollection
from to_weaviate.sources.chroma.client import ChromaError
from to_weaviate.sources.chroma.source import ChromaSource
from to_weaviate.state import State
from to_weaviate.target.schema import SchemaRejected
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


def test_inspect_unknown_collection_hint_names_both_database_settings():
    fakes = FakeFactories(FakeSource(cli_records(10)))
    r = invoke(fakes, "inspect", "chroma", "--collection", "nope", "--json")
    assert r.exit_code == 2
    hint = json.loads(r.stdout)["hint"]
    assert "--database" in hint and "CHROMA_DATABASE" in hint


DB = FakeSource.NAMESPACE


# `asked` is the namespaces argument the source receives; None asks for all.
# `recorded` is the database of each plan entry.
@pytest.mark.parametrize(
    "variable, args, asked, recorded",
    [
        pytest.param(DB, [], [DB], [DB], id="variable"),
        pytest.param(DB, ["--database", "other"], ["other"], [], id="option-wins"),
        pytest.param(None, [], None, [DB], id="neither"),
    ],
)
def test_inspect_databases(tmp_path, variable, args, asked, recorded):
    source = FakeSource(cli_records(10))
    r = invoke(
        FakeFactories(source), "inspect", "chroma", *args, CHROMA_DATABASE=variable
    )
    assert r.exit_code == 0, r.output
    assert source.inspected_namespaces == [asked]
    plan = load_plan(tmp_path / DIR / "plan.yaml")
    assert [e.source.namespace for e in plan.collections] == recorded


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


def all_failed(reason: str = REFUSED) -> WeaviateInsertManyAllFailedError:
    """What client 4.24.1 raises when Weaviate refuses every object of a batch."""
    return WeaviateInsertManyAllFailedError(f"Here is the set of all errors: {reason}")


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
    fakes, client = refusing(10, lambda n: all_failed() if n == 1 else None)
    r = invoke(fakes, "copy", "--json")
    assert r.exit_code == 0, r.output
    assert len(client.writes_per_uuid) == 10
    assert json.loads(r.stdout)["collections"][0]["copied"] == 10


def test_copy_after_a_refused_batch_copies_the_page():
    fakes, client = refusing(10, lambda n: all_failed())
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
    fakes, client = refusing(250, lambda n: all_failed() if n > 1 else None)
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
    ("http://localhost:8099", "HTTP://LocalHost:8099", None, "localhost:8099"),
    ("http://localhost:8099", "http://localhost:8099/", None, "localhost:8099"),
    ("http://localhost:8099", "http://localhost.:8099", None, "localhost:8099"),
    ("http://weaviate.test", "http://weaviate.test:80", None, "weaviate.test:80"),
    ("https://c.weaviate.cloud", "c.weaviate.cloud", None, "c.weaviate.cloud:443"),
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
