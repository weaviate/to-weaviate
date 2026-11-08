import json
import math
import random
import struct
from dataclasses import replace
from pathlib import Path

import pytest
from fakes import FakeFactories, FakeReader, FakeSource, FakeWriter, cli_records, invoke

from to_weaviate import plan as p
from to_weaviate.engine.verify import (
    Check,
    IdDiff,
    check_line,
    diff_ids,
    missing_target,
    verify_counts,
    verify_ids,
    verify_records,
    write_report,
)
from to_weaviate.plan import load_plan, save_plan
from to_weaviate.sources.base import Record, SourceCollection
from to_weaviate.sources.chroma.client import ChromaError
from to_weaviate.target.writer import TargetObject, to_object

DIR = Path(".to-weaviate")


def make_records(n: int, prefix: str = "id") -> list[Record]:
    return [
        Record(
            f"{prefix}-{i}",
            [float(i), 1.0],
            f"doc {i}",
            None,
            {"year": 2000 + i, "score": i / 10},
        )
        for i in range(n)
    ]


def make_entry(mode: str = "keep", tenant: str | None = None) -> p.CollectionPlan:
    vector = (
        p.VectorPlan(
            name="default",
            mode="embed",
            vectorizer=p.VectorizerPlan(module="text2vec-weaviate"),
        )
        if mode == "embed"
        else p.VectorPlan(name="default", mode="keep", distance="cosine")
    )
    return p.CollectionPlan(
        source=p.SourceRef(
            ref=FakeSource.REF,
            namespace=FakeSource.NAMESPACE,
            name=FakeSource.COLLECTION,
            count=0,
            dimension=2,
            distance="cosine",
            embedding=None,
        ),
        target=p.TargetPlan(
            name="Docs",
            tenant=tenant,
            vectors=[vector],
            properties=[
                p.PropertyPlan(source_key="#document", name="document", type="text"),
                p.PropertyPlan(source_key="#id", name="source_id", type="text"),
                p.PropertyPlan(source_key="#uri", name="uri", type="text"),
                p.PropertyPlan(source_key="year", name="year", type="int"),
                p.PropertyPlan(source_key="score", name="score", type="number"),
            ],
        ),
    )


ENTRY = make_entry()


def converted(records: list[Record], entry: p.CollectionPlan) -> list[TargetObject]:
    objects = [to_object(r, entry) for r in records]
    assert all(isinstance(o, TargetObject) for o in objects)
    return objects  # type: ignore[return-value]


def migrated(
    records: list[Record], entry: p.CollectionPlan, skip: set[str] | None = None
) -> tuple[FakeSource, FakeWriter]:
    """A source over `records` and a writer that holds them, except `skip`."""
    writer = FakeWriter()
    writer.write(
        entry, converted([r for r in records if r.id not in (skip or set())], entry)
    )
    return FakeSource(records), writer


def collection_of(source: FakeSource) -> SourceCollection:
    return source.inspect(None, None, None)[0]


def records_check(
    records: list[Record],
    entry: p.CollectionPlan,
    stored: list[TargetObject],
    sample: int | None = None,
) -> Check:
    source = FakeSource(records)
    writer = FakeWriter()
    writer.write(entry, stored)
    return verify_records(
        source,
        FakeReader(writer),
        entry,
        collection_of(source),
        len(records) if sample is None else sample,
        random.Random(0),
    )


def shifted(obj: TargetObject, delta: float) -> TargetObject:
    vector = obj.vectors["default"]
    return replace(obj, vectors={"default": [vector[0] + delta, *vector[1:]]})


def test_counts_match():
    source, writer = migrated(make_records(10), ENTRY)
    check = verify_counts(source, FakeReader(writer), ENTRY, collection_of(source))
    assert (check.name, check.collection, check.passed) == ("count", "Docs", True)
    assert "source count 10" in check.detail and "target count 10" in check.detail


def test_count_mismatch_names_copy():
    source, writer = migrated(make_records(10), ENTRY, skip={"id-3"})
    check = verify_counts(source, FakeReader(writer), ENTRY, collection_of(source))
    assert (check.name, check.collection, check.passed) == ("count", "Docs", False)
    assert "source count 10" in check.detail and "target count 9" in check.detail
    assert "to-weaviate copy" in check.detail and "sync" not in check.detail


def test_target_below_source_says_to_copy_again():
    source, writer = migrated(make_records(10), ENTRY, skip={"id-3"})
    check = verify_counts(source, FakeReader(writer), ENTRY, collection_of(source))
    assert check.detail == (
        "source count 10, target count 9; run to-weaviate copy again. If the "
        "counts still differ, records were skipped or rejected: read "
        "rejects.jsonl. Rejects are final for this directory. After fixing the "
        "cause, copy plan.yaml and inspect.json to a new --dir and run "
        "to-weaviate copy there; objects already in Weaviate are written again "
        "unchanged."
    )


def test_target_above_source_says_no_command_removes_the_extra_objects():
    source, writer = migrated(make_records(10), ENTRY)
    writer.write(ENTRY, converted(make_records(2, prefix="ghost"), ENTRY))
    check = verify_counts(source, FakeReader(writer), ENTRY, collection_of(source))
    assert not check.passed
    assert check.detail == (
        "source count 10, target count 12; the target holds objects whose "
        "source records were deleted. This version has no command that removes "
        "them."
    )


def test_id_diff_finds_one_missing_among_many(tmp_path):
    source, writer = migrated(make_records(10_000), ENTRY, skip={"id-4321"})
    diff = diff_ids(source, FakeReader(writer), ENTRY, collection_of(source), tmp_path)
    assert (diff.missing, diff.extra) == (1, 0)
    assert diff.missing_sample == ["id-4321"] and diff.extra_sample == []
    assert list(tmp_path.iterdir()) == []


def test_id_diff_reports_extra_ids(tmp_path):
    source, writer = migrated(make_records(10), ENTRY)
    writer.write(ENTRY, converted(make_records(2, prefix="ghost"), ENTRY))
    diff = diff_ids(source, FakeReader(writer), ENTRY, collection_of(source), tmp_path)
    assert (diff.missing, diff.extra) == (0, 2)
    assert sorted(diff.extra_sample) == ["ghost-0", "ghost-1"]
    assert diff.missing_sample == []


def test_id_samples_hold_at_most_20_ids(tmp_path):
    absent = {f"id-{i}" for i in range(25)}
    source, writer = migrated(make_records(30), ENTRY, skip=absent)
    writer.write(ENTRY, converted(make_records(25, prefix="ghost"), ENTRY))
    diff = diff_ids(source, FakeReader(writer), ENTRY, collection_of(source), tmp_path)
    assert (diff.missing, diff.extra) == (25, 25)
    assert len(diff.missing_sample) == len(diff.extra_sample) == 20
    assert set(diff.missing_sample) <= absent
    assert all(i.startswith("ghost-") for i in diff.extra_sample)


def test_id_diff_is_scoped_to_the_tenant(tmp_path):
    first, second = make_entry(tenant="t1"), make_entry(tenant="t2")
    source, writer = migrated(make_records(10), first)
    writer.write(second, converted(make_records(5, prefix="other"), second))
    diff = diff_ids(source, FakeReader(writer), first, collection_of(source), tmp_path)
    assert (diff.missing, diff.extra) == (0, 0)


class LostReader(FakeReader):
    def source_ids(self, entry):
        yield "00000000-0000-0000-0000-000000000001", "a"
        raise RuntimeError("connection lost")


def test_id_diff_removes_its_file_when_it_raises(tmp_path):
    source, writer = migrated(make_records(10), ENTRY)
    with pytest.raises(RuntimeError, match="connection lost"):
        diff_ids(source, LostReader(writer), ENTRY, collection_of(source), tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_verify_ids_passes_only_without_differences():
    check = verify_ids(IdDiff(0, 0, [], []), ENTRY)
    assert (check.name, check.collection, check.passed) == ("ids", "Docs", True)
    check = verify_ids(IdDiff(1, 2, ["id-1"], ["ghost-1", "ghost-2"]), ENTRY)
    assert (check.name, check.collection, check.passed) == ("ids", "Docs", False)
    for text in ("missing 1", "extra 2", '"id-1"', '"ghost-1"', '"ghost-2"'):
        assert text in check.detail


COPY_AGAIN = (
    "run to-weaviate copy again. If ids are still missing, records were "
    "skipped or rejected: read rejects.jsonl. Rejects are final for this "
    "directory. After fixing the cause, copy plan.yaml and inspect.json to a "
    "new --dir and run to-weaviate copy there; objects already in Weaviate are "
    "written again unchanged."
)
NO_REMOVE = (
    "the target holds objects whose source records were deleted. This version "
    "has no command that removes them."
)


@pytest.mark.parametrize(
    ("diff", "detail"),
    [
        (IdDiff(0, 0, [], []), "missing 0, extra 0"),
        (
            IdDiff(1, 0, ["a"], []),
            f'missing 1, extra 0; missing ids ["a"]; {COPY_AGAIN}',
        ),
        (IdDiff(0, 1, [], ["b"]), f'missing 0, extra 1; extra ids ["b"]; {NO_REMOVE}'),
        (
            IdDiff(1, 1, ["a"], ["b"]),
            f'missing 1, extra 1; missing ids ["a"]; extra ids ["b"]; {COPY_AGAIN}; '
            f"{NO_REMOVE}",
        ),
    ],
    ids=["none", "missing", "extra", "both"],
)
def test_id_diff_detail_names_the_fix(diff, detail):
    assert verify_ids(diff, ENTRY).detail == detail


def test_records_match():
    records = make_records(10)
    check = records_check(records, ENTRY, converted(records, ENTRY))
    assert (check.name, check.collection, check.passed) == ("records", "Docs", True)
    assert check.detail.startswith("sampled 10, compared 10,")


def test_missing_record_is_reported():
    records = make_records(10)
    stored = converted([r for r in records if r.id != "id-3"], ENTRY)
    check = records_check(records, ENTRY, stored)
    assert (check.name, check.collection, check.passed) == ("records", "Docs", False)
    assert '"id-3"' in check.detail


def test_changed_property_is_reported():
    records = make_records(10)
    stored = converted(records, ENTRY)
    stored[4] = replace(stored[4], properties={**stored[4].properties, "year": 1234})
    check = records_check(records, ENTRY, stored)
    assert not check.passed
    assert "year" in check.detail and '"id-4"' in check.detail
    assert "1234" not in check.detail and "2004" not in check.detail


def test_float_property_is_compared_with_isclose():
    records = make_records(10)
    stored = converted(records, ENTRY)
    score = stored[5].properties["score"]
    stored[5] = replace(
        stored[5], properties={**stored[5].properties, "score": score * (1 + 1e-12)}
    )
    assert stored[5].properties["score"] != score
    check = records_check(records, ENTRY, stored)
    assert check.passed, check.detail


# (source int, int that the target returns, same). Weaviate 1.40.4 returns an
# int above 2**53 rounded to a double. copy rejects an int whose double is
# 2**63, so verify never compares one.
STORED_INTS = [
    (2**53 + 1, 2**53, True),
    (2**63 - 1024, 2**63 - 1024, True),
    (-(2**63), -(2**63), True),
    ([2**53 + 1, 3], [2**53, 3], True),
    (2**53 + 1, 2**53 + 4, False),
    (2003, 2004, False),
]


@pytest.mark.parametrize(("source", "stored", "same"), STORED_INTS)
def test_an_int_is_compared_with_the_double_that_the_target_stores(
    source, stored, same
):
    records = [replace(r, metadata={"year": source}) for r in make_records(3)]
    entry = make_entry()
    if isinstance(source, list):
        year = next(x for x in entry.target.properties if x.name == "year")
        entry.target.properties[entry.target.properties.index(year)] = year.model_copy(
            update={"type": "int[]"}
        )
    objects = converted(records, entry)
    objects[1] = replace(
        objects[1], properties={**objects[1].properties, "year": stored}
    )
    check = records_check(records, entry, objects)
    assert check.passed is same, check.detail


def test_vector_within_tolerance_passes():
    records = make_records(10)
    stored = converted(records, ENTRY)
    stored[2] = shifted(stored[2], 1e-7)
    check = records_check(records, ENTRY, stored)
    assert check.passed, check.detail


def test_vector_outside_tolerance_fails():
    records = make_records(10)
    stored = converted(records, ENTRY)
    stored[2] = shifted(stored[2], 1e-3)
    check = records_check(records, ENTRY, stored)
    assert not check.passed
    assert '"id-2"' in check.detail and "vector default" in check.detail


def test_vector_of_another_length_fails():
    records = make_records(10)
    stored = converted(records, ENTRY)
    stored[6] = replace(stored[6], vectors={"default": [6.0, 1.0, 0.0]})
    check = records_check(records, ENTRY, stored)
    assert not check.passed
    assert '"id-6"' in check.detail and "vector default" in check.detail


def float32(value: float) -> float:
    return float(struct.unpack("<f", struct.pack("<f", value))[0])


def test_source_vector_is_compared_as_float32():
    records = [replace(r, vector=[300.123, 1.0]) for r in make_records(10)]
    assert abs(300.123 - float32(300.123)) > 1e-5
    stored = [
        replace(o, vectors={"default": [float32(300.123), 1.0]})
        for o in converted(records, ENTRY)
    ]
    check = records_check(records, ENTRY, stored)
    assert check.passed, check.detail


def test_source_vector_outside_the_float32_range_fails():
    records = make_records(10)
    stored = converted(records, ENTRY)
    records[4] = replace(records[4], vector=[1e39, 1.0])
    check = records_check(records, ENTRY, stored)
    assert not check.passed
    assert '"id-4": vector default' in check.detail
    assert "float32 range" in check.detail


def test_nan_in_the_source_vector_fails():
    records = make_records(10)
    stored = converted(records, ENTRY)
    records[3] = replace(records[3], vector=[math.nan, 1.0])
    check = records_check(records, ENTRY, stored)
    assert not check.passed and '"id-3": vector default' in check.detail


def test_nan_in_the_stored_vector_fails():
    records = make_records(10)
    stored = converted(records, ENTRY)
    stored[3] = replace(stored[3], vectors={"default": [math.nan, math.nan]})
    check = records_check(records, ENTRY, stored)
    assert not check.passed and '"id-3": vector default' in check.detail


def test_embed_mode_skips_vector_equality():
    entry = make_entry("embed")
    records = make_records(10)
    stored = [
        replace(o, vectors={"default": [9.0, 9.0]}) for o in converted(records, entry)
    ]
    check = records_check(records, entry, stored)
    assert check.passed, check.detail


def test_rejects_are_counted_and_not_compared():
    records = make_records(10)
    odd = Record("odd-1", [1.0, 1.0], "doc", None, {"mystery": 1})
    check = records_check([*records, odd], ENTRY, converted(records, ENTRY))
    assert check.passed, check.detail
    assert "sampled 11, compared 10, rejects skipped 1" in check.detail


class EmptySampleSource(FakeSource):
    """FakeSource whose sample holds no record, as after deletes in the source."""

    def sample(self, collection, size, rng, with_vectors):
        return []


def test_records_fail_when_no_record_was_compared_and_the_source_has_records():
    records = make_records(10)
    source, writer = EmptySampleSource(records), FakeWriter()
    writer.write(ENTRY, converted(records, ENTRY))
    check = verify_records(
        source, FakeReader(writer), ENTRY, collection_of(source), 300, random.Random(0)
    )
    assert (check.name, check.passed) == ("records", False)
    assert check.detail == (
        "sampled 0, compared 0, rejects skipped 0, differing 0; no record was "
        "compared; run to-weaviate verify again"
    )


def test_records_fail_when_every_sampled_record_is_a_reject():
    odd = Record("odd-1", [1.0, 1.0], "doc", None, {"mystery": 1})
    check = records_check([odd, replace(odd, id="odd-2")], ENTRY, [])
    assert not check.passed
    assert check.detail == (
        "sampled 2, compared 0, rejects skipped 2, differing 0; every sampled "
        "record is rejected at copy time; read rejects.jsonl"
    )


def test_records_pass_on_an_empty_source():
    check = records_check([], ENTRY, [], sample=300)
    assert (check.name, check.passed, check.detail) == (
        "records",
        True,
        "source is empty",
    )


def test_records_detail_lists_at_most_10_failures():
    check = records_check(make_records(15), ENTRY, [])
    assert not check.passed and "differing 15" in check.detail
    assert sum(f'"id-{i}"' in check.detail for i in range(15)) == 10


def test_report_lists_every_check(tmp_path):
    checks = [
        Check("count", "Docs", True, "source count 3, target count 3"),
        Check("ids", "Docs", False, "missing 1, extra 0"),
        Check("records", "Docs", True, "sampled 3"),
    ]
    write_report(checks, tmp_path)
    lines = (tmp_path / "report.md").read_text().splitlines()
    assert lines[0].startswith("# ")
    assert [line for line in lines if line.startswith(("PASS", "FAIL"))] == [
        "PASS Docs count: source count 3, target count 3",
        "FAIL Docs ids: missing 1, extra 0",
        "PASS Docs records: sampled 3",
    ]
    assert json.loads((tmp_path / "report.json").read_text()) == {
        "passed": False,
        "checks": [
            {
                "name": "count",
                "collection": "Docs",
                "passed": True,
                "detail": "source count 3, target count 3",
                "tenant": None,
            },
            {
                "name": "ids",
                "collection": "Docs",
                "passed": False,
                "detail": "missing 1, extra 0",
                "tenant": None,
            },
            {
                "name": "records",
                "collection": "Docs",
                "passed": True,
                "detail": "sampled 3",
                "tenant": None,
            },
        ],
    }


def test_check_line_names_the_tenant():
    check = Check("count", "Docs", True, "source count 3, target count 3", "acme")
    assert check_line(check) == (
        "PASS Docs (tenant acme) count: source count 3, target count 3"
    )
    assert check_line(replace(check, tenant=None)) == (
        "PASS Docs count: source count 3, target count 3"
    )


def test_checks_carry_the_tenant_of_the_entry(tmp_path):
    for tenant in ("t1", None):
        entry = make_entry(tenant=tenant)
        source, writer = migrated(make_records(10), entry)
        reader, collection = FakeReader(writer), collection_of(source)
        checks = [
            verify_counts(source, reader, entry, collection),
            verify_ids(diff_ids(source, reader, entry, collection, tmp_path), entry),
            verify_records(source, reader, entry, collection, 5, random.Random(0)),
        ]
        assert [(c.passed, c.tenant) for c in checks] == [(True, tenant)] * 3


def test_missing_collection_is_one_failed_count_check():
    check = missing_target(FakeReader(FakeWriter()), make_entry(tenant="t1"))
    assert check == Check(
        "count",
        "Docs",
        False,
        "collection Docs does not exist; run to-weaviate copy",
        "t1",
    )


def test_missing_tenant_is_one_failed_count_check():
    writer = FakeWriter()
    writer.write(make_entry(tenant="t2"), [])
    check = missing_target(FakeReader(writer), make_entry(tenant="t1"))
    assert check == Check(
        "count",
        "Docs",
        False,
        "tenant t1 of collection Docs does not exist; run to-weaviate copy",
        "t1",
    )


def test_existing_target_gives_no_check():
    writer = FakeWriter()
    writer.write(make_entry(tenant="t1"), [])
    assert missing_target(FakeReader(writer), make_entry(tenant="t1")) is None
    assert missing_target(FakeReader(writer), make_entry()) is None


@pytest.fixture
def in_tmp(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


def copied(writer: FakeWriter | None = None) -> FakeFactories:
    fakes = FakeFactories(FakeSource(cli_records(10)), writer=writer)
    assert invoke(fakes, "inspect", "chroma").exit_code == 0
    assert invoke(fakes, "copy").exit_code in (0, 1)
    return fakes


def test_verify_exits_1_on_failure(in_tmp):
    fakes = copied(FakeWriter(reject_ids={"id-3"}))
    r = invoke(fakes, "verify", "--json")
    assert r.exit_code == 1, r.output
    d = json.loads(r.stdout)
    assert (d["status"], d["code"]) == ("error", "verification_failed")
    assert {
        "name": "count",
        "collection": "Docs",
        "passed": False,
        "tenant": None,
    } == {k: v for k, v in d["checks"][0].items() if k != "detail"}
    assert "3 checks" in d["message"] and "report.md" in d["hint"]
    assert d["report"] == str(DIR / "report.md")
    assert json.loads((DIR / "report.json").read_text())["passed"] is False


def test_verify_passes_after_a_full_copy(in_tmp):
    r = invoke(copied(), "verify", "--json")
    assert r.exit_code == 0, r.output
    d = json.loads(r.stdout)
    assert d["status"] == "success" and d["report"] == str(DIR / "report.md")
    assert [(c["name"], c["passed"]) for c in d["checks"]] == [
        ("count", True),
        ("ids", True),
        ("records", True),
    ]
    assert (DIR / "report.md").exists()
    assert list(DIR.glob("verify-*")) == []


def test_quick_skips_the_id_comparison(in_tmp):
    r = invoke(copied(), "verify", "--quick", "--json")
    assert r.exit_code == 0, r.output
    assert [c["name"] for c in json.loads(r.stdout)["checks"]] == ["count", "records"]


def test_sample_sets_the_number_of_compared_records(in_tmp):
    r = invoke(copied(), "verify", "--sample", "3", "--json")
    assert r.exit_code == 0, r.output
    records = json.loads(r.stdout)["checks"][-1]
    assert records["name"] == "records" and records["detail"].startswith("sampled 3,")


def test_verify_text_prints_one_line_per_check(in_tmp):
    r = invoke(copied(FakeWriter(reject_ids={"id-3"})), "verify")
    assert r.exit_code == 1
    marked = [
        line for line in r.stdout.splitlines() if line.startswith(("PASS", "FAIL"))
    ]
    assert [line.split(":")[0] for line in marked] == [
        "FAIL Docs count",
        "FAIL Docs ids",
        "FAIL Docs records",
    ]
    assert "Error: 3 checks failed." in r.stderr and "report.md" in r.stderr


def test_verify_needs_inspect_json(in_tmp):
    fakes = copied()
    (DIR / "inspect.json").unlink()
    r = invoke(fakes, "verify", "--json")
    assert r.exit_code == 2
    d = json.loads(r.stdout)
    assert d["code"] == "usage" and "to-weaviate inspect chroma" in d["hint"]


def test_verify_names_the_tenant(in_tmp):
    fakes = FakeFactories(FakeSource(cli_records(10)))
    assert invoke(fakes, "inspect", "chroma").exit_code == 0
    plan = load_plan(DIR / "plan.yaml")
    plan.collections[0].target.tenant = "acme"
    save_plan(plan, DIR / "plan.yaml")
    assert invoke(fakes, "copy").exit_code == 0
    r = invoke(fakes, "verify", "--json")
    assert r.exit_code == 0, r.output
    assert [c["tenant"] for c in json.loads(r.stdout)["checks"]] == ["acme"] * 3
    assert "Verifying docs in Docs (tenant acme)" in r.stderr
    report = json.loads((DIR / "report.json").read_text())
    assert [c["tenant"] for c in report["checks"]] == ["acme"] * 3
    assert "PASS Docs (tenant acme) count: " in (DIR / "report.md").read_text()


def test_verify_before_copy_fails_one_count_check(in_tmp):
    fakes = FakeFactories(FakeSource(cli_records(10)))
    assert invoke(fakes, "inspect", "chroma").exit_code == 0
    r = invoke(fakes, "verify", "--json")
    assert r.exit_code == 1, r.output
    d = json.loads(r.stdout)
    assert (d["code"], d["message"]) == ("verification_failed", "1 check failed.")
    assert d["checks"] == [
        {
            "name": "count",
            "collection": "Docs",
            "passed": False,
            "detail": "collection Docs does not exist; run to-weaviate copy",
            "tenant": None,
        }
    ]


@pytest.mark.parametrize(("sample", "exit_code"), [("10000", 0), ("10001", 2)])
def test_sample_accepts_at_most_10000(in_tmp, sample, exit_code):
    r = invoke(copied(), "verify", "--sample", sample, "--json")
    assert r.exit_code == exit_code, r.output
    if exit_code == 2:
        assert r.stdout == "" and "Invalid value" in r.stderr


class FailingIdsSource(FakeSource):
    """FakeSource whose ids() raises `ids_error`, when it is set, after five ids."""

    ids_error: Exception | None = None

    def ids(self, collection):
        for position, record_id in enumerate(super().ids(collection)):
            if self.ids_error is not None and position == 5:
                raise self.ids_error
            yield record_id


def test_failed_verify_leaves_no_old_report(in_tmp):
    source = FailingIdsSource(cli_records(10))
    fakes = FakeFactories(source)
    assert invoke(fakes, "inspect", "chroma").exit_code == 0
    assert invoke(fakes, "copy").exit_code == 0
    assert invoke(fakes, "verify").exit_code == 0
    assert (DIR / "report.md").exists() and (DIR / "report.json").exists()
    source.ids_error = ChromaError(None, "ConnectError: connection refused")
    r = invoke(fakes, "verify", "--json")
    assert r.exit_code == 4, r.output
    assert json.loads(r.stdout)["code"] == "source_unreachable"
    assert not (DIR / "report.md").exists()
    assert not (DIR / "report.json").exists()
    assert list(DIR.glob("verify-*")) == []


def test_verify_waits_before_the_counts_and_reads_the_state_after_the_records(in_tmp):
    fakes = copied()
    fakes.source.calls.clear()
    fakes.source.waits = [(5, 10)]
    fakes.source.compaction = [{"version": 2, "log_position": 10}]
    r = invoke(fakes, "verify", "--json")
    assert r.exit_code == 0, r.output
    calls = fakes.source.calls
    assert calls[0] == "wait_compacted" and calls[-1] == "compaction_state"
    assert {"count", "ids", "sample"} <= set(calls)
    assert (
        "Waiting for Chroma to compact docs: 5 of 10 operations not compacted"
        in r.stderr
    )


def test_verify_stops_with_source_compacting_and_writes_no_report(in_tmp):
    fakes = copied()
    fakes.source.compaction = [
        {"version": 1, "log_position": 10},
        {"version": 2, "log_position": 10},
    ]
    r = invoke(fakes, "verify", "--json")
    assert r.exit_code == 4, r.output
    d = json.loads(r.stdout)
    assert (d["code"], d["collection"]) == ("source_compacting", "docs")
    assert not (DIR / "report.md").exists() and not (DIR / "report.json").exists()
