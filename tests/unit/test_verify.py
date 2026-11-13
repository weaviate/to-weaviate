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
    ID_BATCH,
    Check,
    IdDiff,
    _same,
    check_line,
    diff_ids,
    id_tables,
    missing_source_ids,
    missing_target,
    owned_total,
    source_listed,
    stale_batches,
    stale_count,
    verify_counts,
    verify_ids,
    verify_records,
    write_report,
)
from to_weaviate.ids import to_uuid
from to_weaviate.plan import load_plan, save_plan
from to_weaviate.sources.base import Record, Sample, SourceCollection
from to_weaviate.sources.chroma.client import ChromaError
from to_weaviate.sources.chroma.source import ChromaSource
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
    assert "to-weaviate copy again" in check.detail
    assert "sync --prune" not in check.detail


def test_target_below_source_says_to_copy_again():
    source, writer = migrated(make_records(10), ENTRY, skip={"id-3"})
    check = verify_counts(source, FakeReader(writer), ENTRY, collection_of(source))
    assert check.detail == (
        "source count 10, target count 9; run to-weaviate copy again. If the "
        "counts still differ, records were skipped or rejected: run to-weaviate "
        "sync, and read rejects.jsonl. A reject stays until its cause is fixed; "
        "then to-weaviate sync writes the record. Objects already in Weaviate "
        "are written only when they changed."
    )


def test_target_above_source_points_to_the_ids_check():
    source, writer = migrated(make_records(10), ENTRY)
    writer.write(ENTRY, converted(make_records(2, prefix="ghost"), ENTRY))
    check = verify_counts(source, FakeReader(writer), ENTRY, collection_of(source))
    assert not check.passed
    assert check.detail == (
        "source count 10, target count 12; the target holds objects that are "
        "not in the source; the ids check of a verify without --quick says which "
        "and how to remove them."
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


FOREIGN = TargetObject("00000000-0000-0000-0000-00000000f00d", {"year": 1}, {})


def test_id_tables_lists_stale_objects_with_a_source_id_only(tmp_path):
    source, writer = migrated(make_records(10), ENTRY)
    writer.write(ENTRY, [FOREIGN])
    del source.records[8:]
    reader = FakeReader(writer)
    with id_tables(source, reader, ENTRY, collection_of(source), tmp_path) as db:
        assert (source_listed(db), owned_total(db), stale_count(db)) == (8, 10, 2)
        stale = [pair for batch in stale_batches(db) for pair in batch]
    assert stale == sorted([(to_uuid("id-8"), "id-8"), (to_uuid("id-9"), "id-9")])
    assert list(tmp_path.iterdir()) == []


def test_id_tables_batches_the_stale_and_the_missing_ids(tmp_path):
    source, writer = migrated(make_records(2500), ENTRY)
    absent = source.records[:1200]
    del source.records[:1200]
    source.records.extend(make_records(1100, prefix="new"))
    reader = FakeReader(writer)
    with id_tables(source, reader, ENTRY, collection_of(source), tmp_path) as db:
        stale = list(stale_batches(db))
        missing = list(missing_source_ids(db))
    assert [len(b) for b in stale] == [ID_BATCH, 200]
    assert {i for _, i in sum(stale, [])} == {r.id for r in absent}
    assert [len(b) for b in missing] == [ID_BATCH, 100]
    assert set(sum(missing, [])) == {f"new-{i}" for i in range(1100)}


def test_id_tables_names_its_scratch_directory(tmp_path):
    source, writer = migrated(make_records(10), ENTRY)
    reader = FakeReader(writer)
    with id_tables(
        source, reader, ENTRY, collection_of(source), tmp_path, prefix="sync-"
    ):
        assert [d.name[:5] for d in tmp_path.iterdir()] == ["sync-"]


def test_id_diff_counts_an_object_without_source_id_as_extra_and_names_its_uuid(
    tmp_path,
):
    source, writer = migrated(make_records(10), ENTRY)
    writer.write(ENTRY, [FOREIGN])
    assert list(FakeReader(writer).source_ids(ENTRY))[-1] == (FOREIGN.uuid, None)
    diff = diff_ids(source, FakeReader(writer), ENTRY, collection_of(source), tmp_path)
    assert (diff.missing, diff.extra, diff.foreign) == (0, 1, 1)
    assert diff.extra_sample == [FOREIGN.uuid]


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
    "skipped or rejected: run to-weaviate sync, and read rejects.jsonl. A "
    "reject stays until its cause is fixed; then to-weaviate sync writes the "
    "record. Objects already in Weaviate are written only when they changed."
)
PRUNE_STALE = (
    "the target holds objects whose source records were deleted; stop the "
    "writes to the source and run to-weaviate sync --prune to delete them."
)
FOREIGN_TEXT = (
    "the target holds objects without a source_id, which this tool did not "
    "write; sync --prune leaves them, so delete them in Weaviate if they do not "
    "belong to the collection."
)


@pytest.mark.parametrize(
    ("diff", "detail"),
    [
        (IdDiff(0, 0, [], []), "missing 0, extra 0"),
        (
            IdDiff(1, 0, ["a"], []),
            f'missing 1, extra 0; missing ids ["a"]; {COPY_AGAIN}',
        ),
        (
            IdDiff(0, 1, [], ["b"]),
            f'missing 0, extra 1; extra ids ["b"]; {PRUNE_STALE}',
        ),
        (
            IdDiff(1, 1, ["a"], ["b"]),
            f'missing 1, extra 1; missing ids ["a"]; extra ids ["b"]; {COPY_AGAIN}; '
            f"{PRUNE_STALE}",
        ),
        (
            IdDiff(0, 1, [], ["u"], foreign=1),
            f'missing 0, extra 1; extra ids ["u"]; {FOREIGN_TEXT}',
        ),
        (
            IdDiff(0, 3, [], ["b", "u"], foreign=1),
            f'missing 0, extra 3; extra ids ["b", "u"]; {PRUNE_STALE}; {FOREIGN_TEXT}',
        ),
    ],
    ids=["none", "missing", "extra", "both", "foreign", "stale-and-foreign"],
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


# (source int, value that the target returns, same). The target keeps an int as
# a double and can return it as a float.
STORED_INTS = [
    (2**53, 2**53, True),
    (-(2**53), -(2**53), True),
    (5, 5.0, True),
    (2**53, 2.0**53, True),
    ([2**53, 3], [2.0**53, 3], True),
    (2**53 - 1, 2**53, False),
    (2**53 - 1, 2.0**53, False),
    (-(2**53) + 1, -(2**53), False),
    ([2**53 - 1, 3], [2**53, 3], False),
    (5, 5.5, False),
    (1, True, False),
    (2003, 2004, False),
]


@pytest.mark.parametrize(("source", "stored", "same"), STORED_INTS)
def test_an_int_is_compared_exactly(source, stored, same):
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


@pytest.mark.parametrize(
    ("expected", "found", "same"),
    [
        (2**60 + 1, 2**60, False),
        (2**60 + 1, 2.0**60, False),
        (2**63 - 1, 2.0**63, False),
        (2**60, 2.0**60, True),
        (True, 1, False),
        (True, True, True),
        (0.1, 0.1 * (1 + 1e-12), True),
        (3.0, 3, True),
    ],
)
def test_a_stored_value_is_the_same_only_when_no_int_digit_changed(
    expected, found, same
):
    assert _same(expected, found) is same


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


LARGE = 2**53 + 1
LARGE_REJECT = (
    f"year: int {LARGE} is above 2**53 in magnitude and Weaviate would store it "
    "rounded; set the property type to text, or text[] for a list"
)


def _with_a_large_year(count: int, at: set[int]) -> tuple[list[Record], list[Record]]:
    """Records whose year is LARGE at the indexes `at`, and as 0.1.0 wrote them."""
    rounded = [replace(r, metadata={"year": 2**53}) for r in make_records(count)]
    source = [
        replace(r, metadata={"year": LARGE}) if i in at else r
        for i, r in enumerate(rounded)
    ]
    return source, rounded


def test_a_reject_whose_object_is_in_the_target_is_a_difference():
    source, rounded = _with_a_large_year(3, {1})
    check = records_check(source, ENTRY, converted(rounded, ENTRY))
    assert not check.passed
    assert check.detail == (
        "sampled 3, compared 2, rejects skipped 0, differing 1: "
        f'"id-1": is a reject under this plan ({LARGE_REJECT}), and the target '
        "holds an object for it; the target holds objects for records that this "
        "plan rejects. Fix the cause of each reject, then run to-weaviate sync. "
        "An existing collection keeps its property types, and a changed type "
        "stops sync with plan_mismatch: to change one, copy with another --dir "
        "into a target name that does not exist, or delete the collection first."
    )


def test_a_reject_without_an_object_in_the_target_is_skipped():
    source, rounded = _with_a_large_year(3, {1})
    stored = converted([r for r in rounded if r.id != "id-1"], ENTRY)
    check = records_check(source, ENTRY, stored)
    assert check.passed, check.detail
    assert check.detail == "sampled 3, compared 2, rejects skipped 1, differing 0"


def test_rejects_in_the_target_are_listed_when_every_sampled_record_is_a_reject():
    source, rounded = _with_a_large_year(2, {0, 1})
    check = records_check(source, ENTRY, converted(rounded[:1], ENTRY))
    assert not check.passed
    assert check.detail.startswith(
        "sampled 2, compared 0, rejects skipped 1, differing 1: "
        '"id-0": is a reject under this plan ('
    )
    assert "every sampled record" not in check.detail


def test_one_fetch_reads_the_compared_records_and_the_rejects():
    source_records, rounded = _with_a_large_year(3, {1})
    source, writer = FakeSource(source_records), FakeWriter()
    writer.write(ENTRY, converted(rounded, ENTRY))
    reader = FakeReader(writer)
    asked: list[list[str]] = []
    fetch = reader.fetch
    reader.fetch = lambda entry, uuids: asked.append(uuids) or fetch(entry, uuids)  # type: ignore[method-assign]
    verify_records(source, reader, ENTRY, collection_of(source), 3, random.Random(0))
    assert [sorted(uuids) for uuids in asked] == [
        sorted(to_uuid(r.id) for r in source_records)
    ]


class EmptySampleSource(FakeSource):
    """FakeSource whose sample holds no record, as after deletes in the source."""

    def sample(self, collection, size, rng, with_vectors):
        return Sample([], len(self.records))


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


class EmptyChroma:
    """The ChromaClient calls of a sample of an empty collection; counts the counts."""

    def __init__(self) -> None:
        self.counts = 0

    def count(self, tenant: str, database: str, collection_id: str) -> int:
        self.counts += 1
        return 0


def test_records_of_an_empty_chroma_collection_read_the_count_once():
    client = EmptyChroma()
    source = ChromaSource(client, "t")  # type: ignore[arg-type]
    collection = collection_of(FakeSource([]))
    reader = FakeReader(FakeWriter())
    check = verify_records(source, reader, ENTRY, collection, 300, random.Random(0))
    assert (check.passed, check.detail) == (True, "source is empty")
    assert client.counts == 1


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
