import hashlib
import json
import math
from dataclasses import replace

import pytest
from fakes import FakeSource, FakeWriter

from to_weaviate import plan as p
from to_weaviate.engine.copy import CopyResult, content_hash, copy_collection
from to_weaviate.events import Events
from to_weaviate.ids import to_uuid
from to_weaviate.sources.base import Partition, Record
from to_weaviate.state import State


def make_records(n: int, start: int = 0) -> list[Record]:
    return [
        Record(f"id-{i}", [float(i), 1.0], f"doc {i}", None, {"n": i})
        for i in range(start, start + n)
    ]


C = FakeSource(make_records(1000)).inspect(None, None, None)[0]
ENTRY = p.CollectionPlan(
    source=p.SourceRef(
        ref=C.ref,
        namespace=C.namespace,
        name=C.name,
        count=C.count,
        dimension=C.dimension,
        distance=C.distance,
        embedding=None,
    ),
    target=p.TargetPlan(
        name="Docs",
        vectors=[p.VectorPlan(name="default", mode="keep", distance="cosine")],
        properties=[
            p.PropertyPlan(source_key="#document", name="document", type="text"),
            p.PropertyPlan(source_key="#id", name="source_id", type="text"),
            p.PropertyPlan(source_key="#uri", name="uri", type="text"),
            p.PropertyPlan(source_key="n", name="n", type="int"),
        ],
    ),
)
BASE = Record("a", [0.1, 0.2], "dóc", "file://x", {"k": 1, "j": "v"})


@pytest.fixture
def src() -> FakeSource:
    return FakeSource(make_records(1000))


@pytest.fixture
def state(tmp_path):
    opened = State(tmp_path / "state.sqlite")
    yield opened
    opened.close()


@pytest.fixture
def events(tmp_path):
    opened = Events(tmp_path / "events.jsonl")
    yield opened
    opened.close()


def hashes(records: list[Record]) -> list[tuple[str, bytes]]:
    return [(r.id, content_hash(r)) for r in records]


def test_content_hash_is_blake2b_of_the_json_parts():
    payload = json.dumps(
        ["a", "dóc", "file://x", {"k": 1, "j": "v"}, [0.1, 0.2]],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    expected = hashlib.blake2b(payload.encode("utf-8"), digest_size=16).digest()
    assert content_hash(BASE) == expected


@pytest.mark.parametrize(
    "changed",
    [
        replace(BASE, id="b"),
        replace(BASE, document="other"),
        replace(BASE, uri=None),
        replace(BASE, metadata={"k": 2, "j": "v"}),
        replace(BASE, vector=[0.1, 0.3]),
        replace(BASE, vector=None),
    ],
)
def test_content_hash_changes_with_every_part(changed):
    assert content_hash(changed) != content_hash(BASE)


def test_content_hash_ignores_metadata_key_order():
    assert content_hash(replace(BASE, metadata={"j": "v", "k": 1})) == content_hash(
        BASE
    )


def test_copies_every_record_once(src, state, events):
    writer = FakeWriter()
    result = copy_collection(src, writer, state, events, ENTRY, C, 4)
    assert len(writer.objects) == 1000 and result.copied == 1000
    assert set(writer.writes_per_uuid.values()) == {1}


def test_resumes_after_failure(src, state, events):
    writer = FakeWriter(fail_on_call=3)
    with pytest.raises(RuntimeError, match="call 3$"):
        copy_collection(src, writer, state, events, ENTRY, C, 2)
    writer.fail_on_call = None
    result = copy_collection(src, writer, state, events, ENTRY, C, 2)
    assert len(writer.objects) == 1000 and max(writer.writes_per_uuid.values()) <= 2
    # Only the page of the failed call is written again.
    assert sum(writer.writes_per_uuid.values()) == 1100
    assert result == CopyResult(copied=1000, rejects=0)


def test_second_run_reads_nothing(src, state, events):
    first = copy_collection(src, FakeWriter(), state, events, ENTRY, C, 2)
    reads = src.read_calls
    second = copy_collection(src, FakeWriter(), state, events, ENTRY, C, 2)
    assert src.read_calls == reads
    assert second == first == CopyResult(copied=1000, rejects=0)


def test_rejects_are_recorded(src, state, events):
    writer = FakeWriter(reject_ids={"id-3", "id-700"})
    result = copy_collection(src, writer, state, events, ENTRY, C, 2)
    assert result.rejects == 2 and state.progress()[0].rejects == 2
    assert result.copied == 998 and len(writer.objects) == 998
    kept = src.records[4]
    assert state.unchanged(C.ref, hashes(src.records[3:5])) == {kept.id}


def test_conversion_rejects_are_recorded(state, events):
    records = make_records(1000)
    records[10] = replace(records[10], vector=None)
    writer = FakeWriter()
    result = copy_collection(FakeSource(records), writer, state, events, ENTRY, C, 2)
    assert result == CopyResult(copied=999, rejects=1)
    assert to_uuid("id-10") not in writer.objects


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_vector_value_that_is_not_a_number_is_a_reject(state, events, tmp_path, value):
    records = make_records(1000)
    records[10] = replace(records[10], vector=[1.0, value])
    writer = FakeWriter()
    result = copy_collection(FakeSource(records), writer, state, events, ENTRY, C, 2)
    assert result == CopyResult(copied=999, rejects=1)
    assert to_uuid("id-10") not in writer.objects
    reject = json.loads((tmp_path / "rejects.jsonl").read_text())
    assert (reject["source_id"], reject["reason"]) == (
        "id-10",
        "embedding holds a value that is not a number",
    )


def test_copies_records_added_during_run(src, state, events):
    src.grow_after_partitions(make_records(50, start=1000))
    writer = FakeWriter()
    result = copy_collection(src, writer, state, events, ENTRY, C, 2)
    assert len(writer.objects) == 1050 and result.copied == 1050


def test_later_run_does_not_add_a_second_tail(src, state, events):
    src.grow_after_partitions(make_records(50, start=1000))
    first = copy_collection(src, FakeWriter(), state, events, ENTRY, C, 2)
    reads = src.read_calls
    second = copy_collection(src, FakeWriter(), state, events, ENTRY, C, 2)
    assert src.read_calls == reads
    assert second == first == CopyResult(copied=1050, rejects=0)


def test_resume_with_a_larger_count_covers_the_gap(src, state, events):
    writer = FakeWriter(fail_on_call=3)
    with pytest.raises(RuntimeError, match="call 3$"):
        copy_collection(src, writer, state, events, ENTRY, C, 2)
    src.records.extend(make_records(100, start=1000))
    writer.fail_on_call = None
    grown = replace(C, count=1100)
    result = copy_collection(src, writer, state, events, ENTRY, grown, 2)
    assert len(writer.objects) == 1100 and result.copied == 1100


def test_failed_run_adds_no_tail(src, state, events):
    src.grow_after_partitions(make_records(50, start=1000))
    with pytest.raises(RuntimeError, match="call 3$"):
        copy_collection(src, FakeWriter(fail_on_call=3), state, events, ENTRY, C, 2)
    assert [p for p in state.partitions(C.ref) if p.stop is None] == []


def test_empty_collection_that_grew_gets_a_tail(state, events):
    writer = FakeWriter()
    empty = replace(C, count=0)
    copy_collection(
        FakeSource(make_records(30)), writer, state, events, ENTRY, empty, 2
    )
    assert len(writer.objects) == 30


def test_parallelism_is_bounded(state, events):
    src = FakeSource(make_records(1000), read_delay=0.005)
    # Partitions stored by an earlier run with parallelism 8.
    state.init_collection(C.ref, src.partitions(C, 8), C.count)
    writer = FakeWriter()
    copy_collection(src, writer, state, events, ENTRY, C, 3)
    assert 2 <= src.peak_concurrent_reads <= 3
    assert len(writer.objects) == 1000


@pytest.mark.parametrize(("fail_on_call", "cursor"), [(1, None), (3, "200")])
def test_page_is_not_done_when_its_write_raises(
    src, state, events, fail_on_call, cursor
):
    writer = FakeWriter(fail_on_call=fail_on_call)
    with pytest.raises(RuntimeError, match=f"call {fail_on_call}$"):
        copy_collection(src, writer, state, events, ENTRY, C, 1)
    assert state.pending(C.ref) == [(Partition("0-1000", "0", "1000"), cursor)]
    written = (fail_on_call - 1) * 100
    assert state.progress()[0].copied == written
    failed_page = src.records[written : written + 100]
    assert state.unchanged(C.ref, hashes(failed_page)) == set()


def test_other_workers_stop_after_a_write_raises(src, state, events):
    writer = FakeWriter(fail_on_call=1)
    with pytest.raises(RuntimeError, match="call 1$"):
        copy_collection(src, writer, state, events, ENTRY, C, 2)
    # Without the stop, the other worker reads all five pages of its partition.
    assert src.read_calls <= 3


def test_skip_unchanged_writes_only_changed_records(src, state, events):
    copy_collection(src, FakeWriter(), state, events, ENTRY, C, 2)
    src.records[5] = replace(src.records[5], document="changed")
    state.reset(C.ref)
    writer = FakeWriter()
    copy_collection(src, writer, state, events, ENTRY, C, 2, skip_unchanged=True)
    assert list(writer.objects) == [to_uuid("id-5")]
    assert writer.objects[to_uuid("id-5")].properties["document"] == "changed"


def test_emits_collection_started_and_finished(src, state, events, tmp_path):
    copy_collection(src, FakeWriter(reject_ids={"id-1"}), state, events, ENTRY, C, 2)
    lines = (tmp_path / "events.jsonl").read_text().splitlines()
    named = [e for e in map(json.loads, lines) if e["event"] != "progress"]
    assert [(e["level"], e["event"]) for e in named] == [
        ("info", "collection_started"),
        ("info", "collection_finished"),
    ]
    assert (named[0]["ref"], named[0]["total"]) == (C.ref, 1000)
    assert (named[1]["ref"], named[1]["copied"], named[1]["rejects"]) == (
        C.ref,
        999,
        1,
    )
