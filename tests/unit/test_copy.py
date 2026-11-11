import hashlib
import json
import math
from dataclasses import replace

import pytest
from fakes import FakeSource, FakeWriter

from to_weaviate import plan as p
from to_weaviate.engine.copy import (
    CopyResult,
    content_hash,
    copy_collection,
    plan_hash,
    record_hash,
)
from to_weaviate.events import Events
from to_weaviate.ids import to_uuid
from to_weaviate.sources.base import (
    Partition,
    Record,
    SourceChanged,
    SourceNotCompacted,
)
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
# ENTRY without the property n.
DROPPED = ENTRY.model_copy(
    update={
        "target": ENTRY.target.model_copy(
            update={
                "properties": ENTRY.target.properties[:-1],
                "dropped": [p.DroppedKey(source_key="n", reason="test")],
            }
        )
    }
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
    return [(r.id, record_hash(plan_hash(ENTRY), r)) for r in records]


def test_record_hash_is_blake2b_of_the_plan_hash_and_the_content_hash():
    plan = plan_hash(ENTRY)
    expected = hashlib.blake2b(plan + content_hash(BASE), digest_size=16).digest()
    assert record_hash(plan, BASE) == expected


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
    # The other worker can finish a page after the failure; it stays copied.
    checkpointed = state.progress()[0].copied
    writer.fail_on_call = None
    result = copy_collection(src, writer, state, events, ENTRY, C, 2)
    assert len(writer.objects) == 1000 and max(writer.writes_per_uuid.values()) <= 2
    # Only the page of the failed call is written again.
    assert sum(writer.writes_per_uuid.values()) == 1100
    assert result == CopyResult(copied=1000, rejects=0, written=1000 - checkpointed)


def test_second_run_reads_nothing(src, state, events):
    first = copy_collection(src, FakeWriter(), state, events, ENTRY, C, 2)
    reads = src.read_calls
    second = copy_collection(src, FakeWriter(), state, events, ENTRY, C, 2)
    assert src.read_calls == reads
    assert first == CopyResult(copied=1000, rejects=0, written=1000)
    assert second == CopyResult(copied=1000, rejects=0)


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
    assert result == CopyResult(copied=999, rejects=1, written=999)
    assert to_uuid("id-10") not in writer.objects


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_vector_value_that_is_not_a_number_is_a_reject(state, events, tmp_path, value):
    records = make_records(1000)
    records[10] = replace(records[10], vector=[1.0, value])
    writer = FakeWriter()
    result = copy_collection(FakeSource(records), writer, state, events, ENTRY, C, 2)
    assert result == CopyResult(copied=999, rejects=1, written=999)
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


def test_run_after_a_tail_reads_nothing_when_the_source_did_not_grow(
    src, state, events
):
    src.grow_after_partitions(make_records(50, start=1000))
    first = copy_collection(src, FakeWriter(), state, events, ENTRY, C, 2)
    reads = src.read_calls
    second = copy_collection(src, FakeWriter(), state, events, ENTRY, C, 2)
    assert src.read_calls == reads
    assert first == CopyResult(copied=1050, rejects=0, written=1050)
    assert second == CopyResult(copied=1050, rejects=0)


def test_each_run_copies_the_records_added_since_the_last_run(src, state, events):
    src.grow_after_partitions(make_records(50, start=1000))
    writer = FakeWriter()
    copy_collection(src, writer, state, events, ENTRY, C, 2)
    src.records.extend(make_records(50, start=1050))
    copy_collection(src, writer, state, events, ENTRY, C, 2)
    src.records.extend(make_records(50, start=1100))
    result = copy_collection(src, writer, state, events, ENTRY, C, 2)
    assert result == CopyResult(copied=1150, rejects=0, written=50)
    assert len(writer.objects) == 1150
    assert set(writer.writes_per_uuid.values()) == {1}
    assert state.progress()[0].total == 1150


def test_finished_tail_stores_its_end_offset(src, state, events):
    src.grow_after_partitions(make_records(250, start=1000))
    copy_collection(src, FakeWriter(), state, events, ENTRY, C, 2)
    assert state.partitions(C.ref)[-1] == Partition("1000-", "1000", "1250")


def test_tail_without_an_end_offset_gets_no_later_tail(src, state, events):
    # The state that a run before end offsets left: a finished tail, no stop.
    state.init_collection(C.ref, src.partitions(C, 2), C.count)
    for partition in state.partitions(C.ref):
        state.advance(C.ref, partition.key, None, 500)
    state.add_partition(C.ref, Partition("1000-", "1000", None))
    state.advance(C.ref, "1000-", None, 0)
    src.records.extend(make_records(50, start=1000))
    result = copy_collection(src, FakeWriter(), state, events, ENTRY, C, 2)
    assert src.read_calls == 0 and result.copied == 1000


class OverCount(FakeSource):
    """FakeSource whose count is `extra` above its records."""

    extra = 0

    def count(self, collection):
        return super().count(collection) + self.extra


def test_tail_after_a_tail_that_read_no_records(state, events):
    src = OverCount(make_records(1000))
    src.extra = 5
    writer = FakeWriter()
    copy_collection(src, writer, state, events, ENTRY, C, 2)
    src.extra = 0
    src.records.extend(make_records(5, start=1000))
    result = copy_collection(src, writer, state, events, ENTRY, C, 2)
    assert result.copied == 1005 and len(writer.objects) == 1005


class GrowsDuringTail(FakeSource):
    """FakeSource that appends `during_tail` on the first read of a tail."""

    during_tail: list[Record] = []

    def read(self, collection, partition, cursor, with_vectors):
        if partition.stop is None:
            self.records.extend(self.during_tail)
            self.during_tail = []
        return super().read(collection, partition, cursor, with_vectors)


def test_total_counts_the_records_added_while_the_tail_runs(state, events):
    src = GrowsDuringTail(make_records(1000))
    src.grow_after_partitions(make_records(50, start=1000))
    src.during_tail = make_records(30, start=1050)
    copy_collection(src, FakeWriter(), state, events, ENTRY, C, 2)
    progress = state.progress()[0]
    assert (progress.copied, progress.total) == (1080, 1080)


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


SOURCE_CHANGED = "the source changed since the last run; re-reading from the start"


def _warnings(tmp_path) -> list[dict]:
    lines = (tmp_path / "events.jsonl").read_text().splitlines()
    return [e for e in map(json.loads, lines) if e["level"] == "warning"]


def test_a_later_run_reads_the_records_added_after_a_deletion(src, state, events):
    writer = FakeWriter()
    copy_collection(src, writer, state, events, ENTRY, C, 2)
    del src.records[300:400]
    src.records.extend(make_records(150, start=1000))
    reads, id_reads = src.read_calls, src.id_reads
    result = copy_collection(src, writer, state, events, ENTRY, C, 2)
    assert {to_uuid(r.id) for r in src.records} <= set(writer.objects)
    # One id at offset 999, two id pages back to id-999 at offset 899, then
    # the two pages of the tail from offset 900.
    assert (src.id_reads - id_reads, src.read_calls - reads) == (3, 2)
    assert set(writer.writes_per_uuid.values()) == {1}
    assert state.partitions(C.ref)[-1] == Partition("900-", "900", "1050")
    # No record was read again.
    assert result.reread_from is None


def test_a_deleted_last_read_record_makes_the_run_read_from_the_start(
    src, state, events, tmp_path
):
    writer = FakeWriter()
    copy_collection(src, writer, state, events, ENTRY, C, 2)
    # id-499 and id-999 are the last records of the two partitions.
    src.records = [r for r in src.records if r.id not in {"id-499", "id-999"}]
    src.records.extend(make_records(50, start=1000))
    result = copy_collection(src, writer, state, events, ENTRY, C, 2)
    assert {to_uuid(r.id) for r in src.records} <= set(writer.objects)
    # The records read again keep their uuids, so no object is added twice.
    assert len(writer.objects) == 1050
    assert result.reread_from == 0 and result.copied == 1048
    assert [(e["event"], e["message"]) for e in _warnings(tmp_path)] == [
        ("source_changed", SOURCE_CHANGED)
    ]


def test_a_deleted_last_read_record_falls_back_to_an_earlier_partition(
    src, state, events, tmp_path
):
    writer = FakeWriter()
    copy_collection(src, writer, state, events, ENTRY, C, 2)
    src.records = [r for r in src.records if r.id != "id-999"]
    src.records.extend(make_records(50, start=1000))
    result = copy_collection(src, writer, state, events, ENTRY, C, 2)
    assert {to_uuid(r.id) for r in src.records} <= set(writer.objects)
    # id-499, the last record of the first partition, is still at offset 499.
    assert result.reread_from == 500 and result.copied == 1049
    assert _warnings(tmp_path) == []


def test_a_tail_that_reads_nothing_gives_no_reread_from(src, state, events, tmp_path):
    writer = FakeWriter()
    copy_collection(src, writer, state, events, ENTRY, C, 2)
    # Every record after id-499, the last record of the first partition, is gone.
    del src.records[500:]
    reads = src.read_calls
    result = copy_collection(src, writer, state, events, ENTRY, C, 2)
    assert state.partitions(C.ref)[-1] == Partition("500-", "500", "500")
    assert src.read_calls - reads == 1 and len(writer.objects) == 1000
    assert result.reread_from is None and _warnings(tmp_path) == []


def test_an_unchanged_source_costs_one_id_read(src, state, events):
    writer = FakeWriter()
    copy_collection(src, writer, state, events, ENTRY, C, 2)
    reads, id_reads = src.read_calls, src.id_reads
    result = copy_collection(src, writer, state, events, ENTRY, C, 2)
    assert (src.id_reads - id_reads, src.read_calls - reads) == (1, 0)
    assert result.reread_from is None
    src.records.extend(make_records(50, start=1000))
    result = copy_collection(src, writer, state, events, ENTRY, C, 2)
    assert src.id_reads - id_reads == 2
    assert state.partitions(C.ref)[-1] == Partition("1000-", "1000", "1050")
    assert result.reread_from is None


def test_a_state_without_last_ids_reads_from_the_start(src, state, events, tmp_path):
    # Finished partitions of a state written before the last id was stored.
    state.init_collection(C.ref, src.partitions(C, 2), C.count)
    for partition in state.partitions(C.ref):
        state.advance(C.ref, partition.key, None, 500)
    writer = FakeWriter()
    result = copy_collection(src, writer, state, events, ENTRY, C, 2)
    assert len(writer.objects) == 1000 and src.id_reads == 0
    assert result.reread_from == 0 and result.copied == 1000
    assert [e["message"] for e in _warnings(tmp_path)] == [SOURCE_CHANGED]


@pytest.mark.parametrize("deleted_below", [5, 3], ids=["scan", "probe"])
def test_a_last_read_record_added_again_does_not_hide_the_records_before_it(
    src, state, events, deleted_below
):
    writer = FakeWriter()
    copy_collection(src, writer, state, events, ENTRY, C, 2)
    last = src.records[999]
    # id-999, the last record read, and records below it are deleted; 3
    # records are added, then id-999 again, which goes after them. With 5
    # deleted below it, id-999 is at offset 997; with 3, at offset 999.
    gone = {last.id, *(f"id-{i}" for i in range(900, 900 + deleted_below))}
    src.records = [r for r in src.records if r.id not in gone]
    src.records.extend([*make_records(3, start=1000), last])
    result = copy_collection(src, writer, state, events, ENTRY, C, 2)
    assert {to_uuid(r.id) for r in src.records} <= set(writer.objects)
    # The id before id-999 is not id-998, so the run reads from after id-499.
    assert result.reread_from == 500


def test_a_one_record_partition_has_no_previous_id_and_its_last_id_is_taken(
    state, events, tmp_path
):
    src = FakeSource(make_records(2))
    two = replace(C, count=2)
    writer = FakeWriter()
    copy_collection(src, writer, state, events, ENTRY, two, 2)
    assert state.previous_ids(C.ref) == {}
    del src.records[0]
    src.records.extend(make_records(3, start=2))
    result = copy_collection(src, writer, state, events, ENTRY, two, 2)
    assert {to_uuid(r.id) for r in src.records} <= set(writer.objects)
    # id-1, the last id of the partition 1-2, is at offset 0 now.
    assert state.partitions(C.ref)[-1] == Partition("1-", "1", "4")
    assert result.reread_from is None and _warnings(tmp_path) == []


def test_a_copy_in_progress_in_an_older_state_resumes_without_a_reread(
    src, state, events, tmp_path
):
    # An older state: the first partition finished, the second stopped at 700.
    state.init_collection(C.ref, src.partitions(C, 2), C.count)
    state.advance(C.ref, "0-500", None, 500)
    state.advance(C.ref, "500-1000", "700", 200)
    writer = FakeWriter()
    result = copy_collection(src, writer, state, events, ENTRY, C, 2)
    assert src.read_calls == 3 and len(writer.objects) == 300
    assert result == CopyResult(copied=1000, rejects=0, written=300)
    assert _warnings(tmp_path) == []


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


def test_skip_unchanged_counts_the_skipped_records_as_copied(src, state, events):
    copy_collection(src, FakeWriter(), state, events, ENTRY, C, 2)
    src.records[5] = replace(src.records[5], document="changed")
    result = copy_collection(
        src, FakeWriter(), state, events, ENTRY, C, 2, skip_unchanged=True, fresh=True
    )
    assert (result.copied, result.written) == (1000, 1)
    # status reads this count, so a sync leaves every record counted.
    assert state.progress()[0].copied == 1000


def test_a_copy_skips_no_record(src, state, events):
    result = copy_collection(src, FakeWriter(), state, events, ENTRY, C, 2)
    assert result.written == 1000


def test_record_hashes_follow_the_plan_entry(src, state, events):
    copy_collection(src, FakeWriter(), state, events, ENTRY, C, 2)
    assert state.unchanged(C.ref, hashes(src.records[:2])) == {"id-0", "id-1"}
    other = [(r.id, record_hash(plan_hash(DROPPED), r)) for r in src.records[:2]]
    assert state.unchanged(C.ref, other) == set()


def test_a_copy_keeps_the_plan_hash_that_is_stored(src, state, events):
    copy_collection(src, FakeWriter(), state, events, ENTRY, C, 2)
    copy_collection(src, FakeWriter(), state, events, DROPPED, C, 2)
    assert state.set_plan_hash(C.ref, plan_hash(ENTRY)) is False


def test_plan_hash_ignores_the_reason_of_a_dropped_key():
    def dropping(reason: str) -> p.CollectionPlan:
        return DROPPED.model_copy(
            update={
                "target": DROPPED.target.model_copy(
                    update={"dropped": [p.DroppedKey(source_key="n", reason=reason)]}
                )
            }
        )

    assert plan_hash(dropping("one reason")) == plan_hash(dropping("another"))


def test_plan_hash_ignores_a_field_set_to_its_default():
    explicit = ENTRY.target.model_copy(update={"tenant": None, "dropped": []})
    assert plan_hash(ENTRY.model_copy(update={"target": explicit})) == plan_hash(ENTRY)


def test_plan_hash_follows_the_target_and_not_the_source():
    other_source = ENTRY.model_copy(
        update={"source": ENTRY.source.model_copy(update={"count": 5})}
    )
    assert plan_hash(other_source) == plan_hash(ENTRY)
    assert len(plan_hash(ENTRY)) == 16
    dropped = ENTRY.model_copy(
        update={
            "target": ENTRY.target.model_copy(
                update={
                    "properties": ENTRY.target.properties[:-1],
                    "dropped": [p.DroppedKey(source_key="n", reason="test")],
                }
            )
        }
    )
    assert plan_hash(dropped) != plan_hash(ENTRY)


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


def test_the_wait_comes_before_the_first_page_and_the_state_after_the_last_read(
    src, state, events
):
    src.compaction = [{"version": 1, "log_position": 1000}]
    assert copy_collection(src, FakeWriter(), state, events, ENTRY, C, 2).copied == 1000
    assert src.calls[0] == "wait_compacted" and src.calls[-1] == "compaction_state"
    assert src.calls.count("compaction_state") == 1 and "read" in src.calls


@pytest.mark.parametrize(
    "after",
    [{"version": 2, "log_position": 1000}, {"version": 1, "log_position": 1300}],
    ids=["version", "log_position"],
)
def test_a_changed_state_stops_the_copy_and_the_next_run_reads_from_the_start(
    src, state, events, after
):
    before = {"version": 1, "log_position": 1000}
    src.compaction = [before, after]
    writer = FakeWriter()
    with pytest.raises(SourceChanged) as stopped:
        copy_collection(src, writer, state, events, ENTRY, C, 2)
    assert (stopped.value.collection, stopped.value.before, stopped.value.after) == (
        "docs",
        before,
        after,
    )
    assert state.pending(C.ref) == [(Partition("0-", "0", None), None)]
    assert state.progress()[0].copied == 0 and not state.progress()[0].done
    # `after` is the last item of `compaction`, so the state is unchanged now.
    result = copy_collection(src, writer, state, events, ENTRY, C, 2)
    assert result.copied == 1000
    assert set(writer.writes_per_uuid.values()) == {2}


def test_a_source_without_compaction_is_not_compared(src, state, events):
    assert copy_collection(src, FakeWriter(), state, events, ENTRY, C, 2).copied == 1000
    assert src.calls.count("wait_compacted") == 1
    assert "compaction_state" not in src.calls


def test_the_wait_writes_a_compaction_wait_event_per_change(
    src, state, events, tmp_path
):
    src.waits = [(1200, 40000), (200, 40000)]
    src.compaction = [{"version": 4, "log_position": 40000}]
    seen = []
    copy_collection(
        src,
        FakeWriter(),
        state,
        events,
        ENTRY,
        C,
        2,
        waiting=lambda u, t: seen.append((u, t)),
    )
    lines = (tmp_path / "events.jsonl").read_text().splitlines()
    named = [e for e in map(json.loads, lines) if e["event"] != "progress"]
    assert [
        (e["level"], e["event"], e.get("unindexed"), e.get("total")) for e in named
    ] == [
        ("info", "compaction_wait", 1200, 40000),
        ("info", "compaction_wait", 200, 40000),
        ("info", "collection_started", None, 1000),
        ("info", "collection_finished", None, None),
    ]
    assert named[0]["ref"] == C.ref and seen == [(1200, 40000), (200, 40000)]


def test_a_wait_that_ends_without_compaction_reads_no_page(src, state, events):
    src.wait_error = SourceNotCompacted("docs", 1200, 40000, 600.0)
    with pytest.raises(SourceNotCompacted):
        copy_collection(src, FakeWriter(), state, events, ENTRY, C, 2)
    assert src.read_calls == 0 and state.progress() == []
