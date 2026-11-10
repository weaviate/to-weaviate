import json
import math
import sqlite3
import uuid
from contextlib import closing
from dataclasses import replace
from pathlib import Path

import pytest
from fakes import FakeDeleter, FakeReader, FakeSource, FakeWriter

from to_weaviate import plan as p
from to_weaviate.engine.copy import copy_collection
from to_weaviate.engine.sync import (
    PLAN_CHANGED,
    SourceWriting,
    SyncResult,
    over_prune_limit,
    sync_collection,
)
from to_weaviate.engine.verify import ID_BATCH
from to_weaviate.events import Events
from to_weaviate.ids import to_uuid
from to_weaviate.sources.base import Record, SourceCollection
from to_weaviate.state import State
from to_weaviate.target.writer import TargetObject


def make_records(n: int, start: int = 0) -> list[Record]:
    return [
        Record(f"id-{i}", [float(i), 1.0], f"doc {i}", None, {"n": i})
        for i in range(start, start + n)
    ]


def make_entry(c: SourceCollection, tenant: str | None = None) -> p.CollectionPlan:
    return p.CollectionPlan(
        source=p.SourceRef(
            ref=c.ref,
            namespace=c.namespace,
            name=c.name,
            count=c.count,
            dimension=c.dimension,
            distance=c.distance,
            embedding=None,
        ),
        target=p.TargetPlan(
            name="Docs",
            tenant=tenant,
            vectors=[p.VectorPlan(name="default", mode="keep", distance="cosine")],
            properties=[
                p.PropertyPlan(source_key="#document", name="document", type="text"),
                p.PropertyPlan(source_key="#id", name="source_id", type="text"),
                p.PropertyPlan(source_key="#uri", name="uri", type="text"),
                p.PropertyPlan(source_key="n", name="n", type="int"),
            ],
        ),
    )


C = FakeSource(make_records(1000)).inspect(None, None, None)[0]
ENTRY = make_entry(C)
# ENTRY without the property n: the objects of unchanged records differ.
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
FOREIGN = TargetObject("00000000-0000-0000-0000-00000000f00d", {"document": "x"}, {})


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


def copied(
    src: FakeSource, state: State, events: Events, writer: FakeWriter | None = None
) -> FakeWriter:
    writer = writer or FakeWriter()
    copy_collection(src, writer, state, events, ENTRY, C, 2)
    return writer


def sync(
    src: FakeSource,
    writer: FakeWriter,
    state: State,
    events: Events,
    workdir: Path,
    entry: p.CollectionPlan = ENTRY,
    collection: SourceCollection = C,
    prune: bool = False,
    force: bool = False,
    warn=None,
    delete_error: Exception | None = None,
    parallelism: int = 2,
) -> tuple[SyncResult, FakeDeleter]:
    deleter = FakeDeleter(writer, error=delete_error)
    result = sync_collection(
        src,
        writer,
        FakeReader(writer),
        deleter,
        state,
        events,
        entry,
        collection,
        parallelism,
        prune,
        force,
        workdir,
        warn=warn,
    )
    return result, deleter


def event_lines(workdir: Path) -> list[dict]:
    lines = (workdir / "events.jsonl").read_text().splitlines()
    return [json.loads(line) for line in lines]


def test_a_changed_record_is_written_and_the_others_are_counted_unchanged(
    src, state, events, tmp_path
):
    writer = copied(src, state, events)
    src.records[5] = replace(src.records[5], document="changed")
    result, _ = sync(src, writer, state, events, tmp_path)
    assert (result.copied, result.unchanged, result.stale) == (1, 999, 0)
    assert result.rejects == 0 and result.deleted == 0
    twice = [u for u, n in writer.writes_per_uuid.items() if n == 2]
    assert twice == [to_uuid("id-5")] and set(writer.writes_per_uuid.values()) == {1, 2}
    assert writer.objects[to_uuid("id-5")].properties["document"] == "changed"


def test_an_added_record_is_copied(src, state, events, tmp_path):
    writer = copied(src, state, events)
    src.records.extend(make_records(1, start=1000))
    result, _ = sync(src, writer, state, events, tmp_path)
    assert (result.copied, result.unchanged, result.stale) == (1, 1000, 0)
    assert len(writer.objects) == 1001


def test_a_deleted_record_is_stale_and_kept_without_prune(src, state, events, tmp_path):
    writer = copied(src, state, events)
    del src.records[3]
    result, deleter = sync(src, writer, state, events, tmp_path)
    assert (result.stale, result.deleted, result.prune_refused) == (1, 0, False)
    assert FakeReader(writer).count(ENTRY) == 1000 and deleter.calls == 0


def test_prune_deletes_the_stale_objects_and_their_hashes(src, state, events, tmp_path):
    writer = copied(src, state, events)
    removed = src.records.pop(3)
    result, deleter = sync(src, writer, state, events, tmp_path, prune=True)
    assert (result.stale, result.deleted) == (1, 1)
    assert FakeReader(writer).count(ENTRY) == 999
    assert deleter.deleted == [("Docs", None, to_uuid("id-3"))]
    # The record comes back unchanged: without its hash it is written again.
    src.records.insert(3, removed)
    result, _ = sync(src, writer, state, events, tmp_path, prune=True)
    assert (result.copied, result.unchanged, result.stale) == (1, 999, 0)


def test_prune_above_ten_percent_is_refused_without_force(src, state, events, tmp_path):
    writer = copied(src, state, events)
    del src.records[:101]
    result, deleter = sync(src, writer, state, events, tmp_path, prune=True)
    assert (result.stale, result.deleted, result.prune_refused) == (101, 0, True)
    assert FakeReader(writer).count(ENTRY) == 1000 and deleter.calls == 0
    result, _ = sync(src, writer, state, events, tmp_path, prune=True, force=True)
    assert (result.stale, result.deleted, result.prune_refused) == (101, 101, False)
    assert FakeReader(writer).count(ENTRY) == 899


def test_prune_of_ten_percent_is_allowed(src, state, events, tmp_path):
    writer = copied(src, state, events)
    del src.records[:100]
    result, _ = sync(src, writer, state, events, tmp_path, prune=True)
    assert (result.stale, result.deleted, result.prune_refused) == (100, 100, False)


@pytest.mark.parametrize(
    ("stale", "total", "over"),
    [
        (100, 1000, False),
        (101, 1000, True),
        (1, 10, False),
        (2, 10, True),
        (0, 0, False),
    ],
)
def test_over_prune_limit(stale, total, over):
    assert over_prune_limit(stale, total) is over


class LosingTailSource(FakeSource):
    """A source that loses its last record, once, after a page has read it."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._lost = False

    def read(self, collection, partition, cursor, with_vectors):
        page = super().read(collection, partition, cursor, with_vectors)
        last = self.records[-1].id
        if not self._lost and any(r.id == last for r in page.records):
            self.records.pop()
            self._lost = True
        return page


def test_a_tail_that_reads_records_again_counts_each_record_once(
    src, state, events, tmp_path
):
    writer = copied(src, state, events)
    # The last record goes while the pass reads it, so the tail reads the
    # end of the collection again and supersedes a stored partition.
    lossy = LosingTailSource(src.records)
    result, _ = sync(lossy, writer, state, events, tmp_path)
    assert (result.copied, result.unchanged, result.rejects) == (0, 999, 0)
    assert (result.stale, result.deleted) == (1, 0)
    assert state.progress()[0].copied == 999


def test_a_sync_that_fails_before_its_first_page_keeps_the_progress(
    src, state, events, tmp_path
):
    copied(src, state, events)
    src.wait_error = RuntimeError("source down")
    with pytest.raises(RuntimeError, match="source down"):
        sync(src, FakeWriter(), state, events, tmp_path)
    progress = state.progress()[0]
    assert (progress.copied, progress.rejects, progress.done) == (1000, 0, True)
    assert state.has_progress(C.ref)


def test_a_changed_record_that_the_tail_reads_again_counts_once(
    src, state, events, tmp_path
):
    writer = copied(src, state, events)
    src.records[990] = replace(src.records[990], document="changed")
    # The tail reads id-990 again after the pass wrote it: its hash matches.
    lossy = LosingTailSource(src.records)
    result, _ = sync(lossy, writer, state, events, tmp_path)
    assert (result.copied, result.unchanged) == (1, 998)
    assert state.progress()[0].copied == 999


def test_a_sync_sets_the_total_to_the_count_it_read(src, state, events, tmp_path):
    writer = copied(src, state, events)
    del src.records[995:]
    sync(src, writer, state, events, tmp_path, prune=True)
    progress = state.progress()[0]
    assert (progress.copied, progress.total, progress.done) == (995, 995, True)


def test_an_object_missing_from_the_target_is_written_by_the_same_sync(
    src, state, events, tmp_path
):
    writer = copied(src, state, events)
    # Deleted in Weaviate by something other than the tool.
    del writer.stored[("Docs", None)][to_uuid("id-7")]
    result, _ = sync(src, writer, state, events, tmp_path)
    assert (result.copied, result.unchanged) == (1, 999)
    assert FakeReader(writer).count(ENTRY) == 1000
    result, _ = sync(src, writer, state, events, tmp_path)
    assert (result.copied, result.unchanged) == (0, 1000)


def test_a_rejected_record_is_sent_to_the_target_once_per_sync(
    src, state, events, tmp_path
):
    writer = copied(src, state, events, FakeWriter(reject_ids={"id-3"}))
    calls = writer.calls
    result, _ = sync(src, writer, state, events, tmp_path)
    # The page of id-3 is written; the missing id-3 is not written again.
    assert result.rejects == 1 and writer.calls == calls + 1


class DeleteAndAddDuringListing(FakeSource):
    """Deletes id-0 and adds id-1000 after the first page of an offset listing.

    The listing then skips id-300, which moved into the page already read,
    and the source count stays the same.
    """

    PAGE = 300

    def ids(self, collection):
        offset = 0
        while page := [r.id for r in self.records[offset : offset + self.PAGE]]:
            yield from page
            if offset == 0:
                del self.records[0]
                self.records.extend(make_records(1, start=1000))
            offset += len(page)


def test_prune_keeps_the_object_of_a_record_the_listing_skipped(
    state, events, tmp_path
):
    src = DeleteAndAddDuringListing(make_records(1000))
    writer = copied(src, state, events)
    result, deleter = sync(src, writer, state, events, tmp_path, prune=True)
    assert (result.stale, result.deleted) == (0, 0) and deleter.deleted == []
    assert to_uuid("id-300") in writer.objects
    # id-1000 was added during the listing: the sync wrote it.
    assert (result.copied, result.unchanged) == (1, 1000)
    assert to_uuid("id-1000") in writer.objects


def test_prune_keeps_an_object_whose_source_id_names_a_record_of_the_source(
    src, state, events, tmp_path
):
    writer = copied(src, state, events)
    other = TargetObject(
        "00000000-0000-0000-0000-0000000000a1", {"source_id": "id-1"}, {}
    )
    writer.write(ENTRY, [other])
    result, _ = sync(src, writer, state, events, tmp_path, prune=True)
    assert (result.stale, result.deleted) == (0, 0)
    assert other.uuid in writer.objects
    # The hash of id-1 stays, so the next sync does not write it.
    result, _ = sync(src, writer, state, events, tmp_path, prune=True)
    assert (result.copied, result.unchanged) == (0, 1000)


def test_the_prune_limit_counts_only_objects_with_a_source_id(state, events, tmp_path):
    src = FakeSource(make_records(10))
    c = src.inspect(None, None, None)[0]
    entry = make_entry(c)
    writer = FakeWriter()
    copy_collection(src, writer, state, events, entry, c, 2)
    foreign = [
        TargetObject(str(uuid.UUID(int=i)), {"document": "x"}, {}) for i in range(1, 91)
    ]
    writer.write(entry, foreign)
    del src.records[:2]
    result, deleter = sync(src, writer, state, events, tmp_path, entry, c, prune=True)
    # 2 of the 10 objects with a source_id: above the limit.
    assert (result.stale, result.deleted, result.prune_refused) == (2, 0, True)
    assert deleter.calls == 0


def test_prune_deletes_in_batches(state, events, tmp_path):
    src = FakeSource(make_records(2500))
    c = src.inspect(None, None, None)[0]
    entry = make_entry(c)
    writer = FakeWriter()
    copy_collection(src, writer, state, events, entry, c, 2)
    src.records.clear()
    result, deleter = sync(
        src, writer, state, events, tmp_path, entry, c, prune=True, force=True
    )
    assert (result.stale, result.deleted) == (2500, 2500)
    assert deleter.calls == math.ceil(2500 / ID_BATCH)


def test_a_plan_edit_keeps_the_record_hashes_until_the_records_are_written(
    src, state, events, tmp_path
):
    writer = copied(src, state, events)
    writer.fail_on_call = writer.calls + 1
    with pytest.raises(RuntimeError, match="fake write failure"):
        sync(src, writer, state, events, tmp_path, DROPPED, parallelism=1)
    # The first page failed, so no partition advanced: the target guard of the
    # next run depends on the record hashes alone.
    assert state.progress()[0].copied == 0
    assert state.has_progress(C.ref)
    writer.fail_on_call = None
    result, _ = sync(src, writer, state, events, tmp_path, DROPPED)
    assert (result.copied, result.unchanged) == (1000, 0)
    assert all("n" not in o.properties for o in writer.objects.values())


def test_a_copy_after_a_plan_edit_leaves_the_warning_to_the_next_sync(
    src, state, events, tmp_path
):
    writer = copied(src, state, events)
    copy_collection(src, writer, state, events, DROPPED, C, 2)
    warnings: list[str] = []
    result, _ = sync(
        src, writer, state, events, tmp_path, DROPPED, warn=warnings.append
    )
    assert (result.copied, result.unchanged, result.plan_changed) == (1000, 0, True)
    assert warnings == [PLAN_CHANGED]


def test_record_hashes_from_a_state_without_a_plan_hash_are_written_again(
    src, state, events, tmp_path
):
    writer = copied(src, state, events)
    # A state.sqlite of an earlier version holds record hashes and no plan hash.
    with closing(sqlite3.connect(tmp_path / "state.sqlite")) as db:
        db.execute("DELETE FROM plans")
        db.commit()
    result, _ = sync(src, writer, state, events, tmp_path, DROPPED)
    assert (result.copied, result.unchanged) == (1000, 0)
    assert all("n" not in o.properties for o in writer.objects.values())


def test_a_failed_deletion_leaves_no_hash_behind(src, state, events, tmp_path):
    writer = copied(src, state, events)
    removed = src.records.pop(3)
    with pytest.raises(RuntimeError, match="delete failed"):
        sync(
            src,
            writer,
            state,
            events,
            tmp_path,
            prune=True,
            delete_error=RuntimeError("delete failed"),
        )
    assert FakeReader(writer).count(ENTRY) == 1000
    # The hash went before the deletion, so the record that comes back is
    # written again even though its object was never deleted.
    src.records.insert(3, removed)
    result, _ = sync(src, writer, state, events, tmp_path)
    assert (result.copied, result.unchanged, result.stale) == (1, 999, 0)
    assert writer.writes_per_uuid[to_uuid("id-3")] == 2


def test_a_pass_that_fails_deletes_nothing(src, state, events, tmp_path):
    writer = copied(src, state, events)
    del src.records[3]
    src.records[5] = replace(src.records[5], document="changed")
    writer.fail_on_call = writer.calls + 1
    with pytest.raises(RuntimeError, match="fake write failure"):
        sync(src, writer, state, events, tmp_path, prune=True)
    assert FakeReader(writer).count(ENTRY) == 1000


def test_prune_deletes_only_in_the_tenant_of_the_entry(state, events, tmp_path):
    src_a, src_b = FakeSource(make_records(10)), FakeSource(make_records(10, start=100))
    src_b.REF = "fake-b"
    c_a, c_b = src_a.inspect(None, None, None)[0], src_b.inspect(None, None, None)[0]
    entry_a, entry_b = make_entry(c_a, "a"), make_entry(c_b, "b")
    writer = FakeWriter()
    copy_collection(src_a, writer, state, events, entry_a, c_a, 2)
    copy_collection(src_b, writer, state, events, entry_b, c_b, 2)
    del src_a.records[0]
    result, deleter = sync(
        src_a, writer, state, events, tmp_path, entry_a, c_a, prune=True
    )
    assert (result.stale, result.deleted) == (1, 1)
    reader = FakeReader(writer)
    assert (reader.count(entry_a), reader.count(entry_b)) == (9, 10)
    assert deleter.deleted == [("Docs", "a", to_uuid("id-0"))]


def test_an_object_without_source_id_is_not_stale(src, state, events, tmp_path):
    writer = copied(src, state, events)
    writer.write(ENTRY, [FOREIGN])
    result, deleter = sync(src, writer, state, events, tmp_path, prune=True)
    assert (result.stale, result.deleted) == (0, 0) and deleter.calls == 0
    assert FOREIGN.uuid in writer.objects


class ShrinkingSource(FakeSource):
    """A source that loses its last record while its ids are listed."""

    def ids(self, collection):
        yield from [r.id for r in self.records]
        self.records.pop()


def test_prune_stops_when_the_source_changes_during_the_id_listing(
    state, events, tmp_path
):
    src = ShrinkingSource(make_records(1000))
    writer = copied(src, state, events)
    with pytest.raises(SourceWriting) as info:
        sync(src, writer, state, events, tmp_path, prune=True)
    assert (info.value.before, info.value.listed, info.value.after) == (1000, 1000, 999)
    assert info.value.collection == C.name
    assert FakeReader(writer).count(ENTRY) == 1000
    # Without prune the listing is not guarded: the record popped after the
    # first listing is stale now, and nothing is deleted.
    result, _ = sync(src, writer, state, events, tmp_path)
    assert (result.stale, result.deleted) == (1, 0)


def test_a_changed_plan_entry_writes_every_record_again(src, state, events, tmp_path):
    writer = copied(src, state, events)
    warnings: list[str] = []
    warn = warnings.append
    result, _ = sync(src, writer, state, events, tmp_path, DROPPED, warn=warn)
    assert (result.copied, result.unchanged, result.plan_changed) == (1000, 0, True)
    assert all("n" not in o.properties for o in writer.objects.values())
    assert warnings == [PLAN_CHANGED]
    warned = [e for e in event_lines(tmp_path) if e["event"] == "plan_changed"]
    assert [(e["level"], e["ref"], e["message"]) for e in warned] == [
        ("warning", C.ref, PLAN_CHANGED)
    ]
    result, _ = sync(src, writer, state, events, tmp_path, DROPPED, warn=warn)
    assert (result.copied, result.unchanged, result.plan_changed) == (0, 1000, False)
    assert warnings == [PLAN_CHANGED]


def test_rejects_of_the_copy_are_written_by_the_sync(src, state, events, tmp_path):
    writer = copied(src, state, events, FakeWriter(reject_ids={"id-3"}))
    assert state.progress()[0].rejects == 1 and len(writer.objects) == 999
    writer.reject_ids = set()
    result, _ = sync(src, writer, state, events, tmp_path)
    assert (result.copied, result.unchanged, result.rejects) == (1, 999, 0)
    assert state.progress()[0].rejects == 0 and len(writer.objects) == 1000


def test_emits_sync_finished(src, state, events, tmp_path):
    writer = copied(src, state, events)
    del src.records[3]
    sync(src, writer, state, events, tmp_path, prune=True)
    last = event_lines(tmp_path)[-1]
    assert last["event"] == "sync_finished" and last["level"] == "info"
    keys = ("copied", "unchanged", "rejects", "stale", "deleted")
    assert {k: last[k] for k in keys} == {
        "copied": 0,
        "unchanged": 999,
        "rejects": 0,
        "stale": 1,
        "deleted": 1,
    }
