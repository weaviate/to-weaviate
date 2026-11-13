import hashlib
import json
import math
import threading
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from to_weaviate.events import Events
from to_weaviate.plan import CollectionPlan
from to_weaviate.sources.base import (
    Page,
    Partition,
    Record,
    Source,
    SourceChanged,
    SourceCollection,
)
from to_weaviate.state import CollectionProgress, State
from to_weaviate.target.writer import Reject, TargetObject, Writer, to_object

NOT_A_NUMBER = "embedding holds a value that is not a number"
SOURCE_CHANGED = "the source changed since the last run; re-reading from the start"
REJECTS_HINT = (
    "A reject stays until its cause is fixed; then to-weaviate sync writes the "
    "record. Objects already in Weaviate are written only when they changed."
)


@dataclass(frozen=True)
class CopyResult:
    # Records in Weaviate, skipped unchanged ones included, and rejects, over
    # the runs since the last fresh one.
    copied: int
    rejects: int
    # The start offset of a tail that reads records of earlier runs again.
    reread_from: int | None = None
    # Records this run wrote: not the unchanged ones, not the rejects. A record
    # that changed again before a tail read it twice counts twice.
    written: int = 0


def plan_hash(entry: CollectionPlan) -> bytes:
    """Hash of the target part of a plan entry, which decides each object.

    Fields at their default and the reasons of dropped keys are left out, so
    a field that a later version adds with a default changes no hash. State
    stores these hashes, so the encoding must not change.
    """
    target = entry.target.model_dump(mode="json", exclude_defaults=True)
    for dropped in target.get("dropped", []):
        dropped.pop("reason", None)
    payload = json.dumps(
        target, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.blake2b(payload.encode("utf-8"), digest_size=16).digest()


def record_hash(plan: bytes, record: Record) -> bytes:
    """The hash state stores for `record` written under the plan hash `plan`.

    A plan entry edited since a record was written gives another hash, so a
    sync writes the record again.
    """
    digest = hashlib.blake2b(plan + content_hash(record), digest_size=16)
    return digest.digest()


def content_hash(record: Record) -> bytes:
    """Hash of the record's id, document, uri, metadata and vector.

    State stores these hashes, so the encoding must not change.
    """
    payload = json.dumps(
        [record.id, record.document, record.uri, record.metadata, record.vector],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.blake2b(payload.encode("utf-8"), digest_size=16).digest()


def _not_a_number(record: Record) -> Reject | None:
    """The reject of a record whose vector holds a value that is not a number.

    None, NaN and inf all count as not a number. A source decodes None as NaN.
    The check is here and not in `to_object`, so that `verify` still compares
    such a record with its target object.
    """
    if record.vector is not None and not all(map(math.isfinite, record.vector)):
        return Reject(record.id, NOT_A_NUMBER)
    return None


def _progress(state: State, ref: str) -> CollectionProgress:
    return next(p for p in state.progress() if p.ref == ref)


def _covered(stored: list[Partition]) -> int | None:
    """End offset of the newest stored partition, or None when a tail has no end.

    Stored partitions can come from an earlier run with a smaller count, so
    this end can be lower than the current `collection.count`. A tail stores
    its end offset when it finishes. A finished tail without one comes from a
    state written before tails stored it. After the source offsets shifted,
    an older partition can end above the newest one.
    """
    if any(p.stop is None for p in stored):
        return None
    return int(stored[-1].stop or 0) if stored else 0


def _tail(stored: list[Partition], start: int) -> Partition:
    """A new partition from `start` without a stop.

    A tail that read no records ends at its start, and a tail after a shift
    can start where an older one started, so `{start}-` can be taken. The new
    tail then gets a key with a number from the partition count up.
    """
    keys = {p.key for p in stored}
    key, number = f"{start}-", len(stored)
    while key in keys:
        key, number = f"{start}-{number}", number + 1
    return Partition(key, str(start), None)


def _ids_between(
    source: Source, collection: SourceCollection, start: int, stop: int
) -> list[str]:
    """The ids at the offsets from `start` to `stop`; fewer when the source ends.

    A server can return fewer rows than the limit before its last page.
    """
    ids: list[str] = []
    while start + len(ids) < stop:
        more = source.ids_at(collection, start + len(ids), stop - start - len(ids))
        if not more:
            break
        ids.extend(more)
    return ids[: stop - start]


@dataclass(frozen=True)
class _Mark:
    """The source id of the last record a partition read, and of the one before."""

    last_id: str
    previous_id: str | None


def _ids_backwards(
    source: Source, collection: SourceCollection, below: int
) -> Iterator[tuple[int, str]]:
    """(offset, id) of each record below offset `below`, highest offset first.

    The ids are read without a payload, one page of the source's page size
    at a time, when the caller asks for them.
    """
    stop = below
    while stop > 0:
        start = max(0, stop - source.page_size)
        ids = _ids_between(source, collection, start, stop)
        for offset in reversed(range(start, start + len(ids))):
            yield offset, ids[offset - start]
        stop = start


def _last_read(
    source: Source, collection: SourceCollection, wanted: list[_Mark], below: int
) -> tuple[int, int] | None:
    """(index in `wanted`, offset) of the first of `wanted` found below `below`.

    `wanted` holds marks, newest first. A last id counts as found only when the
    id at the offset below it is the mark's previous id, or when the mark has
    none. The ids are read backwards from offset `below` until `wanted[0]` is
    found or offset 0 is reached.
    """
    rank: dict[str, int] = {}
    for place, mark in enumerate(wanted):
        rank.setdefault(mark.last_id, place)
    best: tuple[int, int] | None = None
    # A last id found at an offset, whose previous id is checked next.
    waiting: tuple[int, int] | None = None
    for offset, record_id in _ids_backwards(source, collection, below):
        if waiting is not None:
            checked, found = waiting
            waiting = None
            if offset == found - 1 and record_id == wanted[checked].previous_id:
                best = (checked, found)
                if checked == 0:
                    break
        index = rank.get(record_id)
        if index is not None and (best is None or index < best[0]):
            if wanted[index].previous_id is not None:
                waiting = (index, offset)
            else:
                best = (index, offset)
                if index == 0:
                    break
    return best


def _unshifted(
    source: Source, collection: SourceCollection, mark: _Mark, covered: int
) -> bool:
    """Whether the ids that end at offset `covered` are the mark's. One read."""
    if mark.previous_id is None or covered < 2:
        return source.ids_at(collection, covered - 1, 1) == [mark.last_id]
    ids = source.ids_at(collection, covered - 2, 2)
    return ids == [mark.previous_id, mark.last_id]


@dataclass(frozen=True)
class _TailPlace:
    start: int
    mark: _Mark | None  # the records at `start - 1` and `start - 2`
    superseded: list[str]  # keys of the partitions whose records the tail reads


def _tail_place(
    source: Source,
    collection: SourceCollection,
    stored: list[Partition],
    marks: dict[str, _Mark],
    covered: int,
    count: int,
) -> _TailPlace:
    """Where the next tail starts, found from the marks of the partitions.

    A source orders its records by insertion: a deletion moves each later
    record one offset to the left, and an addition goes after the last one.
    So a stored last id is at or below the offset it was read at, the records
    after it were not read, and the records before it were. A deleted record
    that is added again goes after the last one too, so a last id counts only
    with its previous id below it. One read of the ids that end at `covered`
    shows that nothing shifted. Otherwise the mark of the newest partition is
    searched, then those of the partitions before it, newest first; the tail
    reads the partitions after the one found again. When none is found, the
    tail starts at 0 and reads every partition again.
    """
    if covered == 0:
        return _TailPlace(0, None, [])
    newest = marks.get(stored[-1].key)
    if newest is not None and _unshifted(source, collection, newest, covered):
        return _TailPlace(covered, newest, [])
    keys = [p.key for p in reversed(stored) if p.key in marks]
    wanted = [marks[key] for key in keys]
    found = (
        _last_read(source, collection, wanted, min(covered, count)) if wanted else None
    )
    if found is None:
        return _TailPlace(0, None, [p.key for p in stored])
    index, offset = found
    position = next(i for i, p in enumerate(stored) if p.key == keys[index])
    return _TailPlace(
        offset + 1, wanted[index], [p.key for p in stored[position + 1 :]]
    )


def _marks(state: State, ref: str) -> dict[str, _Mark]:
    previous = state.previous_ids(ref)
    return {
        key: _Mark(last_id, previous.get(key))
        for key, last_id in state.last_ids(ref).items()
    }


@dataclass(frozen=True)
class _Resume:
    """The stored partitions of a run that continues an earlier one."""

    stored: list[Partition]
    pending: frozenset[str]  # keys of the partitions that are not finished
    marks: dict[str, _Mark]
    warn: Callable[[str], None] | None


def _resume(
    state: State,
    ref: str,
    pending: list[tuple[Partition, str | None]],
    warn: Callable[[str], None] | None,
) -> _Resume | None:
    """None when no stored partition was read: there is no place to check."""
    stored = state.partitions(ref)
    started = any(cursor is not None for _, cursor in pending)
    if not pending or (len(stored) == len(pending) and not started):
        return None
    return _Resume(
        stored, frozenset(part.key for part, _ in pending), _marks(state, ref), warn
    )


@dataclass(frozen=True)
class _ResumePlace:
    cursor: str | None
    # Keys of the partitions whose records are read again. None when the
    # partition reads no record that an earlier run read.
    reread: list[str] | None


def _resume_place(
    source: Source,
    collection: SourceCollection,
    resume: _Resume,
    partition: Partition,
    cursor: str | None,
) -> _ResumePlace:
    """Where a pending partition reads next, checked against the source.

    The order rules of `_tail_place` apply. A partition that read a page
    stores the mark of the record below its cursor. A partition that read
    none has no mark: the records below its start belong to the partition
    before it. When that one is pending, it reads up to its stop, so no
    record is left out and nothing is read here. When it is finished, its
    mark is the record below this start. A tail stores the mark of the
    record below its start when it is added.

    One read of the ids below the position shows that nothing shifted. A
    mark found at a lower offset moves the cursor after it. When the mark is
    gone, the marks of the finished partitions before this one are searched,
    nearest first, and the partition reads from after the one found. When
    none is found, it reads from the start of the first of those finished
    partitions, which is the stop of a pending partition or offset 0. A tail
    whose mark is gone reads from offset 0: the stored partitions do not
    tell which of them the tail reads again.

    A partition without a mark to check, from a state written before marks
    were stored, keeps its cursor.
    """
    same = _ResumePlace(cursor, None)
    position = int(cursor if cursor is not None else partition.start or 0)
    index = next(i for i, p in enumerate(resume.stored) if p.key == partition.key)
    before: list[Partition] = []
    if partition.stop is not None:
        for earlier in reversed(resume.stored[:index]):
            if earlier.key in resume.pending:
                break
            before.append(earlier)
    own = resume.marks.get(partition.key)
    keys = [p.key for p in before if p.key in resume.marks]
    if own is not None:
        keys.insert(0, partition.key)
    elif cursor is not None or not before or before[0].key not in resume.marks:
        return same
    if position == 0:
        return same
    wanted = [resume.marks[key] for key in keys]
    if _unshifted(source, collection, wanted[0], position):
        return same
    found = _last_read(source, collection, wanted, position)
    if found is not None and found[0] == 0:
        return _ResumePlace(str(found[1] + 1), None)
    if found is None:
        if partition.stop is None:
            return _ResumePlace("0", [p.key for p in resume.stored])
        first = before[-1] if before else partition
        return _ResumePlace(
            str(int(first.start or 0)), [partition.key, *(p.key for p in before)]
        )
    reached = next(i for i, p in enumerate(before) if p.key == keys[found[0]])
    return _ResumePlace(
        str(found[1] + 1), [partition.key, *(p.key for p in before[:reached])]
    )


def _tail_end(partition: Partition, cursor: str | None, page: Page) -> str | None:
    """The end offset of a tail whose last page is `page`; None for other pages.

    Starts and cursors are offsets, as `_covered` assumes.
    """
    if partition.stop is not None or page.next_cursor is not None:
        return None
    offset = int(cursor if cursor is not None else partition.start or 0)
    return str(offset + len(page.records))


class _CollectionCopy:
    """Copies the pending partitions of one collection.

    A page is advanced only after its write returned, so a run that stops at
    any point resumes at a cursor whose pages are all written.
    """

    def __init__(
        self,
        source: Source,
        writer: Writer,
        state: State,
        events: Events,
        entry: CollectionPlan,
        collection: SourceCollection,
        skip_unchanged: bool,
    ) -> None:
        self._source = source
        self._writer = writer
        self._state = state
        self._events = events
        self._entry = entry
        self._collection = collection
        self._ref = collection.ref
        self._skip_unchanged = skip_unchanged
        self._with_vectors = any(v.mode == "keep" for v in entry.target.vectors)
        self._plan = plan_hash(entry)
        self._lock = threading.Lock()
        self._copied = _progress(state, self._ref).copied
        self.total = collection.count
        self.written = 0
        # The lowest offset from which a partition reads records again.
        self.reread_from: int | None = None

    def run(
        self,
        pending: list[tuple[Partition, str | None]],
        workers: int,
        resume: _Resume | None = None,
    ) -> None:
        """Copy every pending partition and raise the first worker error.

        After a worker raises, the other workers stop before their next page.
        With `resume`, each worker first checks the place of its partition
        against the source.
        """
        stop = threading.Event()

        def work(partition: Partition, cursor: str | None) -> None:
            try:
                if resume is not None:
                    cursor = self._resume_at(resume, partition, cursor)
                self._copy_partition(partition, cursor, stop)
            except BaseException:
                stop.set()
                raise

        first: BaseException | None = None
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(work, part, cursor) for part, cursor in pending]
            try:
                for future in as_completed(futures):
                    error = future.exception()
                    if error is not None and first is None:
                        first = error
            finally:
                # Leaving the wait early, for example on Ctrl-C, stops the workers.
                stop.set()
        if first is not None:
            raise first

    def _resume_at(
        self, resume: _Resume, partition: Partition, cursor: str | None
    ) -> str | None:
        """The cursor `partition` reads from, after `_resume_place` checked it.

        The moved cursor is not stored: the first page stores the next one,
        and a run that stops before it checks the same marks again.
        """
        place = _resume_place(self._source, self._collection, resume, partition, cursor)
        if place.reread is None:
            return place.cursor
        forgotten = self._state.forget_copied(self._ref, place.reread)
        start = int(place.cursor or 0)
        with self._lock:
            self._copied -= forgotten
            if self.reread_from is None or start < self.reread_from:
                self.reread_from = start
        if start == 0:
            self._events.emit(
                "warning", "source_changed", ref=self._ref, message=SOURCE_CHANGED
            )
            if resume.warn is not None:
                resume.warn(SOURCE_CHANGED)
        return place.cursor

    def _copy_partition(
        self, partition: Partition, cursor: str | None, stop: threading.Event
    ) -> None:
        while not stop.is_set():
            page = self._source.read(
                self._collection, partition, cursor, self._with_vectors
            )
            self._copy_page(partition, page, _tail_end(partition, cursor, page))
            if page.next_cursor is None:
                return
            cursor = page.next_cursor

    def _copy_page(self, partition: Partition, page: Page, end: str | None) -> None:
        written, same = write_records(
            self._writer,
            self._state,
            self._entry,
            self._ref,
            self._plan,
            page.records,
            self._skip_unchanged,
        )
        # An unchanged record counts as copied: the object is in Weaviate.
        self._state.advance(
            self._ref,
            partition.key,
            page.next_cursor,
            written + same,
            stop=end,
            last_id=page.records[-1].id if page.records else None,
            previous_id=page.records[-2].id if len(page.records) > 1 else None,
        )
        with self._lock:
            self._copied += written + same
            self.written += written
            copied = self._copied
        self._events.progress(self._ref, copied, self.total)


def write_records(
    writer: Writer,
    state: State,
    entry: CollectionPlan,
    ref: str,
    plan: bytes,
    records: list[Record],
    skip_unchanged: bool = False,
) -> tuple[int, int]:
    """Write `records` and store their hashes and rejects.

    `plan` is the `plan_hash` of `entry`. With `skip_unchanged`, a record
    whose stored hash matches is not written. Returns the number of records
    written and the number skipped.

    Before the write, the stored hashes of the records it sends match nothing
    any more: Weaviate can store a write and still report it as failed, and a
    record changed back to its old content must then be written again.
    """
    rejects: list[Reject] = []
    converted: list[tuple[Record, TargetObject]] = []
    for record in records:
        result = _not_a_number(record) or to_object(record, entry)
        if isinstance(result, Reject):
            rejects.append(result)
        else:
            converted.append((record, result))
    digests = {record.id: record_hash(plan, record) for record, _ in converted}
    same: set[str] = set()
    if skip_unchanged and digests:
        same = state.unchanged(ref, list(digests.items()))
        converted = [(r, o) for r, o in converted if r.id not in same]
    write_rejects: list[Reject] = []
    if converted:
        state.invalidate_hashes(ref, [r.id for r, _ in converted])
        write_rejects = writer.write(entry, [o for _, o in converted])
    rejects.extend(write_rejects)
    if rejects:
        state.add_rejects(ref, rejects)
    # A write reject names the source id, or the uuid when the plan has no
    # source_id property.
    rejected = {r.source_id for r in write_rejects}
    written = [
        record.id
        for record, obj in converted
        if record.id not in rejected and obj.uuid not in rejected
    ]
    if written:
        state.record_hashes(
            ref, [(record_id, digests[record_id]) for record_id in written]
        )
    return len(written), len(same)


def raise_if_compacted(
    source: Source, collection: SourceCollection, before: dict[str, int] | None
) -> None:
    """Raise SourceChanged when the compaction state differs from `before`.

    `before` None comes from a source without compaction: nothing is read.
    """
    if before is None:
        return
    after = source.compaction_state(collection)
    if after != before:
        raise SourceChanged(collection.name, before, after)


def copy_collection(
    source: Source,
    writer: Writer,
    state: State,
    events: Events,
    entry: CollectionPlan,
    collection: SourceCollection,
    parallelism: int,
    skip_unchanged: bool = False,
    warn: Callable[[str], None] | None = None,
    waiting: Callable[[int, int], None] | None = None,
    fresh: bool = False,
) -> CopyResult:
    """Copy one collection and return its totals.

    Progress is stored in `state`, so a run after a failure resumes where the
    last one stopped. Before its first page, each pending partition of such a
    run checks its place with `_resume_place`, in its own worker, so that
    records deleted from the source between the runs make it skip no record.
    A partition that reads records of earlier runs again sets `reread_from`,
    and one that reads from offset 0 warns as a tail from offset 0 does. With
    `fresh`, the stored read positions and rejects of
    the collection go after `source.wait_compacted` returned, so the run reads
    every record again; a run that stops in the wait keeps them. A fresh run
    stores the count it read after its pass as the total. Otherwise the
    stored total is the highest count a run read. A run stores the plan hash
    of `entry` when none is stored; `sync` compares and replaces it.

    After the stored partitions are finished, every run reads the source
    count. Records beyond the end of the stored partitions are copied by a new
    partition without a stop, a tail. A finished tail stores its end offset,
    so a later run adds the next tail. A finished tail without an end offset,
    from an older state, gets no later tail.

    `_tail_place` decides where the tail starts. When the source offsets
    shifted, the tail is added also without new records, so that the next
    run starts from its end. When the tail reads records of earlier runs
    again, `reread_from` holds its start. A tail from offset 0 writes a
    `source_changed` warning event and passes the same text to `warn`.

    Before the first page, `source.wait_compacted` waits until the source
    has no write left to compact. Each report of the wait writes a
    `compaction_wait` event and goes to `waiting`. After the last read, a
    changed compaction state raises SourceChanged. Before it does, a tail
    from offset 0 supersedes every stored partition, so the next run reads
    the collection again. Pages read before the change used the old order,
    and the marks of `_tail_place` assume insertion order, so they cannot
    find the place of the last read. A record read again is written again
    unchanged. The stop writes no event: no error of the tool writes one,
    and the error document names the collection.
    """
    ref = collection.ref
    before = source.wait_compacted(
        collection, compaction_reporter(events, ref, waiting)
    )
    state.store_plan_hash(ref, plan_hash(entry))
    state.init_collection(
        ref, source.partitions(collection, parallelism), collection.count, fresh=fresh
    )
    events.emit("info", "collection_started", ref=ref, total=collection.count)
    copy = _CollectionCopy(
        source, writer, state, events, entry, collection, skip_unchanged
    )
    pending = state.pending(ref)
    copy.run(pending, parallelism, _resume(state, ref, pending, warn))
    written = copy.written
    count = source.count(collection)
    if fresh:
        state.set_total(ref, count)
    else:
        state.raise_total(ref, count)
    stored = state.partitions(ref)
    covered = _covered(stored)
    reread_from = copy.reread_from
    if covered is not None:
        marks = _marks(state, ref)
        place = _tail_place(source, collection, stored, marks, covered, count)
        shifted = place.start != covered
        if place.superseded and count > place.start:
            reread_from = min(place.start, reread_from or place.start)
        if shifted and place.start == 0:
            events.emit("warning", "source_changed", ref=ref, message=SOURCE_CHANGED)
            if warn is not None:
                warn(SOURCE_CHANGED)
        if shifted or count > covered:
            tail = _tail(stored, place.start)
            mark = place.mark
            state.add_partition(
                ref,
                tail,
                mark.last_id if mark else None,
                place.superseded,
                previous_id=mark.previous_id if mark else None,
            )
            # The superseded counts are 0 now; a new copy reads the count again.
            copy = _CollectionCopy(
                source, writer, state, events, entry, collection, skip_unchanged
            )
            copy.total = count
            copy.run(state.pending(ref), parallelism)
            written += copy.written
            # A tail reads to the last record, which can be past `count` now.
            state.raise_total(ref, source.count(collection))
    try:
        raise_if_compacted(source, collection, before)
    except SourceChanged:
        stored = state.partitions(ref)
        state.add_partition(ref, _tail(stored, 0), None, [p.key for p in stored])
        raise
    progress = _progress(state, ref)
    events.emit(
        "info",
        "collection_finished",
        ref=ref,
        copied=progress.copied,
        rejects=progress.rejects,
    )
    return CopyResult(progress.copied, progress.rejects, reread_from, written)


def compaction_reporter(
    events: Events, ref: str, waiting: Callable[[int, int], None] | None
) -> Callable[[int, int], None]:
    """The `waiting` callback of `Source.wait_compacted` for collection `ref`.

    Each report writes a `compaction_wait` event and goes to `waiting`.
    """

    def report(unindexed: int, total: int) -> None:
        events.emit(
            "info", "compaction_wait", ref=ref, unindexed=unindexed, total=total
        )
        if waiting is not None:
            waiting(unindexed, total)

    return report
