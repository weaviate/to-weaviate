import hashlib
import json
import math
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from to_weaviate.events import Events
from to_weaviate.plan import CollectionPlan
from to_weaviate.sources.base import Page, Partition, Record, Source, SourceCollection
from to_weaviate.state import CollectionProgress, State
from to_weaviate.target.writer import Reject, TargetObject, Writer, to_object

NOT_A_NUMBER = "embedding holds a value that is not a number"
SOURCE_CHANGED = "the source changed since the last run; re-reading from the start"
REJECTS_ARE_FINAL = (
    "Rejects are final for this directory. After fixing the cause, copy "
    "plan.yaml and inspect.json to a new --dir and run to-weaviate copy there; "
    "objects already in Weaviate are written again unchanged."
)


@dataclass(frozen=True)
class CopyResult:
    copied: int
    rejects: int
    # The start offset of a tail that a shift of the source offsets moved.
    reread_from: int | None = None


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


def _last_read(
    source: Source, collection: SourceCollection, wanted: list[str], below: int
) -> tuple[int, int] | None:
    """(index in `wanted`, offset) of the first of `wanted` found below `below`.

    `wanted` holds last ids, newest first. The ids are read backwards from
    offset `below` in pages of the source's page size, until `wanted[0]` is
    found or offset 0 is reached.
    """
    rank: dict[str, int] = {}
    for place, record_id in enumerate(wanted):
        rank.setdefault(record_id, place)
    best: tuple[int, int] | None = None
    stop = below
    while stop > 0 and (best is None or best[0] > 0):
        start = max(0, stop - source.page_size)
        ids = _ids_between(source, collection, start, stop)
        for offset, record_id in enumerate(ids, start):
            index = rank.get(record_id)
            if index is not None and (best is None or index < best[0]):
                best = (index, offset)
        stop = start
    return best


@dataclass(frozen=True)
class _TailPlace:
    start: int
    before: str | None  # the source id of the record at `start - 1`
    superseded: list[str]  # keys of the partitions whose records the tail reads


def _tail_place(
    source: Source,
    collection: SourceCollection,
    stored: list[Partition],
    last_ids: dict[str, str],
    covered: int,
    count: int,
) -> _TailPlace:
    """Where the next tail starts, found from the last ids of the partitions.

    A source orders its records by insertion: a deletion moves each later
    record one offset to the left, and an addition goes after the last one.
    So a stored last id is at or below the offset it was read at, the records
    after it were not read, and the records before it were. One id read at
    `covered - 1` shows that nothing shifted. Otherwise the last id of the
    newest partition is searched, then those of the partitions before it,
    newest first; the tail reads the partitions after the one found again.
    When none is found, the tail starts at 0 and reads every partition again.
    """
    if covered == 0:
        return _TailPlace(0, None, [])
    newest = last_ids.get(stored[-1].key)
    if newest is not None and source.ids_at(collection, covered - 1, 1) == [newest]:
        return _TailPlace(covered, newest, [])
    keys = [p.key for p in reversed(stored) if p.key in last_ids]
    wanted = [last_ids[key] for key in keys]
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
        self._lock = threading.Lock()
        self._copied = _progress(state, self._ref).copied
        self.total = collection.count

    def run(self, pending: list[tuple[Partition, str | None]], workers: int) -> None:
        """Copy every pending partition and raise the first worker error.

        After a worker raises, the other workers stop before their next page.
        """
        stop = threading.Event()

        def work(partition: Partition, cursor: str | None) -> None:
            try:
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
        rejects: list[Reject] = []
        converted: list[tuple[Record, TargetObject]] = []
        for record in page.records:
            result = _not_a_number(record) or to_object(record, self._entry)
            if isinstance(result, Reject):
                rejects.append(result)
            else:
                converted.append((record, result))
        digests = {record.id: content_hash(record) for record, _ in converted}
        if self._skip_unchanged and digests:
            same = self._state.unchanged(self._ref, list(digests.items()))
            converted = [(r, o) for r, o in converted if r.id not in same]
        write_rejects: list[Reject] = []
        if converted:
            write_rejects = self._writer.write(self._entry, [o for _, o in converted])
        rejects.extend(write_rejects)
        if rejects:
            self._state.add_rejects(self._ref, rejects)
        # A write reject names the source id, or the uuid when the plan has no
        # source_id property.
        rejected = {r.source_id for r in write_rejects}
        written = [
            record.id
            for record, obj in converted
            if record.id not in rejected and obj.uuid not in rejected
        ]
        if written:
            self._state.record_hashes(
                self._ref, [(record_id, digests[record_id]) for record_id in written]
            )
        self._state.advance(
            self._ref,
            partition.key,
            page.next_cursor,
            len(written),
            stop=end,
            last_id=page.records[-1].id if page.records else None,
        )
        with self._lock:
            self._copied += len(written)
            copied = self._copied
        self._events.progress(self._ref, copied, self.total)


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
) -> CopyResult:
    """Copy one collection and return its totals over all runs.

    Progress is stored in `state`, so a run after a failure resumes where the
    last one stopped. After the stored partitions are finished, every run
    reads the source count. The stored total is the highest count a run read.
    Records beyond the end of the stored partitions are copied by a new
    partition without a stop, a tail. A finished tail stores its end offset,
    so a later run adds the next tail. A finished tail without an end offset,
    from an older state, gets no later tail.

    `_tail_place` decides where the tail starts. When the source offsets
    shifted, the tail is added also without new records, so that the next
    run starts from its end, and `reread_from` holds its start. A tail from
    offset 0 writes a `source_changed` warning event and passes the same
    text to `warn`.
    """
    ref = collection.ref
    state.init_collection(
        ref, source.partitions(collection, parallelism), collection.count
    )
    events.emit("info", "collection_started", ref=ref, total=collection.count)
    copy = _CollectionCopy(
        source, writer, state, events, entry, collection, skip_unchanged
    )
    copy.run(state.pending(ref), parallelism)
    count = source.count(collection)
    state.raise_total(ref, count)
    stored = state.partitions(ref)
    covered = _covered(stored)
    reread_from = None
    if covered is not None:
        place = _tail_place(
            source, collection, stored, state.last_ids(ref), covered, count
        )
        shifted = place.start != covered
        if shifted:
            reread_from = place.start
        if shifted and place.start == 0:
            events.emit("warning", "source_changed", ref=ref, message=SOURCE_CHANGED)
            if warn is not None:
                warn(SOURCE_CHANGED)
        if shifted or count > covered:
            tail = _tail(stored, place.start)
            state.add_partition(ref, tail, place.before, place.superseded)
            # The superseded counts are 0 now; a new copy reads the count again.
            copy = _CollectionCopy(
                source, writer, state, events, entry, collection, skip_unchanged
            )
            copy.total = count
            copy.run(state.pending(ref), parallelism)
            # A tail reads to the last record, which can be past `count` now.
            state.raise_total(ref, source.count(collection))
    progress = _progress(state, ref)
    events.emit(
        "info",
        "collection_finished",
        ref=ref,
        copied=progress.copied,
        rejects=progress.rejects,
    )
    return CopyResult(progress.copied, progress.rejects, reread_from)
