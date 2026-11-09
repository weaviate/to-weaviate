import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from to_weaviate.events import Events
from to_weaviate.plan import CollectionPlan
from to_weaviate.sources.base import Page, Partition, Record, Source, SourceCollection
from to_weaviate.state import CollectionProgress, State
from to_weaviate.target.writer import Reject, TargetObject, Writer, to_object


@dataclass(frozen=True)
class CopyResult:
    copied: int
    rejects: int


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


def _progress(state: State, ref: str) -> CollectionProgress:
    return next(p for p in state.progress() if p.ref == ref)


def _covered(stored: list[Partition]) -> int | None:
    """End offset of the stored partitions, or None when a tail exists.

    Stored partitions can come from an earlier run with a smaller count, so
    this end can be lower than the current `collection.count`.
    """
    if any(p.stop is None for p in stored):
        return None
    return max((int(p.stop) for p in stored if p.stop is not None), default=0)


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
            self._copy_page(partition, page)
            if page.next_cursor is None:
                return
            cursor = page.next_cursor

    def _copy_page(self, partition: Partition, page: Page) -> None:
        rejects: list[Reject] = []
        converted: list[tuple[Record, TargetObject]] = []
        for record in page.records:
            result = to_object(record, self._entry)
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
        self._state.advance(self._ref, partition.key, page.next_cursor, len(written))
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
) -> CopyResult:
    """Copy one collection and return its totals over all runs.

    Progress is stored in `state`, so a run after a failure resumes where the
    last one stopped. Records beyond the end of the stored partitions are
    copied by one extra partition without a stop. A collection gets at most
    one such partition.
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
    covered = _covered(state.partitions(ref))
    if covered is not None:
        count = source.count(collection)
        if count > covered:
            state.add_partition(ref, Partition(f"{covered}-", str(covered), None))
            copy.total = count
            copy.run(state.pending(ref), parallelism)
    progress = _progress(state, ref)
    events.emit(
        "info",
        "collection_finished",
        ref=ref,
        copied=progress.copied,
        rejects=progress.rejects,
    )
    return CopyResult(progress.copied, progress.rejects)
