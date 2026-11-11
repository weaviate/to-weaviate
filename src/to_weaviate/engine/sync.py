from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from to_weaviate.engine.copy import (
    compaction_reporter,
    copy_collection,
    plan_hash,
    raise_if_compacted,
    write_records,
)
from to_weaviate.engine.verify import (
    id_tables,
    missing_source_ids,
    owned_total,
    source_listed,
    stale_batches,
    stale_count,
)
from to_weaviate.events import Events
from to_weaviate.ids import to_uuid
from to_weaviate.plan import CollectionPlan
from to_weaviate.sources.base import Source, SourceCollection
from to_weaviate.state import State
from to_weaviate.target.deleter import Deleter
from to_weaviate.target.reader import TargetReader
from to_weaviate.target.writer import Writer

PRUNE_LIMIT_PERCENT = 10
PLAN_CHANGED = (
    "the plan entry changed since the last copy; every record is written again"
)
SCRATCH_PREFIX = "sync-"


@dataclass(frozen=True)
class SyncResult:
    copied: int  # records written in this run: new, changed or missing
    unchanged: int
    rejects: int
    # Target objects with a source_id and no source record. After a prune,
    # the objects whose record `Source.get` did not find.
    stale: int
    deleted: int
    plan_changed: bool
    # --prune without --force, and the stale objects are over the limit.
    prune_refused: bool


class SourceWriting(Exception):
    """The source count changed while the ids of `collection` were listed."""

    def __init__(self, collection: str, before: int, listed: int, after: int) -> None:
        super().__init__(
            f"{collection} changed while sync listed its ids: {before} records "
            f"before, {listed} listed, {after} after"
        )
        self.collection = collection
        self.before = before
        self.listed = listed
        self.after = after


def over_prune_limit(stale: int, total: int) -> bool:
    """Whether `stale` objects are more than PRUNE_LIMIT_PERCENT of `total`."""
    return stale * 100 > total * PRUNE_LIMIT_PERCENT


def sync_collection(
    source: Source,
    writer: Writer,
    reader: TargetReader,
    deleter: Deleter,
    state: State,
    events: Events,
    entry: CollectionPlan,
    collection: SourceCollection,
    parallelism: int,
    prune: bool,
    force: bool,
    workdir: Path,
    warn: Callable[[str], None] | None = None,
    waiting: Callable[[int, int], None] | None = None,
) -> SyncResult:
    """Read the whole source again and write the new and changed records.

    `copy_collection` starts fresh: it reads every record and skips those
    whose stored hash matches. Each stored hash holds the plan hash of the
    entry it was written under, so after an edit of the entry none matches
    and every record is written again; this run warns about the edit.

    Then the source ids are listed against the target objects. A record
    without a target object, other than a reject of this run, is read with
    `Source.get` and written. With `prune`, a count that changed around the
    listing raises SourceWriting before anything is deleted. The stale
    objects are deleted unless they are more than PRUNE_LIMIT_PERCENT of the
    objects with a source_id and `force` is not set. Before a batch is
    deleted, `Source.get` reads its source ids again: a record the source
    holds was missed by the listing, and its object stays. The hashes of the
    deleted objects go before the objects, so a record that comes back after
    a failed deletion is written again.
    """
    ref = collection.ref
    plan = plan_hash(entry)
    plan_changed = state.set_plan_hash(ref, plan)
    if plan_changed:
        events.emit("warning", "plan_changed", ref=ref, message=PLAN_CHANGED)
        if warn is not None:
            warn(PLAN_CHANGED)
    result = copy_collection(
        source,
        writer,
        state,
        events,
        entry,
        collection,
        parallelism,
        skip_unchanged=True,
        warn=warn,
        waiting=waiting,
        fresh=True,
    )
    with_vectors = any(v.mode == "keep" for v in entry.target.vectors)
    before = source.count(collection)
    compaction = source.wait_compacted(
        collection, compaction_reporter(events, ref, waiting)
    )
    forgotten = restored = restore_rejects = deleted = 0
    refused = False
    with id_tables(source, reader, entry, collection, workdir, SCRATCH_PREFIX) as db:
        listed = source_listed(db)
        owned = owned_total(db)
        stale = stale_count(db)
        after = source.count(collection)
        raise_if_compacted(source, collection, compaction)
        if prune and not before == listed == after:
            raise SourceWriting(collection.name, before, listed, after)
        for batch in missing_source_ids(db):
            uuids = {i: to_uuid(i) for i in batch}
            rejected = state.rejected_among(ref, [*batch, *uuids.values()])
            ids = [i for i in batch if i not in rejected and uuids[i] not in rejected]
            if not ids:
                continue
            # The pass counted a record with a stored hash as unchanged.
            forgotten += state.delete_hashes(ref, ids)
            records = source.get(collection, ids, with_vectors)
            written, _ = write_records(writer, state, entry, ref, plan, records)
            restored += written
            restore_rejects += len(records) - written
        if prune and stale:
            refused = over_prune_limit(stale, owned) and not force
            if not refused:
                stale = 0
                for pairs in stale_batches(db):
                    live = source.get(collection, [i for _, i in pairs], False)
                    held = {record.id for record in live}
                    gone = [(uuid, i) for uuid, i in pairs if i not in held]
                    if gone:
                        stale += len(gone)
                        state.delete_hashes(ref, [i for _, i in gone])
                        deleted += deleter.delete(entry, [uuid for uuid, _ in gone])
    copied = result.written + restored
    unchanged = max(result.copied - result.written - forgotten, 0)
    rejects = result.rejects + restore_rejects
    events.emit(
        "info",
        "sync_finished",
        ref=ref,
        copied=copied,
        unchanged=unchanged,
        rejects=rejects,
        stale=stale,
        deleted=deleted,
    )
    return SyncResult(
        copied=copied,
        unchanged=unchanged,
        rejects=rejects,
        stale=stale,
        deleted=deleted,
        plan_changed=plan_changed,
        prune_refused=refused,
    )
