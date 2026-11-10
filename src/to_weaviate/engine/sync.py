from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from to_weaviate.engine.copy import copy_collection, plan_hash, raise_if_compacted
from to_weaviate.engine.verify import (
    id_tables,
    missing_source_ids,
    source_listed,
    stale_objects,
    target_total,
)
from to_weaviate.events import Events
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


@dataclass(frozen=True)
class SyncResult:
    copied: int  # records written in this pass: new or changed
    unchanged: int
    rejects: int
    stale: int  # target objects with a source_id and no source record
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
    whose stored hash matches. A plan entry that differs from the one the
    hashes were written under drops them first, so every record is written
    again. Then the source ids are listed against the target objects. The
    hashes of the records without a target object go, so the next sync writes
    them. With `prune`, the stale objects are deleted, after their hashes,
    unless they are over the limit and `force` is not set; the pass must have
    read a source that no one wrote to, so a count that changed around the
    listing raises SourceWriting before anything is deleted.
    """
    ref = collection.ref
    plan_changed = state.set_plan_hash(ref, plan_hash(entry))
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

    def on_wait(unindexed: int, total: int) -> None:
        events.emit(
            "info", "compaction_wait", ref=ref, unindexed=unindexed, total=total
        )
        if waiting is not None:
            waiting(unindexed, total)

    before = source.count(collection)
    compaction = source.wait_compacted(collection, on_wait)
    with id_tables(source, reader, entry, collection, workdir) as db:
        listed = source_listed(db)
        total = target_total(db)
        stale = stale_objects(db)
        after = source.count(collection)
        raise_if_compacted(source, collection, compaction)
        for missing in missing_source_ids(db):
            state.delete_hashes(ref, missing)
    if prune and not before == listed == after:
        raise SourceWriting(collection.name, before, listed, after)
    deleted = 0
    refused = False
    if prune and stale:
        refused = over_prune_limit(len(stale), total) and not force
        if not refused:
            # The hashes go first: after a deletion that fails, a record that
            # comes back must be written again.
            state.delete_hashes(ref, [source_id for _, source_id in stale])
            deleted = deleter.delete(entry, [uuid for uuid, _ in stale])
    copied = result.written
    events.emit(
        "info",
        "sync_finished",
        ref=ref,
        copied=copied,
        unchanged=result.unchanged,
        rejects=result.rejects,
        stale=len(stale),
        deleted=deleted,
    )
    return SyncResult(
        copied,
        result.unchanged,
        result.rejects,
        len(stale),
        deleted,
        plan_changed,
        refused,
    )
