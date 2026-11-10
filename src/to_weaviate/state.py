import json
import sqlite3
import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from to_weaviate.sources.base import Partition
from to_weaviate.target.writer import Reject

BATCH_SIZE = 500

_SCHEMA = """
CREATE TABLE IF NOT EXISTS collections (
    ref TEXT PRIMARY KEY,
    total INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS partitions (
    ref TEXT NOT NULL,
    key TEXT NOT NULL,
    start TEXT,
    stop TEXT,
    cursor TEXT,
    finished INTEGER NOT NULL DEFAULT 0,
    copied INTEGER NOT NULL DEFAULT 0,
    last_id TEXT,
    previous_id TEXT,
    PRIMARY KEY (ref, key)
);
CREATE TABLE IF NOT EXISTS rejects (
    ref TEXT NOT NULL,
    source_id TEXT NOT NULL,
    reason TEXT NOT NULL,
    PRIMARY KEY (ref, source_id)
);
CREATE TABLE IF NOT EXISTS hashes (
    ref TEXT NOT NULL,
    source_id TEXT NOT NULL,
    hash BLOB NOT NULL,
    PRIMARY KEY (ref, source_id)
);
CREATE TABLE IF NOT EXISTS targets (
    ref TEXT PRIMARY KEY,
    host TEXT NOT NULL,
    name TEXT NOT NULL,
    tenant TEXT
);
CREATE TABLE IF NOT EXISTS quantizer_updates (
    ref TEXT PRIMARY KEY
);
CREATE TABLE IF NOT EXISTS plans (
    ref TEXT PRIMARY KEY,
    hash BLOB NOT NULL
);
"""


class StateUnreadable(Exception):
    """The file at `path` is not a state that sqlite can read."""

    def __init__(self, path: Path) -> None:
        super().__init__(f"{path.name} in {path.parent} cannot be read.")
        self.path = path


class StateUnwritable(Exception):
    """The state at `path` cannot be written.

    `reason` says why: the disk is full, the file is read-only, or the
    directory does not allow it.
    """

    def __init__(self, path: Path, reason: str) -> None:
        super().__init__(f"{path.name} in {path.parent} cannot be written: {reason}.")
        self.path = path
        self.reason = reason


# Python 3.10 gives no sqlite error code, only its message.
UNWRITABLE_REASONS = (
    ("database or disk is full", "the disk is full"),
    ("attempt to write a readonly database", "the file is read-only"),
    ("unable to open database file", "the directory does not allow it"),
)


def unwritable_reason(exc: sqlite3.Error) -> str | None:
    """Why sqlite could not write, or None when `exc` says something else."""
    text = str(exc)
    for part, reason in UNWRITABLE_REASONS:
        if part in text:
            return reason
    return None


class StateInUse(Exception):
    """Another process holds a lock on the state at `path`."""

    def __init__(self, path: Path) -> None:
        super().__init__(f"Another to-weaviate process is using {path.parent}.")
        self.path = path


@dataclass(frozen=True)
class CollectionProgress:
    ref: str
    total: int
    copied: int
    rejects: int
    done: bool


@dataclass(frozen=True)
class Target:
    """The place a copy writes to.

    `host` comes from `settings.Endpoint.normalized`. `name` is the collection
    name that Weaviate stores.
    """

    host: str
    name: str
    tenant: str | None


class State:
    """Copy progress in sqlite. Every method commits before it returns.

    Collections and partitions are listed in insertion (rowid) order. A file
    written before the `targets`, `quantizer_updates` or `plans` table existed
    gets the table when it is opened, with no row for any ref. A file written before
    the `last_id` or the `previous_id` column existed gets the column, NULL in
    every row. A file that sqlite cannot read raises StateUnreadable when it is
    opened, and a file that another process keeps locked raises StateInUse.
    """

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._path = path
        self._rejects_path = path.parent / "rejects.jsonl"
        self._lock = threading.Lock()
        try:
            self._db = sqlite3.connect(path, check_same_thread=False)
        except sqlite3.OperationalError as exc:
            reason = unwritable_reason(exc)
            if reason is None:
                raise
            raise StateUnwritable(path, reason) from exc
        try:
            with self._transaction():
                self._db.executescript(_SCHEMA)
                columns = {
                    row[1] for row in self._db.execute("PRAGMA table_info(partitions)")
                }
                for column in ("last_id", "previous_id"):
                    if column not in columns:
                        self._db.execute(
                            f"ALTER TABLE partitions ADD COLUMN {column} TEXT"
                        )
        except (sqlite3.DatabaseError, StateUnwritable) as exc:
            self._db.close()
            if isinstance(exc, StateUnwritable):
                raise
            if isinstance(exc, sqlite3.OperationalError) and "locked" in str(exc):
                raise StateInUse(path) from exc
            raise StateUnreadable(path) from exc

    @contextmanager
    def _transaction(self) -> Iterator[None]:
        """The lock and one transaction.

        A write that the disk, the file or the directory refuses raises
        StateUnwritable instead of the sqlite error.
        """
        try:
            with self._lock, self._db:
                yield
        except sqlite3.OperationalError as exc:
            reason = unwritable_reason(exc)
            if reason is None:
                raise
            raise StateUnwritable(self._path, reason) from exc

    def close(self) -> None:
        with self._lock:
            self._db.close()

    def init_collection(
        self, ref: str, partitions: list[Partition], total: int, fresh: bool = False
    ) -> None:
        """Store `ref` with `total`, and `partitions` when it has none.

        A stored total that is higher stays. With `fresh`, the stored
        partitions and rejects of `ref` go first, in the same transaction.
        """
        with self._transaction():
            if fresh:
                self._db.execute("DELETE FROM partitions WHERE ref = ?", (ref,))
                self._db.execute("DELETE FROM rejects WHERE ref = ?", (ref,))
            self._db.execute(
                "INSERT INTO collections (ref, total) VALUES (?, ?) "
                "ON CONFLICT(ref) DO UPDATE SET total = MAX(total, excluded.total)",
                (ref, total),
            )
            (stored,) = self._db.execute(
                "SELECT COUNT(*) FROM partitions WHERE ref = ?", (ref,)
            ).fetchone()
            if stored == 0:
                self._insert_partitions(ref, partitions)

    def pending(self, ref: str) -> list[tuple[Partition, str | None]]:
        with self._lock:
            rows = self._db.execute(
                "SELECT key, start, stop, cursor FROM partitions "
                "WHERE ref = ? AND finished = 0 ORDER BY rowid",
                (ref,),
            ).fetchall()
        return [
            (Partition(key, start, stop), cursor) for key, start, stop, cursor in rows
        ]

    def partitions(self, ref: str) -> list[Partition]:
        """Every stored partition of `ref`, finished or not."""
        with self._lock:
            rows = self._db.execute(
                "SELECT key, start, stop FROM partitions WHERE ref = ? ORDER BY rowid",
                (ref,),
            ).fetchall()
        return [Partition(key, start, stop) for key, start, stop in rows]

    def advance(
        self,
        ref: str,
        key: str,
        cursor: str | None,
        copied: int,
        stop: str | None = None,
        last_id: str | None = None,
        previous_id: str | None = None,
    ) -> None:
        """Store the cursor of the next page; None finishes the partition.

        A `stop` replaces the stored stop. A finished tail stores its end
        offset this way, in the same commit as its last page. A `last_id`, the
        source id of the last record of the page, replaces the stored one.
        `previous_id` is the id of the record before it on the page; without
        one, the stored last id becomes the previous id.
        """
        with self._transaction():
            # SQLite computes every SET value from the row before the update.
            self._db.execute(
                "UPDATE partitions SET cursor = ?, finished = ?, "
                "copied = copied + ?, stop = COALESCE(?, stop), "
                "previous_id = CASE WHEN ? IS NOT NULL THEN ? "
                "WHEN ? IS NOT NULL THEN last_id ELSE previous_id END, "
                "last_id = COALESCE(?, last_id) "
                "WHERE ref = ? AND key = ?",
                (
                    cursor,
                    1 if cursor is None else 0,
                    copied,
                    stop,
                    previous_id,
                    previous_id,
                    last_id,
                    last_id,
                    ref,
                    key,
                ),
            )

    def add_partition(
        self,
        ref: str,
        partition: Partition,
        last_id: str | None = None,
        superseded: Sequence[str] = (),
        previous_id: str | None = None,
    ) -> None:
        """Store a new unfinished partition; a stored key changes nothing.

        `last_id` is the source id of the record before its start, and
        `previous_id` the id of the record before that one. The copied count
        of each partition key in `superseded` becomes 0 in the same commit,
        because the new partition reads those records again.
        """
        with self._transaction():
            cur = self._db.execute(
                "INSERT OR IGNORE INTO partitions "
                "(ref, key, start, stop, last_id, previous_id) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    ref,
                    partition.key,
                    partition.start,
                    partition.stop,
                    last_id,
                    previous_id,
                ),
            )
            if not cur.rowcount:
                return
            self._db.executemany(
                "UPDATE partitions SET copied = 0 WHERE ref = ? AND key = ?",
                [(ref, key) for key in superseded],
            )

    def last_ids(self, ref: str) -> dict[str, str]:
        """The stored last id of each partition of `ref` that has one, by key."""
        with self._lock:
            rows = self._db.execute(
                "SELECT key, last_id FROM partitions "
                "WHERE ref = ? AND last_id IS NOT NULL ORDER BY rowid",
                (ref,),
            ).fetchall()
        return dict(rows)

    def previous_ids(self, ref: str) -> dict[str, str]:
        """The stored previous id of each partition of `ref` that has one, by key."""
        with self._lock:
            rows = self._db.execute(
                "SELECT key, previous_id FROM partitions "
                "WHERE ref = ? AND previous_id IS NOT NULL ORDER BY rowid",
                (ref,),
            ).fetchall()
        return dict(rows)

    def raise_total(self, ref: str, total: int) -> None:
        """Store `total` for `ref` when it is above the stored total."""
        with self._transaction():
            self._db.execute(
                "UPDATE collections SET total = MAX(total, ?) WHERE ref = ?",
                (total, ref),
            )

    def set_total(self, ref: str, total: int) -> None:
        with self._transaction():
            self._db.execute(
                "UPDATE collections SET total = ? WHERE ref = ?", (total, ref)
            )

    def add_rejects(self, ref: str, rejects: list[Reject]) -> None:
        """Store rejects and append the new ones to rejects.jsonl.

        The file lines are written and closed before the sqlite commit. A crash
        between the two leaves a line without a row, and the retry writes that
        line again. A duplicate line is accepted, a missing line is not.
        """
        with self._transaction():
            added: list[Reject] = []
            for reject in rejects:
                cur = self._db.execute(
                    "INSERT OR IGNORE INTO rejects (ref, source_id, reason) "
                    "VALUES (?, ?, ?)",
                    (ref, reject.source_id, reject.reason),
                )
                if cur.rowcount:
                    added.append(reject)
            if added:
                with self._rejects_path.open("a", encoding="utf-8") as handle:
                    for reject in added:
                        line = {
                            "ref": ref,
                            "source_id": reject.source_id,
                            "reason": reject.reason,
                        }
                        handle.write(json.dumps(line) + "\n")

    def progress(self) -> list[CollectionProgress]:
        with self._lock:
            rows = self._db.execute(
                "SELECT c.ref, c.total, "
                "(SELECT COALESCE(SUM(copied), 0) FROM partitions p "
                "WHERE p.ref = c.ref), "
                "(SELECT COUNT(*) FROM rejects r WHERE r.ref = c.ref), "
                "(SELECT COUNT(*) FROM partitions p "
                "WHERE p.ref = c.ref AND p.finished = 0) + "
                "(SELECT COUNT(*) FROM quantizer_updates q WHERE q.ref = c.ref) "
                "FROM collections c ORDER BY c.rowid"
            ).fetchall()
        return [
            CollectionProgress(ref, total, copied, rejects, pending == 0)
            for ref, total, copied, rejects, pending in rows
        ]

    def expect_quantizer_update(self, ref: str) -> None:
        """Keep `ref` not done until `quantizers_updated(ref)` is called."""
        with self._transaction():
            self._db.execute(
                "INSERT OR IGNORE INTO quantizer_updates (ref) VALUES (?)", (ref,)
            )

    def quantizers_updated(self, ref: str) -> None:
        with self._transaction():
            self._db.execute("DELETE FROM quantizer_updates WHERE ref = ?", (ref,))

    def has_progress(self, ref: str) -> bool:
        """True when `ref` has a copied record, a finished partition or a hash."""
        with self._lock:
            (found,) = self._db.execute(
                "SELECT EXISTS (SELECT 1 FROM partitions WHERE ref = ? AND "
                "(finished = 1 OR copied > 0)) "
                "OR EXISTS (SELECT 1 FROM hashes WHERE ref = ?)",
                (ref, ref),
            ).fetchone()
        return bool(found)

    def targets(self) -> dict[str, Target]:
        """The stored target of each ref that has one."""
        with self._lock:
            rows = self._db.execute(
                "SELECT ref, host, name, tenant FROM targets ORDER BY rowid"
            ).fetchall()
        return {ref: Target(host, name, tenant) for ref, host, name, tenant in rows}

    def set_target(self, ref: str, target: Target) -> None:
        with self._transaction():
            self._db.execute(
                "INSERT INTO targets (ref, host, name, tenant) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(ref) DO UPDATE SET host = excluded.host, "
                "name = excluded.name, tenant = excluded.tenant",
                (ref, target.host, target.name, target.tenant),
            )

    def reset(self, ref: str) -> None:
        with self._transaction():
            self._db.execute("DELETE FROM partitions WHERE ref = ?", (ref,))
            self._db.execute("DELETE FROM rejects WHERE ref = ?", (ref,))

    def set_plan_hash(self, ref: str, digest: bytes) -> bool:
        """Store the hash of the plan entry of `ref`.

        Returns True when another hash was stored. The record hashes stay:
        each one holds the plan hash it was written under, so none of them
        matches under another plan.
        """
        with self._transaction():
            row = self._db.execute(
                "SELECT hash FROM plans WHERE ref = ?", (ref,)
            ).fetchone()
            self._db.execute(
                "INSERT INTO plans (ref, hash) VALUES (?, ?) "
                "ON CONFLICT(ref) DO UPDATE SET hash = excluded.hash",
                (ref, digest),
            )
        return row is not None and bytes(row[0]) != digest

    def store_plan_hash(self, ref: str, digest: bytes) -> None:
        """Store `digest` when `ref` has no plan hash yet."""
        with self._transaction():
            self._db.execute(
                "INSERT OR IGNORE INTO plans (ref, hash) VALUES (?, ?)", (ref, digest)
            )

    def delete_hashes(self, ref: str, source_ids: list[str]) -> int:
        """Delete the hashes of `source_ids`; returns how many were stored."""
        deleted = 0
        with self._transaction():
            for i in range(0, len(source_ids), BATCH_SIZE):
                batch = source_ids[i : i + BATCH_SIZE]
                marks = ",".join("?" * len(batch))
                cursor = self._db.execute(
                    f"DELETE FROM hashes WHERE ref = ? AND source_id IN ({marks})",
                    (ref, *batch),
                )
                deleted += cursor.rowcount
        return deleted

    def rejected_ids(self, ref: str) -> set[str]:
        """The source ids, or uuids, of the stored rejects of `ref`."""
        with self._lock:
            rows = self._db.execute(
                "SELECT source_id FROM rejects WHERE ref = ?", (ref,)
            ).fetchall()
        return {source_id for (source_id,) in rows}

    def record_hashes(self, ref: str, hashes: list[tuple[str, bytes]]) -> None:
        with self._transaction():
            self._db.executemany(
                "INSERT OR REPLACE INTO hashes (ref, source_id, hash) VALUES (?, ?, ?)",
                [(ref, source_id, digest) for source_id, digest in hashes],
            )

    def unchanged(self, ref: str, hashes: list[tuple[str, bytes]]) -> set[str]:
        wanted = dict(hashes)
        ids = list(wanted)
        same: set[str] = set()
        with self._lock:
            for i in range(0, len(ids), BATCH_SIZE):
                batch = ids[i : i + BATCH_SIZE]
                marks = ",".join("?" * len(batch))
                rows = self._db.execute(
                    "SELECT source_id, hash FROM hashes "
                    f"WHERE ref = ? AND source_id IN ({marks})",
                    (ref, *batch),
                ).fetchall()
                same.update(sid for sid, digest in rows if digest == wanted[sid])
        return same

    def _insert_partitions(self, ref: str, partitions: list[Partition]) -> None:
        self._db.executemany(
            "INSERT OR IGNORE INTO partitions (ref, key, start, stop) "
            "VALUES (?, ?, ?, ?)",
            [(ref, p.key, p.start, p.stop) for p in partitions],
        )
