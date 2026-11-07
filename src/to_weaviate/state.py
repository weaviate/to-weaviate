import json
import sqlite3
import threading
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
"""


class StateUnreadable(Exception):
    """The file at `path` is not a state that sqlite can read."""

    def __init__(self, path: Path) -> None:
        super().__init__(f"{path.name} in {path.parent} cannot be read.")
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
    written before the `targets` or the `quantizer_updates` table existed gets
    the table when it is opened, with no row for any ref. A file that sqlite
    cannot read raises StateUnreadable when it is opened.
    """

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._rejects_path = path.parent / "rejects.jsonl"
        self._lock = threading.Lock()
        self._db = sqlite3.connect(path, check_same_thread=False)
        try:
            with self._lock, self._db:
                self._db.executescript(_SCHEMA)
        except sqlite3.DatabaseError as exc:
            self._db.close()
            raise StateUnreadable(path) from exc

    def close(self) -> None:
        with self._lock:
            self._db.close()

    def init_collection(
        self, ref: str, partitions: list[Partition], total: int
    ) -> None:
        """Store `ref` with `total`, and `partitions` when it has none.

        A stored total that is higher stays.
        """
        with self._lock, self._db:
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
    ) -> None:
        """Store the cursor of the next page; None finishes the partition.

        A `stop` replaces the stored stop. A finished tail stores its end
        offset this way, in the same commit as its last page.
        """
        with self._lock, self._db:
            self._db.execute(
                "UPDATE partitions SET cursor = ?, finished = ?, "
                "copied = copied + ?, stop = COALESCE(?, stop) "
                "WHERE ref = ? AND key = ?",
                (cursor, 1 if cursor is None else 0, copied, stop, ref, key),
            )

    def add_partition(self, ref: str, partition: Partition) -> None:
        with self._lock, self._db:
            self._insert_partitions(ref, [partition])

    def raise_total(self, ref: str, total: int) -> None:
        """Store `total` for `ref` when it is above the stored total."""
        with self._lock, self._db:
            self._db.execute(
                "UPDATE collections SET total = MAX(total, ?) WHERE ref = ?",
                (total, ref),
            )

    def add_rejects(self, ref: str, rejects: list[Reject]) -> None:
        """Store rejects and append the new ones to rejects.jsonl.

        The file lines are written and closed before the sqlite commit. A crash
        between the two leaves a line without a row, and the retry writes that
        line again. A duplicate line is accepted, a missing line is not.
        """
        with self._lock, self._db:
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
        with self._lock, self._db:
            self._db.execute(
                "INSERT OR IGNORE INTO quantizer_updates (ref) VALUES (?)", (ref,)
            )

    def quantizers_updated(self, ref: str) -> None:
        with self._lock, self._db:
            self._db.execute("DELETE FROM quantizer_updates WHERE ref = ?", (ref,))

    def has_progress(self, ref: str) -> bool:
        """True when a record of `ref` was copied or a partition of it finished."""
        with self._lock:
            (found,) = self._db.execute(
                "SELECT EXISTS (SELECT 1 FROM partitions WHERE ref = ? AND "
                "(finished = 1 OR copied > 0))",
                (ref,),
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
        with self._lock, self._db:
            self._db.execute(
                "INSERT INTO targets (ref, host, name, tenant) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(ref) DO UPDATE SET host = excluded.host, "
                "name = excluded.name, tenant = excluded.tenant",
                (ref, target.host, target.name, target.tenant),
            )

    def reset(self, ref: str) -> None:
        with self._lock, self._db:
            self._db.execute("DELETE FROM partitions WHERE ref = ?", (ref,))
            self._db.execute("DELETE FROM rejects WHERE ref = ?", (ref,))

    def record_hashes(self, ref: str, hashes: list[tuple[str, bytes]]) -> None:
        with self._lock, self._db:
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
