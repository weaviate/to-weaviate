import itertools
import json
import math
import random
import sqlite3
import struct
import tempfile
from collections.abc import Iterable, Iterator
from contextlib import closing, contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, TypeVar

from to_weaviate.engine.copy import REJECTS_HINT
from to_weaviate.ids import to_uuid
from to_weaviate.plan import CollectionPlan
from to_weaviate.sources.base import Source, SourceCollection
from to_weaviate.target.reader import TargetReader
from to_weaviate.target.writer import Reject, TargetObject, to_object

T = TypeVar("T")

VECTOR_TOLERANCE = 1e-5
ID_SAMPLE = 20
LISTED_FAILURES = 10
ID_BATCH = 1000
REPORT_MD = "report.md"
REPORT_JSON = "report.json"
TARGET_BELOW_SOURCE = (
    "run to-weaviate copy again. If the counts still differ, records were "
    f"skipped or rejected: run to-weaviate sync, and read rejects.jsonl. {REJECTS_HINT}"
)
MISSING_IDS = (
    "run to-weaviate copy again. If ids are still missing, records were "
    f"skipped or rejected: run to-weaviate sync, and read rejects.jsonl. {REJECTS_HINT}"
)
TARGET_ABOVE_SOURCE = (
    "the target holds objects whose source records were deleted; stop the "
    "writes to the source and run to-weaviate sync --prune to delete them."
)
FOREIGN_OBJECTS = (
    "the target holds objects without a source_id, which this tool did not "
    "write; sync --prune leaves them, so delete them in Weaviate if they do not "
    "belong to the collection."
)
# The count check cannot tell stale objects from objects without a source_id.
COUNT_ABOVE_SOURCE = (
    "the target holds objects that are not in the source; the ids check of a "
    "verify without --quick says which and how to remove them."
)
NOTHING_COMPARED = "no record was compared; run to-weaviate verify again"
REJECTS_IN_TARGET = (
    "the target holds objects for records that this plan rejects. Fix the "
    "cause of each reject, then run to-weaviate sync. An existing collection "
    "keeps its property types, and a changed type stops sync with "
    "plan_mismatch: to change one, copy with another --dir into a target name "
    "that does not exist, or delete the collection first."
)
EVERY_SAMPLE_REJECTED = (
    "every sampled record is rejected at copy time; read rejects.jsonl"
)

_TABLES = """
PRAGMA journal_mode = OFF;
PRAGMA synchronous = OFF;
CREATE TABLE source (source_id TEXT PRIMARY KEY, uuid TEXT NOT NULL);
CREATE TABLE target (uuid TEXT PRIMARY KEY, source_id TEXT);
"""
_MISSING = (
    "FROM source s WHERE NOT EXISTS (SELECT 1 FROM target t WHERE t.uuid = s.uuid)"
)
_EXTRA = "FROM target t WHERE NOT EXISTS (SELECT 1 FROM source s WHERE s.uuid = t.uuid)"
# An object without source_id was not written by the tool, and one whose
# source_id names a listed record has a live record, whatever its uuid.
_STALE = (
    f"{_EXTRA} AND t.source_id IS NOT NULL "
    "AND NOT EXISTS (SELECT 1 FROM source s WHERE s.source_id = t.source_id)"
)


@dataclass(frozen=True)
class Check:
    name: str
    collection: str  # target collection name
    passed: bool
    detail: str
    tenant: str | None = None


@dataclass(frozen=True)
class IdDiff:
    missing: int
    extra: int
    missing_sample: list[str]  # source ids, at most ID_SAMPLE
    # The source_id of each extra object, or its uuid when it has none; at
    # most ID_SAMPLE.
    extra_sample: list[str]
    foreign: int = 0  # extra objects without a source_id


def _quoted(value: Any) -> str:
    """JSON text of a source id or a list of them, so that each stays on one line."""
    return json.dumps(value, ensure_ascii=False)


def _check(name: str, entry: CollectionPlan, passed: bool, detail: str) -> Check:
    return Check(name, entry.target.name, passed, detail, entry.target.tenant)


def target_label(collection: str, tenant: str | None) -> str:
    """`collection`, followed by the tenant in parentheses when there is one."""
    return f"{collection} (tenant {tenant})" if tenant else collection


def _batches(items: Iterable[T], size: int) -> Iterator[list[T]]:
    iterator = iter(items)
    while batch := list(itertools.islice(iterator, size)):
        yield batch


def missing_target(reader: TargetReader, entry: CollectionPlan) -> Check | None:
    """A failed count check when the entry's collection or tenant does not exist."""
    if reader.exists(entry):
        return None
    name, tenant = entry.target.name, entry.target.tenant
    without_tenant = entry.model_copy(
        update={"target": entry.target.model_copy(update={"tenant": None})}
    )
    if tenant and reader.exists(without_tenant):
        missing = f"tenant {tenant} of collection {name}"
    else:
        missing = f"collection {name}"
    return _check(
        "count", entry, False, f"{missing} does not exist; run to-weaviate copy"
    )


def verify_counts(
    source: Source,
    reader: TargetReader,
    entry: CollectionPlan,
    collection: SourceCollection,
) -> Check:
    expected = source.count(collection)
    found = reader.count(entry)
    detail = f"source count {expected}, target count {found}"
    if found < expected:
        detail += f"; {TARGET_BELOW_SOURCE}"
    elif found > expected:
        detail += f"; {COUNT_ABOVE_SOURCE}"
    return _check("count", entry, expected == found, detail)


def _load(
    db: sqlite3.Connection,
    source_ids: Iterable[str],
    target_pairs: Iterable[tuple[str, str | None]],
) -> None:
    db.executescript(_TABLES)
    for batch in _batches(((i, to_uuid(i)) for i in source_ids), ID_BATCH):
        db.executemany("INSERT OR IGNORE INTO source VALUES (?, ?)", batch)
        db.commit()
    for pairs in _batches(target_pairs, ID_BATCH):
        db.executemany("INSERT OR IGNORE INTO target VALUES (?, ?)", pairs)
        db.commit()
    db.execute("CREATE INDEX source_uuid ON source (uuid)")


def _count(db: sqlite3.Connection, where: str) -> int:
    return int(db.execute(f"SELECT COUNT(*) {where}").fetchone()[0])


def _counts(db: sqlite3.Connection) -> IdDiff:
    def sample(column: str, where: str) -> list[str]:
        rows = db.execute(f"SELECT {column} {where} LIMIT ?", (ID_SAMPLE,))
        return [str(row[0]) for row in rows]

    return IdDiff(
        missing=_count(db, _MISSING),
        extra=_count(db, _EXTRA),
        missing_sample=sample("s.source_id", _MISSING),
        extra_sample=sample("COALESCE(t.source_id, t.uuid)", _EXTRA),
        foreign=_count(db, f"{_EXTRA} AND t.source_id IS NULL"),
    )


@contextmanager
def id_tables(
    source: Source,
    reader: TargetReader,
    entry: CollectionPlan,
    collection: SourceCollection,
    workdir: Path,
    prefix: str = "verify-",
) -> Iterator[sqlite3.Connection]:
    """A database with the source ids and the target (uuid, source_id) pairs.

    The ids go through a temporary sqlite file in a directory under `workdir`
    whose name starts with `prefix`, so memory does not grow with the
    collection. The directory is removed on exit.
    """
    with tempfile.TemporaryDirectory(prefix=prefix, dir=workdir) as scratch:
        with closing(sqlite3.connect(Path(scratch) / "ids.sqlite")) as db:
            _load(db, source.ids(collection), reader.source_ids(entry))
            yield db


def source_listed(db: sqlite3.Connection) -> int:
    """The number of source ids that `id_tables` listed."""
    return _count(db, "FROM source")


def owned_total(db: sqlite3.Connection) -> int:
    """The number of listed target objects that have a source_id."""
    return _count(db, "FROM target WHERE source_id IS NOT NULL")


def _batches_of(cursor: sqlite3.Cursor) -> Iterator[list[Any]]:
    while rows := cursor.fetchmany(ID_BATCH):
        yield rows


def missing_source_ids(db: sqlite3.Connection) -> Iterator[list[str]]:
    """The source ids without a target object, in batches of ID_BATCH."""
    cursor = db.execute(f"SELECT s.source_id {_MISSING} ORDER BY s.uuid")
    for rows in _batches_of(cursor):
        yield [source_id for (source_id,) in rows]


def stale_count(db: sqlite3.Connection) -> int:
    """The number of objects with a source_id that names no listed record."""
    return _count(db, _STALE)


def foreign_count(
    source: Source,
    reader: TargetReader,
    entry: CollectionPlan,
    collection: SourceCollection,
    workdir: Path,
) -> int:
    """The objects of the entry's target that `sync --prune` would delete now.

    These are objects with a source_id that the source does not hold: objects
    of another source, or of records deleted from this one.
    """
    with id_tables(source, reader, entry, collection, workdir, "ids-") as db:
        return stale_count(db)


def stale_batches(db: sqlite3.Connection) -> Iterator[list[tuple[str, str]]]:
    """(uuid, source_id) of the stale objects, in batches of ID_BATCH."""
    cursor = db.execute(f"SELECT t.uuid, t.source_id {_STALE} ORDER BY t.uuid")
    for rows in _batches_of(cursor):
        yield [(str(uuid), str(source_id)) for uuid, source_id in rows]


def diff_ids(
    source: Source,
    reader: TargetReader,
    entry: CollectionPlan,
    collection: SourceCollection,
    workdir: Path,
) -> IdDiff:
    """Source ids without a target object, and target objects without a source id."""
    with id_tables(source, reader, entry, collection, workdir) as db:
        return _counts(db)


def verify_ids(diff: IdDiff, entry: CollectionPlan) -> Check:
    detail = f"missing {diff.missing}, extra {diff.extra}"
    if diff.missing_sample:
        detail += f"; missing ids {_quoted(diff.missing_sample)}"
    if diff.extra_sample:
        detail += f"; extra ids {_quoted(diff.extra_sample)}"
    if diff.missing:
        detail += f"; {MISSING_IDS}"
    if diff.extra > diff.foreign:
        detail += f"; {TARGET_ABOVE_SOURCE}"
    if diff.foreign:
        detail += f"; {FOREIGN_OBJECTS}"
    return _check("ids", entry, diff.missing == diff.extra == 0, detail)


def _int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _same(expected: Any, found: Any) -> bool:
    """Whether `found` is how the target stores `expected`.

    An int has to come back with the same digits, as an int or as a float.
    """
    if isinstance(expected, list) and isinstance(found, list):
        return len(expected) == len(found) and all(map(_same, expected, found))
    if isinstance(expected, bool) or isinstance(found, bool):
        return type(expected) is type(found) and expected == found
    if _int(expected):
        if isinstance(found, float):
            return found.is_integer() and int(found) == expected
        return _int(found) and expected == found
    if isinstance(expected, float) or isinstance(found, float):
        numbers = (int, float)
        return (
            isinstance(expected, numbers)
            and isinstance(found, numbers)
            and math.isclose(expected, found)
        )
    return bool(expected == found)


def _float32(values: list[float]) -> list[float] | None:
    """`values` rounded to float32, as the target stores them.

    None when a value is outside the float32 range.
    """
    layout = f"<{len(values)}f"
    try:
        return list(struct.unpack(layout, struct.pack(layout, *values)))
    except OverflowError:
        return None


def _vector_problem(
    name: str, expected: list[float], found: list[float] | None
) -> str | None:
    if found is None:
        return f"vector {name} is missing"
    if len(found) != len(expected):
        return f"vector {name} has length {len(found)}, the source has {len(expected)}"
    rounded = _float32(expected)
    if rounded is None:
        return f"vector {name} has a source value outside the float32 range"
    if not all(math.isfinite(value) for value in (*rounded, *found)):
        return f"vector {name} holds a NaN or an infinite value"
    worst = max((abs(a - b) for a, b in zip(rounded, found, strict=True)), default=0.0)
    if worst > VECTOR_TOLERANCE:
        return f"vector {name} differs by up to {worst:.3g}"
    return None


def _problems(
    entry: CollectionPlan, expected: TargetObject, found: TargetObject
) -> list[str]:
    """Names of the differing properties and kept vectors. Values are not quoted."""
    out = [
        f"property {name} differs"
        for name in sorted(set(expected.properties) | set(found.properties))
        if not _same(expected.properties.get(name), found.properties.get(name))
    ]
    for vector in entry.target.vectors:
        if vector.mode == "keep":
            problem = _vector_problem(
                vector.name,
                expected.vectors[vector.name],
                found.vectors.get(vector.name),
            )
            if problem:
                out.append(problem)
    return out


def verify_records(
    source: Source,
    reader: TargetReader,
    entry: CollectionPlan,
    collection: SourceCollection,
    sample: int,
    rng: random.Random,
) -> Check:
    """Compare `sample` source records with their target objects.

    A record that converts to a reject is not compared. It is a difference
    when the target holds an object for it, as after a copy under another plan
    or by a version that did not reject it; otherwise it is counted as
    skipped. An empty sample of an empty source passes. Otherwise the check
    fails when no record was compared.
    """
    with_vectors = any(v.mode == "keep" for v in entry.target.vectors)
    records = source.sample(collection, sample, rng, with_vectors)
    if not records and records.source_count == 0:
        return _check("records", entry, True, "source is empty")
    expected: list[tuple[str, TargetObject]] = []
    rejects: list[Reject] = []
    for record in records:
        obj = to_object(record, entry)
        if isinstance(obj, TargetObject):
            expected.append((record.id, obj))
        else:
            rejects.append(obj)
    stored = reader.fetch(
        entry,
        [*(obj.uuid for _, obj in expected), *(to_uuid(r.source_id) for r in rejects)],
    )
    failures: list[str] = []
    for source_id, obj in expected:
        found = stored.get(obj.uuid)
        problems = (
            ["missing in the target"] if found is None else _problems(entry, obj, found)
        )
        if problems:
            failures.append(f"{_quoted(source_id)}: {', '.join(problems)}")
    in_target = [r for r in rejects if to_uuid(r.source_id) in stored]
    failures.extend(
        f"{_quoted(r.source_id)}: is a reject under this plan ({r.reason}), and "
        "the target holds an object for it"
        for r in in_target
    )
    detail = (
        f"sampled {len(records)}, compared {len(expected)}, "
        f"rejects skipped {len(rejects) - len(in_target)}, differing {len(failures)}"
    )
    if not records:
        detail += f"; {NOTHING_COMPARED}"
    elif failures:
        detail += ": " + "; ".join(failures[:LISTED_FAILURES])
        if len(failures) > LISTED_FAILURES:
            detail += f"; {len(failures) - LISTED_FAILURES} more"
        if in_target:
            detail += f"; {REJECTS_IN_TARGET}"
    elif not expected:
        detail += f"; {EVERY_SAMPLE_REJECTED}"
    return _check("records", entry, bool(expected) and not failures, detail)


def check_line(check: Check) -> str:
    status = "PASS" if check.passed else "FAIL"
    place = target_label(check.collection, check.tenant)
    return f"{status} {place} {check.name}: {check.detail}"


def remove_report(directory: Path) -> None:
    """Remove report.md and report.json from `directory` when they exist."""
    for name in (REPORT_MD, REPORT_JSON):
        (directory / name).unlink(missing_ok=True)


def write_report(checks: list[Check], directory: Path) -> None:
    """Write report.md and report.json into `directory`."""
    # Blank lines keep each check its own paragraph when the markdown renders.
    lines = ["# to-weaviate verification report", *(check_line(c) for c in checks)]
    (directory / REPORT_MD).write_text("\n\n".join(lines) + "\n", encoding="utf-8")
    document = {
        "passed": all(c.passed for c in checks),
        "checks": [asdict(c) for c in checks],
    }
    (directory / REPORT_JSON).write_text(
        json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
