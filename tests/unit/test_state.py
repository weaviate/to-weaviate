import json
import os
import sqlite3
import threading
import time
from pathlib import Path

import pytest
from fakes import old_state_file

from to_weaviate.sources.base import Partition
from to_weaviate.state import (
    CollectionProgress,
    State,
    StateInUse,
    StateUnreadable,
    StateUnwritable,
    Target,
)
from to_weaviate.target.writer import Reject

REF = "default/c1"


_opened: list[State] = []


@pytest.fixture(autouse=True)
def _close_states():
    yield
    while _opened:
        _opened.pop().close()


def make_state(tmp_path: Path) -> State:
    state = State(tmp_path / "state" / "state.sqlite")
    _opened.append(state)
    return state


def parts(*keys: str) -> list[Partition]:
    return [Partition(k, None, None) for k in keys]


def test_finished_partition_not_pending(tmp_path):
    state = make_state(tmp_path)
    state.init_collection(
        REF, [Partition("0-300", "0", "300"), Partition("300-600", "300", "600")], 600
    )
    state.advance(REF, "0-300", None, 300)
    assert [p.key for p, _ in state.pending(REF)] == ["300-600"]


def test_cursor_survives_reopen(tmp_path):
    state = make_state(tmp_path)
    part = Partition("0-300", "0", "300")
    state.init_collection(REF, [part], 300)
    state.advance(REF, "0-300", "150", 150)
    assert make_state(tmp_path).pending(REF) == [(part, "150")]


def test_pending_cursor_is_none_before_first_advance(tmp_path):
    state = make_state(tmp_path)
    state.init_collection(REF, parts("a", "b"), 10)
    assert state.pending(REF) == [
        (Partition("a", None, None), None),
        (Partition("b", None, None), None),
    ]


def test_init_twice_keeps_progress(tmp_path):
    state = make_state(tmp_path)
    state.init_collection(REF, parts("a", "b"), 10)
    state.advance(REF, "a", "5", 5)
    state.advance(REF, "b", None, 5)
    state.init_collection(REF, parts("x", "y", "z"), 20)
    assert state.pending(REF) == [(Partition("a", None, None), "5")]
    assert state.progress() == [CollectionProgress(REF, 20, 10, 0, False)]


def test_init_collection_does_not_lower_the_total(tmp_path):
    state = make_state(tmp_path)
    state.init_collection(REF, parts("a"), 20)
    state.init_collection(REF, parts("a"), 10)
    assert state.progress()[0].total == 20


def test_raise_total_keeps_the_highest_count(tmp_path):
    state = make_state(tmp_path)
    state.init_collection(REF, parts("a"), 10)
    state.raise_total(REF, 15)
    state.raise_total(REF, 12)
    assert state.progress()[0].total == 15


def test_advance_stores_the_end_offset_of_a_finished_tail(tmp_path):
    state = make_state(tmp_path)
    state.init_collection(REF, [Partition("0-10", "0", "10")], 10)
    state.add_partition(REF, Partition("10-", "10", None))
    state.advance(REF, "10-", "12", 2)
    state.advance(REF, "10-", None, 3, stop="15")
    state.advance(REF, "0-10", None, 10)
    assert state.partitions(REF) == [
        Partition("0-10", "0", "10"),
        Partition("10-", "10", "15"),
    ]
    assert state.progress()[0].copied == 15


def test_advance_stores_the_last_id_of_a_page(tmp_path):
    state = make_state(tmp_path)
    state.init_collection(REF, [Partition("0-10", "0", "10")], 10)
    state.advance(REF, "0-10", "5", 5, last_id="r4")
    # An empty last page keeps the last id of the page before it.
    state.advance(REF, "0-10", None, 0)
    assert make_state(tmp_path).last_ids(REF) == {"0-10": "r4"}


def test_add_partition_stores_a_last_id_and_resets_the_superseded_counts(tmp_path):
    state = make_state(tmp_path)
    state.init_collection(
        REF, [Partition("0-5", "0", "5"), Partition("5-10", "5", "10")], 10
    )
    state.advance(REF, "0-5", None, 5, last_id="r4")
    state.advance(REF, "5-10", None, 5, last_id="r9")
    state.add_partition(
        REF, Partition("5-", "5", None), last_id="r4", superseded=["5-10"]
    )
    assert state.last_ids(REF) == {"0-5": "r4", "5-10": "r9", "5-": "r4"}
    assert state.progress()[0].copied == 5


def test_progress_totals(tmp_path):
    state = make_state(tmp_path)
    state.init_collection(REF, parts("a", "b"), 100)
    state.advance(REF, "a", "5", 5)
    state.advance(REF, "a", None, 7)
    state.advance(REF, "b", "9", 9)
    state.add_rejects(REF, [Reject("r1", "bad"), Reject("r2", "bad")])
    assert state.progress() == [CollectionProgress(REF, 100, 21, 2, False)]


def test_progress_follows_first_init_order_and_done(tmp_path):
    state = make_state(tmp_path)
    state.init_collection("z", parts("a"), 1)
    state.init_collection("a", parts("a"), 1)
    state.advance("z", "a", None, 1)
    state.init_collection("z", parts("a"), 1)
    result = state.progress()
    assert [(p.ref, p.done) for p in result] == [("z", True), ("a", False)]


def test_collection_without_partitions_is_done(tmp_path):
    state = make_state(tmp_path)
    state.init_collection(REF, [], 0)
    assert state.progress() == [CollectionProgress(REF, 0, 0, 0, True)]
    assert state.pending(REF) == []


def test_add_partition_appends_unfinished(tmp_path):
    state = make_state(tmp_path)
    state.init_collection(REF, parts("a"), 10)
    state.add_partition(REF, Partition("b", "1", "2"))
    assert [p.key for p, _ in state.pending(REF)] == ["a", "b"]


def test_add_partition_existing_key_changes_nothing(tmp_path):
    state = make_state(tmp_path)
    state.init_collection(REF, [Partition("a", "0", "9")], 10)
    state.advance(REF, "a", "4", 4)
    state.add_partition(REF, Partition("a", "1", "2"))
    assert state.pending(REF) == [(Partition("a", "0", "9"), "4")]
    assert state.progress()[0].copied == 4


def test_partitions_lists_finished_and_pending_in_insertion_order(tmp_path):
    state = make_state(tmp_path)
    state.init_collection(REF, [Partition("b", "5", "9"), Partition("a", "0", "5")], 9)
    state.init_collection("other", [Partition("x", "0", "1")], 1)
    state.advance(REF, "b", None, 4)
    state.add_partition(REF, Partition("0-", "9", None))
    assert state.partitions(REF) == [
        Partition("b", "5", "9"),
        Partition("a", "0", "5"),
        Partition("0-", "9", None),
    ]
    assert state.partitions("missing") == []


def test_add_rejects_twice_counts_once(tmp_path):
    state = make_state(tmp_path)
    state.init_collection(REF, parts("a"), 10)
    state.add_rejects(REF, [Reject("r1", "bad")])
    state.add_rejects(REF, [Reject("r1", "bad"), Reject("r2", "worse")])
    assert state.progress()[0].rejects == 2


def test_rejects_file_has_one_json_line_per_reject(tmp_path):
    state = make_state(tmp_path)
    state.init_collection(REF, parts("a"), 10)
    state.add_rejects(REF, [Reject("r1", "bad"), Reject("r2", "worse")])
    lines = (tmp_path / "state" / "rejects.jsonl").read_text().splitlines()
    assert [json.loads(line) for line in lines] == [
        {"ref": REF, "source_id": "r1", "reason": "bad"},
        {"ref": REF, "source_id": "r2", "reason": "worse"},
    ]


def test_unchanged_returns_ids_with_equal_hash(tmp_path):
    state = make_state(tmp_path)
    state.record_hashes(REF, [("a", b"1"), ("b", b"2")])
    assert state.unchanged(REF, [("a", b"1"), ("b", b"other"), ("c", b"3")]) == {"a"}


def test_record_hashes_replaces_stored_hash(tmp_path):
    state = make_state(tmp_path)
    state.record_hashes(REF, [("a", b"1")])
    state.record_hashes(REF, [("a", b"2")])
    assert state.unchanged(REF, [("a", b"1")]) == set()
    assert state.unchanged(REF, [("a", b"2")]) == {"a"}


def test_hashes_beyond_batch_size(tmp_path):
    state = make_state(tmp_path)
    hashes = [(f"id{i}", str(i).encode()) for i in range(1700)]
    state.record_hashes(REF, hashes)
    assert state.unchanged(REF, hashes) == {h[0] for h in hashes}


def test_reset_keeps_hashes(tmp_path):
    state = make_state(tmp_path)
    state.init_collection(REF, parts("a"), 10)
    state.advance(REF, "a", "5", 5)
    state.record_hashes(REF, [("r1", b"1")])
    state.reset(REF)
    assert state.pending(REF) == []
    assert state.unchanged(REF, [("r1", b"1")]) == {"r1"}


def test_set_plan_hash_stores_the_first_hash_and_keeps_the_record_hashes(tmp_path):
    state = make_state(tmp_path)
    assert state.set_plan_hash(REF, b"p1") is False
    state.record_hashes(REF, [("a", b"1"), ("b", b"2")])
    assert state.set_plan_hash(REF, b"p1") is False
    assert state.unchanged(REF, [("a", b"1"), ("b", b"2")]) == {"a", "b"}


def test_set_plan_hash_with_another_hash_reports_it_and_keeps_the_record_hashes(
    tmp_path,
):
    state = make_state(tmp_path)
    state.set_plan_hash(REF, b"p1")
    state.record_hashes(REF, [("a", b"1"), ("b", b"2")])
    assert state.set_plan_hash(REF, b"p2") is True
    assert state.unchanged(REF, [("a", b"1"), ("b", b"2")]) == {"a", "b"}
    assert state.set_plan_hash(REF, b"p2") is False


def test_set_plan_hash_touches_one_ref_only(tmp_path):
    state = make_state(tmp_path)
    state.set_plan_hash(REF, b"p1")
    state.set_plan_hash("other", b"p1")
    assert state.set_plan_hash("other", b"p2") is True
    assert state.set_plan_hash(REF, b"p1") is False


def test_store_plan_hash_keeps_a_stored_hash(tmp_path):
    state = make_state(tmp_path)
    state.store_plan_hash(REF, b"p1")
    state.store_plan_hash(REF, b"p2")
    assert state.set_plan_hash(REF, b"p1") is False


def test_delete_hashes_removes_the_named_ids_only(tmp_path):
    state = make_state(tmp_path)
    state.record_hashes(REF, [("a", b"1"), ("b", b"2"), ("c", b"3")])
    state.record_hashes("other", [("a", b"1")])
    state.delete_hashes(REF, ["a", "c", "missing"])
    assert state.unchanged(REF, [("a", b"1"), ("b", b"2"), ("c", b"3")]) == {"b"}
    assert state.unchanged("other", [("a", b"1")]) == {"a"}


def test_delete_hashes_beyond_batch_size(tmp_path):
    state = make_state(tmp_path)
    hashes = [(f"id{i}", str(i).encode()) for i in range(1700)]
    state.record_hashes(REF, hashes)
    state.delete_hashes(REF, [h[0] for h in hashes[:1600]])
    assert state.unchanged(REF, hashes) == {h[0] for h in hashes[1600:]}


def test_an_old_state_file_gets_the_plans_table(tmp_path):
    path = tmp_path / "state" / "state.sqlite"
    old_state_file(path, REF, 10, Partition("a", "0", "10"), "5", 5)
    state = State(path)
    _opened.append(state)
    assert state.set_plan_hash(REF, b"p1") is False
    assert state.set_plan_hash(REF, b"p2") is True


def test_reset_clears_stored_rejects_and_keeps_file(tmp_path):
    state = make_state(tmp_path)
    state.init_collection(REF, parts("a"), 10)
    state.add_rejects(REF, [Reject("r1", "bad")])
    state.reset(REF)
    state.init_collection(REF, parts("a"), 10)
    assert state.progress()[0].rejects == 0
    assert len((tmp_path / "state" / "rejects.jsonl").read_text().splitlines()) == 1


def test_rejects_file_failure_leaves_no_rows_and_retry_writes_lines(tmp_path):
    state = make_state(tmp_path)
    state.init_collection(REF, parts("a"), 10)
    rejects_file = tmp_path / "state" / "rejects.jsonl"
    rejects = [Reject("r1", "bad"), Reject("r2", "worse")]
    rejects_file.mkdir()
    with pytest.raises(OSError):
        state.add_rejects(REF, rejects)
    assert state.progress()[0].rejects == 0
    rejects_file.rmdir()
    state.add_rejects(REF, rejects)
    lines = rejects_file.read_text().splitlines()
    assert [json.loads(line)["source_id"] for line in lines] == ["r1", "r2"]
    assert state.progress()[0].rejects == 2


def test_targets_are_stored_per_ref_and_survive_reopen(tmp_path):
    state = make_state(tmp_path)
    assert state.targets() == {}
    state.set_target(REF, Target("localhost:8099", "Docs", None))
    state.set_target("other", Target("c.weaviate.cloud:443", "Other", "t"))
    state.set_target(REF, Target("localhost:8099", "Docs", None))
    assert make_state(tmp_path).targets() == {
        REF: Target("localhost:8099", "Docs", None),
        "other": Target("c.weaviate.cloud:443", "Other", "t"),
    }


def test_set_target_replaces_the_stored_target(tmp_path):
    state = make_state(tmp_path)
    state.set_target(REF, Target("a:80", "Docs", None))
    state.set_target(REF, Target("b:80", "Docs", "t"))
    assert state.targets() == {REF: Target("b:80", "Docs", "t")}


@pytest.mark.parametrize(
    ("cursor", "copied", "expected"),
    [
        (None, 0, True),  # finished, every record rejected
        ("5", 5, True),
        ("5", 0, False),  # one page read, every record rejected
    ],
)
def test_has_progress_after_an_advance(tmp_path, cursor, copied, expected):
    state = make_state(tmp_path)
    state.init_collection(REF, parts("a", "b"), 10)
    assert not state.has_progress(REF)
    state.advance(REF, "a", cursor, copied)
    assert state.has_progress(REF) is expected
    assert not state.has_progress("other")


def test_has_progress_is_false_without_partitions(tmp_path):
    state = make_state(tmp_path)
    state.init_collection(REF, [], 0)
    assert not state.has_progress(REF)


def test_has_progress_with_record_hashes_and_no_partitions(tmp_path):
    state = make_state(tmp_path)
    state.record_hashes(REF, [("a", b"1")])
    assert state.has_progress(REF)
    assert not state.has_progress("other")


def test_init_collection_fresh_replaces_partitions_and_rejects(tmp_path):
    state = make_state(tmp_path)
    state.init_collection(REF, parts("a", "b"), 10)
    state.advance(REF, "a", None, 5)
    state.add_rejects(REF, [Reject("r1", "bad")])
    state.record_hashes(REF, [("r1", b"1")])
    state.init_collection(REF, parts("c"), 8, fresh=True)
    assert state.pending(REF) == [(parts("c")[0], None)]
    progress = state.progress()[0]
    assert (progress.copied, progress.rejects, progress.total) == (0, 0, 10)
    assert state.unchanged(REF, [("r1", b"1")]) == {"r1"}


def test_set_total_replaces_a_higher_total(tmp_path):
    state = make_state(tmp_path)
    state.init_collection(REF, parts("a"), 10)
    state.set_total(REF, 7)
    assert state.progress()[0].total == 7


def test_delete_hashes_returns_the_number_deleted(tmp_path):
    state = make_state(tmp_path)
    state.record_hashes(REF, [("a", b"1"), ("b", b"2")])
    assert state.delete_hashes(REF, ["a", "c"]) == 1


def test_rejected_ids_lists_the_stored_rejects_of_a_ref(tmp_path):
    state = make_state(tmp_path)
    state.init_collection(REF, parts("a"), 10)
    state.add_rejects(REF, [Reject("r1", "bad"), Reject("r2", "bad")])
    state.add_rejects("other", [Reject("r3", "bad")])
    assert state.rejected_ids(REF) == {"r1", "r2"}


def test_state_file_without_targets_opens_and_stores_a_target(tmp_path):
    path = tmp_path / "state" / "state.sqlite"
    old_state_file(path, REF, 10, Partition("0-10", "0", "10"), cursor="5", copied=5)
    state = make_state(tmp_path)
    assert state.targets() == {}
    assert state.pending(REF) == [(Partition("0-10", "0", "10"), "5")]
    assert state.has_progress(REF)
    state.set_target(REF, Target("localhost:8099", "Docs", None))
    assert make_state(tmp_path).targets() == {
        REF: Target("localhost:8099", "Docs", None)
    }


def test_advance_stores_the_id_before_the_last_one(tmp_path):
    state = make_state(tmp_path)
    state.init_collection(
        REF, [Partition("0-5", "0", "5"), Partition("5-10", "5", "10")], 10
    )
    state.advance(REF, "0-5", "4", 4, last_id="r3", previous_id="r2")
    # A page of one record makes the stored last id the previous one, and an
    # empty page keeps both.
    state.advance(REF, "0-5", "5", 1, last_id="r4")
    state.advance(REF, "0-5", None, 0)
    # The first page of a partition, with one record, has no previous id.
    state.advance(REF, "5-10", "6", 1, last_id="r5")
    state.add_partition(REF, Partition("10-", "10", None), "r9", previous_id="r8")
    assert make_state(tmp_path).previous_ids(REF) == {"0-5": "r3", "10-": "r8"}
    assert state.last_ids(REF) == {"0-5": "r4", "5-10": "r5", "10-": "r9"}


def test_state_file_without_previous_ids_gets_the_column(tmp_path):
    path = tmp_path / "state" / "state.sqlite"
    old_state_file(path, REF, 10, Partition("0-10", "0", "10"), cursor="5", copied=5)
    state = make_state(tmp_path)
    assert state.previous_ids(REF) == {}
    state.advance(REF, "0-10", None, 5, last_id="r9", previous_id="r8")
    assert make_state(tmp_path).previous_ids(REF) == {"0-10": "r8"}


def test_state_file_without_last_ids_gets_the_column(tmp_path):
    path = tmp_path / "state" / "state.sqlite"
    old_state_file(path, REF, 10, Partition("0-10", "0", "10"), cursor="5", copied=5)
    state = make_state(tmp_path)
    assert state.last_ids(REF) == {}
    state.advance(REF, "0-10", None, 5, last_id="r9")
    assert make_state(tmp_path).last_ids(REF) == {"0-10": "r9"}


def test_a_pending_quantizer_update_keeps_a_finished_collection_not_done(tmp_path):
    state = make_state(tmp_path)
    state.init_collection(REF, parts("a"), 1)
    state.expect_quantizer_update(REF)
    state.expect_quantizer_update(REF)
    state.advance(REF, "a", None, 1)
    assert state.progress() == [CollectionProgress(REF, 1, 1, 0, False)]
    assert make_state(tmp_path).progress()[0].done is False
    state.quantizers_updated(REF)
    assert state.progress() == [CollectionProgress(REF, 1, 1, 0, True)]


def test_state_file_without_quantizer_updates_reports_done_from_partitions(
    tmp_path,
):
    path = tmp_path / "state" / "state.sqlite"
    old_state_file(path, REF, 10, Partition("0-10", "0", "10"), cursor="5", copied=5)
    state = make_state(tmp_path)
    state.advance(REF, "0-10", None, 5)
    assert state.progress() == [CollectionProgress(REF, 10, 10, 0, True)]


class OverlapProxy:
    """Reports two threads between a first execute and the commit."""

    def __init__(self, conn):
        self._conn = conn
        self._guard = threading.Lock()
        self._owner = None
        self.overlap = False

    def __enter__(self):
        self._conn.__enter__()
        return self

    def __exit__(self, *exc):
        try:
            return self._conn.__exit__(*exc)
        finally:
            self._owner = None

    def execute(self, *args):
        me = threading.get_ident()
        with self._guard:
            if self._owner not in (None, me):
                self.overlap = True
            self._owner = me
        time.sleep(0.002)
        return self._conn.execute(*args)

    def __getattr__(self, name):
        return getattr(self._conn, name)


class NoLock:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def run_workload(state: State, threads_n: int = 4, per_thread: int = 10) -> None:
    state.init_collection(REF, parts("a"), 10_000)

    def work(n: int) -> None:
        try:
            for i in range(per_thread):
                state.advance(REF, "a", str(i), 1)
                state.add_rejects(REF, [Reject(f"{n}-{i}", "bad")])
        except Exception:  # noqa: BLE001
            pass  # sqlite may fail when the lock is removed

    threads = [threading.Thread(target=work, args=(n,)) for n in range(threads_n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


def test_lock_keeps_statement_spans_from_overlapping(tmp_path):
    state = make_state(tmp_path)
    real = state._db
    proxy = OverlapProxy(real)
    state._db = proxy
    run_workload(state)
    state._db = real
    assert not proxy.overlap
    result = state.progress()[0]
    assert result.copied == 40
    assert result.rejects == 40


def test_overlap_proxy_detects_missing_lock(tmp_path):
    state = make_state(tmp_path)
    real = state._db
    proxy = OverlapProxy(real)
    state._db = proxy
    state._lock = NoLock()
    run_workload(state)
    state._db = real
    assert proxy.overlap


def test_a_file_that_is_not_a_database_raises_state_unreadable(tmp_path):
    path = tmp_path / "state.sqlite"
    path.write_bytes(bytes(range(256)) * 16)
    with pytest.raises(StateUnreadable) as info:
        State(path)
    assert info.value.path == path


NOT_ROOT = pytest.mark.skipif(os.geteuid() == 0, reason="root writes read-only files")


@NOT_ROOT
def test_a_read_only_state_file_raises_state_unwritable(tmp_path):
    path = tmp_path / "state.sqlite"
    State(path).close()
    path.chmod(0o444)
    try:
        with pytest.raises(StateUnwritable) as info:
            state = State(path)
            state.init_collection("r", [Partition("0-1", "0", "1")], 1)
    finally:
        path.chmod(0o644)
    assert str(info.value) == (
        f"state.sqlite in {tmp_path} cannot be written: the file is read-only."
    )
    assert info.value.path == path


@NOT_ROOT
def test_a_read_only_directory_raises_state_unwritable(tmp_path):
    directory = tmp_path / "mig"
    directory.mkdir()
    directory.chmod(0o555)
    try:
        with pytest.raises(StateUnwritable) as info:
            State(directory / "state.sqlite")
    finally:
        directory.chmod(0o755)
    assert str(info.value) == (
        f"state.sqlite in {directory} cannot be written: "
        "the directory does not allow it."
    )


class FullDisk:
    """A sqlite connection whose every statement finds the disk full."""

    def __enter__(self) -> "FullDisk":
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def execute(self, *args: object) -> None:
        raise sqlite3.OperationalError("database or disk is full")

    executemany = execute


def test_a_full_disk_during_a_write_raises_state_unwritable(tmp_path):
    state = State(tmp_path / "state.sqlite")
    real = state._db
    state._db = FullDisk()  # type: ignore[assignment]
    try:
        with pytest.raises(StateUnwritable) as info:
            state.init_collection("r", [Partition("0-1", "0", "1")], 1)
    finally:
        state._db = real
        state.close()
    assert str(info.value) == (
        f"state.sqlite in {tmp_path} cannot be written: the disk is full."
    )


def test_another_sqlite_error_during_a_write_is_not_state_unwritable(tmp_path):
    state = State(tmp_path / "state.sqlite")
    real = state._db

    class Broken(FullDisk):
        def execute(self, *args: object) -> None:
            raise sqlite3.OperationalError("no such table: partitions")

        executemany = execute

    state._db = Broken()  # type: ignore[assignment]
    try:
        with pytest.raises(sqlite3.OperationalError):
            state.init_collection("r", [Partition("0-1", "0", "1")], 1)
    finally:
        state._db = real
        state.close()


class LockedConnection:
    """A sqlite connection whose first statement finds the file locked."""

    def __enter__(self) -> "LockedConnection":
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def executescript(self, script: str) -> None:
        raise sqlite3.OperationalError("database is locked")

    def close(self) -> None:
        pass


def test_a_locked_file_raises_state_in_use(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "to_weaviate.state.sqlite3.connect", lambda *a, **k: LockedConnection()
    )
    with pytest.raises(StateInUse) as info:
        State(tmp_path / "state.sqlite")
    assert str(info.value) == f"Another to-weaviate process is using {tmp_path}."
