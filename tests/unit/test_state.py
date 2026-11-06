import json
import threading
import time
from pathlib import Path

import pytest

from to_weaviate.sources.base import Partition
from to_weaviate.state import CollectionProgress, State
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
